"""Batched (B = all rollouts) training and evaluation on the fixed-slot InEKF.

Same loss, regularization, parameters, and filter math as the sequential
path in ``covariance_calibration``; what changes is execution: all rollouts
advance together as one padded batch, chunks are synchronized fixed-length
blocks, and there is one Adam step per synchronized chunk (~T_max/chunk steps
per epoch instead of ~sum(T_i)/chunk). Host<->device sync in the train loop
is limited to the per-chunk grad clip and one logging sync per epoch.
"""

from __future__ import annotations

import copy
import time
from contextlib import contextmanager
from typing import Callable

import numpy as np
import torch

from . import fixed_slot_inekf as fsi
from .covariance_calibration import (
    GROUP_ORDER,
    LAMBDA,
    CalibrationConfig,
    Rollout,
    TrainingResult,
    _peak_memory_gb,
    _reset_peak_memory,
    aggregate_metrics,
    build_covs,
    covariance_regularization,
    fixed_initial_covariance,
    make_cov_modules,
    seed_everything,
    seed_state,
    trajectory_metrics,
    validate_training_splits,
)


def _padded_T(rollouts: dict[str, Rollout], chunk: int) -> int:
    """Pad so every chunk after the seed row has exactly ``chunk`` rows."""
    T_seg = max(r.trim1 - r.trim0 for r in rollouts.values())
    n_chunks = -(-(T_seg - 1) // chunk)
    return 1 + n_chunks * chunk


def _make_batch(rollout_order, rollouts, *, chunk: int,
                dtype: torch.dtype) -> fsi.BatchData:
    rolls = [rollouts[s] for s in rollout_order]
    return fsi.build_batch(rolls, T_pad=_padded_T(rollouts, chunk), dtype=dtype)


def _seed_states(rollout_order, rollouts, P0_fixed, batch, R_kin, device):
    seeds = [seed_state(rollouts[s], rollouts[s].trim0, P0_fixed)
             for s in rollout_order]
    state = fsi.init_state(seeds, device=device)
    return fsi.apply_row0(state, batch.p_meas[:, 0], batch.insert_mask[:, 0],
                          R_kin)


class ChunkGraph:
    """Whole-chunk CUDA-graph capture of fwd + bwd (compile_mode="cuda-graph").

    One graph replays the entire ``chunk``-row batched filter step chain plus
    the body/NIS loss and ``torch.autograd.grad`` into static buffers, killing
    per-step launch overhead entirely. The SPD regularization (eigvalsh is not
    capturable) runs eagerly per chunk and its grads are added afterwards --
    exact by gradient linearity.

    Capture rules learned the hard way (violating either invalidates capture):
    - warmup AND capture must run on the same side stream
      (``torch.cuda.graph(..., stream=side)``), so the params' AccumulateGrad
      nodes are created on the capture stream;
    - no autograd graph that references the params (e.g. a ``build_covs``
      result built outside ``torch.no_grad``) may be kept alive across the
      capture boundary -- it pins AccumulateGrad nodes to the default stream.

    The carried state enters each replay as a constant buffer, i.e. the chunk
    boundary detach of truncated BPTT; additionally the row-0 seed insertion
    is outside the graph, so its (single-row) gradient link into the first
    chunk is cut -- a documented, negligible difference from the eager path.
    """

    def __init__(self, modules, params, batch: fsi.BatchData, *, chunk: int,
                 s_jitter: float, dtype: torch.dtype,
                 state0: fsi.State | None = None,
                 step_fn=None) -> None:
        B = batch.B
        device = batch.imu.device
        self.batch = batch
        self.chunk = chunk
        c = slice(1, 1 + chunk)
        self.imu = batch.imu[:, c].clone()
        self.p_meas = batch.p_meas[:, c].clone()
        self.dt_row = batch.dt_row[:, c].clone()
        self.prop = batch.prop_mask[:, c].clone()
        self.corr = batch.correct_mask[:, c].clone()
        self.ins = batch.insert_mask[:, c].clone()
        self.gt = batch.gt_v_B[:, c].clone()
        self.valid = batch.valid[:, c].clone()
        self.nis_dim = batch.nis_dim[:, c].clone()
        self.X = torch.zeros(B, fsi.DIM_X, fsi.DIM_X, dtype=dtype, device=device)
        self.theta = torch.zeros(B, fsi.DIM_THETA, dtype=dtype, device=device)
        self.P = torch.zeros(B, fsi.DIM_P, fsi.DIM_P, dtype=dtype, device=device)
        self.jc = torch.zeros(B, dtype=dtype, device=device)
        self.ic = torch.zeros(B, dtype=dtype, device=device)
        self.grads = [torch.zeros_like(p) for p in params]
        self.loss_body = torch.zeros((), dtype=dtype, device=device)
        self.nis_term = torch.zeros((), dtype=dtype, device=device)
        self.nis_mean = torch.zeros((), dtype=dtype, device=device)

        step = fsi.step if step_fn is None else step_fn

        def fwd_bwd():
            covs, R_kin = build_covs(modules)
            st = fsi.State(self.X, self.theta, self.P, self.jc, self.ic)
            R_l, v_l, nis_l = [], [], []
            for t in range(chunk):
                st, out = step(st, self.imu[:, t], self.p_meas[:, t],
                               self.dt_row[:, t], self.prop[:, t],
                               self.corr[:, t], self.ins[:, t],
                               covs, R_kin, s_jitter)
                R_l.append(out.R)
                v_l.append(out.v)
                nis_l.append(out.nis)
            R = torch.stack(R_l, dim=1)
            v = torch.stack(v_l, dim=1)
            nis = torch.stack(nis_l, dim=1)
            v_B = torch.einsum("btji,btj->bti", R, v)
            se = ((v_B - self.gt) ** 2).sum(-1)
            loss_body = (se * self.valid).sum() / self.valid.sum().clamp_min(1)
            nis_term = LAMBDA["nis"] * fsi.reg_nis_masked(nis, self.nis_dim)
            grads = torch.autograd.grad(loss_body + nis_term, params)
            for buf, gr in zip(self.grads, grads):
                buf.copy_(gr)
            self.loss_body.copy_(loss_body.detach())
            self.nis_term.copy_(nis_term.detach())
            has = self.nis_dim > 0
            per_dim = torch.where(has, nis.detach() / self.nis_dim.clamp_min(1.0),
                                  torch.zeros_like(nis.detach()))
            self.nis_mean.copy_(per_dim.sum() / has.sum().clamp_min(1))
            # carry: next chunk starts from this chunk's final state
            self.X.copy_(st.X.detach())
            self.theta.copy_(st.theta.detach())
            self.P.copy_(st.P.detach())
            self.jc.copy_(st.jitter_count.detach())
            self.ic.copy_(st.info_count.detach())

        self.load_inputs(1)
        if state0 is not None:
            self.load_state(state0)  # sane numerics during warmup/capture
        if step_fn is not None:
            # warm the _const/_gather caches eagerly: their one-time builders
            # use data-dependent python that dynamo cannot trace
            with torch.no_grad():
                covs_w, R_kin_w = build_covs(modules)
                fsi.step(fsi.State(self.X, self.theta, self.P, self.jc,
                                   self.ic),
                         self.imu[:, 0], self.p_meas[:, 0], self.dt_row[:, 0],
                         self.prop[:, 0], self.corr[:, 0], self.ins[:, 0],
                         covs_w, R_kin_w, s_jitter)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fwd_bwd()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=side):
            fwd_bwd()

    def load_inputs(self, a: int) -> None:
        c = slice(a, a + self.chunk)
        b = self.batch
        self.imu.copy_(b.imu[:, c])
        self.p_meas.copy_(b.p_meas[:, c])
        self.dt_row.copy_(b.dt_row[:, c])
        self.prop.copy_(b.prop_mask[:, c])
        self.corr.copy_(b.correct_mask[:, c])
        self.ins.copy_(b.insert_mask[:, c])
        self.gt.copy_(b.gt_v_B[:, c])
        self.valid.copy_(b.valid[:, c])
        self.nis_dim.copy_(b.nis_dim[:, c])

    def load_state(self, state: fsi.State) -> None:
        self.X.copy_(state.X.detach())
        self.theta.copy_(state.theta.detach())
        self.P.copy_(state.P.detach())
        self.jc.copy_(state.jitter_count.detach())
        self.ic.copy_(state.info_count.detach())

    def replay_chunk(self, a: int) -> None:
        self.load_inputs(a)
        self.graph.replay()


class _StageTimer:
    """Per-chunk stage timing for the train loop (config.profile_stages).

    GPU time per stage comes from pre-allocated, reused CUDA event pairs read
    with a single synchronize at epoch end; CPU time is perf_counter around
    the same block, which attributes host blocking (the per-chunk grad clip
    is a sync point and absorbs whatever GPU work is still queued).
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._free: list[tuple] = []
        self._used: dict[str, list[tuple]] = {}
        self._cpu_ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        pair = self._free.pop() if self._free else (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))
        pair[0].record()
        t0 = time.perf_counter()
        yield
        self._cpu_ms[name] = (self._cpu_ms.get(name, 0.0)
                              + 1e3 * (time.perf_counter() - t0))
        pair[1].record()
        self._used.setdefault(name, []).append(pair)

    def epoch_summary(self) -> tuple[dict, dict] | None:
        """One sync; returns (gpu_ms, cpu_ms) per stage and resets."""
        if not self.enabled:
            return None
        torch.cuda.synchronize()
        gpu = {}
        for name, pairs in self._used.items():
            gpu[name] = sum(a.elapsed_time(b) for a, b in pairs)
            self._free.extend(pairs)
        cpu = dict(self._cpu_ms)
        self._used.clear()
        self._cpu_ms.clear()
        return gpu, cpu


def train_batched(
    train_order: list[str],
    train_rollouts: dict[str, Rollout],
    *,
    validation_order: list[str],
    validation_rollouts: dict[str, Rollout],
    config: CalibrationConfig,
    device: torch.device,
    validation_callback: Callable[[torch.nn.ModuleDict, int], dict] | None = None,
) -> TrainingResult:
    validate_training_splits(train_order, train_rollouts,
                             validation_order, validation_rollouts)
    seed_everything(config.seed, device)
    modules = make_cov_modules(device=device, dtype=config.dtype)
    params = list(modules.parameters())
    optimizer = torch.optim.Adam(
        modules.param_groups(config.lr, config.bias_lr_factor),
        betas=(0.9, 0.999), eps=1e-8,
    )
    P0_fixed = fixed_initial_covariance(device, config.dtype)
    batch = _make_batch(train_order, train_rollouts, chunk=config.chunk,
                        dtype=config.dtype)
    total_train_rows = sum(r.trim1 - r.trim0 - 1
                           for r in train_rollouts.values())
    use_graph = (device.type == "cuda" and config.compile_mode in
                 ("cuda-graph", "cuda-graph-compile"))
    step_fn = None
    if use_graph:
        with torch.no_grad():
            covs0, R_kin0 = build_covs(modules)
            state0 = _seed_states(train_order, train_rollouts, P0_fixed, batch,
                                  R_kin0, device)
        graph_step = (fsi.make_compiled_step("default")
                      if config.compile_mode == "cuda-graph-compile" else None)
        try:
            graph = ChunkGraph(modules, params, batch, chunk=config.chunk,
                               s_jitter=config.s_jitter, dtype=config.dtype,
                               state0=state0, step_fn=graph_step)
        except RuntimeError:
            print("cuda-graph capture failed; falling back to eager")
            use_graph = False
    if not use_graph:
        step_fn = fsi.make_compiled_step(
            None if config.compile_mode in ("cuda-graph", "cuda-graph-compile",
                                            None)
            else config.compile_mode)

    history: list[dict] = []
    chunk_trace: list[float] = []
    best = {
        "validation_body_velocity_rmse_mps": float("inf"),
        "epoch": -1,
        "state": None,
    }
    timer = _StageTimer(config.profile_stages and device.type == "cuda")
    t_train = time.time()
    for epoch in range(config.epochs):
        _reset_peak_memory(device)
        t_epoch = time.time()
        # GPU-accumulated diagnostics; a single sync at epoch end
        body_losses: list[torch.Tensor] = []
        reg_losses: list[torch.Tensor] = []
        nis_chunk_means: list[torch.Tensor] = []
        grad_norms = {name: [] for name in GROUP_ORDER}
        if use_graph:
            with torch.no_grad():
                covs, R_kin = build_covs(modules)
                graph.load_state(_seed_states(train_order, train_rollouts,
                                              P0_fixed, batch, R_kin, device))
        else:
            covs, R_kin = build_covs(modules)
            state = _seed_states(train_order, train_rollouts, P0_fixed, batch,
                                 R_kin, device)
        for a in range(1, batch.T_pad, config.chunk):
            if use_graph:
                with timer.stage("replay"):
                    graph.replay_chunk(a)
                # eigvalsh-based SPD regularization runs outside the graph;
                # grads add up exactly (linearity)
                with timer.stage("reg_bwd"):
                    loss_reg, _ = covariance_regularization(
                        modules, [], [], device=device)
                    optimizer.zero_grad(set_to_none=True)
                    loss_reg.backward()
                with timer.stage("grad_merge"):
                    with torch.no_grad():
                        for p, gbuf in zip(params, graph.grads):
                            p.grad = (gbuf.clone() if p.grad is None
                                      else p.grad + gbuf)
                    loss_body = graph.loss_body.clone()
                    loss_reg = loss_reg.detach() + graph.nis_term
                    nis_chunk_means.append(graph.nis_mean.clone())
            else:
                with timer.stage("forward"):
                    covs, R_kin = build_covs(modules)
                    state, out = fsi.run_rows_fixed(
                        state, batch, slice(a, a + config.chunk), covs, R_kin,
                        s_jitter=config.s_jitter, step_fn=step_fn)
                    v_B = torch.einsum("btji,btj->bti", out["R_WB"], out["v_W"])
                    se = ((v_B - batch.gt_v_B[:, a:a + config.chunk]) ** 2).sum(-1)
                    valid = batch.valid[:, a:a + config.chunk]
                    loss_body = (se * valid).sum() / valid.sum().clamp_min(1)
                    loss_reg, _ = covariance_regularization(
                        modules, [], [], device=device)
                    loss_reg = loss_reg + LAMBDA["nis"] * fsi.reg_nis_masked(
                        out["nis"], out["nis_dim"])
                    loss = loss_body + loss_reg
                with timer.stage("backward"):
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                with torch.no_grad():
                    has = out["nis_dim"] > 0
                    if bool(has.any()):
                        per_dim = out["nis"][has] / out["nis_dim"][has]
                        nis_chunk_means.append(per_dim.mean())
                state = fsi.detach_state(state)
            with timer.stage("grad_diag"):
                with torch.no_grad():
                    for name in GROUP_ORDER:
                        grad = modules[name].raw_tril.grad
                        grad_norms[name].append(
                            grad.norm() if grad is not None
                            else torch.zeros((), dtype=config.dtype,
                                             device=device))
            # per-chunk device sync + fail-fast on non-finite grads (as before)
            with timer.stage("clip"):
                torch.nn.utils.clip_grad_norm_(params, 1.0,
                                               error_if_nonfinite=True)
            with timer.stage("opt_step"):
                optimizer.step()
            body_losses.append(loss_body.detach())
            reg_losses.append(loss_reg.detach())
        # single epoch-end sync for logging
        body_t = torch.stack(body_losses)
        reg_t = torch.stack(reg_losses)
        if not bool(torch.isfinite(body_t).all() and torch.isfinite(reg_t).all()):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}")
        body_np = body_t.cpu().numpy()
        chunk_trace.extend(float(x) for x in body_np)
        jc = graph.jc if use_graph else state.jitter_count
        ic = graph.ic if use_graph else state.info_count
        rec = {
            "epoch": epoch,
            "train_body_loss": float(body_np.mean()),
            "train_reg_loss": float(reg_t.mean()),
            "nis_per_dim_mean": (float(torch.stack(nis_chunk_means).mean())
                                 if nis_chunk_means else None),
            "jitter_events": int(jc.sum().item()),
            "chol_info_events": int(ic.sum().item()),
            "peak_gb": _peak_memory_gb(device),
            "epoch_s": time.time() - t_epoch,
            "rows_per_s": total_train_rows / max(time.time() - t_epoch, 1e-12),
            "groups": modules.summary(),
        }
        for name in GROUP_ORDER:
            rec["groups"][name]["grad_norm_mean"] = float(
                torch.stack(grad_norms[name]).mean())
        stages = timer.epoch_summary()
        if stages is not None:
            rec["stage_ms_gpu"], rec["stage_ms_cpu"] = stages
            print("  stages gpu_ms: "
                  + " ".join(f"{k}={v:.0f}" for k, v in stages[0].items())
                  + " | cpu_ms: "
                  + " ".join(f"{k}={v:.0f}" for k, v in stages[1].items()))
        with torch.no_grad():
            if validation_callback is not None:
                validation = validation_callback(modules, epoch)
            else:
                covs, R_kin = build_covs(modules)
                validation = aggregate_metrics(eval_batched(
                    validation_order, validation_rollouts, covs, R_kin,
                    P0_fixed=P0_fixed, s_jitter=config.s_jitter))
        metric = float(validation["body_velocity_rmse_mps"])
        if not np.isfinite(metric):
            raise FloatingPointError(f"non-finite validation metric at epoch {epoch}")
        rec["validation"] = validation
        history.append(rec)
        if metric < best["validation_body_velocity_rmse_mps"]:
            best = {
                "validation_body_velocity_rmse_mps": metric,
                "epoch": epoch,
                "state": copy.deepcopy(modules.state_dict()),
            }
        nis_text = (f"{rec['nis_per_dim_mean']:.3f}"
                    if rec["nis_per_dim_mean"] is not None else "n/a")
        print(
            f"epoch {epoch:2d}: body {rec['train_body_loss']:.4f} "
            f"reg {rec['train_reg_loss']:.5f} | val {metric:.4f} m/s "
            f"| NIS/dim {nis_text} "
            f"| {rec['epoch_s']:.0f}s ({rec['rows_per_s']:.0f} rows/s) "
            f"| peak {rec['peak_gb']:.2f} GB"
        )
    final_state = copy.deepcopy(modules.state_dict())
    modules.load_state_dict(best["state"])
    return TrainingResult(
        modules=modules,
        optimizer=optimizer,
        history=history,
        best=best,
        chunk_trace=chunk_trace,
        runtime_s=time.time() - t_train,
        lr=config.lr,
        final_state_dict=final_state,
    )


# -----------------------------------------------------------------------------
# batched no-grad evaluation (same metrics as covariance_calibration.eval_replay)


def _active_P_submatrix(P: torch.Tensor, flags_last: np.ndarray) -> torch.Tensor:
    idx = list(range(9))
    for j in np.nonzero(np.asarray(flags_last).astype(bool))[0]:
        idx += [9 + 3 * int(j) + i for i in range(3)]
    idx += [fsi.GROUP + i for i in range(6)]
    return P[idx][:, idx]


def eval_batched(
    rollout_order: list[str],
    rollouts: dict[str, Rollout],
    covs: dict[str, torch.Tensor],
    R_kin: torch.Tensor,
    *,
    P0_fixed: torch.Tensor,
    s_jitter: float,
    block: int = 2000,
) -> dict[str, dict]:
    """No-grad batched replay of all rollouts; per-rollout eval metrics."""
    device = R_kin.device
    rolls = [rollouts[s] for s in rollout_order]
    with torch.no_grad():
        batch = fsi.build_batch(rolls)
        state = _seed_states(rollout_order, rollouts, P0_fixed, batch, R_kin,
                             device)
        R0 = state.X[:, 0:3, 0:3].clone()
        v0 = state.X[:, 0:3, 3].clone()
        p0 = state.X[:, 0:3, 4].clone()
        R_l, v_l, p_l = [R0[:, None]], [v0[:, None]], [p0[:, None]]
        nis_l, nis_dim_l = [], []
        for a in range(1, batch.T_pad, block):
            state, out = fsi.run_rows_fixed(
                state, batch, slice(a, min(a + block, batch.T_pad)), covs,
                R_kin, s_jitter=s_jitter)
            R_l.append(out["R_WB"])
            v_l.append(out["v_W"])
            p_l.append(out["p_W"])
            nis_l.append(out["nis"])
            nis_dim_l.append(out["nis_dim"])
        R_est = torch.cat(R_l, dim=1)
        v_est = torch.cat(v_l, dim=1)
        p_est = torch.cat(p_l, dim=1)
        nis = torch.cat(nis_l, dim=1)
        nis_dim = torch.cat(nis_dim_l, dim=1)
    results: dict[str, dict] = {}
    for i, stem in enumerate(rollout_order):
        roll = rollouts[stem]
        n = roll.trim1 - roll.trim0
        P_act = _active_P_submatrix(state.P[i], roll.flags[roll.trim1 - 1])
        results[stem] = trajectory_metrics(
            R_est[i, :n], v_est[i, :n], p_est[i, :n],
            batch.gt_R_WB[i, :n], batch.gt_v_B[i, :n], batch.gt_p_W[i, :n],
            P_act, nis=nis[i, :n - 1], nis_dim=nis_dim[i, :n - 1],
            jitter_events=int(state.jitter_count[i].item()),
        )
    return results


def evaluate_all_batched(
    rollout_order: list[str],
    rollouts: dict[str, Rollout],
    *,
    covs_initial, R_kin_initial, covs_calibrated, R_kin_calibrated,
    P0_fixed: torch.Tensor,
    s_jitter: float,
) -> dict:
    """Same summary schema as covariance_calibration.evaluate_all."""
    init = eval_batched(rollout_order, rollouts, covs_initial, R_kin_initial,
                        P0_fixed=P0_fixed, s_jitter=s_jitter)
    cal = eval_batched(rollout_order, rollouts, covs_calibrated,
                       R_kin_calibrated, P0_fixed=P0_fixed, s_jitter=s_jitter)
    summary = {"rollouts": {}}
    sse_init = sse_cal = rows_total = 0.0
    for stem in rollout_order:
        c, i0 = cal[stem], init[stem]
        if not (c["finite"] and c["final_P_min_eig"] > -1e-12
                and c["final_P_sym"] < 1e-9):
            raise FloatingPointError(f"final covariance check failed for {stem}")
        sse_init += i0["body_velocity_sse"]
        sse_cal += c["body_velocity_sse"]
        rows_total += c["rows"]
        summary["rollouts"][stem] = {
            "manifest_split_label": rollouts[stem].split_label,
            "rows": c["rows"],
            "body_velocity_rmse_initial_mps": i0["body_velocity_rmse_mps"],
            "body_velocity_rmse_calibrated_mps": c["body_velocity_rmse_mps"],
            "orientation_mean_deg": c["orientation_mean_deg"],
            "orientation_max_deg": c["orientation_max_deg"],
            "position_rmse_m": c["position_rmse_m"],
            "position_final_error_m": c["position_final_error_m"],
            "nis_per_measurement_dim": c["nis_per_measurement_dim"],
            "final_P_min_eig": c["final_P_min_eig"],
            "final_P_sym_residual": c["final_P_sym"],
            "jitter_events": c["jitter_events"],
        }
    summary["aggregate_body_velocity_rmse_initial_mps"] = float(
        np.sqrt(sse_init / rows_total))
    summary["aggregate_body_velocity_rmse_calibrated_mps"] = float(
        np.sqrt(sse_cal / rows_total))
    return summary
