"""Benchmark / profile the InEKF replay: dynamic (baseline) vs fixed-slot.

Reports fwd and fwd+bwd ms/step, rows/s, peak GPU memory, and (with --trace)
exports a torch.profiler chrome trace plus per-row counts of kernel launches,
stream synchronizations, and D2H/H2D copies parsed from the trace.

"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from estimation_calibration_cuda.covariance_calibration import (
    CalibrationConfig,
    build_covs,
    covariance_regularization,
    fixed_initial_covariance,
    load_rollout,
    make_cov_modules,
    seed_state,
)
from estimation_calibration_cuda.invariant_ekf import (
    detach_filter,
    run_rows,
    start_filter,
)

DATA_ROOT = Path("/home/dlc/projects/Estimation-Calibration/data/datasets_v0")
DEFAULT_STEM = "dance1_subject1_20260623_173019"

LAUNCH_EVENTS = {"cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC"}
GRAPH_LAUNCH_EVENTS = {"cudaGraphLaunch", "cuGraphLaunch"}
GRAPH_INSTANTIATE_EVENTS = {
    "cudaGraphInstantiate", "cudaGraphInstantiateWithFlags",
    "cuGraphInstantiate", "cuGraphInstantiateWithFlags",
}
SYNC_EVENTS = {
    "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize",
    "cudaMemcpyAsync",  # counted separately below, kept here for visibility
}


def load_manifest_stems(data_root: Path) -> list[str]:
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    stems = [Path(e["dataset_path"]).stem for e in manifest]
    return sorted(s for s in stems if (data_root / f"{s}.npz").exists())


class DynamicRunner:
    """Training-shaped chunk on the dynamic-dimension InvariantEKF."""

    def __init__(self, roll, modules, config, P0_fixed):
        self.roll = roll
        self.modules = modules
        self.config = config
        self.P0_fixed = P0_fixed
        self.reset()

    def reset(self):
        roll = self.roll
        covs, R_kin = build_covs(self.modules)
        X0, theta0, P0 = seed_state(roll, roll.trim0, self.P0_fixed)
        self.filt = start_filter(X0, theta0, P0, covs, roll.flags[roll.trim0],
                                 roll.p_BC[roll.trim0], R_kin,
                                 s_jitter=self.config.s_jitter)
        self.R_kin = R_kin
        self.cursor = roll.trim0 + 1

    def chunk(self, rows: int, with_grad: bool) -> torch.Tensor:
        roll = self.roll
        a = self.cursor
        b = min(a + rows, roll.trim1)
        if b - a < rows:  # wrap for long benchmark loops
            self.reset()
            a, b = self.cursor, self.cursor + rows
        covs, R_kin = build_covs(self.modules)
        self.filt.covs = covs
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            out = run_rows(
                self.filt, roll.imu[a:b], roll.dt, roll.p_BC[a:b],
                None, None, R_kin,
                collect_nis=with_grad, changes_list=roll.changes[a:b],
            )
            v_B = torch.einsum("tji,tj->ti", out["R_WB"], out["v_W"])
            loss = ((v_B - roll.gt_v_B[a:b]) ** 2).sum(-1).mean()
            if with_grad:
                reg, _ = covariance_regularization(
                    self.modules, out["nis_values"], out["nis_dims"],
                    device=loss.device)
                loss = loss + reg
        self.cursor = b
        detach_filter(self.filt)
        return loss


class FixedRunner:
    """Training-shaped chunk on the fixed-slot batched implementation."""

    def __init__(self, rolls, modules, config, P0_fixed, compile_mode, dtype):
        from estimation_calibration_cuda import fixed_slot_inekf as fsi
        self.fsi = fsi
        self.rolls = rolls
        self.modules = modules
        self.config = config
        self.P0_fixed = P0_fixed
        self.dtype = dtype
        self.step_fn = fsi.make_compiled_step(compile_mode)
        self.reset()

    def reset(self):
        fsi = self.fsi
        rolls = self.rolls
        device = rolls[0].imu.device
        T_pad = max(r.trim1 - r.trim0 for r in rolls)
        self.batch = fsi.build_batch(rolls, T_pad=T_pad, dtype=self.dtype)
        covs, R_kin = build_covs(self.modules)
        states = []
        for r in rolls:
            X0, theta0, P0 = seed_state(r, r.trim0, self.P0_fixed)
            states.append((X0.to(self.dtype), theta0.to(self.dtype),
                           P0.to(self.dtype)))
        self.state = fsi.init_state(states, device=device, dtype=self.dtype)
        self.state = fsi.apply_row0(
            self.state, self.batch.p_meas[:, 0], self.batch.insert_mask[:, 0],
            R_kin.to(self.dtype))
        self.cursor = 1

    def chunk(self, rows: int, with_grad: bool) -> torch.Tensor:
        fsi = self.fsi
        a = self.cursor
        b = a + rows
        if b > self.batch.T_pad:
            self.reset()
            a, b = self.cursor, self.cursor + rows
        covs, R_kin = build_covs(self.modules)
        covs = {k: v.to(self.dtype) for k, v in covs.items()}
        R_kin = R_kin.to(self.dtype)
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            self.state, out = fsi.run_rows_fixed(
                self.state, self.batch, slice(a, b), covs, R_kin,
                s_jitter=self.config.s_jitter, step_fn=self.step_fn)
            v_B = torch.einsum("btji,btj->bti", out["R_WB"], out["v_W"])
            se = ((v_B - self.batch.gt_v_B[:, a:b]) ** 2).sum(-1)
            valid = self.batch.valid[:, a:b]
            loss = (se * valid).sum() / valid.sum()
            if with_grad:
                reg, _ = covariance_regularization(
                    self.modules, [], [], device=loss.device)
                nis_reg = fsi.reg_nis_masked(out["nis"], out["nis_dim"])
                loss = loss + reg + 1e-3 * nis_reg
        self.cursor = b
        self.state = fsi.detach_state(self.state)
        return loss


def _median_stats(wall_times: list[float], rows: int, chunks: int,
                  batch: int) -> dict:
    """median/min/max ms_per_step and rows_per_s over repeated timed passes."""
    n_rows = rows * chunks
    ms = sorted(1e3 * dt / n_rows for dt in wall_times)
    med = float(np.median(ms))
    return {
        "repeats": len(ms),
        "ms_per_step": med,
        "ms_per_step_min": ms[0],
        "ms_per_step_max": ms[-1],
        "rows_per_s": 1e3 * batch / med,
        "batch": batch,
        "wall_s": sum(wall_times),
    }


def timed(runner, rows, chunks, with_grad, params, repeat: int = 5) -> dict:
    # warmup (also triggers compilation)
    loss = runner.chunk(rows, with_grad)
    if with_grad:
        loss.backward()
        for p in params:
            p.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(chunks):
            loss = runner.chunk(rows, with_grad)
            if with_grad:
                loss.backward()
                for p in params:
                    p.grad = None
        torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - t0)
    batch = getattr(getattr(runner, "batch", None), "B", 1)
    result = _median_stats(wall_times, rows, chunks, batch)
    result["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return result


def profile_trace(runner, rows, with_grad, params, trace_path: Path) -> dict:
    from torch.profiler import ProfilerActivity, profile

    loss = runner.chunk(rows, with_grad)  # warmup outside profiler
    if with_grad:
        loss.backward()
        for p in params:
            p.grad = None
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as prof:
        loss = runner.chunk(rows, with_grad)
        if with_grad:
            loss.backward()
            for p in params:
                p.grad = None
        torch.cuda.synchronize()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    return parse_trace(trace_path, rows)


def parse_trace(trace_path: Path, rows: int, chunks: int = 1) -> dict:
    events = json.loads(trace_path.read_text()).get("traceEvents", [])
    launches = syncs = d2h = h2d = 0
    graph_launches = graph_instantiates = kernels = 0
    kernel_time_us = 0.0
    cpu_op_time_us = 0.0
    for e in events:
        name = e.get("name", "")
        cat = e.get("cat", "")
        if name in LAUNCH_EVENTS:
            launches += 1
        elif name in GRAPH_LAUNCH_EVENTS:
            graph_launches += 1
        elif name in GRAPH_INSTANTIATE_EVENTS:
            graph_instantiates += 1
        elif name in ("cudaStreamSynchronize", "cudaDeviceSynchronize",
                      "cudaEventSynchronize"):
            syncs += 1
        if cat == "gpu_memcpy":
            if "DtoH" in name:
                d2h += 1
            elif "HtoD" in name:
                h2d += 1
        elif cat == "kernel":
            kernels += 1
            kernel_time_us += e.get("dur", 0)
        elif cat == "cuda_runtime":
            cpu_op_time_us += e.get("dur", 0)
    return {
        "trace": str(trace_path),
        "rows_profiled": rows,
        "launches_per_row": launches / rows,
        "graph_launches_per_chunk": graph_launches / chunks,
        "graph_launches_per_row": graph_launches / rows,
        "graph_instantiates": graph_instantiates,
        "kernels_per_row": kernels / rows,
        "syncs_total": syncs,
        "d2h_per_row": d2h / rows,
        "h2d_per_row": h2d / rows,
        "gpu_kernel_ms": kernel_time_us / 1e3,
        "cpu_cuda_api_ms": cpu_op_time_us / 1e3,
    }


def run_cuda_graph_bench(args, modules, params, config, P0_fixed, device):
    """Time whole-chunk CUDA-graph replays (fwd+bwd) and export a trace."""
    from estimation_calibration_cuda import fixed_slot_inekf as fsi
    from estimation_calibration_cuda.batched_calibration import ChunkGraph
    from estimation_calibration_cuda.covariance_calibration import seed_state

    stems = load_manifest_stems(args.data_root)[:args.batch] \
        if args.batch > 1 else [args.stem]
    rolls = [load_rollout(args.data_root, s, "bench", config=config,
                          device=device) for s in stems]
    batch = fsi.build_batch(rolls)
    with torch.no_grad():
        covs0, R_kin0 = build_covs(modules)
        state0 = fsi.init_state(
            [seed_state(r, r.trim0, P0_fixed) for r in rolls], device=device)
        state0 = fsi.apply_row0(state0, batch.p_meas[:, 0],
                                batch.insert_mask[:, 0], R_kin0)
    step_fn = None
    if args.compile_mode == "cuda-graph-compile":
        from estimation_calibration_cuda.fixed_slot_inekf import (
            make_compiled_step)
        step_fn = make_compiled_step("default")
    graph = ChunkGraph(modules, params, batch, chunk=args.rows,
                       s_jitter=config.s_jitter, dtype=torch.float64,
                       state0=state0, step_fn=step_fn)
    graph.replay_chunk(1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall_times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        for i in range(args.chunks):
            graph.replay_chunk(1 + (i % 3) * args.rows)
        torch.cuda.synchronize()
        wall_times.append(time.perf_counter() - t0)
    tag = f"fixed_b{batch.B}_c{args.compile_mode}_float64_grad"
    result = {"tag": tag, "rows": args.rows, "chunks": args.chunks}
    result.update(_median_stats(wall_times, args.rows, args.chunks, batch.B))
    result["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    if args.trace:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            graph.replay_chunk(1)
            torch.cuda.synchronize()
        trace_path = args.out / f"{tag}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
        result.update(parse_trace(trace_path, args.rows, chunks=1))
    print(json.dumps(result, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{tag}.summary.json").write_text(json.dumps(result, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impl", choices=["dynamic", "fixed"], default="dynamic")
    ap.add_argument("--compile", dest="compile_mode", default="none",
                    choices=["none", "default", "reduce-overhead",
                             "max-autotune", "cuda-graph",
                             "cuda-graph-compile"])
    ap.add_argument("--batch", type=int, default=1, help="rollouts in batch (fixed impl)")
    ap.add_argument("--rows", type=int, default=200, help="rows per chunk")
    ap.add_argument("--chunks", type=int, default=3, help="timed chunks")
    ap.add_argument("--repeat", type=int, default=5,
                    help="timed passes; ms_per_step is the median")
    ap.add_argument("--with-grad", action="store_true")
    ap.add_argument("--trace", action="store_true", help="export chrome trace")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--stem", default=DEFAULT_STEM)
    ap.add_argument("--out", type=Path, default=Path("runs/profiles"))
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    dtype = {"float64": torch.float64, "float32": torch.float32}[args.dtype]
    config = CalibrationConfig()
    torch.manual_seed(0)
    modules = make_cov_modules(device=device, dtype=torch.float64)
    params = list(modules.parameters())
    P0_fixed = fixed_initial_covariance(device)

    if args.impl == "dynamic":
        roll = load_rollout(args.data_root, args.stem, "bench",
                            config=config, device=device)
        runner = DynamicRunner(roll, modules, config, P0_fixed)
    elif args.compile_mode in ("cuda-graph", "cuda-graph-compile"):
        run_cuda_graph_bench(args, modules, params, config, P0_fixed, device)
        return
    else:
        stems = load_manifest_stems(args.data_root)[:args.batch] \
            if args.batch > 1 else [args.stem]
        rolls = [load_rollout(args.data_root, s, "bench", config=config,
                              device=device) for s in stems]
        runner = FixedRunner(rolls, modules, config, P0_fixed,
                             None if args.compile_mode == "none" else args.compile_mode,
                             dtype)

    tag = (f"{args.impl}_b{args.batch}_c{args.compile_mode}_{args.dtype}"
           f"_{'grad' if args.with_grad else 'fwd'}")
    result = {"tag": tag, "rows": args.rows, "chunks": args.chunks}
    result.update(timed(runner, args.rows, args.chunks, args.with_grad, params,
                        repeat=args.repeat))
    if args.trace:
        trace_rows = min(args.rows, 100)
        result.update(profile_trace(runner, trace_rows, args.with_grad, params,
                                    args.out / f"{tag}.json"))
    print(json.dumps(result, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{tag}.summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
