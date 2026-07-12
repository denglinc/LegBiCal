"""Training smoke: loss decreases, gradients flow, final P sane, both modes."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from estimation_calibration_cuda import fixed_slot_inekf as fsi
from estimation_calibration_cuda.batched_calibration import (
    ChunkGraph,
    eval_batched,
    train_batched,
)
from estimation_calibration_cuda.covariance_calibration import (
    CalibrationConfig,
    build_covs,
    covariance_regularization,
    fixed_initial_covariance,
    make_cov_modules,
    seed_state,
)

from conftest import run_fixed

pytestmark = pytest.mark.external_data


def _truncated(roll, rows=1500):
    """A shortened copy of the rollout for fast training epochs."""
    return dataclasses.replace(roll, trim1=roll.trim0 + rows)


@pytest.mark.parametrize("compile_mode",
                         [None, "cuda-graph", "cuda-graph-compile"])
def test_train_batched_smoke(roll, device, compile_mode):
    config = CalibrationConfig(epochs=2, chunk=300, compile_mode=compile_mode)
    short = _truncated(roll)
    validation = dataclasses.replace(short, stem="v0", split_label="validation")
    result = train_batched(
        ["r0"], {"r0": dataclasses.replace(short, split_label="train")},
        validation_order=["v0"], validation_rollouts={"v0": validation},
        config=config, device=device)
    losses = [h["train_body_loss"] for h in result.history]
    assert all(torch.isfinite(torch.tensor(x)) for x in losses)
    assert losses[-1] < losses[0], losses
    grads = [result.history[-1]["groups"][n]["grad_norm_mean"]
             for n in ("kin_meas", "contact_proc")]
    assert all(g > 0 for g in grads)
    # final covariance gates via the batched evaluator
    covs, R_kin = build_covs(result.modules)
    P0_fixed = fixed_initial_covariance(device)
    res = eval_batched(["r0"], {"r0": short},
                       {k: v.detach() for k, v in covs.items()},
                       R_kin.detach(), P0_fixed=P0_fixed,
                       s_jitter=config.s_jitter)["r0"]
    assert res["finite"]
    assert res["final_P_sym"] < 1e-9
    assert res["final_P_min_eig"] > -1e-12


@pytest.mark.parametrize("compiled", [False, True])
def test_graph_grads_match_eager(roll, device, config, P0_fixed, compiled):
    """One captured chunk reproduces eager grads (body + NIS part).

    The compiled-step capture tolerates inductor's reassociated float ops
    (fusion changes reduction order); the plain capture must be exact."""
    torch.manual_seed(0)
    modules = make_cov_modules(device=device)
    params = list(modules.parameters())
    short = _truncated(roll, rows=901)
    batch = fsi.build_batch([short])
    with torch.no_grad():
        covs0, R_kin0 = build_covs(modules)
        state0 = fsi.init_state([seed_state(short, short.trim0, P0_fixed)],
                                device=device)
        state0 = fsi.apply_row0(state0, batch.p_meas[:, 0],
                                batch.insert_mask[:, 0], R_kin0)
    step_fn = fsi.make_compiled_step("default") if compiled else None
    graph = ChunkGraph(modules, params, batch, chunk=300,
                       s_jitter=config.s_jitter, dtype=torch.float64,
                       state0=state0, step_fn=step_fn)
    graph.load_state(state0)
    graph.replay_chunk(1)
    torch.cuda.synchronize()
    graph_grads = [g.clone() for g in graph.grads]

    covs, R_kin = build_covs(modules)
    st = fsi.State(state0.X, state0.theta, state0.P, state0.jitter_count,
                   state0.info_count)
    st, out = fsi.run_rows_fixed(st, batch, slice(1, 301), covs, R_kin,
                                 s_jitter=config.s_jitter)
    v_B = torch.einsum("btji,btj->bti", out["R_WB"], out["v_W"])
    se = ((v_B - batch.gt_v_B[:, 1:301]) ** 2).sum(-1)
    valid = batch.valid[:, 1:301]
    loss = (se * valid).sum() / valid.sum().clamp_min(1)
    from estimation_calibration_cuda.covariance_calibration import LAMBDA
    loss = loss + LAMBDA["nis"] * fsi.reg_nis_masked(out["nis"], out["nis_dim"])
    eager_grads = torch.autograd.grad(loss, params)
    for gg, eg in zip(graph_grads, eager_grads):
        assert torch.isfinite(gg).all()
        d = float((gg - eg).abs().max())
        tol = 1e-9 * max(1.0, float(eg.abs().max())) if compiled else 1e-12
        assert d <= tol, (d, tol)
