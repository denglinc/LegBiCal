"""Optional per-candidate process covariance checks."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from estimation_calibration_cuda import fixed_slot_inekf as fsi
from estimation_calibration_cuda.batched_calibration import ChunkGraph
from estimation_calibration_cuda.covariance_calibration import (
    LAMBDA,
    build_covs,
    fixed_initial_covariance,
    make_cov_modules,
    seed_state,
)
from estimation_calibration_cuda.data import _episode_to_rollout, load_dataset
from estimation_calibration_cuda.invariant_ekf import run_rows, start_filter

from conftest import dynamic_column_map, map_dynamic_P_to_fixed


def _roll(device, rows=20):
    episode = load_dataset("example").load("train")[0]
    rollout = _episode_to_rollout(episode, device=device, trim_s=0.0)
    return dataclasses.replace(rollout, trim1=rows)


def _covariances(device, *, q_grad=False):
    values = {"Qg": 1e-4, "Qa": 9e-2, "Qc": 1e-2,
              "Qbg": 1e-10, "Qba": 1e-8}
    covariances = {
        key: torch.eye(3, dtype=torch.float64, device=device) * value
        for key, value in values.items()
    }
    covariances["Qc"].requires_grad_(q_grad)
    measurement = torch.eye(3, dtype=torch.float64, device=device) * 4e-4
    return covariances, measurement


def _dynamic(rollout, covariances, measurement, marker="omitted"):
    P0 = fixed_initial_covariance(rollout.imu.device)
    X0, theta0, covariance0 = seed_state(rollout, 0, P0)
    filt = start_filter(
        X0, theta0, covariance0, covariances, rollout.flags[0],
        rollout.p_BC[0], measurement, s_jitter=1e-12)
    kwargs = {} if marker == "omitted" else {
        "contact_process_covariance": None if marker is None else marker[1:]}
    out = run_rows(
        filt, rollout.imu[1:rollout.trim1], rollout.dt,
        rollout.p_BC[1:rollout.trim1], None, None, measurement,
        changes_list=rollout.changes[1:rollout.trim1], **kwargs)
    return filt, out


def _fixed(rollout, covariances, measurement, marker="omitted", step_fn=None):
    batch = fsi.build_batch([rollout])
    P0 = fixed_initial_covariance(rollout.imu.device)
    state = fsi.init_state(
        [seed_state(rollout, 0, P0)], device=rollout.imu.device)
    state = fsi.apply_row0(
        state, batch.p_meas[:, 0], batch.insert_mask[:, 0], measurement)
    kwargs = {} if marker == "omitted" else {
        "contact_process_covariance": marker}
    state, out = fsi.run_rows_fixed(
        state, batch, slice(1, batch.T_pad), covariances, measurement,
        s_jitter=1e-12, step_fn=step_fn, **kwargs)
    return state, out, batch


@pytest.mark.parametrize("backend", ["dynamic", "fixed"])
def test_omitted_and_explicit_none_are_bitwise(backend):
    rollout = _roll(torch.device("cpu"))
    weights = torch.randn(rollout.trim1 - 1, 3, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(17))

    def one(explicit):
        covariances, measurement = _covariances(torch.device("cpu"), q_grad=True)
        if backend == "dynamic":
            state, out = _dynamic(
                rollout, covariances, measurement, None if explicit else "omitted")
            final = (state.X, state.theta, state.P)
            velocity = out["v_W"]
        else:
            state, out, _ = _fixed(
                rollout, covariances, measurement, None if explicit else "omitted")
            final = state
            velocity = out["v_W"][0]
        gradient = torch.autograd.grad((velocity * weights).sum(),
                                       covariances["Qc"])[0]
        return final, out, gradient

    first, out_first, grad_first = one(False)
    second, out_second, grad_second = one(True)
    for key in ("R_WB", "v_W", "p_W"):
        assert torch.equal(out_first[key], out_second[key])
    for left, right in zip(first, second):
        assert torch.equal(left, right)
    assert torch.equal(grad_first, grad_second)


def test_expanded_shared_override_is_bitwise_default():
    rollout = _roll(torch.device("cpu"))
    covariances, measurement = _covariances(torch.device("cpu"))
    default_state, default_out, batch = _fixed(rollout, covariances, measurement)
    shared = covariances["Qc"].expand(
        batch.B, batch.T_pad, fsi.N_SLOTS, 3, 3)
    override_state, override_out, _ = _fixed(
        rollout, covariances, measurement, shared)
    for key in ("R_WB", "v_W", "p_W", "nis"):
        assert torch.equal(default_out[key], override_out[key])
    for left, right in zip(default_state, override_state):
        assert torch.equal(left, right)


def test_heterogeneous_dynamic_fixed_value_and_gradient_parity():
    rollout = _roll(torch.device("cpu"))
    covariances, measurement = _covariances(torch.device("cpu"))
    T, N = rollout.trim1, fsi.N_SLOTS
    scales = torch.linspace(.25, 4., T * N, dtype=torch.float64).reshape(T, N)
    process = covariances["Qc"] * scales[..., None, None]
    dynamic_state, dynamic = _dynamic(rollout, covariances, measurement, process)
    fixed_state, fixed, _ = _fixed(
        rollout, covariances, measurement, process[None])
    for key in ("R_WB", "v_W", "p_W"):
        torch.testing.assert_close(
            dynamic[key], fixed[key][0], rtol=1e-11, atol=1e-12)
    positions = dynamic_column_map(rollout.flags[:T])
    dynamic_P, fixed_P = map_dynamic_P_to_fixed(
        dynamic_state.P, positions, fixed_state.P[0])
    torch.testing.assert_close(dynamic_P, fixed_P, rtol=1e-11, atol=1e-12)

    weights = torch.randn(T - 1, 3, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(23))
    dynamic_scales = scales.clone().requires_grad_()
    fixed_scales = scales.clone().requires_grad_()
    _, dynamic = _dynamic(
        rollout, covariances, measurement,
        covariances["Qc"] * dynamic_scales[..., None, None])
    _, fixed, _ = _fixed(
        rollout, covariances, measurement,
        (covariances["Qc"] * fixed_scales[..., None, None])[None])
    dynamic_grad = torch.autograd.grad(
        (dynamic["v_W"] * weights).sum(), dynamic_scales)[0]
    fixed_grad = torch.autograd.grad(
        (fixed["v_W"][0] * weights).sum(), fixed_scales)[0]
    assert torch.count_nonzero(dynamic_grad) > 0
    torch.testing.assert_close(dynamic_grad, fixed_grad, rtol=1e-10, atol=1e-12)


def _objective(state, out, batch):
    velocity = torch.einsum("btji,btj->bti", out["R_WB"], out["v_W"])
    error = ((velocity - batch.gt_v_B[:, 1:]) ** 2).sum(-1)
    body = (error * batch.valid[:, 1:]).sum() / batch.valid[:, 1:].sum()
    return body + LAMBDA["nis"] * fsi.reg_nis_masked(
        out["nis"], out["nis_dim"])


@pytest.mark.release_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["none", "default", "cuda-graph", "cuda-graph-compile"])
def test_cuda_execution_modes_preserve_values_and_gradients(mode):
    device = torch.device("cuda")
    rollout = _roll(device, rows=9)
    batch = fsi.build_batch([rollout])
    reference_modules = make_cov_modules(device=device)
    with torch.no_grad():
        reference_covariances, reference_measurement = build_covs(reference_modules)
    del reference_modules
    P0 = fixed_initial_covariance(device)
    reference_state0 = fsi.init_state(
        [seed_state(rollout, 0, P0)], device=device)
    reference_state0 = fsi.apply_row0(
        reference_state0, batch.p_meas[:, 0], batch.insert_mask[:, 0],
        reference_measurement)
    base = torch.eye(3, dtype=torch.float64, device=device) \
        .expand(1, batch.T_pad, fsi.N_SLOTS, 3, 3).clone()
    base *= torch.linspace(.002, .02, batch.T_pad * fsi.N_SLOTS,
                           dtype=torch.float64, device=device).reshape(
                               1, batch.T_pad, fsi.N_SLOTS, 1, 1)

    eager_process = base.detach().clone().requires_grad_()
    eager_state, eager_out = fsi.run_rows_fixed(
        reference_state0, batch, slice(1, batch.T_pad),
        reference_covariances, reference_measurement,
        s_jitter=1e-12, contact_process_covariance=eager_process)
    eager_loss = _objective(eager_state, eager_out, batch)
    eager_grad = torch.autograd.grad(eager_loss, eager_process)[0][:, 1:]
    expected_P = eager_state.P.detach().clone()
    expected_loss = eager_loss.detach().clone()
    expected_grad = eager_grad.detach().clone()
    del eager_state, eager_out, eager_loss, eager_grad, eager_process

    modules = make_cov_modules(device=device)
    parameters = list(modules.parameters())
    with torch.no_grad():
        covariances, measurement = build_covs(modules)
        state0 = fsi.init_state([seed_state(rollout, 0, P0)], device=device)
        state0 = fsi.apply_row0(
            state0, batch.p_meas[:, 0], batch.insert_mask[:, 0], measurement)

    process = base.detach().clone().requires_grad_()
    if mode in ("cuda-graph", "cuda-graph-compile"):
        step_fn = (fsi.make_compiled_step("default")
                   if mode == "cuda-graph-compile" else None)
        graph = ChunkGraph(
            modules, parameters, batch, chunk=batch.T_pad - 1,
            s_jitter=1e-12, dtype=torch.float64, state0=state0,
            step_fn=step_fn, contact_process_covariance=process)
        graph.load_state(state0)
        graph.replay_chunk(1)
        torch.cuda.synchronize()
        actual_P = graph.P
        actual_loss = graph.loss_body + graph.nis_term
        actual_grad = graph.process_grad
    else:
        step_fn = fsi.make_compiled_step("default") if mode == "default" else None
        state, out = fsi.run_rows_fixed(
            state0, batch, slice(1, batch.T_pad), covariances, measurement,
            s_jitter=1e-12, step_fn=step_fn,
            contact_process_covariance=process)
        actual_P = state.P
        actual_loss = _objective(state, out, batch)
        actual_grad = torch.autograd.grad(actual_loss, process)[0][:, 1:]
    torch.testing.assert_close(actual_P, expected_P, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-12)
