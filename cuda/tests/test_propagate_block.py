"""Block-structured _propagate vs the dense A/Phi/Adj/Qk reference.

The dense form below is the pre-refactor _propagate verbatim; the block form
must match it in values and gradients for every active-slot pattern, and
dt = 0 rows must stay bitwise no-ops.
"""

from __future__ import annotations

import pytest
import torch

from estimation_calibration_cuda import fixed_slot_inekf as fsi
from estimation_calibration_cuda.fixed_slot_inekf import (
    DIM_P,
    DIM_X,
    GROUP,
    N_SLOTS,
    State,
    _const,
    _slot_blockdiag,
    _slot_blockdiag_k,
    _sym,
    exp_so3,
    skew,
)

from conftest import requires_cuda


def _propagate_dense(state, gyro, accel, dt_row, prop_mask, covs):
    """Pre-refactor dense _propagate (parity oracle for the block rewrite)."""
    X, theta, P = state.X, state.theta, state.P
    B = X.shape[0]
    dtype, device = X.dtype, X.device
    R_old = X[:, 0:3, 0:3]
    v_old = X[:, 0:3, 3]
    p_old = X[:, 0:3, 4]
    dt = dt_row[:, None]

    w = gyro - theta[:, 0:3]
    a = accel - theta[:, 3:6]
    R_pred = R_old @ exp_so3(w * dt)
    acc_w = torch.einsum("bij,bj->bi", R_old, a) + _const("g", dtype, device)
    v_pred = v_old + acc_w * dt
    p_pred = p_old + v_old * dt + 0.5 * acc_w * dt * dt

    X_new = X.clone()
    X_new[:, 0:3, 0:3] = R_pred
    X_new[:, 0:3, 3] = v_pred
    X_new[:, 0:3, 4] = p_pred

    cols = X[:, 0:3, 3:DIM_X]
    SR = skew(cols.transpose(1, 2)) @ R_old[:, None]
    A = torch.zeros(B, DIM_P, DIM_P, dtype=dtype, device=device)
    A[:, 3:6, 0:3] = _const("skew_g", dtype, device)
    A[:, 6:9, 3:6] = _const("I3", dtype, device)
    A[:, 0:3, GROUP:GROUP + 3] = -R_old
    A[:, 3:6, GROUP + 3:] = -R_old
    A[:, 3:GROUP, GROUP:GROUP + 3] = -SR.reshape(B, GROUP - 3, 3)

    Qk = torch.zeros(B, DIM_P, DIM_P, dtype=dtype, device=device)
    Qk[:, 0:3, 0:3] = covs["Qg"]
    Qk[:, 3:6, 3:6] = covs["Qa"]
    slot_q = covs["Qc"] * prop_mask.to(dtype)[:, :, None, None]
    Qk[:, 9:GROUP, 9:GROUP] = _slot_blockdiag(slot_q)
    Qk[:, GROUP:GROUP + 3, GROUP:GROUP + 3] = covs["Qbg"]
    Qk[:, GROUP + 3:, GROUP + 3:] = covs["Qba"]

    I39 = _const("I39", dtype, device)
    Phi = I39 + A * dt[:, :, None]
    Adj = I39.expand(B, -1, -1).clone()
    Adj[:, 0:3, 0:3] = R_old
    Adj[:, 3:GROUP, 0:3] = SR.reshape(B, GROUP - 3, 3)
    R_diag = R_old[:, None].expand(B, DIM_X - 3, 3, 3)
    Adj[:, 3:GROUP, 3:GROUP] = _slot_blockdiag_k(R_diag)
    PhiAdj = Phi @ Adj
    Qk_hat = PhiAdj @ Qk @ PhiAdj.transpose(1, 2) * dt[:, :, None]
    P_new = _sym(Phi @ P @ Phi.transpose(1, 2) + Qk_hat)
    return State(X_new, theta, P_new, state.jitter_count, state.info_count)


def _random_inputs(device, B, seed=0, requires_grad=False, dt_mode="const"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64).to(device)
    X = _const("I13", torch.float64, device).expand(B, -1, -1).clone()
    X[:, 0:3, 0:3] = exp_so3(rnd(B, 3))
    X[:, 0:3, 3:DIM_X] = rnd(B, 3, DIM_X - 3)
    theta = 0.1 * rnd(B, fsi.DIM_THETA)
    A = rnd(B, DIM_P, DIM_P)
    P = _sym(A @ A.transpose(1, 2) / DIM_P
             + 0.1 * _const("I39", torch.float64, device))
    gyro, accel = rnd(B, 3), 3.0 * rnd(B, 3)
    if dt_mode == "const":
        dt_row = torch.full((B,), 0.005, dtype=torch.float64, device=device)
    elif dt_mode == "mixed":  # some exact-zero (padded) rows in the batch
        dt_row = 0.01 * torch.rand(B, generator=g).to(device).double()
        dt_row[::2] = 0.0
    else:
        dt_row = torch.zeros(B, dtype=torch.float64, device=device)
    covs = {}
    for k in ("Qg", "Qa", "Qc", "Qbg", "Qba"):
        Ak = rnd(3, 3)
        covs[k] = Ak @ Ak.T / 3 + 0.05 * _const("I3", torch.float64, device)
    for t in (P, theta, X, gyro, accel, *covs.values()):
        t.requires_grad_(requires_grad)
    zero = torch.zeros(B, dtype=torch.float64, device=device)
    return State(X, theta, P, zero, zero.clone()), gyro, accel, dt_row, covs


def _mask_patterns(device, B):
    pats = []
    for n in range(N_SLOTS + 1):
        row = torch.zeros(N_SLOTS, dtype=torch.bool, device=device)
        row[:n] = True
        pats.append(row.expand(B, -1).clone())
    scatter = torch.zeros(N_SLOTS, dtype=torch.bool, device=device)
    scatter[[1, 4, 7]] = True
    pats.append(scatter.expand(B, -1).clone())
    g = torch.Generator().manual_seed(7)
    pats.append(torch.rand(B, N_SLOTS, generator=g).to(device) < 0.5)
    return pats


def _assert_state_close(out_b, out_d, tol, tag):
    for name in ("X", "theta", "P", "jitter_count", "info_count"):
        d = float((getattr(out_b, name) - getattr(out_d, name)).abs().max())
        assert d <= tol, (tag, name, d)


@requires_cuda
@pytest.mark.parametrize("dt_mode", ["const", "mixed"])
def test_block_matches_dense_all_patterns(device, dt_mode):
    B = 5
    for i, mask in enumerate(_mask_patterns(device, B)):
        state, gyro, accel, dt_row, covs = _random_inputs(
            device, B, seed=i, dt_mode=dt_mode)
        with torch.no_grad():
            out_b = fsi._propagate(state, gyro, accel, dt_row, mask, covs)
            out_d = _propagate_dense(state, gyro, accel, dt_row, mask, covs)
        _assert_state_close(out_b, out_d, 1e-12, f"pattern{i}_{dt_mode}")


@requires_cuda
def test_block_gradients_match_dense(device):
    B = 3
    for i, mask in enumerate(_mask_patterns(device, B)[::3]):
        weights = None

        def loss_of(fn, seed):
            nonlocal weights
            state, gyro, accel, dt_row, covs = _random_inputs(
                device, B, seed=seed, requires_grad=True, dt_mode="mixed")
            out = fn(state, gyro, accel, dt_row, mask, covs)
            if weights is None:
                g = torch.Generator().manual_seed(99)
                weights = [torch.rand(t.shape, generator=g,
                                      dtype=torch.float64).to(device)
                           for t in (out.X, out.theta, out.P)]
            wX, wt, wP = weights
            loss = ((out.X * wX).sum() + (out.theta * wt).sum()
                    + (out.P * wP).sum())
            inputs = (state.P, state.X, state.theta, gyro, accel,
                      *covs.values())
            return torch.autograd.grad(loss, inputs)

        grads_b = loss_of(fsi._propagate, seed=i)
        grads_d = loss_of(_propagate_dense, seed=i)
        names = ("P", "X", "theta", "gyro", "accel",
                 "Qg", "Qa", "Qc", "Qbg", "Qba")
        for gb, gd, name in zip(grads_b, grads_d, names):
            assert torch.isfinite(gb).all(), (i, name)
            d = float((gb - gd).abs().max())
            scale = max(1.0, float(gd.abs().max()))
            assert d <= 1e-10 * scale, (i, name, d, scale)


@requires_cuda
def test_dt_zero_is_bitwise_noop(device):
    B = 4
    state, gyro, accel, _, covs = _random_inputs(device, B, seed=42)
    dt_row = torch.zeros(B, dtype=torch.float64, device=device)
    for mask in (torch.zeros(B, N_SLOTS, dtype=torch.bool, device=device),
                 torch.ones(B, N_SLOTS, dtype=torch.bool, device=device)):
        with torch.no_grad():
            out = fsi._propagate(state, gyro, accel, dt_row, mask, covs)
        assert torch.equal(out.X, state.X)
        assert torch.equal(out.theta, state.theta)
        assert torch.equal(out.P, state.P)
