"""Block-structured _correct vs the dense H/N reference implementation.

The dense form below is the pre-refactor _correct verbatim; the block form
must match it in values and gradients for every active-slot pattern.
"""

from __future__ import annotations

import pytest
import torch

from estimation_calibration_cuda import fixed_slot_inekf as fsi
from estimation_calibration_cuda.fixed_slot_inekf import (
    DIM_M,
    DIM_P,
    DIM_X,
    GROUP,
    N_SLOTS,
    State,
    _const,
    _slot_blockdiag,
    _sym,
    exp_sek3,
    exp_so3,
)

from conftest import requires_cuda

S_JITTER = 1e-10


def _correct_dense(state, p_meas, correct_mask, R_kin, s_jitter):
    """Pre-refactor dense _correct (parity oracle for the block rewrite)."""
    X, theta, P = state.X, state.theta, state.P
    B = X.shape[0]
    dtype, device = X.dtype, X.device
    R_pre = X[:, 0:3, 0:3]
    m = correct_mask.to(dtype)
    mrow = m.repeat_interleave(3, dim=1)

    H = _const("H", dtype, device) * mrow[:, :, None]
    N_blk = R_pre @ R_kin @ R_pre.transpose(1, 2)
    I3 = _const("I3", dtype, device)
    slot_N = torch.where(correct_mask[:, :, None, None],
                         N_blk[:, None], I3.expand(B, N_SLOTS, 3, 3))
    N = _slot_blockdiag(slot_N)

    Z = (torch.einsum("bik,bsk->bsi", R_pre, p_meas)
         + X[:, 0:3, 4][:, None] - X[:, 0:3, 5:DIM_X].transpose(1, 2))
    Z = (Z * m[:, :, None]).reshape(B, DIM_M)

    PHT = P @ H.transpose(1, 2)
    S = H @ PHT + N
    jitter_count = state.jitter_count
    if s_jitter > 0.0:
        jitter_count = jitter_count + (
            torch.diagonal(S, dim1=1, dim2=2).min(dim=1).values
            < 10.0 * s_jitter).to(dtype).detach()
        S = S + s_jitter * _const("I24", dtype, device)
    L, info = torch.linalg.cholesky_ex(S, check_errors=False)
    K = torch.cholesky_solve(PHT.transpose(1, 2), L).transpose(1, 2)
    info_count = state.info_count + (info != 0).to(dtype).detach()

    delta = torch.einsum("bij,bj->bi", K, Z)
    dX = exp_sek3(delta[:, :GROUP])
    X_new = dX @ X
    theta_new = theta + delta[:, GROUP:]
    IKH = _const("I39", dtype, device) - K @ H
    P_new = _sym(IKH @ P @ IKH.transpose(1, 2)
                 + K @ N @ K.transpose(1, 2))
    nis = (Z[:, :, None] * torch.cholesky_solve(Z[:, :, None], L)).sum((1, 2))
    return State(X_new, theta_new, P_new, jitter_count, info_count), nis, N_blk


def _random_inputs(device, B, seed=0, requires_grad=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dd = {"device": device, "dtype": torch.float64}
    rnd = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64).to(device)
    X = _const("I13", torch.float64, device).expand(B, -1, -1).clone()
    X[:, 0:3, 0:3] = exp_so3(rnd(B, 3))
    X[:, 0:3, 3:DIM_X] = rnd(B, 3, DIM_X - 3)
    theta = 0.1 * rnd(B, fsi.DIM_THETA)
    A = rnd(B, DIM_P, DIM_P)
    P = _sym(A @ A.transpose(1, 2) / DIM_P
             + 0.1 * _const("I39", torch.float64, device))
    p_meas = rnd(B, N_SLOTS, 3)
    Ak = rnd(3, 3)
    R_kin = Ak @ Ak.T / 3 + 0.05 * _const("I3", torch.float64, device)
    for t in (P, p_meas, R_kin, theta, X):
        t.requires_grad_(requires_grad)
    zero = torch.zeros(B, dtype=torch.float64, device=device)
    state = State(X, theta, P, zero, zero.clone())
    return state, p_meas, R_kin


def _mask_patterns(device, B):
    """Every active count 0..8 plus non-contiguous / per-row-mixed patterns."""
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
@pytest.mark.parametrize("s_jitter", [0.0, S_JITTER])
def test_block_matches_dense_all_patterns(device, s_jitter):
    B = 5
    for i, mask in enumerate(_mask_patterns(device, B)):
        state, p_meas, R_kin = _random_inputs(device, B, seed=i)
        with torch.no_grad():
            out_b, nis_b, Nb_b = fsi._correct(state, p_meas, mask, R_kin,
                                              s_jitter)
            out_d, nis_d, Nb_d = _correct_dense(state, p_meas, mask, R_kin,
                                                s_jitter)
        tag = f"pattern{i}_jitter{s_jitter}"
        _assert_state_close(out_b, out_d, 1e-12, tag)
        assert float((nis_b - nis_d).abs().max()) <= 1e-10, tag
        assert float((Nb_b - Nb_d).abs().max()) == 0.0, tag


@requires_cuda
def test_block_gradients_match_dense(device):
    B = 3
    for i, mask in enumerate(_mask_patterns(device, B)[::3]):
        weights = None

        def loss_of(fn, seed):
            nonlocal weights
            state, p_meas, R_kin = _random_inputs(device, B, seed=seed,
                                                  requires_grad=True)
            out, nis, _ = fn(state, p_meas, mask, R_kin, S_JITTER)
            if weights is None:
                g = torch.Generator().manual_seed(99)
                weights = [torch.rand(t.shape, generator=g,
                                      dtype=torch.float64).to(device)
                           for t in (out.X, out.theta, out.P, nis)]
            wX, wt, wP, wn = weights
            loss = ((out.X * wX).sum() + (out.theta * wt).sum()
                    + (out.P * wP).sum() + (nis * wn).sum())
            inputs = (state.P, p_meas, R_kin, state.theta, state.X)
            return torch.autograd.grad(loss, inputs)

        grads_b = loss_of(fsi._correct, seed=i)
        grads_d = loss_of(_correct_dense, seed=i)
        for gb, gd, name in zip(grads_b, grads_d,
                                ("P", "p_meas", "R_kin", "theta", "X")):
            assert torch.isfinite(gb).all(), (i, name)
            d = float((gb - gd).abs().max())
            scale = max(1.0, float(gd.abs().max()))
            assert d <= 1e-10 * scale, (i, name, d, scale)


@requires_cuda
def test_all_inactive_is_exact_noop(device):
    B = 4
    state, p_meas, R_kin = _random_inputs(device, B, seed=42)
    mask = torch.zeros(B, N_SLOTS, dtype=torch.bool, device=device)
    with torch.no_grad():
        out, nis, _ = fsi._correct(state, p_meas, mask, R_kin, S_JITTER)
    assert torch.equal(out.X, state.X)
    assert torch.equal(out.theta, state.theta)
    assert torch.equal(out.P, state.P)
    assert float(nis.abs().max()) == 0.0
