"""Gather-based _insert vs the clone/where/blockdiag reference.

The dense form below is the pre-refactor _insert verbatim; the gather form
must match it in values and gradients for every insert pattern, including
same-row multi-slot insertion cross terms (the Hartley-convention trap:
row copy -> column copy of the row-updated matrix -> masked diagonal add).
"""

from __future__ import annotations

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
    _sym,
    exp_so3,
)

from conftest import requires_cuda


def _insert_dense(state, p_meas, insert_mask, R_pre, N_blk):
    """Pre-refactor _insert (parity oracle for the gather rewrite)."""
    X, P = state.X, state.P
    B = X.shape[0]
    dtype = X.dtype
    m = insert_mask[:, :, None]
    p_post = X[:, 0:3, 4]

    new_cols = p_post[:, None] + torch.einsum("bik,bsk->bsi", R_pre, p_meas)
    X_new = X.clone()
    X_new[:, 0:3, 5:DIM_X] = torch.where(
        m.transpose(1, 2), new_cols.transpose(1, 2), X[:, 0:3, 5:DIM_X])

    mr = insert_mask[:, :, None, None]
    P1 = P.clone()
    rows = P[:, 9:GROUP].reshape(B, N_SLOTS, 3, DIM_P)
    P1[:, 9:GROUP] = torch.where(
        mr, P[:, 6:9, :][:, None], rows).reshape(B, GROUP - 9, DIM_P)
    P2 = P1.clone()
    cols = P1[:, :, 9:GROUP].reshape(B, DIM_P, N_SLOTS, 3)
    P2[:, :, 9:GROUP] = torch.where(
        mr.permute(0, 2, 1, 3), P1[:, :, 6:9][:, :, None], cols
    ).reshape(B, DIM_P, GROUP - 9)
    add = _slot_blockdiag(N_blk[:, None] * m[:, :, None].to(dtype))
    P2[:, 9:GROUP, 9:GROUP] = P2[:, 9:GROUP, 9:GROUP] + add
    return State(X_new, state.theta, _sym(P2),
                 state.jitter_count, state.info_count)


def _random_inputs(device, B, seed=0, requires_grad=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64).to(device)
    X = _const("I13", torch.float64, device).expand(B, -1, -1).clone()
    X[:, 0:3, 0:3] = exp_so3(rnd(B, 3))
    X[:, 0:3, 3:DIM_X] = rnd(B, 3, DIM_X - 3)
    theta = 0.1 * rnd(B, fsi.DIM_THETA)
    A = rnd(B, DIM_P, DIM_P)
    P = _sym(A @ A.transpose(1, 2) / DIM_P
             + 0.1 * _const("I39", torch.float64, device))
    p_meas = rnd(B, N_SLOTS, 3)
    R_pre = exp_so3(rnd(B, 3))
    Nb = rnd(B, 3, 3)
    N_blk = Nb @ Nb.transpose(1, 2) / 3
    for t in (P, p_meas, X, N_blk):
        t.requires_grad_(requires_grad)
    zero = torch.zeros(B, dtype=torch.float64, device=device)
    return State(X, theta, P, zero, zero.clone()), p_meas, R_pre, N_blk


def _mask_patterns(device, B):
    """0..8 active, scattered, per-row-mixed, and adjacent multi-insert
    (cross-term) patterns."""
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


@requires_cuda
def test_gather_matches_dense_all_patterns(device):
    B = 5
    for i, mask in enumerate(_mask_patterns(device, B)):
        state, p_meas, R_pre, N_blk = _random_inputs(device, B, seed=i)
        with torch.no_grad():
            out_g = fsi._insert(state, p_meas, mask, R_pre, N_blk)
            out_d = _insert_dense(state, p_meas, mask, R_pre, N_blk)
        for name in ("X", "theta", "P", "jitter_count", "info_count"):
            d = float((getattr(out_g, name) - getattr(out_d, name)).abs().max())
            assert d <= 1e-14, (i, name, d)


@requires_cuda
def test_gather_gradients_match_dense(device):
    B = 3
    for i, mask in enumerate(_mask_patterns(device, B)[::3]):
        weights = None

        def loss_of(fn, seed):
            nonlocal weights
            state, p_meas, R_pre, N_blk = _random_inputs(
                device, B, seed=seed, requires_grad=True)
            out = fn(state, p_meas, mask, R_pre, N_blk)
            if weights is None:
                g = torch.Generator().manual_seed(99)
                weights = [torch.rand(t.shape, generator=g,
                                      dtype=torch.float64).to(device)
                           for t in (out.X, out.P)]
            wX, wP = weights
            loss = (out.X * wX).sum() + (out.P * wP).sum()
            inputs = (state.P, state.X, p_meas, N_blk)
            return torch.autograd.grad(loss, inputs)

        grads_g = loss_of(fsi._insert, seed=i)
        grads_d = loss_of(_insert_dense, seed=i)
        for gg, gd, name in zip(grads_g, grads_d, ("P", "X", "p_meas", "N_blk")):
            assert torch.isfinite(gg).all(), (i, name)
            d = float((gg - gd).abs().max())
            scale = max(1.0, float(gd.abs().max()))
            assert d <= 1e-12 * scale, (i, name, d, scale)


@requires_cuda
def test_all_inactive_is_exact_noop(device):
    B = 4
    state, p_meas, R_pre, N_blk = _random_inputs(device, B, seed=42)
    mask = torch.zeros(B, N_SLOTS, dtype=torch.bool, device=device)
    with torch.no_grad():
        out = fsi._insert(state, p_meas, mask, R_pre, N_blk)
    assert torch.equal(out.X, state.X)
    assert torch.equal(out.theta, state.theta)
    assert torch.equal(out.P, _sym(state.P))
