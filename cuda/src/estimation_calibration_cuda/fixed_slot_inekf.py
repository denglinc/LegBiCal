"""Fixed-slot, static-shape, batched differentiable contact-aided InEKF.

Same math as ``invariant_ekf.InvariantEKF`` (the parity oracle), restructured
for GPU execution: all 8 contact candidate slots are always materialized
(X: (B, 13, 13), P: (B, 39, 39)), contact insertion/removal becomes masked
tensor ops driven by a schedule precomputed on the host, and every step is a
pure tensor->tensor function with static shapes and no data-dependent Python
control flow -- so it batches over rollouts and is torch.compile / CUDA-graph
friendly.

Equivalence to the dynamic-dimension filter (verified by tests/test_parity.py):
- Inactive slots never leak into active blocks: no row of A reads contact
  columns, H columns at inactive slots are zero (so K columns are exactly 0),
  and insertion overwrites the slot's full P row/column.
- Removal is a pure mask clear; insertion reproduces F P F^T + G cov G^T via
  row copy -> column copy (of the row-updated matrix) -> masked diagonal add,
  which also gets same-row multi-insertion cross terms right.
- Rows with no active measurement are exact no-ops (K = 0, exp(0) = I), and
  dt = 0 rows are bitwise no-op propagates -- used for batch padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np
import torch

N_SLOTS = 8
DIM_X = 5 + N_SLOTS            # 13
DIM_THETA = 6
GROUP = 3 * DIM_X - 6          # 33: rotation/velocity/position + slot errors
DIM_P = GROUP + DIM_THETA      # 39
DIM_M = 3 * N_SLOTS            # 24 stacked measurement rows

_SMALL_ANGLE = 1e-4
_CONST_CACHE: dict[tuple, torch.Tensor] = {}


def _const(name: str, dtype, device) -> torch.Tensor:
    """Cached read-only constant tensors (identities, H template, skew(g))."""
    key = (name, dtype, str(device))
    t = _CONST_CACHE.get(key)
    if t is None:
        if name == "I3":
            t = torch.eye(3, dtype=dtype, device=device)
        elif name == "I13":
            t = torch.eye(DIM_X, dtype=dtype, device=device)
        elif name == "I24":
            t = torch.eye(DIM_M, dtype=dtype, device=device)
        elif name == "I39":
            t = torch.eye(DIM_P, dtype=dtype, device=device)
        elif name == "H":
            # slot m rows 3m:3m+3: -I at position error (6:9), +I at slot error
            t = torch.zeros(DIM_M, DIM_P, dtype=dtype, device=device)
            I3 = _const("I3", dtype, device)
            for m in range(N_SLOTS):
                t[3 * m:3 * m + 3, 6:9] = -I3
                t[3 * m:3 * m + 3, 9 + 3 * m:12 + 3 * m] = I3
        elif name == "skew_g":
            t = skew(torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device))
        elif name == "g":
            t = torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device)
        elif name == "skew_W":
            # (3, 9) so that (v @ skew_W).reshape(..., 3, 3) == skew(v)
            t = torch.zeros(3, 9, dtype=dtype, device=device)
            t[2, 1], t[1, 2] = -1.0, 1.0
            t[2, 3], t[0, 5] = 1.0, -1.0
            t[1, 6], t[0, 7] = -1.0, 1.0
        elif name == "prop_basis":
            # (99, 2*39*39): placement basis mapping the batch-dependent
            # entries vals = [R (9), SR (90)] into A (slot 0) and Adj (slot 1);
            # consumed via _gather_const as one gather + sign multiply that
            # replaces every zeros+slice-assign in the A/Adj build
            t = torch.zeros(99, 2 * DIM_P * DIM_P, dtype=dtype, device=device)
            eA = lambda r, c: r * DIM_P + c
            eJ = lambda r, c: DIM_P * DIM_P + r * DIM_P + c
            for i in range(3):
                for j in range(3):
                    v = 3 * i + j                          # R[i, j]
                    t[v, eA(i, GROUP + j)] = -1.0          # A[0:3, 33:36] = -R
                    t[v, eA(3 + i, GROUP + 3 + j)] = -1.0  # A[3:6, 36:39] = -R
                    t[v, eJ(i, j)] = 1.0                   # Adj[0:3, 0:3] = R
                    for k in range(DIM_X - 3):             # Adj blockdiag(R)
                        t[v, eJ(3 + 3 * k + i, 3 + 3 * k + j)] = 1.0
            for k in range(DIM_X - 3):
                for i in range(3):
                    for j in range(3):
                        v = 9 + 9 * k + 3 * i + j          # SR[k][i, j]
                        t[v, eA(3 + 3 * k + i, GROUP + j)] = -1.0
                        t[v, eJ(3 + 3 * k + i, j)] = 1.0
        elif name == "prop_const":
            # constant parts of (A, Adj) stacked to match prop_basis output
            t = torch.zeros(2, DIM_P, DIM_P, dtype=dtype, device=device)
            t[0, 3:6, 0:3] = _const("skew_g", dtype, device)
            t[0, 6:9, 3:6] = _const("I3", dtype, device)
            t[1, GROUP:, GROUP:] = torch.eye(6, dtype=dtype, device=device)
        elif name == "ins_head":
            t = torch.arange(9, device=device)
        elif name == "ins_tail":
            t = torch.arange(GROUP, DIM_P, device=device)
        elif name == "ins_pair":
            # (2, 8, 3) long: [0] = position rows 6,7,8 per slot (the insert
            # source), [1] = each slot's own rows 9..32 (the keep source)
            t = torch.stack([
                torch.arange(6, 9, device=device).repeat(N_SLOTS, 1),
                torch.arange(9, GROUP, device=device).reshape(N_SLOTS, 3)])
        elif name == "Z39_6":
            t = torch.zeros(DIM_P, 6, dtype=dtype, device=device)
        elif name == "corr_basis":
            # (72, 24*24): placement basis for the 3x3 slot-diagonal
            # of the innovation covariance S
            t = torch.zeros(72, DIM_M * DIM_M, dtype=dtype, device=device)
            for k in range(N_SLOTS):
                for i in range(3):
                    for j in range(3):
                        t[9 * k + 3 * i + j,
                          (3 * k + i) * DIM_M + 3 * k + j] = 1.0
        elif name == "ins_basis":
            # (72, 39*39): placement basis for the masked N_blk
            # copies on the slot-diagonal 3x3 blocks
            t = torch.zeros(72, DIM_P * DIM_P, dtype=dtype, device=device)
            for k in range(N_SLOTS):
                for i in range(3):
                    for j in range(3):
                        t[9 * k + 3 * i + j,
                          (9 + 3 * k + i) * DIM_P + 9 + 3 * k + j] = 1.0
        elif name == "qk_basis":
            # (108, 39*39): placement basis for the block-diagonal Qk
            # from vals = [Qg, Qa, slot_q (8 blocks), Qbg, Qba]
            t = torch.zeros(108, DIM_P * DIM_P, dtype=dtype, device=device)
            starts = [0, 3] + [9 + 3 * k for k in range(N_SLOTS)] \
                + [GROUP, GROUP + 3]
            for b, r0 in enumerate(starts):
                for i in range(3):
                    for j in range(3):
                        t[9 * b + 3 * i + j, (r0 + i) * DIM_P + r0 + j] = 1.0
        else:
            raise KeyError(name)
        _CONST_CACHE[key] = t
    return t


def _gather_const(prefix: str, dtype, device):
    """(idx, sign, inv_idx, inv_sign) form of a `<prefix>_basis` placement.

    Forward assembles with index_select + sign multiply; (inv_idx, inv_sign)
    is the padded inverse map (value k appears at positions inv_idx[k, :])
    so the backward is gather + mul + sum -- no scatter/atomics and no
    GEMM: both the (99, 3042) fp64 GEMM (~58 us on GeForce's crippled DGEMM
    path) and indexing_backward (~100 us deterministic index_put) measured
    orders of magnitude slower than these ~2 us kernels."""
    key = (prefix + "_gather", dtype, str(device))
    t = _CONST_CACHE.get(key)
    if t is None:
        basis = _const(prefix + "_basis", dtype, device)
        idx = basis.abs().argmax(0)
        sign = basis.gather(0, idx[None]).squeeze(0)
        idx = torch.where(sign == 0, torch.zeros_like(idx), idx)
        counts = (basis != 0).sum(1)
        D = int(counts.max())
        K = basis.shape[0]
        inv_idx = torch.zeros(K, D, dtype=torch.long, device=device)
        inv_sign = torch.zeros(K, D, dtype=dtype, device=device)
        for k in range(K):
            pos = (basis[k] != 0).nonzero(as_tuple=True)[0]
            inv_idx[k, :len(pos)] = pos
            inv_sign[k, :len(pos)] = basis[k, pos]
        t = (idx, sign, inv_idx.reshape(-1), inv_sign)
        _CONST_CACHE[key] = t
    return t


class _AssembleFn(torch.autograd.Function):
    """out[b, m] = vals[b, idx[m]] * sign[m]; grad via the inverse map."""

    @staticmethod
    def forward(ctx, vals, idx, sign, inv_idx, inv_sign):
        ctx.consts = (inv_idx, inv_sign)
        return torch.index_select(vals, 1, idx) * sign

    @staticmethod
    def backward(ctx, grad):
        inv_idx, inv_sign = ctx.consts
        g = torch.index_select(grad, 1, inv_idx)
        return (g.reshape(grad.shape[0], *inv_sign.shape) * inv_sign).sum(-1), \
            None, None, None, None


def _assemble(prefix: str, vals: torch.Tensor, *shape: int) -> torch.Tensor:
    """Scatter batch-dependent entries into a constant layout. Empty
    positions read vals[:, 0] with sign 0, giving exact +/-0 -- additions
    of signed zero never change a value, so no-op guarantees survive."""
    consts = _gather_const(prefix, vals.dtype, vals.device)
    return _AssembleFn.apply(vals, *consts).reshape(vals.shape[0], *shape)


def skew(v: torch.Tensor) -> torch.Tensor:
    """Batched skew: (..., 3) -> (..., 3, 3). Matches invariant_ekf.skew.

    One GEMM against a constant (3, 9) rearrangement instead of eight
    stacked kernels; each output entry is exactly +/-v_k or 0."""
    W = _const("skew_W", v.dtype, v.device)
    return (v @ W).reshape(*v.shape[:-1], 3, 3)


def _so3_coefficients(theta: torch.Tensor):
    """(sin t/t, (1-cos t)/t^2, (t-sin t)/t^3) with the same small-angle
    series as invariant_ekf._so3_coefficients (STABLE_TRAINING mode)."""
    small = theta < _SMALL_ANGLE
    safe = torch.where(small, torch.ones_like(theta), theta)
    t2 = safe * safe
    a = torch.where(small, 1.0 - theta * theta / 6.0, torch.sin(safe) / safe)
    b = torch.where(small, 0.5 - theta * theta / 24.0, (1.0 - torch.cos(safe)) / t2)
    c = torch.where(small, 1.0 / 6.0 - theta * theta / 120.0,
                    (safe - torch.sin(safe)) / (t2 * safe))
    return a, b, c


def exp_so3(w: torch.Tensor) -> torch.Tensor:
    """Batched SO(3) exp: (B, 3) -> (B, 3, 3)."""
    A = skew(w)
    theta = torch.linalg.norm(w, dim=-1)
    a, b, _ = _so3_coefficients(theta)
    I = _const("I3", w.dtype, w.device)
    return I + a[:, None, None] * A + b[:, None, None] * (A @ A)


def exp_sek3(xi: torch.Tensor) -> torch.Tensor:
    """Batched SE_K(3) exp at fixed K = DIM_X - 3: (B, 33) -> (B, 13, 13)."""
    B = xi.shape[0]
    w = xi[:, :3]
    A = skew(w)
    theta = torch.linalg.norm(w, dim=-1)
    a, b, c = _so3_coefficients(theta)
    I = _const("I3", xi.dtype, xi.device)
    A2 = A @ A
    R = I + a[:, None, None] * A + b[:, None, None] * A2
    Jl = I + b[:, None, None] * A + c[:, None, None] * A2
    cols = torch.einsum("bij,bsj->bis", Jl, xi[:, 3:].reshape(B, DIM_X - 3, 3))
    bottom = _const("I13", xi.dtype, xi.device)[3:].expand(B, -1, -1)
    return torch.cat([torch.cat([R, cols], dim=2), bottom], dim=1)


def _sym(P: torch.Tensor) -> torch.Tensor:
    return 0.5 * (P + P.transpose(-1, -2))


def _slot_blockdiag(blocks: torch.Tensor) -> torch.Tensor:
    """(B, 8, 3, 3) slot blocks -> (B, 24, 24) block-diagonal."""
    B = blocks.shape[0]
    out = torch.zeros(B, N_SLOTS, 3, N_SLOTS, 3,
                      dtype=blocks.dtype, device=blocks.device)
    idx = torch.arange(N_SLOTS, device=blocks.device)
    out[:, idx, :, idx, :] = blocks.transpose(0, 1)
    return out.reshape(B, DIM_M, DIM_M)


# -----------------------------------------------------------------------------
# state and schedule containers


class State(NamedTuple):
    X: torch.Tensor            # (B, 13, 13)
    theta: torch.Tensor        # (B, 6)
    P: torch.Tensor            # (B, 39, 39)
    jitter_count: torch.Tensor  # (B,) on-device near-singularity counter
    info_count: torch.Tensor    # (B,) on-device cholesky info != 0 counter


class RowOut(NamedTuple):
    R: torch.Tensor            # (B, 3, 3)
    v: torch.Tensor            # (B, 3)
    p: torch.Tensor            # (B, 3)
    nis: torch.Tensor          # (B,)


@dataclass
class BatchData:
    """Padded batch of rollout segments with the precomputed slot schedule.

    Local row t maps to global row trim0 + t; row 0 is the seed row
    (insert-only, ``apply_row0``), rows >= 1 are filter steps. Padded rows
    have dt_row = 0 and all-false masks, which makes the step an exact no-op.
    """
    B: int
    T_pad: int
    imu: torch.Tensor           # (B, T, 6)
    p_meas: torch.Tensor        # (B, T, 8, 3)
    gt_R_WB: torch.Tensor       # (B, T, 3, 3)
    gt_v_B: torch.Tensor        # (B, T, 3)
    gt_p_W: torch.Tensor        # (B, T, 3)
    dt_row: torch.Tensor        # (B, T)
    valid: torch.Tensor         # (B, T) bool
    prop_mask: torch.Tensor     # (B, T, 8) bool: active during propagate (flags[k-1])
    correct_mask: torch.Tensor  # (B, T, 8) bool: flags[k-1] & flags[k]
    insert_mask: torch.Tensor   # (B, T, 8) bool: ~flags[k-1] & flags[k]
    nis_dim: torch.Tensor       # (B, T) float: 3 * n corrected slots (0 => no meas)


def build_batch(rolls, *, T_pad: int | None = None,
                dtype: torch.dtype = torch.float64) -> BatchData:
    """Stack trimmed rollout segments into a padded BatchData.

    ``rolls`` are covariance_calibration.Rollout-like objects (imu, p_BC,
    flags, dt, trim0, trim1, gt_v_B attributes).
    """
    device = rolls[0].imu.device
    segs = [(r.trim0, r.trim1 - r.trim0) for r in rolls]
    T = max(s[1] for s in segs) if T_pad is None else T_pad
    B = len(rolls)
    imu = torch.zeros(B, T, 6, dtype=dtype, device=device)
    p_meas = torch.zeros(B, T, N_SLOTS, 3, dtype=dtype, device=device)
    gt_R_WB = torch.zeros(B, T, 3, 3, dtype=dtype, device=device)
    gt_v_B = torch.zeros(B, T, 3, dtype=dtype, device=device)
    gt_p_W = torch.zeros(B, T, 3, dtype=dtype, device=device)
    dt_row = torch.zeros(B, T, dtype=dtype, device=device)
    valid = torch.zeros(B, T, dtype=torch.bool, device=device)
    prop = np.zeros((B, T, N_SLOTS), dtype=bool)
    corr = np.zeros((B, T, N_SLOTS), dtype=bool)
    ins = np.zeros((B, T, N_SLOTS), dtype=bool)
    for b, (r, (t0, L)) in enumerate(zip(rolls, segs)):
        imu[b, :L] = r.imu[t0:t0 + L].to(dtype)
        p_meas[b, :L] = r.p_BC[t0:t0 + L].to(dtype)
        gt_R_WB[b, :L] = r.gt_R_WB[t0:t0 + L].to(dtype)
        gt_v_B[b, :L] = r.gt_v_B[t0:t0 + L].to(dtype)
        gt_p_W[b, :L] = r.gt_p_W[t0:t0 + L].to(dtype)
        dt_row[b, 1:L] = float(r.dt)
        valid[b, :L] = True
        flags = np.asarray(r.flags[t0:t0 + L]).astype(bool)
        prev = np.zeros_like(flags)
        prev[1:] = flags[:-1]           # prev[0] = 0: row 0 is insert-only
        prop[b, :L] = prev
        corr[b, :L] = prev & flags
        ins[b, :L] = ~prev & flags
    to_t = lambda a: torch.as_tensor(a, device=device)
    corr_t = to_t(corr)
    return BatchData(
        B=B, T_pad=T, imu=imu, p_meas=p_meas, gt_R_WB=gt_R_WB,
        gt_v_B=gt_v_B, gt_p_W=gt_p_W,
        dt_row=dt_row, valid=valid,
        prop_mask=to_t(prop), correct_mask=corr_t, insert_mask=to_t(ins),
        nis_dim=3.0 * corr_t.sum(-1).to(dtype),
    )


def init_state(seeds, *, device, dtype: torch.dtype = torch.float64) -> State:
    """Embed (X0 5x5, theta0 6, P0 15x15) seeds into fixed-slot state."""
    B = len(seeds)
    X = _const("I13", dtype, device).expand(B, -1, -1).clone()
    theta = torch.zeros(B, DIM_THETA, dtype=dtype, device=device)
    P = torch.zeros(B, DIM_P, DIM_P, dtype=dtype, device=device)
    for b, (X0, theta0, P0) in enumerate(seeds):
        X[b, 0:5, 0:5] = X0.to(dtype)
        theta[b] = theta0.to(dtype)
        P[b, 0:9, 0:9] = P0[0:9, 0:9].to(dtype)
        P[b, 0:9, GROUP:] = P0[0:9, 9:15].to(dtype)
        P[b, GROUP:, 0:9] = P0[9:15, 0:9].to(dtype)
        P[b, GROUP:, GROUP:] = P0[9:15, 9:15].to(dtype)
    zero = torch.zeros(B, dtype=dtype, device=device)
    return State(X, theta, P, zero, zero.clone())


def detach_state(state: State) -> State:
    return State(*(t.detach() for t in state))


# -----------------------------------------------------------------------------
# filter stages (all batched, static shapes)


def _propagate(state: State, gyro, accel, dt_row, prop_mask, covs) -> State:
    """Propagate with scatter-by-GEMM assembly (tests/test_propagate_block.py
    holds the zeros+slice-assign dense reference).

    Same formulas as the dense build -- P' = sym(Phi P Phi^T + Qk_hat) with
    Phi = I + dt A -- but A, Adj and Qk are assembled entry-exactly by one
    GEMM against constant placement bases instead of zeros + slice
    assignments, and X' by cat instead of clone + slice assignment. In this
    kernel-count-bound regime that swaps ~50 CopySlices/SelectBackward
    launches per step for 2 GEMMs (fwd) + 2 GEMMs (bwd). dt = 0 rows stay
    bitwise no-ops."""
    X, theta, P = state.X, state.theta, state.P
    B = X.shape[0]
    dtype, device = X.dtype, X.device
    R_old = X[:, 0:3, 0:3]
    v_old = X[:, 0:3, 3]
    p_old = X[:, 0:3, 4]
    dt = dt_row[:, None]
    dtb = dt_row[:, None, None]

    w = gyro - theta[:, 0:3]
    a = accel - theta[:, 3:6]
    R_pred = R_old @ exp_so3(w * dt)
    acc_w = torch.einsum("bij,bj->bi", R_old, a) + _const("g", dtype, device)
    v_pred = v_old + acc_w * dt
    p_pred = p_old + v_old * dt + 0.5 * acc_w * dt * dt

    X_new = torch.cat([
        torch.cat([R_pred, v_pred[:, :, None], p_pred[:, :, None],
                   X[:, 0:3, 5:]], dim=2),
        X[:, 3:]], dim=1)

    cols = X[:, 0:3, 3:DIM_X]                        # v, p, slot columns
    SR = skew(cols.transpose(1, 2)) @ R_old[:, None]  # (B, 10, 3, 3)

    # A and Adj assembled with one scatter-by-GEMM (entry-exact vs the
    # zeros+slice-assign build, one-GEMM backward instead of CopySlices)
    vals = torch.cat([R_old.reshape(B, 9), SR.reshape(B, 90)], dim=1)
    AB = (_assemble("prop", vals, 2, DIM_P, DIM_P)
          + _const("prop_const", dtype, device))
    A, Adj = AB[:, 0], AB[:, 1]

    # process noise: Qc only on slots active during this interval
    slot_q = covs["Qc"] * prop_mask.to(dtype)[:, :, None, None]
    vq = torch.cat([covs["Qg"].reshape(1, 9).expand(B, 9),
                    covs["Qa"].reshape(1, 9).expand(B, 9),
                    slot_q.reshape(B, 72),
                    covs["Qbg"].reshape(1, 9).expand(B, 9),
                    covs["Qba"].reshape(1, 9).expand(B, 9)], dim=1)
    Qk = _assemble("qk", vq, DIM_P, DIM_P)

    I39 = _const("I39", dtype, device)
    Phi = I39 + A * dtb
    PhiAdj = Phi @ Adj
    Qk_hat = PhiAdj @ Qk @ PhiAdj.transpose(1, 2) * dtb
    P_new = _sym(Phi @ P @ Phi.transpose(1, 2) + Qk_hat)
    return State(X_new, theta, P_new, state.jitter_count, state.info_count)


def _slot_blockdiag_k(blocks: torch.Tensor) -> torch.Tensor:
    """(B, K, 3, 3) -> (B, 3K, 3K) block diagonal (K = DIM_X - 3)."""
    B, K = blocks.shape[0], blocks.shape[1]
    out = torch.zeros(B, K, 3, K, 3, dtype=blocks.dtype, device=blocks.device)
    idx = torch.arange(K, device=blocks.device)
    out[:, idx, :, idx, :] = blocks.transpose(0, 1)
    return out.reshape(B, 3 * K, 3 * K)


def _correct(state: State, p_meas, correct_mask, R_kin,
             s_jitter: float) -> tuple[State, torch.Tensor, torch.Tensor]:
    """Masked stacked kinematic correction, block-structured: H and the dense
    24x24 gain products are never materialized. Slot m of H is -I at the
    position error and +I at the slot error, so P H^T and H P H^T are slice
    differences of P; N is the active-slot 3x3 block diagonal with identity
    on inactive slots, which decouples them (K's inactive columns are exactly
    0). Same math as the dense form (tests/test_correct_block.py holds the
    dense reference). Returns (state, nis, N_blk) where
    N_blk = R_pre R_kin R_pre^T is reused by the insertion stage."""
    X, theta, P = state.X, state.theta, state.P
    B = X.shape[0]
    dtype, device = X.dtype, X.device
    R_pre = X[:, 0:3, 0:3]
    m = correct_mask.to(dtype)                       # (B, 8)

    N_blk = R_pre @ R_kin @ R_pre.transpose(1, 2)    # (B, 3, 3)
    I3 = _const("I3", dtype, device)
    slot_N = torch.where(correct_mask[:, :, None, None],
                         N_blk[:, None], I3.expand(B, N_SLOTS, 3, 3))

    # innovation rows 0:3 per slot: R p_bc + p - X[0:3, slot], masked
    Z = (torch.einsum("bik,bsk->bsi", R_pre, p_meas)
         + X[:, 0:3, 4][:, None] - X[:, 0:3, 5:DIM_X].transpose(1, 2))
    Z = (Z * m[:, :, None]).reshape(B, DIM_M)

    # PHT = P @ H^T: column block j is (P[:, contact_j] - P[:, position]) m_j
    Pc = P[:, :, 9:9 + DIM_M].reshape(B, DIM_P, N_SLOTS, 3)
    PHT = ((Pc - P[:, :, 6:9][:, :, None])
           * m[:, None, :, None]).reshape(B, DIM_P, DIM_M)
    # S = H @ PHT + N: row block i is (PHT[contact_i] - PHT[position]) m_i
    Sr = PHT[:, 9:9 + DIM_M].reshape(B, N_SLOTS, 3, DIM_M)
    S = ((Sr - PHT[:, 6:9][:, None])
         * m[:, :, None, None]).reshape(B, DIM_M, DIM_M)
    S = S + _assemble("corr", slot_N.reshape(B, 72), DIM_M, DIM_M)
    jitter_count = state.jitter_count
    if s_jitter > 0.0:
        jitter_count = jitter_count + (
            torch.diagonal(S, dim1=1, dim2=2).min(dim=1).values
            < 10.0 * s_jitter).to(dtype).detach()
        S = S + s_jitter * _const("I24", dtype, device)
    L, info = torch.linalg.cholesky_ex(S, check_errors=False)
    # one solve for both the gain (K = PHT S^{-1}) and the NIS whitening
    sol = torch.cholesky_solve(
        torch.cat([PHT.transpose(1, 2), Z[:, :, None]], dim=2), L)
    K = sol[:, :, :DIM_P].transpose(1, 2)            # (B, 39, 24)
    info_count = state.info_count + (info != 0).to(dtype).detach()

    delta = torch.einsum("bij,bj->bi", K, Z)
    dX = exp_sek3(delta[:, :GROUP])
    X_new = dX @ X
    theta_new = theta + delta[:, GROUP:]
    # K @ H by blocks: K's inactive columns are exactly 0, so no extra mask;
    # cat (narrow backward) instead of slice assignment (CopySlices backward).
    # A low-rank Joseph form (KH = C Sel, C = [-sum K_m, K]) was measured at
    # +7 kernels/step in backward vs this KH cat -- rejected, keep the cat.
    Kb = K.reshape(B, DIM_P, N_SLOTS, 3)
    zc = _const("Z39_6", dtype, device).expand(B, -1, -1)
    KH = torch.cat([zc, -Kb.sum(2), K, zc], dim=2)
    IKH = _const("I39", dtype, device) - KH
    M2 = (IKH @ P) @ IKH.transpose(1, 2)
    KN = torch.einsum("bnjk,bjkl->bnjl", Kb, slot_N).reshape(B, DIM_P, DIM_M)
    P_new = _sym(M2 + KN @ K.transpose(1, 2))
    nis = (Z * sol[:, :, DIM_P]).sum(1)
    return State(X_new, theta_new, P_new, jitter_count, info_count), nis, N_blk


def _insert(state: State, p_meas, insert_mask, R_pre, N_blk) -> State:
    """Masked slot insertion: PRE-correction R with POST-correction p and P
    (the Hartley augmentation convention). Exact per-slot equivalent of
    sequential F P F^T + G cov G^T, including same-row multi-insert cross
    terms. The row copy -> column copy of the row-updated matrix is a
    batch-dependent source-row permutation, applied as two gathers instead
    of clone/where chains; the masked diagonal add is one scatter-by-GEMM
    (tests/test_insert_block.py holds the clone/where reference)."""
    X, P = state.X, state.P
    B = X.shape[0]
    dtype, device = X.dtype, X.device
    m = insert_mask[:, :, None]                       # (B, 8, 1)
    p_post = X[:, 0:3, 4]

    new_cols = p_post[:, None] + torch.einsum("bik,bsk->bsi", R_pre, p_meas)
    X_new = torch.cat([
        torch.cat([X[:, 0:3, 0:5],
                   torch.where(m.transpose(1, 2), new_cols.transpose(1, 2),
                               X[:, 0:3, 5:DIM_X])], dim=2),
        X[:, 3:]], dim=1)

    # inserted slot rows/columns read the position band 6:9, everything else
    # reads itself: one (B, 39) source map, applied along rows then columns
    pair = _const("ins_pair", torch.long, device)
    src = torch.cat([
        _const("ins_head", torch.long, device).expand(B, -1),
        torch.where(m, pair[0], pair[1]).reshape(B, GROUP - 9),
        _const("ins_tail", torch.long, device).expand(B, -1)], dim=1)
    P2 = torch.take_along_dim(
        torch.take_along_dim(P, src[:, :, None], dim=1),
        src[:, None, :], dim=2)
    add = _assemble(
        "ins", (N_blk[:, None] * m[:, :, None].to(dtype)).reshape(B, 72),
        DIM_P, DIM_P)
    return State(X_new, state.theta, _sym(P2 + add),
                 state.jitter_count, state.info_count)


def step(state: State, imu, p_meas, dt_row, prop_mask, correct_mask,
         insert_mask, covs, R_kin, s_jitter: float) -> tuple[State, RowOut]:
    """One filter row: propagate -> masked correct -> masked insert.

    Removal needs no operation (mask clear happens in the schedule). Padded
    rows (dt_row = 0, all-false masks) are exact no-ops.
    """
    state = _propagate(state, imu[:, 0:3], imu[:, 3:6], dt_row, prop_mask, covs)
    R_pre = state.X[:, 0:3, 0:3]
    state, nis, N_blk = _correct(state, p_meas, correct_mask, R_kin, s_jitter)
    state = _insert(state, p_meas, insert_mask, R_pre, N_blk)
    out = RowOut(state.X[:, 0:3, 0:3], state.X[:, 0:3, 3], state.X[:, 0:3, 4],
                 nis)
    return state, out


def apply_row0(state: State, p_meas0, insert0, R_kin) -> State:
    """Segment-start row: pure insertion (mirrors ``start_filter``: no slots
    exist yet, so the row-0 kinematic pass augments and never corrects)."""
    R_pre = state.X[:, 0:3, 0:3]
    N_blk = R_pre @ R_kin @ R_pre.transpose(1, 2)
    return _insert(state, p_meas0, insert0, R_pre, N_blk)


def run_rows_fixed(state: State, batch: BatchData, rows: slice, covs,
                   R_kin, *, s_jitter: float = 0.0,
                   step_fn: Callable | None = None) -> tuple[State, dict]:
    """Advance over batch rows [rows.start, rows.stop) (local indices >= 1).

    Returns (state, out) with out tensors shaped (B, T_chunk, ...). The caller
    detaches state at truncated-BPTT chunk boundaries via ``detach_state``.
    """
    fn = step if step_fn is None else step_fn
    R_out, v_out, p_out, nis_out = [], [], [], []
    for t in range(rows.start, rows.stop):
        state, out = fn(state, batch.imu[:, t], batch.p_meas[:, t],
                        batch.dt_row[:, t], batch.prop_mask[:, t],
                        batch.correct_mask[:, t], batch.insert_mask[:, t],
                        covs, R_kin, s_jitter)
        R_out.append(out.R)
        v_out.append(out.v)
        p_out.append(out.p)
        nis_out.append(out.nis)
    return state, {
        "R_WB": torch.stack(R_out, dim=1),
        "v_W": torch.stack(v_out, dim=1),
        "p_W": torch.stack(p_out, dim=1),
        "nis": torch.stack(nis_out, dim=1),
        "nis_dim": batch.nis_dim[:, rows],
    }


def reg_nis_masked(nis: torch.Tensor, nis_dim: torch.Tensor) -> torch.Tensor:
    """(mean(NIS/dim) - 1)^2 over rows that had measurements; exactly the
    masked equivalent of covariance_calibration.reg_nis."""
    has = nis_dim > 0
    per_dim = torch.where(has, nis / nis_dim.clamp_min(1.0),
                          torch.zeros_like(nis))
    count = has.sum().clamp_min(1)
    return (per_dim.sum() / count - 1.0) ** 2


def make_compiled_step(mode: str | None) -> Callable:
    """step compiled with fullgraph=True, or eager for mode None."""
    if mode is None:
        return step
    return torch.compile(step, fullgraph=True, mode=mode)
