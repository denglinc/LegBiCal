"""Differentiable Torch contact-aided right-invariant EKF replay."""

from __future__ import annotations

import numpy as np
import torch

_SMALL_ANGLE = 1e-4

_EYE_CACHE: dict[tuple, torch.Tensor] = {}
_TEMPL_CACHE: dict[tuple, torch.Tensor] = {}


def _eye(n: int, dtype, device) -> torch.Tensor:
    """Cached identity (read-only use; callers must .clone() before mutating).

    Avoids re-launching an eye kernel for every propagate/correct call --
    values are identical to torch.eye, this is purely a launch-count saving.
    """
    key = (n, dtype, str(device))
    e = _EYE_CACHE.get(key)
    if e is None:
        e = torch.eye(n, dtype=dtype, device=device)
        _EYE_CACHE[key] = e
    return e


def _yb_template(dimX: int, col: int, dtype, device) -> torch.Tensor:
    """Cached constant [0,...,0, 1@4, -1@col] vector (read-only).

    This is the constant kinematic ``b`` segment. ``Y`` equals it with rows
    0:3 overwritten by p_bc; both are built from this cache to avoid
    per-measurement host-scalar copies.
    """
    key = ("yb", dimX, col, dtype, str(device))
    t = _TEMPL_CACHE.get(key)
    if t is None:
        t = torch.zeros(dimX, dtype=dtype, device=device)
        t[4] = 1.0
        t[col] = -1.0
        _TEMPL_CACHE[key] = t
    return t


def _h_template(dimP: int, col: int, dtype, device) -> torch.Tensor:
    """Cached constant kinematic H block [-I @ 6:9, +I @ col slice] (read-only)."""
    key = ("h", dimP, col, dtype, str(device))
    t = _TEMPL_CACHE.get(key)
    if t is None:
        t = torch.zeros(3, dimP, dtype=dtype, device=device)
        eye3 = _eye(3, dtype, device)
        t[:, 6:9] = -eye3
        t[:, col_error_slice(col)] = eye3
        _TEMPL_CACHE[key] = t
    return t


def _pi_template(n: int, dimX: int, dtype, device) -> torch.Tensor:
    """Cached constant PI selector (read-only)."""
    key = ("pi", n, dimX, dtype, str(device))
    t = _TEMPL_CACHE.get(key)
    if t is None:
        t = torch.zeros(3 * n, dimX * n, dtype=dtype, device=device)
        for i in range(n):
            t[3 * i:3 * i + 3, dimX * i:dimX * i + 3] = _eye(3, dtype, device)
        _TEMPL_CACHE[key] = t
    return t


def skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros((), dtype=v.dtype, device=v.device)
    row0 = torch.stack([z, -v[2], v[1]])
    row1 = torch.stack([v[2], z, -v[0]])
    row2 = torch.stack([-v[1], v[0], z])
    return torch.stack([row0, row1, row2])


def _so3_coefficients(theta: torch.Tensor):
    """Return (sin(t)/t, (1-cos t)/t^2, (t-sin t)/t^3) with safe small-angle series."""
    small = theta < _SMALL_ANGLE
    safe = torch.where(small, torch.ones_like(theta), theta)
    t2 = safe * safe
    a = torch.where(small, 1.0 - theta * theta / 6.0, torch.sin(safe) / safe)
    b = torch.where(small, 0.5 - theta * theta / 24.0, (1.0 - torch.cos(safe)) / t2)
    c = torch.where(small, 1.0 / 6.0 - theta * theta / 120.0,
                    (safe - torch.sin(safe)) / (t2 * safe))
    return a, b, c


def exp_so3(w: torch.Tensor) -> torch.Tensor:
    A = skew(w)
    theta = torch.linalg.norm(w)
    a, b, _ = _so3_coefficients(theta)
    I = _eye(3, w.dtype, w.device)
    return I + a * A + b * (A @ A)


def exp_sek3(xi: torch.Tensor) -> torch.Tensor:
    """Exponential map for SE_K(3); xi = [phi, xi_1, ..., xi_K] (3 + 3K)."""
    K = (xi.shape[0] - 3) // 3
    w = xi[:3]
    A = skew(w)
    theta = torch.linalg.norm(w)
    a, b, c = _so3_coefficients(theta)
    I = _eye(3, xi.dtype, xi.device)
    A2 = A @ A
    R = I + a * A + b * A2
    Jl = I + b * A + c * A2
    X = torch.eye(3 + K, dtype=xi.dtype, device=xi.device).clone()
    X[0:3, 0:3] = R
    for i in range(K):
        X[0:3, 3 + i] = Jl @ xi[3 + 3 * i: 6 + 3 * i]
    return X


def adjoint_sek3(X: torch.Tensor) -> torch.Tensor:
    """Adjoint of X in SE_K(3): (3+3K, 3+3K) with K = X.cols - 3."""
    K = X.shape[1] - 3
    R = X[0:3, 0:3]
    Adj = torch.zeros(3 + 3 * K, 3 + 3 * K, dtype=X.dtype, device=X.device)
    Adj[0:3, 0:3] = R
    for i in range(K):
        s = 3 + 3 * i
        Adj[s:s + 3, s:s + 3] = R
        Adj[s:s + 3, 0:3] = skew(X[0:3, 3 + i]) @ R
    return Adj


def col_error_slice(col: int) -> slice:
    if col < 3:
        raise ValueError("col_error_slice is for X columns >= 3")
    return slice(3 * col - 6, 3 * col - 3)


class InvariantEKF:
    """Dynamic-dimension, bias-aware, differentiable right-invariant InEKF.

    ``covs`` maps noise-group names to 3x3 SPD tensors (may carry gradients):
    Qg, Qa, Qbg, Qba, Qc. Measurement covariance enters per measurement.
    """

    def __init__(self, X0: torch.Tensor, theta0: torch.Tensor, P0: torch.Tensor,
                 covs: dict[str, torch.Tensor], *, g: torch.Tensor | None = None,
                 s_jitter: float = 0.0) -> None:
        self.X = X0
        self.theta = theta0
        self.P = P0
        self.covs = covs
        dtype, device = X0.dtype, X0.device
        self.g = (torch.tensor([0.0, 0.0, -9.81], dtype=dtype, device=device)
                  if g is None else g)
        self.s_jitter = float(s_jitter)
        # GPU-accumulated near-singularity counter (no per-correction host sync);
        # uses min-diagonal of S as a cheap conservative proxy for min-eig.
        self._jitter_counter = torch.zeros((), dtype=dtype, device=device)
        self._skew_g = skew(self.g)
        self.contacts: dict[int, bool] = {}
        self.estimated_contact_positions: dict[int, int] = {}

    @property
    def jitter_events(self) -> int:
        """Corrections where the jitter was non-negligible vs diag(S).

        Synchronizes with the device -- read after loops, not inside them.
        """
        return int(self._jitter_counter.item())

    # -- dimensions ----------------------------------------------------------
    @property
    def dimX(self) -> int:
        return self.X.shape[1]

    @property
    def dimTheta(self) -> int:
        return self.theta.shape[0]

    @property
    def dimP(self) -> int:
        return self.P.shape[1]

    def _sym(self, P: torch.Tensor) -> torch.Tensor:
        return 0.5 * (P + P.transpose(-1, -2))

    # -- contacts --------------------------------------------------------------
    def set_contacts(self, contacts) -> None:
        for cid, indicator in contacts:
            self.contacts[int(cid)] = bool(indicator)

    # -- propagation -----------------------------------------------------------
    def propagate(
        self,
        gyro: torch.Tensor,
        accel: torch.Tensor,
        dt,
        contact_process_covariance: torch.Tensor | None = None,
    ) -> None:
        """Propagate with an optional ``[N,3,3]`` candidate-id covariance."""
        X_old = self.X
        P_old = self.P
        R_old = X_old[0:3, 0:3]
        v_old = X_old[0:3, 3]
        p_old = X_old[0:3, 4]
        dtype, device = X_old.dtype, X_old.device
        dt = torch.as_tensor(dt, dtype=dtype, device=device)

        w = gyro - self.theta[0:3]
        a = accel - self.theta[3:6]

        R_pred = R_old @ exp_so3(w * dt)
        acc_w = R_old @ a + self.g
        v_pred = v_old + acc_w * dt
        p_pred = p_old + v_old * dt + 0.5 * acc_w * dt * dt

        X_new = X_old.clone()
        X_new[0:3, 0:3] = R_pred
        X_new[0:3, 3] = v_pred
        X_new[0:3, 4] = p_pred

        dimX, dimP, dimTheta = self.dimX, self.dimP, self.dimTheta
        bias = dimP - dimTheta
        A = torch.zeros(dimP, dimP, dtype=dtype, device=device)
        A[3:6, 0:3] = self._skew_g
        A[6:9, 3:6] = _eye(3, dtype, device)
        A[0:3, bias:bias + 3] = -R_old
        A[3:6, bias + 3:bias + 6] = -R_old
        for i in range(3, dimX):
            A[3 * i - 6:3 * i - 3, bias:bias + 3] = -skew(X_old[0:3, i]) @ R_old

        Qk = torch.zeros(dimP, dimP, dtype=dtype, device=device)
        Qk[0:3, 0:3] = self.covs["Qg"]
        Qk[3:6, 3:6] = self.covs["Qa"]
        if contact_process_covariance is None:
            for col in self.estimated_contact_positions.values():
                sl = col_error_slice(col)
                Qk[sl, sl] = self.covs["Qc"]
        else:
            if contact_process_covariance.ndim != 3 \
                    or contact_process_covariance.shape[-2:] != (3, 3):
                raise ValueError("contact_process_covariance must have shape [N,3,3]")
            for candidate_id, col in self.estimated_contact_positions.items():
                if candidate_id >= contact_process_covariance.shape[0]:
                    raise ValueError("active candidate is outside contact_process_covariance")
                sl = col_error_slice(col)
                Qk[sl, sl] = contact_process_covariance[candidate_id]
        Qk[bias:bias + 3, bias:bias + 3] = self.covs["Qbg"]
        Qk[bias + 3:bias + 6, bias + 3:bias + 6] = self.covs["Qba"]

        I = _eye(dimP, dtype, device)
        Phi = I + A * dt
        Adj = _eye(dimP, dtype, device).clone()
        Adj[:dimP - dimTheta, :dimP - dimTheta] = adjoint_sek3(X_old)
        PhiAdj = Phi @ Adj
        Qk_hat = PhiAdj @ Qk @ PhiAdj.transpose(0, 1) * dt

        self.X = X_new
        self.P = self._sym(Phi @ P_old @ Phi.transpose(0, 1) + Qk_hat)

    # -- generic correction ------------------------------------------------------
    def correct(self, Y: torch.Tensor, b: torch.Tensor, H: torch.Tensor,
                N: torch.Tensor, PI: torch.Tensor) -> torch.Tensor:
        """Right-invariant update. Returns per-call NIS (scalar tensor)."""
        dimX, dimP, dimTheta = self.dimX, self.dimP, self.dimTheta
        dtype, device = self.X.dtype, self.X.device
        P = self.P
        PHT = P @ H.transpose(0, 1)
        S = H @ PHT + N
        if self.s_jitter > 0.0:
            # accumulate near-singularity count on-device (min diag(S) as a
            # cheap proxy for min eig; no host synchronization per correction)
            with torch.no_grad():
                self._jitter_counter += (
                    torch.diagonal(S).min() < 10.0 * self.s_jitter).to(dtype)
            S = S + self.s_jitter * _eye(S.shape[0], dtype, device)
        K = torch.linalg.solve(S, PHT.transpose(0, 1)).transpose(0, 1)

        n_copies = Y.shape[0] // dimX
        if n_copies == 1:
            BigX = self.X
        else:
            BigX = torch.zeros(n_copies * dimX, n_copies * dimX, dtype=dtype, device=device)
            for i in range(n_copies):
                BigX[i * dimX:(i + 1) * dimX, i * dimX:(i + 1) * dimX] = self.X

        Z = BigX @ Y - b
        Z_sel = PI @ Z
        delta = K @ Z_sel
        dX = exp_sek3(delta[:dimP - dimTheta])

        self.X = dX @ self.X
        self.theta = self.theta + delta[dimP - dimTheta:]

        IKH = _eye(dimP, dtype, device) - K @ H
        self.P = self._sym(IKH @ P @ IKH.transpose(0, 1) + K @ N @ K.transpose(0, 1))

        nis = Z_sel @ torch.linalg.solve(S, Z_sel)
        return nis

    # -- contact kinematics -------------------------------------------------------
    def correct_kinematics(self, measurements) -> list[tuple[torch.Tensor, int]]:
        """measurements: iterable of (id, p_bc (3,) tensor, cov3 (3,3) tensor).

        Returns [(nis, dim)] for the stacked correction (empty if none).
        Mirrors InEKF::CorrectKinematics: correct -> remove -> augment, with the
        PRE-correction rotation used for N, the new-column mean, and G.
        """
        R_pre = self.X[0:3, 0:3]
        Y_list, b_list, H_list, N_blocks = [], [], [], []
        remove_contacts: list[tuple[int, int]] = []
        new_contacts = []
        used_ids: list[int] = []
        dtype, device = self.X.dtype, self.X.device

        for cid, p_bc, cov3 in measurements:
            cid = int(cid)
            if cid in used_ids:
                continue
            used_ids.append(cid)
            if cid not in self.contacts:
                continue
            indicated = self.contacts[cid]
            found = cid in self.estimated_contact_positions
            if (not indicated) and found:
                remove_contacts.append((cid, self.estimated_contact_positions[cid]))
            elif indicated and not found:
                new_contacts.append((cid, p_bc, cov3))
            elif indicated and found:
                col = self.estimated_contact_positions[cid]
                dimX, dimP = self.dimX, self.dimP
                yb = _yb_template(dimX, col, dtype, device)
                Y = yb.clone()
                Y[0:3] = p_bc
                Y_list.append(Y)
                b_list.append(yb)  # read-only constant; identical to the direct build
                H_list.append(_h_template(dimP, col, dtype, device))
                N_blocks.append(R_pre @ cov3 @ R_pre.transpose(0, 1))

        nis_out: list[tuple[torch.Tensor, int]] = []
        if Y_list:
            n = len(Y_list)
            dimX = self.dimX
            Y = torch.cat(Y_list)
            b = torch.cat(b_list)
            H = torch.cat(H_list, dim=0)
            N = torch.zeros(3 * n, 3 * n, dtype=dtype, device=device)
            for i, blk in enumerate(N_blocks):
                N[3 * i:3 * i + 3, 3 * i:3 * i + 3] = blk
            PI = _pi_template(n, dimX, dtype, device)
            nis = self.correct(Y, b, H, N, PI)
            nis_out.append((nis, 3 * n))

        # remove (descending column order == C++ in-order removal + reindex)
        for cid, col in sorted(remove_contacts, key=lambda x: x[1], reverse=True):
            del self.estimated_contact_positions[cid]
            keep_x = [i for i in range(self.dimX) if i != col]
            idx_x = torch.tensor(keep_x, dtype=torch.long, device=device)
            self.X = self.X.index_select(0, idx_x).index_select(1, idx_x)
            sl = col_error_slice(col)
            keep_p = [i for i in range(self.dimP) if not (sl.start <= i < sl.stop)]
            idx_p = torch.tensor(keep_p, dtype=torch.long, device=device)
            self.P = self.P.index_select(0, idx_p).index_select(1, idx_p)
            for key, old_col in self.estimated_contact_positions.items():
                if old_col > col:
                    self.estimated_contact_positions[key] = old_col - 1

        # augment (PRE-correction R, POST-correction p and P, as in C++)
        for cid, p_bc, cov3 in new_contacts:
            old_dimX, old_dimP = self.dimX, self.dimP
            dimTheta = self.dimTheta
            group = old_dimP - dimTheta
            p = self.X[0:3, 4]
            X_aug = torch.eye(old_dimX + 1, dtype=dtype, device=device).clone()
            X_aug[:old_dimX, :old_dimX] = self.X
            X_aug[0:3, old_dimX] = p + R_pre @ p_bc
            F = torch.zeros(old_dimP + 3, old_dimP, dtype=dtype, device=device)
            F[0:group, 0:group] = _eye(group, dtype, device)
            F[group:group + 3, 6:9] = _eye(3, dtype, device)
            F[group + 3:, group:] = _eye(dimTheta, dtype, device)
            G = torch.zeros(old_dimP + 3, 3, dtype=dtype, device=device)
            G[group:group + 3, :] = R_pre
            self.P = self._sym(F @ self.P @ F.transpose(0, 1)
                               + G @ cov3 @ G.transpose(0, 1))
            self.X = X_aug
            self.estimated_contact_positions[cid] = old_dimX

        return nis_out


def start_filter(X0: torch.Tensor, theta0: torch.Tensor, P0: torch.Tensor,
                 covs: dict[str, torch.Tensor], flags_row0, p_meas_row0: torch.Tensor,
                 R_kin_pos: torch.Tensor, *, s_jitter: float = 0.0) -> InvariantEKF:
    """Initialize a filter at a segment start: set the row-0 contact state and
    apply the row-0 kinematic correction (mirrors the rollout-start behavior of
    ``replay_inekf_torch``)."""
    N = p_meas_row0.shape[0]
    filt = InvariantEKF(X0, theta0, P0, covs, s_jitter=s_jitter)
    filt.set_contacts([(i, bool(flags_row0[i])) for i in range(N)])
    filt.correct_kinematics([(i, p_meas_row0[i], R_kin_pos) for i in range(N)])
    return filt


def run_rows(filt: InvariantEKF, imu_step: torch.Tensor, dt,
             p_meas: torch.Tensor, flags, prev_flags_row, R_kin_pos: torch.Tensor,
             *, collect_nis: bool = False, changes_list=None,
             contact_process_covariance: torch.Tensor | None = None):
    """Advance an existing filter over a block of rows (for truncated BPTT).

    Row r in the block: propagate with ``imu_step[r]`` (the IMU sample driving
    the interval ending at row r), apply contact changes of ``flags[r]`` vs the
    previous row (``prev_flags_row`` for r=0), then correct with
    ``p_meas[r]``. The filter (state, covariance, contact maps) is mutated in
    place and carries across successive calls.

    ``changes_list`` optionally provides the per-row contact-change pairs
    precomputed by ``precompute_contact_changes`` (a CPU cost saving; the
    events are identical to the flags-derived ones, and ``flags`` /
    ``prev_flags_row`` are then ignored).
    """
    n_rows, N = p_meas.shape[0], p_meas.shape[1]
    if changes_list is None:
        flags = flags.astype(bool) if not isinstance(flags, torch.Tensor) else \
            flags.detach().cpu().numpy().astype(bool)
        changes_list = [
            [(i, bool(flags[r, i]))
             for i in range(N) if flags[r, i] != (prev_flags_row[i] if r == 0
                                                  else flags[r - 1, i])]
            for r in range(n_rows)
        ]
    cand_ids = list(range(N))
    if contact_process_covariance is not None and contact_process_covariance.shape != \
            (n_rows, N, 3, 3):
        raise ValueError(
            "contact_process_covariance must match [rows,N,3,3], got "
            f"{tuple(contact_process_covariance.shape)}")
    dt = torch.as_tensor(dt, dtype=filt.X.dtype, device=filt.X.device)  # one H2D copy
    R_out, v_out, p_out = [], [], []
    nis_values: list[torch.Tensor] = []
    nis_dims: list[int] = []
    for r in range(n_rows):
        if contact_process_covariance is None:
            filt.propagate(imu_step[r, 0:3], imu_step[r, 3:6], dt)
        else:
            filt.propagate(
                imu_step[r, 0:3], imu_step[r, 3:6], dt,
                contact_process_covariance[r])
        changes = changes_list[r]
        if changes:
            filt.set_contacts(changes)
        p_row = p_meas[r]
        out = filt.correct_kinematics([(i, p_row[i], R_kin_pos) for i in cand_ids])
        if collect_nis:
            for nis, dim in out:
                nis_values.append(nis)
                nis_dims.append(dim)
        R_out.append(filt.X[0:3, 0:3])
        v_out.append(filt.X[0:3, 3])
        p_out.append(filt.X[0:3, 4])
    return {
        "R_WB": torch.stack(R_out),
        "v_W": torch.stack(v_out),
        "p_W": torch.stack(p_out),
        "nis_values": nis_values,
        "nis_dims": nis_dims,
    }


def detach_filter(filt: InvariantEKF) -> None:
    """Detach the carried filter state at a truncated-BPTT chunk boundary."""
    filt.X = filt.X.detach()
    filt.theta = filt.theta.detach()
    filt.P = filt.P.detach()


def precompute_contact_changes(flags) -> list:
    """Per-row contact-change pairs for ``run_rows(changes_list=...)``.

    Row 0 entries are the changes vs the previous row OUTSIDE the block, so
    the caller should slice this list to match its rows: for a block starting
    at rollout row a, pass ``changes[a:b]`` where ``changes`` was built from
    the full flags array (row 0 of the rollout gets all-True-vs-unknown
    handled by ``start_filter``, so index 0 here is unused at rollout start).
    """
    flags = np.asarray(flags).astype(bool)
    T, N = flags.shape
    changes = [[] for _ in range(T)]
    for r in range(1, T):
        diff = np.nonzero(flags[r] != flags[r - 1])[0]
        if diff.size:
            changes[r] = [(int(i), bool(flags[r, i])) for i in diff]
    return changes


def replay_inekf_torch(
    X0: torch.Tensor,
    theta0: torch.Tensor,
    P0: torch.Tensor,
    covs: dict[str, torch.Tensor],
    imu: torch.Tensor,          # (T, 6) [gyro, accel]
    dt: float,
    p_meas: torch.Tensor,       # (T, N, 3) candidate positions in frame I
    contact_flags,              # (T, N) bool array/tensor: fixed event schedule
    R_kin_pos: torch.Tensor,    # (3, 3) kinematic position measurement covariance
    *,
    collect_nis: bool = False,
    s_jitter: float = 0.0,
    contact_process_covariance: torch.Tensor | None = None,
):
    """Replay with a fixed contact-event schedule (deterministic, differentiable).

    Step convention (matches the NumPy G1 replay): step 0 records the initial
    state; at step k >= 1 the filter propagates over [t[k-1], t[k]] using the
    IMU sample of row k-1, then applies contact events of row k, then corrects
    with the kinematic measurements of row k for candidates whose contact
    state is known. Ground truth is not an input.
    """
    T, N = p_meas.shape[0], p_meas.shape[1]
    flags = contact_flags
    if isinstance(flags, torch.Tensor):
        flags = flags.detach().cpu().numpy()
    flags = flags.astype(bool)
    if contact_process_covariance is not None \
            and contact_process_covariance.shape != (T, N, 3, 3):
        raise ValueError("contact_process_covariance must have shape [T,N,3,3]")

    filt = InvariantEKF(X0, theta0, P0, covs, s_jitter=s_jitter)
    filt.set_contacts([(i, bool(flags[0, i])) for i in range(N)])
    meas0 = [(i, p_meas[0, i], R_kin_pos) for i in range(N)]
    filt.correct_kinematics(meas0)

    R_out = [filt.X[0:3, 0:3]]
    v_out = [filt.X[0:3, 3]]
    p_out = [filt.X[0:3, 4]]
    nis_values: list[torch.Tensor] = []
    nis_dims: list[int] = []
    dt = torch.as_tensor(dt, dtype=X0.dtype, device=X0.device)  # one H2D copy

    for k in range(1, T):
        if contact_process_covariance is None:
            filt.propagate(imu[k - 1, 0:3], imu[k - 1, 3:6], dt)
        else:
            filt.propagate(
                imu[k - 1, 0:3], imu[k - 1, 3:6], dt,
                contact_process_covariance[k])
        changes = [(i, bool(flags[k, i])) for i in range(N) if flags[k, i] != flags[k - 1, i]]
        if changes:
            filt.set_contacts(changes)
        meas = [(i, p_meas[k, i], R_kin_pos) for i in range(N)]
        out = filt.correct_kinematics(meas)
        if collect_nis:
            for nis, dim in out:
                nis_values.append(nis)
                nis_dims.append(dim)
        R_out.append(filt.X[0:3, 0:3])
        v_out.append(filt.X[0:3, 3])
        p_out.append(filt.X[0:3, 4])

    return {
        "R_WB": torch.stack(R_out),
        "v_W": torch.stack(v_out),
        "p_W": torch.stack(p_out),
        "nis_values": nis_values,
        "nis_dims": nis_dims,
        "final_X": filt.X,
        "final_theta": filt.theta,
        "final_P": filt.P,
        "final_contacts": dict(filt.contacts),
        "final_estimated_contact_positions": dict(filt.estimated_contact_positions),
        "jitter_events": filt.jitter_events,
    }
