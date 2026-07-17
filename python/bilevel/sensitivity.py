"""Adjoint KKT pullback for the three-dimensional FIE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from casadi import DM
from scipy.sparse.linalg import factorized

from .robot import MeasurementBundle


@dataclass(frozen=True)
class SensitivityResult:
    gradient: np.ndarray
    kkt_inf_norm: float


def _casadi_to_csc(matrix, shape: tuple[int, int]) -> sp.csc_matrix:
    sparsity = matrix.sparsity()
    rows = np.asarray(sparsity.row(), dtype=np.int32)
    colptr = np.asarray(sparsity.colind(), dtype=np.int32)
    values = np.asarray(matrix.nonzeros(), dtype=np.float64)
    return sp.csc_matrix((values, rows, colptr), shape=shape)


class SparseKktFactorization:
    """Cache the sparse factorization for repeated pullbacks at one solution."""

    def __init__(self) -> None:
        self._matrix: sp.csc_matrix | None = None
        self._solve_transpose = None

    def prepare(self, matrix: sp.csc_matrix) -> None:
        matrix = matrix.tocsc()
        if self._same_matrix(matrix):
            return
        self._solve_transpose = factorized(matrix.T.tocsc())
        self._matrix = matrix.copy()

    def solve_transpose(self, rhs: np.ndarray) -> np.ndarray:
        if self._solve_transpose is None:
            raise RuntimeError("KKT factorization has not been prepared")
        return np.asarray(self._solve_transpose(rhs), dtype=float)

    def _same_matrix(self, matrix: sp.csc_matrix) -> bool:
        cached = self._matrix
        return (
            cached is not None
            and cached.shape == matrix.shape
            and np.array_equal(cached.indptr, matrix.indptr)
            and np.array_equal(cached.indices, matrix.indices)
            and np.array_equal(cached.data, matrix.data)
        )


class EstimatorSensitivity:
    """Apply the implicit first-order derivative with one adjoint solve."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.factorization = SparseKktFactorization()

    def pullback(
        self,
        state_traj: np.ndarray,
        noise_traj: np.ndarray,
        costate_traj: np.ndarray,
        measurement: MeasurementBundle,
        controls: np.ndarray,
        contacts: np.ndarray,
        prior: np.ndarray,
        theta_core: np.ndarray,
        g_meas: np.ndarray,
        g_tip_jacobian: np.ndarray,
        loss_state_gradient: np.ndarray,
    ) -> SensitivityResult:
        horizon = g_meas.shape[0] - 1
        n_state = self.estimator.n_state
        x_vec = np.asarray(state_traj, dtype=float).reshape(-1, 1)
        w_vec = np.asarray(noise_traj, dtype=float).reshape(-1, 1)
        lambda_vec = np.asarray(costate_traj, dtype=float).reshape(-1, 1)
        y_vec = np.asarray(measurement.y, dtype=float).reshape(-1, 1)
        u_vec = np.asarray(controls[:-1, :], dtype=float).reshape(-1, 1)
        c_vec = np.asarray(contacts, dtype=float).reshape(-1, 1)
        g_vec = DM(g_meas.reshape(-1, 1))
        theta_core = np.asarray(theta_core, dtype=float).reshape(-1)

        arguments = {
            "s": x_vec,
            "n": w_vec,
            "costate": lambda_vec,
            "y": y_vec,
            "u": u_vec,
            "c": c_vec,
            "prior": prior,
            "tp": theta_core.tolist(),
            "G": g_vec,
        }
        kkt_value = self.estimator.KKT_fn(**arguments)["KKT_fn"]
        d_kkt_z = self.estimator.dKKT_Z_fn(**arguments)["dKKT_Z_fn"]

        n_system = int(kkt_value.size1())
        fz = _casadi_to_csc(d_kkt_z, (n_system, n_system))
        fz = (0.5 * (fz + fz.T)).tocsc()

        state_gradient = np.asarray(loss_state_gradient, dtype=float).reshape(-1)
        expected = (horizon + 1) * n_state
        if state_gradient.size != expected:
            raise ValueError(
                f"state loss gradient has size {state_gradient.size}; expected {expected}"
            )
        adjoint_rhs = np.zeros(n_system, dtype=float)
        adjoint_rhs[:expected] = state_gradient
        self.factorization.prepare(fz)
        adjoint = self.factorization.solve_transpose(adjoint_rhs)

        vjp_arguments = {**arguments, "adjoint": adjoint}
        core_gradient = -np.asarray(
            self.estimator.KKT_tp_vjp_fn(**vjp_arguments)["KKT_tp_vjp"]
        ).reshape(-1)
        y_pullback = np.asarray(
            self.estimator.KKT_Y_vjp_fn(**vjp_arguments)["KKT_Y_vjp"]
        ).reshape(-1)
        g_pullback = np.asarray(
            self.estimator.KKT_G_vjp_fn(**vjp_arguments)["KKT_G_vjp"]
        ).reshape(-1)
        tip_gradient = -np.asarray(measurement.dy_dtip.T @ y_pullback).reshape(-1)
        for k in range(horizon + 1):
            col0 = k * 24 * 9
            dg_k = g_tip_jacobian[k]
            tip_gradient -= np.asarray(
                dg_k.T @ g_pullback[col0 : col0 + 24 * 9]
            ).reshape(-1)

        return SensitivityResult(
            gradient=np.concatenate([core_gradient, tip_gradient]),
            kkt_inf_norm=float(np.linalg.norm(kkt_value.full(), np.inf)),
        )
