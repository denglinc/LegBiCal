"""Sparse Gauss--Newton KKT adjoint for covariance calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import splu

from .estimator import PrimeSolution


@dataclass(frozen=True)
class AdjointResult:
    gradient: np.ndarray
    adjoint: np.ndarray


class GaussNewtonKKTAdjoint:
    """Differentiate the eliminated process-noise smoother.

    The approximation matches the Gauss--Newton Hessian consumed by FDDP: it
    keeps all first derivatives of PRIME contact dynamics and drops dynamics
    second derivatives. Covariance mixed derivatives are exact.
    """

    _BASE_Q = np.arange(3)
    _BASE_V = np.arange(7, 10)

    def __init__(self) -> None:
        measurement_sigma = np.array(
            [0.002, 0.002, 0.003, 0.006, 0.006, 0.006, 0.006,
             0.020, 0.020, 0.025, 0.040, 0.040, 0.040, 0.040]
        )
        process_sigma = np.array(
            [0.001, 0.001, 0.0015, 0.012, 0.012, 0.012, 0.012,
             0.030, 0.030, 0.0375, 0.160, 0.160, 0.160, 0.160]
        )
        self.measurement_weight0 = measurement_sigma ** -2
        self.process_weight0 = process_sigma ** -2

    def weights(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.asarray(theta, dtype=float)
        measurement = self.measurement_weight0.copy()
        process = self.process_weight0.copy()
        measurement[self._BASE_Q] *= np.exp(-2.0 * theta[0])
        measurement[self._BASE_V] *= np.exp(-2.0 * theta[1])
        process[self._BASE_Q] *= np.exp(-2.0 * theta[2])
        process[self._BASE_V] *= np.exp(-2.0 * theta[3])
        return measurement, process

    def upper_loss_and_state_gradient(
        self, solution: PrimeSolution
    ) -> tuple[float, np.ndarray]:
        residual = solution.state - solution.truth
        # The requested supervised calibration target is x,z,vx,vz.
        scale = np.array([0.02, 0.02, 0.20, 0.20])
        indices = np.array([0, 1, 7, 8])
        normalized = residual[:, indices] / scale
        loss = 0.5 * np.mean(normalized ** 2)
        gradient = np.zeros_like(solution.state)
        gradient[:, indices] = residual[:, indices] / (scale ** 2)
        gradient[:, indices] /= normalized.size
        return float(loss), gradient

    def differentiate(
        self, solution: PrimeSolution, dloss_dx: np.ndarray
    ) -> AdjointResult:
        x = solution.state
        y = solution.measurement
        process_residual = solution.process
        dynamics = solution.dynamics
        knots, nx = x.shape
        measurement_weight, process_weight = self.weights(solution.theta)
        wy = np.diag(measurement_weight)
        wp = np.diag(process_weight)
        hessian = lil_matrix((knots * nx, knots * nx), dtype=float)

        def block(k: int) -> slice:
            return slice(k * nx, (k + 1) * nx)

        for k in range(knots):
            hessian[block(k), block(k)] += wy
        # Arrival prior uses the same nominal covariance in this implementation.
        hessian[block(0), block(0)] += wy
        for k, a in enumerate(dynamics):
            hessian[block(k), block(k)] += (
                a.T @ wp @ a - solution.dynamics_H_correction[k]
            )
            hessian[block(k + 1), block(k + 1)] += wp
            cross = -a.T @ wp
            hessian[block(k), block(k + 1)] += cross
            hessian[block(k + 1), block(k)] += cross.T

        hessian = csc_matrix(hessian)
        factor = splu(hessian, permc_spec="COLAMD")
        adjoint = factor.solve(np.asarray(dloss_dx, dtype=float).reshape(-1))

        g_theta = np.zeros((knots * nx, 5))
        measurement_residual = x - y
        measurement_masks = [self._BASE_Q, self._BASE_V]
        for parameter, indices in enumerate(measurement_masks):
            derivative_weight = np.zeros(nx)
            derivative_weight[indices] = -2.0 * measurement_weight[indices]
            for k in range(knots):
                g_theta[block(k), parameter] += (
                    derivative_weight * measurement_residual[k]
                )
            g_theta[block(0), parameter] += (
                derivative_weight * measurement_residual[0]
            )

        process_masks = [self._BASE_Q, self._BASE_V]
        for local_parameter, indices in enumerate(process_masks):
            parameter = local_parameter + 2
            derivative_weight = np.zeros(nx)
            derivative_weight[indices] = -2.0 * process_weight[indices]
            for k, (a, residual) in enumerate(zip(dynamics, process_residual)):
                weighted = derivative_weight * residual
                g_theta[block(k), parameter] -= a.T @ weighted
                g_theta[block(k + 1), parameter] += weighted

        # Shin geometry affects only the contact dynamics. f_theta and A_theta
        # are inexpensive local central derivatives at the converged trajectory;
        # the horizon coupling remains the single analytical KKT adjoint.
        for k, (a, a_theta, f_theta, residual) in enumerate(zip(
            dynamics, solution.dynamics_A_shin, solution.dynamics_shin,
            process_residual
        )):
            weighted = process_weight * residual
            weighted_f = process_weight * f_theta
            g_theta[block(k), 4] += -a_theta.T @ weighted + a.T @ weighted_f
            g_theta[block(k + 1), 4] -= weighted_f

        gradient = -(adjoint @ g_theta)
        return AdjointResult(
            gradient=np.asarray(gradient), adjoint=adjoint,
        )
