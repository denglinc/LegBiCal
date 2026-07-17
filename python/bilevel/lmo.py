"""Linear minimization oracle for the Frank-Wolfe upper level."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cvxpy as cp

from .config import FrankWolfeConfig, WeightParameterLayout


@dataclass(frozen=True)
class LMOResult:
    point: np.ndarray
    status: str
    objective: float


class LinearMinimizationOracle:
    """CVXPY formulation of the feasible-set LMO."""

    def __init__(self, layout: WeightParameterLayout, config: FrankWolfeConfig):
        self.layout = layout
        self.config = config
        size = layout.total_size
        self._variable = cp.Variable(size)
        self._gradient = cp.Parameter(size)
        self._theta = cp.Parameter(size)
        self._lower = cp.Parameter(size)
        self._upper = cp.Parameter(size)
        self._adaptive_box = cp.Parameter(size, nonneg=True)
        self._trust_region = cp.Parameter(size, nonneg=True)
        constraints = self._constraints()
        self._problem = cp.Problem(
            cp.Minimize(self._gradient @ self._variable), constraints
        )

    def solve(self, gradient: np.ndarray, theta: np.ndarray) -> LMOResult:
        gradient = np.asarray(gradient, dtype=float).reshape(-1)
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if gradient.size != self.layout.total_size:
            raise ValueError(
                f"gradient has size {gradient.size}; expected {self.layout.total_size}"
            )
        if theta.size != self.layout.total_size:
            raise ValueError(
                f"theta has size {theta.size}; expected {self.layout.total_size}"
            )

        lower, upper = self.variable_bounds(theta)
        adaptive_box = self.config.adaptive_abs_box_scale * np.maximum(
            1.0, np.abs(theta)
        )
        adaptive_box = np.maximum(adaptive_box, np.maximum(np.abs(lower), np.abs(upper)))
        distance_to_box = np.maximum(
            0.0, np.maximum(lower - theta, theta - upper)
        )
        trust_region = np.maximum(
            self.config.trust_region_radius, distance_to_box
        )
        self._gradient.value = gradient
        self._theta.value = theta
        self._lower.value = lower
        self._upper.value = upper
        self._adaptive_box.value = adaptive_box
        self._trust_region.value = trust_region

        solver = getattr(cp, self.config.lmo_solver, self.config.lmo_solver)
        try:
            self._problem.solve(solver=solver, verbose=False, warm_start=True)
        except cp.error.SolverError:
            if self.config.lmo_solver == "CLARABEL":
                raise
            self._problem.solve(
                solver=cp.CLARABEL, verbose=False, warm_start=True
            )

        if self._variable.value is None:
            raise RuntimeError(f"LMO failed with status {self._problem.status}")
        return LMOResult(
            point=np.asarray(self._variable.value, dtype=float).reshape(-1),
            status=str(self._problem.status),
            objective=float(self._problem.value),
        )

    def variable_bounds(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return scalar lower/upper bounds without PSD constraints."""

        cfg = self.config
        layout = self.layout
        theta = np.asarray(theta, dtype=float).reshape(-1)

        lb = -cfg.big_box * np.ones(layout.total_size)
        ub = cfg.big_box * np.ones(layout.total_size)

        lb[layout.arrival_slice] = cfg.core_min
        ub[layout.arrival_slice] = cfg.core_max

        for block in (
            *layout.measurement_covariance_slices,
            *layout.noise_covariance_slices,
        ):
            lb[block.start : block.start + 3] = cfg.eps_diag
            ub[block.start : block.start + 3] = cfg.core_max
            lb[block.start + 3 : block.stop] = -cfg.core_max
            ub[block.start + 3 : block.stop] = cfg.core_max

        lb[layout.random_walk_slice] = cfg.eps_diag
        ub[layout.random_walk_slice] = cfg.core_max

        lb[layout.swing_slice] = cfg.qswing_min
        ub[layout.swing_slice] = max(cfg.qswing_max, float(np.max(theta[layout.swing_slice])))

        lb[layout.stance_slice] = cfg.qstance_min
        ub[layout.stance_slice] = max(cfg.qstance_max, float(np.max(theta[layout.stance_slice])))

        lb[layout.tip_slice] = -cfg.tip_bound
        ub[layout.tip_slice] = cfg.tip_bound
        lb[layout.base_slice] = -cfg.base_bound
        ub[layout.base_slice] = cfg.base_bound
        return lb, ub

    def _constraints(self) -> list:
        cfg = self.config
        layout = self.layout
        idx_noise = layout.noise_slice.start
        x_opt = self._variable

        constraints = [
            cp.abs(x_opt) <= self._adaptive_box,
            cp.abs(x_opt - self._theta) <= self._trust_region,
            x_opt >= self._lower,
            x_opt <= self._upper,
        ]

        meas = layout.measurement_slice.start
        for offset in (meas, meas + 6, idx_noise, idx_noise + 6):
            block = self._symmetric_3x3(x_opt[offset : offset + 6])
            constraints += [
                block >> cfg.eps_psd * np.eye(3),
                cp.trace(block) <= cfg.trace_cap,
            ]

        constraints += [
            x_opt[idx_noise + 12 : idx_noise + 15] >= cfg.eps_diag,
            x_opt[idx_noise + 15 : idx_noise + 18] >= cfg.eps_diag,
            x_opt[idx_noise + 18 : idx_noise + 21] >= cfg.eps_diag,
            x_opt[idx_noise + 21 : idx_noise + 24] >= cfg.eps_diag,
        ]
        return constraints

    @staticmethod
    def _symmetric_3x3(v6):
        return cp.bmat(
            [
                [v6[0], v6[3], v6[4]],
                [v6[3], v6[1], v6[5]],
                [v6[4], v6[5], v6[2]],
            ]
        )
