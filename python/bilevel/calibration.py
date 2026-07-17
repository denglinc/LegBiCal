"""High-level Frank-Wolfe calibration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from casadi import DM

from .codegen import CodegenFunctions
from .config import BilevelConfig, WeightParameterLayout, default_weight_vector
from .data_io import LeggedDataset
from .estimator.full_information import FullInformationEstimator
from .lmo import LinearMinimizationOracle
from .losses import TrajectoryLoss
from .models import Models
from .robot import B1RobotModel, MeasurementBundle
from .sensitivity import EstimatorSensitivity


LOGGER = logging.getLogger(__name__)


@dataclass
class EstimatorSolution:
    state: np.ndarray
    noise: np.ndarray
    costate: np.ndarray


@dataclass
class CalibrationState:
    theta: np.ndarray
    solution: EstimatorSolution
    measurement: MeasurementBundle
    g_meas: np.ndarray
    loss: float


@dataclass(frozen=True)
class CalibrationResult:
    theta: np.ndarray
    theta_history: np.ndarray
    loss_history: np.ndarray
    gradient_history: np.ndarray
    state_trajectory: np.ndarray


class FrankWolfeCalibrator:
    """Estimator-in-the-loop calibration for the B1 dataset."""

    def __init__(
        self,
        config: BilevelConfig,
        dataset: LeggedDataset,
        robot: B1RobotModel,
        codegen: CodegenFunctions,
        models: Models | None = None,
        estimator: FullInformationEstimator | None = None,
        loss: TrajectoryLoss | None = None,
    ):
        self.config = config
        self.dataset = dataset
        self.window = dataset.window(
            config.dataset.start_idx,
            config.dataset.horizon,
        )
        self.robot = robot
        self.codegen = codegen
        self.models = models or self._build_models()
        self.estimator = estimator or self._build_estimator(self.models)
        self.loss = loss or TrajectoryLoss()
        self.layout = WeightParameterLayout(self.estimator.n_state)
        self.sensitivity = EstimatorSensitivity(self.estimator)
        self.lmo = LinearMinimizationOracle(self.layout, config.frank_wolfe)
        self.prior = self.robot.initial_state_prior(self.window.x[0, :])
        self.g_meas_zero = self.robot.build_measurement_jacobians(
            self.window.q, self.window.v, self.window.u
        )
        self.g_tip_jacobian = self.robot.build_measurement_jacobian_derivatives(
            self.codegen, self.window.q, self.window.v, self.window.u
        )
        self.measurement_zero = self.robot.build_measurements(
            self.window.q, self.window.v, self.window.u, np.zeros(12)
        )
        self.estimator.prepare(
            self.window.u,
            self.prior,
            self.window.horizon,
            self.window.contact,
            self.measurement_zero.y,
            self.measurement_zero.dy_dtip,
            self.g_meas_zero,
            self.g_tip_jacobian,
        )

    def run(self) -> CalibrationResult:
        self.estimator.load_or_build_derivatives(str(self.config.casadi_cache_dir))
        state = self._initial_state()

        max_iter = self.config.frank_wolfe.max_iterations
        theta_history = np.full((max_iter + 1, self.layout.total_size), np.nan)
        loss_history = np.full(max_iter, np.nan)
        gradient_history = np.full(max_iter, np.nan)
        theta_history[0, :] = state.theta

        for iteration in range(1, max_iter + 1):
            LOGGER.info("iteration=%d loss=%.6g", iteration, state.loss)
            loss_history[iteration - 1] = state.loss

            gradient, kkt_inf = self._gradient(state)
            gradient_norm = float(np.linalg.norm(gradient))
            gradient_history[iteration - 1] = gradient_norm
            LOGGER.info("kkt_inf=%.6g gradient_norm=%.6g", kkt_inf, gradient_norm)

            lmo_result = self.lmo.solve(gradient, state.theta)
            LOGGER.debug("LMO status: %s", lmo_result.status)

            direction = lmo_result.point - state.theta
            frank_wolfe_gap = -float(gradient @ direction)
            if frank_wolfe_gap <= (
                self.config.frank_wolfe.gap_tolerance * max(1.0, abs(state.loss))
            ):
                theta_history = theta_history[:iteration, :]
                loss_history = loss_history[:iteration]
                gradient_history = gradient_history[:iteration]
                break
            next_state, gamma, expected = self._armijo(state, gradient, direction)
            actual = next_state.loss - state.loss

            theta_history[iteration, :] = next_state.theta
            state = next_state

            LOGGER.info(
                "step=%.3e predicted_change=%.6g actual_change=%.6g",
                gamma, expected, actual,
            )

        return CalibrationResult(
            theta=state.theta,
            theta_history=theta_history,
            loss_history=loss_history,
            gradient_history=gradient_history,
            state_trajectory=state.solution.state,
        )

    def _build_models(self) -> Models:
        models = Models(self.config.effective_dt)
        models.build_models()
        return models

    def _build_estimator(self, models: Models) -> FullInformationEstimator:
        estimator = FullInformationEstimator(
            self.config.dataset.horizon,
            self.config.effective_dt,
            solver_config=self.config.fatrop,
        )
        estimator.set_state_variable(models.xa)
        estimator.set_output_variable(models.y)
        estimator.set_control_variable(models.u)
        estimator.set_noise_variable(models.w)
        estimator.set_models(models.models_mhe)
        estimator.set_cost_models()
        return estimator

    def _initial_state(self) -> CalibrationState:
        theta_core = np.asarray(default_weight_vector(), dtype=float)
        if theta_core.size != self.layout.core_size:
            raise ValueError(
                f"default weight vector has size {theta_core.size}; "
                f"expected {self.layout.core_size}"
            )
        theta = np.concatenate([theta_core, np.zeros(12), np.zeros(3)])
        measurement = self.robot.build_measurements(
            self.window.q, self.window.v, self.window.u, theta[self.layout.tip_slice]
        )
        g_meas = self._measurement_jacobians(theta[self.layout.tip_slice])
        solution = self._solve_estimator(theta_core, theta[self.layout.tip_slice])
        return CalibrationState(
            theta=theta,
            solution=solution,
            measurement=measurement,
            g_meas=g_meas,
            loss=self._loss_value(solution.state, theta[self.layout.base_slice]),
        )

    def _solve_estimator(
        self,
        theta_core: np.ndarray,
        tip_offset: np.ndarray,
    ) -> EstimatorSolution:
        opt_sol = self.estimator.solve(theta_core, tip_offset)
        return EstimatorSolution(
            state=opt_sol["state_traj_opt"],
            noise=opt_sol["noise_traj_opt"],
            costate=opt_sol["costate"],
        )

    def _loss_value(self, state_traj: np.ndarray, base_offset: np.ndarray) -> float:
        attitude_loss = self._attitude_loss(state_traj)
        return self.loss.evaluate(
            state_traj,
            self.window.x,
            self.window.foot,
            attitude_loss,
            base_offset,
        ).value

    def _gradient(self, state: CalibrationState) -> tuple[np.ndarray, float]:
        dloss_dx = self.loss.state_gradient(
            state.solution.state,
            self.window.x,
            self.window.foot,
            self._attitude_gradient(state.solution.state),
            state.theta[self.layout.base_slice],
        )
        sensitivity = self.sensitivity.pullback(
            state.solution.state,
            state.solution.noise,
            state.solution.costate,
            state.measurement,
            self.window.u,
            self.window.contact,
            self.prior,
            state.theta[: self.layout.core_size],
            state.g_meas,
            self.g_tip_jacobian,
            dloss_dx,
        )
        base_gradient = self.loss.base_offset_gradient(
            state.solution.state,
            self.window.x,
            state.theta[self.layout.base_slice],
        )
        return (
            np.concatenate([sensitivity.gradient, base_gradient]),
            sensitivity.kkt_inf_norm,
        )

    def _armijo(
        self,
        state: CalibrationState,
        gradient: np.ndarray,
        direction: np.ndarray,
    ) -> tuple[CalibrationState, float, float]:
        gamma = self.config.frank_wolfe.armijo_gamma_init
        linear_model = float(gradient @ direction)

        for _ in range(self.config.frank_wolfe.armijo_max_steps):
            theta_candidate = state.theta + gamma * direction
            measurement = self.robot.build_measurements(
                self.window.q,
                self.window.v,
                self.window.u,
                theta_candidate[self.layout.tip_slice],
            )
            g_meas = self._measurement_jacobians(
                theta_candidate[self.layout.tip_slice]
            )
            try:
                solution = self._solve_estimator(
                    theta_candidate[: self.layout.core_size],
                    theta_candidate[self.layout.tip_slice],
                )
            except RuntimeError:
                gamma *= self.config.frank_wolfe.armijo_beta
                if gamma < self.config.frank_wolfe.armijo_min_step:
                    break
                continue
            loss_candidate = self._loss_value(
                solution.state, theta_candidate[self.layout.base_slice]
            )
            candidate = CalibrationState(
                theta=theta_candidate,
                solution=solution,
                measurement=measurement,
                g_meas=g_meas,
                loss=loss_candidate,
            )
            rhs = (
                state.loss
                + self.config.frank_wolfe.armijo_rho * gamma * linear_model
            )
            if loss_candidate <= rhs:
                return candidate, gamma, gamma * linear_model
            gamma *= self.config.frank_wolfe.armijo_beta
            if gamma < self.config.frank_wolfe.armijo_min_step:
                break

        raise RuntimeError("Armijo line search did not find an acceptable step")

    def _measurement_jacobians(self, tip_offset: np.ndarray) -> np.ndarray:
        tip_offset = np.asarray(tip_offset, dtype=float).reshape(12)
        delta = self.g_tip_jacobian @ tip_offset
        blocks = [
            delta[k].reshape(24, 9, order="F") for k in range(self.window.length)
        ]
        return self.g_meas_zero + np.stack(blocks, axis=0)

    def _attitude_gradient(self, state_traj: np.ndarray) -> np.ndarray:
        q_est = state_traj[:, 9:13]
        q_mocap = self.window.x[:, 3:7]
        grad = self.estimator.dL_dQ_fn(
            q=DM(q_est.reshape(-1, 1)),
            qm=DM(q_mocap.reshape(-1, 1)),
        )["dL_dQ"]
        return np.asarray(grad, dtype=float).reshape(self.window.length, 4)

    def _attitude_loss(self, state_traj: np.ndarray) -> float:
        q_est = state_traj[:, 9:13]
        q_mocap = self.window.x[:, 3:7]
        value = self.estimator.L_att_fn(
            q=DM(q_est.reshape(-1, 1)),
            qm=DM(q_mocap.reshape(-1, 1)),
        )["L"]
        return float(value)
