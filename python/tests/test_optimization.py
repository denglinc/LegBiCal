"""Smoke and directional-gradient checks for the optimized 3-D pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from casadi import DM

from bilevel.codegen import CodegenLibraryLoader
from bilevel.config import (
    BilevelConfig,
    DatasetConfig,
    FrankWolfeConfig,
    WeightParameterLayout,
    default_weight_vector,
)
from bilevel.data_io import DatasetLoader
from bilevel.estimator.full_information import FullInformationEstimator
from bilevel.losses import TrajectoryLoss
from bilevel.lmo import LinearMinimizationOracle
from bilevel.models import Models
from bilevel.robot import B1RobotModel
from bilevel.sensitivity import EstimatorSensitivity


def test_optimization() -> None:
    root = Path(__file__).resolve().parents[2]
    config = BilevelConfig(
        repo_root=root,
        dataset=DatasetConfig(start_idx=22000, horizon=3),
    )
    window = DatasetLoader(config).load().window(
        config.dataset.start_idx, config.dataset.horizon
    )
    robot = B1RobotModel.from_config(config)
    codegen = CodegenLibraryLoader(config.external_lib_dir).load()

    models = Models(config.effective_dt)
    models.build_models()
    estimator = FullInformationEstimator(
        window.horizon, config.effective_dt, config.fatrop
    )
    estimator.set_state_variable(models.xa)
    estimator.set_output_variable(models.y)
    estimator.set_control_variable(models.u)
    estimator.set_noise_variable(models.w)
    estimator.set_models(models.models_mhe)
    estimator.set_cost_models()
    estimator.diffKKT()
    estimator.diffquat()

    prior = robot.initial_state_prior(window.x[0, :])
    g_zero = robot.build_measurement_jacobians(window.q, window.v, window.u)
    dg_dtip = robot.build_measurement_jacobian_derivatives(
        codegen, window.q, window.v, window.u
    )
    measurement_zero = robot.build_measurements(window.q, window.v, window.u, np.zeros(12))
    estimator.prepare(
        window.u, prior, window.horizon, window.contact,
        measurement_zero.y, measurement_zero.dy_dtip, g_zero, dg_dtip,
    )
    solver_identity = id(estimator._solver)

    theta_core = np.asarray(default_weight_vector(), dtype=float)
    tip = np.zeros(12)
    measurement = robot.build_measurements(window.q, window.v, window.u, tip)
    g_meas = measurement_jacobians(g_zero, dg_dtip, tip)
    lower = estimator.solve(theta_core, tip)
    repeated = estimator.solve(theta_core, tip)
    assert repeated is lower
    assert id(estimator._solver) == solver_identity

    loss = TrajectoryLoss()
    attitude_gradient = np.asarray(
        estimator.dL_dQ_fn(
            q=DM(lower["state_traj_opt"][:, 9:13].reshape(-1, 1)),
            qm=DM(window.x[:, 3:7].reshape(-1, 1)),
        )["dL_dQ"]
    ).reshape(window.length, 4)
    loss_gradient = loss.state_gradient(
        lower["state_traj_opt"],
        window.x,
        window.foot,
        attitude_gradient,
    )
    sensitivity_engine = EstimatorSensitivity(estimator)
    pullback_arguments = (
        lower["state_traj_opt"],
        lower["noise_traj_opt"],
        lower["costate"],
        measurement,
        window.u,
        window.contact,
        prior,
        theta_core,
        g_meas,
        dg_dtip,
        loss_gradient,
    )
    sensitivity = sensitivity_engine.pullback(*pullback_arguments)
    factorization_identity = id(sensitivity_engine.factorization._solve_transpose)
    repeated_sensitivity = sensitivity_engine.pullback(*pullback_arguments)
    assert id(sensitivity_engine.factorization._solve_transpose) == factorization_identity
    np.testing.assert_allclose(repeated_sensitivity.gradient, sensitivity.gradient)

    direction = np.zeros(12)
    direction[2] = 1.0
    step = 2e-4
    plus = lower_loss(
        estimator, robot, loss, window, prior, theta_core,
        g_zero, dg_dtip, step * direction,
    )
    minus = lower_loss(
        estimator, robot, loss, window, prior, theta_core,
        g_zero, dg_dtip, -step * direction,
    )
    finite_difference = (plus - minus) / (2.0 * step)
    adjoint = float(sensitivity.gradient[theta_core.size + 2])
    relative_error = abs(adjoint - finite_difference) / max(
        1.0, abs(adjoint), abs(finite_difference)
    )
    assert np.isfinite(sensitivity.gradient).all()
    assert relative_error < 2e-2
    assert id(estimator._solver) == solver_identity
    np.testing.assert_allclose(
        np.linalg.norm(lower["state_traj_opt"][1:, 9:13], axis=1),
        1.0,
        atol=1e-10,
    )
    assert abs(np.linalg.norm(lower["state_traj_opt"][0, 9:13]) - 1.0) < 1e-3

    core_direction = theta_core.copy()
    core_step = 1e-4
    core_plus = lower_loss(
        estimator, robot, loss, window, prior,
        theta_core + core_step * core_direction,
        g_zero, dg_dtip, tip,
    )
    core_minus = lower_loss(
        estimator, robot, loss, window, prior,
        theta_core - core_step * core_direction,
        g_zero, dg_dtip, tip,
    )
    core_finite_difference = (core_plus - core_minus) / (2.0 * core_step)
    core_adjoint = float(sensitivity.gradient[: theta_core.size] @ core_direction)
    core_relative_error = abs(core_adjoint - core_finite_difference) / max(
        1.0, abs(core_adjoint), abs(core_finite_difference)
    )
    assert core_relative_error < 2e-2


def test_lmo_problem_is_reused() -> None:
    layout = WeightParameterLayout(28)
    theta = np.concatenate([default_weight_vector(), np.zeros(15)])
    oracle = LinearMinimizationOracle(layout, FrankWolfeConfig())
    problem_identity = id(oracle._problem)
    first = oracle.solve(np.ones(layout.total_size), theta)
    second = oracle.solve(-np.ones(layout.total_size), theta)
    assert id(oracle._problem) == problem_identity
    assert first.status.startswith("optimal")
    assert second.status.startswith("optimal")


def measurement_jacobians(
    g_zero: np.ndarray, dg_dtip: np.ndarray, tip: np.ndarray
) -> np.ndarray:
    delta = dg_dtip @ np.asarray(tip, dtype=float).reshape(12)
    return g_zero + np.stack(
        [row.reshape(24, 9, order="F") for row in delta], axis=0
    )


def lower_loss(
    estimator,
    robot,
    loss,
    window,
    prior,
    theta_core,
    g_zero,
    dg_dtip,
    tip,
) -> float:
    measurement = robot.build_measurements(window.q, window.v, window.u, tip)
    g_meas = measurement_jacobians(g_zero, dg_dtip, tip)
    solution = estimator.solve(theta_core, tip)
    attitude = float(
        estimator.L_att_fn(
            q=DM(solution["state_traj_opt"][:, 9:13].reshape(-1, 1)),
            qm=DM(window.x[:, 3:7].reshape(-1, 1)),
        )["L"]
    )
    return loss.evaluate(
        solution["state_traj_opt"], window.x, window.foot, attitude
    ).value


if __name__ == "__main__":
    test_optimization()
