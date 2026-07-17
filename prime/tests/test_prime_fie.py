#!/usr/bin/env python3
"""Numerical smoke and directional-gradient checks for the PRIME FIE."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stride_prime.calibration import CalibrationProblem
from stride_prime.estimator import PrimeEstimator


def test_prime_fie() -> None:
    data_file = ROOT.parent / "matlab/data/stride_demo.mat"
    estimator = PrimeEstimator(ROOT, data_file, knots=40)
    problem = CalibrationProblem(estimator)
    theta = np.zeros(5)
    analytical = problem.evaluate(theta).gradient
    direction = np.array([0.2, -0.1, 0.3, -0.2, 0.1])
    direction /= np.linalg.norm(direction)
    step = 2e-3
    adjoint = problem.adjoint
    plus = estimator.solve(theta + step * direction)
    minus = estimator.solve(theta - step * direction)
    plus_loss = adjoint.upper_loss_and_state_gradient(plus)[0]
    minus_loss = adjoint.upper_loss_and_state_gradient(minus)[0]
    finite_difference = (plus_loss - minus_loss) / (2 * step)
    directional = float(analytical @ direction)
    relative_error = abs(directional - finite_difference) / max(
        1e-15, abs(directional), abs(finite_difference)
    )
    assert relative_error < 1e-2

    for method in ("sqp", "frank-wolfe", "adam"):
        smoke = CalibrationProblem(
            PrimeEstimator(ROOT, data_file, knots=40)
        ).run(method, theta, 1)
        assert np.isfinite(smoke["loss"])
        assert np.all(np.isfinite(smoke["theta"]))


if __name__ == "__main__":
    test_prime_fie()
