#!/usr/bin/env python3
"""Run contact-aware covariance and shin-geometry calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from stride_prime.calibration import CalibrationProblem
from stride_prime.estimator import PrimeEstimator


ROOT = Path(__file__).resolve().parent


def calibrate(
    method: str = "sqp", iterations: int = 10,
    data_file: Path | None = None, knots: int = 80,
) -> dict:
    if data_file is None:
        data_file = ROOT.parent / "matlab/data/stride_demo.mat"
    estimator = PrimeEstimator(ROOT, data_file, knots=knots)
    problem = CalibrationProblem(estimator)
    theta0 = np.array([0.65, -0.65, 0.55, -0.55, 0.030])
    return problem.run(method, theta0, iterations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=("sqp", "frank-wolfe", "adam"), default="sqp"
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--knots", type=int, default=80)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    result = calibrate(args.method, args.iterations, args.data, args.knots)
    print(f"{result['method']}: theta={result['theta']}")


if __name__ == "__main__":
    main()
