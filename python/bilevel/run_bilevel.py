"""Command-line entry point for B1 bi-level calibration."""

from __future__ import annotations

import argparse
import importlib
import logging

from .config import BilevelConfig, DatasetConfig, FrankWolfeConfig


def _require_runtime_dependencies() -> None:
    missing = []
    for module in ("casadi", "pinocchio", "cvxpy"):
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Install the conda/pip packages described in README.md."
        )

    import casadi as cs

    if not cs.has_nlpsol("fatrop"):
        raise RuntimeError(
            "CasADi is installed, but the Fatrop NLP plugin is not available. "
            "Install a CasADi/Fatrop build that provides nlpsol('fatrop')."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=22000)
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--iterations", type=int, default=75)
    parser.add_argument("--lmo-solver", default="CLARABEL")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    _require_runtime_dependencies()

    from .calibration import FrankWolfeCalibrator
    from .codegen import CodegenLibraryLoader
    from .data_io import DatasetLoader
    from .robot import B1RobotModel

    config = BilevelConfig(
        dataset=DatasetConfig(start_idx=args.start, horizon=args.horizon),
        frank_wolfe=FrankWolfeConfig(
            max_iterations=args.iterations, lmo_solver=args.lmo_solver
        ),
    )
    dataset = DatasetLoader(config).load()
    robot = B1RobotModel.from_config(config)
    codegen = CodegenLibraryLoader(config.external_lib_dir).load()
    FrankWolfeCalibrator(config, dataset, robot, codegen).run()


if __name__ == "__main__":
    main()
