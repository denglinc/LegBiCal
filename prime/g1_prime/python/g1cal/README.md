# `g1cal` package

Python control plane for the G1 covariance-calibration release. It owns the
content-addressed lower oracle, upper methods, public CLI, and viewer-neutral
rendering inputs.

## Modules

| Module | Responsibility |
|---|---|
| [`cli.py`](cli.py) | `g1cal` commands and argument validation |
| [`calibration.py`](calibration.py) | Two-clip `CalibrationOracle`, strict gates, and released covariance values |
| [`optimizers.py`](optimizers.py) | SQP--BFGS, Frank--Wolfe--SDP, and strict result selection |
| [`gradient.py`](gradient.py) | Whole-estimator central finite difference |
| [`covariance.py`](covariance.py) | Versioned block-isotropic covariance parameterization |
| [`backend.py`](backend.py), [`horizon_solver.py`](horizon_solver.py), [`attempts.py`](attempts.py) | Native execution, long-horizon continuation, immutable attempts, and promotion |
| [`loss.py`](loss.py), [`gt_clips.py`](gt_clips.py) | SE(3)-log trajectory loss and ground-truth coordinate adapters |
| [`profiles.py`](profiles.py), [`paths.py`](paths.py) | Frozen model profiles, hashes, and repository path policy |
| [`rendering.py`](rendering.py) | Meshcat export and passive MuJoCo replay entry points |
| [`visualization/`](visualization/README.md) | Full Drake/pydrake visualization graph and backend implementations |

The supported user interface is the `g1cal` CLI documented in the
[implementation README](../../README.md). Modules remain importable for
advanced orchestration.

Return to the [Python package source](../README.md).
