# `legbical` MATLAB package

Package boundary for the structured lower estimator and interchangeable
upper-level calibration methods.

`config.options` defines the problem used by `FIEGraph` and
`FatropEstimator`. The lower layer owns estimation and local derivatives;
`CalibrationProblem` exposes only its trajectory, loss gradient, and
KKT-adjoint pullback to SQP--BFGS, Frank--Wolfe, and projected Adam.

## Subpackages

| Path | Responsibility |
|---|---|
| [`+estimation/`](+estimation/README.md) | Stage graph, Fatrop solve, warm start, KKT factorization, and adjoint pullback |
| [`+calibration/`](+calibration/README.md) | Parameterization, supervised objective, and three upper updates |
| [`+config/`](+config/README.md) | One declared set of dimensions, horizons, bounds, solver options, and loss weights |

Return to the [MATLAB implementation](../README.md).
