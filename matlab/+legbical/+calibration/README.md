# Calibration subpackage

Shared upper objective and interchangeable parameter updates.

| Class | Responsibility |
|---|---|
| [`CalibrationProblem.m`](CalibrationProblem.m) | Evaluates the estimator, trajectory loss, and KKT-adjoint gradient |
| [`CovarianceParameterization.m`](CovarianceParameterization.m) | Maps the compact parameter vector to bounded covariance and kinematic quantities |
| [`SqpBfgsOptimizer.m`](SqpBfgsOptimizer.m) | Constrained SQP update with a BFGS curvature model |
| [`FrankWolfeOptimizer.m`](FrankWolfeOptimizer.m) | Feasible conditional-gradient update through the covariance oracle |
| [`ProjectedAdamOptimizer.m`](ProjectedAdamOptimizer.m) | First-order adaptive update followed by projection to declared bounds |

Every optimizer consumes the same `CalibrationProblem.evaluate` contract, so
method comparisons do not change the lower estimator or supervised loss.

Return to the [`legbical` package](../README.md).
