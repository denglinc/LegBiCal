# Estimation subpackage

Stage-structured full-information estimation and implicit differentiation.

| Class | Responsibility |
|---|---|
| [`FIEGraph.m`](FIEGraph.m) | Builds state/noise stages, dynamics, arrival/process/measurement costs, contact-dependent covariance, and sparse KKT derivatives |
| [`FatropEstimator.m`](FatropEstimator.m) | Solves the structured NLP, carries primal/dual warm starts, and exposes the adjoint pullback |
| [`KktSystem.m`](KktSystem.m) | Prepares the sparse KKT matrix once and solves the transposed system for upper gradients |

`FIEGraph` preserves the temporal block structure expected by Fatrop rather
than flattening the problem into a generic dense NLP. State and covariance
derivatives are generated from the same CasADi graph used by the solve.

Return to the [`legbical` package](../README.md).
