# MATLAB implementation

A stage-structured Fatrop full-information estimator with an adjoint KKT
gradient. SQP--BFGS, Frank--Wolfe, and projected Adam share the same covariance
and kinematic calibration problem.

## Contents

| Path | Responsibility |
|---|---|
| [`+legbical/`](+legbical/README.md) | Calibration, estimator, KKT, and configuration package |
| [`assets/`](assets/) | Paper figures generated from the experiments |
| [`data/`](data/) | STRIDE signals and precomputed kinematic quantities |
| [`run_calibration.m`](run_calibration.m) | Public calibration entry point |
| [`setup.m`](setup.m) | Local MATLAB and CasADi path setup |

## Run

MATLAB, Optimization Toolbox, and a CasADi build with Fatrop are required. Set
`CASADI_MATLAB_PATH` when CasADi is not already on the MATLAB path.

```matlab
cd matlab
calibration = run_calibration(Method="sqp", Horizon="demo");
```

`Method` also accepts `"frank-wolfe"` and `"adam"`; `Horizon="full"` uses the
complete stored trajectory.

Return to the [repository overview](../README.md).
