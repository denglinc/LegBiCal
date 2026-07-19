# LegBiCal: Estimation & Calibration for Legged Robots

Reference implementations for *Simultaneous Calibration of Noise Covariance
and Kinematics for State Estimation of Legged Robots via Bi-level
Optimization* ([arXiv:2510.11539](https://arxiv.org/abs/2510.11539)).

[Explore the calibrated G1 PRIME visualization](https://dlinc3.github.io/LegBiCal/).

## Implementations

| Directory | Estimator and calibration path |
|---|---|
| [`cuda/`](cuda/README.md) | Batched Torch CPU/CUDA covariance calibration for a contact-aided InEKF |
| [`prime/`](prime/README.md) | Contact-aware PRIME FDDP implementations for STRIDE and Unitree G1 |
| [`matlab/`](matlab/README.md) | Stage-structured Fatrop FIE with covariance and kinematic calibration |
| [`python/`](python/README.md) | Hardware-oriented B1 FIE with sparse-adjoint bilevel calibration |

Each implementation is self-contained; see its README.
