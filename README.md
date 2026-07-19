# LegBiCal

Reference implementations for *Simultaneous Calibration of Noise Covariance
and Kinematics for State Estimation of Legged Robots via Bi-level
Optimization* ([arXiv:2510.11539](https://arxiv.org/abs/2510.11539)).

[Explore the calibrated G1 PRIME visualization](https://dlinc3.github.io/LegBiCal/).

## Implementations

| Priority | Directory | Estimator and calibration path |
|---:|---|---|
| 1 | [`cuda/`](cuda/README.md) | Batched Torch CPU/CUDA covariance calibration for a contact-aided InEKF |
| 2 | [`prime/`](prime/README.md) | Contact-aware PRIME FDDP implementations for STRIDE and Unitree G1 |
| 3 | [`matlab/`](matlab/README.md) | Stage-structured Fatrop FIE with covariance and kinematic calibration |
| 4 | [`python/`](python/README.md) | Hardware-oriented B1 FIE with sparse-adjoint bilevel calibration |

Each implementation owns its environment, commands, and architecture notes.
Start from its README and follow the local links one level at a time.

## Project information

- Citation metadata: [`CITATION.cff`](CITATION.cff)
- Repository license: [`LICENSE`](LICENSE)
- Third-party software and data: [`THIRD_PARTY.md`](THIRD_PARTY.md)
