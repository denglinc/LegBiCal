# STRIDE native estimator

[`prime_fie.cpp`](prime_fie.cpp) is the narrow C++ boundary around the planar
PRIME FDDP lower problem.

It loads the fixed STRIDE URDF and aligned CSV horizon, maps the five calibrated
scales into estimator weights and shin geometry, solves the contact-aware
trajectory, and writes states plus the local dynamics needed by the Python KKT
adjoint. The executable keeps PRIME/contact mechanics native while exposing a
stable tabular contract to `PrimeEstimator`.

Return to the [STRIDE implementation](../README.md).
