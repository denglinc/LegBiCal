# C++ overlay

Native G1 full-information estimation layered on the pinned PRIME library.
The parent superbuild compiles PRIME first and then this overlay.

## Contents

| Path | Responsibility |
|---|---|
| [`apps/`](apps/) | `g1_motion_fie` lower-solver command-line executable |
| [`bindings/`](bindings/) | `_g1cal_cpp` in-process pybind entry point |
| [`include/g1cal/`](include/g1cal/) | Motion actions, problem assembly, covariance precision, simulation, preprocessing, and contact profiles |
| [`CMakeLists.txt`](CMakeLists.txt) | Overlay targets and PRIME linkage |

Build from the implementation root:

```bash
./scripts/build.sh
```

The executable and pybind module compile the same estimator translation unit,
so subprocess and in-process lower solves share one numerical implementation.

Return to the [G1 implementation](../README.md).
