# STRIDE PRIME implementation

Contact-aware covariance and shin-geometry calibration for the planar STRIDE
model. A native PRIME FDDP estimator supplies trajectories and local dynamics;
Python computes the upper loss, sensitivity, and parameter update.

## Contents

| Path | Responsibility |
|---|---|
| [`native/`](native/README.md) | C++ PRIME full-information estimator executable |
| [`stride_prime/`](stride_prime/README.md) | Python estimator boundary, sensitivity, and optimizers |
| [`model/`](model/) | Planar STRIDE URDF used by the estimator |
| [`vendor/PRIME/`](vendor/PRIME/) | Compact pinned PRIME/Crocoddyl source subset |
| [`run_calibration.py`](run_calibration.py) | Command-line calibration entry point |

The example reads [`../../matlab/data/stride_demo.mat`](../../matlab/data/stride_demo.mat).

## Build and run

```bash
cd prime/stride_prime
conda env create -f environment.yml
conda activate stride-prime
./build.sh
python run_calibration.py --method sqp --iterations 10 --knots 80
```

`--method` accepts `sqp`, `frank-wolfe`, and `adam`. Use `--data PATH` for a
compatible MAT file and `--knots` to select the estimator horizon.

The vendored source is derived from
[well-robotics/PRIME](https://github.com/well-robotics/PRIME) commit
`b848ceecd451f4786ce39dcefa59e96dbaa369ba` under BSD 3-Clause. See
[`THIRD_PARTY.md`](THIRD_PARTY.md) for the local dependency record.

Return to the [PRIME implementations](../README.md).
