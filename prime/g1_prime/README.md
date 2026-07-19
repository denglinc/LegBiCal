# G1 PRIME covariance calibration

Reproducible bilevel covariance calibration for the Unitree G1. PRIME FDDP and
contact Newton solves form the lower problem; SQP--BFGS or Frank--Wolfe--SDP
minimizes the upper SE(3)-log trajectory loss.

[Open the calibrated G1 visualization](https://dlinc3.github.io/LegBiCal/).

| Clip | Meshcat 0.5x | Meshcat 1x | MuJoCo |
|---|---|---|---|
| run1 | [Open](https://dlinc3.github.io/LegBiCal/media/run1_calibrated.html) | [Open](https://dlinc3.github.io/LegBiCal/media/run1_calibrated.html?speed=1) | `g1cal replay --clip run1` |
| run2 | [Open](https://dlinc3.github.io/LegBiCal/media/run2_calibrated.html) | [Open](https://dlinc3.github.io/LegBiCal/media/run2_calibrated.html?speed=1) | `g1cal replay --clip run2` |

Meshcat defaults to `0.5x`. Visualizations are built on top of PRIME's
excellent estimator (well-robotics/PRIME, BSD-3).

The released upper problem varies the joint-position measurement block and
directly minimizes SE(3)-log loss on the two 501-state clips; all other
covariance coordinates remain fixed, and no accuracy beyond these clips is
claimed.

## Quickstart

The demo renders the shipped calibrated solutions without rerunning the
estimator.

```bash
conda env create -f environment.yml
conda activate g1cal
./scripts/build.sh
python -m pip install -e .
g1cal demo --out out/demo
```

Run either lower problem at the released covariance:

```bash
g1cal solve --clip run1 --covariance data/calibrated/precision.csv
g1cal solve --clip run2 --covariance calibrated
```

MuJoCo replay prescribes `qpos`/`qvel` and calls `mj_forward` at 50 Hz.

```bash
g1cal replay --clip run1 --source calibrated
```

## Repository map

| Directory | Responsibility |
|---|---|
| [`configs/`](configs/README.md) | Lower-solver, replay-scene, and visualization configuration |
| [`cpp/`](cpp/README.md) | PRIME overlay, lower-solver executable, pybind module, and contact model |
| [`data/`](data/README.md) | Released clips, calibrated covariance, and reference solutions |
| [`docs/`](docs/README.md) | GitHub Pages landing page and deployment instructions |
| [`models/`](models/README.md) | Pinned G1 URDF, MJCF, meshes, contact frames, and manifest |
| [`python/`](python/README.md) | Installable `g1cal` package and complete Drake visualization architecture |
| [`scripts/`](scripts/) | Build, Pages generation, and release-maintenance entry points |
| [`third_party/`](third_party/) | Pinned PRIME source and preserved notices |

## Calibration architecture

`CalibrationOracle` maps each optimizer coordinate to block covariance and
precision, runs both 501-state PRIME lower problems, and returns their mean
SE(3)-log loss. Whole-estimator central differences supply the SQP--BFGS or
Frank--Wolfe--SDP update.

For visualization, a saved solution becomes an immutable
`MotionForceSequence`; `MotionForcePlaybackSystem` owns Drake time and
publishes one `VisualizationFrame` contract to the Meshcat and MuJoCo
LeafSystems. See the full [Drake/pydrake ownership and port
structure](python/g1cal/visualization/README.md).

## Commands

| Command | Purpose |
|---|---|
| `g1cal demo` | Render both shipped calibrated solutions to Meshcat HTML |
| `g1cal solve` | Run one lower estimator at a selected covariance |
| `g1cal calibrate` | Run SQP--BFGS or Frank--Wolfe--SDP upper updates |
| `g1cal select` | Select the lowest strict evaluated covariance |
| `g1cal render` | Render one saved solution to self-contained Meshcat HTML |
| `g1cal replay` | Open the passive interactive MuJoCo viewer |

Use `g1cal COMMAND --help` for the exact arguments.

## Reproduce the calibration

Run either method through the shared content-addressed oracle and strict
promotion gate:

```bash
g1cal calibrate --optimizer sqp-bfgs --max-iterations 2 \
  --out out/calibration
g1cal calibrate --optimizer frank-wolfe-sdp --max-iterations 2 \
  --out out/calibration
g1cal select --out out/calibration
```

Lower attempts are immutable; whole-estimator central differences drive both
methods, and the Frank--Wolfe SDP oracle is checked against the analytic
interval endpoint.

## Acknowledgments and licenses

Built on the excellent work of
[PRIME](https://github.com/well-robotics/PRIME) (well-robotics), BSD-3. PRIME
provides the lower estimator's contact machinery; its license and notices are
preserved under [`third_party/PRIME/`](third_party/PRIME/VENDORED.md).
Visualizations are built on top of PRIME's excellent estimator and use its
experiment scene palette.

Unitree G1 model provenance and license text are recorded in
[`models/g1/NOTICE.md`](models/g1/NOTICE.md). Repository code in this subtree is
BSD-3-Clause; third-party material remains subject to its notices.

Return to the [PRIME implementations](../README.md).
