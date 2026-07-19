# CUDA implementation

A compact Torch package for split-safe covariance calibration of a
contact-aided invariant EKF.

## Implementation

- Eight fixed contact slots convert insertion and removal events into masks.
  Every rollout is padded once into a static batch, with no data-dependent
  Python control flow in the filter step.
- `torch.compile(fullgraph=True)` fuses the tensor-only InEKF step.
  `cuda-graph-compile` additionally captures fixed-chunk forward and backward
  replay to reduce kernel-launch overhead.
- Cached constants, gather-based matrix assembly, and synchronized chunks
  avoid repeated allocation, scatter/atomics, and tiny float64 GEMMs while
  preserving autograd.
- A dynamic-dimension InEKF remains as the mathematical parity oracle; the
  fixed-slot implementation changes execution shape, not filter equations.
- Six full-SPD covariance blocks use scaled Cholesky factors with positive
  softplus diagonals, conditioning regularization, and explicit covariance
  floors.

## Contents

| Path | Responsibility |
|---|---|
| [`src/estimation_calibration_cuda/`](src/estimation_calibration_cuda/README.md) | Public API, data contracts, dynamic/fixed-slot InEKF, batching, compilation, and artifacts |
| [`notebooks/`](notebooks/README.md) | Differentiation tutorial and reproducible CUDA benchmark |
| [`benchmarks/`](benchmarks/) | Lightweight replay profiler |
| [`tests/`](tests/) | Numeric blocks, gradients, parity, datasets, training, and release-surface coverage |

## Install

Python 3.10--3.14 and Torch 2.11 or newer are required. Install the Torch build
appropriate for the CPU or CUDA environment first.

```bash
cd cuda
python -m pip install .
```

Use `python -m pip install '.[notebooks]'` for the notebook environment.

## Quick start

The installed package contains a small train/validation/test dataset named
`example`; it needs no external files.

```bash
estimation-calibration-cuda train example -o run \
  --device cpu --compile none --epochs 2 --chunk 32
estimation-calibration-cuda evaluate example --checkpoint run/checkpoint.pt \
  --device cpu
estimation-calibration-cuda inspect run
```

A run contains exactly four files:

| File | Purpose |
|---|---|
| `checkpoint.pt` | Current training state and validation-selected state |
| `covariances.npz` | Validation-selected covariance matrices |
| `metrics.json` | Train/validation history and optional one-time test result |
| `manifest.json` | Schema, execution facts, and hashes of the other files |

Training opens only train episodes. Validation body-velocity RMSE selects the
saved covariance state. Test arrays are opened only by the explicit
`evaluate` command, and a run accepts that write once.

## Notebooks

The committed notebooks retain their outputs. These are machine-specific
execution records, not estimation-quality claims.

| Notebook | Recorded result |
|---|---|
| [CUDA graph + compile benchmark](notebooks/covariance_calibration_run.ipynb) | RTX 5090 Laptop, Torch 2.12/CUDA 13: **0.8967 ms/step**, **7,806 batched rows/s**, **0.1718 GB** peak |
| [SO(3) covariance tuning](notebooks/covariance_tuning_tutorial.ipynb) | Scalar graph gradients matched central finite differences at about `1e-13`; AdamW reduced loss from `7.282e-03` to `1.182e-04` in `32.37 s` |

On the recorded benchmark, the fixed replay ran at about **0.8–0.9 ms/step**.

See [`notebooks/README.md`](notebooks/README.md) for the experiment shapes and
rerun requirements.

## Python API

```python
from estimation_calibration_cuda import (
    CalibrationConfig,
    calibrate,
    evaluate,
    load_dataset,
)

dataset = load_dataset("example")
result = calibrate(
    dataset,
    CalibrationConfig(device="cpu", compile_mode="none", epochs=2, chunk=32),
    output_dir="run",
)
test_metrics = evaluate(
    dataset, checkpoint="run/checkpoint.pt", split="test", device="cpu"
)
```

Custom dataset schemas, resume behavior, and module boundaries are documented
in the [package README](src/estimation_calibration_cuda/README.md).

## Test

```bash
python -m pip install -e '.[dev]'
pytest
```

Numeric replay uses float64. Public episodes support at most eight contact
candidates; contact schedules are explicit binary inputs, and an optional
`contact_process_covariance` supplies contact-point process noise. CPU supports
eager and default compile modes, while CUDA is required for graph modes.

Return to the [repository overview](../README.md).
