# Estimation Calibration CUDA

A compact Torch package for split-safe covariance calibration of a contact-aided invariant EKF.

## Implementation

- Eight fixed contact slots turn events into masks, so rollouts pad once into one
  static batch with no data-dependent Python control flow.
- `torch.compile(fullgraph=True)` fuses the tensor-only EKF step;
  `cuda-graph-compile` captures fixed-chunk forward/backward replay to cut launch overhead.
- Cached tensors and gather-based assembly avoid repeated allocation,
  scatter/atomics, and slow tiny float64 GEMMs while preserving autograd.

## Notebooks

- [Covariance tuning tutorial](notebooks/covariance_tuning_tutorial.ipynb): a small scalar-to-Torch walkthrough.
- [CUDA graph + compile benchmark](notebooks/covariance_calibration_run.ipynb): a repeatable runtime record. An RTX 5090 Laptop GPU with Torch 2.12 and CUDA 13 measured about **0.8–0.9 ms/step** and **8k batched rows/s**; rerun it for your machine.

## Install

Python 3.10–3.14 and Torch 2.11 or newer are required.
Install the Torch build appropriate for your CPU or CUDA environment first,
then install this subproject:

```bash
python -m pip install .
```

Use `python -m pip install '.[notebooks]'` to run the notebooks.

## Quick start

The installed package contains a small train/validation/test dataset named
`example`. It runs without a checkout or external data.

```bash
estimation-calibration-cuda train example -o run \
  --device cpu --compile none --epochs 2 --chunk 32
estimation-calibration-cuda evaluate example --checkpoint run/checkpoint.pt \
  --device cpu
estimation-calibration-cuda inspect run
```

A run contains exactly four files:

| file | purpose |
|---|---|
| `checkpoint.pt` | current training state and validation-selected state |
| `covariances.npz` | validation-selected covariance matrices |
| `metrics.json` | train/validation history and optional one-time test result |
| `manifest.json` | schema, execution facts, and hashes of the other files |

## Python API

The public API has six names:

```python
from estimation_calibration_cuda import (
    CalibrationConfig,
    CalibrationEpisode,
    CalibrationResult,
    calibrate,
    evaluate,
    load_dataset,
)

dataset = load_dataset("example")
config = CalibrationConfig(
    device="cpu", compile_mode="none", epochs=2, chunk=32
)
result = calibrate(dataset, config, output_dir="run")
test_metrics = evaluate(
    dataset, checkpoint="run/checkpoint.pt", split="test", device="cpu"
)
```

Training uses only `train` episodes. Validation body-velocity RMSE selects the
saved covariance state. Test arrays are opened only by the explicit
`evaluate` call, and a run accepts that write once.

## Dataset format

`load_dataset(PATH)` expects `PATH/dataset_manifest.json` and one NPZ per
episode. The manifest schema is:

```json
{
  "schema_version": "estimation-calibration-dataset-v1",
  "episodes": [
    {
      "name": "walk-01",
      "split": "train",
      "source_id": "recording-01",
      "file": "walk-01.npz",
      "sha256": "64-lowercase-hex-characters"
    }
  ]
}
```

Each NPZ contains:

| key | shape | dtype |
|---|---|---|
| `time_s` | `[T]` | float32/float64 |
| `imu` | `[T,6]` | float32/float64 |
| `p_BC` | `[T,N,3]` | float32/float64 |
| `contact_flags` | `[T,N]` | bool |
| `gt_R_WB` | `[T,3,3]` | float32/float64 |
| `gt_v_W` | `[T,3]` | float32/float64 |
| `gt_p_W` | `[T,3]` | float32/float64 |
| `row_valid` | `[T]` | optional bool |

Candidate count `N` must be between one and eight. Timestamps must be finite,
strictly increasing, and near-uniform. If present, true `row_valid` rows must
be one contiguous interval; split an episode instead of placing a hole in the
mask. Files, stems, hashes, and recording lineage are checked across splits
before replay starts.

## Resume

Resume only occurs at an epoch boundary. `epochs` is the new total target:

```bash
estimation-calibration-cuda train example -o resume-run \
  --device cpu --compile none --epochs 2 --chunk 32
estimation-calibration-cuda train example -o resume-run \
  --device cpu --compile none --epochs 4 --chunk 32 --resume
```

Dataset identity and every value except the epoch target must match; CUDA graphs are recreated.

The dynamic and fixed-slot replays accept optional `contact_process_covariance`.
Binary flags still control propagation, correction, insertion, and removal.

## Limits

- Numeric replay and calibration use float64.
- Public episodes support at most eight candidates and pad once at batching.
- Contact schedules are explicit binary inputs; this package does not infer them.
- CPU supports eager and default compile modes; CUDA is required for graph modes.
- Calibration quality depends on dataset coverage and is not asserted by software gates.

## License

The MIT license in `cuda/LICENSE` applies to this `cuda/` subproject only.
