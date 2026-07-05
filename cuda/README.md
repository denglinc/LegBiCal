# Estimation Calibration CUDA

Torch/CUDA implementation of covariance tuning through differentiable filter replay. A contact-aided right-invariant InEKF is replayed over rollout data in Torch tensors, the covariance parameters are trainable SPD blocks, and training runs truncated BPTT through rollout chunks with a standard Torch optimizer. This is a practical CUDA path for covariance tuning, not the symbolic KKT/Fatrop implementation from the paper.

## Estimator

This CUDA implementation uses the contact-aided invariant EKF estimator from Hartley et al. (2020) on Unitree G1. For estimator details, see “Contact-aided invariant extended Kalman filtering for robot state estimation”: https://doi.org/10.1177/0278364919894385

## Code Path

- `src/estimation_calibration_cuda/invariant_ekf.py` — right-invariant InEKF replay with dynamic contact insertion/removal, written as a Torch tensor graph so the replay stays differentiable with respect to the process and kinematic measurement covariances. `replay_inekf_torch` runs a full rollout; `start_filter` / `run_rows` / `detach_filter` split the same replay into blocks for truncated BPTT.
- `src/estimation_calibration_cuda/covariance_calibration.py` — rollout loading from `.npz` files (including the derived contact schedule), SPD covariance modules, the chunked-BPTT training loop with SPD regularization, evaluation, plots, checkpoints, and the CLI.
- `notebooks/covariance_tuning_tutorial.ipynb` — compact SO(3) example introducing the computation graph and gradient flow.
- `notebooks/covariance_calibration_run.ipynb` — thin runner around the library code that keeps saved outputs visible; re-running the training cell starts a new run.

## Environment

This folder is a standalone uv project.

```bash
cd cuda
uv sync --extra cu130 --extra notebooks
```

CPU inspection:

```bash
uv sync --extra cpu --extra notebooks
```

Torch check:

```bash
uv run python - <<'PY'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
PY
```

The training path expects CUDA float64. CPU mode is mainly for imports, reading notebooks, and inspecting saved outputs.

## Data

```text
datasets_v0/
├── dataset_manifest.json
├── <rollout>.npz
└── <rollout>.features.npz
```

The `.npz` file carries IMU, ground truth, and timing; the `.features.npz` file carries the candidate foot kinematics used for measurements and the contact schedule.

## Run

```bash
uv run estimation-calibration-cuda train \
  --data-root /path/to/datasets_v0 \
  --outputs runs/covariance_calibration \
  --epochs 20 \
  --chunk 300 \
  --lr 1e-2
```

Summarize an existing output directory without touching the GPU:

```bash
uv run estimation-calibration-cuda summarize \
  --outputs runs/covariance_calibration
```

## Outputs

Training writes `calibrated_covariances.npz` and `initial_covariances.npz`, `calibration_checkpoint.pt`, `full_spd_training_log.json` and `full_spd_eval_summary.json`, and a `plots/` directory with training-curve, eigenvalue, condition-number, correlation, and NIS diagnostics.
