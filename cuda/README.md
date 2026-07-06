# Estimation Calibration CUDA

Differentiable Torch/CUDA covariance calibration for a contact-aided
right-invariant EKF on Unitree G1 data. The filter replay is unrolled through
time, covariance blocks are trainable SPD parameters, and training uses
chunked BPTT.

Scope: the default run uses all 7 rollouts for calibration. The reported
metrics are in-sample calibration metrics, not held-out generalization.

## Lineage

This implementation follows the computation-graph view of
[Backprop KF](https://dl.acm.org/doi/10.5555/3157382.3157587) and the legged
robot contact-aided InEKF setting of
[Lin et al., CoRL/PMLR 2022](https://proceedings.mlr.press/v164/lin22b.html).
This release does not include a learned contact-event network; contact
schedules come from provided features, and the learned parameters are
covariance blocks.

## Code Map

- `src/estimation_calibration_cuda/invariant_ekf.py`: original dynamic filter,
  kept as the parity oracle and `--exec sequential` reference.
- `src/estimation_calibration_cuda/fixed_slot_inekf.py`: static-shape,
  fixed-slot, batched fast path.
- `src/estimation_calibration_cuda/batched_calibration.py`: batched trainer,
  batched eval, and whole-chunk CUDA graph capture.
- `src/estimation_calibration_cuda/covariance_calibration.py`: data loading,
  covariance modules, sequential trainer, CLI.
- `benchmarks/profile_replay.py`: profiler and benchmark harness.
- `tests/`: parity, gradient, graph, padding, batching, and smoke tests.
- `docs/gpu_execution_playbook.md`: concise GPU execution notes for future
  estimator changes.

## Environment

`pyproject.toml` also supports uv as a fallback dependency record:

```bash
uv sync --extra cu130 --extra notebooks --extra dev
```

The training path expects CUDA float64.

## Data

Expected dataset root:

```text
/home/dlc/projects/Estimation-Calibration/data/datasets_v0
```

Each rollout has `<stem>.npz` and `<stem>.features.npz`; the manifest is
`dataset_manifest.json`.

## Run

```bash
PYTHONPATH=src /home/dlc/miniforge3/envs/legged_opt/bin/python \
  -m estimation_calibration_cuda.covariance_calibration train \
  --data-root /home/dlc/projects/Estimation-Calibration/data/datasets_v0 \
  --outputs runs/covariance_calibration \
  --epochs 20
```

Defaults are `--exec batched --compile cuda-graph`. Use `--exec sequential`
for the original dynamic-dimension reference path.

## Test And Profile

```bash
PYTHONPATH=src /home/dlc/miniforge3/envs/legged_opt/bin/python -m pytest tests/

PYTHONPATH=src /home/dlc/miniforge3/envs/legged_opt/bin/python \
  benchmarks/profile_replay.py --impl fixed --batch 7 --rows 300 --chunks 10 \
  --with-grad --compile cuda-graph --trace
```

## GPU Execution Summary

The original CUDA replay was launch-bound: dynamic state dimension, per-row
Python control flow, tiny matrix kernels, and hidden device-to-host syncs. The
fast path uses fixed contact slots, masks, batched rollouts, and one CUDA graph
replay per 300-row chunk.

| path | fwd+bwd | throughput | launches/row |
|---|---:|---:|---:|
| dynamic baseline | 6.33 ms/step | 158 rows/s | 1262 |
| fixed-slot + chunk CUDA graph | 1.57 ms/step | 4451 rows/s | 0.03 |

Full 20-epoch calibration: 82 min sequential reference to 8.0 min batched
CUDA graph. Aggregate calibrated vB RMSE: 1.73 vs 1.70 sequential reference.

Future estimator contract:

```python
schedule = build_schedule(host_metadata)
carry = init_carry(seed, batch)
carry, out = step(carry, inputs[:, t], schedule[:, t], params)
```

Keep the hot path tensor-only, batch-first, static-shape, and free of
data-dependent Python branching.

## Outputs

Training writes `initial_covariances.npz`, `calibrated_covariances.npz`,
`calibration_checkpoint.pt`, `full_spd_training_log.json`,
`full_spd_eval_summary.json`, and diagnostic plots under the selected output
directory.
