# `estimation_calibration_cuda`

Torch implementation of split-aware covariance calibration around a
contact-aided right-invariant EKF.

## Data flow

```mermaid
flowchart LR
  D[Dataset manifest + NPZ episodes] --> V[data.py validation]
  V --> A[api.py run contract]
  A --> B[batched_calibration.py]
  B --> F[fixed_slot_inekf.py]
  B --> C[covariance_calibration.py]
  F --> O[checkpoint + covariances + metrics + manifest]
  C --> O
  R[invariant_ekf.py<br/>dynamic parity oracle] -. equation parity .-> F
```

## Modules

| Module | Responsibility |
|---|---|
| [`api.py`](api.py) | Atomic run lifecycle, resume identity, validation selection, one-time test evaluation, and artifact hashes |
| [`cli.py`](cli.py) | `train`, `evaluate`, and `inspect` command boundary |
| [`data.py`](data.py) | Immutable episodes, split/lineage checks, manifest hashes, and portable NPZ loading |
| [`covariance_calibration.py`](covariance_calibration.py) | SPD parameterization, losses, regularization, configuration, and sequential reference trainer |
| [`batched_calibration.py`](batched_calibration.py) | All-rollout static batching, synchronized chunks, `torch.compile`, and CUDA Graph execution |
| [`fixed_slot_inekf.py`](fixed_slot_inekf.py) | Static-shape 8-slot differentiable InEKF used by the fast path |
| [`invariant_ekf.py`](invariant_ekf.py) | Dynamic-dimension differentiable reference filter |
| [`data_paths.py`](data_paths.py) | Optional external-data paths used by profiling and release coverage |
| [`__init__.py`](__init__.py) | Six-name public API |

## Key technical choices

- The fixed state uses eight materialized contact slots, `X ∈ R^(13×13)` and
  `P ∈ R^(39×39)`. Inactive slots are masked and overwritten on insertion, so
  they cannot leak into active covariance blocks.
- Batch padding uses `dt = 0` no-op rows. One synchronized chunk therefore has
  the same static shape for every episode and can be compiled or captured.
- `SPD3` constructs each covariance as `L Lᵀ`: softplus keeps the Cholesky
  diagonal positive, explicit floors prevent collapse, and off-diagonal scale
  controls conditioning.
- CUDA Graph capture owns forward and backward for a whole chunk. The
  non-capturable SPD regularizer runs eagerly and its gradient is added by
  linearity before the optimizer update.
- Dataset identity, execution mode, RNG state, and every output hash are part
  of the run contract. Resume is allowed only at epoch boundaries with an
  unchanged identity.

## Dataset contract

`load_dataset(PATH)` expects `dataset_manifest.json` and one NPZ per episode.
The manifest records `name`, `split`, `source_id`, `file`, and `sha256`.

Each NPZ contains:

| Key | Shape |
|---|---|
| `time_s` | `[T]` |
| `imu` | `[T, 6]` |
| `p_BC` | `[T, N, 3]` |
| `contact_flags` | `[T, N]` |
| `gt_R_WB` | `[T, 3, 3]` |
| `gt_v_W`, `gt_p_W` | `[T, 3]` |
| `row_valid` | Optional `[T]` contiguous validity mask |

`N` is between one and eight. Timestamps are finite, strictly increasing, and
near-uniform. Recording lineage is checked across train, validation, and test
before replay begins.

Return to the [CUDA implementation](../../README.md).
