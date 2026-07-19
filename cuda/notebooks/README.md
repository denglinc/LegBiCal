# CUDA notebooks

Two executable records connect the small differentiable examples to the
package's fixed-slot CUDA path.

| Notebook | Scope |
|---|---|
| [`covariance_tuning_tutorial.ipynb`](covariance_tuning_tutorial.ipynb) | Scalar autograd, SO(3) exp/log, finite-difference gradient check, diagonal Q/R learning, and captured CUDA training |
| [`covariance_calibration_run.ipynb`](covariance_calibration_run.ipynb) | Fixed-slot batch-7, 300-row chunk, forward/backward benchmark with five timing repetitions |

## Recorded outputs

- Tutorial: graph gradients agree with central finite differences at roughly
  `1e-13`; the saved RTX 5090 run reduces AdamW loss from `7.282e-03` to
  `1.182e-04` in `32.37 s`.
- Benchmark: RTX 5090 Laptop, Torch `2.12.0+cu130`, CUDA `13.0`, median
  `0.8967 ms/step`, `7,806` batched rows/s, and `0.1718 GB` peak memory.

Install the notebook extra from `cuda/`:

```bash
python -m pip install -e '.[notebooks]'
```

The benchmark reads real rollouts from `LEG_BICAL_DATA_ROOT` and writes new
measurements under the ignored `runs/notebook_benchmark/` directory. Recorded
numbers describe that hardware/software configuration only.

Return to the [CUDA implementation](../README.md).
