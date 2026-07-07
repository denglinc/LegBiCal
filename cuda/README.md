# Estimation Calibration CUDA

Differentiable covariance calibration for a contact-aided right-invariant EKF.
The estimator replay is a recurrent computation graph; the learned parameters
are float64 SPD covariance blocks, trained through chunked BPTT.
The full notebook run trains 1,469,080 supervised BPTT time steps in 228.4 s.

## Lineage

This implementation follows the computation-graph view of
[Backprop KF](https://dl.acm.org/doi/10.5555/3157382.3157587) and the legged
robot contact-aided InEKF setting of
[Lin et al., CoRL/PMLR 2022](https://proceedings.mlr.press/v164/lin22b.html).
This release does not include a learned contact-event network; contact
schedules come from provided features, and the learned parameters are
covariance blocks.

## Approach

The estimator replay is an unrolled differentiable program: gradients pass
through propagation, correction, covariance updates, and any future learned
block. The same mechanism would support, for example, an observation-conditioned
contact/stick/slip covariance model inside the filter.

The dynamic InEKF stays as the oracle. The training path uses eight fixed
contact slots plus masks, preserving the math while making the hot path
batch-first, static-shape, and capturable.

The first full-training reference was direct autograd/eager dynamic replay,
which took about 82 min for 20 epochs. CUDA graph capture removes CPU launch
overhead by replaying each 300-row forward+backward chunk as one graph; without
it, even fixed-slot eager remained launch-bound at roughly 5.76 ms/step and
769 kernels/row.

The block/gather rewrite uses the block-sparse/symmetric InEKF structure in
propagate, insert, and correct, replacing dense zero-fill, slice-assign,
copy/scatter, and blockdiag assembly. Dense pre-refactor code is kept in tests
as the value/gradient oracle.

The compiled-step path captures `torch.compile(step, fullgraph=True)` inside
the manual whole-chunk CUDA graph, so Inductor/Triton fuses the remaining
elementwise kernels while the outer graph keeps one launch per chunk.

#### Example Results

RTX 5090 Laptop GPU, CUDA float64, B=7, 300 rows/chunk, fwd+bwd, median of 5:

| path | ms/step | rows/s | kernels/row |
|---|---:|---:|---:|
| fixed-slot eager | 5.761 | 1,215 | 768.7 |
| +CUDA graph baseline | 1.428 | 4,901 | 782.8 |
| +block/gather rewrite | 1.333 | 5,250 | 607.1 |
| +compiled step in graph | 0.799 | 8,760 | 185.7 |

| item | value |
|---|---:|
| rollouts | 7 |
| rows/epoch | 73,454 |
| epochs | 20 |
| supervised BPTT time steps | 1,469,080 |
| training wall-clock | 228.4 s |
| effective throughput | 6,433 steps/s |
| selected epoch | 11 |
| aggregate vB RMSE | 1.939752 -> 1.732404 |

See
[`notebooks/covariance_calibration_run.ipynb`](notebooks/covariance_calibration_run.ipynb)
for the run output.

## Commands

Replace the Python executable and dataset root with your own CUDA-enabled
environment and local `datasets_v0` path.

```bash
PYTHONPATH=src /path/to/your/cuda/python \
  -m estimation_calibration_cuda.covariance_calibration train \
  --data-root /path/to/datasets_v0 \
  --outputs runs/covariance_calibration_cuda_graph_compile \
  --epochs 20 \
  --compile cuda-graph-compile
```

```bash
PYTHONPATH=src /path/to/your/cuda/python -m pytest tests/

PYTHONPATH=src /path/to/your/cuda/python \
  benchmarks/profile_replay.py --impl fixed --batch 7 --rows 300 --chunks 10 \
  --with-grad --compile cuda-graph-compile --trace --repeat 5 \
  --data-root /path/to/datasets_v0
```

`notebooks/covariance_tuning_tutorial.ipynb` is the small SO(3) computation
graph analogue; `notebooks/covariance_calibration_run.ipynb` is the full CUDA
calibration run.
