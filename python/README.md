# Python

Three-dimensional bilevel calibration for the B1 platform. A stage-structured
Fatrop FIE forms the lower level; a sparse adjoint supplies the first-order
gradient; and a semidefinite Frank--Wolfe oracle updates covariance and
kinematic parameters.

The fixed-window solver, primal/dual warm start, derivative functions, KKT
factorization, and convex oracle are reused. Data, URDF, and portable generated
C sources are packaged under `bilevel/resources`; native kinematic functions
are compiled once into the user cache.

## Run

```bash
conda create -n legbical -c conda-forge python=3.12 pinocchio casadi
conda activate legbical
python -m pip install -e '.[dev]'
estimation-calibration --horizon 3000 --iterations 75
```

CasADi must include the Fatrop plugin. CLARABEL is the default open-source LMO
solver; another CVXPY solver can be selected with `--lmo-solver`.

## Test

```bash
pytest python/tests
```
