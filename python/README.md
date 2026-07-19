# Python implementation

A hardware-oriented B1 implementation using a stage-structured Fatrop FIE, a
sparse adjoint, and a semidefinite Frank--Wolfe oracle for covariance and
kinematic calibration.

## Contents

| Path | Responsibility |
|---|---|
| [`bilevel/`](bilevel/README.md) | Data, robot kinematics, estimator, sensitivity, loss, oracle, and calibration loop |
| [`tools/`](tools/) | Maintainer utility for regenerating portable kinematic C code |

The repository-root [`pyproject.toml`](../pyproject.toml) owns this package and
the `estimation-calibration` command. Data, URDF, and portable generated C
sources live under `bilevel/resources/`; native functions compile once into
the user cache.

## Run

From the repository root:

```bash
conda create -n legbical -c conda-forge python=3.12 pinocchio casadi
conda activate legbical
python -m pip install -e .
estimation-calibration --horizon 3000 --iterations 75
```

CasADi must provide the Fatrop plugin. CLARABEL is the default open-source
linear minimization oracle; another installed CVXPY solver can be selected
with `--lmo-solver`.

Return to the [repository overview](../README.md).
