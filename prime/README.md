# PRIME

Contact-aware full-information calibration based on PRIME and Crocoddyl FDDP.
The upper-level gradient is obtained from the block-banded optimality system;
SQP--BFGS, Frank--Wolfe, and projected Adam use the same lower-level estimate.

## Run

```bash
conda env create -f environment.yml
conda activate stride-prime
./build.sh
python run_calibration.py --method sqp
```

The example uses `../matlab/data/stride_demo.mat`. The other upper-level
methods are selected with `--method frank-wolfe` and `--method adam`.
