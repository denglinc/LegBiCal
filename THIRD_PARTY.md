# Third-party material

The repository is distributed under the root MIT license except where a file
or subtree carries its own notice.

- `prime/vendor/PRIME/` is a compact derivative of
  [well-robotics/PRIME](https://github.com/well-robotics/PRIME), pinned from
  commit `b848ceecd451f4786ce39dcefa59e96dbaa369ba`. It retains the BSD
  3-Clause notice in `prime/vendor/PRIME/LICENSE` and includes Crocoddyl code.
- CasADi, Fatrop, Pinocchio, Eigen, Boost, NumPy, SciPy, CVXPY, and Torch are
  external dependencies and are not relicensed here.
- `matlab/data/stride_demo.mat` contains project experiment signals and
  precomputed kinematic quantities. STRIDE controller and simulator sources
  are not included. The corresponding STRIDE source revision was
  `37d32610ec758a2da2734c662201022fe4a3231d`.
- `python/bilevel/resources/data/b1.npz` contains the project B1 hardware
  trajectory used by the Python example. `python/bilevel/resources/robot/B1.urdf`
  is the robot description required to interpret that trajectory.
- `matlab/assets/` contains figures generated for the associated paper.
