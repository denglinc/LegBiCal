# Third-party software

The fast FIE requires CasADi and Fatrop; these binaries are not duplicated in
this directory. Their license notices are distributed with the corresponding
CasADi package.

The contact-aware FIE includes a compact source tree derived from PRIME and
Crocoddyl. Its BSD 3-Clause license is retained at
`prime/vendor/PRIME/LICENSE`. Pinocchio, Eigen, Boost, NumPy, and SciPy are
installed through the environment specification and retain their respective
licenses.

The compact PRIME tree is derived from
`https://github.com/well-robotics/PRIME` at commit
`b848ceecd451f4786ce39dcefa59e96dbaa369ba`.

`../matlab/data/stride_demo.mat` contains the numerical signals and precomputed
kinematic quantities needed by the example. STRIDE controller and simulator
sources are not included.
