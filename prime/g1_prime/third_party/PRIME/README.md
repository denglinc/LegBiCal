# PRIME: Physically-consistent Robotic Inertial and Motion Estimation for Legged and Humanoid Robots

Robotics: Science and Systems (RSS) 2026

PRIME, **Physically-consistent Robotic Inertial and Motion Estimation**, is a
research codebase for reconstructing dynamically consistent motion, contact
forces, and inertial parameters for legged and humanoid robots.

The method formulates contact-rich motion reconstruction as a parameter
full-information estimation problem. Given measured kinematics and actuator
commands, PRIME refines the robot trajectory while jointly estimating
frictional contact interactions and physically consistent inertial parameters.
The implementation uses differentiable Anitescu-style contact dynamics with
smoothed complementarity and solves the resulting problem with
Crocoddyl/FDDP.

<p align="center">
  <a href="https://arxiv.org/abs/2605.17681"><img src="https://img.shields.io/badge/arXiv-2605.17681-b31b1b.svg" alt="arXiv"></a>
  <a href="https://jkangkjr.github.io/PRIME-project"><img src="https://img.shields.io/badge/Project-Page-2b6cb0.svg" alt="Project Page"></a>
  <a href="https://www.youtube.com/embed/5pG6b9hvz-Y?autoplay=1&mute=1&loop=1&playlist=5pG6b9hvz-Y&playsinline=1&rel=0/"><img src="https://img.shields.io/badge/Video-Demo-ff6f00.svg" alt="Video"></a>
</p>

<p align="center">
  <img src="media/FirstPlot_new.png" alt="PRIME pipeline from real robot motion and sensing to physics-consistent motion, inertia, and contact estimation" width="78%">
</p>

PRIME is built on [Crocoddyl](https://github.com/loco-3d/crocoddyl) as the
optimization backend and preserves Crocoddyl's BSD-3-Clause license and
attribution.

## Citing PRIME

If you use this code in academic work, please cite both PRIME and the upstream
Crocoddyl paper. See `CITATION.cff` for citation metadata.

```bibtex
@misc{kang2026PRIME,
      title={PRIME: Physically-consistent Robotic Inertial and Motion Estimation for Legged and Humanoid Robots},
      author={Jiarong Kang and Kunzhao Ren and Tao Pang and Xiaobin Xiong},
      year={2026},
      eprint={2605.17681},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.17681},
}
```
```bibtex
@inproceedings{mastalli20crocoddyl,
  author={Mastalli, Carlos and Budhiraja, Rohan and Merkt, Wolfgang and Saurel, Guilhem and Hammoud, Bilal
  and Naveau, Maximilien and Carpentier, Justin and Righetti, Ludovic and Vijayakumar, Sethu and Mansard, Nicolas},
  title={{Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control}},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year={2020}
}
```

## PRIME Offers

- Contact-implicit optimization based on analytic
  smoothed Anitescu-style frictional contact dynamics.
- Joint trajectory estimation, contact-force reconstruction, and inertial
  parameter identification from kinematics and actuator sensing.
- Contact annotation tools for real-robot locomotion logs without relying on contact-related sensors.
- Self-contained Unitree G1 and Go2 experiments for both real and simulated
  robot data, with XML configs, robot descriptions, results, and visualizers.

## Example Results

PRIME reconstructs physics-consistent motion, contact, and inertial estimates
across humanoid and quadruped experiments from kinematics and actuator sensing.

<p align="center">
  <a href="media/PRIME_1_cropped.mp4">
    <img src="media/PRIME_1_cropped.gif" alt="Animated PRIME result preview for Go2 motion reconstruction" width="95%">
  </a>
</p>

<p align="center">
  <a href="media/PRIME_2_cropped_trimmed.mp4">
    <img src="media/PRIME_2_cropped_trimmed.gif" alt="Animated PRIME result preview for G1 motion reconstruction" width="95%">
  </a>
</p>

<p align="center">
  <a href="media/PRIME_3_cropped_trimmed.mp4">
    <img src="media/PRIME_3_cropped_trimmed.gif" alt="Animated PRIME result preview for additional motion reconstruction examples" width="95%">
  </a>
</p>

<p align="center">
  <img src="media/Go2_compare.png" alt="Go2 real and optimized motion comparison with contact identification" width="95%">
</p>

<p align="center">
  <img src="media/G1_real.png" alt="G1 real-world motion reconstruction and contact identification result" width="95%">
</p>

## Repository Layout

```text
PRIME/
├── include/
│   └── crocoddyl/
│       ├── contact_id/
│       │   ├── actions/          # Contact-ID action models
│       │   ├── actuations/       # Contact-ID actuation models
│       │   ├── anitescu/         # Differentiable Anitescu contact utilities
│       │   ├── config/           # XML config structures and parser helpers
│       │   ├── problem/          # Contact-ID problem builder
│       │   └── states/           # Parameter-augmented multibody state
│       └── multibody/            # Crocoddyl compatibility and upstream code
│
├── src/
│   ├── contact_id/
│   │   └── problem/              # Contact-ID implementation sources
│   └── multibody/                # Crocoddyl compatibility and upstream code
│
├── experiments/
│   ├── common/                   # Shared XML runner utilities
│   │   ├── contact_id_model.hpp
│   │   ├── contact_id_outputs.hpp
│   │   └── contact_id_preprocess.hpp
│   │
experiment
│   ├── G1_real_dance_1/          # Unitree G1 real dance sequence
│   ├── G1_real_dance_2/          # Unitree G1 real dance sequence
│   ├── Go2_real_belly_plate_4.6kg/
│   ├── Go2_sim_cz_-0.1m/
│   └── Go2_sim_m_+3kg/
│
│       Each experiment folder follows:
│       ├── CMakeLists.txt
│       ├── config/               # XML configs
│       ├── data/                 # Input q/v/u CSV measurements
│       ├── descriptions/         # URDF, SRDF, and meshes
│       │   ├── urdf/
│       │   ├── srdf/
│       │   ├── mjcf/
│       │   └── meshes/
│       ├── results/              # Generated motion estimates, parameters , force estimates
│       ├── src/                  # Experiment executable source
│       └── visualizer/           # Meshcat visualizer
│
├── benchmark/                    # Benchmark inherited from Crocoddyl
├── examples/                     # Examples inherited from Crocoddyl
├── unittest/                     # Unit tests inherited from Crocoddyl
├── cmake/                        # CMake helper modules
├── CMakeLists.txt
├── package.xml
├── README.md
├── CITATION.cff
├── AUTHORS.md
├── NOTICE.md
├── THIRD_PARTY_NOTICES.md
└── LICENSE
```

## Dependencies

The C++ build follows the [Crocoddyl](https://github.com/loco-3d/crocoddyl)
dependency stack. This repository has been tested with the following versions:

| Dependency | Tested version |
| --- | --- |
| CMake | `3.22.1` |
| C++ compiler | GCC/G++ `11.4.0` |
| Eigen | `3.4.0` |
| Boost | `1.74.0` |
| Pinocchio | `3.4.0` |
| hpp-fcl | `3.0.0` |
| Ipopt | `3.11.9` |
| example-robot-data | `4.2.0` |
| eigenpy | `3.10.3` |

For the experiment executables in this repository, the recommended build keeps
the Python interface and upstream examples disabled.

## Build

A minimal C++ build is:

```bash
cmake -S . -B build -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_EXAMPLES=OFF
cmake --build build --target g1_real_dance_1 -j2
```

Experiments are added by `experiments/CMakeLists.txt` when an experiment folder
contains its own `CMakeLists.txt`. Current experiment targets can be built by
name:

```bash
cmake --build build --target g1_real_dance_1 -j2
cmake --build build --target g1_real_dance_2 -j2
cmake --build build --target go2_sim_cz_m0p1 -j2
cmake --build build --target go2_sim_m_p3kg -j2
cmake --build build --target go2_real_belly_plate_4p6kg -j2
```

The project depends on the Crocoddyl stack, including Pinocchio, Eigen, Boost,
and solver dependencies available in your CMake configuration.

## Run Experiments

Each experiment executable takes one XML config:

```bash
<experiment_executable> <experiment_config.xml>
```

Go2 simulation with +3 kg inertial edit:

```bash
cmake -S . -B build -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_EXAMPLES=OFF
cmake --build build --target go2_sim_m_p3kg -j2
build/experiments/Go2_sim_m_+3kg/go2_sim_m_p3kg \
  experiments/Go2_sim_m_+3kg/config/Go2_sim_m_+3kg.xml
```

G1 real dance sequence:

```bash
cmake -S . -B build -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_EXAMPLES=OFF
cmake --build build --target g1_real_dance_1 -j2
build/experiments/G1_real_dance_1/g1_real_dance_1 \
  experiments/G1_real_dance_1/config/g1_real_dance_1.xml
```

The same pattern is used for future robot modules:

```text
experiments/<ExperimentName>/CMakeLists.txt
experiments/<ExperimentName>/src/<target>.cpp
experiments/<ExperimentName>/config/<config>.xml
build/experiments/<ExperimentName>/<target> <config>.xml
```

For a parser/configuration check without running the solver, set:

```xml
dry_run="true"
```

in the `<solver>` block. A successful dry run prints the number of configured
contact frames and identified links.

Generated outputs are written to the `<outputs directory="...">` configured in
XML. Typical outputs include:

```text
xs_log.csv            preprocessed initial/reference states
us_log.csv            preprocessed controls
xs_results_fddp.csv   optimized state trajectory
us_results_fddp.csv   optimized controls
u0_results_fddp.csv   identified parameter update at the first node
inertia_identification.txt
                       readable inertial-parameter identification report
xs_rollout.csv        dynamics rollout from optimized controls
f_rollout.csv         reconstructed contact-force log
```

## XML Interface

Each robot experiment is configured through XML:

```text
<robot>          URDF, optional SRDF, reference configuration
<contacts>       ordered contact frame names
<identification> link or joint names/indices and optional parameter offsets
<data>           CSV paths for q, v, u and preprocessing flags
<solver>         timestep, horizon, down-sampling, iterations, callbacks
<weights>        arrival, measurement, dynamics, contact, and friction weights
<outputs>        output directory and filenames
```

The example weights are rough manual tunings chosen to demonstrate the method
on the provided logs; they are not claimed to be optimal. Automatic or
calibrated gain selection is a useful next step. One related direction is our
recent ICRA 2026 work,
[denglinc/LegBiCal](https://github.com/denglinc/LegBiCal)
([arXiv:2510.11539](https://arxiv.org/abs/2510.11539)), although PRIME's
contact-implicit dynamics can introduce additional coupling beyond the
calibration cases studied there.

For particularly noisy hardware logs, PRIME can also be run as a staged
continuation problem. A first solve uses a smaller `kappa`, which gives a
smoother contact model and usually an easier convergence basin; later stages
warm-start from the previous solution while increasing `kappa` toward the
target value. This keeps the final solution closer to the desired rigid-contact
behavior while avoiding the hardest initialization directly from raw sensing.
See `experiments/Go2_real_belly_plate_4.6kg/config/go2_real_belly_plate_4.6kg.xml`
for a staged real-robot example.


## Adding Experiments

Robot experiments should be self-contained:

```text
experiments/<Robot>/
  config/
  data/
  descriptions/
  results/
  src/
  visualizer/
```

The shared helpers in `experiments/common/` handle XML validation, model
loading, CSV preprocessing, torso-to-base conversion, terrain projection,
initial guess construction, and output logging. Robot-specific runners should
stay thin: parse XML, prepare data, build the contact-ID problem, solve, and
save results. For normal experiment use, the source code should not need to be
edited; add or modify XML configs, robot descriptions, data logs, and
visualizers inside the experiment folder instead.

## Relationship To Crocoddyl

This project is not an official Crocoddyl release. It builds on Crocoddyl's
optimal-control and FDDP infrastructure, then adds PRIME/differentiable contact dynamics,
state augmentation, identification utilities, XML experiment runners, and robot
experiments.

Upstream Crocoddyl resources:

- Repository: https://github.com/loco-3d/crocoddyl
- Documentation: https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/master/doxygen-html/
- Publications: https://github.com/loco-3d/crocoddyl/blob/master/PUBLICATIONS.md

## License And Attribution

This repository is distributed under the BSD-3-Clause license while preserving
the license and attribution requirements of Crocoddyl and other third-party
dependencies. See:

- `LICENSE`
- `NOTICE.md`
- `AUTHORS.md`
- `THIRD_PARTY_NOTICES.md`
- `CITATION.cff`
