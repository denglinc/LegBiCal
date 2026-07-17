"""Configuration and parameter layout for the estimation calibration pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Sequence


DEFAULT_FOOT_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
DEFAULT_FOOT_Z_OFFSETS = (0.01960054, 0.02402977, 0.04499581, 0.03318461)


@dataclass
class DatasetConfig:
    """Dataset window and preprocessing options."""

    start_idx: int = 22000
    horizon: int = 3000
    dt: float = 0.005
    downsample_factor: int = 1
    contact_threshold: float = 100.0
    foot_z_offsets: Sequence[float] = DEFAULT_FOOT_Z_OFFSETS


@dataclass
class FrankWolfeConfig:
    """Outer-loop optimization and feasible-set settings."""

    max_iterations: int = 75
    armijo_rho: float = 1e-4
    armijo_beta: float = 0.5
    armijo_gamma_init: float = 0.25
    armijo_max_steps: int = 20
    armijo_min_step: float = 1e-8
    gap_tolerance: float = 1e-8

    tip_bound: float = 0.10
    base_bound: float = 0.50

    big_box: float = 1e6
    core_min: float = 1e-9
    core_max: float = 1e6
    qswing_min: float = 1e-3
    qswing_max: float = 1e6
    qstance_min: float = 1e6
    qstance_max: float = 1e8

    eps_psd: float = 1e-9
    eps_diag: float = 1e-9
    trace_cap: float = 1e6
    trust_region_radius: float = 1e5
    adaptive_abs_box_scale: float = 1e3
    lmo_solver: str = "CLARABEL"


@dataclass
class FatropConfig:
    """CasADi/Fatrop options for the lower-level FIE solve."""

    structure_detection: str = "manual"
    expand: bool = False
    print_time: bool = False
    error_on_fail: bool = True
    verbose: bool = False
    tolerance: float = 1e-8
    max_iterations: int = 50
    extra_options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class WeightParameterLayout:
    """Index map for theta = [FIE weights | tip offsets | base offset]."""

    n_state: int
    measurement_len: int = 12
    noise_len: int = 24

    @property
    def core_size(self) -> int:
        return self.n_state + self.measurement_len + self.noise_len

    @property
    def measurement_slice(self) -> slice:
        return slice(self.n_state, self.n_state + self.measurement_len)

    @property
    def noise_slice(self) -> slice:
        start = self.measurement_slice.stop
        return slice(start, start + self.noise_len)

    @property
    def tip_slice(self) -> slice:
        return slice(self.core_size, self.core_size + 12)

    @property
    def base_slice(self) -> slice:
        return slice(self.core_size + 12, self.core_size + 15)

    @property
    def total_size(self) -> int:
        return self.core_size + 15

    @property
    def arrival_slice(self) -> slice:
        return slice(0, self.n_state)

    @property
    def measurement_covariance_slices(self) -> tuple[slice, slice]:
        start = self.measurement_slice.start
        return slice(start, start + 6), slice(start + 6, start + 12)

    @property
    def noise_covariance_slices(self) -> tuple[slice, slice]:
        start = self.noise_slice.start
        return slice(start, start + 6), slice(start + 6, start + 12)

    @property
    def random_walk_slice(self) -> slice:
        start = self.noise_slice.start
        return slice(start + 12, start + 18)

    @property
    def swing_slice(self) -> slice:
        start = self.noise_slice.start
        return slice(start + 18, start + 21)

    @property
    def stance_slice(self) -> slice:
        start = self.noise_slice.start
        return slice(start + 21, start + 24)


@dataclass
class BilevelConfig:
    """Top-level runtime configuration."""

    repo_root: Path | None = None
    resource_dir: Path | None = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    frank_wolfe: FrankWolfeConfig = field(default_factory=FrankWolfeConfig)
    fatrop: FatropConfig = field(default_factory=FatropConfig)
    foot_frame_names: Sequence[str] = DEFAULT_FOOT_NAMES

    data_path: Path = field(init=False)
    urdf_path: Path = field(init=False)
    casadi_cache_dir: Path = field(init=False)
    external_lib_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.repo_root = Path.cwd() if self.repo_root is None else Path(self.repo_root)
        self.repo_root = self.repo_root.resolve()
        resource_dir = (
            Path(str(files("bilevel").joinpath("resources")))
            if self.resource_dir is None
            else Path(self.resource_dir)
        )
        resource_dir = resource_dir.resolve()
        self.resource_dir = resource_dir
        self.data_path = resource_dir / "data" / "b1.npz"
        self.urdf_path = resource_dir / "robot" / "B1.urdf"
        self.casadi_cache_dir = (
            self.repo_root / ".cache" / "casadi" / f"B1_H{self.dataset.horizon}"
        )
        self.external_lib_dir = resource_dir / "codegen"

    @property
    def effective_dt(self) -> float:
        return self.dataset.dt * self.dataset.downsample_factor


def default_weight_vector() -> list[float]:
    """Initial theta core used by the original B1 experiments."""

    w_arrival = [
        50000, 50000, 50000,
        30000, 30000, 30000,
        10000, 10000, 10000,
        80000, 80000, 80000, 80000,
        5000, 5000, 5000,
        20000, 20000, 20000,
        20000, 20000, 20000,
        20000, 20000, 20000,
        20000, 20000, 20000,
    ]

    w_measurement = [
        600, 600, 600,
        0, 0, 0,
        600, 600, 600,
        0, 0, 0,
    ]

    w_noise = [
        70000, 70000, 70000,
        0, 0, 0,
        70000, 70000, 70000,
        0, 0, 0,
        2000, 2000, 2000,
        2000, 2000, 2000,
        400, 400, 400,
        1000000, 1000000, 1000000,
    ]

    return w_arrival + w_measurement + w_noise
