"""Typed boundary around the native contact-aware full-information estimator."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class PrimeSolution:
    theta: np.ndarray
    state: np.ndarray
    truth: np.ndarray
    measurement: np.ndarray
    process: np.ndarray
    dynamics: np.ndarray
    dynamics_shin: np.ndarray
    dynamics_A_shin: np.ndarray
    dynamics_H_correction: np.ndarray


class PrimeEstimator:
    """Run PRIME contact dynamics and FDDP for one parameter vector."""

    _FROST_TO_PIN = np.array([0, 1, 2, 5, 6, 3, 4])

    def __init__(
        self,
        root: Path,
        data_file: Path,
        knots: int = 80,
        max_iterations: int = 220,
        kappa: float = 200.0,
    ) -> None:
        self.root = Path(root).resolve()
        self.binary = self.root / ".build/prime_fie"
        self.model = self.root / "model/stride_frost_planar.urdf"
        self.knots = knots
        self.max_iterations = max_iterations
        self.kappa = kappa
        self._dataset = self.root / ".build" / "stride_demo.csv"
        data_file = Path(data_file)
        if (
            not self._dataset.exists()
            or self._dataset.stat().st_mtime_ns < data_file.stat().st_mtime_ns
        ):
            self._write_dataset(data_file)

    def solve(self, theta: np.ndarray) -> PrimeSolution:
        theta = np.asarray(theta, dtype=float).reshape(5)
        scales = np.exp(theta)
        command = [
            str(self.binary), str(self.model), str(self._dataset),
            "-", "0", str(self.knots),
            str(self.max_iterations), f"{self.kappa:.17g}",
            *[f"{value:.17g}" for value in scales],
        ]
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        table = np.genfromtxt(StringIO(completed.stdout), delimiter=",", names=True)
        state = self._state(table, "est")
        truth = self._state(table, "gt")
        measurement = self._state(table, "meas")
        process = np.column_stack([table[f"process{i}"] for i in range(14)])[:-1]
        dynamics = self._matrix_history(table, "dynamics_A")
        dynamics_A_shin = self._matrix_history(table, "dynamics_A_shin")
        dynamics_H = self._matrix_history(table, "dynamics_H_correction")
        dynamics_shin = np.column_stack(
            [table[f"dynamics_shin{i}"] for i in range(14)]
        )[:-1]
        return PrimeSolution(
            theta=theta, state=state, truth=truth, measurement=measurement,
            process=process, dynamics=dynamics,
            dynamics_shin=dynamics_shin,
            dynamics_A_shin=dynamics_A_shin,
            dynamics_H_correction=dynamics_H,
        )

    def _write_dataset(self, data_file: Path) -> None:
        value = loadmat(data_file, squeeze_me=True, struct_as_record=False)["prime"]
        arrays = [
            np.asarray(value.t), np.asarray(value.qGroundTruth),
            np.asarray(value.vGroundTruth), np.asarray(value.torqueGroundTruth),
            np.asarray(value.qMeasurement), np.asarray(value.vMeasurement),
            np.asarray(value.torqueMeasurement),
        ]
        names = ["t"]
        for prefix, count in (("q_gt", 7), ("v_gt", 7), ("u_gt", 4),
                              ("q_meas", 7), ("v_meas", 7), ("u_meas", 4)):
            names.extend(f"{prefix}{i}" for i in range(count))
        self._dataset.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._dataset.with_suffix(".tmp")
        np.savetxt(
            temporary, np.column_stack(arrays), delimiter=",",
            header=",".join(names), comments="", fmt="%.17g",
        )
        os.replace(temporary, self._dataset)

    @classmethod
    def _state(cls, table: np.ndarray, kind: str) -> np.ndarray:
        q = np.column_stack([table[f"q_{kind}{i}"] for i in range(7)])
        v = np.column_stack([table[f"v_{kind}{i}"] for i in range(7)])
        return np.column_stack((q[:, cls._FROST_TO_PIN], v[:, cls._FROST_TO_PIN]))

    def _matrix_history(self, table: np.ndarray, prefix: str) -> np.ndarray:
        flat = np.column_stack([table[f"{prefix}{i}"] for i in range(196)])
        return flat[:-1].reshape(self.knots - 1, 14, 14)
