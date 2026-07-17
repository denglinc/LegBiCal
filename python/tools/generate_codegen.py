"""Generate and sanity-check B1 CasADi foot-kinematics code.

The runtime package loads the compiled libraries generated from these CasADi
functions. This script keeps the source-generation path and finite-difference
checks in one place so the checked-in C sources can be reproduced.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import casadi as cs
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin


RESOURCE_DIR = Path(__file__).resolve().parents[1] / "bilevel" / "resources"
CODEGEN_DIR = RESOURCE_DIR / "codegen"
URDF_PATH = str(RESOURCE_DIR / "robot" / "B1.urdf")

DEFAULT_FOOT_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
FOOT_NAMES = list(DEFAULT_FOOT_NAMES)

POSITION_FUNCTION_NAME = "pf_and_J"
VELOCITY_FUNCTION_NAME = "yv_and_J"
POSITION_C_FILENAME = "pf_and_J_codegen.c"
VELOCITY_C_FILENAME = "yv_and_J_codegen.c"


def skew_cs(v):
    """CasADi SX skew-symmetric matrix."""
    v0, v1, v2 = v[0], v[1], v[2]
    z = cs.SX(0)
    return cs.vertcat(
        cs.hcat([z, -v2, v1]),
        cs.hcat([v2, z, -v0]),
        cs.hcat([-v1, v0, z]),
    )


def unvec_colmajor(vec: np.ndarray, rows: int, cols: int) -> np.ndarray:
    return np.reshape(vec, (rows, cols), order="F")


def unflatten_jac_colmajor(jac_flat: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Convert d vec(M) / d theta into tensor[:, :, theta_index]."""
    jac_flat = np.asarray(jac_flat, dtype=float)
    return np.stack(
        [unvec_colmajor(jac_flat[:, k], rows, cols) for k in range(jac_flat.shape[1])],
        axis=2,
    )


@dataclass(frozen=True)
class B1CodegenConfig:
    """File paths and generation options for B1 codegen."""

    urdf_path: Path = Path(URDF_PATH)
    foot_names: Sequence[str] = DEFAULT_FOOT_NAMES
    output_dir: Path | None = CODEGEN_DIR
    shared_offset: bool = False
    base_ang_slice: tuple[int, int] = (3, 6)

    def __post_init__(self) -> None:
        object.__setattr__(self, "urdf_path", Path(self.urdf_path).expanduser().resolve())
        object.__setattr__(self, "foot_names", tuple(self.foot_names))
        if self.output_dir is not None:
            object.__setattr__(
                self,
                "output_dir",
                Path(self.output_dir).expanduser().resolve(),
            )


@dataclass(frozen=True)
class B1ModelContext:
    """Pinocchio model, data, and resolved foot frame IDs."""

    model: pin.Model
    data: object
    foot_frame_ids: tuple[int, ...]
    foot_names: tuple[str, ...]

    @classmethod
    def from_config(cls, config: B1CodegenConfig) -> "B1ModelContext":
        model = pin.buildModelFromUrdf(str(config.urdf_path), pin.JointModelFreeFlyer())
        data = model.createData()
        foot_frame_ids = tuple(cls._resolve_frame_id(model, name) for name in config.foot_names)
        return cls(
            model=model,
            data=data,
            foot_frame_ids=foot_frame_ids,
            foot_names=tuple(config.foot_names),
        )

    @staticmethod
    def _resolve_frame_id(model: pin.Model, frame_name: str) -> int:
        frame_id = model.getFrameId(frame_name)
        if frame_id == len(model.frames):
            raise RuntimeError(f"Frame '{frame_name}' not found in model.")
        return int(frame_id)


class B1Kinematics:
    """Numeric B1 foot measurement utilities used by sanity checks."""

    @staticmethod
    def compute_foot_positions(
        model: pin.Model,
        data: object,
        q_zero: np.ndarray,
        foot_frame_ids: Sequence[int],
        offset_calf: np.ndarray,
    ) -> np.ndarray:
        pin.forwardKinematics(model, data, q_zero)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q_zero)

        foot_positions = np.zeros(3 * len(foot_frame_ids), dtype=float)
        for leg_index, frame_id in enumerate(foot_frame_ids):
            row = slice(3 * leg_index, 3 * leg_index + 3)
            parent_joint_id = model.frames[frame_id].parentJoint
            parent_rotation = np.asarray(data.oMi[parent_joint_id].rotation, dtype=float)
            nominal_position = np.asarray(data.oMf[frame_id].translation, dtype=float)
            foot_positions[row] = nominal_position + parent_rotation @ offset_calf[row]
        return foot_positions

    @staticmethod
    def compute_foot_velocities(
        model: pin.Model,
        data: object,
        q_zero: np.ndarray,
        v_zero: np.ndarray,
        omega: np.ndarray,
        foot_frame_ids: Sequence[int],
        offset_calf: np.ndarray,
    ) -> np.ndarray:
        pin.forwardKinematics(model, data, q_zero, v_zero)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q_zero)

        omega = np.asarray(omega, dtype=float).reshape(3)
        foot_velocities = np.zeros(3 * len(foot_frame_ids), dtype=float)
        joint_start = 6
        joint_stop = 6 + model.nv - 6

        for leg_index, frame_id in enumerate(foot_frame_ids):
            row = slice(3 * leg_index, 3 * leg_index + 3)
            frame_jacobian = pin.getFrameJacobian(
                model,
                data,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            linear_jacobian = np.asarray(frame_jacobian[0:3, joint_start:joint_stop])
            nominal_position = np.asarray(data.oMf[frame_id].translation, dtype=float)
            base_velocity = -(
                linear_jacobian @ v_zero[joint_start:joint_stop]
                + np.cross(omega, nominal_position)
            )

            parent_joint_id = model.frames[frame_id].parentJoint
            parent_rotation = np.asarray(data.oMi[parent_joint_id].rotation, dtype=float)
            offset_world = parent_rotation @ offset_calf[row]
            joint_jacobian = np.asarray(
                pin.getJointJacobian(
                    model,
                    data,
                    parent_joint_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
            )
            offset_velocity = (
                -pin.skew(offset_world)
                @ joint_jacobian[3:6, joint_start:joint_stop]
                @ v_zero[joint_start:joint_stop]
                + np.cross(omega, offset_world)
            )
            foot_velocities[row] = base_velocity - offset_velocity
        return foot_velocities


class CasadiKinematicsCodegen:
    """Build CasADi external functions for B1 foot position and velocity."""

    def __init__(
        self,
        model: pin.Model,
        foot_frame_ids: Sequence[int],
        *,
        shared_offset: bool = False,
        output_dir: Path | None = None,
        base_ang_slice: tuple[int, int] = (3, 6),
    ) -> None:
        self.model = model
        self.foot_frame_ids = tuple(int(frame_id) for frame_id in foot_frame_ids)
        self.shared_offset = shared_offset
        self.output_dir = Path(output_dir) if output_dir is not None else None
        # Retained for the original helper signature; omega is now an explicit input.
        self.base_ang_slice = base_ang_slice

    def build_position_function(
        self,
        *,
        generate: bool = True,
        c_filename: str | Path = POSITION_C_FILENAME,
    ):
        """Build pf_and_J(q, theta) and optionally write generated C code."""
        cmodel, cdata = self._casadi_model()
        q_joint = cs.SX.sym("q", cmodel.nq - 7)
        theta = self._theta_symbol(len(self.foot_frame_ids))
        q_zero = self._zero_base_configuration(q_joint)

        cpin.forwardKinematics(cmodel, cdata, q_zero)
        cpin.updateFramePlacements(cmodel, cdata)

        y_blocks = []
        for leg_index, frame_id in enumerate(self.foot_frame_ids):
            parent_joint_id = cmodel.frames[frame_id].parentJoint
            parent_rotation = cdata.oMi[parent_joint_id].rotation
            nominal_position = cdata.oMf[frame_id].translation
            theta_leg = self._theta_for_leg(theta, leg_index)
            y_blocks.append(nominal_position + parent_rotation @ theta_leg)

        y = cs.vertcat(*y_blocks)
        jy = cs.jacobian(y, q_joint)
        dy_dtheta = cs.jacobian(y, theta)
        d_jy_dtheta = cs.jacobian(cs.vec(jy), theta)

        function = cs.Function(
            POSITION_FUNCTION_NAME,
            [q_joint, theta],
            [y, jy, dy_dtheta, d_jy_dtheta],
            ["q", "theta"],
            ["y", "Jy", "DyDtheta", "dJy_dtheta"],
        )

        if generate:
            self._generate_c(function, c_filename)
        return function

    def build_velocity_function(
        self,
        *,
        generate: bool = True,
        c_filename: str | Path = VELOCITY_C_FILENAME,
    ):
        """Build yv_and_J(q, v, omega, theta) and optionally write C code."""
        if len(self.foot_frame_ids) != 4:
            raise ValueError("This generated velocity function assumes four B1 feet.")

        cmodel, cdata = self._casadi_model()
        q_joint = cs.SX.sym("q", cmodel.nq - 7)
        v_joint = cs.SX.sym("v", cmodel.nv - 6)
        omega = cs.SX.sym("omega", 3)
        theta = self._theta_symbol(len(self.foot_frame_ids))

        q_zero = self._zero_base_configuration(q_joint)
        v_zero = cs.vertcat(cs.DM.zeros(6), v_joint)
        joint_start = 6
        joint_stop = 6 + cmodel.nv - 6

        cpin.forwardKinematics(cmodel, cdata, q_zero, v_zero)
        cpin.updateFramePlacements(cmodel, cdata)
        cpin.computeJointJacobians(cmodel, cdata, q_zero)

        velocity_blocks = []
        for leg_index, frame_id in enumerate(self.foot_frame_ids):
            parent_joint_id = cmodel.frames[frame_id].parentJoint
            parent_rotation = cdata.oMi[parent_joint_id].rotation
            frame_jacobian = cpin.getFrameJacobian(
                cmodel,
                cdata,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            joint_jacobian = cpin.getJointJacobian(
                cmodel,
                cdata,
                parent_joint_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            nominal_position = cdata.oMf[frame_id].translation
            theta_leg = self._theta_for_leg(theta, leg_index)
            offset_world = parent_rotation @ theta_leg

            linear_jacobian = frame_jacobian[0:3, joint_start:joint_stop]
            joint_angular_jacobian = joint_jacobian[3:6, joint_start:joint_stop]
            base_velocity = -((linear_jacobian @ v_joint) + cs.cross(omega, nominal_position))
            offset_velocity = (
                -skew_cs(offset_world) @ joint_angular_jacobian @ v_joint
                + cs.cross(omega, offset_world)
            )
            velocity_blocks.append(base_velocity - offset_velocity)

        foot_velocity = cs.vertcat(*velocity_blocks)
        jy_q = cs.jacobian(foot_velocity, q_joint)
        jy_v = cs.jacobian(foot_velocity, v_joint)
        jy_omega = cs.jacobian(foot_velocity, omega)
        dy_dtheta = cs.jacobian(foot_velocity, theta)

        function = cs.Function(
            VELOCITY_FUNCTION_NAME,
            [q_joint, v_joint, omega, theta],
            [
                foot_velocity,
                jy_q,
                jy_v,
                jy_omega,
                dy_dtheta,
                cs.jacobian(cs.vec(jy_q), theta),
                cs.jacobian(cs.vec(jy_v), theta),
                cs.jacobian(cs.vec(jy_omega), theta),
            ],
            ["q", "v", "omega", "theta"],
            [
                "v_foot",
                "Jy_q",
                "Jy_v",
                "Jy_omega",
                "DyDtheta",
                "dJy_q_dtheta",
                "dJy_v_dtheta",
                "dJy_omega_dtheta",
            ],
        )

        if generate:
            self._generate_c(function, c_filename)
        return function

    def _casadi_model(self):
        try:
            cmodel = cpin.Model(self.model)
        except Exception as exc:
            raise RuntimeError(
                "Could not construct a CasADi Pinocchio model from the Pinocchio model."
            ) from exc
        return cmodel, cmodel.createData()

    def _theta_symbol(self, foot_count: int):
        return cs.SX.sym("theta", 3 if self.shared_offset else 3 * foot_count)

    def _theta_for_leg(self, theta, leg_index: int):
        if self.shared_offset:
            return theta
        return theta[3 * leg_index : 3 * leg_index + 3]

    @staticmethod
    def _zero_base_configuration(q_joint):
        return cs.vertcat(cs.DM.zeros(6), cs.DM(1), q_joint)

    def _generate_c(self, function, c_filename: str | Path) -> Path:
        path = self._resolve_codegen_path(c_filename)
        if path.parent not in (Path("."), Path("")):
            path.parent.mkdir(parents=True, exist_ok=True)
        generator = cs.CodeGenerator(str(path))
        generator.add(function)
        generator.generate()
        print(f"[ok] Wrote {path}")
        return path

    def _resolve_codegen_path(self, filename: str | Path) -> Path:
        path = Path(filename)
        if path.is_absolute() or self.output_dir is None:
            return path
        return self.output_dir / path


@dataclass(frozen=True)
class SanityCheckResult:
    name: str
    frobenius_norm: float
    relative_norm: float
    max_abs: float


class CodegenSanityChecker:
    """Finite-difference checks for the generated symbolic functions."""

    def __init__(
        self,
        context: B1ModelContext,
        builder: CasadiKinematicsCodegen,
        *,
        eps_q: float = 1e-7,
        eps_theta: float = 1e-7,
    ) -> None:
        self.context = context
        self.builder = builder
        self.eps_q = eps_q
        self.eps_theta = eps_theta

    def run(self, f_pf=None, f_yv=None) -> list[SanityCheckResult]:
        f_pf = f_pf or self.builder.build_position_function(generate=False)
        f_yv = f_yv or self.builder.build_velocity_function(generate=False)

        q0, v_joint, omega, theta = self._sample_inputs()
        out_pf = f_pf(q=q0[7:], theta=theta)
        out_yv = f_yv(q=q0[7:], v=v_joint, omega=omega, theta=theta)

        y = self._vector(out_pf["y"])
        jy = self._dense(out_pf["Jy"])
        dy_dtheta = self._dense(out_pf["DyDtheta"])
        d_jy_dtheta = self._dense(out_pf["dJy_dtheta"])

        v_foot = self._vector(out_yv["v_foot"])
        jy_q = self._dense(out_yv["Jy_q"])
        jy_v = self._dense(out_yv["Jy_v"])
        jy_omega = self._dense(out_yv["Jy_omega"])
        d_jy_q_dtheta = self._dense(out_yv["dJy_q_dtheta"])
        d_jy_v_dtheta = self._dense(out_yv["dJy_v_dtheta"])
        d_jy_omega_dtheta = self._dense(out_yv["dJy_omega_dtheta"])

        print("\n=== Base quantities ===")
        print("y shape:", y.shape)
        print("Jy shape:", jy.shape)
        print("DyDtheta shape:", dy_dtheta.shape)
        print("v_foot shape:", v_foot.shape)
        print("Jy_q shape:", jy_q.shape)
        print("Jy_v shape:", jy_v.shape)
        print("Jy_omega shape:", jy_omega.shape)

        rows = y.size
        cols_q = jy_q.shape[1]
        cols_v = jy_v.shape[1]
        cols_omega = jy_omega.shape[1]

        d_jy_tensor = unflatten_jac_colmajor(d_jy_dtheta, rows, jy.shape[1])
        d_jy_q_tensor = unflatten_jac_colmajor(d_jy_q_dtheta, rows, cols_q)
        d_jy_v_tensor = unflatten_jac_colmajor(d_jy_v_dtheta, rows, cols_v)
        d_jy_omega_tensor = unflatten_jac_colmajor(d_jy_omega_dtheta, rows, cols_omega)

        print("\n=== Flattened Jacobians ===")
        print("dJy_dtheta shape:", d_jy_dtheta.shape)
        print("dJy_q_dtheta shape:", d_jy_q_dtheta.shape)
        print("dJy_v_dtheta shape:", d_jy_v_dtheta.shape)
        print("dJy_omega_dtheta shape:", d_jy_omega_dtheta.shape)

        print("\n=== Unflattened tensors ===")
        print("dJy_tensor shape:", d_jy_tensor.shape)
        print("dJy_q_tensor shape:", d_jy_q_tensor.shape)
        print("dJy_v_tensor shape:", d_jy_v_tensor.shape)
        print("dJy_omega_tensor shape:", d_jy_omega_tensor.shape)
        print("dJy_q_tensor[0:3, 0:3, 0]:\n", d_jy_q_tensor[0:3, 0:3, 0])
        print("dJy_v_tensor[0:3, 0:3, 0]:\n", d_jy_v_tensor[0:3, 0:3, 0])
        print("dJy_omega_tensor[0:3, 0:3, 0]:\n", d_jy_omega_tensor[0:3, 0:3, 0])

        results = [
            self._check_numeric_position_value(q0, theta, y),
            self._check_numeric_velocity_value(q0, v_joint, omega, theta, v_foot),
            self._check_position_jy_fd(f_pf, q0, theta, y, jy),
            self._check_position_theta_fd(f_pf, q0, theta, y, dy_dtheta),
            self._check_position_djy_theta_fd(f_pf, q0, theta, jy, d_jy_tensor),
            self._check_velocity_djy_q_theta_fd(
                f_yv,
                q0,
                v_joint,
                omega,
                theta,
                jy_q,
                d_jy_q_tensor,
            ),
        ]

        print("\n=== Sanity summary ===")
        for result in results:
            print(
                f"{result.name}: "
                f"fro={result.frobenius_norm:.3e}, "
                f"rel={result.relative_norm:.3e}, "
                f"max={result.max_abs:.3e}"
            )
        return results

    def _sample_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q0 = pin.neutral(self.context.model)
        v_joint = np.zeros(self.context.model.nv - 6, dtype=float)
        v_joint[:3] = [0.2, -0.1, 0.05]
        omega = np.array([0.1, -0.2, 0.3], dtype=float)
        if self.builder.shared_offset:
            theta = np.array([0.05, 0.02, -0.03], dtype=float)
        else:
            theta = np.tile(np.array([0.05, 0.02, -0.03], dtype=float), len(self.context.foot_frame_ids))
        return q0, v_joint, omega, theta

    def _check_numeric_position_value(
        self,
        q0: np.ndarray,
        theta: np.ndarray,
        y: np.ndarray,
    ) -> SanityCheckResult:
        theta_full = self._expand_shared_theta(theta)
        y_numeric = B1Kinematics.compute_foot_positions(
            self.context.model,
            self.context.data,
            q0.copy(),
            self.context.foot_frame_ids,
            theta_full,
        )
        return self._compare("numeric position value", y_numeric, y)

    def _check_numeric_velocity_value(
        self,
        q0: np.ndarray,
        v_joint: np.ndarray,
        omega: np.ndarray,
        theta: np.ndarray,
        v_foot: np.ndarray,
    ) -> SanityCheckResult:
        v_zero = np.zeros(self.context.model.nv, dtype=float)
        v_zero[6:] = v_joint
        theta_full = self._expand_shared_theta(theta)
        v_numeric = B1Kinematics.compute_foot_velocities(
            self.context.model,
            self.context.data,
            q0.copy(),
            v_zero,
            omega,
            self.context.foot_frame_ids,
            theta_full,
        )
        return self._compare("numeric velocity value", v_numeric, v_foot)

    def _check_position_jy_fd(
        self,
        f_pf,
        q0: np.ndarray,
        theta: np.ndarray,
        y: np.ndarray,
        jy: np.ndarray,
    ) -> SanityCheckResult:
        q_joint = np.array(q0[7:], dtype=float)
        fd_jy = np.zeros_like(jy)
        for col in range(fd_jy.shape[1]):
            q_perturbed = q_joint.copy()
            q_perturbed[col] += self.eps_q
            y_plus = self._vector(f_pf(q=q_perturbed, theta=theta)["y"])
            fd_jy[:, col] = (y_plus - y) / self.eps_q
        return self._compare("position Jy finite difference", fd_jy, jy)

    def _check_position_theta_fd(
        self,
        f_pf,
        q0: np.ndarray,
        theta: np.ndarray,
        y: np.ndarray,
        dy_dtheta: np.ndarray,
    ) -> SanityCheckResult:
        q_joint = np.array(q0[7:], dtype=float)
        fd_dy = np.zeros_like(dy_dtheta)
        for col in range(fd_dy.shape[1]):
            theta_perturbed = theta.copy()
            theta_perturbed[col] += self.eps_theta
            y_plus = self._vector(f_pf(q=q_joint, theta=theta_perturbed)["y"])
            fd_dy[:, col] = (y_plus - y) / self.eps_theta
        return self._compare("position DyDtheta finite difference", fd_dy, dy_dtheta)

    def _check_position_djy_theta_fd(
        self,
        f_pf,
        q0: np.ndarray,
        theta: np.ndarray,
        jy: np.ndarray,
        d_jy_tensor: np.ndarray,
    ) -> SanityCheckResult:
        q_joint = np.array(q0[7:], dtype=float)
        fd_tensor = np.zeros_like(d_jy_tensor)
        for col in range(theta.size):
            theta_perturbed = theta.copy()
            theta_perturbed[col] += self.eps_theta
            jy_plus = self._dense(f_pf(q=q_joint, theta=theta_perturbed)["Jy"])
            fd_tensor[:, :, col] = (jy_plus - jy) / self.eps_theta
        return self._compare("position dJy/dtheta finite difference", fd_tensor, d_jy_tensor)

    def _check_velocity_djy_q_theta_fd(
        self,
        f_yv,
        q0: np.ndarray,
        v_joint: np.ndarray,
        omega: np.ndarray,
        theta: np.ndarray,
        jy_q: np.ndarray,
        d_jy_q_tensor: np.ndarray,
    ) -> SanityCheckResult:
        q_joint = np.array(q0[7:], dtype=float)
        fd_tensor = np.zeros_like(d_jy_q_tensor)
        for col in range(theta.size):
            theta_perturbed = theta.copy()
            theta_perturbed[col] += self.eps_theta
            jy_q_plus = self._dense(
                f_yv(q=q_joint, v=v_joint, omega=omega, theta=theta_perturbed)["Jy_q"]
            )
            fd_tensor[:, :, col] = (jy_q_plus - jy_q) / self.eps_theta
        return self._compare("velocity dJy_q/dtheta finite difference", fd_tensor, d_jy_q_tensor)

    def _expand_shared_theta(self, theta: np.ndarray) -> np.ndarray:
        if not self.builder.shared_offset:
            return theta
        return np.tile(theta, len(self.context.foot_frame_ids))

    @staticmethod
    def _dense(value) -> np.ndarray:
        if hasattr(value, "full"):
            value = value.full()
        return np.asarray(value, dtype=float)

    @classmethod
    def _vector(cls, value) -> np.ndarray:
        return cls._dense(value).reshape(-1)

    @staticmethod
    def _compare(name: str, observed: np.ndarray, expected: np.ndarray) -> SanityCheckResult:
        diff = np.asarray(observed, dtype=float) - np.asarray(expected, dtype=float)
        fro_norm = float(np.linalg.norm(diff))
        rel_norm = fro_norm / max(1.0, float(np.linalg.norm(expected)))
        max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
        return SanityCheckResult(name, fro_norm, rel_norm, max_abs)


def compute_pf_meas(model, data, q_zero, fids, offset_calf):
    """Backward-compatible wrapper for the original script function."""
    return B1Kinematics.compute_foot_positions(model, data, q_zero, fids, offset_calf)


def compute_yv_kin(model, data, q_zero, v_zero, omega, fids, offset_calf):
    """Backward-compatible wrapper for the original script function."""
    return B1Kinematics.compute_foot_velocities(
        model,
        data,
        q_zero,
        v_zero,
        omega,
        fids,
        offset_calf,
    )


def build_pf_and_J_codegen(model: pin.Model, fids: list[int], shared_offset: bool = False):
    """Backward-compatible wrapper that writes pf_and_J_codegen.c in cwd."""
    return CasadiKinematicsCodegen(
        model,
        fids,
        shared_offset=shared_offset,
        output_dir=None,
    ).build_position_function(generate=True, c_filename=POSITION_C_FILENAME)


def build_yv_and_J_codegen(
    model: pin.Model,
    fids,
    shared_offset: bool = False,
    base_ang_slice: tuple[int, int] = (3, 6),
    c_filename: str = VELOCITY_C_FILENAME,
):
    """Backward-compatible wrapper that writes yv_and_J_codegen.c in cwd."""
    return CasadiKinematicsCodegen(
        model,
        fids,
        shared_offset=shared_offset,
        output_dir=None,
        base_ang_slice=base_ang_slice,
    ).build_velocity_function(generate=True, c_filename=c_filename)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=Path(URDF_PATH))
    parser.add_argument("--output-dir", type=Path, default=CODEGEN_DIR)
    parser.add_argument("--shared-offset", action="store_true")
    parser.add_argument("--generate", action="store_true", help="write generated C sources")
    parser.add_argument("--sanity-check", action="store_true", help="run finite-difference checks")
    parser.add_argument("--position-c", default=POSITION_C_FILENAME)
    parser.add_argument("--velocity-c", default=VELOCITY_C_FILENAME)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = B1CodegenConfig(
        urdf_path=args.urdf,
        output_dir=args.output_dir,
        shared_offset=args.shared_offset,
    )
    context = B1ModelContext.from_config(config)
    builder = CasadiKinematicsCodegen(
        context.model,
        context.foot_frame_ids,
        shared_offset=config.shared_offset,
        output_dir=config.output_dir,
        base_ang_slice=config.base_ang_slice,
    )

    print("[ok] Foot frame IDs:", list(context.foot_frame_ids))

    f_pf = None
    f_yv = None
    if args.generate:
        f_pf = builder.build_position_function(generate=True, c_filename=args.position_c)
        f_yv = builder.build_velocity_function(generate=True, c_filename=args.velocity_c)

    if args.sanity_check:
        checker = CodegenSanityChecker(context, builder)
        checker.run(f_pf=f_pf, f_yv=f_yv)

    if not args.generate and not args.sanity_check:
        print("[info] Use --generate and/or --sanity-check for codegen work.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
