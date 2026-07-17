"""Robot-specific kinematics used by the B1 calibration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin

from .config import BilevelConfig, DEFAULT_FOOT_NAMES
from .kinematics import (
    build_zeroed,
    compute_Gp_leg_blocks_body,
    compute_Gv_leg_blocks_body,
    compute_pf_meas,
    compute_yv_kin,
    dGk_dtip_from_codegen,
    pack_G24x9,
)


@dataclass(frozen=True)
class MeasurementBundle:
    y: np.ndarray
    dy_dtip: np.ndarray


class B1RobotModel:
    """Pinocchio model and B1-specific measurement construction."""

    def __init__(
        self,
        urdf_path: str | Path,
        foot_frame_names: tuple[str, ...] = DEFAULT_FOOT_NAMES,
    ):
        self.urdf_path = Path(urdf_path)
        self.model = pin.buildModelFromUrdf(
            str(self.urdf_path), pin.JointModelFreeFlyer()
        )
        self.data = self.model.createData()
        self.foot_frame_names = tuple(foot_frame_names)
        self.foot_frame_ids = [self._frame_id(name) for name in self.foot_frame_names]

    @classmethod
    def from_config(cls, config: BilevelConfig) -> "B1RobotModel":
        return cls(config.urdf_path, tuple(config.foot_frame_names))

    def _frame_id(self, name: str) -> int:
        frame_id = self.model.getFrameId(name)
        if frame_id == len(self.model.frames):
            raise ValueError(f"foot frame '{name}' is not present in {self.urdf_path}")
        return frame_id

    @staticmethod
    def zero_base(q_i: np.ndarray, v_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return build_zeroed(q_i, v_i)

    def build_measurements(
        self,
        q_meas: np.ndarray,
        v_meas: np.ndarray,
        u_meas: np.ndarray,
        tip_offset: np.ndarray,
    ) -> MeasurementBundle:
        tip_offset = np.asarray(tip_offset, dtype=float).reshape(12)
        pf_list: list[np.ndarray] = []
        velocity_list: list[np.ndarray] = []
        dy_blocks: list[np.ndarray] = []

        for k in range(q_meas.shape[0]):
            q_zero, v_zero = self.zero_base(q_meas[k, :], v_meas[k, :])
            pf_k, j_pf = compute_pf_meas(
                self.model, self.data, q_zero, self.foot_frame_ids, tip_offset
            )
            yv_k, j_v = compute_yv_kin(
                self.model,
                self.data,
                q_zero,
                v_zero,
                u_meas[k, 3:6],
                self.foot_frame_ids,
                tip_offset,
            )
            pf_list.append(pf_k)
            velocity_list.append(yv_k)
            dy_blocks.append(np.vstack([j_v, j_pf]))

        y = np.hstack([np.vstack(velocity_list), np.vstack(pf_list)])
        return MeasurementBundle(y=y, dy_dtip=np.vstack(dy_blocks))

    def build_measurement_jacobians(
        self,
        q_meas: np.ndarray,
        v_meas: np.ndarray,
        u_meas: np.ndarray,
    ) -> np.ndarray:
        g_blocks: list[np.ndarray] = []
        for k in range(q_meas.shape[0]):
            q_zero, v_zero = self.zero_base(q_meas[k, :], v_meas[k, :])
            g_velocity = compute_Gv_leg_blocks_body(
                self.model,
                self.data,
                q_zero,
                v_zero,
                u_meas[k, 3:6],
                self.foot_frame_ids,
            )
            g_position = compute_Gp_leg_blocks_body(
                self.model, self.data, q_zero, self.foot_frame_ids
            )
            g_blocks.append(pack_G24x9(g_velocity, g_position))
        return np.stack(g_blocks, axis=0)

    def dG_dtip(
        self,
        f_yv,
        f_pf,
        q_i: np.ndarray,
        v_i: np.ndarray,
        u_i: np.ndarray,
    ) -> np.ndarray:
        q_zero, v_zero = self.zero_base(q_i, v_i)
        return dGk_dtip_from_codegen(
            f_yv,
            f_pf,
            self.model,
            self.data,
            q_zero,
            v_zero,
            u_i[3:6],
            self.foot_frame_ids,
        )

    def build_measurement_jacobian_derivatives(
        self,
        codegen,
        q_meas: np.ndarray,
        v_meas: np.ndarray,
        u_meas: np.ndarray,
    ) -> np.ndarray:
        """Precompute d vec(G_k) / d tip for the fixed trajectory."""

        blocks = [
            self.dG_dtip(
                codegen.foot_velocity,
                codegen.foot_position,
                q_meas[k, :],
                v_meas[k, :],
                u_meas[k, :],
            )
            for k in range(q_meas.shape[0])
        ]
        return np.stack(blocks, axis=0)

    def initial_state_prior(self, x0: np.ndarray) -> np.ndarray:
        """Build the original FIE prior from the first mocap sample."""

        q0 = np.asarray(x0[3:7], dtype=float)
        q0 = q0 / (np.linalg.norm(q0) + 1e-16)
        rotation = pin.Quaternion(q0).toRotationMatrix()
        velocity_world = rotation @ np.asarray(x0[19:22], dtype=float).reshape(3, 1)

        q_full = np.zeros(self.model.nq)
        q_full[0:3] = x0[0:3]
        q_full[3:7] = q0
        n_joint_q = self.model.nq - 7
        q_full[7 : 7 + n_joint_q] = x0[7 : 7 + n_joint_q]

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        feet = np.zeros((12, 1))
        for leg_idx, frame_id in enumerate(self.foot_frame_ids):
            feet[3 * leg_idx : 3 * leg_idx + 3, 0] = self.data.oMf[
                frame_id
            ].translation

        return np.vstack(
            [
                np.asarray(x0[0:3], dtype=float).reshape(3, 1),
                velocity_world.reshape(3, 1),
                np.zeros((3, 1)),
                q0.reshape(4, 1),
                np.zeros((3, 1)),
                feet,
            ]
        )

    def rotated_tip_offsets(
        self, q_meas: np.ndarray, tip_offset: np.ndarray
    ) -> np.ndarray:
        """Return per-step world-frame tip offsets with shape (T, 4, 3)."""

        tip_offset = np.asarray(tip_offset, dtype=float).reshape(12)
        offsets = np.zeros((q_meas.shape[0], 4, 3))
        local_data = self.model.createData()
        for k, q_i in enumerate(q_meas):
            q_zero, _ = build_zeroed(q_i, np.zeros(self.model.nv))
            pin.forwardKinematics(self.model, local_data, q_zero)
            pin.updateFramePlacements(self.model, local_data)
            for leg_idx, frame_id in enumerate(self.foot_frame_ids):
                frame = self.model.frames[frame_id]
                joint_id = (
                    frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent
                )
                rotation = np.asarray(local_data.oMi[joint_id].rotation, dtype=float)
                offsets[k, leg_idx, :] = rotation @ tip_offset[
                    3 * leg_idx : 3 * leg_idx + 3
                ]
        return offsets
