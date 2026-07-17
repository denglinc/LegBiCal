"""Upper-level trajectory loss and gradients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass(frozen=True)
class TrajectoryLossWeights:
    position: float = 1.0
    velocity: float = 3.0
    foothold: float = 0.5
    attitude: float = 2.0


@dataclass(frozen=True)
class LossEvaluation:
    value: float
    breakdown: dict[str, float]


class TrajectoryLoss:
    """Paper-level objective evaluated on the B1 mocap window."""

    def __init__(self, weights: TrajectoryLossWeights | None = None):
        self.weights = weights or TrajectoryLossWeights()

    def evaluate(
        self,
        state_traj: np.ndarray,
        x_ground_truth: np.ndarray,
        foot_ground_truth: np.ndarray,
        attitude_loss: float,
        base_offset: np.ndarray | None = None,
    ) -> LossEvaluation:
        if base_offset is None:
            base_offset = np.zeros(3)
        base_offset = np.asarray(base_offset, dtype=float).reshape(3)

        p_est = np.asarray(state_traj[:, 0:3], dtype=float)
        v_est = np.asarray(state_traj[:, 3:6], dtype=float)
        foot_est = np.asarray(state_traj[:, 16:28], dtype=float).reshape(-1, 4, 3)

        p_gt = np.asarray(x_ground_truth[:, 0:3], dtype=float)
        q_gt = self._normalized_quaternions(x_ground_truth[:, 3:7])
        v_gt = self._body_velocity_to_world(q_gt, x_ground_truth[:, 19:22])
        foot_gt = np.asarray(foot_ground_truth, dtype=float).reshape(-1, 4, 3)

        position_error = 0.0
        for k in range(p_est.shape[0]):
            rotation = pin.Quaternion(q_gt[k]).toRotationMatrix()
            residual = p_est[k] + rotation @ base_offset - p_gt[k]
            position_error += float(residual @ residual)

        velocity_error = float(np.sum((v_est - v_gt) ** 2))
        foot_error = float(np.sum((foot_est - foot_gt) ** 2))
        attitude_error = float(attitude_loss)

        total = (
            self.weights.position * position_error
            + self.weights.velocity * velocity_error
            + self.weights.foothold * foot_error
            + self.weights.attitude * attitude_error
        )
        return LossEvaluation(
            value=float(total),
            breakdown={
                "E_p": position_error,
                "E_v": velocity_error,
                "E_pf": foot_error,
                "L_q": attitude_error,
                "w_p": self.weights.position,
                "w_v": self.weights.velocity,
                "w_pfoot": self.weights.foothold,
                "w_q": self.weights.attitude,
            },
        )

    def state_gradient(
        self,
        state_traj: np.ndarray,
        x_ground_truth: np.ndarray,
        foot_ground_truth: np.ndarray,
        attitude_gradient: np.ndarray,
        base_offset: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the loss differential in the complete 3-D FIE coordinates."""

        if base_offset is None:
            base_offset = np.zeros(3)
        base_offset = np.asarray(base_offset, dtype=float).reshape(3)

        h1 = state_traj.shape[0]
        if attitude_gradient.shape != (h1, 4):
            raise ValueError(
                f"attitude gradient must have shape {(h1, 4)}, "
                f"got {attitude_gradient.shape}"
            )

        p_est = np.asarray(state_traj[:, 0:3], dtype=float)
        v_est = np.asarray(state_traj[:, 3:6], dtype=float)
        foot_est = np.asarray(state_traj[:, 16:28], dtype=float).reshape(h1, 4, 3)

        p_gt = np.asarray(x_ground_truth[:, 0:3], dtype=float)
        q_gt = self._normalized_quaternions(x_ground_truth[:, 3:7])
        v_gt = self._body_velocity_to_world(q_gt, x_ground_truth[:, 19:22])
        foot_gt = np.asarray(foot_ground_truth, dtype=float).reshape(h1, 4, 3)

        blocks: list[np.ndarray] = []
        for k in range(h1):
            rotation = pin.Quaternion(q_gt[k]).toRotationMatrix()
            position_residual = p_est[k] + rotation @ base_offset - p_gt[k]
            dp = 2.0 * self.weights.position * position_residual
            dv = 2.0 * self.weights.velocity * (v_est[k] - v_gt[k])
            dq = self.weights.attitude * attitude_gradient[k].reshape(4)
            dpf = 2.0 * self.weights.foothold * (foot_est[k] - foot_gt[k])
            blocks.append(
                np.concatenate(
                    [dp, dv, np.zeros(3), dq, np.zeros(3), dpf.reshape(12)]
                )
            )
        return np.concatenate(blocks)

    def base_offset_gradient(
        self,
        state_traj: np.ndarray,
        x_ground_truth: np.ndarray,
        base_offset: np.ndarray,
    ) -> np.ndarray:
        p_est = np.asarray(state_traj[:, 0:3], dtype=float)
        p_gt = np.asarray(x_ground_truth[:, 0:3], dtype=float)
        q_gt = self._normalized_quaternions(x_ground_truth[:, 3:7])
        base_offset = np.asarray(base_offset, dtype=float).reshape(3)

        gradient = np.zeros(3, dtype=float)
        for k in range(p_est.shape[0]):
            rotation = pin.Quaternion(q_gt[k]).toRotationMatrix()
            residual = p_est[k] + rotation @ base_offset - p_gt[k]
            gradient += 2.0 * self.weights.position * (rotation.T @ residual)
        return gradient

    @staticmethod
    def _normalized_quaternions(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-16)

    @staticmethod
    def _body_velocity_to_world(q_world_body: np.ndarray, velocity_body: np.ndarray) -> np.ndarray:
        velocity_world = np.zeros_like(velocity_body, dtype=float)
        for k in range(q_world_body.shape[0]):
            rotation = pin.Quaternion(q_world_body[k]).toRotationMatrix()
            velocity_world[k] = rotation @ velocity_body[k]
        return velocity_world
