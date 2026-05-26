"""Control backends for the single-paddle receive task."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .env_cfg import ControlCfg


@dataclass
class ControlBackend:
    """Direct paddle control backend.

    The method signature is kept intentionally small so a later OpenArm IK
    backend can implement the same `apply` contract.
    """

    control_cfg: ControlCfg
    actuator_ids: tuple[int, ...]

    def apply(
        self,
        action: np.ndarray,
        data: mujoco.MjData,
        *,
        prev_cmd_qpos: np.ndarray,
        nominal_cmd_qpos: np.ndarray | None,
    ) -> np.ndarray:
        """Apply an action under the configured control mode."""

        mode = str(self.control_cfg.control_mode)
        prev_cmd_qpos = np.asarray(prev_cmd_qpos, dtype=np.float64)
        nominal_cmd_qpos = (
            prev_cmd_qpos if nominal_cmd_qpos is None else np.asarray(nominal_cmd_qpos, dtype=np.float64)
        )

        if mode == "direct_delta":
            target = prev_cmd_qpos + self._scale_pose_delta(
                action,
                linear_scale=self.control_cfg.action_scale_m,
                angular_scale=self.control_cfg.action_scale_rad,
            )
        elif mode == "planner_only":
            target = nominal_cmd_qpos.copy()
        elif mode == "planner_residual":
            target = nominal_cmd_qpos + self._scale_pose_delta(
                action,
                linear_scale=self.control_cfg.residual_scale_m,
                angular_scale=self.control_cfg.residual_scale_rad,
            )
        else:
            raise ValueError(f"Unsupported control_mode: {mode}")

        return self._clip_and_write(target, data)

    def _scale_pose_delta(
        self,
        action: np.ndarray,
        *,
        linear_scale: tuple[float, float, float],
        angular_scale: tuple[float, float, float],
    ) -> np.ndarray:
        """Scale a normalized 6-DoF action into task joint deltas."""

        return np.array(
            [
                action[0] * linear_scale[0],
                action[1] * linear_scale[1],
                action[2] * linear_scale[2],
                action[3] * angular_scale[0],
                action[4] * angular_scale[1],
                action[5] * angular_scale[2],
            ],
            dtype=np.float64,
        )

    def _clip_and_write(
        self,
        target: np.ndarray,
        data: mujoco.MjData,
    ) -> np.ndarray:
        """Clip a task-space paddle pose to limits and write actuator targets."""

        limits = (
            self.control_cfg.x_range_m,
            self.control_cfg.y_range_m,
            self.control_cfg.z_range_m,
            self.control_cfg.roll_range_rad,
            self.control_cfg.pitch_range_rad,
            self.control_cfg.yaw_range_rad,
        )
        clipped = np.asarray(target, dtype=np.float64).copy()
        for i, (lower, upper) in enumerate(limits):
            clipped[i] = float(np.clip(clipped[i], lower, upper))
            data.ctrl[self.actuator_ids[i]] = clipped[i]
        return clipped
