"""Runtime state and reset/event helpers."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..utils.serve_dataset import ServeSample


@dataclass
class ContactState:
    ball_table: bool = False
    ball_paddle: bool = False
    ball_net: bool = False
    ball_floor: bool = False


@dataclass
class RuntimeState:
    has_hit: bool = False
    paddle_hit_count: int = 0
    own_side_bounce_count: int = 0
    crossed_net_after_hit: bool = False
    net_contact_after_hit: bool = False
    predicted_intercept: np.ndarray | None = None
    predicted_landing: np.ndarray | None = None
    planner_valid: bool = False
    planned_hit_pos: np.ndarray | None = None
    planned_hit_euler: np.ndarray | None = None
    planned_hit_time_s: float = 0.0
    planned_cmd_qpos: np.ndarray | None = None
    planned_cmd_qvel: np.ndarray | None = None
    planned_outgoing_vel: np.ndarray | None = None
    target_landing_xy: np.ndarray | None = None
    success: bool = False
    failure_reason: str = ""


def reset_runtime_state() -> RuntimeState:
    """Create a fresh episode rule-state."""

    return RuntimeState()


def apply_serve_sample(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ball_joint_qpos_adr: int,
    ball_joint_qvel_adr: int,
    sample: ServeSample,
) -> None:
    """Apply a sampled ball state to the MuJoCo data."""

    del model
    data.qpos[ball_joint_qpos_adr : ball_joint_qpos_adr + 3] = sample.position
    data.qpos[ball_joint_qpos_adr + 3 : ball_joint_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qvel[ball_joint_qvel_adr : ball_joint_qvel_adr + 3] = sample.linear_velocity
    data.qvel[ball_joint_qvel_adr + 3 : ball_joint_qvel_adr + 6] = sample.angular_velocity
