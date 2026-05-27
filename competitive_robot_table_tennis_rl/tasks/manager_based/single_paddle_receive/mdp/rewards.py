"""Reward helpers for the single-paddle receive task."""

from __future__ import annotations

import numpy as np

from ..env_cfg import RewardCfg
from .events import RuntimeState
from .transforms import normalize


def _wrapped_angle_norm(delta: np.ndarray) -> float:
    """Return the norm of wrapped Euler-angle deltas."""

    wrapped = (np.asarray(delta, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.linalg.norm(wrapped))


def dense_reward(
    *,
    reward_cfg: RewardCfg,
    runtime_state: RuntimeState,
    ball_vel: np.ndarray,
    paddle_qpos: np.ndarray,
    paddle_qvel: np.ndarray,
    predicted_landing: np.ndarray | None,
    target_landing_xy: np.ndarray | None,
    action: np.ndarray,
    prev_action: np.ndarray,
) -> dict[str, float]:
    """Compute dense shaping terms."""

    reward = {
        "survive": reward_cfg.survive,
        "tracking_pos": 0.0,
        "tracking_rot": 0.0,
        "tracking_vel": 0.0,
        "outgoing_direction": 0.0,
        "landing_shape": 0.0,
        "action_rate_penalty": -reward_cfg.action_rate_penalty * float(np.sum((action - prev_action) ** 2)),
        "action_mag_penalty": -reward_cfg.action_mag_penalty * float(np.sum(action**2)),
        "qvel_penalty": -reward_cfg.qvel_penalty * float(np.sum(paddle_qvel**2)),
    }

    if (
        not runtime_state.has_hit
        and runtime_state.planner_valid
        and runtime_state.planned_cmd_qpos is not None
        and runtime_state.planned_cmd_qvel is not None
    ):
        pos_err = float(np.linalg.norm(paddle_qpos[:3] - runtime_state.planned_cmd_qpos[:3]))
        rot_err = _wrapped_angle_norm(paddle_qpos[3:] - runtime_state.planned_cmd_qpos[3:])
        vel_err = float(np.linalg.norm(paddle_qvel - runtime_state.planned_cmd_qvel))
        reward["tracking_pos"] = reward_cfg.tracking_pos * float(np.exp(-pos_err / reward_cfg.tracking_pos_sigma_m))
        reward["tracking_rot"] = reward_cfg.tracking_rot * float(np.exp(-rot_err / reward_cfg.tracking_rot_sigma_rad))
        reward["tracking_vel"] = reward_cfg.tracking_vel * float(np.exp(-vel_err / reward_cfg.tracking_vel_sigma_m_s))

    if (
        runtime_state.has_hit
        and not runtime_state.crossed_net_after_hit
        and runtime_state.planned_outgoing_vel is not None
    ):
        desired_dir = normalize(runtime_state.planned_outgoing_vel)
        actual_dir = normalize(ball_vel)
        direction_score = float(np.clip(np.dot(desired_dir, actual_dir), 0.0, 1.0))
        reward["outgoing_direction"] = reward_cfg.outgoing_direction * direction_score

    if runtime_state.has_hit and predicted_landing is not None and target_landing_xy is not None:
        err = float(np.linalg.norm(predicted_landing[:2] - target_landing_xy[:2]))
        reward["landing_shape"] = reward_cfg.landing_shape * float(np.exp(-err / reward_cfg.landing_sigma_m))

    return reward


def reward_for_legal_hit(reward_cfg: RewardCfg, ball_vel: np.ndarray) -> dict[str, float]:
    """Reward a first legal hit and immediate outgoing trajectory."""

    forward_component = float(np.clip((ball_vel[1] - reward_cfg.send_forward_min_speed_m_s) / reward_cfg.send_forward_scale_m_s, 0.0, 1.0))
    lift_component = float(np.clip((ball_vel[2] - reward_cfg.lift_min_m_s) / reward_cfg.lift_scale_m_s, 0.0, 1.0))
    return {
        "legal_hit": reward_cfg.legal_hit,
        "send_forward": reward_cfg.send_forward * forward_component,
        "lift": reward_cfg.lift * lift_component,
    }


def reward_for_cross_net(reward_cfg: RewardCfg) -> dict[str, float]:
    """Reward a successful net crossing."""

    return {"cross_net": reward_cfg.cross_net}


def reward_for_success(reward_cfg: RewardCfg) -> dict[str, float]:
    """Reward a successful opponent-side landing."""

    return {"landing": reward_cfg.landing}


def penalty(name: str, value: float) -> dict[str, float]:
    """Return a single named penalty term."""

    return {name: value}
