"""Observation assembly for the single-paddle receive task."""

from __future__ import annotations

import numpy as np

from ..env_cfg import TABLE_HALF_LENGTH_M, TABLE_HALF_WIDTH_M
from .events import RuntimeState


def build_observation(
    *,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    paddle_qpos: np.ndarray,
    paddle_qvel: np.ndarray,
    prev_action: np.ndarray,
    runtime_state: RuntimeState,
) -> np.ndarray:
    """Build the flat observation vector."""

    pos_scale = np.array([TABLE_HALF_WIDTH_M, TABLE_HALF_LENGTH_M, 0.60], dtype=np.float64)
    linear_vel_scale = 5.0
    angular_vel_scale = 30.0
    angle_scale = np.pi
    rel_ball = (ball_pos - paddle_qpos[:3]) / pos_scale
    planned_hit_pos_rel = (
        (runtime_state.planned_hit_pos - paddle_qpos[:3]) / pos_scale
        if runtime_state.planned_hit_pos is not None
        else np.zeros(3, dtype=np.float64)
    )
    planned_hit_euler = (
        runtime_state.planned_hit_euler / angle_scale
        if runtime_state.planned_hit_euler is not None
        else np.zeros(3, dtype=np.float64)
    )
    planned_cmd_qpos = (
        runtime_state.planned_cmd_qpos
        if runtime_state.planned_cmd_qpos is not None
        else np.zeros(6, dtype=np.float64)
    )
    planned_cmd_qvel = (
        runtime_state.planned_cmd_qvel
        if runtime_state.planned_cmd_qvel is not None
        else np.zeros(6, dtype=np.float64)
    )
    planned_cmd_pos_rel = (planned_cmd_qpos[:3] - paddle_qpos[:3]) / pos_scale
    planned_cmd_euler = planned_cmd_qpos[3:] / angle_scale
    planned_outgoing_vel = (
        runtime_state.planned_outgoing_vel / linear_vel_scale
        if runtime_state.planned_outgoing_vel is not None
        else np.zeros(3, dtype=np.float64)
    )
    target_landing_xy = (
        runtime_state.target_landing_xy / np.array([TABLE_HALF_WIDTH_M, TABLE_HALF_LENGTH_M], dtype=np.float64)
        if runtime_state.target_landing_xy is not None
        else np.zeros(2, dtype=np.float64)
    )
    rule_flags = np.array(
        [
            float(runtime_state.has_hit),
            float(runtime_state.own_side_bounce_count),
            float(runtime_state.paddle_hit_count),
            float(runtime_state.crossed_net_after_hit),
            float(runtime_state.planner_valid),
        ],
        dtype=np.float64,
    )
    obs = np.concatenate(
        [
            ball_pos / pos_scale,
            ball_vel / linear_vel_scale,
            np.concatenate([paddle_qpos[:3] / pos_scale, paddle_qpos[3:] / angle_scale]),
            np.concatenate([paddle_qvel[:3] / linear_vel_scale, paddle_qvel[3:] / angular_vel_scale]),
            rel_ball,
            planned_hit_pos_rel,
            planned_hit_euler,
            np.array([runtime_state.planned_hit_time_s], dtype=np.float64),
            planned_cmd_pos_rel,
            planned_cmd_euler,
            planned_outgoing_vel,
            target_landing_xy,
            rule_flags,
            prev_action,
        ]
    )
    return obs.astype(np.float32, copy=False)
