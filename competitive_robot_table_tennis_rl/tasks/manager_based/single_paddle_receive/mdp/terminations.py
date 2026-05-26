"""Termination helpers for the single-paddle receive task."""

from __future__ import annotations

import numpy as np

from ..env_cfg import TerminationCfg
from .events import RuntimeState


def terminal_from_state(
    runtime_state: RuntimeState,
    ball_pos: np.ndarray,
    termination_cfg: TerminationCfg,
) -> tuple[bool, str]:
    """Check terminal conditions that do not require contact context."""

    if runtime_state.success:
        return True, "success"
    if runtime_state.failure_reason:
        return True, runtime_state.failure_reason
    if ball_pos[2] < termination_cfg.min_ball_z_m:
        return True, "ball_floor"
    if abs(ball_pos[0]) > termination_cfg.max_abs_x_m or abs(ball_pos[1]) > termination_cfg.max_abs_y_m:
        return True, "ball_out_of_bounds"
    return False, ""
