"""Planner data types for the single-paddle receive task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BallTrajectoryPoint:
    """A predicted ball state at a future time."""

    t_s: float
    pos: np.ndarray
    vel: np.ndarray


@dataclass(frozen=True)
class HitPlan:
    """A candidate nominal impact plan for the paddle."""

    valid: bool
    hit_time_s: float
    hit_pos: np.ndarray
    hit_euler: np.ndarray
    hit_vel: np.ndarray
    outgoing_vel_des: np.ndarray
    target_landing_xy: np.ndarray
    cost: float

    @classmethod
    def invalid(cls, target_landing_xy: np.ndarray | None = None) -> "HitPlan":
        """Return an invalid placeholder plan."""

        landing_xy = (
            np.asarray(target_landing_xy, dtype=np.float64).reshape(2)
            if target_landing_xy is not None
            else np.zeros(2, dtype=np.float64)
        )
        zeros3 = np.zeros(3, dtype=np.float64)
        return cls(
            valid=False,
            hit_time_s=0.0,
            hit_pos=zeros3.copy(),
            hit_euler=zeros3.copy(),
            hit_vel=zeros3.copy(),
            outgoing_vel_des=zeros3.copy(),
            target_landing_xy=landing_xy,
            cost=float("inf"),
        )


@dataclass(frozen=True)
class PaddleCommand:
    """A nominal paddle command in task joint coordinates."""

    qpos: np.ndarray
    qvel: np.ndarray
