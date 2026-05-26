"""Ball trajectory prediction for the single-paddle receive task."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..env_cfg import DEFAULT_GRAVITY_M_S2
from .planner_types import BallTrajectoryPoint


class BallPredictor:
    """Gravity-only ball trajectory predictor."""

    def __init__(self, gravity_m_s2: float = DEFAULT_GRAVITY_M_S2) -> None:
        self.gravity = np.array([0.0, 0.0, -float(gravity_m_s2)], dtype=np.float64)

    def predict_point(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        t_s: float,
    ) -> BallTrajectoryPoint:
        """Predict the ball state at a single future time."""

        t_s = float(t_s)
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        pred_pos = pos + vel * t_s + 0.5 * self.gravity * (t_s**2)
        pred_vel = vel + self.gravity * t_s
        return BallTrajectoryPoint(t_s=t_s, pos=pred_pos, vel=pred_vel)

    def predict(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        times_s: Iterable[float],
    ) -> list[BallTrajectoryPoint]:
        """Predict the ball state over a sequence of future times."""

        return [self.predict_point(pos, vel, t_s) for t_s in times_s]
