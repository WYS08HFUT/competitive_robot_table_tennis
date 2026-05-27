"""Ball trajectory prediction for the single-paddle receive task."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..env_cfg import BALL_RADIUS_M, DEFAULT_GRAVITY_M_S2, TABLE_TOP_Z_M
from ..mdp.transforms import point_in_table_bounds
from .planner_types import BallTrajectoryPoint


class BallPredictor:
    """Ball trajectory predictor with an optional one-bounce receive model."""

    def __init__(
        self,
        gravity_m_s2: float = DEFAULT_GRAVITY_M_S2,
        *,
        table_bounce_restitution_z: float = 0.88,
        table_bounce_damping_xy: float = 0.96,
    ) -> None:
        self.gravity = np.array([0.0, 0.0, -float(gravity_m_s2)], dtype=np.float64)
        self.gravity_m_s2 = float(gravity_m_s2)
        self.table_bounce_restitution_z = float(table_bounce_restitution_z)
        self.table_bounce_damping_xy = float(table_bounce_damping_xy)

    def predict_point(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        t_s: float,
        *,
        allow_table_bounce: bool = False,
        bounce_has_occurred: bool = False,
    ) -> BallTrajectoryPoint:
        """Predict the ball state at a single future time."""

        t_s = float(t_s)
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        if allow_table_bounce:
            return self.predict_point_with_table_bounce(
                pos,
                vel,
                t_s,
                bounce_has_occurred=bounce_has_occurred,
            )
        pred_pos = pos + vel * t_s + 0.5 * self.gravity * (t_s**2)
        pred_vel = vel + self.gravity * t_s
        return BallTrajectoryPoint(t_s=t_s, pos=pred_pos, vel=pred_vel)

    def predict_point_with_table_bounce(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        t_s: float,
        *,
        bounce_has_occurred: bool = False,
    ) -> BallTrajectoryPoint:
        """Predict the ball state with a single table bounce on the receive side."""

        t_s = float(t_s)
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        if t_s <= 0.0:
            return BallTrajectoryPoint(t_s=t_s, pos=pos.copy(), vel=vel.copy())

        if bounce_has_occurred:
            pred_pos = pos + vel * t_s + 0.5 * self.gravity * (t_s**2)
            pred_vel = vel + self.gravity * t_s
            return BallTrajectoryPoint(t_s=t_s, pos=pred_pos, vel=pred_vel)

        landing = self._predict_table_bounce(pos, vel, max_time_s=t_s)
        if landing is None:
            pred_pos = pos + vel * t_s + 0.5 * self.gravity * (t_s**2)
            pred_vel = vel + self.gravity * t_s
            return BallTrajectoryPoint(t_s=t_s, pos=pred_pos, vel=pred_vel)

        bounce_pos, bounced_vel, t_bounce = landing
        if t_s <= t_bounce:
            pred_pos = pos + vel * t_s + 0.5 * self.gravity * (t_s**2)
            pred_vel = vel + self.gravity * t_s
            return BallTrajectoryPoint(t_s=t_s, pos=pred_pos, vel=pred_vel)

        dt_after_bounce = t_s - t_bounce
        pred_pos = bounce_pos + bounced_vel * dt_after_bounce + 0.5 * self.gravity * (dt_after_bounce**2)
        pred_vel = bounced_vel + self.gravity * dt_after_bounce
        return BallTrajectoryPoint(t_s=t_s, pos=pred_pos, vel=pred_vel)

    def predict(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        times_s: Iterable[float],
        *,
        allow_table_bounce: bool = False,
        bounce_has_occurred: bool = False,
    ) -> list[BallTrajectoryPoint]:
        """Predict the ball state over a sequence of future times."""

        return [
            self.predict_point(
                pos,
                vel,
                t_s,
                allow_table_bounce=allow_table_bounce,
                bounce_has_occurred=bounce_has_occurred,
            )
            for t_s in times_s
        ]

    def _predict_table_bounce(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        *,
        max_time_s: float,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Return bounce state if the ball will hit the table plane in time."""

        t_bounce = self._first_positive_root(
            -0.5 * self.gravity_m_s2,
            float(vel[2]),
            float(pos[2] - (TABLE_TOP_Z_M + BALL_RADIUS_M)),
            max_time_s=max_time_s,
        )
        if t_bounce is None:
            return None

        bounce_pos = pos + vel * t_bounce + 0.5 * self.gravity * (t_bounce**2)
        if not point_in_table_bounds(float(bounce_pos[0]), float(bounce_pos[1])):
            return None
        vz_impact = float(vel[2] - self.gravity_m_s2 * t_bounce)
        bounced_vel = np.array(
            [
                float(vel[0]) * self.table_bounce_damping_xy,
                float(vel[1]) * self.table_bounce_damping_xy,
                abs(vz_impact) * self.table_bounce_restitution_z,
            ],
            dtype=np.float64,
        )
        bounce_pos = np.array(
            [float(bounce_pos[0]), float(bounce_pos[1]), TABLE_TOP_Z_M + BALL_RADIUS_M],
            dtype=np.float64,
        )
        return bounce_pos, bounced_vel, float(t_bounce)

    @staticmethod
    def _first_positive_root(
        a: float,
        b: float,
        c: float,
        *,
        max_time_s: float,
    ) -> float | None:
        """Return the smallest non-negative quadratic root within a horizon."""

        eps = 1e-8
        if abs(a) < eps:
            if abs(b) < eps:
                return None
            root = -c / b
            if root < 0.0 or root > max_time_s:
                return None
            return root

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        sqrt_disc = float(np.sqrt(disc))
        roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
        roots = [root for root in roots if 0.0 <= root <= max_time_s]
        if not roots:
            return None
        return min(roots)
