"""Analytic impact planner for the single-paddle receive task."""

from __future__ import annotations

from math import atan2, sqrt

import numpy as np

from ..env_cfg import BALL_RADIUS_M, TABLE_TOP_Z_M, ControlCfg, PlannerCfg
from ..mdp.transforms import normalize
from .ball_predictor import BallPredictor
from .planner_types import HitPlan


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""

    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _angle_delta(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return wrapped Euler-angle deltas."""

    return np.array([_wrap_angle(float(a - b)) for a, b in zip(lhs, rhs)], dtype=np.float64)


class ImpactPlanner:
    """Search a reachable nominal hit state for the current incoming ball."""

    def __init__(
        self,
        planner_cfg: PlannerCfg,
        control_cfg: ControlCfg,
        *,
        ball_predictor: BallPredictor,
    ) -> None:
        self.planner_cfg = planner_cfg
        self.control_cfg = control_cfg
        self.ball_predictor = ball_predictor

    def plan(
        self,
        *,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        paddle_qpos: np.ndarray,
        target_landing_xy: np.ndarray,
    ) -> HitPlan:
        """Return the lowest-cost reachable impact plan."""

        target_landing_xy = np.asarray(target_landing_xy, dtype=np.float64).reshape(2)
        times_s = np.linspace(
            self.planner_cfg.hit_time_min_s,
            self.planner_cfg.hit_time_max_s,
            self.planner_cfg.hit_time_samples,
            dtype=np.float64,
        )

        best: HitPlan | None = None
        for t_s in times_s:
            point = self.ball_predictor.predict_point(ball_pos, ball_vel, float(t_s))
            if not self._inside_workspace(point.pos):
                continue
            if not (self.planner_cfg.min_hit_z_m <= float(point.pos[2]) <= self.planner_cfg.max_hit_z_m):
                continue

            outgoing_vel = self._solve_outgoing_velocity(point.pos, target_landing_xy)
            if outgoing_vel is None:
                continue

            hit_euler = self._estimate_paddle_euler(point.vel, outgoing_vel)
            hit_vel = outgoing_vel - point.vel
            cost = self._plan_cost(
                paddle_qpos=paddle_qpos,
                hit_pos=point.pos,
                hit_euler=hit_euler,
                hit_time_s=float(t_s),
            )
            candidate = HitPlan(
                valid=True,
                hit_time_s=float(t_s),
                hit_pos=np.asarray(point.pos, dtype=np.float64),
                hit_euler=hit_euler,
                hit_vel=hit_vel,
                outgoing_vel_des=outgoing_vel,
                target_landing_xy=target_landing_xy,
                cost=cost,
            )
            if best is None or candidate.cost < best.cost:
                best = candidate

        return best if best is not None else HitPlan.invalid(target_landing_xy)

    def _inside_workspace(self, hit_pos: np.ndarray) -> bool:
        return (
            self.control_cfg.x_range_m[0] <= float(hit_pos[0]) <= self.control_cfg.x_range_m[1]
            and self.control_cfg.y_range_m[0] <= float(hit_pos[1]) <= self.control_cfg.y_range_m[1]
            and self.control_cfg.z_range_m[0] <= float(hit_pos[2]) <= self.control_cfg.z_range_m[1]
        )

    def _solve_outgoing_velocity(
        self,
        hit_pos: np.ndarray,
        target_landing_xy: np.ndarray,
    ) -> np.ndarray | None:
        target_pos = np.array(
            [float(target_landing_xy[0]), float(target_landing_xy[1]), TABLE_TOP_Z_M + BALL_RADIUS_M],
            dtype=np.float64,
        )
        t_s = float(self.planner_cfg.desired_flight_time_s)
        gravity = self.ball_predictor.gravity
        outgoing_vel = (target_pos - hit_pos - 0.5 * gravity * (t_s**2)) / max(t_s, 1e-6)
        speed = float(np.linalg.norm(outgoing_vel))
        if speed > self.planner_cfg.max_outgoing_speed_m_s:
            return None
        if outgoing_vel[1] <= 0.0:
            return None
        return outgoing_vel

    def _estimate_paddle_euler(
        self,
        incoming_vel: np.ndarray,
        outgoing_vel: np.ndarray,
    ) -> np.ndarray:
        incoming_dir = normalize(incoming_vel)
        outgoing_dir = normalize(outgoing_vel)
        normal = normalize(outgoing_dir - incoming_dir)
        if float(np.linalg.norm(normal)) < 1e-8:
            normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        yaw = atan2(-float(normal[0]), float(normal[1]))
        planar = sqrt(float(normal[0] ** 2 + normal[1] ** 2))
        pitch = atan2(float(normal[2]), max(planar, 1e-6))
        roll = 0.0
        return np.array([roll, pitch, _wrap_angle(yaw)], dtype=np.float64)

    def _plan_cost(
        self,
        *,
        paddle_qpos: np.ndarray,
        hit_pos: np.ndarray,
        hit_euler: np.ndarray,
        hit_time_s: float,
    ) -> float:
        pos_cost = float(np.linalg.norm(np.asarray(paddle_qpos[:3], dtype=np.float64) - hit_pos))
        rot_cost = float(np.linalg.norm(_angle_delta(np.asarray(paddle_qpos[3:], dtype=np.float64), hit_euler)))
        return pos_cost + 0.15 * rot_cost + 0.20 * float(hit_time_s)
