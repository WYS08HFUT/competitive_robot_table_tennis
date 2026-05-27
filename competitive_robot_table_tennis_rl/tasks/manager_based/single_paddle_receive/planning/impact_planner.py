"""Analytic impact planner for the single-paddle receive task."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, sqrt

import numpy as np

from ..env_cfg import (
    BALL_RADIUS_M,
    PADDLE_CONTACT_HALF_THICKNESS_M,
    TABLE_TOP_Z_M,
    ControlCfg,
    PlannerCfg,
)
from ..mdp.transforms import normalize, paddle_normal_from_euler
from .ball_predictor import BallPredictor
from .planner_types import HitPlan


_REJECT_HIT_Z_OUT_OF_RANGE = "hit_z_out_of_range"
_REJECT_OUTGOING_SPEED_TOO_HIGH = "outgoing_speed_too_high"
_REJECT_OUTGOING_NOT_FORWARD = "outgoing_not_forward"
_REJECT_WORKSPACE_OUTSIDE = "workspace_outside"
_REJECT_UNREACHABLE_IN_TIME = "unreachable_in_time"


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""

    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _angle_delta(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return wrapped Euler-angle deltas."""

    return np.array([_wrap_angle(float(a - b)) for a, b in zip(lhs, rhs)], dtype=np.float64)


@dataclass
class _HitCandidate:
    """Internal candidate state used during impact-plan search."""

    t_s: float
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    outgoing_vel: np.ndarray
    hit_pos: np.ndarray
    hit_euler: np.ndarray
    hit_vel: np.ndarray
    linear_time_s: float = 0.0
    angular_time_s: float = 0.0
    reach_slack_s: float = 0.0
    valid: bool = True
    reject_reason: str = ""
    cost: float = float("inf")


def _sample_times(planner_cfg: PlannerCfg) -> np.ndarray:
    """Return the candidate hit times evaluated by the planner."""

    return np.linspace(
        planner_cfg.hit_time_min_s,
        planner_cfg.hit_time_max_s,
        planner_cfg.hit_time_samples,
        dtype=np.float64,
    )


def _solve_outgoing_velocity(
    *,
    ball_hit_pos: np.ndarray,
    target_landing_xy: np.ndarray,
    desired_flight_time_s: float,
    gravity: np.ndarray,
) -> np.ndarray:
    """Return the unconstrained outgoing velocity toward the target landing point."""

    target_pos = np.array(
        [float(target_landing_xy[0]), float(target_landing_xy[1]), TABLE_TOP_Z_M + BALL_RADIUS_M],
        dtype=np.float64,
    )
    flight_time_s = max(float(desired_flight_time_s), 1e-6)
    return (target_pos - np.asarray(ball_hit_pos, dtype=np.float64) - 0.5 * gravity * (flight_time_s**2)) / flight_time_s


def _estimate_paddle_euler(
    incoming_vel: np.ndarray,
    outgoing_vel: np.ndarray,
) -> np.ndarray:
    """Estimate a paddle orientation from incoming and desired outgoing ball velocity."""

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


def _compute_contact_pose(
    ball_hit_pos: np.ndarray,
    hit_euler: np.ndarray,
) -> np.ndarray:
    """Convert a desired ball contact point into a paddle-center contact pose."""

    normal = paddle_normal_from_euler(hit_euler)
    contact_offset = normal * (BALL_RADIUS_M + PADDLE_CONTACT_HALF_THICKNESS_M)
    return np.asarray(ball_hit_pos, dtype=np.float64) - contact_offset


def _make_candidate(
    *,
    t_s: float,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    target_landing_xy: np.ndarray,
    desired_flight_time_s: float,
    gravity: np.ndarray,
) -> _HitCandidate:
    """Construct a raw contact candidate from a future ball state."""

    outgoing_vel = _solve_outgoing_velocity(
        ball_hit_pos=ball_pos,
        target_landing_xy=target_landing_xy,
        desired_flight_time_s=desired_flight_time_s,
        gravity=gravity,
    )
    hit_euler = _estimate_paddle_euler(ball_vel, outgoing_vel)
    hit_pos = _compute_contact_pose(ball_pos, hit_euler)
    hit_vel = outgoing_vel - ball_vel
    return _HitCandidate(
        t_s=float(t_s),
        ball_pos=np.asarray(ball_pos, dtype=np.float64),
        ball_vel=np.asarray(ball_vel, dtype=np.float64),
        outgoing_vel=outgoing_vel,
        hit_pos=hit_pos,
        hit_euler=hit_euler,
        hit_vel=hit_vel,
    )


def _inside_workspace(hit_pos: np.ndarray, *, control_cfg: ControlCfg) -> bool:
    """Return whether a candidate paddle center lies inside the allowed workspace."""

    return (
        control_cfg.x_range_m[0] <= float(hit_pos[0]) <= control_cfg.x_range_m[1]
        and control_cfg.y_range_m[0] <= float(hit_pos[1]) <= control_cfg.y_range_m[1]
        and control_cfg.z_range_m[0] <= float(hit_pos[2]) <= control_cfg.z_range_m[1]
    )


def _clip_qpos(qpos: np.ndarray, *, lower_limits: np.ndarray, upper_limits: np.ndarray) -> np.ndarray:
    """Clip a paddle task-space pose to configured bounds."""

    clipped = np.asarray(qpos, dtype=np.float64).copy()
    clipped[:3] = np.clip(clipped[:3], lower_limits[:3], upper_limits[:3])
    clipped[3:] = np.clip(clipped[3:], lower_limits[3:], upper_limits[3:])
    return clipped


def _stage_qpos(
    hit_qpos: np.ndarray,
    *,
    stage_standoff_m: float,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> np.ndarray:
    """Return the staged pre-contact pose behind the ball along the paddle normal."""

    stage_qpos = np.asarray(hit_qpos, dtype=np.float64).copy()
    hit_normal = paddle_normal_from_euler(stage_qpos[3:])
    stage_qpos[:3] -= hit_normal * float(stage_standoff_m)
    return _clip_qpos(stage_qpos, lower_limits=lower_limits, upper_limits=upper_limits)


def _validate_candidate(
    candidate: _HitCandidate,
    *,
    paddle_qpos: np.ndarray,
    planner_cfg: PlannerCfg,
    control_cfg: ControlCfg,
    linear_max_speed: float,
    angular_max_speed: float,
    stage_standoff_m: float,
    stage_release_time_s: float,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> _HitCandidate:
    """Apply all candidate rejection checks and annotate reachability metrics."""

    if not (planner_cfg.min_hit_z_m <= float(candidate.ball_pos[2]) <= planner_cfg.max_hit_z_m):
        candidate.valid = False
        candidate.reject_reason = _REJECT_HIT_Z_OUT_OF_RANGE
        return candidate

    outgoing_speed = float(np.linalg.norm(candidate.outgoing_vel))
    if outgoing_speed > planner_cfg.max_outgoing_speed_m_s:
        candidate.valid = False
        candidate.reject_reason = _REJECT_OUTGOING_SPEED_TOO_HIGH
        return candidate

    if candidate.outgoing_vel[1] <= 0.0:
        candidate.valid = False
        candidate.reject_reason = _REJECT_OUTGOING_NOT_FORWARD
        return candidate

    if not _inside_workspace(candidate.hit_pos, control_cfg=control_cfg):
        candidate.valid = False
        candidate.reject_reason = _REJECT_WORKSPACE_OUTSIDE
        return candidate

    current_qpos = np.asarray(paddle_qpos, dtype=np.float64)
    hit_qpos = np.concatenate([candidate.hit_pos, candidate.hit_euler]).astype(np.float64, copy=False)
    release_time_s = min(float(stage_release_time_s), max(0.5 * candidate.t_s, 1e-3))
    stage_qpos = _stage_qpos(
        hit_qpos,
        stage_standoff_m=stage_standoff_m,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )

    candidate.angular_time_s = float(np.linalg.norm(_angle_delta(hit_qpos[3:], current_qpos[3:]))) / angular_max_speed
    if candidate.t_s <= release_time_s + 1e-6:
        candidate.linear_time_s = float(np.linalg.norm(hit_qpos[:3] - current_qpos[:3])) / linear_max_speed
        required_time_s = max(candidate.linear_time_s, candidate.angular_time_s)
        candidate.reach_slack_s = candidate.t_s - required_time_s
        if required_time_s > candidate.t_s + 1e-6:
            candidate.valid = False
            candidate.reject_reason = _REJECT_UNREACHABLE_IN_TIME
        return candidate

    stage_budget_s = candidate.t_s - release_time_s
    time_to_stage_s = float(np.linalg.norm(stage_qpos[:3] - current_qpos[:3])) / linear_max_speed
    time_stage_to_hit_s = float(np.linalg.norm(hit_qpos[:3] - stage_qpos[:3])) / linear_max_speed
    candidate.linear_time_s = time_to_stage_s + time_stage_to_hit_s
    required_time_s = max(candidate.linear_time_s, candidate.angular_time_s)
    candidate.reach_slack_s = candidate.t_s - required_time_s
    if (
        time_to_stage_s > stage_budget_s + 1e-6
        or time_stage_to_hit_s > release_time_s + 1e-6
        or candidate.angular_time_s > candidate.t_s + 1e-6
    ):
        candidate.valid = False
        candidate.reject_reason = _REJECT_UNREACHABLE_IN_TIME
    return candidate


def _score_candidate(
    candidate: _HitCandidate,
    *,
    paddle_qpos: np.ndarray,
    time_weight: float,
    rot_weight: float,
    slack_weight: float,
) -> float:
    """Return the candidate score used to choose the best hit plan."""

    pos_cost = float(np.linalg.norm(np.asarray(paddle_qpos[:3], dtype=np.float64) - candidate.hit_pos))
    rot_cost = float(np.linalg.norm(_angle_delta(np.asarray(paddle_qpos[3:], dtype=np.float64), candidate.hit_euler)))
    return pos_cost + rot_weight * rot_cost + time_weight * candidate.t_s - slack_weight * candidate.reach_slack_s


def _candidate_to_hit_plan(candidate: _HitCandidate, *, target_landing_xy: np.ndarray) -> HitPlan:
    """Convert a validated, scored candidate into the planner's public output type."""

    return HitPlan(
        valid=True,
        hit_time_s=float(candidate.t_s),
        ball_hit_pos=np.asarray(candidate.ball_pos, dtype=np.float64),
        hit_pos=np.asarray(candidate.hit_pos, dtype=np.float64),
        hit_euler=np.asarray(candidate.hit_euler, dtype=np.float64),
        incoming_ball_vel=np.asarray(candidate.ball_vel, dtype=np.float64),
        hit_vel=np.asarray(candidate.hit_vel, dtype=np.float64),
        outgoing_vel_des=np.asarray(candidate.outgoing_vel, dtype=np.float64),
        target_landing_xy=np.asarray(target_landing_xy, dtype=np.float64).reshape(2),
        cost=float(candidate.cost),
    )


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
        self.linear_max_speed = max(float(planner_cfg.apf_linear_max_speed_m_s), 1e-6)
        self.angular_max_speed = max(float(planner_cfg.apf_angular_max_speed_rad_s), 1e-6)
        self.stage_standoff_m = float(planner_cfg.apf_stage_standoff_m)
        self.stage_release_time_s = float(planner_cfg.apf_stage_release_time_s)
        self.lower_limits = np.array(
            [
                control_cfg.x_range_m[0],
                control_cfg.y_range_m[0],
                control_cfg.z_range_m[0],
                control_cfg.roll_range_rad[0],
                control_cfg.pitch_range_rad[0],
                control_cfg.yaw_range_rad[0],
            ],
            dtype=np.float64,
        )
        self.upper_limits = np.array(
            [
                control_cfg.x_range_m[1],
                control_cfg.y_range_m[1],
                control_cfg.z_range_m[1],
                control_cfg.roll_range_rad[1],
                control_cfg.pitch_range_rad[1],
                control_cfg.yaw_range_rad[1],
            ],
            dtype=np.float64,
        )

    def plan(
        self,
        *,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        paddle_qpos: np.ndarray,
        target_landing_xy: np.ndarray,
        bounce_has_occurred: bool = False,
    ) -> HitPlan:
        """Return the lowest-cost reachable impact plan."""

        target_landing_xy = np.asarray(target_landing_xy, dtype=np.float64).reshape(2)
        times_s = _sample_times(self.planner_cfg)

        best: _HitCandidate | None = None
        for t_s in times_s:
            point = self.ball_predictor.predict_point(
                ball_pos,
                ball_vel,
                float(t_s),
                allow_table_bounce=True,
                bounce_has_occurred=bounce_has_occurred,
            )
            candidate = _make_candidate(
                t_s=float(t_s),
                ball_pos=point.pos,
                ball_vel=point.vel,
                target_landing_xy=target_landing_xy,
                desired_flight_time_s=self.planner_cfg.desired_flight_time_s,
                gravity=self.ball_predictor.gravity,
            )
            candidate = _validate_candidate(
                candidate,
                paddle_qpos=np.asarray(paddle_qpos, dtype=np.float64),
                planner_cfg=self.planner_cfg,
                control_cfg=self.control_cfg,
                linear_max_speed=self.linear_max_speed,
                angular_max_speed=self.angular_max_speed,
                stage_standoff_m=self.stage_standoff_m,
                stage_release_time_s=self.stage_release_time_s,
                lower_limits=self.lower_limits,
                upper_limits=self.upper_limits,
            )
            if not candidate.valid:
                continue

            candidate.cost = _score_candidate(
                candidate,
                paddle_qpos=paddle_qpos,
                time_weight=0.20,
                rot_weight=0.15,
                slack_weight=0.30,
            )
            if best is None or candidate.cost < best.cost:
                best = candidate

        return HitPlan.invalid(target_landing_xy) if best is None else _candidate_to_hit_plan(best, target_landing_xy=target_landing_xy)
