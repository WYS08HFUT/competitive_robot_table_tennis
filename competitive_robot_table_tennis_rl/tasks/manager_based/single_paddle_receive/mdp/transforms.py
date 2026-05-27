"""Coordinate-frame and prediction helpers."""

from __future__ import annotations

from math import sqrt

import numpy as np

from ..env_cfg import BALL_RADIUS_M, DEFAULT_GRAVITY_M_S2, TABLE_HALF_LENGTH_M, TABLE_HALF_WIDTH_M, TABLE_TOP_Z_M


def safe_norm(vector: np.ndarray) -> float:
    """Return a stable L2 norm."""

    return float(np.linalg.norm(vector))


def speed_norm(vector: np.ndarray) -> float:
    """Return the linear speed."""

    return safe_norm(vector)


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a normalized vector or zeros if nearly singular."""

    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return np.zeros_like(vector)
    return vector / norm


def _first_positive_root(a: float, b: float, c: float, max_time_s: float | None = None) -> float | None:
    """Return the smallest non-negative quadratic root."""

    eps = 1e-8
    if abs(a) < eps:
        if abs(b) < eps:
            return None
        root = -c / b
        if root < 0.0:
            return None
        if max_time_s is not None and root > max_time_s:
            return None
        return root

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = sqrt(disc)
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    roots = [root for root in roots if root >= 0.0]
    if not roots:
        return None
    root = min(roots)
    if max_time_s is not None and root > max_time_s:
        return None
    return root


def predict_ballistic_landing(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    z_target: float = TABLE_TOP_Z_M + BALL_RADIUS_M,
    gravity: float = DEFAULT_GRAVITY_M_S2,
    max_time_s: float = 2.5,
) -> tuple[float, float, float] | None:
    """Predict the first landing point on a z plane under gravity only."""

    t = _first_positive_root(-0.5 * gravity, float(ball_vel[2]), float(ball_pos[2] - z_target), max_time_s=max_time_s)
    if t is None:
        return None
    x = float(ball_pos[0] + ball_vel[0] * t)
    y = float(ball_pos[1] + ball_vel[1] * t)
    return (x, y, t)


def compute_intercept_point_at_y(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    y_target: float,
    gravity: float = DEFAULT_GRAVITY_M_S2,
    max_time_s: float = 1.5,
) -> tuple[float, float, float, float] | None:
    """Estimate where the ball reaches a y plane."""

    vy = float(ball_vel[1])
    if abs(vy) < 1e-8:
        return None
    t = (y_target - float(ball_pos[1])) / vy
    if t < 0.0 or t > max_time_s:
        return None
    x = float(ball_pos[0] + ball_vel[0] * t)
    z = float(ball_pos[2] + ball_vel[2] * t - 0.5 * gravity * t * t)
    return (x, y_target, z, t)


def predict_post_bounce_intercept_at_y(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    y_target: float,
    *,
    gravity: float = DEFAULT_GRAVITY_M_S2,
    z_target: float = TABLE_TOP_Z_M + BALL_RADIUS_M,
    restitution_z: float = 0.88,
    damping_xy: float = 0.96,
    max_time_s: float = 2.0,
) -> tuple[float, float, float, float] | None:
    """Estimate a y-plane intercept after one table bounce.

    This lightweight helper is intentionally approximate. It assumes the ball
    first lands on the table plane, flips the vertical velocity with a fixed
    restitution, damps the planar velocity slightly, and then continues under
    gravity. That is enough for reset filtering and shaping on the receive task.
    """

    landing = predict_ballistic_landing(
        ball_pos,
        ball_vel,
        z_target=z_target,
        gravity=gravity,
        max_time_s=max_time_s,
    )
    if landing is None:
        return None

    landing_x, landing_y, t_landing = landing
    vz_impact = float(ball_vel[2] - gravity * t_landing)
    bounced_vel = np.array(
        [
            float(ball_vel[0]) * damping_xy,
            float(ball_vel[1]) * damping_xy,
            abs(vz_impact) * restitution_z,
        ],
        dtype=np.float64,
    )
    bounced_pos = np.array([landing_x, landing_y, z_target], dtype=np.float64)
    post = compute_intercept_point_at_y(
        bounced_pos,
        bounced_vel,
        y_target=y_target,
        gravity=gravity,
        max_time_s=max_time_s - t_landing,
    )
    if post is None:
        return None
    x, y, z, t_post = post
    return (x, y, z, t_landing + t_post)


def point_in_table_bounds(x: float, y: float, margin: float = 0.0) -> bool:
    """Return True if a point lies over the table surface footprint."""

    return (
        -TABLE_HALF_WIDTH_M + margin <= x <= TABLE_HALF_WIDTH_M - margin
        and -TABLE_HALF_LENGTH_M + margin <= y <= TABLE_HALF_LENGTH_M - margin
    )


def point_on_opponent_side(y: float) -> bool:
    """Return True if a point lies on the opponent half for the receiver."""

    return y > 0.0


def point_on_own_side(y: float) -> bool:
    """Return True if a point lies on the receiver half."""

    return y < 0.0


def paddle_normal_from_xmat(xmat: np.ndarray) -> np.ndarray:
    """Return the paddle-face normal, assuming the local y-axis is the face normal."""

    mat = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    return normalize(mat[:, 1])


def paddle_normal_from_euler(euler: np.ndarray) -> np.ndarray:
    """Approximate the paddle-face normal from the planner Euler convention."""

    roll, pitch, yaw = np.asarray(euler, dtype=np.float64)
    del roll
    cos_pitch = float(np.cos(pitch))
    return normalize(
        np.array(
            [
                -float(np.sin(yaw)) * cos_pitch,
                float(np.cos(yaw)) * cos_pitch,
                float(np.sin(pitch)),
            ],
            dtype=np.float64,
        )
    )
