import numpy as np

from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.mdp.transforms import (
    compute_intercept_point_at_y,
    point_in_table_bounds,
    point_on_opponent_side,
    point_on_own_side,
    predict_ballistic_landing,
)


def test_compute_intercept_point_at_y() -> None:
    ball_pos = np.array([0.1, 1.2, 0.4], dtype=np.float64)
    ball_vel = np.array([0.0, -5.0, -1.0], dtype=np.float64)
    intercept = compute_intercept_point_at_y(ball_pos, ball_vel, y_target=-0.6)
    assert intercept is not None
    x, y, z, t = intercept
    assert np.isclose(y, -0.6)
    assert t > 0.0
    assert np.isclose(x, 0.1)
    assert z < ball_pos[2]


def test_predict_ballistic_landing() -> None:
    ball_pos = np.array([0.0, 0.5, 1.1], dtype=np.float64)
    ball_vel = np.array([0.0, 2.0, -0.5], dtype=np.float64)
    landing = predict_ballistic_landing(ball_pos, ball_vel)
    assert landing is not None
    x, y, t = landing
    assert t > 0.0
    assert np.isclose(x, 0.0)
    assert y > 0.5


def test_table_side_helpers() -> None:
    assert point_in_table_bounds(0.0, 0.0)
    assert not point_in_table_bounds(1.0, 0.0)
    assert point_on_own_side(-0.2)
    assert not point_on_own_side(0.2)
    assert point_on_opponent_side(0.2)
    assert not point_on_opponent_side(-0.2)
