from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

import competitive_robot_table_tennis_rl  # noqa: F401
from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.env import (
    SinglePaddleServeReceiveEnv,
)
from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.env_cfg import (
    TASK_IDS,
)
from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.utils.paths import prepare_run_dir, resolve_repo_path


def test_registry_and_reset_step() -> None:
    env = gym.make(TASK_IDS.easy)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert info["serve_id"] is not None
    ball_pos = env.unwrapped.ball_pos
    ball_vel = env.unwrapped.ball_vel
    assert ball_pos[1] > 0.0
    assert ball_vel[1] < 0.0
    next_obs, reward, terminated, truncated, step_info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "reward_terms" in step_info
    env.close()


def test_run_dir_creation() -> None:
    run_dir = prepare_run_dir(TASK_IDS.easy, "pytest")
    assert run_dir.exists()
    assert "runs/tasks" in str(run_dir)


def test_asset_contains_required_names() -> None:
    xml_path = resolve_repo_path(
        "competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/assets/"
        "single_paddle_receive_dataset_frame.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    required = [
        ("geom", "geom_ball"),
        ("geom", "geom_table"),
        ("geom", "geom_net"),
        ("geom", "paddle_collision"),
        ("site", "paddle_center_site"),
        ("site", "ball_site"),
        ("actuator", "paddle_x_pos"),
        ("actuator", "paddle_y_pos"),
        ("actuator", "paddle_z_pos"),
    ]
    type_map = {
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
    }
    for kind, name in required:
        assert mujoco.mj_name2id(model, type_map[kind], name) >= 0


def test_runtime_success_flag_can_be_triggered() -> None:
    env = SinglePaddleServeReceiveEnv(difficulty="easy")
    obs, _ = env.reset(seed=1)
    del obs
    env.state.has_hit = True
    env.state.success = True
    _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert terminated
    assert info["episode_success"] is True
    env.close()
