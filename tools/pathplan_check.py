from dataclasses import replace
import time
import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer

import competitive_robot_table_tennis_rl  # noqa: F401
from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.env_cfg import make_task_cfg

cfg = make_task_cfg("easy")
cfg = replace(cfg, control=replace(cfg.control, control_mode="planner_only"))
episode_num=20
env = gym.make("TableTennis-SinglePaddleServeReceiveEasy-v0", cfg=cfg)


def _print_plan_trace(env_unwrapped, *, episode: int) -> None:
    hit_plan = env_unwrapped.current_hit_plan
    if not hit_plan.valid:
        print("plan", {"episode": episode, "planner_valid": False})
        return
    print(
        "plan",
        {
            "episode": episode,
            "predicted_contact_time_s": round(float(hit_plan.hit_time_s), 6),
            "predicted_contact_xyz": np.round(hit_plan.ball_hit_pos, 6),
            "planned_paddle_contact_pose": np.round(hit_plan.hit_pos, 6),
            "planned_paddle_euler": np.round(hit_plan.hit_euler, 6),
        },
    )

try:
    with mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data) as viewer:
        for episode in range(episode_num):
            obs, info = env.reset(seed=episode)
            contact_logged = False
            print("reset", {
                "episode": episode,
                "control_mode": info["control_mode"],
                "planner_valid": info["planner_valid"],
                "target_landing_xy": info["target_landing_xy"],
            })
            _print_plan_trace(env.unwrapped, episode=episode)
            viewer.sync()

            total_reward = 0.0
            for step in range(160):
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                obs, reward, terminated, truncated, step_info = env.step(action)
                total_reward += reward

                if not contact_logged and env.unwrapped.current_hit_plan.valid:
                    planned_contact_time_s = (
                        env.unwrapped.path_planner.start_time_s + env.unwrapped.current_hit_plan.hit_time_s
                    )
                    if env.unwrapped.episode_time_s >= planned_contact_time_s:
                        print(
                            "contact",
                            {
                                "episode": episode,
                                "step": step + 1,
                                "planned_contact_time_s": round(float(planned_contact_time_s), 6),
                                "sim_time_s": round(float(env.unwrapped.episode_time_s), 6),
                                "predicted_contact_xyz": np.round(env.unwrapped.current_hit_plan.ball_hit_pos, 6),
                                "actual_ball_xyz": np.round(env.unwrapped.ball_pos, 6),
                                "actual_paddle_qpos": np.round(env.unwrapped.paddle_qpos, 6),
                                "actual_paddle_center_xyz": np.round(env.unwrapped.paddle_pos, 6),
                            },
                        )
                        contact_logged = True

                if not viewer.is_running():
                    raise SystemExit

                viewer.sync()
                time.sleep(info["control_timestep_s"])

                if terminated or truncated:
                    print("done", {
                        "episode": episode,
                        "step": step + 1,
                        "total_reward": total_reward,
                        "episode_success": step_info.get("episode_success"),
                        "terminal_reason": step_info.get("terminal_reason"),
                        "planner_valid": step_info.get("planner_valid"),
                    })
                    break
finally:
    env.close()
