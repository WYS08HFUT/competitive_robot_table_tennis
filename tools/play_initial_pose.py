"""Inspect the initial paddle pose in the canonical dataset-frame scene."""

from __future__ import annotations

import argparse
import os
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

import competitive_robot_table_tennis_rl  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and optionally override the initial paddle pose.")
    parser.add_argument("--task-id", default="TableTennis-SinglePaddleServeReceiveEasy-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--serve-id", type=int, default=None)
    parser.add_argument("--paddle-qpos", nargs=6, type=float, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--mujoco-gl", default=None)
    parser.add_argument("--show-contact-points", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    args = parser.parse_args()
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    env = gym.make(args.task_id)
    options = {}
    if args.serve_id is not None:
        options["serve_id"] = args.serve_id
    if args.paddle_qpos is not None:
        options["paddle_qpos"] = args.paddle_qpos

    obs, info = env.reset(seed=args.seed, options=options or None)
    print("reset", {"obs_shape": tuple(obs.shape), **info})
    print(
        "initial_pose",
        {
            "commanded_paddle_qpos": env.unwrapped.commanded_paddle_qpos.tolist(),
            "paddle_qpos": env.unwrapped.paddle_qpos.tolist(),
            "paddle_pos": env.unwrapped.paddle_pos.tolist(),
            "paddle_normal": env.unwrapped.paddle_normal.tolist(),
            "ball_pos": env.unwrapped.ball_pos.tolist(),
            "ball_vel": env.unwrapped.ball_vel.tolist(),
        },
    )

    print(
        "interactive_viewer",
        {
            "hint": "Close the MuJoCo viewer window or press Ctrl+C in the terminal to stop playback.",
            "control_timestep_s": info["control_timestep_s"],
        },
    )
    with mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = args.show_contact_points
        viewer.sync()
        total_reward = 0.0
        for step in range(args.steps):
            loop_start = time.perf_counter()
            obs, reward, terminated, truncated, step_info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            del obs
            total_reward += reward
            if not viewer.is_running():
                print("viewer_closed", {"step": step, "total_reward": total_reward})
                break
            viewer.sync()
            if terminated or truncated:
                print("done", {"step": step + 1, "total_reward": total_reward, **step_info})
                break
            if not args.no_realtime:
                elapsed = time.perf_counter() - loop_start
                sleep_s = max(0.0, info["control_timestep_s"] - elapsed)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
        else:
            print("partial_rollout", {"steps": args.steps, "total_reward": total_reward})

    env.close()


if __name__ == "__main__":
    main()
