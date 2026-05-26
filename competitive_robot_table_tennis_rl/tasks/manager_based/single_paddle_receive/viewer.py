"""Simple viewer and smoke rollout for the single-paddle receive task."""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

import competitive_robot_table_tennis_rl  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a short single-paddle receive rollout.")
    parser.add_argument("--task-id", default="TableTennis-SinglePaddleServeReceiveEasy-v0")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make(args.task_id, render_mode="rgb_array")
    obs, info = env.reset(seed=args.seed)
    print("reset", {"obs_shape": tuple(obs.shape), **info})

    total_reward = 0.0
    for step in range(args.steps):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        try:
            frame = env.render()
            if step == 0:
                print("frame_shape", frame.shape)
        except RuntimeError as exc:
            print("render_unavailable", str(exc))
            break
        if terminated or truncated:
            print("done", {"step": step + 1, "total_reward": total_reward, **info})
            break
    else:
        print("partial_rollout", {"steps": args.steps, "total_reward": total_reward})
    env.close()


if __name__ == "__main__":
    main()
