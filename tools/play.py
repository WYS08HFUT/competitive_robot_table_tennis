"""Playback helper for a trained policy checkpoint."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

import competitive_robot_table_tennis_rl  # noqa: F401
from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.agents.mjlab.policy import load_policy_checkpoint, require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained policy checkpoint in the receive task.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-id", default='TableTennis-SinglePaddleServeReceiveEasy-v0')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--mujoco-gl", default=None)
    parser.add_argument("--show-contact-points", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    args = parser.parse_args()
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    torch, _ = require_torch()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    probe_env = gym.make(args.task_id or "TableTennis-SinglePaddleServeReceiveEasy-v0")
    obs, _ = probe_env.reset(seed=args.seed)
    policy, train_cfg = load_policy_checkpoint(
        checkpoint_path,
        obs_dim=int(np.prod(obs.shape)),
        act_dim=int(np.prod(probe_env.action_space.shape)),
    )
    task_id = args.task_id or str(train_cfg["task_id"])
    probe_env.close()

    env = gym.make(task_id)
    print("checkpoint", {"path": str(checkpoint_path), "train_cfg": train_cfg})
    try:
        with mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = args.show_contact_points
            for episode in range(args.episodes):
                obs, info = env.reset(seed=args.seed + episode)
                viewer.sync()
                print("reset", {"episode": episode, "obs_shape": tuple(obs.shape), **info})
                total_reward = 0.0
                for step in range(args.steps):
                    loop_start = time.perf_counter()
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        mean, std = policy(obs_tensor)
                        if args.deterministic:
                            action = torch.tanh(mean)
                        else:
                            action = torch.tanh(torch.normal(mean, std))
                    obs, reward, terminated, truncated, step_info = env.step(
                        action.squeeze(0).cpu().numpy().astype(np.float32)
                    )
                    total_reward += reward
                    if not viewer.is_running():
                        print("viewer_closed", {"episode": episode, "step": step, "total_reward": total_reward})
                        return
                    viewer.sync()
                    if terminated or truncated:
                        print(
                            "done",
                            {
                                "episode": episode,
                                "step": step + 1,
                                "total_reward": total_reward,
                                "episode_success": step_info.get("episode_success"),
                                "terminal_reason": step_info.get("terminal_reason"),
                            },
                        )
                        break
                    if not args.no_realtime:
                        elapsed = time.perf_counter() - loop_start
                        sleep_s = max(0.0, info["control_timestep_s"] - elapsed)
                        if sleep_s > 0.0:
                            time.sleep(sleep_s)
                else:
                    print("partial_rollout", {"episode": episode, "steps": args.steps, "total_reward": total_reward})
    finally:
        env.close()


if __name__ == "__main__":
    main()
