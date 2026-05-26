"""Dataset validation and serve playback helper."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np

import competitive_robot_table_tennis_rl  # noqa: F401
from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.env_cfg import DatasetCfg
from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.utils.serve_dataset import filter_serves, load_serves


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and play dataset serves inside the receive task.")
    parser.add_argument("--task-id", default="TableTennis-SinglePaddleServeReceiveEasy-v0")
    parser.add_argument("--serve-id", type=int, default=None)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--difficulty", choices=("easy", "base", "full"), default="easy")
    parser.add_argument("--save-video", default=None, help="Optional .mp4 or .gif output path for rollout playback.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--mujoco-gl",
        default=None,
        help="Optional MuJoCo GL backend override. Example: glfw for interactive desktop, egl for headless export.",
    )
    parser.add_argument(
        "--show-contact-points",
        action="store_true",
        help="Show MuJoCo contact points in the interactive viewer.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Run the interactive viewer as fast as possible instead of sleeping to match control timestep.",
    )
    args = parser.parse_args()
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl

    dataset_cfg = DatasetCfg()
    serves = load_serves("serves.json")
    bucket_ids = filter_serves(
        serves,
        difficulty=args.difficulty,
        dataset_cfg=dataset_cfg,
        workspace_x=(-0.60, 0.60),
        workspace_y=(-1.25, -0.05),
        workspace_z=(0.05, 0.60),
    )
    print(
        "dataset_summary",
        {
            "total_serves": len(serves),
            "difficulty": args.difficulty,
            "filtered_count": len(bucket_ids),
            "first_ids": bucket_ids[:10],
        },
    )

    serve_id = args.serve_id if args.serve_id is not None else int(bucket_ids[args.seed % len(bucket_ids)])
    sample = next(sample for sample in serves if sample.id == serve_id)
    print(
        "selected_serve",
        {
            "id": sample.id,
            "position": sample.position.tolist(),
            "linear_velocity": sample.linear_velocity.tolist(),
            "angular_velocity": sample.angular_velocity.tolist(),
            "spin_norm": sample.spin_norm,
        },
    )

    render_mode = "rgb_array" if args.save_video else None
    env = gym.make(args.task_id, render_mode=render_mode)
    obs, info = env.reset(seed=args.seed, options={"serve_id": serve_id})
    print("reset", {"obs_shape": tuple(obs.shape), **info})
    frames: list[np.ndarray] = []
    video_path = Path(args.save_video).expanduser() if args.save_video else None

    def maybe_capture_frame() -> bool:
        try:
            frame = env.render()
        except RuntimeError as exc:
            print("render_unavailable", str(exc))
            return False
        if video_path is not None:
            frames.append(frame)
        elif not frames:
            print("frame_shape", frame.shape)
        return True

    if video_path is not None:
        maybe_capture_frame()
        total_reward = 0.0
        for step in range(args.steps):
            obs, reward, terminated, truncated, step_info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32)
            )
            del obs
            total_reward += reward
            if step == 0:
                print(
                    "step0_state",
                    {
                        "ball_pos": env.unwrapped.ball_pos.tolist(),
                        "ball_vel": env.unwrapped.ball_vel.tolist(),
                        "paddle_qpos": env.unwrapped.paddle_qpos.tolist(),
                        "paddle_pos": env.unwrapped.paddle_pos.tolist(),
                    },
                )
            if not maybe_capture_frame():
                break
            if terminated or truncated:
                print(
                    "done",
                    {
                        "step": step + 1,
                        "total_reward": total_reward,
                        "terminal_reason": step_info.get("terminal_reason"),
                        "episode_success": step_info.get("episode_success"),
                    },
                )
                break
        else:
            print("partial_rollout", {"steps": args.steps, "total_reward": total_reward})
    else:
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
                if step == 0:
                    print(
                        "step0_state",
                        {
                            "ball_pos": env.unwrapped.ball_pos.tolist(),
                            "ball_vel": env.unwrapped.ball_vel.tolist(),
                            "paddle_qpos": env.unwrapped.paddle_qpos.tolist(),
                            "paddle_pos": env.unwrapped.paddle_pos.tolist(),
                        },
                    )
                if not viewer.is_running():
                    print("viewer_closed", {"step": step, "total_reward": total_reward})
                    break
                viewer.sync()
                if terminated or truncated:
                    print(
                        "done",
                        {
                            "step": step + 1,
                            "total_reward": total_reward,
                            "terminal_reason": step_info.get("terminal_reason"),
                            "episode_success": step_info.get("episode_success"),
                        },
                    )
                    break
                if not args.no_realtime:
                    elapsed = time.perf_counter() - loop_start
                    sleep_s = max(0.0, info["control_timestep_s"] - elapsed)
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
            else:
                print("partial_rollout", {"steps": args.steps, "total_reward": total_reward})

    env.close()
    if video_path is not None and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=args.fps)
        print("saved_video", {"path": str(video_path), "frames": len(frames), "fps": args.fps})


if __name__ == "__main__":
    main()
