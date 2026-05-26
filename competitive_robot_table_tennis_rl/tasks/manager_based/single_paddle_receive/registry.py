"""Gymnasium task registry for the single-paddle receive task."""

from __future__ import annotations

import gymnasium as gym

from .env_cfg import TASK_IDS


def register_tasks() -> None:
    """Register all single-paddle serve-receive task variants once."""

    specs = {
        TASK_IDS.base: {"difficulty": "base"},
        TASK_IDS.easy: {"difficulty": "easy"},
        TASK_IDS.curriculum: {"difficulty": "curriculum"},
    }
    for task_id, kwargs in specs.items():
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point=(
                "competitive_robot_table_tennis_rl.tasks.manager_based."
                "single_paddle_receive.env:SinglePaddleServeReceiveEnv"
            ),
            kwargs=kwargs,
        )
