"""Single-paddle serve-receive task package."""

from .env import SinglePaddleServeReceiveEnv
from .env_cfg import TASK_IDS, TaskCfg, make_task_cfg
from .registry import register_tasks

register_tasks()

__all__ = [
    "SinglePaddleServeReceiveEnv",
    "TASK_IDS",
    "TaskCfg",
    "make_task_cfg",
    "register_tasks",
]
