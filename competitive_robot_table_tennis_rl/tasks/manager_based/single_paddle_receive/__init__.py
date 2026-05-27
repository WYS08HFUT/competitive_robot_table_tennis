"""Single-paddle serve-receive task package."""

from .env_cfg import TASK_IDS, TaskCfg, make_task_cfg
try:
    from .env import SinglePaddleServeReceiveEnv
    from .registry import register_tasks
except ModuleNotFoundError as exc:
    if exc.name != "gymnasium":
        raise
    SinglePaddleServeReceiveEnv = None

    def register_tasks() -> None:
        """Allow planner imports when Gym dependencies are unavailable."""

        return None

register_tasks()

__all__ = [
    "SinglePaddleServeReceiveEnv",
    "TASK_IDS",
    "TaskCfg",
    "make_task_cfg",
    "register_tasks",
]
