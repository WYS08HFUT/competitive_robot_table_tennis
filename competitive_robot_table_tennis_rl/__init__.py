"""Competitive robot table tennis RL package."""

try:
    from .tasks.manager_based.single_paddle_receive.registry import register_tasks
except ModuleNotFoundError as exc:
    if exc.name != "gymnasium":
        raise

    def register_tasks() -> None:
        """Allow lightweight planner imports without the Gym runtime."""

        return None

register_tasks()

__all__ = ["register_tasks"]
