"""Competitive robot table tennis RL package."""

from .tasks.manager_based.single_paddle_receive.registry import register_tasks

register_tasks()

__all__ = ["register_tasks"]
