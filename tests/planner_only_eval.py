"""Helpers for planner-only batch validation on the receive task."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.env import (
    SinglePaddleServeReceiveEnv,
)
from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.env_cfg import (
    TaskCfg,
    make_task_cfg,
)


@dataclass(frozen=True)
class PlannerOnlyEvalMetrics:
    """Aggregated planner-only rollout metrics."""

    episodes: int
    total_steps: int
    mean_episode_reward: float
    planner_valid_reset_rate: float
    planner_valid_step_rate: float
    legal_hit_rate: float
    cross_net_rate: float
    opponent_landing_rate: float
    floor_rate: float
    out_of_bounds_rate: float
    timeout_rate: float
    terminal_reason_counts: dict[str, int]

    def as_dict(self) -> dict[str, float | int | dict[str, int]]:
        """Return a JSON-like summary for debugging output."""

        return {
            "episodes": self.episodes,
            "total_steps": self.total_steps,
            "mean_episode_reward": self.mean_episode_reward,
            "planner_valid_reset_rate": self.planner_valid_reset_rate,
            "planner_valid_step_rate": self.planner_valid_step_rate,
            "legal_hit_rate": self.legal_hit_rate,
            "cross_net_rate": self.cross_net_rate,
            "opponent_landing_rate": self.opponent_landing_rate,
            "floor_rate": self.floor_rate,
            "out_of_bounds_rate": self.out_of_bounds_rate,
            "timeout_rate": self.timeout_rate,
            "terminal_reason_counts": dict(self.terminal_reason_counts),
        }


@dataclass(frozen=True)
class PlannerOnlyAcceptanceGate:
    """Recommended easy-bucket contact gate before PPO residual training."""

    min_planner_valid_reset_rate: float = 0.95
    min_planner_valid_step_rate: float = 0.65
    min_legal_hit_rate: float = 0.25
    max_floor_rate: float = 0.75
    max_out_of_bounds_rate: float = 0.75


def make_planner_only_cfg(difficulty: str = "easy") -> TaskCfg:
    """Return a task config with the nominal planner driving the paddle alone."""

    cfg = make_task_cfg(difficulty)
    return replace(cfg, control=replace(cfg.control, control_mode="planner_only"))


def run_planner_only_batch(
    *,
    episodes: int,
    difficulty: str = "easy",
    seed_offset: int = 0,
) -> PlannerOnlyEvalMetrics:
    """Roll out the task with zero residual action and summarize planner quality."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")

    cfg = make_planner_only_cfg(difficulty)
    env = SinglePaddleServeReceiveEnv(difficulty=difficulty, cfg=cfg)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    planner_valid_reset_count = 0
    planner_valid_step_count = 0
    total_steps = 0
    legal_hits = 0
    cross_nets = 0
    opponent_landings = 0
    episode_reward_sum = 0.0
    terminal_reason_counts: Counter[str] = Counter()

    try:
        for episode_idx in range(episodes):
            _, info = env.reset(seed=seed_offset + episode_idx)
            planner_valid_reset_count += int(info.get("planner_valid", False))

            episode_reward = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                _, reward, terminated, truncated, step_info = env.step(zero_action)
                total_steps += 1
                planner_valid_step_count += int(step_info.get("planner_valid", False))
                episode_reward += float(reward)

            legal_hits += int(env.state.has_hit)
            cross_nets += int(env.state.crossed_net_after_hit)
            opponent_landings += int(env.state.success)
            episode_reward_sum += episode_reward
            terminal_reason_counts[str(step_info.get("terminal_reason", "unknown"))] += 1
    finally:
        env.close()

    floor_count = terminal_reason_counts.get("ball_floor", 0)
    out_of_bounds_count = terminal_reason_counts.get("ball_out_of_bounds", 0)
    timeout_count = terminal_reason_counts.get("timeout", 0)

    return PlannerOnlyEvalMetrics(
        episodes=episodes,
        total_steps=total_steps,
        mean_episode_reward=episode_reward_sum / float(episodes),
        planner_valid_reset_rate=planner_valid_reset_count / float(episodes),
        planner_valid_step_rate=planner_valid_step_count / float(max(total_steps, 1)),
        legal_hit_rate=legal_hits / float(episodes),
        cross_net_rate=cross_nets / float(episodes),
        opponent_landing_rate=opponent_landings / float(episodes),
        floor_rate=floor_count / float(episodes),
        out_of_bounds_rate=out_of_bounds_count / float(episodes),
        timeout_rate=timeout_count / float(episodes),
        terminal_reason_counts=dict(terminal_reason_counts),
    )


def assert_planner_only_acceptance(
    metrics: PlannerOnlyEvalMetrics,
    gate: PlannerOnlyAcceptanceGate | None = None,
) -> None:
    """Assert the recommended easy-bucket planner-only contact gate before PPO."""

    gate = PlannerOnlyAcceptanceGate() if gate is None else gate
    summary = metrics.as_dict()

    assert metrics.planner_valid_reset_rate >= gate.min_planner_valid_reset_rate, summary
    assert metrics.planner_valid_step_rate >= gate.min_planner_valid_step_rate, summary
    assert metrics.legal_hit_rate >= gate.min_legal_hit_rate, summary
    assert metrics.floor_rate <= gate.max_floor_rate, summary
    assert metrics.out_of_bounds_rate <= gate.max_out_of_bounds_rate, summary
