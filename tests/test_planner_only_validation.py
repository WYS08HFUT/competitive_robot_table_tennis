from __future__ import annotations

import os

import numpy as np
import pytest

from tests.planner_only_eval import assert_planner_only_acceptance, run_planner_only_batch


def test_planner_only_batch_eval_reports_consistent_metrics() -> None:
    metrics = run_planner_only_batch(episodes=2, difficulty="easy", seed_offset=10)
    summary = metrics.as_dict()

    assert metrics.episodes == 2
    assert metrics.total_steps > 0
    assert sum(metrics.terminal_reason_counts.values()) == metrics.episodes
    assert np.isfinite(metrics.mean_episode_reward)
    assert 0.0 <= metrics.planner_valid_reset_rate <= 1.0
    assert 0.0 <= metrics.planner_valid_step_rate <= 1.0
    assert 0.0 <= metrics.legal_hit_rate <= 1.0
    assert 0.0 <= metrics.cross_net_rate <= 1.0
    assert 0.0 <= metrics.opponent_landing_rate <= 1.0
    assert 0.0 <= metrics.floor_rate <= 1.0
    assert 0.0 <= metrics.out_of_bounds_rate <= 1.0
    assert 0.0 <= metrics.timeout_rate <= 1.0
    assert set(summary["terminal_reason_counts"].keys()) == set(metrics.terminal_reason_counts)


def test_planner_only_batch_eval_keeps_nominal_planner_alive_on_easy_bucket() -> None:
    metrics = run_planner_only_batch(episodes=3, difficulty="easy", seed_offset=20)
    summary = metrics.as_dict()

    assert metrics.planner_valid_reset_rate > 0.0, summary
    assert metrics.planner_valid_step_rate > 0.0, summary


def test_planner_only_acceptance_gate_before_ppo() -> None:
    if os.environ.get("RUN_PLANNER_ONLY_GATE") != "1":
        pytest.skip("Set RUN_PLANNER_ONLY_GATE=1 to enforce the planner-only contact gate before PPO.")

    metrics = run_planner_only_batch(episodes=20, difficulty="easy", seed_offset=100)
    assert_planner_only_acceptance(metrics)
