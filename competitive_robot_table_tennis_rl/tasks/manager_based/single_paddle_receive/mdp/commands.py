"""Serve sampling and curriculum selection."""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

from ..env_cfg import CurriculumCfg
from ..utils.serve_dataset import ServeSample


def resolve_bucket_name(difficulty: str, curriculum_level: int) -> str:
    """Resolve the actual dataset bucket for a task difficulty."""

    if difficulty != "curriculum":
        return difficulty
    return {0: "easy", 1: "base"}.get(curriculum_level, "full")


def update_curriculum_level(
    history: deque[bool],
    curriculum_level: int,
    curriculum_cfg: CurriculumCfg,
) -> int:
    """Promote curriculum level based on recent success rate."""

    if len(history) < curriculum_cfg.success_window:
        return curriculum_level
    success_rate = sum(history) / len(history)
    if curriculum_level == 0 and success_rate >= curriculum_cfg.easy_to_base_success_rate:
        return 1
    if curriculum_level == 1 and success_rate >= curriculum_cfg.base_to_full_success_rate:
        return 2
    return curriculum_level


def sample_reset_state(
    rng: np.random.Generator,
    samples_by_id: dict[int, ServeSample],
    candidate_ids: Sequence[int],
) -> ServeSample:
    """Sample a serve state uniformly from a filtered id list."""

    sample_id = int(rng.choice(candidate_ids))
    return samples_by_id[sample_id]
