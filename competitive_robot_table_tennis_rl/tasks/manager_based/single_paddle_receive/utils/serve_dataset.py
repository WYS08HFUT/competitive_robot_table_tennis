"""Serve dataset loading and filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable

import numpy as np

from ..env_cfg import DatasetCfg
from ..mdp.transforms import predict_post_bounce_intercept_at_y, speed_norm
from .paths import resolve_repo_path


@dataclass(frozen=True)
class ServeSample:
    id: int
    pos_x: float
    pos_y: float
    pos_z: float
    vel_x: float
    vel_y: float
    vel_z: float
    w_vel_x: float
    w_vel_y: float
    w_vel_z: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.pos_x, self.pos_y, self.pos_z], dtype=np.float64)

    @property
    def linear_velocity(self) -> np.ndarray:
        return np.array([self.vel_x, self.vel_y, self.vel_z], dtype=np.float64)

    @property
    def angular_velocity(self) -> np.ndarray:
        return np.array([self.w_vel_x, self.w_vel_y, self.w_vel_z], dtype=np.float64)

    @property
    def spin_norm(self) -> float:
        return sqrt(self.w_vel_x**2 + self.w_vel_y**2 + self.w_vel_z**2)


def load_serves(path: str | Path) -> list[ServeSample]:
    """Load serve samples from JSON."""

    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("Serve dataset must be a list of objects")
    samples: list[ServeSample] = []
    required = {
        "id",
        "pos_x",
        "pos_y",
        "pos_z",
        "vel_x",
        "vel_y",
        "vel_z",
        "w_vel_x",
        "w_vel_y",
        "w_vel_z",
    }
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("Serve entry must be a JSON object")
        missing = required.difference(entry)
        if missing:
            raise ValueError(f"Serve entry missing keys: {sorted(missing)}")
        samples.append(ServeSample(**{key: entry[key] for key in required}))
    return sorted(samples, key=lambda sample: sample.id)


def _intercept_in_workspace(
    sample: ServeSample,
    dataset_cfg: DatasetCfg,
    workspace_x: tuple[float, float],
    workspace_y: tuple[float, float],
    workspace_z: tuple[float, float],
) -> bool:
    y_target = max(workspace_y[0], min(-0.65, workspace_y[1]))
    intercept = predict_post_bounce_intercept_at_y(
        sample.position,
        sample.linear_velocity,
        y_target=y_target,
        max_time_s=dataset_cfg.max_intercept_horizon_s,
        restitution_z=dataset_cfg.table_bounce_restitution_z,
        damping_xy=dataset_cfg.table_bounce_damping_xy,
    )
    if intercept is None:
        return False
    x, y, z, _ = intercept
    return (
        workspace_x[0] <= x <= workspace_x[1]
        and workspace_y[0] <= y <= workspace_y[1]
        and workspace_z[0] <= z <= workspace_z[1]
    )


def filter_serves(
    samples: Iterable[ServeSample],
    difficulty: str,
    dataset_cfg: DatasetCfg,
    workspace_x: tuple[float, float],
    workspace_y: tuple[float, float],
    workspace_z: tuple[float, float],
) -> list[int]:
    """Return serve ids allowed for a given difficulty bucket."""

    allowed: list[int] = []
    for sample in samples:
        if sample.pos_y <= 0.0 or sample.vel_y >= 0.0:
            continue
        if not _intercept_in_workspace(sample, dataset_cfg, workspace_x, workspace_y, workspace_z):
            continue

        speed = speed_norm(sample.linear_velocity)
        if difficulty == "easy":
            if not (dataset_cfg.easy_speed_range_m_s[0] <= speed <= dataset_cfg.easy_speed_range_m_s[1]):
                continue
            if not (dataset_cfg.easy_height_range_m[0] <= sample.pos_z <= dataset_cfg.easy_height_range_m[1]):
                continue
            if sample.spin_norm > dataset_cfg.easy_spin_max_rad_s:
                continue
        elif difficulty == "base":
            if sample.spin_norm > dataset_cfg.base_spin_max_rad_s:
                continue
        elif difficulty != "full":
            raise ValueError(f"Unknown difficulty bucket: {difficulty}")
        allowed.append(sample.id)
    if not allowed:
        raise ValueError(f"No serve samples survived filtering for difficulty={difficulty}")
    return allowed
