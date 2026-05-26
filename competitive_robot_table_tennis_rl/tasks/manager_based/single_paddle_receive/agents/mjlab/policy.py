"""Policy helpers for MJLab-namespaced training and playback."""

from __future__ import annotations

from pathlib import Path


def require_torch():
    """Import torch modules with a task-specific error."""

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "Torch is required for policy training/playback. "
            "Install requirements-train.txt inside the mujoco conda env."
        ) from exc
    return torch, nn


def build_policy(obs_dim: int, act_dim: int, hidden_dim: int):
    """Build the feed-forward Gaussian actor-critic policy network."""

    torch, nn = require_torch()

    class Policy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, act_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        def forward(self, x):
            mean = self.actor(x)
            std = torch.exp(self.log_std).expand_as(mean)
            return mean, std

        def value(self, x):
            return self.critic(x).squeeze(-1)

    return Policy()


def load_policy_checkpoint(checkpoint_path: str | Path, obs_dim: int, act_dim: int):
    """Load a saved policy checkpoint and rebuild the matching network."""

    torch, _ = require_torch()
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    train_cfg = checkpoint["train_cfg"]
    policy = build_policy(obs_dim, act_dim, int(train_cfg["hidden_dim"]))
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy, train_cfg
