"""Single-environment PPO trainer for the serve-receive task."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np

import competitive_robot_table_tennis_rl  # noqa: F401
from .policy import build_policy, require_torch
from ...utils.paths import prepare_run_dir

torch = None


@dataclass
class TrainCfg:
    task_id: str
    timesteps: int
    seed: int
    lr: float
    hidden_dim: int
    gamma: float
    gae_lambda: float
    clip_coef: float
    value_coef: float
    entropy_coef: float
    rollout_steps: int
    update_epochs: int
    minibatch_size: int
    max_grad_norm: float
    target_kl: float | None
    run_name: str
    device: str


def _atanh_clipped(x, *, eps: float = 1e-6):
    """Stable inverse tanh for bounded actions in [-1, 1]."""

    x = x.clamp(min=-1.0 + eps, max=1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _sample_squashed_action(dist):
    """Sample a tanh-squashed Gaussian action and its corrected log-prob."""

    pre_tanh = dist.rsample()
    action = torch.tanh(pre_tanh)
    log_prob = _squashed_log_prob(dist, pre_tanh, action)
    return action, log_prob


def _squashed_log_prob(dist, pre_tanh, action, *, eps: float = 1e-6):
    """Log-probability of a tanh-squashed Gaussian sample."""

    gaussian_log_prob = dist.log_prob(pre_tanh).sum(dim=-1)
    log_det_jacobian = torch.log(1.0 - action.pow(2) + eps).sum(dim=-1)
    return gaussian_log_prob - log_det_jacobian


def _explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Return explained variance for critic diagnostics."""

    var_y = float(np.var(y_true))
    if var_y <= 1e-8:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)




def _safe_mean(values) -> float:
    """Mean helper that returns 0.0 for empty sequences."""

    return float(np.mean(values)) if values else 0.0


def _log_scalars(writer, prefix: str, metrics: dict[str, float], step: int) -> None:
    """Write a group of scalar metrics to TensorBoard."""

    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", float(value), step)


def _log_prefixed_items(writer, prefix: str, values: dict[str, float], step: int) -> None:
    """Write dynamic scalar dictionaries such as reward terms or terminal counts."""

    for key, value in sorted(values.items()):
        writer.add_scalar(f"{prefix}/{key}", float(value), step)


def _rolling_terminal_reason_rates(reasons: deque[str]) -> dict[str, float]:
    """Return terminal-reason fractions over a rolling window."""

    if not reasons:
        return {}
    counts: defaultdict[str, int] = defaultdict(int)
    for reason in reasons:
        counts[str(reason)] += 1
    total = float(len(reasons))
    return {key: value / total for key, value in counts.items()}


def _suffix_keys(values: dict[str, float], suffix: str) -> dict[str, float]:
    """Return a metric dict with a shared suffix appended to all keys."""

    return {f"{key}{suffix}": value for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO policy on the serve-receive task.")
    parser.add_argument("--task-id", default="TableTennis-SinglePaddleServeReceiveEasy-v0")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--run-name", default="ppo")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    global torch
    torch, _ = require_torch()
    from torch.distributions import Normal
    from torch.utils.tensorboard import SummaryWriter

    cfg = TrainCfg(
        task_id=args.task_id,
        timesteps=args.timesteps,
        seed=args.seed,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        run_name=args.run_name,
        device=args.device,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    env = gym.make(cfg.task_id)
    obs, _ = env.reset(seed=cfg.seed)
    obs_dim = int(np.prod(obs.shape))
    act_dim = int(np.prod(env.action_space.shape))
    policy = build_policy(obs_dim, act_dim, cfg.hidden_dim).to(cfg.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    run_dir = prepare_run_dir(cfg.task_id, cfg.run_name)
    metrics_path = run_dir / "metrics.jsonl"
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    checkpoint_interval_steps = max(1, int(cfg.timesteps * 0.05))
    next_checkpoint_step = checkpoint_interval_steps
    rolling_window = 50
    recent_returns: deque[float] = deque(maxlen=rolling_window)
    recent_lengths: deque[int] = deque(maxlen=rolling_window)
    recent_successes: deque[float] = deque(maxlen=rolling_window)
    recent_terminal_reasons: deque[str] = deque(maxlen=rolling_window)
    writer.add_text(
        "algorithm/name",
        "PPO (single-env actor-critic with Gaussian policy, GAE, clipped surrogate objective)",
        0,
    )
    writer.add_text("algorithm/parallelism", "single-environment sequential rollout collection", 0)
    writer.add_text("run/config", json.dumps(asdict(cfg), indent=2), 0)

    def save_checkpoint(path, *, include_optimizer: bool) -> None:
        checkpoint = {
            "policy_state_dict": policy.state_dict(),
            "train_cfg": asdict(cfg),
            "steps": total_steps,
            "episodes": episode_idx,
            "updates": update_idx,
        }
        if include_optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, path)

    total_steps = 0
    episode_idx = 0
    update_idx = 0

    current_episode_return = 0.0
    current_episode_length = 0
    current_reward_term_sums: defaultdict[str, float] = defaultdict(float)
    current_entropy_values: list[float] = []
    current_action_std_values: list[float] = []
    current_action_abs_values: list[float] = []
    current_action_saturation_values: list[float] = []
    current_policy_mean_abs_values: list[float] = []
    train_wall_start_s = time.perf_counter()

    def reset_episode_accumulators() -> None:
        current_reward_term_sums.clear()
        current_entropy_values.clear()
        current_action_std_values.clear()
        current_action_abs_values.clear()
        current_action_saturation_values.clear()
        current_policy_mean_abs_values.clear()

    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            while total_steps < cfg.timesteps:
                rollout_start_s = time.perf_counter()
                rollout_len = min(cfg.rollout_steps, cfg.timesteps - total_steps)

                obs_buf = np.zeros((rollout_len, obs_dim), dtype=np.float32)
                action_buf = np.zeros((rollout_len, act_dim), dtype=np.float32)
                reward_buf = np.zeros(rollout_len, dtype=np.float32)
                done_buf = np.zeros(rollout_len, dtype=np.float32)
                logprob_buf = np.zeros(rollout_len, dtype=np.float32)
                value_buf = np.zeros(rollout_len, dtype=np.float32)

                rollout_reward_term_sums: defaultdict[str, float] = defaultdict(float)
                rollout_entropy_values: list[float] = []
                rollout_action_std_values: list[float] = []
                rollout_action_abs_values: list[float] = []
                rollout_action_saturation_values: list[float] = []
                rollout_policy_mean_abs_values: list[float] = []

                for step in range(rollout_len):
                    obs_buf[step] = obs
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
                    with torch.no_grad():
                        mean, std = policy(obs_tensor)
                        value = policy.value(obs_tensor)
                        dist = Normal(mean, std)
                        env_action, log_prob = _sample_squashed_action(dist)

                    action_buf[step] = env_action.squeeze(0).cpu().numpy().astype(np.float32)
                    value_buf[step] = float(value.squeeze(0).cpu())
                    logprob_buf[step] = float(log_prob.squeeze(0).cpu())

                    entropy_value = float(dist.entropy().sum(dim=-1).mean().cpu())
                    action_std_value = float(std.mean().cpu())
                    action_abs_value = float(env_action.abs().mean().cpu())
                    action_saturation_value = float((env_action.abs() > 0.95).float().mean().cpu())
                    policy_mean_abs_value = float(mean.abs().mean().cpu())

                    rollout_entropy_values.append(entropy_value)
                    rollout_action_std_values.append(action_std_value)
                    rollout_action_abs_values.append(action_abs_value)
                    rollout_action_saturation_values.append(action_saturation_value)
                    rollout_policy_mean_abs_values.append(policy_mean_abs_value)

                    current_entropy_values.append(entropy_value)
                    current_action_std_values.append(action_std_value)
                    current_action_abs_values.append(action_abs_value)
                    current_action_saturation_values.append(action_saturation_value)
                    current_policy_mean_abs_values.append(policy_mean_abs_value)

                    next_obs, reward, terminated, truncated, info = env.step(
                        env_action.squeeze(0).cpu().numpy().astype(np.float32)
                    )

                    reward_buf[step] = float(reward)
                    done_buf[step] = float(terminated or truncated)
                    current_episode_return += float(reward)
                    current_episode_length += 1
                    total_steps += 1

                    for key, value in info.get("reward_terms", {}).items():
                        reward_value = float(value)
                        current_reward_term_sums[key] += reward_value
                        rollout_reward_term_sums[key] += reward_value

                    obs = next_obs

                    if terminated or truncated:
                        terminal_reason = str(info.get("terminal_reason", "unknown"))
                        success = bool(info.get("episode_success", False))
                        recent_returns.append(current_episode_return)
                        recent_lengths.append(current_episode_length)
                        recent_successes.append(float(success))
                        recent_terminal_reasons.append(terminal_reason)

                        record = {
                            "episode": episode_idx,
                            "steps": total_steps,
                            "episode_length": current_episode_length,
                            "episode_return": current_episode_return,
                            "success": success,
                            "terminal_reason": terminal_reason,
                            "rolling_return_mean": float(np.mean(recent_returns)),
                            "rolling_success_rate": float(np.mean(recent_successes)),
                            "rolling_length_mean": float(np.mean(recent_lengths)),
                            "policy_entropy_mean": float(np.mean(current_entropy_values)) if current_entropy_values else 0.0,
                            "action_std_mean": float(np.mean(current_action_std_values)) if current_action_std_values else 0.0,
                            "action_abs_mean": float(np.mean(current_action_abs_values)) if current_action_abs_values else 0.0,
                            "action_saturation_frac": float(np.mean(current_action_saturation_values)) if current_action_saturation_values else 0.0,
                            "policy_mean_abs": float(np.mean(current_policy_mean_abs_values)) if current_policy_mean_abs_values else 0.0,
                            "reward_terms": dict(sorted(current_reward_term_sums.items())),
                        }
                        metrics_file.write(json.dumps(record) + "\n")
                        metrics_file.flush()
                        print(record)

                        perf_metrics = {
                            "episode_return": current_episode_return,
                            "return_mean_50": _safe_mean(recent_returns),
                            "episode_length_mean_50": _safe_mean(recent_lengths),
                            "success_rate_50": _safe_mean(recent_successes),
                        }

                        _log_scalars(writer, "Perf", perf_metrics, total_steps)
                        _log_prefixed_items(writer, "Env/reward", current_reward_term_sums, total_steps)
                        _log_prefixed_items(
                            writer,
                            "Env/termination",
                            _suffix_keys(_rolling_terminal_reason_rates(recent_terminal_reasons), "_rate_50"),
                            total_steps,
                        )

                        episode_idx += 1
                        current_episode_return = 0.0
                        current_episode_length = 0
                        reset_episode_accumulators()
                        obs, _ = env.reset()

                next_value = 0.0
                if done_buf[rollout_len - 1] == 0.0:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
                    with torch.no_grad():
                        next_value = float(policy.value(obs_tensor).squeeze(0).cpu())

                advantages = np.zeros(rollout_len, dtype=np.float32)
                last_gae = 0.0
                for step in reversed(range(rollout_len)):
                    if step == rollout_len - 1:
                        next_nonterminal = 1.0 - done_buf[step]
                        next_val = next_value
                    else:
                        next_nonterminal = 1.0 - done_buf[step]
                        next_val = value_buf[step + 1]
                    delta = reward_buf[step] + cfg.gamma * next_val * next_nonterminal - value_buf[step]
                    last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
                    advantages[step] = last_gae
                returns = advantages + value_buf

                b_obs = torch.as_tensor(obs_buf, dtype=torch.float32, device=cfg.device)
                b_actions = torch.as_tensor(action_buf, dtype=torch.float32, device=cfg.device)
                b_old_logprobs = torch.as_tensor(logprob_buf, dtype=torch.float32, device=cfg.device)
                b_advantages = torch.as_tensor(advantages, dtype=torch.float32, device=cfg.device)
                b_returns = torch.as_tensor(returns, dtype=torch.float32, device=cfg.device)
                b_values = torch.as_tensor(value_buf, dtype=torch.float32, device=cfg.device)

                adv_mean = b_advantages.mean()
                adv_std = b_advantages.std(unbiased=False)
                b_advantages = (b_advantages - adv_mean) / (adv_std + 1e-8)

                policy_loss_values = []
                value_loss_values = []
                entropy_loss_values = []
                total_loss_values = []
                approx_kl_values = []
                clipfrac_values = []
                grad_norm_values = []
                raw_returns_mean = float(np.mean(returns))
                raw_returns_std = float(np.std(returns))
                early_stop_triggered = False

                batch_size = rollout_len
                minibatch_size = min(cfg.minibatch_size, batch_size)
                for _ in range(cfg.update_epochs):
                    permutation = np.random.permutation(batch_size)
                    for start in range(0, batch_size, minibatch_size):
                        idx = permutation[start : start + minibatch_size]
                        mb_obs = b_obs[idx]
                        mb_actions = b_actions[idx]
                        mb_old_logprobs = b_old_logprobs[idx]
                        mb_advantages = b_advantages[idx]
                        mb_returns = b_returns[idx]
                        mb_old_values = b_values[idx]

                        mean, std = policy(mb_obs)
                        dist = Normal(mean, std)
                        mb_pre_tanh = _atanh_clipped(mb_actions)
                        new_logprob = _squashed_log_prob(dist, mb_pre_tanh, mb_actions)
                        entropy = dist.entropy().sum(dim=-1).mean()
                        new_values = policy.value(mb_obs)

                        log_ratio = new_logprob - mb_old_logprobs
                        ratio = log_ratio.exp()
                        surrogate_1 = ratio * mb_advantages
                        surrogate_2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * mb_advantages
                        policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                        value_loss_unclipped = (new_values - mb_returns).pow(2)
                        value_clipped = mb_old_values + torch.clamp(new_values - mb_old_values, -cfg.clip_coef, cfg.clip_coef)
                        value_loss_clipped = (value_clipped - mb_returns).pow(2)
                        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                        entropy_loss = entropy
                        total_loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss

                        optimizer.zero_grad()
                        total_loss.backward()
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm).cpu())
                        optimizer.step()

                        approx_kl = float(((ratio - 1.0) - log_ratio).mean().detach().cpu())
                        clipfrac = float((torch.abs(ratio - 1.0) > cfg.clip_coef).float().mean().detach().cpu())

                        policy_loss_values.append(float(policy_loss.detach().cpu()))
                        value_loss_values.append(float(value_loss.detach().cpu()))
                        entropy_loss_values.append(float(entropy_loss.detach().cpu()))
                        total_loss_values.append(float(total_loss.detach().cpu()))
                        approx_kl_values.append(approx_kl)
                        clipfrac_values.append(clipfrac)
                        grad_norm_values.append(grad_norm)

                    if cfg.target_kl is not None and approx_kl_values:
                        if float(np.mean(approx_kl_values)) > cfg.target_kl:
                            early_stop_triggered = True
                            break
                with torch.no_grad():
                    critic_values = policy.value(b_obs).detach().cpu().numpy()
                explained_variance = _explained_variance(critic_values, returns)

                param_sq_sum = 0.0
                for param in policy.parameters():
                    param_sq_sum += float(param.detach().pow(2).sum().cpu())
                param_norm = param_sq_sum**0.5
                rollout_duration_s = max(1e-6, time.perf_counter() - rollout_start_s)
                steps_per_second = rollout_len / rollout_duration_s
                train_elapsed_s = max(1e-6, time.perf_counter() - train_wall_start_s)

                loss_metrics = {
                    "policy": _safe_mean(policy_loss_values),
                    "value": _safe_mean(value_loss_values),
                    "entropy": _safe_mean(entropy_loss_values),
                    "total": _safe_mean(total_loss_values),
                    "kl": _safe_mean(approx_kl_values),
                    "clip_fraction": _safe_mean(clipfrac_values),
                }
                policy_metrics = {
                    "entropy_mean": _safe_mean(rollout_entropy_values),
                    "action_std_mean": _safe_mean(rollout_action_std_values),
                    "action_saturation_frac": _safe_mean(rollout_action_saturation_values),
                }
                train_metrics = {
                    "mean_step_reward": float(np.mean(reward_buf)),
                    "fps": steps_per_second,
                    "done_fraction": float(np.mean(done_buf)),
                    "grad_norm": _safe_mean(grad_norm_values),
                    "explained_variance": explained_variance,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                _log_scalars(writer, "Loss", loss_metrics, update_idx)
                _log_scalars(writer, "Policy", policy_metrics, update_idx)
                _log_scalars(writer, "Train", train_metrics, update_idx)
                _log_prefixed_items(writer, "Env/reward_rollout", rollout_reward_term_sums, update_idx)

                while total_steps >= next_checkpoint_step and next_checkpoint_step <= cfg.timesteps:
                    checkpoint_name = f"policy_step_{next_checkpoint_step:08d}.pt"
                    save_checkpoint(run_dir / checkpoint_name, include_optimizer=True)
                    next_checkpoint_step += checkpoint_interval_steps

                update_idx += 1
    finally:
        writer.close()
        save_checkpoint(run_dir / "policy.pt", include_optimizer=True)
        print({"run_dir": str(run_dir), "episodes": episode_idx, "steps": total_steps, "updates": update_idx})
        env.close()


if __name__ == "__main__":
    main()
