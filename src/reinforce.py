"""REINFORCE (Monte-Carlo Policy Gradient) training algorithms.

Two variants are implemented:

1. ``reinforce`` — the vanilla algorithm exactly as presented in the slides
   (Exercise 3). For every episode we collect a trajectory, compute the
   discounted returns G_t, normalise them, and minimise
   ``loss = - sum_t log pi(a_t|s_t) * G_t``.

2. ``reinforce_with_baseline`` — the same Monte-Carlo policy gradient but the
   return is replaced by the *advantage* ``A_t = G_t - V(s_t)``, where V is a
   learned value baseline. The value network is trained on the **raw** returns
   ``G_t`` (stationary targets) while only the *advantages* are normalised for
   the policy gradient. This keeps the estimator unbiased while reducing its
   variance, which usually means faster and more stable learning.

Both functions keep the best policy seen during training (by rolling avg-100)
and reload it before returning, so evaluation reflects the best checkpoint
rather than a possibly-degraded final one (REINFORCE is high-variance).

Both functions share the same signature/return type so the Gradio app and the
training script can treat them interchangeably.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .policy import Policy, PolicyWithValue


@dataclass
class TrainResult:
    """Container for everything produced during a training run."""

    scores: List[float] = field(default_factory=list)          # return per episode
    avg_scores: List[float] = field(default_factory=list)      # rolling mean (100)
    episode_lengths: List[int] = field(default_factory=list)   # steps per episode
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    algo: str = "reinforce"
    best_avg: float = float("-inf")          # best rolling avg-100 reached
    solved_episode: Optional[int] = None     # first episode with avg-100 >= 200


def _discounted_returns(rewards: List[float], gamma: float, max_steps: int) -> deque:
    """Compute G_t for every timestep using the slide's appendleft trick."""
    returns: deque = deque(maxlen=max_steps)
    n_steps = len(rewards)
    for t in range(n_steps)[::-1]:
        disc_return_t = returns[0] if len(returns) > 0 else 0.0
        returns.appendleft(gamma * disc_return_t + rewards[t])
    return returns


def _track_best(result: TrainResult, policy, i_episode: int,
                best_state: Optional[dict]) -> Optional[dict]:
    """Snapshot the policy when it reaches a new best rolling avg-100, and record
    the first episode that crosses the ``solved`` threshold (avg-100 >= 200)."""
    avg = result.avg_scores[-1]
    if avg > result.best_avg:
        result.best_avg = avg
        best_state = deepcopy(policy.state_dict())
    if result.solved_episode is None and avg >= 200.0:
        result.solved_episode = i_episode
    return best_state


def reinforce(
    policy: Policy,
    optimizer: optim.Optimizer,
    env,
    n_training_episodes: int,
    max_steps: int,
    gamma: float,
    device: torch.device,
    print_every: int = 100,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> TrainResult:
    """Vanilla REINFORCE, faithful to the slides (with gymnasium step API)."""
    result = TrainResult(algo="reinforce")
    scores_deque: deque = deque(maxlen=100)
    best_state: Optional[dict] = None

    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        state, _ = env.reset()

        for _ in range(max_steps):
            action, log_prob = policy.act(state, device)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break

        total_reward = float(sum(rewards))
        scores_deque.append(total_reward)
        result.scores.append(total_reward)
        result.avg_scores.append(float(np.mean(scores_deque)))
        result.episode_lengths.append(len(rewards))

        returns = _discounted_returns(rewards, gamma, max_steps)
        eps = np.finfo(np.float32).eps.item()
        returns_t = torch.tensor(list(returns), dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + eps)

        policy_loss = [-log_prob * R for log_prob, R in zip(saved_log_probs, returns_t)]
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        result.policy_losses.append(float(policy_loss.item()))
        result.value_losses.append(0.0)
        best_state = _track_best(result, policy, i_episode, best_state)

        if progress_cb is not None:
            progress_cb(i_episode, n_training_episodes, result.avg_scores[-1])
        if print_every and i_episode % print_every == 0:
            print(f"[REINFORCE] Episode {i_episode}\tAverage Score: {result.avg_scores[-1]:.2f}")

    if best_state is not None:
        policy.load_state_dict(best_state)
    return result


def reinforce_with_baseline(
    policy: PolicyWithValue,
    optimizer: optim.Optimizer,
    env,
    n_training_episodes: int,
    max_steps: int,
    gamma: float,
    device: torch.device,
    print_every: int = 100,
    value_coef: float = 0.5,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> TrainResult:
    """REINFORCE with a learned value baseline (advantage = G_t - V(s_t))."""
    result = TrainResult(algo="reinforce_baseline")
    scores_deque: deque = deque(maxlen=100)
    best_state: Optional[dict] = None

    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs: List[torch.Tensor] = []
        saved_values: List[torch.Tensor] = []
        rewards: List[float] = []
        state, _ = env.reset()

        for _ in range(max_steps):
            action, log_prob, value = policy.act(state, device)
            saved_log_probs.append(log_prob)
            saved_values.append(value)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                break

        total_reward = float(sum(rewards))
        scores_deque.append(total_reward)
        result.scores.append(total_reward)
        result.avg_scores.append(float(np.mean(scores_deque)))
        result.episode_lengths.append(len(rewards))

        returns = _discounted_returns(rewards, gamma, max_steps)
        eps = np.finfo(np.float32).eps.item()
        returns_t = torch.tensor(list(returns), dtype=torch.float32)

        values_t = torch.cat(saved_values).to("cpu")
        # Value network is trained to predict the RAW return G_t (a stationary
        # target). The advantage is G_t - V(s_t) with the baseline detached, then
        # normalised so the policy gradient stays well-scaled and low-variance.
        value_loss = F.mse_loss(values_t, returns_t)
        advantages = returns_t - values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

        policy_loss = torch.cat(
            [-log_prob * A for log_prob, A in zip(saved_log_probs, advantages)]
        ).sum()
        loss = policy_loss + value_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        result.policy_losses.append(float(policy_loss.item()))
        result.value_losses.append(float(value_loss.item()))
        best_state = _track_best(result, policy, i_episode, best_state)

        if progress_cb is not None:
            progress_cb(i_episode, n_training_episodes, result.avg_scores[-1])
        if print_every and i_episode % print_every == 0:
            print(
                f"[REINFORCE+Baseline] Episode {i_episode}\t"
                f"Average Score: {result.avg_scores[-1]:.2f}"
            )

    if best_state is not None:
        policy.load_state_dict(best_state)
    return result
