"""Policy and value networks for the LunarLander REINFORCE agents.

The ``Policy`` network follows the architecture from the slides (Exercise 3):
    State -> FC1 -> ReLU -> FC2 (hidden*2) -> ReLU -> FC3 -> softmax

``PolicyWithValue`` pairs that policy with a *separate* value network used by the
REINFORCE-with-baseline agent to estimate the state-value V(s) and reduce
gradient variance. The two networks are kept independent on purpose: the value
function is trained on the (large-magnitude) raw returns, so sharing a trunk
would let the value loss dominate and corrupt the policy features.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    """Vanilla policy network (matches the slide architecture)."""

    def __init__(self, s_size: int, a_size: int, h_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size * 2)
        self.fc3 = nn.Linear(h_size * 2, a_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state: np.ndarray, device: torch.device):
        """Sample an action from the policy distribution.

        Returns the chosen action and the log-probability of that action.
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state_t).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    @torch.no_grad()
    def act_greedy(self, state: np.ndarray, device: torch.device) -> int:
        """Pick the most likely action (used for deterministic evaluation)."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state_t).cpu()
        return int(torch.argmax(probs, dim=1).item())


class PolicyWithValue(nn.Module):
    """Policy network plus an independent value network.

    Used by the REINFORCE-with-baseline agent. The value network V(s) is trained
    to predict the discounted return G_t and is subtracted from the return to
    form the advantage A_t = G_t - V(s_t), which lowers the variance of the
    gradient estimate without introducing bias (Sutton & Barto, 13.4).

    Policy and value use *separate* parameters so the value loss (computed on
    raw, large-magnitude returns) cannot overwhelm the policy gradient.
    """

    def __init__(self, s_size: int, a_size: int, h_size: int = 64):
        super().__init__()
        # Policy network (same architecture as ``Policy``).
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size * 2)
        self.policy_head = nn.Linear(h_size * 2, a_size)
        # Independent value network V(s).
        self.v_fc1 = nn.Linear(s_size, h_size)
        self.v_fc2 = nn.Linear(h_size, h_size * 2)
        self.value_head = nn.Linear(h_size * 2, 1)

    def forward(self, x: torch.Tensor):
        p = F.relu(self.fc1(x))
        p = F.relu(self.fc2(p))
        probs = F.softmax(self.policy_head(p), dim=1)
        v = F.relu(self.v_fc1(x))
        v = F.relu(self.v_fc2(v))
        value = self.value_head(v)
        return probs, value

    def act(self, state: np.ndarray, device: torch.device):
        """Sample an action; return action, log-prob and the value estimate."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs, value = self.forward(state_t)
        probs = probs.cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), value.squeeze(0)

    @torch.no_grad()
    def act_greedy(self, state: np.ndarray, device: torch.device) -> int:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs, _ = self.forward(state_t)
        return int(torch.argmax(probs.cpu(), dim=1).item())
