"""Shared hyper-parameters and helpers."""

from __future__ import annotations

import torch

ENV_ID = "LunarLander-v3"  # v2 from the slides is deprecated; v3 is identical.

# Default hyper-parameters (close to the slide values, tuned for CPU training).
DEFAULT_HPARAMS = {
    "h_size": 64,
    "n_training_episodes": 3000,
    "max_steps": 1000,
    "gamma": 0.99,
    "lr": 1e-3,
    "print_every": 100,
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
