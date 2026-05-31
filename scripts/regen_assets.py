"""Regenerate the static plots and gameplay GIFs in ``assets/``.

Run after (re)training so the committed artifacts match the current
``checkpoints/*_history.json`` and ``checkpoints/*.pt``:

    python -m scripts.regen_assets
"""

from __future__ import annotations

import json
import os

from src import plotting
from src.config import get_device
from src.evaluate import record_episode_gif
from src.policy import Policy, PolicyWithValue

ROOT = os.path.dirname(os.path.dirname(__file__))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
ASSET_DIR = os.path.join(ROOT, "assets")
DEVICE = get_device()


def _load_histories() -> dict:
    histories = {}
    for algo in ("reinforce", "reinforce_baseline"):
        path = os.path.join(CKPT_DIR, f"{algo}_history.json")
        with open(path) as f:
            histories[algo] = json.load(f)
    return histories


def _load_policy(algo: str):
    ckpt = __import__("torch").load(
        os.path.join(CKPT_DIR, f"{algo}.pt"), map_location=DEVICE, weights_only=False
    )
    cls = Policy if algo == "reinforce" else PolicyWithValue
    policy = cls(ckpt["s_size"], ckpt["a_size"], ckpt["h_size"])
    policy.load_state_dict(ckpt["state_dict"])
    return policy.to(DEVICE).eval()


def _load_experiment() -> dict:
    path = os.path.join(CKPT_DIR, "experiment.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main() -> None:
    os.makedirs(ASSET_DIR, exist_ok=True)
    histories = _load_histories()
    exp = _load_experiment()

    figs = {
        "learning_curve.png": plotting.learning_curve(
            histories, "Learning curve (seed 0) — REINFORCE vs REINFORCE+Baseline"
        ),
        "comparison_bars.png": plotting.comparison_bars(histories, exp),
        "reward_dist.png": plotting.reward_distribution(histories),
        "episode_length.png": plotting.episode_length_curve(histories),
    }
    for name, fig in figs.items():
        out = os.path.join(ASSET_DIR, name)
        fig.savefig(out, bbox_inches="tight")
        print(f"saved {out}")

    for algo, fname in (("reinforce", "reinforce_play.gif"),
                        ("reinforce_baseline", "reinforce_baseline_play.gif")):
        policy = _load_policy(algo)
        out = os.path.join(ASSET_DIR, fname)
        _, reward = record_episode_gif(policy, DEVICE, out, seed=42, greedy=True)
        print(f"saved {out} (reward={reward:.1f})")


if __name__ == "__main__":
    main()
