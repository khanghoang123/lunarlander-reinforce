"""Aggregate the multi-seed runs in ``checkpoints/exp/`` for both algorithms.

The DEMO_SEED run's checkpoint + history are copied to
``checkpoints/<algo>.pt`` / ``checkpoints/<algo>_history.json`` (these drive the
Gradio demo and the static plots) so both algorithms are shown on the *same*
seed (fair head-to-head). A combined ``checkpoints/experiment.json`` is written
with per-seed values and mean ± std aggregates across all seeds for the report.

    python -m scripts.finalize_experiment
"""

from __future__ import annotations

import glob
import json
import os
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
CKPT = os.path.join(ROOT, "checkpoints")
EXP = os.path.join(CKPT, "exp")
ALGOS = ("reinforce", "reinforce_baseline")
DEMO_SEED = 0  # committed checkpoints use this seed (same for both algorithms)


def _load_runs(algo: str) -> list:
    runs = []
    for path in sorted(glob.glob(os.path.join(EXP, f"{algo}_seed*.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def main() -> None:
    aggregate = {}
    for algo in ALGOS:
        runs = _load_runs(algo)
        if not runs:
            raise SystemExit(f"No experiment runs found for {algo} in {EXP}")
        evals = np.array([r["eval_mean"] for r in runs])
        bests = np.array([r["best_avg"] for r in runs])
        solved = [r["solved_episode"] for r in runs]
        n_solved = sum(1 for s in solved if s is not None)
        all_eval = [x for r in runs for x in r["eval_rewards"]]
        succ = 100.0 * float(np.mean([1.0 if x >= 200 else 0.0 for x in all_eval]))

        rep = next((r for r in runs if r["seed"] == DEMO_SEED), runs[0])
        rep_seed = rep["seed"]

        shutil.copy(os.path.join(EXP, f"{algo}_seed{rep_seed}.pt"),
                    os.path.join(CKPT, f"{algo}.pt"))
        # The demo/plots expect a flat history dict (same schema as train.py).
        hist = {k: rep[k] for k in (
            "scores", "avg_scores", "episode_lengths", "best_avg",
            "solved_episode", "eval_mean", "eval_std", "eval_rewards",
            "train_time_sec")}
        hist["algo"] = algo
        hist["hparams"] = {"episodes": len(rep["scores"]), "lr": 1e-3,
                           "gamma": 0.99, "h_size": 64, "max_steps": 1000,
                           "seed": rep_seed}
        with open(os.path.join(CKPT, f"{algo}_history.json"), "w") as f:
            json.dump(hist, f)

        aggregate[algo] = {
            "n_seeds": len(runs),
            "seeds": [r["seed"] for r in runs],
            "eval_mean_avg": float(evals.mean()), "eval_mean_std": float(evals.std()),
            "per_seed_eval": [round(float(x), 1) for x in evals],
            "per_seed_eval_std": [round(float(r["eval_std"]), 1) for r in runs],
            "best_avg_mean": float(bests.mean()), "best_avg_std": float(bests.std()),
            "per_seed_best_avg": [round(float(x), 1) for x in bests],
            "n_solved": n_solved, "per_seed_solved": solved,
            "success_rate_pct": round(succ, 1),
            "per_seed_eval_std_mean": round(float(np.mean([r["eval_std"] for r in runs])), 1),
            "train_time_mean": float(np.mean([r["train_time_sec"] for r in runs])),
            "demo_seed": rep_seed,
        }
        print(f"{algo}: demo seed={rep_seed} | eval {evals.mean():.1f}±{evals.std():.1f} "
              f"| best_avg {bests.mean():.1f}±{bests.std():.1f} | solved {n_solved}/{len(runs)} "
              f"| success {succ:.0f}%")

    with open(os.path.join(CKPT, "experiment.json"), "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nWrote {os.path.join(CKPT, 'experiment.json')}")


if __name__ == "__main__":
    main()
