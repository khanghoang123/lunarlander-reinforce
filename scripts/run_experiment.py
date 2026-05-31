"""Multi-seed comparison of REINFORCE vs REINFORCE+Baseline.

REINFORCE is high-variance, so a single seed is unreliable. This runs several
seeds per algorithm, saves a per-seed history + checkpoint under
``checkpoints/exp/``, and writes an aggregate JSON (``checkpoints/experiment.json``)
with mean ± std of the key metrics across seeds.

    python -m scripts.run_experiment --seeds 0 1 2 --episodes 3000 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from src.config import ENV_ID, get_device
from src.evaluate import evaluate_agent
from src.policy import Policy, PolicyWithValue
from src.reinforce import reinforce, reinforce_with_baseline

ROOT = os.path.dirname(os.path.dirname(__file__))
EXP_DIR = os.path.join(ROOT, "checkpoints", "exp")


def run_one(algo: str, seed: int, episodes: int, lr: float, gamma: float,
            h_size: int, max_steps: int) -> dict:
    device = get_device()
    torch.manual_seed(seed)
    env = gym.make(ENV_ID)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    if algo == "reinforce":
        policy = Policy(s_size, a_size, h_size).to(device)
        train_fn = reinforce
    else:
        policy = PolicyWithValue(s_size, a_size, h_size).to(device)
        train_fn = reinforce_with_baseline
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    t0 = time.time()
    result = train_fn(policy, optimizer, env, episodes, max_steps, gamma, device,
                      print_every=0)
    train_time = time.time() - t0

    eval_env = gym.make(ENV_ID)
    mean_r, std_r, all_r = evaluate_agent(eval_env, policy, device, max_steps,
                                          n_eval_episodes=50, greedy=False,
                                          seed=list(range(50)))
    os.makedirs(EXP_DIR, exist_ok=True)
    torch.save({"algo": algo, "state_dict": policy.state_dict(),
                "s_size": int(s_size), "a_size": int(a_size), "h_size": int(h_size)},
               os.path.join(EXP_DIR, f"{algo}_seed{seed}.pt"))
    hist = {
        "algo": algo, "seed": seed,
        "scores": result.scores, "avg_scores": result.avg_scores,
        "episode_lengths": result.episode_lengths,
        "best_avg": result.best_avg, "solved_episode": result.solved_episode,
        "eval_mean": mean_r, "eval_std": std_r, "eval_rewards": all_r,
        "train_time_sec": train_time,
    }
    with open(os.path.join(EXP_DIR, f"{algo}_seed{seed}.json"), "w") as f:
        json.dump(hist, f)
    print(f"[{algo} seed{seed}] eval {mean_r:.1f}±{std_r:.1f} | best_avg "
          f"{result.best_avg:.1f} | solved {result.solved_episode} | {train_time:.0f}s")
    return hist


def aggregate(per_seed: dict) -> dict:
    out = {}
    for algo, runs in per_seed.items():
        evals = [r["eval_mean"] for r in runs]
        bests = [r["best_avg"] for r in runs]
        solved = [r["solved_episode"] for r in runs if r["solved_episode"] is not None]
        # success rate: fraction of all eval episodes (across seeds) that land (>=200)
        all_eval = [x for r in runs for x in r["eval_rewards"]]
        succ = float(np.mean([1.0 if x >= 200 else 0.0 for x in all_eval])) * 100
        out[algo] = {
            "n_seeds": len(runs),
            "eval_mean_avg": float(np.mean(evals)), "eval_mean_std": float(np.std(evals)),
            "best_avg_mean": float(np.mean(bests)), "best_avg_std": float(np.std(bests)),
            "n_solved": len(solved), "solved_episode_mean": (float(np.mean(solved)) if solved else None),
            "success_rate_pct": succ,
            "per_seed_eval": evals, "per_seed_best_avg": bests,
            "per_seed_solved": [r["solved_episode"] for r in runs],
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--h_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--algos", nargs="+", default=["reinforce", "reinforce_baseline"])
    args = p.parse_args()

    per_seed = {a: [] for a in args.algos}
    for algo in args.algos:
        for seed in args.seeds:
            per_seed[algo].append(
                run_one(algo, seed, args.episodes, args.lr, args.gamma,
                        args.h_size, args.max_steps))

    agg = aggregate(per_seed)
    with open(os.path.join(ROOT, "checkpoints", "experiment.json"), "w") as f:
        json.dump({"config": vars(args), "aggregate": agg}, f, indent=2)
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
