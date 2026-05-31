"""Gradio demo for Exercise 3 — REINFORCE on LunarLander.

Three tabs:
  1. Overview & Comparison — pre-computed results for REINFORCE vs
     REINFORCE+Baseline (learning curves, distributions, bar chart, table).
  2. Watch the agent — render a trained agent landing as a GIF.
  3. Interactive training — train an agent live with custom hyper-parameters
     and watch the learning curve update in real time.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict

import gradio as gr
import gymnasium as gym
import pandas as pd
import torch
import torch.optim as optim

from src import plotting
from src.config import DEFAULT_HPARAMS, ENV_ID, get_device
from src.evaluate import evaluate_agent, record_episode_gif
from src.policy import Policy, PolicyWithValue
from src.reinforce import reinforce, reinforce_with_baseline

ROOT = os.path.dirname(__file__)
CKPT_DIR = os.path.join(ROOT, "checkpoints")
DEVICE = get_device()

ALGO_LABELS = {
    "reinforce": "REINFORCE (vanilla, theo slide)",
    "reinforce_baseline": "REINFORCE + Baseline (cải tiến)",
}


# --------------------------------------------------------------------------- #
# Loading pre-computed results
# --------------------------------------------------------------------------- #
def load_histories() -> Dict[str, dict]:
    histories: Dict[str, dict] = {}
    for algo in ("reinforce", "reinforce_baseline"):
        path = os.path.join(CKPT_DIR, f"{algo}_history.json")
        if os.path.exists(path):
            with open(path) as f:
                histories[algo] = json.load(f)
    return histories


def load_experiment() -> Dict[str, dict]:
    """Multi-seed aggregate written by scripts.finalize_experiment (optional)."""
    path = os.path.join(CKPT_DIR, "experiment.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def build_summary_table(histories: Dict[str, dict]) -> pd.DataFrame:
    exp = load_experiment()
    rows = []
    for algo, hist in histories.items():
        scores = hist.get("avg_scores", [])
        solved_at = next((i for i, v in enumerate(scores) if v >= 200), None)
        row = {
            "Thuật toán": plotting.LABELS.get(algo, algo),
            "Eval (demo, seed 0)": f"{hist.get('eval_mean', 0):.1f} ± {hist.get('eval_std', 0):.1f}",
            "Đạt ≥200 (seed 0)": f"ep {solved_at}" if solved_at is not None else "chưa",
        }
        agg = exp.get(algo)
        if agg:
            row["Eval TB (3 seed)"] = f"{agg['eval_mean_avg']:.1f} ± {agg['eval_mean_std']:.1f}"
            row["Eval-std TB ↓ (3 seed)"] = f"{agg['per_seed_eval_std_mean']:.1f}"
            row["Best avg-100 (3 seed)"] = f"{agg['best_avg_mean']:.1f} ± {agg['best_avg_std']:.1f}"
            row["Solved (3 seed)"] = f"{agg['n_solved']}/{agg['n_seeds']}"
        else:
            row["Best avg-100"] = f"{max(scores):.1f}" if scores else "-"
            row["Train time (s)"] = f"{hist.get('train_time_sec', 0):.0f}"
        rows.append(row)
    return pd.DataFrame(rows)


def refresh_overview():
    histories = load_histories()
    if not histories:
        empty = pd.DataFrame([{"Thông báo": "Chưa có checkpoint. Hãy chạy: python -m src.train"}])
        return None, None, None, None, empty
    return (
        plotting.learning_curve(histories, "Learning curve (seed 0) — REINFORCE vs REINFORCE+Baseline"),
        plotting.comparison_bars(histories, load_experiment()),
        plotting.reward_distribution(histories),
        plotting.episode_length_curve(histories),
        build_summary_table(histories),
    )


# --------------------------------------------------------------------------- #
# Watch the agent
# --------------------------------------------------------------------------- #
def _load_policy(algo: str):
    path = os.path.join(CKPT_DIR, f"{algo}.pt")
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if algo == "reinforce":
        policy = Policy(ckpt["s_size"], ckpt["a_size"], ckpt["h_size"])
    else:
        policy = PolicyWithValue(ckpt["s_size"], ckpt["a_size"], ckpt["h_size"])
    policy.load_state_dict(ckpt["state_dict"])
    policy.to(DEVICE).eval()
    return policy


def watch_agent(algo_label: str, seed: int):
    algo = {v: k for k, v in ALGO_LABELS.items()}[algo_label]
    policy = _load_policy(algo)
    if policy is None:
        return None, f"Chưa có checkpoint cho {algo}. Hãy train trước."
    out_path = os.path.join(tempfile.mkdtemp(), f"{algo}_play.gif")
    gif, reward = record_episode_gif(policy, DEVICE, out_path, seed=int(seed), greedy=True)
    return gif, f"Tổng reward của episode: **{reward:.1f}** (seed={int(seed)})"


# --------------------------------------------------------------------------- #
# Interactive training
# --------------------------------------------------------------------------- #
def interactive_train(algo_label: str, episodes: int, lr: float, gamma: float,
                      h_size: int, progress=gr.Progress()):
    algo = {v: k for k, v in ALGO_LABELS.items()}[algo_label]
    torch.manual_seed(0)
    env = gym.make(ENV_ID)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    max_steps = DEFAULT_HPARAMS["max_steps"]

    if algo == "reinforce":
        policy = Policy(s_size, a_size, int(h_size)).to(DEVICE)
        optimizer = optim.Adam(policy.parameters(), lr=lr)
        train_fn = reinforce
    else:
        policy = PolicyWithValue(s_size, a_size, int(h_size)).to(DEVICE)
        optimizer = optim.Adam(policy.parameters(), lr=lr)
        train_fn = reinforce_with_baseline

    def cb(i, n, avg):
        progress((i / n), desc=f"Episode {i}/{n} — avg-100: {avg:.1f}")

    result = train_fn(policy, optimizer, env, int(episodes), max_steps, gamma,
                      DEVICE, print_every=0, progress_cb=cb)

    eval_env = gym.make(ENV_ID)
    mean_r, std_r, _ = evaluate_agent(eval_env, policy, DEVICE, max_steps,
                                      n_eval_episodes=20, greedy=False,
                                      seed=list(range(20)))
    fig = plotting.live_curve(result.scores, result.avg_scores, algo)
    msg = (f"Hoàn tất {episodes} episodes.\n\n"
           f"**Eval mean reward = {mean_r:.1f} ± {std_r:.1f}** (20 episodes)\n\n"
           f"Best avg-100 trong lúc train = {max(result.avg_scores):.1f}")
    return fig, msg


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
INTRO = """
# 🚀 Exercise 3 — REINFORCE trên LunarLander
**Policy Gradient (Monte-Carlo REINFORCE)** điều khiển tàu đổ bộ hạ cánh đúng bãi đáp.
So sánh **REINFORCE thuần (theo slide)** với **REINFORCE + Baseline** (thêm value head để giảm phương sai gradient).

- *State*: 8 chiều · *Action*: Discrete(4) · *Solved*: avg reward ≥ 200
- Môi trường: `LunarLander-v3` (bản v2 trong slide đã deprecated, kiến trúc giống hệt)
"""


def build_app():
    with gr.Blocks(title="REINFORCE LunarLander") as demo:
        gr.Markdown(INTRO)

        with gr.Tab("📊 Tổng quan & So sánh"):
            gr.Markdown("Kết quả đã huấn luyện sẵn (3000 episodes, lr=1e-3). "
                        "Learning curve/phân phối là **run seed 0**; cột so sánh & bảng có "
                        "thêm **trung bình ± std qua 3 seed** để so sánh công bằng "
                        "(REINFORCE phương sai cao).")
            refresh_btn = gr.Button("🔄 Tải / làm mới kết quả", variant="primary")
            table = gr.Dataframe(label="Bảng tổng kết", interactive=False)
            with gr.Row():
                lc_plot = gr.Plot(label="Learning curve")
                bar_plot = gr.Plot(label="So sánh hiệu năng cuối")
            with gr.Row():
                dist_plot = gr.Plot(label="Phân phối reward (đánh giá)")
                len_plot = gr.Plot(label="Độ dài episode")
            refresh_btn.click(refresh_overview,
                              outputs=[lc_plot, bar_plot, dist_plot, len_plot, table])
            demo.load(refresh_overview,
                      outputs=[lc_plot, bar_plot, dist_plot, len_plot, table])

        with gr.Tab("🎬 Xem agent chơi"):
            gr.Markdown("Render 1 episode của agent đã huấn luyện (greedy).")
            with gr.Row():
                algo_play = gr.Radio(list(ALGO_LABELS.values()),
                                     value=ALGO_LABELS["reinforce_baseline"],
                                     label="Chọn agent")
                seed_play = gr.Number(value=42, label="Seed", precision=0)
            play_btn = gr.Button("▶️ Cho agent chơi", variant="primary")
            gif_out = gr.Image(label="Agent landing", type="filepath")
            reward_out = gr.Markdown()
            play_btn.click(watch_agent, [algo_play, seed_play], [gif_out, reward_out])

        with gr.Tab("🧪 Huấn luyện tương tác"):
            gr.Markdown("Tự chỉnh siêu tham số và xem learning curve cập nhật.")
            with gr.Row():
                algo_tr = gr.Radio(list(ALGO_LABELS.values()),
                                   value=ALGO_LABELS["reinforce"], label="Thuật toán")
            with gr.Row():
                ep_tr = gr.Slider(100, 2000, value=400, step=100, label="Số episodes")
                lr_tr = gr.Slider(1e-4, 5e-2, value=1e-3, step=1e-4, label="Learning rate")
            with gr.Row():
                gamma_tr = gr.Slider(0.90, 0.999, value=0.99, step=0.001, label="Gamma")
                h_tr = gr.Slider(16, 128, value=64, step=16, label="Hidden size")
            train_btn = gr.Button("🚀 Bắt đầu huấn luyện", variant="primary")
            tr_plot = gr.Plot(label="Live learning curve")
            tr_msg = gr.Markdown()
            train_btn.click(interactive_train,
                            [algo_tr, ep_tr, lr_tr, gamma_tr, h_tr],
                            [tr_plot, tr_msg])

        gr.Markdown("---\n📖 Xem `docs/EXPLAINER_vi.md` để hiểu chi tiết thuật toán & dự án.")
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860,
               theme=gr.themes.Soft(primary_hue="blue"))
