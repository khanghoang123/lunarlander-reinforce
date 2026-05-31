---
name: testing-gradio-demo
description: Test the LunarLander REINFORCE Gradio demo end-to-end. Use when verifying the app.py UI (overview/comparison, watch-agent, interactive training) or any change to the REINFORCE training/plotting/eval code.
---

# Testing the LunarLander REINFORCE Gradio demo

## Setup & launch
- Project deps (gymnasium[box2d], torch, gradio, matplotlib, imageio, pandas) are installed in the active Python (pyenv `python3`). The old `/home/ubuntu/.venv-rl` is NOT guaranteed to exist in a fresh session — just use `python3`.
- From repo root: `PYTHONPATH=$(pwd) python3 app.py` → serves at `http://localhost:7860`.
- Pre-trained checkpoints + history JSON live in `checkpoints/` (`<algo>.pt`, `<algo>_history.json`, plus `experiment.json` with the 3-seed aggregate). Static plots/GIFs live in `assets/`. If missing, regenerate: `python -m scripts.run_experiment --seeds 0 1 2 --episodes 3000 --lr 1e-3 && python -m scripts.finalize_experiment && python -m scripts.regen_assets`.
- Maximize the browser before recording: `wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz`.

## What to verify (3 tabs)
1. **Tổng quan & So sánh (overview)** — auto-loads on page open. Table has 2 rows with columns: `Eval (demo, seed 0)`, `Đạt ≥200 (seed 0)`, `Eval TB (3 seed)`, `Eval-std TB ↓ (3 seed)`, `Best avg-100 (3 seed)`, `Solved (3 seed)`. Committed demo checkpoints use **seed 0**. Expected (lr=1e-3, 3000 eps): REINFORCE seed-0 eval `143.3 ± 91.5`, solved `chưa`; Baseline seed-0 eval `235.5 ± 88.9`, solved `ep 1239`. 3-seed means: REINFORCE `151.9 ± 35.8` (std 96.8); Baseline `177.6 ± 41.5` (std 52.2). Bar chart shows the **3-seed** mean (152 vs 178); learning curve is the seed-0 run (baseline crosses 200, vanilla stays below).
2. **Xem agent chơi (watch-agent)** — select an algo + seed, click `▶️ Cho agent chơi`. Takes a few seconds. Confirm a GIF appears and a reward line shows (Baseline @ seed=42 ≈ `155.6`).
3. **Huấn luyện tương tác (interactive training)** — pick algo, set episodes/lr/gamma/hidden size, click `🚀 Bắt đầu huấn luyện`. Progress bar advances; on completion a live learning curve + eval text appear. A 300-episode run takes ~20s; judge by the upward avg-100 trend (won't reach +200, that needs ~1200+ eps).

## Gotchas
- **Interactive lr default is now 1e-3** (slider), which trains cleanly. Avoid lr ≥ 0.01 (REINFORCE diverges).
- **LunarLander-v3**, not v2 (slide's v2 is deprecated in modern gymnasium; drop-in compatible).
- **torch.load needs `weights_only=False`** (PyTorch 2.6+) because checkpoints store config ints alongside the state_dict.
- The value network is now **separate** from the policy and is trained on **raw returns G_t** (only the advantage is normalised); the agent reloads its **best avg-100 checkpoint** before eval.
- GIF rendering uses a headless render; if it hangs, confirm `box2d` is installed (needs swig + build-essential to compile).
- Interactive training reward staying negative at low episode counts is expected, NOT a bug — judge by the upward trend.
- Gradio 6: the `theme` is passed in `app.launch(...)`, not the `Blocks(...)` constructor.

## Devin Secrets Needed
- None. Everything runs locally; no external auth required.
