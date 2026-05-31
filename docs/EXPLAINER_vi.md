# 📖 Giải thích chi tiết — REINFORCE trên LunarLander (Exercise 3)

Tài liệu này giúp bạn hiểu **kiến thức lý thuyết** đằng sau Policy Gradient / REINFORCE và **cách dự án này được tổ chức**. Đọc xong bạn sẽ nắm được: bài toán là gì, thuật toán hoạt động ra sao, vì sao thêm *baseline* giúp cải thiện, và mỗi file code làm gì.

---

## 1. Bài toán: LunarLander

LunarLander là môi trường điều khiển một tàu đổ bộ hạ cánh đúng vào bãi đáp (giữa 2 lá cờ).

| Thành phần | Mô tả |
|---|---|
| **State** | Vector 8 chiều: vị trí (x, y), vận tốc (vx, vy), góc nghiêng, vận tốc góc, 2 cờ báo chân trái/phải chạm đất |
| **Action** | `Discrete(4)`: 0 = không làm gì, 1 = bắn động cơ trái, 2 = bắn động cơ chính, 3 = bắn động cơ phải |
| **Reward** | +100 khi đáp đúng bãi, −100 khi đâm, −0.3 mỗi bước (phạt thời gian), +10 mỗi chân chạm đất, phạt nhiên liệu khi bắn động cơ |
| **Solved** | Reward trung bình ≥ **200** trên 100 episode |
| **Loại môi trường** | Stochastic (ngẫu nhiên) |

> ⚠️ Slide dùng `gym` + `LunarLander-v2`. Bản này đã bị *deprecated*; dự án dùng `gymnasium` + `LunarLander-v3` — **giống hệt** về state/action/reward, chỉ khác API (`reset()` trả thêm `info`, `step()` trả `terminated`/`truncated` thay cho `done`).

---

## 2. Policy Gradient là gì?

Có 2 nhóm phương pháp lớn trong RL:

- **Value-based** (vd. Q-Learning, DQN): học hàm giá trị `Q(s, a)`, rồi suy ra chính sách bằng cách chọn action có Q lớn nhất.
- **Policy-based** (Policy Gradient): **học trực tiếp chính sách** `π_θ(a|s)` — một mạng nơ-ron nhận state, xuất ra **phân phối xác suất trên các action**, không cần thông qua hàm giá trị.

Chính sách được tham số hoá bằng mạng nơ-ron với tham số `θ`:

```
π_θ(a | s) = P(action = a | state = s; θ)
```

### Mục tiêu huấn luyện

Tối đa hoá **kỳ vọng tổng phần thưởng tích luỹ** của một trajectory `τ` (chuỗi state–action):

```
maximize  J(θ) = E_{τ ~ π_θ} [ R(τ) ]
```

với `R(τ) = r₁ + γ·r₂ + γ²·r₃ + ...` (γ là discount factor).

Vì là bài toán *tối đa hoá*, ta dùng **Gradient Ascent**:

```
θ ← θ + α · ∇_θ J(θ)
```

### Policy Gradient Theorem

Gradient của mục tiêu có thể ước lượng bằng (Monte-Carlo):

```
∇_θ J(θ) ≈ Σ_t  ∇_θ log π_θ(aₜ | sₜ) · R(τ)
```

Trực giác: **tăng xác suất** của những action dẫn tới return cao, **giảm xác suất** action dẫn tới return thấp — có trọng số theo `R`.

---

## 3. Thuật toán REINFORCE (Monte-Carlo Policy Gradient)

REINFORCE là cách hiện thực đơn giản nhất của Policy Gradient. Pseudocode (theo slide):

```
Algorithm REINFORCE
1. Khởi tạo policy model π_θ
2. repeat:
3.   Sinh 1 episode S₀,A₀,r₀,…,S_{T-1},A_{T-1},r_{T-1} theo π_θ
4.   for t từ T-1 về 0:
5.     Gₜ = Σ_{k=t}^{T-1} γ^{k-t} · r_k        # return tích luỹ từ bước t
6.   L(θ) = (1/T) Σ_t Gₜ · log π_θ(Aₜ|Sₜ)
7.   Cập nhật θ theo ∇L(θ)
```

### Các chi tiết quan trọng trong code

1. **Tính return ngược (`appendleft`)** — duyệt rewards từ cuối về đầu:
   `Gₜ = γ·G_{t+1} + rₜ`. Xem `src/reinforce.py::_discounted_returns`.

2. **Chuẩn hoá return** (giảm phương sai):
   ```python
   returns = (returns - returns.mean()) / (returns.std() + eps)
   ```

3. **Dấu âm trong loss** — PyTorch *tối thiểu hoá*, nên để thực hiện *gradient ascent* ta thêm dấu trừ:
   ```python
   policy_loss = Σ_t (-log_prob_t * Gₜ)
   ```

4. **Lấy mẫu action** — dùng `Categorical(probs).sample()` để cân bằng *exploration/exploitation*; khi đánh giá có thể chọn greedy (argmax).

---

## 4. Kiến trúc Policy Network

Đúng theo slide:

```
State (8) ─► FC1 (h) ─► ReLU ─► FC2 (h*2) ─► ReLU ─► FC3 (4) ─► softmax ─► phân phối action
```

Mặc định `h = 64`. Xem `src/policy.py::Policy`.

---

## 5. Cải tiến: REINFORCE **with Baseline** (so sánh chính của dự án)

### Vấn đề của REINFORCE thuần
Ước lượng gradient của REINFORCE có **phương sai (variance) rất lớn** vì dùng nguyên return `Gₜ` (một con số dao động mạnh giữa các episode). Hệ quả: học **chậm và không ổn định** — thực nghiệm cho thấy với learning rate cao nó còn **phân kỳ** (reward tụt xuống ~ −600).

### Ý tưởng baseline
Trừ đi một **baseline** `b(sₜ)` khỏi return mà **không làm lệch (bias)** gradient:

```
∇_θ J(θ) ≈ Σ_t ∇_θ log π_θ(aₜ|sₜ) · ( Gₜ − b(sₜ) )
```

Chọn `b(sₜ) = V(sₜ)` (giá trị kỳ vọng của state) thì `Aₜ = Gₜ − V(sₜ)` chính là **advantage**: "action này tốt hơn / tệ hơn mức trung bình bao nhiêu". Việc trừ baseline **giảm variance** → học nhanh và ổn định hơn.

### Hiện thực trong dự án
- Mạng `PolicyWithValue` gồm **2 mạng tách rời (separate networks)**: mạng policy (FC1→FC2→`policy_head`) và mạng value độc lập (v_FC1→v_FC2→`value_head` → V(s)). Tách riêng để **value loss không lấn át** gradient của policy (xem mục ⚠️ bên dưới).
- Loss gồm 2 phần:
  ```python
  value_loss = MSE( V(sₜ), Gₜ )                       # train V trên RETURN GỐC Gₜ (target dừng/stationary)
  Aₜ = Gₜ − V(sₜ).detach()                            # advantage, detach baseline → không bias
  Aₜ = (Aₜ − mean(A)) / (std(A) + eps)                # CHỈ chuẩn hoá ADVANTAGE cho policy gradient
  policy_loss = Σₜ (−log_probₜ · Aₜ)
  loss = policy_loss + 0.5 · value_loss
  ```
- `advantage` dùng `V(s).detach()` để baseline **chỉ giảm variance**, không tạo bias cho gradient của policy.

> ⚠️ **Vì sao train V trên return *gốc* chứ không phải return *đã chuẩn hoá*?**
> Nếu chuẩn hoá return *theo từng episode* rồi lấy đó làm target cho `V`, thì cùng 1 state sẽ có target khác nhau tuỳ tổng return của episode → **mục tiêu không dừng (non-stationary)**, làm hỏng ý nghĩa của baseline. Cách đúng (Sutton & Barto §13.4): huấn luyện `V` trên `Gₜ` gốc (target ổn định), và **chỉ chuẩn hoá advantage** để giữ policy gradient gọn về thang đo, ít phương sai.
>
> Vì `Gₜ` của LunarLander có biên độ lớn (hàng trăm) → `value_loss` lớn. Nếu policy & value **dùng chung trunk**, gradient từ value loss sẽ lấn át và phá đặc trưng của policy (thực nghiệm: entropy sụp về ~0, reward rơi xuống −3000). Vì vậy ta **tách riêng 2 mạng**.

### Lưu best-checkpoint (chống dao động cuối)
REINFORCE phương sai cao nên policy *cuối cùng* có thể tệ hơn policy tốt nhất giữa chừng. Cả 2 thuật toán đều **lưu lại bộ trọng số đạt best avg-100** trong lúc train rồi **nạp lại trước khi đánh giá** (`src/reinforce.py::_track_best`), nên số liệu eval phản ánh checkpoint tốt nhất chứ không phải checkpoint bị sụp ở cuối.

Xem `src/reinforce.py::reinforce_with_baseline`.

### Kỳ vọng kết quả
REINFORCE + Baseline **đạt mốc 200 (solved)** trong khi REINFORCE thuần thường **chững lại quanh ~130–140 và không solve**; đường học của bản baseline cũng **mượt và cao hơn**. Bảng/biểu đồ so sánh được hiển thị trực tiếp trong Gradio demo (tab "Tổng quan & So sánh") — số liệu cụ thể đọc động từ `checkpoints/*_history.json`.

---

## 6. Cấu trúc dự án

```
lunarlander-reinforce/
├── app.py                  # Gradio demo (3 tab: So sánh / Xem agent / Train tương tác)
├── requirements.txt
├── src/
│   ├── policy.py           # Policy network + PolicyWithValue (có value head)
│   ├── reinforce.py        # 2 thuật toán: reinforce & reinforce_with_baseline
│   ├── train.py            # CLI huấn luyện, lưu checkpoint + history JSON
│   ├── evaluate.py         # đánh giá agent + render GIF tàu đáp
│   ├── plotting.py         # các biểu đồ matplotlib dùng trong demo
│   └── config.py           # siêu tham số mặc định
├── checkpoints/            # .pt + *_history.json (sinh ra sau khi train)
└── docs/EXPLAINER_vi.md    # tài liệu này
```

### Luồng hoạt động
1. `train.py` huấn luyện → lưu `checkpoints/<algo>.pt` và `<algo>_history.json` (chứa scores, eval, thời gian train).
2. `app.py` đọc các file history để vẽ biểu đồ so sánh, load `.pt` để cho agent chơi (xuất GIF), và cho phép train lại tương tác.

---

## 7. Cách chạy

```bash
pip install -r requirements.txt

# Huấn luyện cả 2 agent
python -m src.train --algo reinforce          --episodes 3000 --lr 1e-3
python -m src.train --algo reinforce_baseline --episodes 3000 --lr 1e-3

# Chạy demo
python app.py        # mở http://localhost:7860
```

---

## 8. Công thức tóm tắt

| Khái niệm | Công thức |
|---|---|
| Chính sách | `π_θ(a\|s) = softmax(NN_θ(s))` |
| Return | `Gₜ = Σ_{k=t}^{T-1} γ^{k-t} r_k` |
| Mục tiêu | `J(θ) = E_{τ~π_θ}[R(τ)]` |
| Gradient (REINFORCE) | `∇J ≈ Σ_t ∇ log π_θ(aₜ\|sₜ) · Gₜ` |
| Gradient (+Baseline) | `∇J ≈ Σ_t ∇ log π_θ(aₜ\|sₜ) · (Gₜ − V(sₜ))` |
| Loss PyTorch | `−Σ_t log π_θ(aₜ\|sₜ) · (advantage)ₜ` |

---

## 9. Hạn chế & hướng phát triển tiếp
- REINFORCE là *on-policy, Monte-Carlo* → kém hiệu quả dữ liệu (mỗi episode dùng 1 lần).
- Có thể tiến tới **Actor-Critic / A2C** (bootstrap từng bước thay vì đợi hết episode), rồi **PPO** (giới hạn bước cập nhật bằng clipping) để ổn định và mạnh hơn nữa.
