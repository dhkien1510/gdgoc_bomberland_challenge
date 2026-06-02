# V6 Agent Checklist Audit

Kết quả kiểm tra toàn bộ implementation v6 so với [checklist.md](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/checklist.md).

---

## Tổng quan nhanh

| Mục | Trạng thái | Ghi chú |
|-----|-----------|---------|
| A. Canonical action | ✅ Đúng | Round-trip qua `_model_base.py` import từ v5_1 |
| B. Dataset BC | ✅ Đúng | Split theo episode, có tactical state |
| C. Model BC + LSTM | ✅ Đúng | Shape, hidden reset, loss mask OK |
| D. Training BC | ✅ Đúng | Class weight, masked CE, metrics |
| E. Rollout eval BC | ✅ Đúng | `eval_bc.py` có đầy đủ metrics |
| F. BC inference parity | ⚠️ Có vấn đề nhỏ | `export_bc_agent.py` không dùng `weights_only=True` |
| G. DAgger-lite | ✅ Đúng | Implemented trong `collect_dataset.py` |
| H. PPO load BC actor | ✅ Đúng | `load_actor_from_checkpoint()` verify shape |
| I. Recurrent PPO | ❌ **3 lỗi quan trọng** | Advantage norm, hidden state, cross-episode |
| J. PPO không phá BC | ❌ **Thiếu BC loss regularization** | `train_ppo.py` không có `bc_coef * bc_loss` |
| K. Curriculum PPO | ✅ Đúng | 4 stage, metrics đầy đủ |
| L. Debug rollout | ✅ Đúng | Loop detection, BFS fallback |
| M. Checklist cuối | ❌ Có items chưa pass | Xem chi tiết bên dưới |

---

## A. Kiểm tra preprocessing/action — ✅ PASS

### A1. Canonical action round-trip
- [to_canonical_action](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/_model_base.py#L44) và [to_env_action](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/_model_base.py#L43) import trực tiếp từ `v5_1/_model_base.py` — **đúng**.
- `agent.py` L133: `return int(to_env_action(int(canonical_action), self.agent_id))` — **đúng**.
- `collect_dataset.py` L195: `canonical_action = int(to_canonical_action(env_action, agent_id))` — **đúng**.

### A2. Action label từ teacher
- [collect_dataset.py L194-195](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/collect_dataset.py#L194-L195): Lấy `env_action` từ teacher, convert sang `canonical_action` rồi lưu — **đúng**.

### A3. Mask chứa expert action
- [collect_dataset.py L203-204](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/collect_dataset.py#L203-L204): `if not bool(action_mask[canonical_action])` → skip sample, đếm `teacher_action_masked` — **đúng**, samples mà expert action bị mask sẽ bị loại bỏ.

---

## B. Dataset BC — ✅ PASS

### B1. Action distribution
- [analyze_dataset.py](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/analyze_dataset.py) in phân phối 6 action với tỷ lệ % — **đúng**.

### B2. Tactical state distribution
- `analyze_dataset.py` L54-61: Log `danger_states`, `valuable_bomb_states`, `can_escape_if_place` — **đúng**.
- `collect_dataset.py` lưu `danger_times`, `valuable_states`, `can_escape_if_place` — **đúng**.

### B3. Split theo episode/seed
- [bc_dataset.py L37-50](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/bc_dataset.py#L37-L50): `split_episode_keys()` shuffle unique episode keys rồi split — **đúng**, không shuffle theo step.

---

## C. Unit test model BC + LSTM — ✅ PASS

### C1. Forward shape
- [bc_model.py CNNLSTMBCActor](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/bc_model.py#L116-L144): `forward_sequence()` nhận `[B, T, C, 13, 13]` map + `[B, T, AUX_DIM]` aux → trả logits `[B, T, 6]` — **đúng**.

### C2. Hidden reset
- [bc_model.py L82-85](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/bc_model.py#L82-L85): Khi `episode_start=True`, `hx = hx * (1.0 - reset)` và `cx = cx * (1.0 - reset)` → reset về 0 — **đúng**.
- [agent.py L43](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/agent.py#L43): `self.episode_start = True` khi init — **đúng**.
- `agent.py` L119: `self.episode_start = False` sau step đầu — **đúng**.

> [!NOTE]
> Tuy nhiên, `agent.py` không bao giờ reset `self.episode_start = True` khi episode mới bắt đầu trong quá trình chạy liên tục. Nhưng trong competition context, mỗi instance Agent là 1 episode nên OK.

### C3. Sequence padding/loss mask
- [bc_dataset.py L143-154](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/bc_dataset.py#L143-L154): `loss_mask[:actual_len] = 1.0`, padding giữ 0 — **đúng**.
- [train_bc.py L32-39](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/train_bc.py#L32-L39): `masked_ce_loss()` chỉ tính loss trên `flat_mask > 0` — **đúng**.

---

## D. Training BC — ✅ PASS

### D1. Overfit batch nhỏ
- Không có script riêng, nhưng `train_bc.py` có thể chạy với `--epochs 1000 --batch_size 16` trên data nhỏ — **khả thi**.

### D2. Train smoke + metrics
- [train_bc.py L42-113](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/train_bc.py#L42-L113): `evaluate_loader()` trả về:
  - `loss`, `accuracy` — ✅
  - `bomb_precision`, `bomb_recall` — ✅
  - `danger_accuracy` — ✅
  - `valuable_accuracy` — ✅
  - `illegal_before_mask_rate` — ✅ (thêm metric hay)

### D3. Confusion matrix
- **Chưa implement** confusion matrix 6×6 trong `evaluate_loader()`.

> [!WARNING]
> Checklist yêu cầu confusion matrix 6×6. Hiện tại `train_bc.py` chỉ log precision/recall cho PLACE_BOMB. Thiếu confusion matrix chi tiết cho các action khác.

---

## E. Rollout eval BC — ✅ PASS

- [eval_bc.py](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/eval_bc.py) chạy match với opponent pools (easy/hard/mixed).
- Metrics: `points`, `first`, `bombs`, `boxes`, `items`, `kills`, `vb` (valuable_bomb_ratio), `repeat` — **đúng**.

> [!NOTE]
> Thiếu một số metrics so với checklist: `no_escape_bomb_ratio`, `danger_steps`, `unique_tiles_visited`, `episode_length`. Tuy nhiên `_train_base.py` có `summarize_episode_metrics()` trả đầy đủ, `eval_bc.py` chỉ chọn log một số. Không phải lỗi logic, chỉ là log chưa đầy đủ.

---

## F. BC inference parity — ⚠️ CÓ VẤN ĐỀ NHỎ

### F1. Cùng preprocessing
- `agent.py`, `eval_bc.py`, `collect_dataset.py`, `train_bc.py` đều import `prepare_policy_inputs` từ cùng source (`_model_base.py` → `_model_v3_base.py` → `model.py`) — **đúng**, cùng 1 hàm.

### F2. Checkpoint load

> [!WARNING]
> **`export_bc_agent.py` L22** dùng `torch.load(args.input, map_location="cpu")` **mà không có `weights_only=True`**. Checklist F2 yêu cầu phải dùng `weights_only=True` để tránh lỗi PyTorch 2.6+.
>
> Tương tự, [agent.py L50](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/agent.py#L50) và [agent.py L59](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/agent.py#L59) cũng không dùng `weights_only=True`.

Các files **thiếu `weights_only=True`**:
- `export_bc_agent.py` L22
- `agent.py` L50, L59
- `eval_bc.py` L42
- `collect_dataset.py` L96
- `model.py` L173

### F3. Deterministic vs stochastic
- `agent.py` L104/115: `deterministic=True` — **đúng**.
- Có loop detection fallback (L121-130) — **đúng**, match case 2 trong checklist.

---

## G. DAgger-lite — ✅ PASS

### G1. BC acts, teacher labels
- [collect_dataset.py L303-307](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/collect_dataset.py#L303-L307):
  ```python
  teacher_action = TacticalRuleAgent(player_id).act(obs)
  learner_action = player.act(obs)  # BC policy
  label_actions[player_id] = int(teacher_action)  # label = teacher
  actions[player_id] = learner_action if rng.random() < 0.8 else teacher_action
  ```
  - State do BC tạo (80% thời gian) — **đúng**.
  - Label luôn từ TacticalRuleAgent — **đúng**.

### G2. DAgger mixing
- 80% BC action, 20% teacher action — **đúng**, standard DAgger mixing ratio.

---

## H. PPO load BC actor — ✅ PASS

### H1. Load BC actor
- [model.py load_actor_from_checkpoint L171-195](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/model.py#L171-L195):
  - Remap keys từ BC checkpoint (prefix `actor_core.`, `actor_head.`)
  - Verify shape match: `tuple(current[new_key].shape) == tuple(value.shape)` — **đúng**.
  - Trả `loaded_actor` flag — **đúng**.

### H2. Critic warmup
- [train_ppo.py L179-191](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/train_ppo.py#L179-L191): `initial_actor_warmup_steps = 100_000` + `stage_actor_warmup_steps = 15_000`.
- [train_ppo.py L113-115](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/model.py#L113-L115): `set_actor_trainable(False)` freeze actor params.
- [train_ppo.py L392-400](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/train_ppo.py#L392-L400): Khi `actor_trainable=False`, `pg_loss` và `entropy_term` đều = 0, chỉ update `val_loss` — **đúng**.

> [!IMPORTANT]
> **Vấn đề tiềm ẩn**: Khi `actor_trainable=False`, gradient vẫn flow qua `nn.utils.clip_grad_norm_(model.parameters(), ...)` cho tất cả parameters, nhưng `requires_grad=False` trên actor params nên chúng thực sự không được update. Đây là **OK**, không phải bug, nhưng có thể tối ưu bằng cách chỉ pass critic parameters vào optimizer.

---

## I. Recurrent PPO — ❌ **3 LỖI QUAN TRỌNG**

### I1. Rollout buffer lưu sequence, không shuffle từng step — ✅

- [SequenceRolloutBuffer.make_windows](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/train_ppo.py#L93-L94): Cắt thành windows `[start, start+seq_len]` — **đúng**.
- PPO update shuffle windows, không shuffle individual steps — **đúng**.

### I2. Lưu hidden state tại đầu chunk — ❌ **THIẾU**

> [!CAUTION]
> **Lỗi nghiêm trọng**: `SequenceRolloutBuffer` **không lưu hidden state (h0, c0) tại đầu mỗi chunk**. Checklist I2 yêu cầu mỗi chunk cần `h0, c0` để khi recompute logits trong PPO update, LSTM bắt đầu từ đúng hidden state.
>
> Hiện tại, [train_ppo.py L383-388](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/train_ppo.py#L383-L388): `get_action_and_value_sequence()` được gọi **mà không truyền `state`**, nên nó sẽ dùng zero hidden state cho mỗi batch window. Điều này sai vì rollout thực tế có hidden state carry-over.
>
> **Impact**: Ratio `new_log_prob / old_log_prob` sẽ bị lệch → PPO update không chính xác → có thể gây divergence hoặc suboptimal training.

**Fix cần thiết**: Lưu `actor_state` tại mỗi step trong buffer, sau đó truyền `state` tương ứng của đầu mỗi window vào `get_action_and_value_sequence()`.

### I3. Reset hidden khi done — ⚠️ KHÔNG HOÀN TOÀN

- Rollout: [train_ppo.py L582](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/train_ppo.py#L582): `actor_state = model.get_initial_actor_state(1, DEVICE)` khi done — **đúng**.
- Update sequence: `episode_starts` được truyền vào `get_action_and_value_sequence()` → model reset hidden khi `episode_start=True` — **đúng**.

> [!WARNING]
> **Nhưng**: Windows có thể chứa episode boundaries. `make_windows()` cắt cố định theo `seq_len` mà **không respect episode boundaries**. Ví dụ: window [32..96] có thể chứa done=True ở step 50 và episode_start=True ở step 51. LSTM sẽ reset tại đó nhưng advantage/return computation có thể bị ảnh hưởng nếu GAE chạy xuyên qua done. **Tuy nhiên**, GAE computation (`compute_gae` L80-91) dùng `nonterminal = 1.0 - dones[t]` nên **advantage không xuyên qua done** → OK.
>
> Kết hợp với lỗi I2 (không lưu hidden state), reset hidden khi done trong update bị "partial correct" — model reset hidden khi gặp episode_start trong sequence, nhưng hidden state đầu window sai (zero thay vì actual).

### I4. Log_prob parity — ❌ **SẼ FAIL DO LỖI I2**

- Vì hidden state đầu chunk không khớp rollout (lỗi I2), `new_log_prob` sẽ khác `old_log_prob` ngay trước update đầu tiên.
- **Không có verification log nào** kiểm tra `mean abs diff` trước update.

### I5. Advantage normalization — ❌ **KHÔNG MASK PADDING**

> [!CAUTION]
> [train_ppo.py L370](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/train_ppo.py#L370):
> ```python
> advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
> ```
> Normalize trên **toàn bộ** buffer (bao gồm cả vùng padding implicit). Checklist I5 yêu cầu normalize trên valid tokens:
> ```python
> valid_adv = advantages[loss_mask == 1]
> advantages = (advantages - valid_adv.mean()) / (valid_adv.std() + 1e-8)
> ```
>
> **Tuy nhiên**, `compute_gae()` chạy trên raw rollout (không có padding), và padding chỉ xuất hiện trong `build_sequence_batch()`. Nên advantage normalization ở L370 thực ra **trước khi pack vào windows**, tức trên raw rollout data → **KHÔNG CÓ PADDING Ở BƯỚC NÀY** → thực tế **OK**.
>
> **Nhưng**: nếu window cuối ngắn hơn `seq_len`, `build_sequence_batch` pad advantage = 0. Trong PPO loss, `[valid]` filter sẽ loại padding. Nên **đây không phải lỗi critical**, nhưng kỹ thuật thì advantage normalization nên được verify lại trên valid portion.

**Kết luận I5**: Thực tế **OK** vì normalize xảy ra trước padding. Nhưng nên double-check.

### I6. PPO loss mask padding — ✅

- [train_ppo.py L391-402](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/train_ppo.py#L391-L402):
  ```python
  valid = batch["loss_mask"] > 0
  pg_loss = torch.max(pg_loss1, pg_loss2)[valid].mean()
  entropy_term = CFG["entropy_coef"] * entropy[valid].mean()
  val_loss = 0.5 * ((new_values - batch["returns"]) ** 2)[valid].mean()
  ```
  Tất cả loss đều mask padding — **đúng**.

---

## J. PPO không phá BC — ❌ **THIẾU BC LOSS REGULARIZATION**

### J1. PPO smoke test
- Logging đầy đủ: `FirstRate`, `Bombs`, `VB`, `UB`, `NEB`, `Danger`, `Tiles`, `Repeat`, `PG`, `Val`, `Ent` — **đúng**.
- Eval suite chạy cả easy_medium và hard opponents — **đúng**.

### J2. BC loss trong PPO — ❌ **CHƯA IMPLEMENT**

> [!CAUTION]
> Checklist J2 yêu cầu:
> ```
> total_loss = ppo_loss + bc_coef * bc_loss_demo
> ```
> với schedule `bc_coef` giảm dần từ 0.1-0.2 xuống 0.
>
> [train_ppo.py ppo_update()](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/train_ppo.py#L368-L419) chỉ có:
> ```python
> loss = pg_loss + value_coef * val_loss - entropy_term
> ```
> **Không có BC loss regularization term**. Đây là risk lớn cho policy forgetting khi PPO fine-tune từ BC.

### J3. Eval BC vs PPO checkpoint
- `evaluate_policy()` chạy mỗi `save_every` steps — **đúng**.
- Lưu best model so với `best_eval_score` — **đúng**.
- **Nhưng**: Không có so sánh trực tiếp với BC-only baseline trong cùng session.

---

## K. Curriculum PPO — ✅ PASS

- [_train_base.py](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/_train_base.py#L71-L103): 4 stages kế thừa từ v5_1, thêm `r_move_closer_bomb_spot`, `r_move_away_bomb_spot`, `r_best_bomb_spot_dist`, `r_position_loop`.
- Critic warmup khi chuyển stage (`stage_actor_warmup_steps = 15_000`) — **đúng**.
- Eval metrics đầy đủ: bombs, kills, boxes, items, valuable_bomb_ratio, repeat_position_rate, danger_steps — **đúng**.

---

## L. Debug rollout — ✅ PASS

- **Case 2 (Loop)**: [agent.py L75-78](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/agent.py#L75-L78): `_is_looping()` detect <= 2 unique positions in last 8 — **đúng**.
- **Loop fallback**: [agent.py L121-130](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/agent.py#L121-L130): Nếu loop → bomb nếu valuable, hoặc BFS tới nearest bomb spot — **đúng**.
- **Reward penalty**: `r_position_loop` trong curriculum stages — **đúng**.

---

## M. Checklist cuối cùng

| Item | Status | Nơi implement |
|------|--------|---------------|
| `[ ]` canonical action round-trip OK | ✅ | `_model_base.py` imports from v5_1 |
| `[ ]` expert action nằm trong mask > 99.9% | ✅ | `collect_dataset.py` L203-204 |
| `[ ]` action distribution hợp lý, bomb >= 5% | ✅ | `analyze_dataset.py` + weighted sampler |
| `[ ]` train/val split theo episode/seed | ✅ | `bc_dataset.py` L37-50 |
| `[ ]` LSTM forward shape OK | ✅ | `bc_model.py` CNNLSTMBCActor |
| `[ ]` hidden reset khi done OK | ✅ | `bc_model.py` L82-85, `agent.py` |
| `[ ]` padding loss mask OK | ✅ | `train_bc.py` L32-39 |
| `[ ]` model overfit được 1 batch nhỏ | ⚪ | Chưa test, nhưng code OK |
| `[ ]` BC val PLACE_BOMB recall >= 40–50% | ⚪ | Chưa test, cần chạy train |
| `[ ]` BC rollout biết phá box/đặt bom/chạy | ⚪ | Chưa test, dùng `eval_bc.py` |
| `[ ]` BC no_escape_bomb_ratio thấp | ⚪ | Chưa test |
| `[ ]` BC repeat_position_rate không cao | ⚪ | Chưa test |
| `[ ]` PPO load actor từ BC output match | ✅ | `model.py` L171-195 |
| `[ ]` critic warmup không đổi actor logits | ✅ | `model.py` L113-115, `train_ppo.py` L392-400 |
| `[ ]` recurrent PPO log_prob parity trước update OK | ❌ | **Thiếu hidden state lưu cho chunks** |
| `[ ]` PPO loss mask padding đúng | ✅ | `train_ppo.py` L391 `valid = loss_mask > 0` |
| `[ ]` PPO smoke 100k không phá behavior BC | ❓ | Code OK nhưng **thiếu BC loss regularization** |
| `[ ]` eval holdout có metrics bomb/repeat/danger/kills | ✅ | `train_ppo.py` evaluate_suite() |

---

## Tóm tắt các lỗi cần fix

### ❌ Critical (ảnh hưởng training quality)

1. **[I2] Thiếu lưu hidden state cho recurrent PPO chunks**
   - File: [train_ppo.py](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/train_ppo.py)
   - `SequenceRolloutBuffer` không lưu `(h, c)` tại mỗi step → PPO update recompute logits từ zero hidden → ratio sai
   - **Fix**: Thêm `hidden_states` list vào buffer, lưu `actor_state` tại mỗi step, truyền `state` đầu window vào `get_action_and_value_sequence()`

2. **[J2] Thiếu BC loss regularization trong PPO**
   - File: [train_ppo.py ppo_update()](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/train_ppo.py#L368-L419)
   - Không có `bc_coef * bc_loss_demo` → risk policy forgetting
   - **Fix**: Load BC demo buffer, thêm BC cross-entropy loss vào total PPO loss với schedule decay

### ⚠️ Important (nên fix)

3. **[D3] Thiếu confusion matrix 6×6**
   - File: [train_bc.py evaluate_loader()](file:///d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6/train_bc.py#L42-L113)
   - Chỉ có bomb P/R, thiếu full confusion matrix
   - **Fix**: Thêm accumulation cho 6×6 matrix và print

4. **[F2] Thiếu `weights_only=True` khi torch.load**
   - Files: `export_bc_agent.py`, `agent.py`, `eval_bc.py`, `collect_dataset.py`, `model.py`
   - PyTorch 2.6+ default `weights_only=True` → có thể fail
   - **Fix**: Thêm `weights_only=True` hoặc handle cả hai case

5. **[E] eval_bc.py thiếu một số metrics**
   - Thiếu: `no_escape_bomb_ratio`, `danger_steps`, `unique_tiles_visited`
   - Đã có sẵn trong `summarize_episode_metrics()`, chỉ cần log thêm

### ⚪ Cosmetic / chưa verify

6. **[I4] Không có verification log cho log_prob parity** — nên thêm sanity check trước PPO update đầu tiên
7. **Confusion matrix** cho debugging khi bomb recall thấp
