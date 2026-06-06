Dưới đây là pipeline kiểm tra theo kiểu **từ thấp lên cao**: nếu fail ở tầng dưới thì không cần chạy tầng trên. Mục tiêu là phát hiện lỗi implement BC/PPO trước khi bạn tốn 3–10 giờ train.

---

# A. Kiểm tra preprocessing/action trước tiên

## A1. Test canonical action

Mục tiêu: đảm bảo label BC không bị sai hướng ở agent 1/2/3.

Tạo test nhỏ:

```python
for agent_id in range(4):
    for env_action in range(6):
        canonical = to_canonical_action(env_action, agent_id)
        restored = to_env_action(canonical, agent_id)
        assert restored == env_action, (agent_id, env_action, canonical, restored)
print("canonical action round-trip OK")
```

Nếu fail ở đây, BC sẽ học sai hướng di chuyển.

## A2. Test action label từ teacher

Khi collect BC, teacher trả `env_action`. Dataset phải lưu:

```python
canonical_action = to_canonical_action(env_action, agent_id)
```

Sau đó khi inference:

```python
env_action = to_env_action(canonical_action, agent_id)
```

Bạn nên log 20 step đầu của 4 agent_id:

```text
agent_id, env_action_teacher, canonical_label, restored_env_action
```

`restored_env_action` phải bằng `env_action_teacher`.

## A3. Test mask có chứa expert action

Với mỗi sample BC:

```python
assert action_mask[canonical_action] == True
```

Nếu tỷ lệ fail > 0.1% thì có mismatch giữa teacher action và model mask.

Log:

```text
expert_action_mask_violation_rate
```

Mục tiêu:

```text
< 0.1%
```

Nếu cao, nguyên nhân thường là: collect dùng mask khác train, canonical sai, hoặc teacher chọn action mà mask PPO không cho phép.

---

# B. Kiểm tra dataset BC

## B1. Thống kê action distribution

Sau collect, in:

```text
STOP
LEFT
RIGHT
UP
DOWN
PLACE_BOMB
```

Mục tiêu tương đối:

```text
PLACE_BOMB: 5%–15%
MOVE total: 65%–85%
STOP: 5%–20%
```

Nếu `PLACE_BOMB < 2%`, BC rất dễ học không đặt bom. Nếu `PLACE_BOMB > 25%`, dataset có thể quá lệch về bomb.

## B2. Thống kê tactical state distribution

Nếu bạn đã thêm feature như v5_1/v5_2, log:

```text
can_escape_if_place %
can_hit_if_place %
boxes_hit_if_place > 0 %
danger_time <= 2 %
attack_pressure %
same_line_clear %
valuable_bomb_spot_exists %
```

V5_1 trước đó đã đưa nhiều tactical aux như `can_escape_if_place`, `can_hit_if_place`, `boxes_hit_if_place`, `attack_pressure`, `same_line_clear`; đây là đúng hướng vì reward/mask dùng chính các khái niệm này. 

Mục tiêu: dataset phải có đủ danger/bomb/pressure state, không chỉ đi bộ bình thường.

## B3. Split theo episode/seed, không split theo step

Đảm bảo:

```text
train episodes != val episodes != test episodes
```

Không random split từng transition, vì như vậy cùng episode rơi vào cả train/val và val accuracy bị ảo.

---

# C. Unit test model BC + LSTM

## C1. Test forward shape

Với batch giả:

```python
B = 4
T = 16
map_feats = torch.randn(B, T, C, 13, 13)
aux_feats = torch.randn(B, T, AUX_DIM)
action_masks = torch.ones(B, T, 6).bool()

logits, hidden = model(map_feats, aux_feats)
assert logits.shape == (B, T, 6)
```

Nếu có critic:

```python
assert values.shape == (B, T)
```

## C2. Test hidden reset

Với LSTM, khi episode done, hidden state phải reset.

Test đơn giản:

```python
# chạy sequence A rồi B với reset
# chạy sequence A rồi B không reset
# output của B phải khác nếu không reset, nhưng inference thật phải reset khi done
```

Trong rollout eval, log:

```text
hidden_reset_on_done = True
```

Nếu không reset, model sẽ mang memory từ episode trước sang episode sau.

## C3. Test sequence padding/loss mask

Nếu sequence cuối ngắn hơn `seq_len`, loss ở padding phải bằng 0.

Kiểm tra:

```python
loss = (ce * loss_mask).sum() / loss_mask.sum()
```

Không dùng CE trực tiếp trên padding action.

---

# D. Kiểm tra training BC

## D1. Overfit một batch nhỏ

Lấy 256–1024 sequence, train 200–1000 update.

Kỳ vọng:

```text
train loss giảm rất mạnh
train action accuracy > 95%
PLACE_BOMB recall trên batch nhỏ > 90%
```

Nếu không overfit được batch nhỏ, lỗi nằm ở model/loss/mask/data.

## D2. Train smoke dataset

Train trên 2k episodes.

Theo dõi:

```text
train CE
val CE
overall accuracy
PLACE_BOMB precision
PLACE_BOMB recall
danger-state accuracy
valuable-bomb-state accuracy
```

Mục tiêu ban đầu:

```text
overall accuracy: không quá quan trọng
PLACE_BOMB recall: >= 40–50%
danger-state accuracy: tăng dần
val CE giảm ổn định
```

Nếu overall accuracy cao nhưng bomb recall thấp, model đang học “chỉ di chuyển”.

## D3. Confusion matrix

In confusion matrix 6×6:

```text
expert action vs predicted action
```

Đặc biệt xem hàng `PLACE_BOMB`:

```text
expert PLACE_BOMB → predicted PLACE_BOMB bao nhiêu %
```

Nếu expert bomb thường bị predict thành STOP/MOVE, cần:

```text
- tăng class weight cho PLACE_BOMB
- oversample sequence có bomb
- collect thêm farm/pressure states
```

---

# E. Rollout eval cho BC

Đây là phần quan trọng nhất. Đừng chỉ nhìn accuracy.

Chạy local match với BC agent:

```bash
python scripts/participant/run_local_match.py \
  --agent_paths agent/agent/bc_lstm SimpleRuleAgent SmarterRuleAgent TacticalRuleAgent \
  --num_episodes 20 \
  --max_steps 500 \
  --seed 1
```

Không visualize trước; chạy nhiều episode lấy metric.

Log cho mỗi episode:

```text
rank / points
bombs_per_episode
boxes_per_episode
items_per_episode
kills_per_episode
valuable_bomb_ratio
useless_bomb_ratio
no_escape_bomb_ratio
danger_steps
unique_tiles_visited
repeat_position_rate
episode_length
```

V5_1 đã có các metric kiểu `bombs_per_episode`, `valuable_bomb_ratio`, `useless_bomb_ratio`, `no_escape_bomb_ratio`, `danger_steps`, `unique_tiles`, `repeat_position_rate`; bạn nên dùng lại cho BC eval.

## BC đạt yêu cầu nếu:

```text
bombs_per_episode >= 1.5
boxes_per_episode > 0
no_escape_bomb_ratio gần 0
repeat_position_rate không cao bất thường
unique_tiles_visited đủ lớn
visualize thấy: đi tới box → đặt bom → chạy → quay lại item/target
```

Nếu BC không đạt các tiêu chí này, **chưa chuyển sang PPO**.

---

# F. Kiểm tra BC inference parity

Đây là lỗi rất hay gặp: train tốt nhưng agent.py chạy khác.

## F1. Dùng cùng preprocessing

Trong collect, train, inference phải dùng cùng:

```python
prepare_policy_inputs(...)
```

Không được có bản `_model_base.py` khác nhau giữa `bc/` và `agent/agent/bc_lstm/`.

## F2. Test checkpoint load

Vì PyTorch 2.6 đổi default `weights_only=True`, bạn nên export pure weights:

```python
torch.save(model.state_dict(), "bc_lstm_weights.pth")
```

Trong agent.py:

```python
state_dict = torch.load(path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
```

Tránh lưu checkpoint có metadata numpy như lỗi bạn đã gặp trước đó.

## F3. Test deterministic vs stochastic

BC eval nên dùng deterministic:

```python
action = logits.argmax(dim=-1)
```

Nhưng nếu deterministic gây loop, thử sampling có temperature nhỏ:

```python
dist = Categorical(logits=logits / temperature)
```

với:

```text
temperature = 0.8–1.0
```

Tuy nhiên nếu phải dựa vào sampling để không loop, BC vẫn chưa ổn.

---

# G. Kiểm tra DAgger-lite nếu có

Nếu bạn đã implement DAgger-lite:

## G1. BC acts, teacher labels

Trong DAgger, state phải do BC policy tạo ra, nhưng action label phải từ TacticalRuleAgent:

```text
state: visited by BC
label: tactical_action(state)
```

Không lấy action BC làm label.

## G2. So sánh dataset trước/sau DAgger

Sau DAgger, dataset nên có nhiều state kiểu:

```text
danger recovery
bad position
near enemy
loop-prone state
```

Nếu DAgger chỉ thêm toàn state bình thường, nó không chữa distribution drift nhiều.

BC+RL paper nhấn mạnh BC dễ bị distribution drift khi policy gặp states khác training set, nên DAgger-lite là đúng hướng để giảm lỗi tích lũy. 

---

# H. Kiểm tra PPO trước khi fine-tune BC

Sau khi BC đạt rollout tốt, mới test PPO.

## H1. Load BC actor đúng chưa?

Nếu PPO model có cùng CNN/LSTM/actor head, test:

```python
bc_logits = bc_model(obs_seq)
ppo_logits = ppo_model.actor(obs_seq)
max_diff = (bc_logits - ppo_logits).abs().max()
```

Ngay sau load, `max_diff` phải rất nhỏ nếu kiến trúc giống nhau.

Nếu PPO có thêm critic nhưng actor giống BC, actor output phải match.

## H2. Critic warmup

Nếu bạn freeze actor và train critic trước, log:

```text
actor_trainable = False
policy logits không đổi
value loss giảm
```

Test:

```python
before = actor_logits(obs)
# critic warmup update
after = actor_logits(obs)
assert max_abs(before - after) < 1e-6
```

Nếu actor logits đổi trong critic warmup, freeze sai.

---

# I. Kiểm tra Recurrent PPO implementation

Nếu bạn dùng PPO + LSTM, đây là phần dễ sai nhất.

## I1. Rollout buffer phải lưu sequence, không shuffle từng step

Không được train LSTM bằng minibatch random từng transition như PPO cũ. PPO cũ của bạn shuffle index từng step trong buffer; cách đó phù hợp MLP/CNN, không phù hợp LSTM. 

Recurrent PPO phải update theo chunk:

```text
[B, T, ...]
```

Ví dụ:

```text
chunk_len = 32 hoặc 64
batch_size = số chunk
```

## I2. Lưu hidden state tại đầu chunk

Mỗi chunk cần:

```text
h0, c0
map_feats[t:t+T]
aux_feats[t:t+T]
actions
old_log_probs
returns
advantages
dones
loss_mask
```

Nếu không lưu hidden state đầu chunk, LSTM update không khớp rollout.

## I3. Reset hidden khi done

Trong rollout:

```python
if done:
    h = zeros
    c = zeros
```

Trong update sequence, khi `done[t] = True`, hidden sau đó cũng phải reset hoặc dùng episode chunk không crossing done.

Cách đơn giản nhất: **cắt sequence không vượt qua episode boundary**.

## I4. Kiểm tra log_prob parity

Ngay sau rollout, trước khi update, recompute log_prob trên cùng sequence với hidden state đã lưu.

```python
diff = new_log_prob - old_log_prob
```

Trước update, diff phải gần 0:

```text
mean abs diff < 1e-5 đến 1e-4
```

Nếu diff lớn, PPO ratio sai vì hidden/preprocessing/mask không khớp.

## I5. Advantage normalization

Normalize advantage trên valid tokens:

```python
valid_adv = advantages[loss_mask == 1]
advantages = (advantages - valid_adv.mean()) / (valid_adv.std() + 1e-8)
```

Không tính padding vào mean/std.

## I6. PPO loss phải mask padding

Policy loss, value loss, entropy đều phải nhân `loss_mask`.

```python
loss = (loss_per_token * loss_mask).sum() / loss_mask.sum()
```

---

# J. Kiểm tra PPO không phá BC

## J1. PPO smoke 50k–100k steps

Chạy PPO rất ngắn trước.

Log:

```text
avg reward
eval score
KL to old policy
entropy
value loss
BC behavior metrics
bombs_per_episode
repeat_position_rate
```

Nếu sau 50k PPO mà agent tệ hơn BC rõ rệt:

```text
lr quá cao
entropy quá cao
bc loss/kl regularization quá thấp
critic advantage nhiễu
curriculum quá khó
```

## J2. Giữ BC loss trong PPO

Trong giai đoạn đầu PPO, thêm demo BC loss:

```text
total_loss = ppo_loss + bc_coef * bc_loss_demo
```

Lịch:

```text
0–500k PPO steps:  bc_coef = 0.1–0.2
500k–1.5M:         bc_coef = 0.03–0.05
sau đó:            0.0–0.01
```

Bài Pommerman cũng không dùng PPO thuần ngay mà kết hợp imitation, reward shaping, action filter, curriculum để tránh policy forgetting/degeneration. 

## J3. Eval BC checkpoint vs PPO checkpoint

Luôn giữ BC checkpoint làm baseline:

```text
BC-only
PPO-100k
PPO-500k
PPO-1M
```

Nếu PPO không vượt BC mà còn giảm bomb quality, rollback.

---

# K. Kiểm tra curriculum PPO

PPO fine-tune không nên vào ngay Tactical/Genius/self-play.

Pipeline test:

```text
Phase 0: BC-only eval
Phase 1: PPO vs Random/Simple/BoxFarmer
Phase 2: PPO vs Simple/Smarter
Phase 3: PPO vs Smarter/Genius/Tactical
Phase 4: PPO self-play mix
```

Ở mỗi phase, tiêu chí chuyển phase không chỉ là step count, mà nên có metric:

```text
phase1 pass:
boxes_per_episode >= 2
no_escape_bomb_ratio < 5%

phase2 pass:
danger_steps giảm
repeat_position_rate thấp

phase3 pass:
kills_per_episode tăng
valuable_bomb_ratio tốt

phase4:
avg_points/unique_first tăng trên holdout
```

---

# L. Debug nếu rollout tệ

## Case 1: BC/PPO không đặt bom

Kiểm tra:

```text
PLACE_BOMB recall trên val
mask[5] trong rollout có thường True không?
expert bomb action có bị mask violation không?
bomb class weight/oversampling đủ chưa?
```

Nếu `mask[5]` thường False: value mask quá chặt hoặc agent không tới bomb spot.
Nếu `mask[5]` True nhưng model không chọn: BC imbalance hoặc PPO làm quên bomb.

## Case 2: Đi lên xuống loop

Kiểm tra:

```text
repeat_position_rate
first_action_to_bomb_spot feature có đúng không?
LSTM hidden reset đúng không?
deterministic argmax có kẹt không?
```

Thêm temporary fallback để debug:

```text
nếu loop > 8 step:
    dùng BFS tới nearest valuable bomb spot
```

Nếu fallback sửa ngay, lỗi là navigation policy chưa học đủ.

## Case 3: Đặt bom rồi chết

Kiểm tra:

```text
no_escape_bomb_ratio
has_escape_after_placing_bomb strict chưa?
danger penalty đủ chưa?
mask có cấm no-escape bomb không?
```

## Case 4: PPO làm tệ hơn BC

Kiểm tra:

```text
actor lr
entropy coef
bc_coef
target KL
critic warmup
opponent curriculum
```

---

# M. Checklist cuối cùng

Bạn có thể dùng checklist này trước khi train dài:

```text
[ ] canonical action round-trip OK
[ ] expert action nằm trong mask > 99.9%
[ ] action distribution hợp lý, bomb >= 5%
[ ] train/val split theo episode/seed
[ ] LSTM forward shape OK
[ ] hidden reset khi done OK
[ ] padding loss mask OK
[ ] model overfit được 1 batch nhỏ
[ ] BC val PLACE_BOMB recall >= 40–50%
[ ] BC rollout biết phá box/đặt bom/chạy
[ ] BC no_escape_bomb_ratio thấp
[ ] BC repeat_position_rate không cao
[ ] PPO load actor từ BC output match
[ ] critic warmup không đổi actor logits
[ ] recurrent PPO log_prob parity trước update OK
[ ] PPO loss mask padding đúng
[ ] PPO smoke 100k không phá behavior BC
[ ] eval holdout có metrics bomb/repeat/danger/kills
```

Kết luận: pipeline kiểm tra đúng nhất là **unit test canonical/mask → dataset analysis → BC overfit test → BC rollout metrics → PPO recurrent parity test → PPO smoke test**. Đừng chạy train dài nếu chưa qua các bước này, vì đa số lỗi BC/PPO Bomberland xuất hiện ở canonical action, mask mismatch, LSTM hidden reset, hoặc PPO update sequence sai.
