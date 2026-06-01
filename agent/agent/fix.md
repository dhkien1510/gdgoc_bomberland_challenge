Dưới đây là các vấn đề **sai/rất dễ làm PPO học lệch** trong implement hiện tại của bạn. Kết luận chính: mô hình không chỉ “chưa đủ mạnh”, mà đang được huấn luyện với **reward sai credit**, **self-play gần như không hoạt động**, **không action mask**, và **train lệch agent 0/top-left**, nên thắng easy bot 80% nhưng rơi mạnh khi gặp medium là rất hợp lý.

## 1. Lỗi lớn nhất: self-play pool gần như không bao giờ được thêm checkpoint

Trong `train_ppo.py`, bạn đặt:

```python
"n_steps": 2048
"save_every": 200_000
...
if global_step % CFG["save_every"] == 0:
    pool.add_checkpoint(model.state_dict())
```

Nhưng `global_step` tăng theo từng block 2048, tức là các mốc sẽ là `2048, 4096, 6144, ...`. Vì `200_000` **không chia hết cho 2048**, điều kiện `global_step % 200_000 == 0` hầu như không bao giờ đúng. Hệ quả: checkpoint không được thêm vào pool, `selfplay_prob=0.5` không có tác dụng, và training thực tế chủ yếu là đấu với baseline theo curriculum chứ không phải self-play. Phần cấu hình `n_steps`, `selfplay_prob`, `save_every` nằm ở đầu file, còn việc chỉ thêm checkpoint trong block `if global_step % save_every == 0` nằm cuối loop train.  

Sửa ngay:

```python
last_save_step = 0

# trong while loop, sau ppo_update:
if global_step - last_save_step >= CFG["save_every"]:
    last_save_step = global_step
    ...
    pool.add_checkpoint(model.state_dict())
```

Đây là lỗi ưu tiên số 1, vì nó làm toàn bộ ý tưởng “self-play PPO” bị sai thực tế.

## 2. Reward đang credit sai: agent được thưởng box/kills do người khác gây ra

Trong `compute_reward`, bạn tính:

```python
boxes_destroyed = np.sum((prev_map == 2) & (curr_map != 2))
reward += boxes_destroyed * r_box_destroy
```

Cách này thưởng cho agent của bạn **mọi box bị phá trên toàn bản đồ**, kể cả box do bot khác phá. Tương tự, phần kill:

```python
if p_prev[i][2] == 1 and p_curr[i][2] == 0:
    reward += r_kill
```

sẽ thưởng cho agent nếu bất kỳ đối thủ nào chết, kể cả chết do tự sát, do bot khác đặt bom, hoặc chết cùng lúc không liên quan đến mình. Đây là reward leakage rất nặng: mô hình học “ở gần/đợi người khác làm việc” cũng có reward. 

Với easy bot, leakage này vẫn có thể tạo win-rate cao vì easy bot hay tự chết/di chuyển kém. Nhưng sang medium bot, reward giả biến mất hoặc không đủ, nên policy sụp.

Sửa hướng:

```python
# Đừng thưởng box_destroy toàn map nếu không biết owner.
# Tối thiểu: chỉ thưởng box bị phá trong blast zone của bomb mình nổ ở step đó.
# Nếu engine không trả owner event, tự track bombs của mình và thời điểm nổ.
```

Nếu chưa track được chính xác, tốt hơn là giảm `r_box_destroy` rất thấp hoặc bỏ hẳn, rồi dùng reward terminal/rank mạnh hơn.

## 3. Item capacity reward cũng sai vì bạn dùng `bombs_left` như “capacity”

Trong guide, `players` có dạng `[row, col, alive, bombs_left, bomb_radius_bonus]`; không có trường “max capacity” riêng. `bombs_left` thay đổi khi đặt bom và khi bom nổ/trả lại capacity, nên đoạn này:

```python
prev_c = int(p_prev[agent_id][3])
curr_c = int(p_curr[agent_id][3])
if curr_r > prev_r or curr_c > prev_c:
    reward += r_item_collect
```

có thể thưởng nhầm khi bom nổ xong làm `bombs_left` tăng trở lại, không nhất thiết là nhặt item capacity.  

Sửa: chỉ thưởng item nếu vị trí trước/sau cho thấy agent đứng lên tile item, hoặc nếu engine có thống kê item collected thì dùng thống kê đó. Không nên suy ra capacity item từ `bombs_left`.

## 4. Bạn train cố định `agent_id = 0`, nhưng khi nộp agent có thể được gán id 0–3

Trong training loop bạn hard-code:

```python
agent_id = 0   # Ta train agent 0 (góc top-left)
```

Trong competition, `Agent.__init__(agent_id)` có thể nhận 0, 1, 2 hoặc 3, và guide yêu cầu dùng `self.agent_id` để lấy trạng thái đúng của agent.  

Vấn đề: policy của bạn chỉ học opening, hướng di chuyển, pattern né bom từ góc top-left. Khi được assign bottom-right/top-right/bottom-left, biểu diễn “self channel” vẫn đúng, nhưng phân phối trạng thái theo vị trí tuyệt đối khác hẳn. CNN không tự biết “đối xứng bản đồ” nếu bạn không augment/normalize.

Sửa một trong hai cách:

```python
# Cách 1: mỗi episode random agent_id cần train
agent_id = random.randint(0, 3)
opponents = [pool.get_opponent(i, global_step) for i in range(4) if i != agent_id]
```

Hoặc tốt hơn: canonicalize observation bằng rotate/flip để agent luôn nhìn mình như ở góc top-left, rồi map ngược action khi act.

## 5. Không action mask: PPO đang học trên rất nhiều action vô nghĩa/illegal

Game có 6 action, nhưng rất nhiều action không hợp lệ theo từng state: đi vào wall/box, đi vào bomb cũ, đặt bom khi `bombs_left=0`, đặt bom trên tile đã có bomb. Guide ghi rõ các ràng buộc movement và bomb placement này. 

Hiện tại actor luôn sample từ `Categorical(logits=logits)` trên đủ 6 action, không mask invalid action ở train hoặc inference. 

Hệ quả: với easy bot, đứng ngẫu nhiên hoặc action invalid đôi khi vẫn sống. Với medium bot, một vài action invalid/stuck là chết hoặc mất cơ hội. Đây là lý do rất hay gặp khiến PPO nhìn có vẻ train được nhưng policy yếu.

Sửa bắt buộc: tạo `valid_action_mask(obs, agent_id)` và set logits invalid thành `-1e9` cả khi train lẫn inference.

```python
def masked_logits(logits, mask):
    return logits.masked_fill(~mask.bool(), -1e9)
```

Đặc biệt action `PLACE_BOMB` nên bị mask nếu:

* `bombs_left <= 0`
* tile hiện tại đã có bomb
* không có đường thoát an toàn sau khi đặt bom, nếu bạn muốn safety mask mạnh hơn.

## 6. Danger channel đang thiếu thông tin sống còn

Trong `encode_obs`, bạn có channel bomb timer và blast zone, nhưng blast zone được gán `timer / 7.0`. Nghĩa là bom còn 7 bước có giá trị 1.0, bom còn 1 bước chỉ 0.14. Nếu coi đây là “độ nguy hiểm”, nó bị ngược trực giác: bom sắp nổ lại có giá trị nhỏ hơn. Ngoài ra blast zone không đánh dấu ô đặt bom, không mô hình hóa chain reaction, và không tách danger theo “nổ trong 1 bước / 2 bước / 3 bước”. 

Guide có chain reaction và bom nổ theo 4 hướng, dừng ở wall/box, agent trong vùng nổ bị loại ngay. 

Sửa nên làm:

```text
channel danger_t1: ô sẽ nổ trong 1 step
channel danger_t2: ô sẽ nổ trong <=2 step
channel danger_t3: ô sẽ nổ trong <=3 step
channel bomb_block: vị trí bomb đang chặn đường
channel my_bomb / enemy_bomb
channel escape_reachable_after_bomb: optional nhưng rất mạnh
```

Không nhất thiết cần LSTM ngay; chỉ cần danger map đúng và action mask đã tăng rất nhiều.

## 7. Reward terminal không khớp scoring thật của cuộc thi

Competition xếp hạng theo elimination order; nếu đến step 500, tie-break là kills, boxes destroyed, items collected, bombs placed. 

Nhưng reward của bạn chỉ có `r_win` khi `done and len(survivors)==1 and survivors[0]==agent_id`. Nếu trận bị truncate ở step 500, agent không nhận terminal reward tương ứng với rank/tie-break. 

Vậy policy không học đúng mục tiêu leaderboard. Nó có thể học “sống lâu” hoặc “ăn reward shaping”, nhưng không tối ưu thứ hạng thật.

Sửa: terminal reward nên dựa trên rank:

```python
# ví dụ
rank_reward = {
    1: +20,
    2: +5,
    3: -5,
    4: -20,
}
```

Nếu engine có stats kills/boxes/items/bombs, dùng đúng tie-break để tính rank ở truncated.

## 8. Entropy hơi cao và inference stochastic có thể làm tụt mạnh

Bạn đặt `entropy_coef = 0.05`, khá cao với action space chỉ 6 hành động. Nó ép policy giữ tính ngẫu nhiên lâu. Nếu trong `agent.py` bạn gọi `get_action_inference(..., deterministic=False)` như default trong `model.py`, agent khi thi đấu sẽ tiếp tục sample ngẫu nhiên thay vì chọn action tốt nhất.  

Sửa:

* Khi evaluate/submit: `deterministic=True`.
* Giảm entropy về `0.005` hoặc `0.01`.
* Theo dõi entropy; nếu sau train entropy vẫn gần `log(6)=1.79`, policy còn quá random.

## 9. Kiến trúc CNN flatten lớn, nhưng thiếu feature chiến thuật

Model hiện dùng 3 conv rồi flatten toàn bộ `64*13*13=10816`, nối với 2 aux features `[bombs_left, radius_bonus]`. 

Vấn đề không phải là model quá nhỏ; vấn đề là feature chưa đủ chiến thuật:

* Không có `agent_id`/canonical perspective.
* Không có opponent bombs_left/radius.
* Không có owner bomb rõ ràng.
* Không có action validity.
* Không có escape route/safe tile distance.
* Không có “đặt bom ở đây có phá box/đe dọa enemy không”.

Với Bomberman, medium bot thắng vì rule-based biết BFS né bom, tìm item, chọn chỗ đặt bom. Một CNN thuần, reward nhiễu, không mask, rất khó tự phát hiện các quy tắc này.

## 10. Hướng đi trong file tóm tắt là đúng, nhưng implement hiện tại chưa làm phần quan trọng nhất

File hướng tiếp theo của bạn đã nhắc BC pre-training và PPO+LSTM; BC giúp vượt sparse reward, còn LSTM giúp xử lý delayed reward do bom nổ sau vài bước. 

Nhưng implement hiện tại là PPO từ đầu, không BC, không LSTM, reward lại nhiễu. Vì vậy nó dễ học “mẹo thắng easy” thay vì học policy Bomberman thật.

## Thứ tự sửa khuyến nghị

1. **Sửa checkpoint/self-play trước**: dùng `global_step - last_save_step >= save_every`.
2. **Thêm action mask** ở cả train và inference.
3. **Train random/canonical agent_id**, không chỉ agent 0.
4. **Sửa reward credit**: bỏ reward box/kill nếu không xác định được owner; dùng terminal rank reward mạnh hơn.
5. **Sửa danger channels**: tách danger theo thời gian nổ, đánh dấu bomb cell, mô phỏng chain reaction.
6. **Evaluate theo ma trận rõ ràng**: agent_id 0/1/2/3 × random/simple/smarter/box_farmer/genius/tactical, không chỉ win-rate tổng.
7. Sau khi ổn định, thêm **BC từ tactical/genius** rồi fine-tune PPO.

Nhận định thẳng: kết quả 80% vs easy nhưng 25% vs medium không bất thường. Với code hiện tại, PPO đang học trong một môi trường reward bị nhiễu và curriculum bị hỏng self-play; medium bot chỉ đơn giản là đủ tốt để làm lộ các lỗi đó.
