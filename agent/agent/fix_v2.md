Dưới đây là bản tổng hợp toàn diện và tường tận những điểm nghẽn, lỗi logic cốt lõi trong phần cài đặt PPO (`model.py` và `train_ppo.py`) cùng phương án đại tu hệ thống phần thưởng (Reward Shaping) dành riêng cho mô hình của bạn. Phần này sẽ hoàn toàn lược bỏ mảng Học bắt chước (Imitation Learning) để bạn tập trung tối ưu hóa thuật toán tăng cường thuần túy.

---

## PHẦN 1: Lỗi Kiến trúc mạng & Xử lý trạng thái (`model.py`)

### 1. Bất đối xứng góc nhìn (The Corner Asymmetry)

* 
**Vấn đề:** Trận đấu sắp xếp 4 Agent ở 4 góc tuyệt đối của bản đồ: Agent 0 ở $(1, 1)$, Agent 1 ở $(11, 11)$, v.v.. Khi bạn truyền ma trận thô vào CNN, tọa độ này mang tính tuyệt đối. Nếu Agent đóng vai Agent 0, hành động đúng để vào trung tâm là di chuyển **Xuống** và **Phải**. Nhưng ở lượt sau, nếu mạng đóng vai Agent 1, hành động đúng lại là **Lên** và **Trái**. Trọng số mạng CNN sau khi `Flatten` đi vào lớp `Linear` sẽ liên tục bị xung đột và triệt tiêu lẫn nhau vì cùng một mục tiêu (vào giữa) nhưng phân phối không gian lại đảo ngược.


* 
**Giải pháp:** Áp dụng cơ chế **Perspective Normalization** (Agent-Centric). Trước khi đưa Tensor vào CNN, dựa vào `agent_id` để lật hoặc xoay ma trận sao cho từ góc nhìn của mạng Neural, **Agent của bạn luôn nằm ở góc Top-Left $(1, 1)$**.


* 
**Lưu ý kỹ thuật:** Khi mạng xuất ra hành động, bạn phải thực hiện phép ánh xạ ngược (Inverse Mapping) để chuyển hành động từ góc nhìn Normalization về hành động thực tế gửi cho Engine môi trường. Kỹ thuật này giúp không gian mẫu cần học thu hẹp đi **4 lần**.



### 2. Nghẽn cổ chai dữ liệu không gian (CNN Bottleneck)

* 
**Vấn đề:** Hãy nhìn vào lớp nén phẳng của bạn: `nn.Flatten()` nén toàn bộ thông tin từ $64 \times 13 \times 13$ thành một vector phẳng $10,816$ chiều, rồi ép đột ngột xuống $512$ chiều qua một tầng `nn.Linear` duy nhất. Phép nén tuyến tính quá gắt này phá hủy các đặc trưng quan hệ không gian cục bộ ngắn hạn (ví dụ: khoảng cách giữa mình và quả bom ngay cạnh).


* 
**Giải pháp:** Cho phép các lớp Conv giảm chiều không gian một cách mượt mà. Hãy thêm `stride=2` hoặc tích hợp thêm tầng `nn.MaxPool2d` giữa các lớp CNN sao cho ma trận đầu ra cuối cùng trước khi Flatten chỉ còn kích thước khoảng $3 \times 3$ hoặc $5 \times 5$.



### 3. Mô hình thuần phản xạ (Thiếu bộ nhớ LSTM)

* 
**Vấn đề:** Game Bomberman mang đặc trưng **Trì hoãn phần thưởng (Delayed Reward)** rất nặng. Quả bom sau khi đặt phải mất 7 bước thời gian mới phát nổ để mang lại kết quả. Mạng MLP/CNN thuần túy (như mô hình hiện tại của bạn) chỉ có thể nhìn nhận trạng thái tĩnh tại bước thời gian hiện tại, hoàn toàn không có khả năng liên kết hành động đặt bom ở quá khứ với phần thưởng ở tương lai.


* 
**Giải pháp:** Chèn một tầng `nn.LSTM` (hoặc `nn.GRU`) vào giữa khối trích xuất đặc trưng CNN và hai đầu ra Actor/Critic. Bộ nhớ của mạng tuần hoàn sẽ lưu giữ chuỗi hành động qua các cổng (gates), giúp mạng Critic đánh giá đúng giá trị trạng thái dài hạn.



---

## PHẦN 2: Lỗi thuật toán & Chiến lược huấn luyện PPO (`train_ppo.py`)

### 1. "Over-masking" — Triệt tiêu tín hiệu học tập của PPO

* 
**Vấn đề:** Hàm `has_escape_after_placing_bomb` chạy thuật toán BFS dài tới 7 bước để chặn Agent đặt bom nếu không tìm thấy đường thoát. Action Masking quá chặt chẽ ở giai đoạn đầu khiến Agent bị "bọc trong lồng kính". Khi Agent luôn bị ép chọn hành động an toàn bởi một thuật toán heuristic có sẵn, mạng Neural không bao giờ trải nghiệm lỗi sai thực tế, dẫn đến mạng Critic không có cơ hội nhận tín hiệu phạt lỗi sai biệt thời gian (`TD-error`) để tự rút ra bài học thế nào là nguy hiểm.


* 
**Giải pháp:** Nới lỏng Mask trong $400,000$ bước huấn luyện đầu tiên. Chỉ nên mask những hành động chắc chắn đâm vào tường hoặc đi trực tiếp vào ngọn lửa đang nổ ngay ở bước tiếp theo ($t+1$). Hãy để thuật toán PPO tự học cách tính toán rủi ro dài hạn thông qua hàm phần thưởng phạt.



### 2. Biên độ phần thưởng quá lớn (Reward Scale Variance)

* 
**Vấn đề:** Hiện tại, điểm sống sót bước đi là `+0.0001` nhưng điểm giết địch là `+15.0`, và chênh lệch Rank Rewards lên tới `60.0` điểm ($+30.0$ và $-30.0$). Thang đo chênh lệch quá lớn này khiến hàm Critic ($V(s)$) bị sốc Gradient khi Agent đang nhận chuỗi reward tuần tự rất nhỏ bỗng dính cú hích cực đại ở cuối trận. Hệ quả là `val_loss` tăng vọt, kéo theo `pg_loss` bị cập nhật quá đà, bẻ gãy các kỹ năng cơ bản mạng vừa mới tối ưu.


* 
**Giải pháp:** Khống chế dải điểm (Scale) nằm trong khoảng hẹp tĩnh từ $-2.0$ đến $+2.0$ giống như thiết kế của các bài báo khoa học chuẩn.



---

## PHẦN 3: Đại tu hệ thống phần thưởng (Reward System Overhaul)

Hàm phần thưởng cũ của bạn quá thưa thớt (Sparse Reward). Chúng ta cần chuyển dịch sang **Phần thưởng dày đặc (Dense Rewards)** để định hướng Agent di chuyển, săn mồi và né bom một cách chi tiết.

Dưới đây là bảng thiết kế hệ thống phần thưởng tối ưu được chuẩn hóa biên độ hẹp mà bạn cần thay thế vào hàm `compute_reward`:

| Tên sự kiện / Trạng thái | Giá trị Reward mới | Mục tiêu định hướng hành vi cho Agent |
| --- | --- | --- |
| **Agent bị chết (Death)** | `-0.5` | Phạt nặng nhưng vừa đủ để không làm sập Gradient của Critic.

 |
| **Tiêu diệt 1 kẻ địch** | `+1.0` | Thưởng thúc đẩy tính tấn công chủ động thay vì trốn chạy.

 |
| **Thắng trận (Last Man Standing)** | `+1.0` | Mục tiêu tối thượng cuối cùng của tập episodic.

 |
| **Phá hủy thành công 1 hộp** | `+0.1` | Khuyến khích mở đường giải phóng không gian và tìm item.

 |
| **Đạt vị trí gần địch nhất lịch sử** | `+0.1` | Phá vỡ trạng thái an toàn cục bộ, thúc đẩy đi săn lùng.

 |
| **Lại gần đối thủ hơn bước trước** | `+0.002` | Tinh chỉnh hành vi di chuyển (tính bằng khoảng cách Manhattan).

 |
| **Đi ra xa đối thủ hơn bước trước** | `-0.002` | Phạt hành vi bỏ chạy thụ động hoặc đi vòng vô nghĩa.

 |
| **Hình phạt mỗi bước (Step Penalty)** | `-0.01` | **Quan trọng:** Thay thế hoàn toàn cho `+0.0001` cũ. Ép Agent phải dứt điểm trận đấu nhanh.

 |
| **Đứng trong ô nằm trong tầm bom** | `-0.000666` | Cảnh báo chủ động cho mạng Critic về vùng nguy hiểm sắp nổ.

 |
| **Né vào ô an toàn khi có bom** | `+0.002` | Thưởng thúc đẩy hành vi tìm chỗ nấp (cover) khi bom đếm ngược.

 |

---

### Đoạn mã gợi ý tích hợp tính toán khoảng cách Manhattan vào `train_ppo.py`

Bạn có thể đưa logic này vào để tính toán phần thưởng định hướng không gian cho Agent:

```python
# Tọa độ hiện tại và quá khứ của Agent
my_pos = (int(obs["players"][agent_id][0]), int(obs["players"][agent_id][1]))
prev_my_pos = (int(prev_obs["players"][agent_id][0]), int(prev_obs["players"][agent_id][1]))

def get_min_manhattan_dist(players_matrix, current_agent_pos):
    min_dist = 999.0
    for idx, p in enumerate(players_matrix):
        if idx != agent_id and bool(p[2]): # Đối thủ còn sống
            dist = abs(current_agent_pos[0] - int(p[0])) + abs(current_agent_pos[1] - int(p[1]))
            if dist < min_dist:
                min_dist = dist
    return min_dist

curr_enemy_dist = get_min_manhattan_dist(obs["players"], my_pos)
prev_enemy_dist = get_min_manhattan_dist(prev_obs["players"], prev_my_pos)

# Áp dụng phần thưởng dẫn đường ngắn hạn
if curr_enemy_dist < prev_enemy_dist:
    reward += 0.002   # Đi lại gần địch hơn
elif curr_enemy_dist > prev_enemy_dist:
    reward += -0.002  # Đi lùi xa địch

```

Bằng việc chuẩn hóa lại mô hình và áp dụng tập phần thưởng dày đặc này, đồ thị lỗi `val_loss` của mô hình sẽ hội tụ mượt mà và bền vững hơn rất nhiều.

