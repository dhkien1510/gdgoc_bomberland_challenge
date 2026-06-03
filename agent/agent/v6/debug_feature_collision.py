import hashlib
import numpy as np
import torch
from collections import defaultdict, Counter

from bc_dataset import load_bc_shards, split_episode_keys, BCSequenceDataset

ACTION_NAMES = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "BOMB"]

data_dir = "agent/agent/v6/bc_data_smoke/farm"
seq_len = 1
stride = 1
seed = 42
overfit_sequences = 256

arrays = load_bc_shards(data_dir)
train_keys, val_keys, test_keys = split_episode_keys(arrays["episode_keys"], seed=seed)
ds = BCSequenceDataset(arrays, train_keys, seq_len=seq_len, stride=stride)

rng = np.random.default_rng(seed)
indices = np.arange(len(ds))
rng.shuffle(indices)
chosen = indices[:overfit_sequences].tolist()

groups = defaultdict(list)

for idx in chosen:
    s = ds[idx]

    map_feats = s["map_feats"].numpy()
    aux_feats = s["aux_feats"].numpy()
    action_masks = s["action_masks"].numpy()
    action = int(s["actions"].item())

    # Hash toàn bộ input mà model nhìn thấy.
    h = hashlib.md5()
    h.update(map_feats.tobytes())
    h.update(aux_feats.tobytes())
    h.update(action_masks.tobytes())
    key = h.hexdigest()

    groups[key].append(action)

num_unique = len(groups)
num_total = len(chosen)

conflict_groups = []
for key, actions in groups.items():
    c = Counter(actions)
    if len(c) > 1:
        conflict_groups.append((key, len(actions), c))

print("Total samples:", num_total)
print("Unique feature groups:", num_unique)
print("Duplicate samples:", num_total - num_unique)
print("Conflict groups:", len(conflict_groups))

total_conflict_samples = sum(n for _, n, _ in conflict_groups)
print("Samples inside conflict groups:", total_conflict_samples)

print("\nTop 20 conflict groups:")
conflict_groups = sorted(conflict_groups, key=lambda x: x[1], reverse=True)

for key, n, c in conflict_groups[:20]:
    readable = {ACTION_NAMES[a]: count for a, count in c.items()}
    print("n=", n, readable)

# Upper-bound accuracy nếu model chỉ thấy các feature này.
# Với mỗi feature group, tốt nhất model chọn action majority trong group.
max_correct = 0
for actions in groups.values():
    c = Counter(actions)
    max_correct += max(c.values())

print("\nTheoretical max accuracy from exact features:")
print(f"{max_correct}/{num_total} = {max_correct / num_total:.2%}")

# Action distribution
all_actions = []
for actions in groups.values():
    all_actions.extend(actions)

print("\nAction distribution:")
for a, count in Counter(all_actions).items():
    print(ACTION_NAMES[a], count)