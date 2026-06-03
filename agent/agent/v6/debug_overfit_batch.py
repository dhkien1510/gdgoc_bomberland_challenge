from pathlib import Path
import torch
import numpy as np

from bc_dataset import load_bc_shards, split_episode_keys, BCSequenceDataset

ACTION_NAMES = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "BOMB"]

data_dir = "agent/agent/v6/bc_data_smoke/farm"
seq_len = 64
stride = 64
seed = 42
overfit_sequences = 32

arrays = load_bc_shards(data_dir)
train_keys, val_keys, test_keys = split_episode_keys(arrays["episode_keys"], seed=seed)
ds = BCSequenceDataset(arrays, train_keys, seq_len=seq_len, stride=stride)

# giống logic overfit_sequences: lấy 32 sequence ngẫu nhiên cố định
rng = np.random.default_rng(seed)
indices = np.arange(len(ds))
rng.shuffle(indices)
chosen = indices[:overfit_sequences].tolist()

print("Dataset len:", len(ds))
print("Chosen indices:", chosen)

total_actions = torch.zeros(6, dtype=torch.long)
total_mask_true = torch.zeros(6, dtype=torch.long)
expert_mask_violation = 0
total_valid = 0

episode_ids = []
steps_minmax = []

for idx in chosen:
    sample = ds[idx]

    actions = sample["actions"]
    masks = sample["action_masks"]
    loss_mask = sample["loss_mask"] > 0
    episode_starts = sample["episode_starts"]

    valid_actions = actions[loss_mask]
    total_actions += torch.bincount(valid_actions, minlength=6).cpu()

    valid_masks = masks[loss_mask].bool()
    total_mask_true += valid_masks.sum(dim=0).cpu()

    ok = valid_masks.gather(-1, valid_actions.unsqueeze(-1)).squeeze(-1)
    expert_mask_violation += int((~ok).sum().item())
    total_valid += int(loss_mask.sum().item())

    if "episode_ids" in sample:
        episode_ids.append(sample["episode_ids"][loss_mask].unique().tolist())

    if "steps" in sample:
        s = sample["steps"][loss_mask]
        steps_minmax.append((int(s.min().item()), int(s.max().item())))

print("\nAction distribution in chosen 32 sequences:")
for i, name in enumerate(ACTION_NAMES):
    print(f"{name:>5}: {int(total_actions[i])}")

print("\nAction ratio:")
for i, name in enumerate(ACTION_NAMES):
    print(f"{name:>5}: {float(total_actions[i] / max(total_actions.sum(), 1)):.2%}")

print("\nMask true rate:")
for i, name in enumerate(ACTION_NAMES):
    print(f"{name:>5}: {float(total_mask_true[i] / max(total_valid, 1)):.2%}")

print("\nExpert action mask violation:", expert_mask_violation, "/", total_valid)
print("Total valid steps:", total_valid)
print("Episode starts first sample:", ds[chosen[0]]["episode_starts"][:10])
print("First sample actions:", ds[chosen[0]]["actions"][:64].tolist())
print("First sample loss_mask sum:", int((ds[chosen[0]]["loss_mask"] > 0).sum().item()))

if steps_minmax:
    print("First 10 step ranges:", steps_minmax[:10])
if episode_ids:
    print("First 10 episode_id uniques:", episode_ids[:10])