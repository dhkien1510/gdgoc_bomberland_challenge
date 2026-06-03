import torch
import numpy as np

from bc_dataset import load_bc_shards, split_episode_keys, BCSequenceDataset

data_dir = "agent/agent/v6/bc_data_smoke/farm"
seq_len = 1
stride = 1
seed = 42

arrays = load_bc_shards(data_dir)
train_keys, val_keys, test_keys = split_episode_keys(arrays["episode_keys"], seed=seed)
ds = BCSequenceDataset(arrays, train_keys, seq_len=seq_len, stride=stride)

idx = 0

a = ds[idx]
b = ds[idx]

keys = ["map_feats", "aux_feats", "action_masks", "actions", "loss_mask", "episode_starts"]

print("Dataset len:", len(ds))
print("Testing index:", idx)

for k in keys:
    if k not in a:
        print(k, "missing")
        continue

    av = a[k]
    bv = b[k]

    if torch.is_tensor(av):
        same = torch.equal(av, bv)
        max_diff = (av.float() - bv.float()).abs().max().item()
    else:
        same = np.array_equal(av, bv)
        max_diff = np.max(np.abs(np.asarray(av) - np.asarray(bv)))

    print(f"{k}: same={same}, max_diff={max_diff}")

print("action a:", a["actions"].tolist())
print("action b:", b["actions"].tolist())
print("mask a:", a["action_masks"].tolist())
print("mask b:", b["action_masks"].tolist())
print("aux a:", a["aux_feats"].flatten()[:20].tolist())
print("aux b:", b["aux_feats"].flatten()[:20].tolist())