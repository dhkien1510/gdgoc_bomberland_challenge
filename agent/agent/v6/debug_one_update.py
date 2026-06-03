import torch
import torch.nn.functional as F
import numpy as np

from bc_dataset import load_bc_shards, split_episode_keys, BCSequenceDataset
from bc_model import CNNLSTMBCActor


def ce_from_logits(logits, targets, loss_mask):
    b, t, a = logits.shape
    flat_logits = logits.reshape(b * t, a)
    flat_targets = targets.reshape(b * t)
    flat_mask = loss_mask.reshape(b * t) > 0
    losses = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return losses[flat_mask].mean()


data_dir = "agent/agent/v6/bc_data_smoke/farm"
seq_len = 1
stride = 1
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

arrays = load_bc_shards(data_dir)
train_keys, val_keys, test_keys = split_episode_keys(arrays["episode_keys"], seed=seed)
ds = BCSequenceDataset(arrays, train_keys, seq_len=seq_len, stride=stride)

# lấy đúng 256 sample giống overfit seq_len=1
rng = np.random.default_rng(seed)
indices = np.arange(len(ds))
rng.shuffle(indices)
chosen = indices[:256].tolist()

# lấy một batch nhỏ cố định
batch_indices = chosen[:32]
samples = [ds[i] for i in batch_indices]

def stack(key):
    return torch.stack([s[key] for s in samples], dim=0)

map_feats = stack("map_feats").to(device).float()
aux_feats = stack("aux_feats").to(device).float()
actions = stack("actions").to(device).long()
loss_mask = stack("loss_mask").to(device).float()
action_masks = stack("action_masks").to(device).bool()
episode_starts = stack("episode_starts").to(device).bool()

print("device:", device)
print("map_feats shape:", tuple(map_feats.shape))
print("aux_feats shape:", tuple(aux_feats.shape))
print("actions shape:", tuple(actions.shape))
print("actions bincount:", torch.bincount(actions.reshape(-1).cpu(), minlength=6).tolist())
print("loss_mask sum:", float(loss_mask.sum().item()))

model = CNNLSTMBCActor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()

with torch.no_grad():
    logits_before, _ = model.forward_sequence(
        map_feats,
        aux_feats,
        action_mask_seq=None,
        episode_start_mask=episode_starts,
    )
    pred_before = logits_before.argmax(dim=-1)
    loss_before = ce_from_logits(logits_before, actions, loss_mask)

print("loss_before:", float(loss_before.item()))
print("pred_before bincount:", torch.bincount(pred_before.reshape(-1).cpu(), minlength=6).tolist())

# lưu parameter đầu
params_before = {
    name: p.detach().clone()
    for name, p in model.named_parameters()
    if p.requires_grad
}

# train 1 update
logits, _ = model.forward_sequence(
    map_feats,
    aux_feats,
    action_mask_seq=None,
    episode_start_mask=episode_starts,
)
loss = ce_from_logits(logits, actions, loss_mask)

optimizer.zero_grad(set_to_none=True)
loss.backward()

print("loss_used_for_backward:", float(loss.item()))

# grad norm
total_grad_norm = 0.0
nonzero_grad_params = 0
for name, p in model.named_parameters():
    if p.grad is not None:
        g = float(p.grad.detach().norm().item())
        if g > 0:
            nonzero_grad_params += 1
        total_grad_norm += g
        if "actor_head" in name or "lstm" in name or "pre_lstm" in name:
            print("grad", name, g)

print("nonzero_grad_params:", nonzero_grad_params)
print("total_grad_norm_sum:", total_grad_norm)

optimizer.step()

with torch.no_grad():
    logits_after, _ = model.forward_sequence(
        map_feats,
        aux_feats,
        action_mask_seq=None,
        episode_start_mask=episode_starts,
    )
    pred_after = logits_after.argmax(dim=-1)
    loss_after = ce_from_logits(logits_after, actions, loss_mask)

print("loss_after:", float(loss_after.item()))
print("pred_after bincount:", torch.bincount(pred_after.reshape(-1).cpu(), minlength=6).tolist())

# parameter diff
max_param_diff = 0.0
changed_params = 0
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    diff = float((p.detach() - params_before[name]).abs().max().item())
    if diff > 0:
        changed_params += 1
    max_param_diff = max(max_param_diff, diff)

print("changed_params:", changed_params)
print("max_param_diff:", max_param_diff)

# logits diff
logit_diff = float((logits_after - logits_before).abs().max().item())
print("max_logit_diff_after_one_update:", logit_diff)