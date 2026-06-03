import torch
import torch.nn.functional as F
import numpy as np

from bc_dataset import load_bc_shards, split_episode_keys, BCSequenceDataset
from bc_model import CNNLSTMBCActor


ACTION_NAMES = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "BOMB"]


def ce_from_logits(logits, targets, loss_mask, class_weights=None):
    b, t, a = logits.shape
    flat_logits = logits.reshape(b * t, a)
    flat_targets = targets.reshape(b * t)
    flat_mask = loss_mask.reshape(b * t) > 0

    losses = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="none",
        weight=class_weights,
    )
    return losses[flat_mask].mean()


@torch.no_grad()
def eval_batch(model, map_feats, aux_feats, actions, loss_mask, episode_starts):
    logits, _ = model.forward_sequence(
        map_feats,
        aux_feats,
        action_mask_seq=None,
        episode_start_mask=episode_starts,
    )
    pred = logits.argmax(dim=-1)

    valid = loss_mask > 0
    correct = ((pred == actions) & valid).sum().item()
    total = valid.sum().item()

    acc = correct / max(total, 1)

    pred_flat = pred[valid].reshape(-1).cpu()
    true_flat = actions[valid].reshape(-1).cpu()

    pred_counts = torch.bincount(pred_flat, minlength=6)
    true_counts = torch.bincount(true_flat, minlength=6)

    bomb_true = (true_flat == 5)
    bomb_pred = (pred_flat == 5)
    bomb_tp = ((true_flat == 5) & (pred_flat == 5)).sum().item()

    bomb_precision = bomb_tp / max(bomb_pred.sum().item(), 1)
    bomb_recall = bomb_tp / max(bomb_true.sum().item(), 1)

    loss = ce_from_logits(logits, actions, loss_mask)

    return {
        "loss": loss.item(),
        "acc": acc,
        "pred_counts": pred_counts.tolist(),
        "true_counts": true_counts.tolist(),
        "bomb_precision": bomb_precision,
        "bomb_recall": bomb_recall,
    }


data_dir = "agent/agent/v6/bc_data_smoke/farm"
seq_len = 1
stride = 1
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

arrays = load_bc_shards(data_dir)
train_keys, val_keys, test_keys = split_episode_keys(arrays["episode_keys"], seed=seed)
ds = BCSequenceDataset(arrays, train_keys, seq_len=seq_len, stride=stride)

rng = np.random.default_rng(seed)
indices = np.arange(len(ds))
rng.shuffle(indices)
chosen = indices[:256].tolist()

# fixed batch: 32 samples đầu trong chosen set
batch_indices = chosen[:32]
samples = [ds[i] for i in batch_indices]

def stack(key):
    return torch.stack([s[key] for s in samples], dim=0)

map_feats = stack("map_feats").to(device).float()
aux_feats = stack("aux_feats").to(device).float()
actions = stack("actions").to(device).long()
loss_mask = stack("loss_mask").to(device).float()
episode_starts = stack("episode_starts").to(device).bool()

class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.5], device=device)

print("device:", device)
print("map_feats:", tuple(map_feats.shape))
print("actions bincount:", torch.bincount(actions.reshape(-1).cpu(), minlength=6).tolist())

model = CNNLSTMBCActor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(1, 501):
    model.train()

    logits, _ = model.forward_sequence(
        map_feats,
        aux_feats,
        action_mask_seq=None,
        episode_start_mask=episode_starts,
    )

    loss = ce_from_logits(logits, actions, loss_mask, class_weights)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        stats = eval_batch(model, map_feats, aux_feats, actions, loss_mask, episode_starts)
        print(
            f"step {step:4d} | "
            f"loss {stats['loss']:.4f} | "
            f"acc {stats['acc']:.2%} | "
            f"bomb P/R {stats['bomb_precision']:.2%}/{stats['bomb_recall']:.2%} | "
            f"pred {stats['pred_counts']} | true {stats['true_counts']}"
        )