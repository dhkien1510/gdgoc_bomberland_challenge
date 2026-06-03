import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader, Subset

from bc_dataset import load_bc_shards, split_episode_keys, BCSequenceDataset
from bc_model import CNNLSTMBCActor


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
def eval_loader(model, loader, device):
    model.eval()

    total_correct = 0
    total_valid = 0

    true_counts = torch.zeros(6, dtype=torch.long)
    pred_counts = torch.zeros(6, dtype=torch.long)

    bomb_tp = 0
    bomb_true = 0
    bomb_pred = 0

    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        map_feats = batch["map_feats"].to(device).float()

        # DEBUG: nếu dataset đang lưu [B,T,H,W,C], đổi sang [B,T,C,H,W]
        map_feats = map_feats.permute(0, 1, 4, 2, 3).contiguous()
        aux_feats = batch["aux_feats"].to(device).float()
        actions = batch["actions"].to(device).long()
        loss_mask = batch["loss_mask"].to(device).float()
        episode_starts = batch["episode_starts"].to(device).bool()

        logits, _ = model.forward_sequence(
            map_feats,
            aux_feats,
            action_mask_seq=None,
            episode_start_mask=episode_starts,
        )

        loss = ce_from_logits(logits, actions, loss_mask)
        total_loss += float(loss.item())
        total_batches += 1

        pred = logits.argmax(dim=-1)
        valid = loss_mask > 0

        total_correct += int(((pred == actions) & valid).sum().item())
        total_valid += int(valid.sum().item())

        y = actions[valid].detach().cpu().reshape(-1)
        p = pred[valid].detach().cpu().reshape(-1)

        true_counts += torch.bincount(y, minlength=6)
        pred_counts += torch.bincount(p, minlength=6)

        bomb_tp += int(((y == 5) & (p == 5)).sum().item())
        bomb_true += int((y == 5).sum().item())
        bomb_pred += int((p == 5).sum().item())

    acc = total_correct / max(total_valid, 1)
    bomb_precision = bomb_tp / max(bomb_pred, 1)
    bomb_recall = bomb_tp / max(bomb_true, 1)

    return {
        "loss": total_loss / max(total_batches, 1),
        "acc": acc,
        "bomb_precision": bomb_precision,
        "bomb_recall": bomb_recall,
        "true_counts": true_counts.tolist(),
        "pred_counts": pred_counts.tolist(),
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

subset = Subset(ds, chosen)

loader = DataLoader(
    subset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)

eval_loader_fixed = DataLoader(
    subset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

model = CNNLSTMBCActor().to(device)
for p in model.actor_core.parameters():
    p.requires_grad = False
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-2,
)
class_weights = torch.tensor([1, 1, 1, 1, 1, 1.5], dtype=torch.float32, device=device)

print("device:", device)
print("subset size:", len(subset))

for epoch in range(1, 201):
    model.train()

    train_loss = 0.0
    batches = 0

    for batch in loader:
        map_feats = batch["map_feats"].to(device).float()

        # DEBUG: nếu dataset đang lưu [B,T,H,W,C], đổi sang [B,T,C,H,W]
        map_feats = map_feats.permute(0, 1, 4, 2, 3).contiguous()
        aux_feats = batch["aux_feats"].to(device).float()
        actions = batch["actions"].to(device).long()
        loss_mask = batch["loss_mask"].to(device).float()
        episode_starts = batch["episode_starts"].to(device).bool()

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

        train_loss += float(loss.item())
        batches += 1

    if epoch in [1, 2, 5, 10, 20, 50, 100, 150, 200]:
        stats = eval_loader(model, eval_loader_fixed, device)
        print(
            f"epoch {epoch:3d} | "
            f"train_loss {train_loss / max(batches, 1):.4f} | "
            f"eval_loss {stats['loss']:.4f} | "
            f"acc {stats['acc']:.2%} | "
            f"bomb P/R {stats['bomb_precision']:.2%}/{stats['bomb_recall']:.2%} | "
            f"pred {stats['pred_counts']} | true {stats['true_counts']}"
        )