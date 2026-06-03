import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader, Subset

from bc_dataset import load_bc_shards, split_episode_keys, BCSequenceDataset
from bc_model import CNNLSTMBCActor


def collect_embeddings(model, loader, device, use_permute=False):
    model.eval()

    xs = []
    ys = []

    with torch.no_grad():
        for batch in loader:
            map_feats = batch["map_feats"].to(device).float()

            if use_permute:
                map_feats = map_feats.permute(0, 1, 4, 2, 3).contiguous()

            aux_feats = batch["aux_feats"].to(device).float()
            actions = batch["actions"].to(device).long()
            episode_starts = batch["episode_starts"].to(device).bool()
            loss_mask = batch["loss_mask"].to(device).float()

            actor_out, _ = model.actor_core.forward_sequence(
                map_feats,
                aux_feats,
                state=None,
                episode_start_mask=episode_starts,
            )

            valid = loss_mask > 0
            xs.append(actor_out[valid].detach().cpu())
            ys.append(actions[valid].detach().cpu())

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def train_probe(x, y, steps=1000, lr=1e-2):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = x.to(device).float()
    y = y.to(device).long()

    probe = nn.Linear(x.shape[-1], 6).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    for step in range(1, steps + 1):
        logits = probe(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step in [1, 10, 50, 100, 200, 500, 1000]:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == y).float().mean().item()

                true_counts = torch.bincount(y.cpu(), minlength=6)
                pred_counts = torch.bincount(pred.cpu(), minlength=6)

                bomb_true = y == 5
                bomb_pred = pred == 5
                bomb_tp = ((y == 5) & (pred == 5)).sum().item()

                bomb_p = bomb_tp / max(bomb_pred.sum().item(), 1)
                bomb_r = bomb_tp / max(bomb_true.sum().item(), 1)

                print(
                    f"step {step:4d} | "
                    f"loss {loss.item():.4f} | "
                    f"acc {acc:.2%} | "
                    f"bomb P/R {bomb_p:.2%}/{bomb_r:.2%} | "
                    f"pred {pred_counts.tolist()} | true {true_counts.tolist()}"
                )


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
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

print("device:", device)
print("subset size:", len(subset))

for use_permute in [False, True]:
    print("\n=== use_permute =", use_permute, "===")

    model = CNNLSTMBCActor().to(device)

    x, y = collect_embeddings(
        model=model,
        loader=loader,
        device=device,
        use_permute=use_permute,
    )

    print("embedding shape:", tuple(x.shape))
    print("embedding mean/std:", float(x.mean()), float(x.std()))
    print("label counts:", torch.bincount(y, minlength=6).tolist())

    train_probe(x, y, steps=1000, lr=1e-2)