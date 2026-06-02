"""
Train a CNN+LSTM behavioral cloning actor on v6 BC shards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bc_dataset import BCSequenceDataset, load_bc_shards, split_episode_keys
from bc_model import CNNLSTMBCActor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parent / "bc_actor.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bomb_weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def masked_ce_loss(logits, targets, loss_mask, class_weights):
    batch, seq, num_actions = logits.shape
    flat_logits = logits.reshape(batch * seq, num_actions)
    flat_targets = targets.reshape(batch * seq)
    flat_mask = loss_mask.reshape(batch * seq) > 0
    losses = F.cross_entropy(flat_logits, flat_targets, reduction="none", weight=class_weights)
    losses = losses[flat_mask]
    return losses.mean() if losses.numel() > 0 else torch.zeros((), device=logits.device)


@torch.no_grad()
def evaluate_loader(model, loader, device):
    total_loss = 0.0
    total_steps = 0
    total_correct = 0
    bomb_tp = 0
    bomb_fp = 0
    bomb_fn = 0
    danger_correct = 0
    danger_total = 0
    valuable_correct = 0
    valuable_total = 0
    illegal_before_mask = 0

    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0], device=device)
    for batch in loader:
        map_feats = batch["map_feats"].to(device)
        aux_feats = batch["aux_feats"].to(device)
        action_masks = batch["action_masks"].to(device)
        actions = batch["actions"].to(device)
        episode_starts = batch["episode_starts"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        danger_times = batch["danger_times"].to(device)
        valuable_states = batch["valuable_states"].to(device)

        raw_logits, _ = model.forward_sequence(
            map_feats,
            aux_feats,
            action_mask_seq=None,
            episode_start_mask=episode_starts,
        )
        raw_pred = raw_logits.argmax(dim=-1)
        illegal_before_mask += int(((~action_masks.gather(-1, raw_pred.unsqueeze(-1)).squeeze(-1)) & (loss_mask > 0)).sum().item())

        logits, _ = model.forward_sequence(
            map_feats,
            aux_feats,
            action_mask_seq=action_masks,
            episode_start_mask=episode_starts,
        )
        loss = masked_ce_loss(logits, actions, loss_mask, class_weights)
        preds = logits.argmax(dim=-1)
        valid = loss_mask > 0

        total_loss += float(loss.item()) * int(valid.sum().item())
        total_steps += int(valid.sum().item())
        total_correct += int(((preds == actions) & valid).sum().item())

        pred_bomb = (preds == 5) & valid
        true_bomb = (actions == 5) & valid
        bomb_tp += int((pred_bomb & true_bomb).sum().item())
        bomb_fp += int((pred_bomb & ~true_bomb).sum().item())
        bomb_fn += int((~pred_bomb & true_bomb).sum().item())

        danger_mask = (danger_times >= 0) & (danger_times <= 3) & valid
        danger_total += int(danger_mask.sum().item())
        danger_correct += int(((preds == actions) & danger_mask).sum().item())

        valuable_mask = valuable_states & valid
        valuable_total += int(valuable_mask.sum().item())
        valuable_correct += int(((preds == actions) & valuable_mask).sum().item())

    denom = max(total_steps, 1)
    return {
        "loss": total_loss / denom,
        "accuracy": total_correct / denom,
        "bomb_precision": bomb_tp / max(bomb_tp + bomb_fp, 1),
        "bomb_recall": bomb_tp / max(bomb_tp + bomb_fn, 1),
        "danger_accuracy": danger_correct / max(danger_total, 1),
        "valuable_accuracy": valuable_correct / max(valuable_total, 1),
        "illegal_before_mask_rate": illegal_before_mask / denom,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arrays = load_bc_shards(args.data_dir)
    train_keys, val_keys, _test_keys = split_episode_keys(arrays["episode_keys"], seed=args.seed)
    train_ds = BCSequenceDataset(arrays, train_keys, seq_len=args.seq_len, stride=args.stride)
    val_ds = BCSequenceDataset(arrays, val_keys, seq_len=args.seq_len, stride=args.stride)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_ds.make_sampler(),
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CNNLSTMBCActor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, args.bomb_weight], device=device)

    best_score = float("-inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen_steps = 0
        for batch in train_loader:
            map_feats = batch["map_feats"].to(device)
            aux_feats = batch["aux_feats"].to(device)
            action_masks = batch["action_masks"].to(device)
            actions = batch["actions"].to(device)
            episode_starts = batch["episode_starts"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            logits, _ = model.forward_sequence(
                map_feats,
                aux_feats,
                action_mask_seq=action_masks,
                episode_start_mask=episode_starts,
            )
            loss = masked_ce_loss(logits, actions, loss_mask, class_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += float(loss.item()) * int((loss_mask > 0).sum().item())
            seen_steps += int((loss_mask > 0).sum().item())

        train_loss = running_loss / max(seen_steps, 1)
        val_stats = evaluate_loader(model, val_loader, device)
        score = val_stats["accuracy"] + 0.25 * val_stats["bomb_recall"] - 0.2 * val_stats["illegal_before_mask_rate"]

        print(
            f"Epoch {epoch:>2} | train_loss {train_loss:.4f} | "
            f"val_loss {val_stats['loss']:.4f} | acc {val_stats['accuracy']:.2%} | "
            f"bomb P/R {val_stats['bomb_precision']:.2%}/{val_stats['bomb_recall']:.2%} | "
            f"danger {val_stats['danger_accuracy']:.2%} | valuable {val_stats['valuable_accuracy']:.2%} | "
            f"illegal_pre_mask {val_stats['illegal_before_mask_rate']:.2%}"
        )

        if score >= best_score:
            best_score = score
            torch.save(model.state_dict(), output_path)
            print(f"  -> saved best BC actor to {output_path}")


if __name__ == "__main__":
    main()
