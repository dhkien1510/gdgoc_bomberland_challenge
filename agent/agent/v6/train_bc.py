"""
Train a CNN+LSTM behavioral cloning actor on v6 BC shards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from bc_dataset import BCSequenceDataset, load_bc_shards, split_episode_keys
from bc_model import CNNLSTMBCActor

ACTION_NAMES = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "BOMB"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, nargs="+", required=True)
    parser.add_argument("--scenario_weight", type=float, nargs="+", default=None)
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parent / "bc_actor.pth"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bomb_weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_scenario_weights(data_dirs: list[str], scenario_weights: list[float] | None):
    if scenario_weights is None:
        return [1.0] * len(data_dirs)
    if len(scenario_weights) == 1 and len(data_dirs) > 1:
        return [float(scenario_weights[0])] * len(data_dirs)
    if len(scenario_weights) != len(data_dirs):
        raise ValueError(
            f"--scenario_weight expects either 1 value or one per --data_dir. "
            f"Got {len(scenario_weights)} weights for {len(data_dirs)} data dirs."
        )
    return [float(value) for value in scenario_weights]


def build_multi_dir_datasets(data_dirs: list[str], scenario_weights: list[float], seq_len: int, stride: int, seed: int):
    train_datasets = []
    val_datasets = []
    scenario_infos = []

    for idx, data_dir in enumerate(data_dirs):
        arrays = load_bc_shards(data_dir)
        train_keys, val_keys, _test_keys = split_episode_keys(arrays["episode_keys"], seed=seed + idx)
        train_ds = BCSequenceDataset(arrays, train_keys, seq_len=seq_len, stride=stride)
        val_ds = BCSequenceDataset(arrays, val_keys, seq_len=seq_len, stride=stride)
        scenario_name = Path(data_dir).name

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        scenario_infos.append(
            {
                "name": scenario_name,
                "data_dir": data_dir,
                "weight": scenario_weights[idx],
                "train_sequences": len(train_ds),
                "val_sequences": len(val_ds),
            }
        )

    return train_datasets, val_datasets, scenario_infos


def make_balanced_concat_sampler(datasets: list[BCSequenceDataset], scenario_weights: list[float]):
    sample_weights = []
    for dataset, scenario_weight in zip(datasets, scenario_weights):
        if len(dataset) == 0:
            continue
        bucket_weights = dataset.bucket_weights()
        scenario_base = float(scenario_weight) / float(len(dataset))
        sample_weights.extend((scenario_base * bucket_weights).tolist())
    weights = torch.tensor(sample_weights, dtype=torch.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


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
    confusion = torch.zeros((6, 6), dtype=torch.int64)

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
        valid_targets = actions[valid].detach().cpu()
        valid_preds = preds[valid].detach().cpu()
        for target, pred in zip(valid_targets.tolist(), valid_preds.tolist()):
            confusion[target, pred] += 1

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
        "confusion_matrix": confusion.numpy(),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scenario_weights = normalize_scenario_weights(args.data_dir, args.scenario_weight)
    train_datasets, val_datasets, scenario_infos = build_multi_dir_datasets(
        args.data_dir,
        scenario_weights,
        seq_len=args.seq_len,
        stride=args.stride,
        seed=args.seed,
    )

    train_ds = ConcatDataset(train_datasets)
    val_ds = ConcatDataset(val_datasets)
    train_sampler = make_balanced_concat_sampler(train_datasets, scenario_weights)

    print("Scenario sampling plan:")
    total_scenario_weight = sum(scenario_weights)
    for info in scenario_infos:
        target_ratio = info["weight"] / max(total_scenario_weight, 1e-8)
        print(
            f"  {info['name']}: target {target_ratio:.2%} | "
            f"train seq {info['train_sequences']:,} | val seq {info['val_sequences']:,} | "
            f"source {info['data_dir']}"
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=0)
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
        cm = val_stats["confusion_matrix"]
        cm_rows = " | ".join(
            f"{ACTION_NAMES[row]}:{','.join(str(int(v)) for v in cm[row])}"
            for row in range(cm.shape[0])
        )
        print(f"  -> Confusion {cm_rows}")

        if score >= best_score:
            best_score = score
            torch.save(model.state_dict(), output_path)
            print(f"  -> saved best BC actor to {output_path}")


if __name__ == "__main__":
    main()
