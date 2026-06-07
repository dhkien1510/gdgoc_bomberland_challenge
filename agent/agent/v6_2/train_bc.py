"""
Train a CNN+LSTM behavioral cloning actor for v6_2.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from bc_dataset import BCSequenceDataset, load_bc_shards, split_episode_keys
from bc_model import CNNLSTMBCActor

ACTION_NAMES = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "BOMB"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="public_replay_top1", choices=["public_replay_top1", "generic"])
    parser.add_argument("--data_dir", type=str, nargs="+", required=True)
    parser.add_argument("--scenario_weight", type=float, nargs="+", default=None)
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parent / "bc_actor.pth"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head_warmup_epochs", type=int, default=0)
    parser.add_argument("--head_warmup_lr", type=float, default=1e-2)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--finetune_lr", type=float, default=8e-5)
    parser.add_argument("--finetune_patience", type=int, default=6)
    parser.add_argument("--bomb_weight", type=float, default=1.25)
    parser.add_argument("--raw_ce_coef", type=float, default=0.6)
    parser.add_argument("--masked_ce_coef", type=float, default=0.4)
    parser.add_argument("--illegal_action_coef", type=float, default=0.04)
    parser.add_argument("--illegal_margin", type=float, default=0.1)
    parser.add_argument("--overfit_sequences", type=int, default=0)
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


def maybe_limit_sequences(
    datasets: list[BCSequenceDataset],
    scenario_infos: list[dict],
    overfit_sequences: int,
    seed: int,
):
    if overfit_sequences <= 0:
        return datasets, scenario_infos

    limited_datasets: list[torch.utils.data.Subset] = []
    updated_infos: list[dict] = []
    rng = np.random.default_rng(seed)

    for dataset, info in zip(datasets, scenario_infos):
        take = min(len(dataset), overfit_sequences)
        if take <= 0:
            limited_datasets.append(torch.utils.data.Subset(dataset, []))
            cloned = dict(info)
            cloned["train_sequences"] = 0
            updated_infos.append(cloned)
            continue
        indices = np.arange(len(dataset))
        rng.shuffle(indices)
        chosen = indices[:take].tolist()
        limited_datasets.append(torch.utils.data.Subset(dataset, chosen))
        cloned = dict(info)
        cloned["train_sequences"] = take
        updated_infos.append(cloned)
    return limited_datasets, updated_infos


def make_balanced_concat_sampler(datasets: list[BCSequenceDataset], scenario_weights: list[float]):
    sample_weights = []
    for dataset, scenario_weight in zip(datasets, scenario_weights):
        if len(dataset) == 0:
            continue
        if isinstance(dataset, Subset):
            if not hasattr(dataset.dataset, "bucket_weights"):
                raise TypeError("Subset base dataset must define bucket_weights()")
            bucket_weights = dataset.dataset.bucket_weights()[dataset.indices]
        else:
            bucket_weights = dataset.bucket_weights()
        scenario_base = float(scenario_weight) / float(len(dataset))
        sample_weights.extend((scenario_base * bucket_weights).tolist())
    weights = torch.tensor(sample_weights, dtype=torch.float32)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def ce_from_logits(logits, targets, loss_mask, class_weights):
    batch, seq, num_actions = logits.shape
    flat_logits = logits.reshape(batch * seq, num_actions)
    flat_targets = targets.reshape(batch * seq)
    flat_mask = loss_mask.reshape(batch * seq) > 0
    losses = F.cross_entropy(flat_logits, flat_targets, reduction="none", weight=class_weights)
    losses = losses[flat_mask]
    return losses.mean() if losses.numel() > 0 else torch.zeros((), device=logits.device)


def mask_logits(raw_logits, action_masks):
    return raw_logits.masked_fill(~action_masks.bool(), -1e9)


def invalid_action_mass_loss(raw_logits, action_masks, loss_mask):
    flat_masks = action_masks.reshape(-1, action_masks.shape[-1]).bool()
    flat_valid_steps = loss_mask.reshape(-1) > 0
    if not torch.any(flat_valid_steps):
        return torch.zeros((), device=raw_logits.device)

    probs = torch.softmax(raw_logits, dim=-1).reshape(-1, raw_logits.shape[-1])[flat_valid_steps]
    masks = flat_masks[flat_valid_steps]
    invalid_mass = (probs * (~masks).float()).sum(dim=-1)
    return invalid_mass.mean()


def configure_train_phase(model: CNNLSTMBCActor, args, epoch: int, current_phase: str | None):
    warmup_active = args.head_warmup_epochs > 0 and epoch <= args.head_warmup_epochs
    next_phase = "head_warmup" if warmup_active else "finetune"
    if next_phase == current_phase:
        return current_phase, None

    if next_phase == "head_warmup":
        for param in model.actor_core.parameters():
            param.requires_grad = False
        for param in model.actor_head.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.head_warmup_lr,
        )
    else:
        for param in model.actor_core.parameters():
            param.requires_grad = True
        for param in model.actor_head.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(
            [
                {"params": model.actor_core.parameters(), "lr": args.backbone_lr},
                {"params": model.actor_head.parameters(), "lr": args.finetune_lr},
            ]
        )

    return next_phase, optimizer


def apply_preset_defaults(args):
    if args.preset == "generic":
        return args
    # public_replay_top1: smaller but stronger dataset, so use more overlap and
    # a slightly stronger invalid/bomb weighting to preserve rare tactical
    # decisions from strong agents.
    return args


@torch.no_grad()
def evaluate_loader(
    model,
    loader,
    device,
    bomb_weight: float,
    raw_ce_coef: float,
    masked_ce_coef: float,
    invalid_action_coef: float,
):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_steps = 0
    total_correct = 0
    total_raw_correct = 0
    bomb_tp = 0
    bomb_fp = 0
    bomb_fn = 0
    pred_bomb_count = 0
    true_bomb_count = 0
    danger_correct = 0
    danger_total = 0
    valuable_correct = 0
    valuable_total = 0
    illegal_before_mask = 0
    invalid_mass_total = 0.0
    confusion = torch.zeros((6, 6), dtype=torch.int64)

    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, bomb_weight], device=device)
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
        valid = loss_mask > 0
        total_raw_correct += int(((raw_pred == actions) & valid).sum().item())

        masked_logits = mask_logits(raw_logits, action_masks)
        raw_ce = ce_from_logits(raw_logits, actions, loss_mask, class_weights)
        masked_ce = ce_from_logits(masked_logits, actions, loss_mask, class_weights)
        invalid_mass = invalid_action_mass_loss(raw_logits, action_masks, loss_mask)
        loss = raw_ce_coef * raw_ce + masked_ce_coef * masked_ce + invalid_action_coef * invalid_mass
        invalid_mass_total += float(invalid_mass.item()) * int((loss_mask > 0).sum().item())
        preds = masked_logits.argmax(dim=-1)

        total_loss += float(loss.item()) * int(valid.sum().item())
        total_steps += int(valid.sum().item())
        total_correct += int(((preds == actions) & valid).sum().item())
        valid_targets = actions[valid].detach().cpu()
        valid_preds = preds[valid].detach().cpu()
        for target, pred in zip(valid_targets.tolist(), valid_preds.tolist()):
            confusion[target, pred] += 1

        pred_bomb = (preds == 5) & valid
        true_bomb = (actions == 5) & valid
        pred_bomb_count += int(pred_bomb.sum().item())
        true_bomb_count += int(true_bomb.sum().item())
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
    stats = {
        "loss": total_loss / denom,
        "accuracy": total_correct / denom,
        "raw_accuracy": total_raw_correct / denom,
        "bomb_precision": bomb_tp / max(bomb_tp + bomb_fp, 1),
        "bomb_recall": bomb_tp / max(bomb_tp + bomb_fn, 1),
        "pred_bomb_rate": pred_bomb_count / denom,
        "true_bomb_rate": true_bomb_count / denom,
        "danger_accuracy": danger_correct / max(danger_total, 1),
        "valuable_accuracy": valuable_correct / max(valuable_total, 1),
        "illegal_before_mask_rate": illegal_before_mask / denom,
        "invalid_action_mass": invalid_mass_total / denom,
        "confusion_matrix": confusion.numpy(),
    }
    if was_training:
        model.train()
    return stats


def main():
    args = parse_args()
    args = apply_preset_defaults(args)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.illegal_margin != 0.1:
        warnings.warn(
            "--illegal_margin is ignored in the current BC objective. "
            "The old margin loss was removed in favor of invalid_action_mass_loss.",
            stacklevel=2,
        )

    scenario_weights = normalize_scenario_weights(args.data_dir, args.scenario_weight)
    train_datasets, val_datasets, scenario_infos = build_multi_dir_datasets(
        args.data_dir,
        scenario_weights,
        seq_len=args.seq_len,
        stride=args.stride,
        seed=args.seed,
    )
    train_datasets, scenario_infos = maybe_limit_sequences(
        train_datasets,
        scenario_infos,
        overfit_sequences=args.overfit_sequences,
        seed=args.seed,
    )

    train_ds = ConcatDataset(train_datasets)
    val_ds = ConcatDataset(val_datasets)
    train_sampler = None if args.overfit_sequences > 0 else make_balanced_concat_sampler(train_datasets, scenario_weights)

    print("Scenario sampling plan:")
    total_scenario_weight = sum(scenario_weights)
    for info in scenario_infos:
        target_ratio = info["weight"] / max(total_scenario_weight, 1e-8)
        print(
            f"  {info['name']}: target {target_ratio:.2%} | "
            f"train seq {info['train_sequences']:,} | val seq {info['val_sequences']:,} | "
            f"source {info['data_dir']}"
        )
    if args.overfit_sequences > 0:
        print(f"Overfit debug mode: limiting each scenario to <= {args.overfit_sequences} train sequences")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=0,
    )
    if args.overfit_sequences > 0:
        print("\nDEBUG first 5 train batches:")

        for bi, batch in zip(range(5), train_loader):
            actions = batch["actions"]
            loss_mask = batch["loss_mask"] > 0

            counts = torch.bincount(
                actions[loss_mask].reshape(-1).cpu(),
                minlength=6,
            )

            print(
                f"batch {bi} | "
                f"valid={int(loss_mask.sum().item())} | "
                f"action_counts={counts.tolist()}"
            )

        print("DEBUG end first 5 train batches\n")

    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CNNLSTMBCActor().to(device)
    if args.head_warmup_epochs <= 0 and args.lr != 3e-4:
        warnings.warn(
            "--lr is ignored unless --head_warmup_epochs=0 and legacy single-phase training is restored. "
            "Use --head_warmup_lr, --backbone_lr, and --finetune_lr instead.",
            stacklevel=2,
        )
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, args.bomb_weight], device=device)

    best_score = float("-inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    head_warmup_output_path = output_path.with_name(f"{output_path.stem}_head_warmup{output_path.suffix}")
    finetune_output_path = output_path.with_name(f"{output_path.stem}_finetune{output_path.suffix}")
    phase_name = None
    optimizer = None
    best_phase_scores = {
        "head_warmup": float("-inf"),
        "finetune": float("-inf"),
    }
    best_phase_epochs = {
        "head_warmup": None,
        "finetune": None,
    }
    finetune_epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        phase_name, maybe_optimizer = configure_train_phase(model, args, epoch, phase_name)
        if maybe_optimizer is not None:
            optimizer = maybe_optimizer
            print(
                f"Entering phase: {phase_name} | "
                f"head_warmup_epochs={args.head_warmup_epochs} | "
                f"head_warmup_lr={args.head_warmup_lr:.2e} | "
                f"backbone_lr={args.backbone_lr:.2e} | finetune_lr={args.finetune_lr:.2e}"
            )

        model.train()
        running_loss = 0.0
        running_raw_ce = 0.0
        running_masked_ce = 0.0
        running_invalid_mass = 0.0
        seen_steps = 0
        for batch in train_loader:
            map_feats = batch["map_feats"].to(device)
            aux_feats = batch["aux_feats"].to(device)
            action_masks = batch["action_masks"].to(device)
            actions = batch["actions"].to(device)
            episode_starts = batch["episode_starts"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            raw_logits, _ = model.forward_sequence(
                map_feats,
                aux_feats,
                action_mask_seq=None,
                episode_start_mask=episode_starts,
            )
            masked_logits = mask_logits(raw_logits, action_masks)
            raw_ce = ce_from_logits(raw_logits, actions, loss_mask, class_weights)
            masked_ce = ce_from_logits(masked_logits, actions, loss_mask, class_weights)
            invalid_loss = invalid_action_mass_loss(
                raw_logits,
                action_masks,
                loss_mask,
            )
            loss = (
                args.raw_ce_coef * raw_ce
                + args.masked_ce_coef * masked_ce
                + args.illegal_action_coef * invalid_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += float(loss.item()) * int((loss_mask > 0).sum().item())
            running_raw_ce += float(raw_ce.item()) * int((loss_mask > 0).sum().item())
            running_masked_ce += float(masked_ce.item()) * int((loss_mask > 0).sum().item())
            running_invalid_mass += float(invalid_loss.item()) * int((loss_mask > 0).sum().item())
            seen_steps += int((loss_mask > 0).sum().item())

        train_loss = running_loss / max(seen_steps, 1)
        train_raw_ce = running_raw_ce / max(seen_steps, 1)
        train_masked_ce = running_masked_ce / max(seen_steps, 1)
        train_invalid_mass = running_invalid_mass / max(seen_steps, 1)
        train_eval_stats = None
        if args.overfit_sequences > 0:
            train_eval_stats = evaluate_loader(
                model,
                train_eval_loader,
                device,
                bomb_weight=args.bomb_weight,
                raw_ce_coef=args.raw_ce_coef,
                masked_ce_coef=args.masked_ce_coef,
                invalid_action_coef=args.illegal_action_coef,
            )
        val_stats = evaluate_loader(
            model,
            val_loader,
            device,
            bomb_weight=args.bomb_weight,
            raw_ce_coef=args.raw_ce_coef,
            masked_ce_coef=args.masked_ce_coef,
            invalid_action_coef=args.illegal_action_coef,
        )
        score = (
            val_stats["accuracy"]
            + 0.10 * val_stats["raw_accuracy"]
            + 0.20 * val_stats["bomb_recall"]
            + 0.10 * val_stats["bomb_precision"]
            + 0.10 * val_stats["danger_accuracy"]
            + 0.08 * val_stats["valuable_accuracy"]
            - 0.30 * val_stats["illegal_before_mask_rate"]
            - 0.10 * val_stats["invalid_action_mass"]
            - 0.10 * abs(val_stats["pred_bomb_rate"] - val_stats["true_bomb_rate"])
        )

        print(
            f"Epoch {epoch:>2} | phase {phase_name} | train_loss {train_loss:.4f} | "
            f"val_loss {val_stats['loss']:.4f} | acc/raw {val_stats['accuracy']:.2%}/{val_stats['raw_accuracy']:.2%} | "
            f"bomb P/R {val_stats['bomb_precision']:.2%}/{val_stats['bomb_recall']:.2%} | "
            f"bomb pred/true {val_stats['pred_bomb_rate']:.2%}/{val_stats['true_bomb_rate']:.2%} | "
            f"danger {val_stats['danger_accuracy']:.2%} | valuable {val_stats['valuable_accuracy']:.2%} | "
            f"illegal_pre_mask {val_stats['illegal_before_mask_rate']:.2%} | "
            f"raw_ce {train_raw_ce:.4f} | masked_ce {train_masked_ce:.4f} | "
            f"invalid_mass train/val {train_invalid_mass:.4f}/{val_stats['invalid_action_mass']:.4f}"
        )
        if train_eval_stats is not None:
            print(
                f"  -> Overfit train_eval acc/raw {train_eval_stats['accuracy']:.2%}/{train_eval_stats['raw_accuracy']:.2%} | "
                f"bomb P/R {train_eval_stats['bomb_precision']:.2%}/{train_eval_stats['bomb_recall']:.2%} | "
                f"bomb pred/true {train_eval_stats['pred_bomb_rate']:.2%}/{train_eval_stats['true_bomb_rate']:.2%} | "
                f"illegal_pre_mask {train_eval_stats['illegal_before_mask_rate']:.2%} | "
                f"invalid_mass {train_eval_stats['invalid_action_mass']:.4f}"
            )
        cm = val_stats["confusion_matrix"]
        cm_rows = " | ".join(
            f"{ACTION_NAMES[row]}:{','.join(str(int(v)) for v in cm[row])}"
            for row in range(cm.shape[0])
        )
        print(f"  -> Confusion {cm_rows}")
        if train_eval_stats is not None:
            train_cm = train_eval_stats["confusion_matrix"]
            train_cm_rows = " | ".join(
                f"{ACTION_NAMES[row]}:{','.join(str(int(v)) for v in train_cm[row])}"
                for row in range(train_cm.shape[0])
            )
            print(f"  -> Overfit Train Confusion {train_cm_rows}")

        phase_best_path = head_warmup_output_path if phase_name == "head_warmup" else finetune_output_path
        if score >= best_phase_scores[phase_name]:
            best_phase_scores[phase_name] = score
            best_phase_epochs[phase_name] = epoch
            torch.save(model.state_dict(), phase_best_path)
            print(
                f"  -> saved best {phase_name} checkpoint to {phase_best_path} "
                f"(epoch {epoch}, score {score:.6f})"
            )
            if phase_name == "finetune":
                finetune_epochs_without_improve = 0
        elif phase_name == "finetune":
            finetune_epochs_without_improve += 1

        if score >= best_score:
            best_score = score
            torch.save(model.state_dict(), output_path)
            print(f"  -> saved best BC actor to {output_path}")

        if (
            phase_name == "finetune"
            and args.finetune_patience > 0
            and finetune_epochs_without_improve >= args.finetune_patience
        ):
            print(
                f"Early stopping finetune after {finetune_epochs_without_improve} non-improving epochs | "
                f"best_finetune_epoch={best_phase_epochs['finetune']} "
                f"best_finetune_score={best_phase_scores['finetune']:.6f}"
            )
            break

    print(
        "Training summary | "
        f"best_head_warmup_epoch={best_phase_epochs['head_warmup']} "
        f"best_head_warmup_score={best_phase_scores['head_warmup']:.6f} | "
        f"best_finetune_epoch={best_phase_epochs['finetune']} "
        f"best_finetune_score={best_phase_scores['finetune']:.6f} | "
        f"best_overall_score={best_score:.6f}"
    )


if __name__ == "__main__":
    main()
