"""
Shard loader and sequence dataset for v6_2 BC training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def load_bc_shards(data_dir: str | Path):
    data_dir = Path(data_dir)
    shard_paths = sorted(data_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {data_dir}")

    merged: dict[str, list[np.ndarray]] = {}
    episode_keys_parts: list[np.ndarray] = []
    for shard_idx, path in enumerate(shard_paths):
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                if key == "episode_ids":
                    raw_episode_ids = data[key].astype(np.int64)
                    episode_keys_parts.append(raw_episode_ids + shard_idx * 1_000_000_000)
                else:
                    merged.setdefault(key, []).append(data[key])

    arrays = {key: np.concatenate(parts, axis=0) for key, parts in merged.items()}
    arrays["episode_keys"] = np.concatenate(episode_keys_parts, axis=0)
    return arrays


def split_episode_keys(episode_keys: np.ndarray, train_ratio=0.8, val_ratio=0.1, seed: int = 42):
    unique_keys = np.unique(episode_keys)
    rng = np.random.default_rng(seed)
    keys = unique_keys.copy()
    rng.shuffle(keys)

    n_total = len(keys)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_keys = keys[:n_train]
    val_keys = keys[n_train:n_train + n_val]
    test_keys = keys[n_train + n_val:]
    return train_keys, val_keys, test_keys


@dataclass(frozen=True)
class SequenceSample:
    episode_key: int
    start: int
    end: int
    bucket: str


class BCSequenceDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        selected_episode_keys: np.ndarray,
        seq_len: int = 64,
        stride: int = 32,
    ):
        self.arrays = arrays
        self.seq_len = seq_len
        self.selected_set = {int(v) for v in selected_episode_keys.tolist()}
        self.samples: list[SequenceSample] = []
        self._build_samples(stride)

    def _build_samples(self, stride: int):
        episode_keys = self.arrays["episode_keys"]
        steps = self.arrays["steps"]
        order = np.lexsort((steps, episode_keys))
        ordered_episode_keys = episode_keys[order]

        start = 0
        while start < len(order):
            episode_key = int(ordered_episode_keys[start])
            end = start + 1
            while end < len(order) and int(ordered_episode_keys[end]) == episode_key:
                end += 1
            if episode_key in self.selected_set:
                episode_indices = order[start:end]
                self._append_episode_sequences(episode_key, episode_indices, stride)
            start = end

    def _append_episode_sequences(self, episode_key: int, episode_indices: np.ndarray, stride: int):
        episode_actions = self.arrays["actions"][episode_indices]
        episode_danger = self.arrays["danger_times"][episode_indices]
        episode_valuable = self.arrays["valuable_states"][episode_indices]

        length = len(episode_indices)
        for start in range(0, max(length, 1), stride):
            end = min(start + self.seq_len, length)
            window_actions = episode_actions[start:end]
            window_danger = episode_danger[start:end]
            window_valuable = episode_valuable[start:end]
            if np.any(window_actions == 5):
                bucket = "bomb"
            elif np.any(window_danger >= 0) and np.any(window_danger <= 3):
                bucket = "danger"
            elif np.any(window_valuable):
                bucket = "valuable"
            else:
                bucket = "normal"
            self.samples.append(SequenceSample(episode_key, int(start), int(end), bucket))
            if end >= length:
                break

    def bucket_weights(self):
        weights = []
        for sample in self.samples:
            if sample.bucket == "bomb":
                weights.append(1.9)
            elif sample.bucket == "danger":
                weights.append(1.7)
            elif sample.bucket == "valuable":
                weights.append(1.35)
            else:
                weights.append(1.0)
        return np.asarray(weights, dtype=np.float32)

    def make_sampler(self):
        weights = torch.tensor(self.bucket_weights(), dtype=torch.float32)
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        mask = self.arrays["episode_keys"] == sample.episode_key
        indices = np.nonzero(mask)[0]
        indices = indices[np.argsort(self.arrays["steps"][indices])]
        indices = indices[sample.start:sample.end]

        actual_len = len(indices)
        seq_len = self.seq_len
        map_feats = np.zeros((seq_len, *self.arrays["map_feats"].shape[1:]), dtype=np.float32)
        aux_feats = np.zeros((seq_len, self.arrays["aux_feats"].shape[1]), dtype=np.float32)
        action_masks = np.zeros((seq_len, self.arrays["action_masks"].shape[1]), dtype=np.bool_)
        actions = np.zeros((seq_len,), dtype=np.int64)
        episode_starts = np.zeros((seq_len,), dtype=np.bool_)
        loss_mask = np.zeros((seq_len,), dtype=np.float32)
        danger_times = np.full((seq_len,), -1, dtype=np.int64)
        valuable_states = np.zeros((seq_len,), dtype=np.bool_)

        if actual_len > 0:
            map_feats[:actual_len] = self.arrays["map_feats"][indices].astype(np.float32)
            aux_feats[:actual_len] = self.arrays["aux_feats"][indices].astype(np.float32)
            action_masks[:actual_len] = self.arrays["action_masks"][indices]
            actions[:actual_len] = self.arrays["actions"][indices]
            danger_times[:actual_len] = self.arrays["danger_times"][indices]
            valuable_states[:actual_len] = self.arrays["valuable_states"][indices]
            loss_mask[:actual_len] = 1.0
            episode_starts[0] = sample.start == 0

        return {
            "map_feats": torch.from_numpy(map_feats),
            "aux_feats": torch.from_numpy(aux_feats),
            "action_masks": torch.from_numpy(action_masks),
            "actions": torch.from_numpy(actions),
            "episode_starts": torch.from_numpy(episode_starts),
            "loss_mask": torch.from_numpy(loss_mask),
            "danger_times": torch.from_numpy(danger_times),
            "valuable_states": torch.from_numpy(valuable_states),
            "bucket": sample.bucket,
        }


__all__ = [
    "BCSequenceDataset",
    "load_bc_shards",
    "split_episode_keys",
]
