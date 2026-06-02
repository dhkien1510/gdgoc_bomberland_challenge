"""
Quick dataset diagnostics for BC shards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bc_dataset import load_bc_shards

ACTION_NAMES = ["STOP", "LEFT", "RIGHT", "UP", "DOWN", "PLACE_BOMB"]
SCENARIO_NAMES = {
    0: "farm",
    1: "survive",
    2: "pressure",
    3: "selfplay",
    4: "late",
    5: "dagger",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    arrays = load_bc_shards(Path(args.data_dir))

    num_samples = len(arrays["actions"])
    num_episodes = len(np.unique(arrays["episode_keys"]))
    print(f"Samples: {num_samples:,}")
    print(f"Episodes: {num_episodes:,}")

    action_counts = np.bincount(arrays["actions"], minlength=6)
    print("\nAction distribution:")
    for idx, name in enumerate(ACTION_NAMES):
        pct = 100.0 * action_counts[idx] / max(num_samples, 1)
        print(f"  {name:<11} {action_counts[idx]:>8,}  ({pct:5.2f}%)")

    scenario_counts = np.bincount(arrays["scenario_ids"], minlength=6)
    print("\nScenario distribution:")
    for idx, count in enumerate(scenario_counts):
        if count == 0:
            continue
        pct = 100.0 * count / max(num_samples, 1)
        print(f"  {SCENARIO_NAMES.get(idx, idx):<11} {count:>8,}  ({pct:5.2f}%)")

    danger_mask = arrays["danger_times"] >= 0
    valuable_mask = arrays["valuable_states"].astype(bool)
    can_escape_mask = arrays["can_escape_if_place"].astype(bool)

    print("\nState ratios:")
    print(f"  danger states         {100.0 * danger_mask.mean():5.2f}%")
    print(f"  valuable bomb states  {100.0 * valuable_mask.mean():5.2f}%")
    print(f"  can escape if place   {100.0 * can_escape_mask.mean():5.2f}%")

    step_buckets = {
        "0-49": ((arrays["steps"] >= 0) & (arrays["steps"] < 50)).mean(),
        "50-149": ((arrays["steps"] >= 50) & (arrays["steps"] < 150)).mean(),
        "150-299": ((arrays["steps"] >= 150) & (arrays["steps"] < 300)).mean(),
        "300+": (arrays["steps"] >= 300).mean(),
    }
    print("\nStep buckets:")
    for name, ratio in step_buckets.items():
        print(f"  {name:<8} {100.0 * ratio:5.2f}%")


if __name__ == "__main__":
    main()
