"""
Export a BC checkpoint as a plain actor state_dict for inference/PPO init.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parent / "bc_actor.pth"))
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    torch.save(state_dict, args.output)
    print(f"Exported BC actor weights to {args.output}")


if __name__ == "__main__":
    main()
