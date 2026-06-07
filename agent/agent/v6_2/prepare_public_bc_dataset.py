"""
Convert public replay JSON manifests into BC shard_*.npz files for v6_2.

This bootstraps v6_2 BC from strong public replay data before the rest of the
v6_2 training stack is finalized.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent.parent
V6_1_DIR = _HERE.parent / "v6_1"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V6_1_DIR) not in sys.path:
    sys.path.insert(0, str(V6_1_DIR))

_MODEL_SPEC = importlib.util.spec_from_file_location("_v6_1_model_public_bc", V6_1_DIR / "model.py")
_MODEL = importlib.util.module_from_spec(_MODEL_SPEC)
assert _MODEL_SPEC.loader is not None
_MODEL_SPEC.loader.exec_module(_MODEL)

build_bomb_state = _MODEL.build_bomb_state
can_hit_enemy_if_place = _MODEL.can_hit_enemy_if_place
count_boxes_if_place = _MODEL.count_boxes_if_place
current_tile_danger_time = _MODEL.current_tile_danger_time
has_escape_after_placing_bomb = _MODEL.has_escape_after_placing_bomb
prepare_policy_inputs = _MODEL.prepare_policy_inputs
to_canonical_action = _MODEL.to_canonical_action

BC_MASK_CURRENT_STEP = 0
BC_MASK_WARMUP_STEPS = 10**9
BC_MASK_VALUE_BOMB_STEPS = 10**9
SCENARIO_PUBLIC_REPLAY = 6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest.csv from downloaded public replays")
    parser.add_argument("--output_dir", type=str, default=str(_HERE / "bc_data_public"))
    parser.add_argument("--agent_id", type=str, default="", help="Optional agent id to validate against manifest rows")
    parser.add_argument("--max_matches", type=int, default=0, help="0 means use every row in the manifest")
    parser.add_argument("--episodes_per_shard", type=int, default=100)
    parser.add_argument("--min_rank", type=int, default=0)
    parser.add_argument("--max_rank", type=int, default=1)
    parser.add_argument("--allow_masked_actions", type=int, default=0)
    return parser.parse_args()


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def normalize_obs(frame: dict) -> dict:
    return {
        "map": frame["map"],
        "players": frame["players"],
        "bombs": frame["bombs"],
    }


def init_buffers() -> dict[str, list]:
    return {
        "map_feats": [],
        "aux_feats": [],
        "action_masks": [],
        "actions": [],
        "dones": [],
        "agent_ids": [],
        "episode_ids": [],
        "steps": [],
        "scenario_ids": [],
        "danger_times": [],
        "valuable_states": [],
        "can_escape_if_place": [],
    }


def flush_shard(output_dir: Path, shard_idx: int, buffers: dict[str, list]) -> int:
    if not buffers["actions"]:
        return shard_idx
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"shard_{shard_idx:05d}.npz"
    np.savez_compressed(
        shard_path,
        map_feats=np.asarray(buffers["map_feats"], dtype=np.float16),
        aux_feats=np.asarray(buffers["aux_feats"], dtype=np.float32),
        action_masks=np.asarray(buffers["action_masks"], dtype=np.bool_),
        actions=np.asarray(buffers["actions"], dtype=np.int64),
        dones=np.asarray(buffers["dones"], dtype=np.bool_),
        agent_ids=np.asarray(buffers["agent_ids"], dtype=np.int8),
        episode_ids=np.asarray(buffers["episode_ids"], dtype=np.int32),
        steps=np.asarray(buffers["steps"], dtype=np.int16),
        scenario_ids=np.asarray(buffers["scenario_ids"], dtype=np.int8),
        danger_times=np.asarray(buffers["danger_times"], dtype=np.int8),
        valuable_states=np.asarray(buffers["valuable_states"], dtype=np.bool_),
        can_escape_if_place=np.asarray(buffers["can_escape_if_place"], dtype=np.bool_),
    )
    for key in buffers:
        buffers[key].clear()
    return shard_idx + 1


def append_sample(
    canonical_obs: dict,
    map_feat,
    aux_feat,
    action_mask,
    canonical_action: int,
    agent_id: int,
    episode_id: int,
    step_idx: int,
    buffers: dict[str, list],
):
    bomb_state = build_bomb_state(canonical_obs)
    danger = current_tile_danger_time(canonical_obs, agent_id, bomb_state)
    valuable = can_hit_enemy_if_place(canonical_obs, agent_id) or count_boxes_if_place(canonical_obs, agent_id) > 0
    can_escape = has_escape_after_placing_bomb(canonical_obs, agent_id)

    buffers["map_feats"].append(map_feat.numpy().astype(np.float16))
    buffers["aux_feats"].append(aux_feat.numpy().astype(np.float32))
    buffers["action_masks"].append(action_mask.numpy().astype(np.bool_))
    buffers["actions"].append(canonical_action)
    buffers["dones"].append(False)
    buffers["agent_ids"].append(agent_id)
    buffers["episode_ids"].append(episode_id)
    buffers["steps"].append(step_idx)
    buffers["scenario_ids"].append(SCENARIO_PUBLIC_REPLAY)
    buffers["danger_times"].append(-1 if danger is None else int(danger))
    buffers["valuable_states"].append(bool(valuable))
    buffers["can_escape_if_place"].append(bool(can_escape))


def convert_replay(
    replay: dict,
    replay_agent_index: int,
    episode_id: int,
    buffers: dict[str, list],
    allow_masked_actions: bool,
) -> tuple[int, int]:
    history = replay["history"]
    saved_steps = 0
    masked_steps = 0

    for frame_idx in range(len(history) - 1):
        obs_frame = history[frame_idx]
        next_frame = history[frame_idx + 1]
        actions = next_frame.get("actions")
        if actions is None:
            continue

        obs = normalize_obs(obs_frame)
        players = np.asarray(obs["players"], dtype=np.int64)
        if int(players[replay_agent_index][2]) != 1:
            continue

        env_action = int(actions[replay_agent_index])
        canonical_action = int(to_canonical_action(env_action, replay_agent_index))
        canonical_obs, map_feat, aux_feat, action_mask = prepare_policy_inputs(
            obs,
            replay_agent_index,
            current_step=BC_MASK_CURRENT_STEP,
            warmup_steps=BC_MASK_WARMUP_STEPS,
            value_bomb_mask_steps=BC_MASK_VALUE_BOMB_STEPS,
            eval_mode=False,
        )
        if not bool(action_mask[canonical_action]):
            masked_steps += 1
            if not allow_masked_actions:
                continue

        append_sample(
            canonical_obs=canonical_obs,
            map_feat=map_feat,
            aux_feat=aux_feat,
            action_mask=action_mask,
            canonical_action=canonical_action,
            agent_id=replay_agent_index,
            episode_id=episode_id,
            step_idx=int(obs_frame["step"]),
            buffers=buffers,
        )
        saved_steps += 1

    return saved_steps, masked_steps


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    rows = load_manifest_rows(manifest_path)
    selected_rows = [
        row
        for row in rows
        if args.min_rank <= int(row["rank"]) <= args.max_rank
        and (not args.agent_id or row["agent_id"] == args.agent_id)
    ]
    if args.max_matches > 0:
        selected_rows = selected_rows[: args.max_matches]

    buffers = init_buffers()
    shard_idx = 0
    total_saved_steps = 0
    total_masked_steps = 0
    kept_matches = 0
    skipped_missing = 0

    for episode_id, row in enumerate(selected_rows):
        json_path = Path(row["json_path"])
        if not json_path.is_absolute():
            json_path = ROOT / json_path
        if not json_path.exists():
            skipped_missing += 1
            print(f"[skip] missing replay json: {json_path}")
            continue

        replay = json.loads(json_path.read_text(encoding="utf-8"))
        replay_agent_index = int(row["replay_index"])
        saved_steps, masked_steps = convert_replay(
            replay=replay,
            replay_agent_index=replay_agent_index,
            episode_id=episode_id,
            buffers=buffers,
            allow_masked_actions=bool(args.allow_masked_actions),
        )
        total_saved_steps += saved_steps
        total_masked_steps += masked_steps
        kept_matches += 1

        if kept_matches % args.episodes_per_shard == 0:
            shard_idx = flush_shard(output_dir, shard_idx, buffers)

    shard_idx = flush_shard(output_dir, shard_idx, buffers)

    summary = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "selected_matches": len(selected_rows),
        "processed_matches": kept_matches,
        "missing_matches": skipped_missing,
        "saved_steps": total_saved_steps,
        "masked_steps_seen": total_masked_steps,
        "episodes_per_shard": int(args.episodes_per_shard),
        "rank_range": [int(args.min_rank), int(args.max_rank)],
        "agent_id": args.agent_id,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Selected matches: {len(selected_rows)}")
    print(f"Processed matches: {kept_matches}")
    print(f"Missing matches: {skipped_missing}")
    print(f"Saved steps: {total_saved_steps}")
    print(f"Masked steps seen: {total_masked_steps}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
