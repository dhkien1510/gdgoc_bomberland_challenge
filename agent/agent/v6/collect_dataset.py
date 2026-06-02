"""
Scenario-based BC dataset collector using v5.2 preprocessing.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent.parent
BASELINE_DIR = ROOT / "agent"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from box_farmer_agent import BoxFarmerAgent
from genius_rule_agent import GeniusRuleAgent
from random_agent import RandomAgent
from simple_rule_agent import SimpleRuleAgent
from smarter_rule_agent import SmarterRuleAgent
from tactical_rule_agent import TacticalRuleAgent
from bc_model import CNNLSTMBCActor
from engine.game import BomberEnv
from model import (
    VALUE_BOMB_MASK_STEPS,
    build_bomb_state,
    can_hit_enemy_if_place,
    count_boxes_if_place,
    current_tile_danger_time,
    has_escape_after_placing_bomb,
    prepare_policy_inputs,
    to_canonical_action,
    to_env_action,
)

SCENARIOS = {
    "farm": {
        "id": 0,
        "opponents": [(RandomAgent, 0.4), (SimpleRuleAgent, 0.4), (BoxFarmerAgent, 0.2)],
        "collect_all_perspectives": False,
    },
    "survive": {
        "id": 1,
        "opponents": [(SimpleRuleAgent, 0.5), (BoxFarmerAgent, 0.25), (SmarterRuleAgent, 0.25)],
        "collect_all_perspectives": False,
    },
    "pressure": {
        "id": 2,
        "opponents": [(SmarterRuleAgent, 0.5), (GeniusRuleAgent, 0.25), (TacticalRuleAgent, 0.25)],
        "collect_all_perspectives": False,
    },
    "selfplay": {
        "id": 3,
        "opponents": [(TacticalRuleAgent, 1.0)],
        "collect_all_perspectives": True,
    },
    "late": {
        "id": 4,
        "opponents": [(GeniusRuleAgent, 0.4), (TacticalRuleAgent, 0.4), (SmarterRuleAgent, 0.2)],
        "collect_all_perspectives": False,
    },
    "dagger": {
        "id": 5,
        "opponents": [(SimpleRuleAgent, 0.35), (SmarterRuleAgent, 0.35), (TacticalRuleAgent, 0.3)],
        "collect_all_perspectives": False,
    },
}
MASK_STEP_FOR_BC = VALUE_BOMB_MASK_STEPS


def weighted_choice(rng: random.Random, entries):
    total = sum(weight for _item, weight in entries)
    pick = rng.random() * total
    running = 0.0
    for item, weight in entries:
        running += weight
        if pick <= running:
            return item
    return entries[-1][0]


class BCPolicyWrapper:
    def __init__(self, checkpoint_path: str | Path, agent_id: int, deterministic: bool = True):
        self.agent_id = int(agent_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNLSTMBCActor()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.state = self.model.get_initial_state(1, self.device)
        self.deterministic = deterministic
        self.episode_start = True

    def reset(self):
        self.state = self.model.get_initial_state(1, self.device)
        self.episode_start = True

    def act(self, obs):
        _, map_feat, aux_feat, action_mask = prepare_policy_inputs(
            obs,
            self.agent_id,
            current_step=MASK_STEP_FOR_BC,
            value_bomb_mask_steps=VALUE_BOMB_MASK_STEPS,
            eval_mode=False,
        )
        with torch.no_grad():
            action, next_state = self.model.act_step(
                map_feat.unsqueeze(0).to(self.device),
                aux_feat.unsqueeze(0).to(self.device),
                action_mask=action_mask.unsqueeze(0).to(self.device),
                state=self.state,
                deterministic=self.deterministic,
                episode_start=torch.tensor([self.episode_start], device=self.device),
            )
        self.state = next_state
        self.episode_start = False
        return to_env_action(int(action.item()), self.agent_id)


def remaining_boxes(obs: dict) -> int:
    return int(np.sum(np.asarray(obs["map"]) == 2))


def should_keep_sample(obs: dict, agent_id: int, step_idx: int, scenario_name: str) -> bool:
    if not bool(obs["players"][agent_id][2]):
        return False
    if scenario_name != "late":
        return True
    bomb_state = build_bomb_state(obs)
    danger = current_tile_danger_time(obs, agent_id, bomb_state)
    enemy_dist = min(
        [
            abs(int(obs["players"][agent_id][0]) - int(p[0])) + abs(int(obs["players"][agent_id][1]) - int(p[1]))
            for i, p in enumerate(obs["players"])
            if i != agent_id and int(p[2]) == 1
        ]
        or [999]
    )
    return step_idx >= 150 or remaining_boxes(obs) <= 20 or (danger is not None and danger <= 4) or enemy_dist <= 5


def scenario_teacher_ids(scenario_name: str, rng: random.Random):
    if SCENARIOS[scenario_name]["collect_all_perspectives"]:
        return [0, 1, 2, 3]
    return [rng.randrange(4)]


def build_agents_for_episode(
    scenario_name: str,
    rng: random.Random,
    bc_policy_path: str | None,
):
    teacher_ids = scenario_teacher_ids(scenario_name, rng)
    players = []
    for player_id in range(4):
        if scenario_name == "selfplay":
            players.append(TacticalRuleAgent(player_id))
        elif scenario_name == "dagger" and player_id in teacher_ids and bc_policy_path:
            players.append(BCPolicyWrapper(bc_policy_path, player_id))
        elif player_id in teacher_ids:
            players.append(TacticalRuleAgent(player_id))
        else:
            agent_cls = weighted_choice(rng, SCENARIOS[scenario_name]["opponents"])
            players.append(agent_cls(player_id))
    return players, teacher_ids


def collect_episode_samples(
    obs: dict,
    label_actions: list[int],
    episode_id: int,
    step_idx: int,
    scenario_name: str,
    teacher_ids: list[int],
    buffers: dict[str, list],
    illegal_counter: dict[str, int],
):
    scenario_id = SCENARIOS[scenario_name]["id"]
    for agent_id in teacher_ids:
        if not should_keep_sample(obs, agent_id, step_idx, scenario_name):
            continue

        env_action = int(label_actions[agent_id])
        canonical_action = int(to_canonical_action(env_action, agent_id))
        canonical_obs, map_feat, aux_feat, action_mask = prepare_policy_inputs(
            obs,
            agent_id,
            current_step=MASK_STEP_FOR_BC,
            value_bomb_mask_steps=VALUE_BOMB_MASK_STEPS,
            eval_mode=False,
        )
        if not bool(action_mask[canonical_action]):
            illegal_counter["teacher_action_masked"] += 1
            continue

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
        buffers["scenario_ids"].append(scenario_id)
        buffers["danger_times"].append(-1 if danger is None else int(danger))
        buffers["valuable_states"].append(bool(valuable))
        buffers["can_escape_if_place"].append(bool(can_escape))


def flush_shard(output_dir: Path, shard_idx: int, buffers: dict[str, list]):
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
    for key in list(buffers.keys()):
        buffers[key].clear()
    return shard_idx + 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, required=True)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS.keys()), required=True)
    parser.add_argument("--output_dir", type=str, default=str(_HERE / "bc_data"))
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--episodes_per_shard", type=int, default=200)
    parser.add_argument("--bc_policy_ckpt", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    buffers = {
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
    illegal_counter = {"teacher_action_masked": 0}
    total_steps_saved = 0
    shard_idx = 0

    for episode_offset in range(args.num_episodes):
        episode_id = args.seed_start + episode_offset
        seed = args.seed_start + episode_offset
        rng = random.Random(seed)
        env = BomberEnv(max_steps=args.max_steps, seed=seed)
        obs = env.reset(seed=seed)
        players, teacher_ids = build_agents_for_episode(args.scenario, rng, args.bc_policy_ckpt)
        for player in players:
            if hasattr(player, "reset"):
                player.reset()

        done = False
        step_idx = 0
        while not done:
            actions = [0] * 4
            label_actions = [0] * 4
            for player_id, player in enumerate(players):
                if int(obs["players"][player_id][2]) != 1:
                    continue
                if args.scenario == "dagger" and player_id in teacher_ids and args.bc_policy_ckpt:
                    teacher_action = TacticalRuleAgent(player_id).act(obs)
                    learner_action = player.act(obs)
                    label_actions[player_id] = int(teacher_action)
                    actions[player_id] = learner_action if rng.random() < 0.8 else teacher_action
                else:
                    act = int(player.act(obs))
                    actions[player_id] = act
                    label_actions[player_id] = act

            collect_episode_samples(
                obs,
                label_actions,
                episode_id=episode_id,
                step_idx=step_idx,
                scenario_name=args.scenario,
                teacher_ids=teacher_ids,
                buffers=buffers,
                illegal_counter=illegal_counter,
            )
            total_steps_saved = len(buffers["actions"])
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step_idx += 1

        if (episode_offset + 1) % args.episodes_per_shard == 0:
            shard_idx = flush_shard(output_dir, shard_idx, buffers)
            print(
                f"Collected {episode_offset + 1}/{args.num_episodes} episodes | "
                f"saved samples: {total_steps_saved:,} | masked teacher labels skipped: {illegal_counter['teacher_action_masked']}"
            )

    shard_idx = flush_shard(output_dir, shard_idx, buffers)
    print(
        f"Done. Episodes: {args.num_episodes} | shards: {shard_idx} | "
        f"masked teacher labels skipped: {illegal_counter['teacher_action_masked']}"
    )


if __name__ == "__main__":
    main()
