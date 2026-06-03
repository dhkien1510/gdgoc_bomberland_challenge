"""
Rollout evaluation for the BC actor before PPO fine-tuning.
"""

from __future__ import annotations

import argparse
from collections import deque
import importlib.util
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

_MODEL_SPEC = importlib.util.spec_from_file_location("_v6_model_eval", _HERE / "model.py")
_MODEL = importlib.util.module_from_spec(_MODEL_SPEC)
assert _MODEL_SPEC.loader is not None
_MODEL_SPEC.loader.exec_module(_MODEL)

from box_farmer_agent import BoxFarmerAgent
from genius_rule_agent import GeniusRuleAgent
from random_agent import RandomAgent
from simple_rule_agent import SimpleRuleAgent
from smarter_rule_agent import SmarterRuleAgent
from tactical_rule_agent import TacticalRuleAgent
from bc_model import CNNLSTMBCActor
from engine.game import BomberEnv
import _train_base as base

prepare_policy_inputs = _MODEL.prepare_policy_inputs
to_env_action = _MODEL.to_env_action
ACTION_PLACE_BOMB = _MODEL.ACTION_PLACE_BOMB
MASK_WARMUP_STEPS = _MODEL.MASK_WARMUP_STEPS
VALUE_BOMB_MASK_STEPS = _MODEL.VALUE_BOMB_MASK_STEPS
can_hit_enemy_if_place = _MODEL.can_hit_enemy_if_place
count_boxes_if_place = _MODEL.count_boxes_if_place
nearest_valuable_bomb_spot_info = _MODEL.nearest_valuable_bomb_spot_info


def _load_checkpoint(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class BCAgentEval:
    def __init__(self, checkpoint_path: str | Path, agent_id: int, deterministic: bool = True):
        self.agent_id = int(agent_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_step = 0
        self.recent_positions = deque(maxlen=12)
        self.recent_actions = deque(maxlen=12)
        self.model = CNNLSTMBCActor()
        checkpoint = _load_checkpoint(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.state = self.model.get_initial_state(1, self.device)
        self.episode_start = True
        self.deterministic = deterministic

    def reset(self):
        self.state = self.model.get_initial_state(1, self.device)
        self.episode_start = True
        self.current_step = 0
        self.recent_positions.clear()
        self.recent_actions.clear()

    def _is_looping(self) -> bool:
        if len(self.recent_positions) < 8:
            return False
        return len(set(list(self.recent_positions)[-8:])) <= 2

    def act(self, obs):
        self.current_step += 1
        my = obs["players"][self.agent_id]
        pos = (int(my[0]), int(my[1]))
        self.recent_positions.append(pos)

        canonical_obs, map_feat, aux_feat, action_mask = prepare_policy_inputs(
            obs,
            self.agent_id,
            current_step=VALUE_BOMB_MASK_STEPS,
            warmup_steps=MASK_WARMUP_STEPS,
            value_bomb_mask_steps=VALUE_BOMB_MASK_STEPS,
            eval_mode=False,
        )
        with torch.no_grad():
            action, self.state = self.model.act_step(
                map_feat.unsqueeze(0).to(self.device),
                aux_feat.unsqueeze(0).to(self.device),
                action_mask=action_mask.unsqueeze(0).to(self.device),
                state=self.state,
                deterministic=self.deterministic,
                episode_start=torch.tensor([self.episode_start], device=self.device),
            )
        canonical_action = int(action.item())
        self.episode_start = False

        if self._is_looping():
            if bool(action_mask[ACTION_PLACE_BOMB]) and (
                can_hit_enemy_if_place(canonical_obs, self.agent_id)
                or count_boxes_if_place(canonical_obs, self.agent_id) > 0
            ):
                canonical_action = ACTION_PLACE_BOMB
            else:
                first_action, _dist, _count = nearest_valuable_bomb_spot_info(canonical_obs, self.agent_id)
                if first_action is not None and bool(action_mask[int(first_action)]):
                    canonical_action = int(first_action)

        self.recent_actions.append(canonical_action)
        return to_env_action(canonical_action, self.agent_id)


def opponent_pool(name: str):
    if name == "easy":
        return [RandomAgent, SimpleRuleAgent, BoxFarmerAgent]
    if name == "hard":
        return [SmarterRuleAgent, GeniusRuleAgent, TacticalRuleAgent]
    return [RandomAgent, SimpleRuleAgent, BoxFarmerAgent, SmarterRuleAgent, GeniusRuleAgent, TacticalRuleAgent]


def run_match(checkpoint_path: str, pool_name: str, num_matches: int, seed_base: int):
    total_points = 0.0
    total_first = 0.0
    total_unique_first = 0.0
    total_shared_first = 0.0
    total_rank = 0.0
    total_bombs = 0.0
    total_boxes = 0.0
    total_items = 0.0
    total_kills = 0.0
    total_danger = 0.0
    total_no_escape = 0.0
    total_tiles = 0.0
    total_repeat = 0.0
    total_vb = 0.0
    opponents_pool = opponent_pool(pool_name)

    for match_idx in range(num_matches):
        seed = seed_base + match_idx
        rng = random.Random(seed)
        env = BomberEnv(max_steps=500, seed=seed)
        agent_id = rng.randrange(4)
        obs = env.reset(seed=seed)

        learners = {agent_id: BCAgentEval(checkpoint_path, agent_id)}
        opponents = {
            player_id: rng.choice(opponents_pool)(player_id)
            for player_id in range(4)
            if player_id != agent_id
        }
        alive_mask = [bool(player.alive) for player in env.players]
        death_order = []
        episode_ctx = base.make_episode_context(obs, agent_id)

        done = False
        while not done:
            actions = [0] * 4
            if env.players[agent_id].alive:
                actions[agent_id] = learners[agent_id].act(obs)
            for opponent_id, opponent in opponents.items():
                if env.players[opponent_id].alive:
                    actions[opponent_id] = int(opponent.act(obs))

            prev_obs = base.clone_obs(obs)
            prev_stats = base.clone_stats(env.players[agent_id])
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            base.record_deaths(env.players, alive_mask, death_order)
            curr_stats = base.clone_stats(env.players[agent_id])
            base.compute_reward(
                prev_obs,
                obs,
                prev_stats,
                curr_stats,
                agent_id,
                done,
                None,
                episode_ctx,
                base.get_stage(0),
            )

        ranks = base.compute_competition_ranks(env.players, death_order, alive_mask)
        first_group_size = sum(1 for rank in ranks if rank == 0)
        final_stats = base.clone_stats(env.players[agent_id])
        metrics = base.summarize_episode_metrics(episode_ctx, final_stats)

        total_points += base.RANK_TO_POINTS[ranks[agent_id]]
        total_first += float(ranks[agent_id] == 0)
        total_unique_first += float(ranks[agent_id] == 0 and first_group_size == 1)
        total_shared_first += float(ranks[agent_id] == 0 and first_group_size > 1)
        total_rank += float(ranks[agent_id])
        total_bombs += metrics["bombs_per_episode"]
        total_boxes += metrics["boxes_per_episode"]
        total_items += metrics["items_per_episode"]
        total_kills += metrics["kills_per_episode"]
        total_danger += metrics["danger_steps_per_episode"]
        total_no_escape += metrics["no_escape_bomb_ratio"]
        total_tiles += metrics["unique_tiles_visited"]
        total_repeat += metrics["repeat_position_rate"]
        total_vb += metrics["valuable_bomb_ratio"]

    denom = max(num_matches, 1)
    print(
        f"Pool {pool_name} | points {total_points / denom:.3f} | first {total_first / denom:.2%} "
        f"(unique {total_unique_first / denom:.2%}, shared {total_shared_first / denom:.2%}) | "
        f"rank {total_rank / denom:.2f} | bombs {total_bombs / denom:.2f} | "
        f"boxes {total_boxes / denom:.2f} | items {total_items / denom:.2f} | "
        f"kills {total_kills / denom:.2f} | vb {total_vb / denom:.2%} | "
        f"no_escape {total_no_escape / denom:.2%} | danger {total_danger / denom:.1f} | "
        f"tiles {total_tiles / denom:.1f} | repeat {total_repeat / denom:.2%}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(_HERE / "bc_actor.pth"))
    parser.add_argument("--pool", choices=["easy", "hard", "mixed"], default="mixed")
    parser.add_argument("--num_matches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_match(args.checkpoint, args.pool, args.num_matches, args.seed)
