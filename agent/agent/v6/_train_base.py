"""
V5.2 training helpers: tactical reward shaping plus bomb-spot navigation.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import random
import sys
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent.parent
BASELINE_DIR = ROOT / "agent"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _model_v3_base import (
    MASK_WARMUP_STEPS,
    CNNActorCriticV3,
    VALUE_BOMB_MASK_STEPS,
    bfs_first_action_to_targets,
    build_bomb_state,
    can_hit_enemy_if_place,
    count_boxes_if_place,
    current_tile_danger_time,
    enemy_same_row_or_col_with_clear_path,
    has_attack_pressure,
    has_escape_after_placing_bomb,
    nearest_enemy_distance,
    nearest_valuable_bomb_spot_info,
    prepare_policy_inputs,
    to_env_action,
)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(BASELINE_DIR) not in sys.path:
    sys.path.append(str(BASELINE_DIR))

from box_farmer_agent import BoxFarmerAgent
from genius_rule_agent import GeniusRuleAgent
from random_agent import RandomAgent
from simple_rule_agent import SimpleRuleAgent
from smarter_rule_agent import SmarterRuleAgent
from tactical_rule_agent import TacticalRuleAgent
from engine.game import BomberEnv

_BASE_PATH = _HERE.parent / "v5_1" / "_train_base.py"
_SPEC = importlib.util.spec_from_file_location("_v5_1_train_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BASE)

CFG = copy.deepcopy(_BASE.CFG)
CFG["reward_clip"] = 3.0
DEVICE = _BASE.DEVICE
RANK_TO_POINTS = _BASE.RANK_TO_POINTS
print(f"Using device: {DEVICE}")

CURRICULUM_STAGES = copy.deepcopy(_BASE.CURRICULUM_STAGES)
CURRICULUM_STAGES[0]["reward"].update(
    {
        "r_move_closer_bomb_spot": 0.006,
        "r_move_away_bomb_spot": -0.002,
        "r_best_bomb_spot_dist": 0.02,
        "r_position_loop": -0.02,
    }
)
CURRICULUM_STAGES[1]["reward"].update(
    {
        "r_move_closer_bomb_spot": 0.004,
        "r_move_away_bomb_spot": -0.002,
        "r_best_bomb_spot_dist": 0.015,
        "r_position_loop": -0.025,
    }
)
CURRICULUM_STAGES[2]["reward"].update(
    {
        "r_move_closer_bomb_spot": 0.002,
        "r_move_away_bomb_spot": -0.001,
        "r_best_bomb_spot_dist": 0.008,
        "r_position_loop": -0.03,
    }
)
CURRICULUM_STAGES[3]["reward"].update(
    {
        "r_move_closer_bomb_spot": 0.002,
        "r_move_away_bomb_spot": -0.001,
        "r_best_bomb_spot_dist": 0.008,
        "r_position_loop": -0.03,
    }
)


def get_stage(current_step: int):
    for stage in CURRICULUM_STAGES:
        if stage["end_step"] is None or current_step < stage["end_step"]:
            return stage
    return CURRICULUM_STAGES[-1]


clone_obs = _BASE.clone_obs
clone_stats = _BASE.clone_stats
sync_linear_lr = _BASE.sync_linear_lr
record_deaths = _BASE.record_deaths
compute_competition_ranks = _BASE.compute_competition_ranks
temporary_random_seed = _BASE.temporary_random_seed
mask_eval_mode = _BASE.mask_eval_mode
weighted_choice = _BASE.weighted_choice
RolloutBuffer = _BASE.RolloutBuffer
ppo_update = _BASE.ppo_update
ActiveOpponentPool = _BASE.ActiveOpponentPool
ModelOpponent = _BASE.ModelOpponent


def danger_time(obs: dict, agent_id: int) -> int | None:
    bomb_state = build_bomb_state(obs)
    return current_tile_danger_time(obs, agent_id, bomb_state)


def danger_penalty_value(reward_cfg: dict, current_danger: int | None) -> float:
    if current_danger is None:
        return 0.0
    if current_danger <= 1:
        return reward_cfg["r_danger_critical"]
    if current_danger <= 2:
        return reward_cfg["r_danger_soon"]
    if current_danger <= 4:
        return reward_cfg["r_danger_far"]
    return 0.0


def is_position_loop(recent_positions: deque[tuple[int, int]]) -> bool:
    if len(recent_positions) < 8:
        return False
    last = list(recent_positions)
    if len(set(last[-6:])) == 1:
        return True
    if len(set(last[-8:])) == 2:
        return True
    return False


def make_episode_context(obs: dict, agent_id: int) -> dict:
    start_row = int(obs["players"][agent_id][0])
    start_col = int(obs["players"][agent_id][1])
    _, initial_bomb_spot_dist, _ = nearest_valuable_bomb_spot_info(obs, agent_id)
    recent_positions: deque[tuple[int, int]] = deque(maxlen=12)
    recent_positions.append((start_row, start_col))
    return {
        "best_enemy_dist": nearest_enemy_distance(obs, agent_id),
        "best_bomb_spot_dist": initial_bomb_spot_dist,
        "bombs_placed": 0,
        "valuable_bombs": 0,
        "box_bombs": 0,
        "useless_bombs": 0,
        "no_escape_bombs": 0,
        "danger_steps": 0,
        "visited_tiles": {(start_row, start_col)},
        "repeat_steps": 0,
        "recent_positions": recent_positions,
        "recent_actions": deque(maxlen=12),
    }


def summarize_episode_metrics(episode_ctx: dict, final_stats: dict) -> dict:
    return _BASE.summarize_episode_metrics(episode_ctx, final_stats)


def compute_reward(
    prev_obs,
    obs,
    prev_stats,
    curr_stats,
    agent_id,
    done,
    rank,
    episode_ctx,
    stage,
    canonical_action: int | None = None,
):
    reward_cfg = stage["reward"]
    reward = reward_cfg["step_penalty"]

    was_alive = bool(prev_obs["players"][agent_id][2])
    if was_alive and not bool(obs["players"][agent_id][2]):
        reward += reward_cfg["r_death"]

    reward += (curr_stats["kills"] - prev_stats["kills"]) * reward_cfg["r_kill"]
    reward += (curr_stats["boxes"] - prev_stats["boxes"]) * reward_cfg["r_box_destroy"]
    reward += (curr_stats["items"] - prev_stats["items"]) * reward_cfg["r_item_collect"]
    bombs_placed = curr_stats["bombs"] - prev_stats["bombs"]
    reward += bombs_placed * reward_cfg["r_bomb_place"]
    if bombs_placed > 0:
        episode_ctx["bombs_placed"] += bombs_placed
        escape = has_escape_after_placing_bomb(prev_obs, agent_id)
        enemy_hit = can_hit_enemy_if_place(prev_obs, agent_id)
        boxes_hit = count_boxes_if_place(prev_obs, agent_id)
        if not escape:
            reward += reward_cfg["r_no_escape_bomb"]
            episode_ctx["no_escape_bombs"] += bombs_placed
        elif enemy_hit:
            reward += reward_cfg["r_valuable_bomb_enemy"]
            episode_ctx["valuable_bombs"] += bombs_placed
        elif boxes_hit > 0:
            reward += reward_cfg["r_valuable_bomb_box"] * min(boxes_hit, 2)
            episode_ctx["box_bombs"] += bombs_placed
        else:
            reward += reward_cfg["r_useless_bomb"]
            episode_ctx["useless_bombs"] += bombs_placed

    prev_enemy_dist = nearest_enemy_distance(prev_obs, agent_id)
    curr_enemy_dist = nearest_enemy_distance(obs, agent_id)
    prev_danger = danger_time(prev_obs, agent_id)
    curr_danger = danger_time(obs, agent_id)
    if (
        prev_enemy_dist > 0.0
        and curr_enemy_dist > 0.0
        and prev_danger is None
        and curr_danger is None
        and has_attack_pressure(prev_obs, agent_id)
    ):
        if curr_enemy_dist < prev_enemy_dist:
            reward += reward_cfg["r_move_closer"]
        elif curr_enemy_dist > prev_enemy_dist:
            reward += reward_cfg["r_move_away"]

    best_enemy_dist = episode_ctx["best_enemy_dist"]
    if (
        curr_enemy_dist > 0.0
        and curr_enemy_dist < best_enemy_dist
        and curr_danger is None
        and has_attack_pressure(obs, agent_id)
    ):
        reward += reward_cfg["r_best_enemy_dist"]
        episode_ctx["best_enemy_dist"] = curr_enemy_dist
    elif curr_enemy_dist > 0.0:
        episode_ctx["best_enemy_dist"] = min(best_enemy_dist, curr_enemy_dist)

    _, prev_spot_dist, _ = nearest_valuable_bomb_spot_info(prev_obs, agent_id)
    _, curr_spot_dist, _ = nearest_valuable_bomb_spot_info(obs, agent_id)
    if prev_danger is None and curr_danger is None:
        if curr_spot_dist < prev_spot_dist:
            reward += reward_cfg["r_move_closer_bomb_spot"]
        elif curr_spot_dist > prev_spot_dist and prev_spot_dist < 999:
            reward += reward_cfg["r_move_away_bomb_spot"]

    best_spot_dist = episode_ctx.get("best_bomb_spot_dist", 999)
    if curr_spot_dist < best_spot_dist:
        reward += reward_cfg["r_best_bomb_spot_dist"]
        episode_ctx["best_bomb_spot_dist"] = curr_spot_dist
    elif curr_spot_dist < 999:
        episode_ctx["best_bomb_spot_dist"] = min(best_spot_dist, curr_spot_dist)

    reward += danger_penalty_value(reward_cfg, curr_danger)
    if curr_danger is not None:
        episode_ctx["danger_steps"] += 1
    if prev_danger is not None and curr_danger is None:
        reward += reward_cfg["r_escape_danger"]

    curr_row = int(obs["players"][agent_id][0])
    curr_col = int(obs["players"][agent_id][1])
    if (curr_row, curr_col) in episode_ctx["visited_tiles"]:
        episode_ctx["repeat_steps"] += 1
    episode_ctx["visited_tiles"].add((curr_row, curr_col))
    episode_ctx["recent_positions"].append((curr_row, curr_col))
    if canonical_action is not None:
        episode_ctx["recent_actions"].append(int(canonical_action))

    if curr_danger is None and is_position_loop(episode_ctx["recent_positions"]):
        reward += reward_cfg["r_position_loop"]

    if done and rank is not None:
        reward += reward_cfg["rank_rewards"].get(rank, 0.0)

    return float(np.clip(reward, -CFG["reward_clip"], CFG["reward_clip"]))


def start_episode(env, pool, global_step):
    obs = env.reset()
    agent_id = random.randint(0, 3)
    opponents = {
        player_id: pool.get_opponent(player_id, global_step)
        for player_id in range(4)
        if player_id != agent_id
    }
    alive_mask = [bool(player.alive) for player in env.players]
    death_order = []
    episode_ctx = make_episode_context(obs, agent_id)
    return obs, agent_id, opponents, alive_mask, death_order, episode_ctx


__all__ = [
    "ActiveOpponentPool",
    "BomberEnv",
    "BoxFarmerAgent",
    "CFG",
    "CNNActorCriticV3",
    "DEVICE",
    "GeniusRuleAgent",
    "MASK_WARMUP_STEPS",
    "ModelOpponent",
    "RANK_TO_POINTS",
    "RandomAgent",
    "RolloutBuffer",
    "SimpleRuleAgent",
    "SmarterRuleAgent",
    "TacticalRuleAgent",
    "VALUE_BOMB_MASK_STEPS",
    "clone_obs",
    "clone_stats",
    "compute_competition_ranks",
    "compute_reward",
    "count_boxes_if_place",
    "danger_time",
    "get_stage",
    "has_attack_pressure",
    "has_escape_after_placing_bomb",
    "mask_eval_mode",
    "nearest_enemy_distance",
    "nearest_valuable_bomb_spot_info",
    "ppo_update",
    "prepare_policy_inputs",
    "record_deaths",
    "start_episode",
    "summarize_episode_metrics",
    "sync_linear_lr",
    "temporary_random_seed",
    "to_env_action",
    "weighted_choice",
]
