"""
Version 3 compatibility stack for V5.2 helpers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BASE_SPEC = importlib.util.spec_from_file_location("_v6_2_model_base", _HERE / "_model_base.py")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
assert _BASE_SPEC.loader is not None
_BASE_SPEC.loader.exec_module(_BASE)

ACTION_DELTAS = _BASE.ACTION_DELTAS
ACTION_DOWN = _BASE.ACTION_DOWN
ACTION_LEFT = _BASE.ACTION_LEFT
ACTION_PLACE_BOMB = _BASE.ACTION_PLACE_BOMB
ACTION_RIGHT = _BASE.ACTION_RIGHT
ACTION_STOP = _BASE.ACTION_STOP
ACTION_UP = _BASE.ACTION_UP
AUX_DIM = _BASE.AUX_DIM
BASE_AUX_DIM = _BASE.BASE_AUX_DIM
ITEM_AUX_DIM = _BASE.ITEM_AUX_DIM
BOMB_TIMER_MAX = _BASE.BOMB_TIMER_MAX
GRID_SIZE = _BASE.GRID_SIZE
MASK_WARMUP_STEPS = _BASE.MASK_WARMUP_STEPS
VALUE_BOMB_MASK_STEPS = _BASE.VALUE_BOMB_MASK_STEPS
NUM_ACTIONS = _BASE.NUM_ACTIONS
NUM_CHANNELS = _BASE.NUM_CHANNELS
SAFE_BOMB_HORIZON = _BASE.SAFE_BOMB_HORIZON
CNNActorCriticV2 = _BASE.CNNActorCriticV2
bfs_first_action_to_targets = _BASE.bfs_first_action_to_targets
build_bomb_state = _BASE.build_bomb_state
can_hit_enemy_if_place = _BASE.can_hit_enemy_if_place
canonicalize_obs = _BASE.canonicalize_obs
clone_obs_with_player_at = _BASE.clone_obs_with_player_at
count_boxes_if_place = _BASE.count_boxes_if_place
current_tile_danger_time = _BASE.current_tile_danger_time
encode_aux = _BASE.encode_aux
encode_item_aux = _BASE.encode_item_aux
encode_obs = _BASE.encode_obs
enemy_same_row_or_col_with_clear_path = _BASE.enemy_same_row_or_col_with_clear_path
has_attack_pressure = _BASE.has_attack_pressure
has_escape_after_placing_bomb = _BASE.has_escape_after_placing_bomb
masked_logits = _BASE.masked_logits
nearest_enemy_distance = _BASE.nearest_enemy_distance
nearest_valuable_bomb_spot_info = _BASE.nearest_valuable_bomb_spot_info
prepare_policy_inputs = _BASE.prepare_policy_inputs
to_canonical_action = _BASE.to_canonical_action
to_env_action = _BASE.to_env_action
valuable_bomb_spots = _BASE.valuable_bomb_spots
valid_action_mask = _BASE.valid_action_mask


class CNNActorCriticV3(CNNActorCriticV2):
    """Compatibility wrapper for older curriculum code."""


__all__ = [
    "ACTION_DELTAS",
    "ACTION_DOWN",
    "ACTION_LEFT",
    "ACTION_PLACE_BOMB",
    "ACTION_RIGHT",
    "ACTION_STOP",
    "ACTION_UP",
    "AUX_DIM",
    "BASE_AUX_DIM",
    "BOMB_TIMER_MAX",
    "CNNActorCriticV3",
    "GRID_SIZE",
    "ITEM_AUX_DIM",
    "MASK_WARMUP_STEPS",
    "VALUE_BOMB_MASK_STEPS",
    "NUM_ACTIONS",
    "NUM_CHANNELS",
    "SAFE_BOMB_HORIZON",
    "bfs_first_action_to_targets",
    "build_bomb_state",
    "can_hit_enemy_if_place",
    "canonicalize_obs",
    "clone_obs_with_player_at",
    "count_boxes_if_place",
    "current_tile_danger_time",
    "encode_aux",
    "encode_item_aux",
    "encode_obs",
    "enemy_same_row_or_col_with_clear_path",
    "has_attack_pressure",
    "has_escape_after_placing_bomb",
    "masked_logits",
    "nearest_enemy_distance",
    "nearest_valuable_bomb_spot_info",
    "prepare_policy_inputs",
    "to_canonical_action",
    "to_env_action",
    "valuable_bomb_spots",
    "valid_action_mask",
]
