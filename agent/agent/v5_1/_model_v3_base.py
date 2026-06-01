"""
Version 3 model stack for Bomberland PPO.

V3 keeps the V2 perspective-normalized encoder and masking pipeline, and
provides versioned symbols so training/inference code can evolve separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
sys.modules.pop("_model_base", None)

from _model_base import (
    ACTION_DELTAS,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_PLACE_BOMB,
    ACTION_RIGHT,
    ACTION_STOP,
    ACTION_UP,
    AUX_DIM,
    BOMB_TIMER_MAX,
    GRID_SIZE,
    MASK_WARMUP_STEPS,
    VALUE_BOMB_MASK_STEPS,
    NUM_ACTIONS,
    NUM_CHANNELS,
    SAFE_BOMB_HORIZON,
    CNNActorCriticV2,
    build_bomb_state,
    can_hit_enemy_if_place,
    canonicalize_obs,
    count_boxes_if_place,
    current_tile_danger_time,
    enemy_same_row_or_col_with_clear_path,
    encode_aux,
    encode_obs,
    has_attack_pressure,
    has_escape_after_placing_bomb,
    masked_logits,
    nearest_enemy_distance,
    prepare_policy_inputs,
    to_canonical_action,
    to_env_action,
    valid_action_mask,
)


class CNNActorCriticV3(CNNActorCriticV2):
    """
    V3 currently reuses the V2 network architecture.

    The curriculum and reward redesign live in the local train stack; keeping the
    network stable makes it easier to attribute gains or regressions to the
    training changes.
    """


__all__ = [
    "ACTION_DELTAS",
    "ACTION_DOWN",
    "ACTION_LEFT",
    "ACTION_PLACE_BOMB",
    "ACTION_RIGHT",
    "ACTION_STOP",
    "ACTION_UP",
    "AUX_DIM",
    "BOMB_TIMER_MAX",
    "CNNActorCriticV3",
    "GRID_SIZE",
    "MASK_WARMUP_STEPS",
    "VALUE_BOMB_MASK_STEPS",
    "NUM_ACTIONS",
    "NUM_CHANNELS",
    "SAFE_BOMB_HORIZON",
    "build_bomb_state",
    "can_hit_enemy_if_place",
    "canonicalize_obs",
    "count_boxes_if_place",
    "current_tile_danger_time",
    "enemy_same_row_or_col_with_clear_path",
    "encode_aux",
    "encode_obs",
    "has_attack_pressure",
    "has_escape_after_placing_bomb",
    "masked_logits",
    "nearest_enemy_distance",
    "prepare_policy_inputs",
    "to_canonical_action",
    "to_env_action",
    "valid_action_mask",
]
