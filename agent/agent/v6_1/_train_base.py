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
sys.modules.pop("_model_base", None)
sys.modules.pop("_model_v3_base", None)

_MODEL_V3_SPEC = importlib.util.spec_from_file_location(
    "_v6_model_v3_base_train_base", _HERE / "_model_v3_base.py"
)
_MODEL_V3 = importlib.util.module_from_spec(_MODEL_V3_SPEC)
assert _MODEL_V3_SPEC.loader is not None
_MODEL_V3_SPEC.loader.exec_module(_MODEL_V3)

MASK_WARMUP_STEPS = _MODEL_V3.MASK_WARMUP_STEPS
CNNActorCriticV3 = _MODEL_V3.CNNActorCriticV3
VALUE_BOMB_MASK_STEPS = _MODEL_V3.VALUE_BOMB_MASK_STEPS
bfs_first_action_to_targets = _MODEL_V3.bfs_first_action_to_targets
build_bomb_state = _MODEL_V3.build_bomb_state
can_hit_enemy_if_place = _MODEL_V3.can_hit_enemy_if_place
count_boxes_if_place = _MODEL_V3.count_boxes_if_place
current_tile_danger_time = _MODEL_V3.current_tile_danger_time
enemy_same_row_or_col_with_clear_path = _MODEL_V3.enemy_same_row_or_col_with_clear_path
has_attack_pressure = _MODEL_V3.has_attack_pressure
has_escape_after_placing_bomb = _MODEL_V3.has_escape_after_placing_bomb
nearest_enemy_distance = _MODEL_V3.nearest_enemy_distance
nearest_valuable_bomb_spot_info = _MODEL_V3.nearest_valuable_bomb_spot_info
prepare_policy_inputs = _MODEL_V3.prepare_policy_inputs
to_env_action = _MODEL_V3.to_env_action

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
from engine.map import Map

_BASE_PATH = _HERE / "_train_base_v5_1.py"
_SPEC = importlib.util.spec_from_file_location("_v6_1_train_base_v5_1", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BASE)

CFG = copy.deepcopy(_BASE.CFG)
CFG["reward_clip"] = 3.0
DEVICE = _BASE.DEVICE
RANK_TO_POINTS = _BASE.RANK_TO_POINTS
print(f"Using device: {DEVICE}")

CURRICULUM_STAGES = copy.deepcopy(_BASE.CURRICULUM_STAGES)
RESOURCE_CONTROL_STAGE_START = 1_700_000
RESOURCE_CONTROL_REWARD = {
    "r_item_capacity": 0.24,
    "r_item_radius": 0.26,
    "r_item_other": 0.05,
    "r_safe_item_progress_coef": 0.012,
    "r_safe_item_progress_clip": 0.025,
    "r_near_item_progress": 0.025,
    "r_near_item_wrong_way": -0.005,
    "r_ignore_near_safe_item": -0.005,
    "near_safe_item_dist": 3,
    "tiebreak_start_step": 300,
    "tiebreak_kill_diff_coef": 0.08,
    "tiebreak_box_diff_coef": 0.03,
    "tiebreak_item_diff_coef": 0.04,
    "tiebreak_bomb_diff_coef": 0.0,
    "tiebreak_delta_clip": 0.03,
    "resource_adv_bomb_coef": 0.0,
    "resource_adv_radius_coef": 0.02,
    "resource_adv_delta_clip": 0.03,
}
CURRICULUM_STAGES[1]["end_step"] = 500_000
CURRICULUM_STAGES[1]["baseline_bots"] = [
    (SimpleRuleAgent, 0.4),
    (SmarterRuleAgent, 0.6),
]
CURRICULUM_STAGES[1]["reward"].update(
    {
        "r_death": -0.55,
        "r_kill": 0.6,
        "r_box_destroy": 0.05,
        "r_item_collect": 0.06,
        "r_best_enemy_dist": 0.03,
        "r_danger_critical": -0.20,
        "r_danger_soon": -0.08,
        "r_danger_far": -0.02,
        "r_escape_danger": 0.03,
        "r_valuable_bomb_enemy": 0.18,
        "r_valuable_bomb_box": 0.03,
        "r_useless_bomb": -0.12,
        "r_no_escape_bomb": -0.50,
        "rank_rewards": {0: 1.0, 1: 0.2, 2: -0.2, 3: -0.8},
    }
)
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
        "r_move_closer_bomb_spot": 0.002,
        "r_move_away_bomb_spot": -0.001,
        "r_best_bomb_spot_dist": 0.008,
        "r_position_loop": -0.03,
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
CURRICULUM_STAGES[3]["end_step"] = RESOURCE_CONTROL_STAGE_START
CURRICULUM_STAGES.append(
    {
        "name": "phase_resource_control",
        "end_step": None,
        "baseline_bots": [
            (TacticalRuleAgent, 0.5),
            (GeniusRuleAgent, 0.3),
            (SmarterRuleAgent, 0.2),
        ],
        "selfplay_prob": 0.30,
        "reward": {
            "step_penalty": -0.01,
            "r_death": -1.0,
            "r_kill": 1.6,
            "r_box_destroy": 0.035,
            "r_item_collect": 0.10,
            "r_bomb_place": 0.0,
            "r_move_closer": 0.0005,
            "r_move_away": -0.0005,
            "r_best_enemy_dist": 0.004,
            "r_danger_critical": -0.24,
            "r_danger_soon": -0.10,
            "r_danger_far": -0.025,
            "r_escape_danger": 0.05,
            "r_valuable_bomb_enemy": 0.24,
            "r_valuable_bomb_box": 0.035,
            "r_useless_bomb": -0.12,
            "r_no_escape_bomb": -0.65,
            "rank_rewards": {0: 2.0, 1: 0.45, 2: -0.55, 3: -2.0},
            "r_move_closer_bomb_spot": 0.001,
            "r_move_away_bomb_spot": -0.0005,
            "r_best_bomb_spot_dist": 0.004,
            "r_position_loop": -0.03,
            **RESOURCE_CONTROL_REWARD,
        },
    }
)

ITEM_CONTROL_EASY_REWARD = copy.deepcopy(CURRICULUM_STAGES[-1]["reward"])

ITEM_CONTROL_EASY_REWARD.update(
    {
        # Tổng quát
        "step_penalty": -0.008,
        "r_death": -0.9,

        # Combat giảm nhẹ để agent không chase quá gắt
        "r_kill": 1.2,
        "r_move_closer": 0.0003,
        "r_move_away": -0.0003,
        "r_best_enemy_dist": 0.002,

        # Box + item mạnh hơn
        "r_box_destroy": 0.04,
        "r_item_collect": 0.12,
        "r_item_capacity": 0.30,
        "r_item_radius": 0.32,
        "r_item_other": 0.05,

        # Bomb vẫn giữ value, không spam
        "r_bomb_place": 0.0,
        "r_valuable_bomb_enemy": 0.18,
        "r_valuable_bomb_box": 0.04,
        "r_useless_bomb": -0.12,
        "r_no_escape_bomb": -0.60,

        # Danger
        "r_danger_critical": -0.24,
        "r_danger_soon": -0.10,
        "r_danger_far": -0.025,
        "r_escape_danger": 0.05,

        # Bomb spot vẫn có nhưng nhẹ
        "r_move_closer_bomb_spot": 0.001,
        "r_move_away_bomb_spot": -0.0005,
        "r_best_bomb_spot_dist": 0.004,

        # Item navigation mạnh hơn một chút
        "r_safe_item_progress_coef": 0.015,
        "r_safe_item_progress_clip": 0.03,
        "r_near_item_progress": 0.04,
        "r_near_item_wrong_way": -0.005,
        "r_ignore_near_safe_item": -0.005,
        "near_safe_item_dist": 3,

        # Tie-break vẫn giữ, không reward bomb diff
        "tiebreak_kill_diff_coef": 0.08,
        "tiebreak_box_diff_coef": 0.03,
        "tiebreak_item_diff_coef": 0.05,
        "tiebreak_bomb_diff_coef": 0.0,

        # 1v1 resource
        "resource_adv_bomb_coef": 0.0,
        "resource_adv_radius_coef": 0.02,

        # Rank reward nhẹ hơn resource_control hard một chút
        "rank_rewards": {0: 1.6, 1: 0.35, 2: -0.45, 3: -1.6},
    }
)

CURRICULUM_STAGES.append(
    {
        "name": "phase_item_control_easy",
        "end_step": None,
        "baseline_bots": [
            (SimpleRuleAgent, 0.25),
            (SmarterRuleAgent, 0.45),
            (BoxFarmerAgent, 0.20),
            (TacticalRuleAgent, 0.10),
        ],
        "selfplay_prob": 0.05,
        "reward": ITEM_CONTROL_EASY_REWARD,
    }
)

# v6_1 curriculum override: inherit v6 codebase but switch to item/escape-first training.
RESOURCE_CONTROL_STAGE_START = 1_700_000
ITEM_REWARD_COMMON = {
    "r_item_capacity": 0.30,
    "r_item_radius": 0.32,
    "r_item_other": 0.05,
    "r_safe_item_progress_coef": 0.015,
    "r_safe_item_progress_clip": 0.03,
    "r_near_item_progress": 0.05,
    "r_near_item_wrong_way": -0.015,
    "r_ignore_near_safe_item": -0.02,
    "r_contested_item_progress_bonus": 0.025,
    "r_contested_item_take_bonus": 0.06,
    "r_destroy_item_capacity": -0.18,
    "r_destroy_item_radius": -0.20,
    "r_destroy_item_other": -0.06,
    "near_safe_item_dist": 3,
    "item_priority_dist": 4,
    "tiebreak_start_step": 300,
    "tiebreak_kill_diff_coef": 0.08,
    "tiebreak_box_diff_coef": 0.03,
    "tiebreak_item_diff_coef": 0.05,
    "tiebreak_bomb_diff_coef": 0.0,
    "tiebreak_delta_clip": 0.03,
    "resource_adv_bomb_coef": 0.0,
    "resource_adv_radius_coef": 0.02,
    "resource_adv_delta_clip": 0.03,
    "duel_loop_window": 8,
    "duel_loop_unique_max": 2,
    "r_duel_loop_penalty": -0.035,
    "duel_safe_item_dist": 5,
    "r_duel_safe_item_pressure": 0.015,
    "r_duel_item_stall": -0.015,
    "r_duel_commit_enemy_bonus": 0.08,
}

RESOURCE_CONTROL_END_STEP = 1_500_000
LATE_GAME_CLOSER_END_STEP = 1_650_000

CURRICULUM_STAGES = [
    {
        "name": "farm_box_safe",
        "end_step": 350_000,
        "baseline_bots": [(SimpleRuleAgent, 0.40), (BoxFarmerAgent, 0.40), (SmarterRuleAgent, 0.20)],
        "selfplay_prob": 0.0,
        "reward": {
            "step_penalty": -0.008,
            "r_death": -0.70,
            "r_kill": 0.6,
            "r_box_destroy": 0.06,
            "r_item_collect": 0.06,
            "r_bomb_place": 0.0,
            "r_move_closer": 0.0002,
            "r_move_away": -0.0002,
            "r_best_enemy_dist": 0.001,
            "r_danger_critical": -0.22,
            "r_danger_soon": -0.09,
            "r_danger_far": -0.025,
            "r_escape_danger": 0.04,
            "r_valuable_bomb_enemy": 0.12,
            "r_valuable_bomb_box": 0.05,
            "r_useless_bomb": -0.12,
            "r_no_escape_bomb": -0.60,
            "rank_rewards": {0: 0.8, 1: 0.15, 2: -0.15, 3: -0.6},
            "r_move_closer_bomb_spot": 0.004,
            "r_move_away_bomb_spot": -0.0015,
            "r_best_bomb_spot_dist": 0.012,
            "r_position_loop": -0.03,
            **ITEM_REWARD_COMMON,
        },
    },
    {
        "name": "easy_item_taken",
        "end_step": 700_000,
        "baseline_bots": [(SimpleRuleAgent, 0.40), (SmarterRuleAgent, 0.40), (BoxFarmerAgent, 0.20)],
        "selfplay_prob": 0.0,
        "reward": {
            "step_penalty": -0.008,
            "r_death": -0.85,
            "r_kill": 0.9,
            "r_box_destroy": 0.045,
            "r_item_collect": 0.13,
            "r_bomb_place": 0.0,
            "r_move_closer": 0.0002,
            "r_move_away": -0.0002,
            "r_best_enemy_dist": 0.001,
            "r_danger_critical": -0.24,
            "r_danger_soon": -0.10,
            "r_danger_far": -0.025,
            "r_escape_danger": 0.05,
            "r_valuable_bomb_enemy": 0.15,
            "r_valuable_bomb_box": 0.045,
            "r_useless_bomb": -0.12,
            "r_no_escape_bomb": -0.65,
            "rank_rewards": {0: 1.2, 1: 0.25, 2: -0.25, 3: -1.0},
            "r_move_closer_bomb_spot": 0.001,
            "r_move_away_bomb_spot": -0.0005,
            "r_best_bomb_spot_dist": 0.004,
            "r_position_loop": -0.03,
            **ITEM_REWARD_COMMON,
        },
    },
    {
        "name": "dense_bomb_escape",
        "end_step": 950_000,
        "baseline_bots": [(SmarterRuleAgent, 0.50), (BoxFarmerAgent, 0.30), (TacticalRuleAgent, 0.20)],
        "selfplay_prob": 0.0,
        "reward": {
            "step_penalty": -0.01,
            "r_death": -1.1,
            "r_kill": 0.9,
            "r_box_destroy": 0.035,
            "r_item_collect": 0.10,
            "r_bomb_place": 0.0,
            "r_move_closer": 0.0002,
            "r_move_away": -0.0002,
            "r_best_enemy_dist": 0.001,
            "r_danger_critical": -0.32,
            "r_danger_soon": -0.16,
            "r_danger_far": -0.04,
            "r_escape_danger": 0.09,
            "r_valuable_bomb_enemy": 0.16,
            "r_valuable_bomb_box": 0.035,
            "r_useless_bomb": -0.14,
            "r_no_escape_bomb": -0.85,
            "rank_rewards": {0: 1.3, 1: 0.30, 2: -0.35, 3: -1.4},
            "r_move_closer_bomb_spot": 0.0005,
            "r_move_away_bomb_spot": -0.0005,
            "r_best_bomb_spot_dist": 0.002,
            "r_position_loop": -0.035,
            **ITEM_REWARD_COMMON,
        },
    },
    {
        "name": "pressure_kill",
        "end_step": 1_300_000,
        "baseline_bots": [(SmarterRuleAgent, 0.40), (TacticalRuleAgent, 0.40), (GeniusRuleAgent, 0.20)],
        "selfplay_prob": 0.10,
        "reward": {
            "step_penalty": -0.01,
            "r_death": -1.0,
            "r_kill": 1.5,
            "r_box_destroy": 0.035,
            "r_item_collect": 0.10,
            "r_bomb_place": 0.0,
            "r_move_closer": 0.0005,
            "r_move_away": -0.0005,
            "r_best_enemy_dist": 0.004,
            "r_danger_critical": -0.26,
            "r_danger_soon": -0.11,
            "r_danger_far": -0.025,
            "r_escape_danger": 0.05,
            "r_valuable_bomb_enemy": 0.24,
            "r_valuable_bomb_box": 0.035,
            "r_useless_bomb": -0.12,
            "r_no_escape_bomb": -0.70,
            "rank_rewards": {0: 1.8, 1: 0.40, 2: -0.50, 3: -1.8},
            "r_move_closer_bomb_spot": 0.001,
            "r_move_away_bomb_spot": -0.0005,
            "r_best_bomb_spot_dist": 0.004,
            "r_position_loop": -0.03,
            **ITEM_REWARD_COMMON,
        },
    },
    {
        "name": "resource_control",
        "end_step": RESOURCE_CONTROL_END_STEP,
        "baseline_bots": [(TacticalRuleAgent, 0.45), (GeniusRuleAgent, 0.35), (SmarterRuleAgent, 0.20)],
        "selfplay_prob": 0.20,
        "reward": {
            "step_penalty": -0.01,
            "r_death": -1.0,
            "r_kill": 1.6,
            "r_box_destroy": 0.04,
            "r_item_collect": 0.11,
            "r_bomb_place": 0.0,
            "r_move_closer": 0.0005,
            "r_move_away": -0.0005,
            "r_best_enemy_dist": 0.004,
            "r_danger_critical": -0.26,
            "r_danger_soon": -0.11,
            "r_danger_far": -0.025,
            "r_escape_danger": 0.05,
            "r_valuable_bomb_enemy": 0.24,
            "r_valuable_bomb_box": 0.04,
            "r_useless_bomb": -0.12,
            "r_no_escape_bomb": -0.70,
            "rank_rewards": {0: 2.0, 1: 0.45, 2: -0.55, 3: -2.0},
            "r_move_closer_bomb_spot": 0.001,
            "r_move_away_bomb_spot": -0.0005,
            "r_best_bomb_spot_dist": 0.004,
            "r_position_loop": -0.03,
            "late_step_start": 350,
            "ffa_reward_add": {
                "r_kill": -0.45,
                "r_move_closer": -0.0004,
                "r_move_away": +0.0002,
                "r_best_enemy_dist": -0.0030,
                "r_danger_critical": -0.05,
                "r_danger_soon": -0.03,
                "r_escape_danger": +0.02,
                "r_valuable_bomb_enemy": -0.08,
                "r_item_collect": +0.02,
                "r_item_capacity": +0.04,
                "r_item_radius": +0.04,
            },
            "duel_reward_add": {
                "r_kill": +0.55,
                "r_move_closer": +0.0010,
                "r_best_enemy_dist": +0.0050,
                "r_valuable_bomb_enemy": +0.10,
                "r_no_escape_bomb": +0.08,
                "resource_adv_radius_coef": +0.03,
                "tiebreak_item_diff_coef": -0.02,
                "tiebreak_box_diff_coef": -0.01,
                "tiebreak_kill_diff_coef": +0.03,
            },
            "late_step_reward_add": {
                "tiebreak_box_diff_coef": +0.02,
                "tiebreak_item_diff_coef": +0.02,
            },
            **ITEM_REWARD_COMMON,
        },
    },
    {
        "name": "late_game_closer",
        "end_step": LATE_GAME_CLOSER_END_STEP,
        "baseline_bots": [(TacticalRuleAgent, 0.40), (GeniusRuleAgent, 0.35), (SmarterRuleAgent, 0.10), (BoxFarmerAgent, 0.15)],
        "selfplay_prob": 0.35,
        "reward": {
            "step_penalty": -0.01,
            "r_death": -1.0,
            "r_kill": 1.8,
            "r_box_destroy": 0.035,
            "r_item_collect": 0.10,
            "r_bomb_place": 0.0,
            "r_move_closer": 0.0004,
            "r_move_away": -0.0003,
            "r_best_enemy_dist": 0.003,
            "r_danger_critical": -0.28,
            "r_danger_soon": -0.12,
            "r_danger_far": -0.03,
            "r_escape_danger": 0.06,
            "r_valuable_bomb_enemy": 0.22,
            "r_valuable_bomb_box": 0.03,
            "r_useless_bomb": -0.13,
            "r_no_escape_bomb": -0.72,
            "rank_rewards": {0: 2.2, 1: 0.55, 2: -0.55, 3: -2.1},
            "r_move_closer_bomb_spot": 0.0007,
            "r_move_away_bomb_spot": -0.0004,
            "r_best_bomb_spot_dist": 0.003,
            "r_position_loop": -0.035,
            "late_step_start": 320,
            "ffa_reward_add": {
                "r_kill": -0.70,
                "r_move_closer": -0.0005,
                "r_move_away": +0.0002,
                "r_best_enemy_dist": -0.0020,
                "r_danger_critical": -0.05,
                "r_danger_soon": -0.03,
                "r_escape_danger": +0.02,
                "r_valuable_bomb_enemy": -0.09,
                "r_item_collect": +0.01,
                "r_item_capacity": +0.03,
                "r_item_radius": +0.03,
            },
            "duel_reward_add": {
                "r_kill": +0.80,
                "r_move_closer": +0.0014,
                "r_best_enemy_dist": +0.0060,
                "r_valuable_bomb_enemy": +0.13,
                "r_no_escape_bomb": +0.10,
                "resource_adv_radius_coef": +0.04,
                "tiebreak_kill_diff_coef": +0.04,
                "tiebreak_item_diff_coef": -0.03,
                "tiebreak_box_diff_coef": -0.02,
            },
            "late_step_reward_add": {
                "tiebreak_kill_diff_coef": +0.02,
                "tiebreak_box_diff_coef": +0.02,
            },
            **ITEM_REWARD_COMMON,
        },
    },
    {
        "name": "selfplay_hard",
        "end_step": None,
        "baseline_bots": [(TacticalRuleAgent, 0.35), (GeniusRuleAgent, 0.35), (SmarterRuleAgent, 0.30)],
        "selfplay_prob": 0.45,
        "reward": {
            "step_penalty": -0.01,
            "r_death": -1.0,
            "r_kill": 1.7,
            "r_box_destroy": 0.035,
            "r_item_collect": 0.10,
            "r_bomb_place": 0.0,
            "r_move_closer": 0.0005,
            "r_move_away": -0.0005,
            "r_best_enemy_dist": 0.004,
            "r_danger_critical": -0.26,
            "r_danger_soon": -0.11,
            "r_danger_far": -0.025,
            "r_escape_danger": 0.05,
            "r_valuable_bomb_enemy": 0.25,
            "r_valuable_bomb_box": 0.035,
            "r_useless_bomb": -0.12,
            "r_no_escape_bomb": -0.70,
            "rank_rewards": {0: 2.2, 1: 0.50, 2: -0.60, 3: -2.2},
            "r_move_closer_bomb_spot": 0.001,
            "r_move_away_bomb_spot": -0.0005,
            "r_best_bomb_spot_dist": 0.004,
            "r_position_loop": -0.03,
            "late_step_start": 320,
            "ffa_reward_add": {
                "r_kill": -0.55,
                "r_move_closer": -0.0004,
                "r_move_away": +0.0002,
                "r_best_enemy_dist": -0.0025,
                "r_danger_critical": -0.04,
                "r_danger_soon": -0.02,
                "r_escape_danger": +0.02,
                "r_valuable_bomb_enemy": -0.07,
                "r_item_collect": +0.01,
                "r_item_capacity": +0.03,
                "r_item_radius": +0.03,
            },
            "duel_reward_add": {
                "r_kill": +0.65,
                "r_move_closer": +0.0012,
                "r_best_enemy_dist": +0.0050,
                "r_valuable_bomb_enemy": +0.12,
                "r_no_escape_bomb": +0.08,
                "resource_adv_radius_coef": +0.03,
                "tiebreak_kill_diff_coef": +0.04,
                "tiebreak_item_diff_coef": -0.02,
            },
            "late_step_reward_add": {
                "tiebreak_kill_diff_coef": +0.02,
                "tiebreak_box_diff_coef": +0.02,
            },
            **ITEM_REWARD_COMMON,
        },
    },
]

ITEM_SHAPING_STAGES = {
    "farm_box_safe",
    "easy_item_taken",
    "dense_bomb_escape",
    "pressure_kill",
    "resource_control",
    "selfplay_hard",
}


def get_stage(current_step: int):
    for stage in CURRICULUM_STAGES:
        if stage["end_step"] is None or current_step < stage["end_step"]:
            return stage
    return CURRICULUM_STAGES[-1]


def get_stage_by_name(stage_name: str):
    aliases = {
        "phase_resource_control": "resource_control",
        "phase_item_control_easy": "easy_item_taken",
        "phase_late_game_closer": "late_game_closer",
        "phase4_full": "selfplay_hard",
    }
    stage_name = aliases.get(stage_name, stage_name)
    for stage in CURRICULUM_STAGES:
        if stage["name"] == stage_name:
            return stage
    raise KeyError(f"Unknown curriculum stage: {stage_name}")


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


def _apply_reward_add(base_reward_cfg: dict, reward_add: dict | None) -> dict:
    if not reward_add:
        return base_reward_cfg
    merged = dict(base_reward_cfg)
    for key, delta in reward_add.items():
        base_value = merged.get(key)
        if isinstance(base_value, (int, float)) and isinstance(delta, (int, float)):
            merged[key] = float(base_value) + float(delta)
        else:
            merged[key] = delta
    return merged


def effective_reward_cfg(stage: dict, alive_count: int, current_step: int | None) -> dict:
    reward_cfg = stage["reward"]
    dynamic_keys = ("ffa_reward_add", "duel_reward_add", "late_step_reward_add")
    if not any(key in reward_cfg for key in dynamic_keys):
        return reward_cfg

    effective = reward_cfg
    if alive_count >= 3:
        effective = _apply_reward_add(effective, reward_cfg.get("ffa_reward_add"))
    elif alive_count == 2:
        effective = _apply_reward_add(effective, reward_cfg.get("duel_reward_add"))

    late_step_start = reward_cfg.get("late_step_start")
    if late_step_start is not None and current_step is not None and int(current_step) >= int(late_step_start):
        effective = _apply_reward_add(effective, reward_cfg.get("late_step_reward_add"))
    return effective


def is_position_loop(recent_positions: deque[tuple[int, int]]) -> bool:
    if len(recent_positions) < 8:
        return False
    last = list(recent_positions)
    if len(set(last[-6:])) == 1:
        return True
    if len(set(last[-8:])) == 2:
        return True
    return False


def is_duel_stalemate(episode_ctx: dict, reward_cfg: dict) -> bool:
    window = int(reward_cfg.get("duel_loop_window", 8))
    unique_max = int(reward_cfg.get("duel_loop_unique_max", 2))
    recent_positions = episode_ctx.get("recent_positions")
    if recent_positions is None or len(recent_positions) < window:
        return False
    tail = list(recent_positions)[-window:]
    return len(set(tail)) <= unique_max


def alive_agent_ids(obs: dict) -> list[int]:
    players = np.asarray(obs["players"], dtype=np.int64)
    return [idx for idx in range(len(players)) if int(players[idx][2]) == 1]


def active_bombs_owned(obs: dict, agent_id: int) -> int:
    bombs = np.asarray(obs["bombs"], dtype=np.int64)
    if bombs.size == 0:
        return 0
    if bombs.ndim == 1:
        bombs = bombs.reshape(1, -1)
    return int(np.sum(bombs[:, 3] == agent_id))


def inferred_bomb_capacity(obs: dict, agent_id: int) -> int:
    players = np.asarray(obs["players"], dtype=np.int64)
    bombs_left = int(players[agent_id][3])
    return bombs_left + active_bombs_owned(obs, agent_id)


def has_clear_kill_opportunity(obs: dict, agent_id: int) -> bool:
    return has_escape_after_placing_bomb(obs, agent_id) and can_hit_enemy_if_place(obs, agent_id)


def nearest_safe_item_dist(obs: dict, agent_id: int) -> int | None:
    info = nearest_safe_item_info(obs, agent_id)
    if info is None:
        return None
    return int(info["dist"])


def nearest_safe_item_info(obs: dict, agent_id: int) -> dict | None:
    grid = np.asarray(obs["map"], dtype=np.int64)
    targets = [
        (row, col, int(grid[row, col]))
        for row in range(1, grid.shape[0] - 1)
        for col in range(1, grid.shape[1] - 1)
        if int(grid[row, col]) in (Map.ITEM_RADIUS, Map.ITEM_CAPACITY)
    ]
    if not targets:
        return None
    target_positions = {(row, col) for row, col, _tile in targets}
    _first_action, dist = bfs_first_action_to_targets(obs, agent_id, target_positions, avoid_danger_leq=2)
    if dist >= 999:
        return None

    players = np.asarray(obs["players"], dtype=np.int64)
    my_row = int(players[agent_id][0])
    my_col = int(players[agent_id][1])

    best_pos = None
    best_tile = None
    best_manhattan = 999
    for row, col, tile in targets:
        md = abs(row - my_row) + abs(col - my_col)
        if md < best_manhattan:
            best_manhattan = md
            best_pos = (row, col)
            best_tile = tile

    contested = False
    if best_pos is not None:
        enemy_min = 999
        for enemy_id in alive_agent_ids(obs):
            if enemy_id == agent_id:
                continue
            er = int(players[enemy_id][0])
            ec = int(players[enemy_id][1])
            enemy_min = min(enemy_min, abs(er - best_pos[0]) + abs(ec - best_pos[1]))
        contested = enemy_min <= best_manhattan + 1

    return {
        "dist": int(dist),
        "pos": best_pos,
        "tile": best_tile,
        "contested": bool(contested),
    }


def _item_positions(obs: dict) -> dict[tuple[int, int], int]:
    grid = np.asarray(obs["map"], dtype=np.int64)
    items: dict[tuple[int, int], int] = {}
    for row in range(1, grid.shape[0] - 1):
        for col in range(1, grid.shape[1] - 1):
            tile = int(grid[row, col])
            if tile in (Map.ITEM_RADIUS, Map.ITEM_CAPACITY):
                items[(row, col)] = tile
    return items


def _bomb_blast_tiles(obs: dict, row: int, col: int, radius: int) -> set[tuple[int, int]]:
    grid = np.asarray(obs["map"], dtype=np.int64)
    tiles = {(int(row), int(col))}
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for step in range(1, max(int(radius), 0) + 1):
            nr = int(row) + dr * step
            nc = int(col) + dc * step
            if nr < 0 or nr >= grid.shape[0] or nc < 0 or nc >= grid.shape[1]:
                break
            cell = int(grid[nr, nc])
            if cell == Map.WALL:
                break
            tiles.add((nr, nc))
            if cell == Map.BOX:
                break
    return tiles


def destroyed_items_by_own_bomb(
    prev_obs: dict,
    obs: dict,
    agent_id: int,
    all_prev_stats: list[dict] | None = None,
    all_curr_stats: list[dict] | None = None,
) -> list[int]:
    prev_items = _item_positions(prev_obs)
    if not prev_items:
        return []

    curr_items = _item_positions(obs)
    removed_positions = [pos for pos in prev_items if pos not in curr_items]
    if not removed_positions:
        return []

    picked_positions: set[tuple[int, int]] = set()
    if all_prev_stats and all_curr_stats:
        players = np.asarray(obs["players"], dtype=np.int64)
        for player_id in range(min(len(all_prev_stats), len(all_curr_stats), len(players))):
            prev_item_count = int(all_prev_stats[player_id].get("items", 0))
            curr_item_count = int(all_curr_stats[player_id].get("items", 0))
            if curr_item_count > prev_item_count:
                picked_positions.add((int(players[player_id][0]), int(players[player_id][1])))

    own_blast_tiles: set[tuple[int, int]] = set()
    bombs = np.asarray(prev_obs["bombs"], dtype=np.int64)
    if bombs.size > 0:
        if bombs.ndim == 1:
            bombs = bombs.reshape(1, -1)
        players_prev = np.asarray(prev_obs["players"], dtype=np.int64)
        for bomb in bombs:
            owner = int(bomb[3])
            if owner != agent_id:
                continue
            radius = 1 + int(players_prev[owner][4])
            own_blast_tiles.update(_bomb_blast_tiles(prev_obs, int(bomb[0]), int(bomb[1]), radius))

    destroyed_tiles = []
    for pos in removed_positions:
        if pos in picked_positions:
            continue
        if pos in own_blast_tiles:
            destroyed_tiles.append(prev_items[pos])
    return destroyed_tiles


def infer_picked_item_type(prev_obs: dict, obs: dict, agent_id: int, item_delta: int) -> str | None:
    if item_delta <= 0:
        return None
    players = np.asarray(obs["players"], dtype=np.int64)
    row = int(players[agent_id][0])
    col = int(players[agent_id][1])
    prev_grid = np.asarray(prev_obs["map"], dtype=np.int64)
    prev_tile = int(prev_grid[row, col])
    if prev_tile == Map.ITEM_CAPACITY:
        return "capacity"
    if prev_tile == Map.ITEM_RADIUS:
        return "radius"
    return "other"


def _player_competition_tuple(stats: dict) -> tuple[int, int, int, int]:
    return (
        int(stats.get("kills", 0)),
        int(stats.get("boxes", 0)),
        int(stats.get("items", 0)),
        int(stats.get("bombs", 0)),
    )


def tie_break_advantage(stats_list: list[dict] | None, agent_id: int, reward_cfg: dict) -> float | None:
    if not stats_list or agent_id >= len(stats_list):
        return None
    enemy_tuples = [
        _player_competition_tuple(stats)
        for idx, stats in enumerate(stats_list)
        if idx != agent_id
    ]
    if not enemy_tuples:
        return None
    my_tuple = _player_competition_tuple(stats_list[agent_id])
    best_enemy_tuple = max(enemy_tuples)
    return (
        reward_cfg["tiebreak_kill_diff_coef"] * (my_tuple[0] - best_enemy_tuple[0])
        + reward_cfg["tiebreak_box_diff_coef"] * (my_tuple[1] - best_enemy_tuple[1])
        + reward_cfg["tiebreak_item_diff_coef"] * (my_tuple[2] - best_enemy_tuple[2])
        + reward_cfg["tiebreak_bomb_diff_coef"] * (my_tuple[3] - best_enemy_tuple[3])
    )


def resource_advantage(obs: dict, agent_id: int, reward_cfg: dict) -> float | None:
    players = np.asarray(obs["players"], dtype=np.int64)
    alive_ids = alive_agent_ids(obs)
    if len(alive_ids) != 2 or agent_id not in alive_ids:
        return None
    enemy_id = alive_ids[0] if alive_ids[1] == agent_id else alive_ids[1]
    my_capacity = inferred_bomb_capacity(obs, agent_id)
    enemy_capacity = inferred_bomb_capacity(obs, enemy_id)
    my_radius = 1 + int(players[agent_id][4])
    enemy_radius = 1 + int(players[enemy_id][4])
    return (
        reward_cfg["resource_adv_bomb_coef"] * (my_capacity - enemy_capacity)
        + reward_cfg["resource_adv_radius_coef"] * (my_radius - enemy_radius)
    )


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
        "safe_item_opportunities": 0,
        "safe_item_taken": 0,
        "near_item_ignored": 0,
        "item_progress_events": 0,
        "item_progress_reward_total": 0.0,
        "duel_stall_steps": 0,
        "duel_item_stall_events": 0,
        "duel_commit_bomb_events": 0,
        "visited_tiles": {(start_row, start_col)},
        "repeat_steps": 0,
        "recent_positions": recent_positions,
        "recent_actions": deque(maxlen=12),
    }


def summarize_episode_metrics(episode_ctx: dict, final_stats: dict) -> dict:
    metrics = _BASE.summarize_episode_metrics(episode_ctx, final_stats)

    safe_item_opps = int(episode_ctx.get("safe_item_opportunities", 0))
    safe_item_taken = int(episode_ctx.get("safe_item_taken", 0))
    near_item_ignored = int(episode_ctx.get("near_item_ignored", 0))

    boxes = max(int(final_stats.get("boxes", 0)), 1)
    bombs = max(int(final_stats.get("bombs", 0)), 1)

    metrics.update(
        {
            "safe_item_opportunities": float(safe_item_opps),
            "safe_item_taken": float(safe_item_taken),
            "safe_item_take_rate": safe_item_taken / max(safe_item_opps, 1),
            "near_item_ignore_rate": near_item_ignored / max(safe_item_opps, 1),
            "items_per_box": float(final_stats.get("items", 0)) / boxes,
            "boxes_per_bomb": float(final_stats.get("boxes", 0)) / bombs,
            "item_progress_events": float(episode_ctx.get("item_progress_events", 0)),
            "item_progress_reward_total": float(episode_ctx.get("item_progress_reward_total", 0.0)),
            "duel_stall_steps": float(episode_ctx.get("duel_stall_steps", 0)),
            "duel_item_stall_events": float(episode_ctx.get("duel_item_stall_events", 0)),
            "duel_commit_bomb_events": float(episode_ctx.get("duel_commit_bomb_events", 0)),
        }
    )
    return metrics


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
    all_prev_stats: list[dict] | None = None,
    all_curr_stats: list[dict] | None = None,
    current_step: int | None = None,
):
    alive_count = len(alive_agent_ids(obs))
    reward_cfg = effective_reward_cfg(stage, alive_count, current_step)
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
            if alive_count == 2:
                reward += reward_cfg.get("r_duel_commit_enemy_bonus", 0.0)
                episode_ctx["duel_commit_bomb_events"] += bombs_placed
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

    if stage["name"] in ITEM_SHAPING_STAGES:
        item_delta = curr_stats["items"] - prev_stats["items"]
        picked_item_type = infer_picked_item_type(prev_obs, obs, agent_id, item_delta)
        if picked_item_type == "capacity":
            reward += reward_cfg["r_item_capacity"]
        elif picked_item_type == "radius":
            reward += reward_cfg["r_item_radius"]
        elif picked_item_type is not None:
            reward += reward_cfg["r_item_other"]

        kill_opportunity = has_clear_kill_opportunity(prev_obs, agent_id)
        prev_item_info = nearest_safe_item_info(prev_obs, agent_id)
        curr_item_info = nearest_safe_item_info(obs, agent_id)
        item_priority_override = (
            prev_item_info is not None
            and prev_item_info["dist"] <= reward_cfg["item_priority_dist"]
        )
        safe_for_item = (
            (prev_danger is None or prev_danger > 2)
            and (curr_danger is None or curr_danger > 2)
            and (not kill_opportunity or item_priority_override)
        )
        if safe_for_item:
            prev_item_dist = None if prev_item_info is None else int(prev_item_info["dist"])
            curr_item_dist = None if curr_item_info is None else int(curr_item_info["dist"])

            if (
                prev_item_dist is not None
                and prev_item_dist <= reward_cfg["near_safe_item_dist"]
            ):
                episode_ctx["safe_item_opportunities"] += 1

                if item_delta > 0:
                    episode_ctx["safe_item_taken"] += 1
                    if bool(prev_item_info.get("contested", False)):
                        reward += reward_cfg["r_contested_item_take_bonus"]

            if prev_item_dist is not None and curr_item_dist is not None:
                item_progress = prev_item_dist - curr_item_dist
                progress_reward = float(
                    np.clip(
                        reward_cfg["r_safe_item_progress_coef"] * item_progress,
                        -reward_cfg["r_safe_item_progress_clip"],
                        reward_cfg["r_safe_item_progress_clip"],
                    )
                )
                reward += progress_reward

                if item_progress != 0:
                    episode_ctx["item_progress_events"] += 1
                    episode_ctx["item_progress_reward_total"] += progress_reward

                if prev_item_dist <= reward_cfg["near_safe_item_dist"]:
                    if curr_item_dist < prev_item_dist:
                        reward += reward_cfg["r_near_item_progress"]
                        if bool(prev_item_info.get("contested", False)):
                            reward += reward_cfg["r_contested_item_progress_bonus"]
                    elif item_delta <= 0 and curr_item_dist > prev_item_dist:
                        reward += reward_cfg["r_near_item_wrong_way"]
                        episode_ctx["near_item_ignored"] += 1

            if (
                item_delta <= 0
                and prev_item_dist is not None
                and prev_item_dist <= reward_cfg["near_safe_item_dist"]
                and curr_item_dist is not None
                and curr_item_dist > prev_item_dist
            ):
                reward += reward_cfg["r_ignore_near_safe_item"]

        destroyed_items = destroyed_items_by_own_bomb(
            prev_obs,
            obs,
            agent_id,
            all_prev_stats=all_prev_stats,
            all_curr_stats=all_curr_stats,
        )
        for destroyed_tile in destroyed_items:
            if destroyed_tile == Map.ITEM_CAPACITY:
                reward += reward_cfg["r_destroy_item_capacity"]
            elif destroyed_tile == Map.ITEM_RADIUS:
                reward += reward_cfg["r_destroy_item_radius"]
            else:
                reward += reward_cfg["r_destroy_item_other"]

        if (
            (current_step is not None and current_step >= reward_cfg["tiebreak_start_step"])
            or alive_count <= 2
        ):
            prev_tb_score = tie_break_advantage(all_prev_stats, agent_id, reward_cfg)
            curr_tb_score = tie_break_advantage(all_curr_stats, agent_id, reward_cfg)
            if prev_tb_score is not None and curr_tb_score is not None:
                reward += float(
                    np.clip(
                        curr_tb_score - prev_tb_score,
                        -reward_cfg["tiebreak_delta_clip"],
                        reward_cfg["tiebreak_delta_clip"],
                    )
                )

        if alive_count == 2:
            if curr_danger is None and is_duel_stalemate(episode_ctx, reward_cfg):
                reward += reward_cfg.get("r_duel_loop_penalty", 0.0)
                episode_ctx["duel_stall_steps"] += 1

            if safe_for_item and prev_item_info is not None:
                prev_item_dist = int(prev_item_info["dist"])
                duel_item_dist_limit = int(
                    reward_cfg.get("duel_safe_item_dist", reward_cfg["item_priority_dist"])
                )
                if prev_item_dist <= duel_item_dist_limit:
                    curr_item_dist = None if curr_item_info is None else int(curr_item_info["dist"])
                    if item_delta <= 0:
                        if curr_item_dist is not None and curr_item_dist < prev_item_dist:
                            reward += reward_cfg.get("r_duel_safe_item_pressure", 0.0)
                        elif curr_item_dist is None or curr_item_dist >= prev_item_dist:
                            reward += reward_cfg.get("r_duel_item_stall", 0.0)
                            episode_ctx["duel_item_stall_events"] += 1

            prev_resource_adv = resource_advantage(prev_obs, agent_id, reward_cfg)
            curr_resource_adv = resource_advantage(obs, agent_id, reward_cfg)
            if prev_resource_adv is not None and curr_resource_adv is not None:
                reward += float(
                    np.clip(
                        curr_resource_adv - prev_resource_adv,
                        -reward_cfg["resource_adv_delta_clip"],
                        reward_cfg["resource_adv_delta_clip"],
                    )
                )

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
    "RESOURCE_CONTROL_STAGE_START",
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
    "get_stage_by_name",
    "has_attack_pressure",
    "has_clear_kill_opportunity",
    "has_escape_after_placing_bomb",
    "infer_picked_item_type",
    "is_duel_stalemate",
    "mask_eval_mode",
    "nearest_safe_item_dist",
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
