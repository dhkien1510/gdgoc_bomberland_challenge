"""
V5.2 shared model and preprocessing helpers for Bomberland PPO.
"""

from __future__ import annotations

import importlib.util
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_BASE_PATH = _HERE.parent / "v5_1" / "_model_base.py"
_SPEC = importlib.util.spec_from_file_location("_v5_1_model_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BASE)

GRID_SIZE = _BASE.GRID_SIZE
BOMB_TIMER_MAX = _BASE.BOMB_TIMER_MAX
NUM_ACTIONS = _BASE.NUM_ACTIONS
NUM_CHANNELS = _BASE.NUM_CHANNELS
AUX_DIM = 18
MASK_WARMUP_STEPS = _BASE.MASK_WARMUP_STEPS
SAFE_BOMB_HORIZON = _BASE.SAFE_BOMB_HORIZON
VALUE_BOMB_MASK_STEPS = 1_300_000

ACTION_STOP = _BASE.ACTION_STOP
ACTION_LEFT = _BASE.ACTION_LEFT
ACTION_RIGHT = _BASE.ACTION_RIGHT
ACTION_UP = _BASE.ACTION_UP
ACTION_DOWN = _BASE.ACTION_DOWN
ACTION_PLACE_BOMB = _BASE.ACTION_PLACE_BOMB
ACTION_DELTAS = _BASE.ACTION_DELTAS

_iter_bombs = _BASE._iter_bombs
_flip_flags = _BASE._flip_flags
canonicalize_obs = _BASE.canonicalize_obs
_swap_axis_actions = _BASE._swap_axis_actions
to_env_action = _BASE.to_env_action
to_canonical_action = _BASE.to_canonical_action
_bomb_radius = _BASE._bomb_radius
_walkable_tile = _BASE._walkable_tile
_compute_explosion_tiles = _BASE._compute_explosion_tiles
build_bomb_state = _BASE.build_bomb_state
current_tile_danger_time = _BASE.current_tile_danger_time
encode_obs = _BASE.encode_obs
masked_logits = _BASE.masked_logits


def nearest_enemy_distance(obs: dict, agent_id: int) -> float:
    return _BASE.nearest_enemy_distance(obs, agent_id)


def safe_until(earliest_danger: dict, pos: tuple[int, int], until_step: int) -> bool:
    danger_time = earliest_danger.get(pos)
    return danger_time is None or danger_time > until_step


def has_escape_after_placing_bomb(obs: dict, agent_id: int) -> bool:
    players = np.asarray(obs["players"], dtype=np.int64)
    grid = np.asarray(obs["map"], dtype=np.int64)

    if agent_id < 0 or agent_id >= len(players):
        return False

    row, col, alive, bombs_left, radius_bonus = (int(v) for v in players[agent_id][:5])
    if alive == 0 or bombs_left <= 0:
        return False

    virtual_bomb = {
        "row": row,
        "col": col,
        "timer": int(BOMB_TIMER_MAX),
        "owner": agent_id,
        "radius": 1 + radius_bonus,
    }
    bomb_state = build_bomb_state(obs, extra_bomb=virtual_bomb)
    virtual = bomb_state["virtual_bomb"]
    if virtual is None:
        return False

    earliest_danger = bomb_state["earliest_danger"]
    bomb_positions = bomb_state["bomb_positions"]
    virtual_blast_tiles = virtual["blast_tiles"]
    virtual_explode_at = virtual["explode_at"]
    max_depth = min(SAFE_BOMB_HORIZON, max(virtual_explode_at - 1, 0))

    def safe_at_time(pos: tuple[int, int], step: int) -> bool:
        danger_time = earliest_danger.get(pos)
        return danger_time is None or danger_time > step

    if not safe_at_time((row, col), 1):
        return False

    queue = deque([(1, row, col)])
    visited = {(1, row, col)}

    while queue:
        step, cur_row, cur_col = queue.popleft()

        if (
            step < virtual_explode_at
            and (cur_row, cur_col) not in virtual_blast_tiles
            and safe_until(earliest_danger, (cur_row, cur_col), virtual_explode_at)
        ):
            return True

        if step >= max_depth:
            continue

        next_step = step + 1
        for dr, dc in ((0, 0), *ACTION_DELTAS.values()):
            nr, nc = cur_row + dr, cur_col + dc
            moved = (nr, nc) != (cur_row, cur_col)

            if not _walkable_tile(grid, nr, nc):
                continue

            if moved:
                occupied_until = bomb_positions.get((nr, nc))
                if occupied_until is not None and occupied_until > next_step:
                    continue

            if not safe_at_time((nr, nc), next_step):
                continue

            state = (next_step, nr, nc)
            if state in visited:
                continue
            visited.add(state)
            queue.append(state)

    return False


def _virtual_bomb_info(obs: dict, agent_id: int) -> tuple[set[tuple[int, int]], int]:
    return _BASE._virtual_bomb_info(obs, agent_id)


def count_boxes_if_place(obs: dict, agent_id: int) -> int:
    return _BASE.count_boxes_if_place(obs, agent_id)


def can_hit_enemy_if_place(obs: dict, agent_id: int) -> bool:
    return _BASE.can_hit_enemy_if_place(obs, agent_id)


def enemy_same_row_or_col_with_clear_path(obs: dict, agent_id: int) -> bool:
    return _BASE.enemy_same_row_or_col_with_clear_path(obs, agent_id)


def has_attack_pressure(obs: dict, agent_id: int) -> bool:
    nearest_enemy_dist = nearest_enemy_distance(obs, agent_id)
    return (
        can_hit_enemy_if_place(obs, agent_id)
        or enemy_same_row_or_col_with_clear_path(obs, agent_id)
        or (0.0 < nearest_enemy_dist <= 3.0)
    )


def clone_obs_with_player_at(obs: dict, agent_id: int, row: int, col: int) -> dict:
    new_obs = {
        "map": np.array(obs["map"], copy=True),
        "players": np.array(obs["players"], copy=True),
        "bombs": np.array(obs["bombs"], copy=True),
    }
    new_obs["players"][agent_id, 0] = row
    new_obs["players"][agent_id, 1] = col
    return new_obs


def valuable_bomb_spots(obs: dict, agent_id: int, require_escape: bool = True) -> set[tuple[int, int]]:
    grid = np.asarray(obs["map"], dtype=np.int64)
    bomb_state = build_bomb_state(obs)
    blocked_bomb_tiles = set(bomb_state["bomb_positions"].keys())
    spots = set()

    for row in range(1, GRID_SIZE - 1):
        for col in range(1, GRID_SIZE - 1):
            if not _walkable_tile(grid, row, col):
                continue
            if (row, col) in blocked_bomb_tiles:
                continue

            test_obs = clone_obs_with_player_at(obs, agent_id, row, col)
            if require_escape and not has_escape_after_placing_bomb(test_obs, agent_id):
                continue

            boxes_hit = count_boxes_if_place(test_obs, agent_id)
            enemy_hit = can_hit_enemy_if_place(test_obs, agent_id)
            if enemy_hit or boxes_hit > 0:
                spots.add((row, col))

    return spots


def bfs_first_action_to_targets(
    obs: dict,
    agent_id: int,
    targets: set[tuple[int, int]],
    avoid_danger_leq: int = 2,
) -> tuple[int | None, int]:
    if not targets:
        return None, 999

    grid = np.asarray(obs["map"], dtype=np.int64)
    players = np.asarray(obs["players"], dtype=np.int64)
    bombs = build_bomb_state(obs)

    start = (int(players[agent_id][0]), int(players[agent_id][1]))
    bomb_tiles = set(bombs["bomb_positions"].keys())

    def safe(pos: tuple[int, int]) -> bool:
        danger = bombs["earliest_danger"].get(pos)
        return danger is None or danger > avoid_danger_leq

    queue = deque([(start[0], start[1], None, 0)])
    visited = {start}

    while queue:
        row, col, first_action, dist = queue.popleft()
        if (row, col) in targets:
            return first_action, dist

        for action, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = row + dr, col + dc
            pos = (nr, nc)
            if pos in visited:
                continue
            if not _walkable_tile(grid, nr, nc):
                continue
            if pos in bomb_tiles:
                continue
            if not safe(pos):
                continue
            visited.add(pos)
            queue.append((nr, nc, action if first_action is None else first_action, dist + 1))

    return None, 999


def nearest_valuable_bomb_spot_info(obs: dict, agent_id: int) -> tuple[int | None, int, int]:
    spots = valuable_bomb_spots(obs, agent_id, require_escape=True)
    first_action, dist = bfs_first_action_to_targets(obs, agent_id, spots)
    return first_action, dist, len(spots)


def encode_aux(obs: dict, agent_id: int, bomb_state=None) -> torch.Tensor:
    players = np.asarray(obs["players"], dtype=np.float32)
    player = players[agent_id]

    enemies = [
        abs(int(player[0]) - int(other[0])) + abs(int(player[1]) - int(other[1]))
        for i, other in enumerate(players)
        if i != agent_id and int(other[2]) == 1
    ]
    nearest_enemy_dist = min(enemies) if enemies else 0.0
    alive_enemy_count = sum(1 for i, other in enumerate(players) if i != agent_id and int(other[2]) == 1)

    if bomb_state is None:
        bomb_state = build_bomb_state(obs)
    current_danger = current_tile_danger_time(obs, agent_id, bomb_state)
    can_escape_if_place = has_escape_after_placing_bomb(obs, agent_id)
    can_hit_if_place = can_hit_enemy_if_place(obs, agent_id)
    boxes_hit_if_place = count_boxes_if_place(obs, agent_id)
    same_line_clear = enemy_same_row_or_col_with_clear_path(obs, agent_id)
    attack_pressure = has_attack_pressure(obs, agent_id)
    if current_danger is None:
        danger_time_norm = 0.0
    else:
        danger_time_norm = (BOMB_TIMER_MAX + 1.0 - min(float(current_danger), BOMB_TIMER_MAX)) / BOMB_TIMER_MAX

    first_action_to_bomb_spot, dist_to_bomb_spot, num_bomb_spots = nearest_valuable_bomb_spot_info(obs, agent_id)
    target_action_onehot = [0.0] * 5
    if first_action_to_bomb_spot is None:
        target_action_onehot[0] = 1.0
    else:
        target_action_onehot[int(first_action_to_bomb_spot)] = 1.0

    aux = [
        float(player[3]) / 5.0,
        float(player[4]) / 4.0,
        float(nearest_enemy_dist) / (2.0 * (GRID_SIZE - 1)),
        float(alive_enemy_count) / 3.0,
        1.0 if current_danger is not None and current_danger <= 2 else 0.0,
        danger_time_norm,
        1.0 if can_escape_if_place else 0.0,
        1.0 if can_hit_if_place else 0.0,
        min(float(boxes_hit_if_place), 4.0) / 4.0,
        1.0 if attack_pressure else 0.0,
        1.0 if same_line_clear else 0.0,
        min(float(dist_to_bomb_spot), 20.0) / 20.0,
        min(float(num_bomb_spots), 10.0) / 10.0,
        *target_action_onehot,
    ]
    return torch.tensor(aux, dtype=torch.float32)


def valid_action_mask(
    obs: dict,
    agent_id: int,
    bomb_state=None,
    current_step: int = 0,
    warmup_steps: int = MASK_WARMUP_STEPS,
    value_bomb_mask_steps: int = VALUE_BOMB_MASK_STEPS,
    eval_mode: bool = False,
) -> torch.Tensor:
    players = np.asarray(obs["players"], dtype=np.int64)
    grid = np.asarray(obs["map"], dtype=np.int64)
    if bomb_state is None:
        bomb_state = build_bomb_state(obs)

    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)

    if agent_id < 0 or agent_id >= len(players):
        return torch.from_numpy(mask)

    row, col, alive, bombs_left, _radius_bonus = (int(v) for v in players[agent_id][:5])
    if alive == 0:
        return torch.from_numpy(mask)

    def safe_at_time(pos: tuple[int, int], step: int) -> bool:
        danger_time = bomb_state["earliest_danger"].get(pos)
        return danger_time is None or danger_time > step

    strict_bomb_mask = eval_mode or current_step >= warmup_steps
    strict_value_bomb_mask = current_step >= value_bomb_mask_steps

    mask[ACTION_STOP] = safe_at_time((row, col), 1)
    bomb_tiles = set(bomb_state["bomb_positions"].keys())

    for action, (dr, dc) in ACTION_DELTAS.items():
        nr, nc = row + dr, col + dc
        if not _walkable_tile(grid, nr, nc):
            continue
        if (nr, nc) in bomb_tiles:
            continue
        if not safe_at_time((nr, nc), 1):
            continue
        mask[action] = True

    if bombs_left > 0 and (row, col) not in bomb_tiles and safe_at_time((row, col), 1):
        escape = has_escape_after_placing_bomb(obs, agent_id)
        valuable = can_hit_enemy_if_place(obs, agent_id) or count_boxes_if_place(obs, agent_id) > 0
        if (not strict_bomb_mask or escape) and (not strict_value_bomb_mask or valuable):
            mask[ACTION_PLACE_BOMB] = True

    if not mask.any():
        mask[ACTION_STOP] = True

    return torch.from_numpy(mask)


def prepare_policy_inputs(
    obs: dict,
    agent_id: int,
    current_step: int = 0,
    warmup_steps: int = MASK_WARMUP_STEPS,
    value_bomb_mask_steps: int = VALUE_BOMB_MASK_STEPS,
    eval_mode: bool = False,
):
    canonical_obs = canonicalize_obs(obs, agent_id)
    bomb_state = build_bomb_state(canonical_obs)
    map_feat = encode_obs(canonical_obs, agent_id, bomb_state)
    aux_feat = encode_aux(canonical_obs, agent_id, bomb_state)
    action_mask = valid_action_mask(
        canonical_obs,
        agent_id,
        bomb_state=bomb_state,
        current_step=current_step,
        warmup_steps=warmup_steps,
        value_bomb_mask_steps=value_bomb_mask_steps,
        eval_mode=eval_mode,
    )
    return canonical_obs, map_feat, aux_feat, action_mask


class CNNActorCriticV2(nn.Module):
    """
    Shared-trunk baseline used only for compatibility with V3-style wrappers.
    """

    def __init__(self, num_actions: int = NUM_ACTIONS):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
        )

        cnn_out = 128 * 3 * 3
        fc_in = cnn_out + AUX_DIM
        self.shared_fc = nn.Sequential(
            nn.Linear(fc_in, 384),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(256, num_actions)
        self.critic_head = nn.Linear(256, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward(self, map_feat: torch.Tensor, aux_feat: torch.Tensor):
        cnn_out = self.cnn(map_feat)
        x = torch.cat([cnn_out, aux_feat], dim=1)
        x = self.shared_fc(x)
        logits = self.actor_head(x)
        value = self.critic_head(x)
        return logits, value

    def get_action_and_value(self, map_feat, aux_feat, action=None, action_mask=None):
        from torch.distributions import Categorical

        logits, value = self.forward(map_feat, aux_feat)
        logits = masked_logits(logits, action_mask)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    @torch.no_grad()
    def get_action_inference(self, map_feat, aux_feat, deterministic=False, action_mask=None):
        from torch.distributions import Categorical

        logits, _ = self.forward(map_feat, aux_feat)
        logits = masked_logits(logits, action_mask)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = Categorical(logits=logits).sample()
        return action.item() if action.numel() == 1 else action


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
    "CNNActorCriticV2",
    "GRID_SIZE",
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
