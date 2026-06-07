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
_BASE_PATH = _HERE / "_model_base_v5_1.py"
_SPEC = importlib.util.spec_from_file_location("_v6_1_model_base_v5_1", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BASE)

GRID_SIZE = _BASE.GRID_SIZE
BOMB_TIMER_MAX = _BASE.BOMB_TIMER_MAX
NUM_ACTIONS = _BASE.NUM_ACTIONS
NUM_CHANNELS = _BASE.NUM_CHANNELS
BASE_AUX_DIM = 18
ITEM_AUX_DIM = 17
AUX_DIM = BASE_AUX_DIM + ITEM_AUX_DIM
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

ITEM_RADIUS_TILE = 3
ITEM_CAPACITY_TILE = 4


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
    if action_mask is None:
        return logits

    if action_mask.dim() == 1:
        action_mask = action_mask.unsqueeze(0)
    action_mask = action_mask.to(device=logits.device, dtype=torch.bool)

    if action_mask.shape != logits.shape:
        try:
            action_mask = action_mask.expand_as(logits)
        except RuntimeError as exc:
            raise ValueError(
                f"action_mask shape {tuple(action_mask.shape)} is incompatible with logits shape {tuple(logits.shape)}"
            ) from exc

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_mask = action_mask.reshape(-1, action_mask.shape[-1])

    invalid_rows = ~flat_mask.any(dim=-1)
    if invalid_rows.any():
        flat_mask = flat_mask.clone()
        flat_mask[invalid_rows, ACTION_STOP] = True

    masked = flat_logits.masked_fill(~flat_mask, -1e9)
    return masked.view_as(logits)


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


def _safe_norm_dist(dist: int | None, max_dist: int = 20) -> float:
    if dist is None or dist >= 999:
        return 1.0
    return min(float(dist) / float(max_dist), 1.0)


def _one_hot_action(action: int | None, num_actions: int = NUM_ACTIONS) -> np.ndarray:
    out = np.zeros(num_actions, dtype=np.float32)
    if action is not None and 0 <= int(action) < num_actions:
        out[int(action)] = 1.0
    return out


def _item_type_one_hot(tile_value: int | None) -> np.ndarray:
    out = np.zeros(3, dtype=np.float32)
    if tile_value == ITEM_CAPACITY_TILE:
        out[1] = 1.0
    elif tile_value == ITEM_RADIUS_TILE:
        out[2] = 1.0
    else:
        out[0] = 1.0
    return out


def _active_bombs_owned(obs: dict, agent_id: int) -> int:
    bombs = np.asarray(obs["bombs"], dtype=np.int64)
    if bombs.size == 0:
        return 0
    if bombs.ndim == 1:
        bombs = bombs.reshape(1, -1)
    return int(np.sum(bombs[:, 3] == agent_id))


def _inferred_bomb_capacity(obs: dict, agent_id: int) -> int:
    players = np.asarray(obs["players"], dtype=np.int64)
    bombs_left = int(players[agent_id][3])
    return bombs_left + _active_bombs_owned(obs, agent_id)


def _alive_agent_ids(obs: dict) -> list[int]:
    players = np.asarray(obs["players"], dtype=np.int64)
    return [idx for idx in range(len(players)) if int(players[idx][2]) == 1]


def _nearest_safe_item_info(obs: dict, agent_id: int):
    grid = np.asarray(obs["map"], dtype=np.int64)
    targets = []
    for row in range(1, grid.shape[0] - 1):
        for col in range(1, grid.shape[1] - 1):
            tile = int(grid[row, col])
            if tile in (ITEM_RADIUS_TILE, ITEM_CAPACITY_TILE):
                targets.append((row, col, tile))

    if not targets:
        return None, None, None, 0.0

    target_set = {(r, c) for r, c, _tile in targets}
    first_action, dist = bfs_first_action_to_targets(
        obs,
        agent_id,
        target_set,
        avoid_danger_leq=2,
    )

    if dist >= 999:
        return None, None, None, 0.0

    players = np.asarray(obs["players"], dtype=np.int64)
    my_pos = (int(players[agent_id][0]), int(players[agent_id][1]))

    best_tile = None
    best_manhattan = 999
    best_pos = None
    for row, col, tile in targets:
        md = abs(row - my_pos[0]) + abs(col - my_pos[1])
        if md < best_manhattan:
            best_manhattan = md
            best_tile = tile
            best_pos = (row, col)

    contested = 0.0
    if best_pos is not None:
        enemy_min = 999
        for enemy_id in _alive_agent_ids(obs):
            if enemy_id == agent_id:
                continue
            er = int(players[enemy_id][0])
            ec = int(players[enemy_id][1])
            enemy_min = min(enemy_min, abs(er - best_pos[0]) + abs(ec - best_pos[1]))
        contested = 1.0 if enemy_min <= best_manhattan + 1 else 0.0

    return int(first_action) if first_action is not None else None, int(dist), best_tile, contested


def encode_item_aux(obs: dict, agent_id: int, bomb_state=None) -> np.ndarray:
    first_action, dist, item_tile, contested = _nearest_safe_item_info(obs, agent_id)

    action_oh = _one_hot_action(first_action, NUM_ACTIONS)
    dist_feat = np.array([_safe_norm_dist(dist, 20)], dtype=np.float32)
    type_oh = _item_type_one_hot(item_tile)

    players = np.asarray(obs["players"], dtype=np.int64)
    my_capacity = _inferred_bomb_capacity(obs, agent_id)
    my_radius = 1 + int(players[agent_id][4])

    enemy_caps = []
    enemy_radii = []
    for enemy_id in _alive_agent_ids(obs):
        if enemy_id == agent_id:
            continue
        enemy_caps.append(_inferred_bomb_capacity(obs, enemy_id))
        enemy_radii.append(1 + int(players[enemy_id][4]))

    enemy_max_capacity = max(enemy_caps) if enemy_caps else 0
    enemy_max_radius = max(enemy_radii) if enemy_radii else 1

    resource_feats = np.array(
        [
            min(my_capacity / 8.0, 1.0),
            min(my_radius / 8.0, 1.0),
            min(enemy_max_capacity / 8.0, 1.0),
            min(enemy_max_radius / 8.0, 1.0),
            np.clip((my_capacity - enemy_max_capacity) / 8.0, -1.0, 1.0),
            np.clip((my_radius - enemy_max_radius) / 8.0, -1.0, 1.0),
            float(contested),
        ],
        dtype=np.float32,
    )

    return np.concatenate([action_oh, dist_feat, type_oh, resource_feats], axis=0)


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
    item_aux = encode_item_aux(obs, agent_id, bomb_state)
    aux = np.concatenate([np.asarray(aux, dtype=np.float32), item_aux], axis=0).astype(np.float32)
    return torch.from_numpy(aux)


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
