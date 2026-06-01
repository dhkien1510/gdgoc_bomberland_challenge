"""
Shared model and observation helpers for Bomberland PPO.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn

NUM_CHANNELS = 9
GRID_SIZE = 13
NUM_ACTIONS = 6
BOMB_TIMER_MAX = 7.0
SAFE_BOMB_HORIZON = 7

ACTION_STOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DOWN = 4
ACTION_PLACE_BOMB = 5

# Match the engine's action IDs exactly. The names are unconventional there:
# LEFT/RIGHT change the row axis, UP/DOWN change the column axis.
ACTION_DELTAS = {
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
}


def _iter_bombs(obs: dict):
    bombs = np.asarray(obs["bombs"], dtype=np.int64)
    if bombs.ndim != 2 or bombs.shape[0] == 0:
        return []
    return bombs


def _bomb_radius(players: np.ndarray, owner: int) -> int:
    if 0 <= owner < len(players):
        return 1 + int(players[owner][4])
    return 1


def _compute_explosion_tiles(grid: np.ndarray, row: int, col: int, radius: int):
    tiles = {(row, col)}
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        for step in range(1, radius + 1):
            nr, nc = row + dr * step, col + dc * step
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                break
            cell = int(grid[nr, nc])
            if cell == 1:
                break
            tiles.add((nr, nc))
            if cell == 2:
                break
    return tiles


def _build_bomb_state(obs: dict, extra_bomb: dict | None = None):
    grid = np.asarray(obs["map"], dtype=np.int64)
    players = np.asarray(obs["players"], dtype=np.int64)

    bombs = []
    for bomb in _iter_bombs(obs):
        row, col, timer, owner = (int(v) for v in bomb[:4])
        bombs.append(
            {
                "row": row,
                "col": col,
                "timer": max(1, min(timer, int(BOMB_TIMER_MAX))),
                "owner": owner,
                "radius": _bomb_radius(players, owner),
                "is_virtual": False,
            }
        )

    if extra_bomb is not None:
        bombs.append(
            {
                "row": int(extra_bomb["row"]),
                "col": int(extra_bomb["col"]),
                "timer": int(extra_bomb.get("timer", BOMB_TIMER_MAX)),
                "owner": int(extra_bomb["owner"]),
                "radius": int(extra_bomb["radius"]),
                "is_virtual": True,
            }
        )

    for bomb in bombs:
        bomb["blast_tiles"] = _compute_explosion_tiles(
            grid,
            bomb["row"],
            bomb["col"],
            bomb["radius"],
        )
        bomb["explode_at"] = bomb["timer"]

    changed = True
    while changed:
        changed = False
        for source in bombs:
            source_time = source["explode_at"]
            for target in bombs:
                if source is target:
                    continue
                if (target["row"], target["col"]) in source["blast_tiles"] and target["explode_at"] > source_time:
                    target["explode_at"] = source_time
                    changed = True

    earliest_danger = {}
    bomb_positions = {}
    virtual_bomb = None

    for bomb in bombs:
        position = (bomb["row"], bomb["col"])
        explode_at = bomb["explode_at"]
        bomb_positions[position] = min(bomb_positions.get(position, explode_at), explode_at)
        for tile in bomb["blast_tiles"]:
            earliest_danger[tile] = min(earliest_danger.get(tile, explode_at), explode_at)
        if bomb["is_virtual"]:
            virtual_bomb = bomb

    return {
        "bombs": bombs,
        "earliest_danger": earliest_danger,
        "bomb_positions": bomb_positions,
        "virtual_bomb": virtual_bomb,
    }


def _walkable_tile(grid: np.ndarray, row: int, col: int) -> bool:
    return (
        0 < row < GRID_SIZE - 1
        and 0 < col < GRID_SIZE - 1
        and int(grid[row, col]) not in (1, 2)
    )


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
    bomb_state = _build_bomb_state(obs, extra_bomb=virtual_bomb)
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

    # PLACE_BOMB consumes the current action, so the agent must survive step 1
    # while still on the current tile before any movement is possible.
    if not safe_at_time((row, col), 1):
        return False

    queue = deque([(1, row, col)])
    visited = {(1, row, col)}

    while queue:
        step, cur_row, cur_col = queue.popleft()

        if (
            step < virtual_explode_at
            and (cur_row, cur_col) not in virtual_blast_tiles
            and safe_at_time((cur_row, cur_col), step)
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


def encode_obs(obs: dict, agent_id: int) -> torch.Tensor:
    """
    Encode the full observation into a (9, 13, 13) float tensor.
    """
    grid = np.asarray(obs["map"], dtype=np.float32)
    players = np.asarray(obs["players"], dtype=np.float32)
    bomb_state = _build_bomb_state(obs)

    channels = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    channels[0] = (grid == 0).astype(np.float32)
    channels[1] = (grid == 1).astype(np.float32)
    channels[2] = (grid == 2).astype(np.float32)
    channels[3] = (grid == 3).astype(np.float32)
    channels[4] = (grid == 4).astype(np.float32)

    for i, player in enumerate(players):
        row, col, alive = int(player[0]), int(player[1]), int(player[2])
        if alive and 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            channels[5 if i == agent_id else 6, row, col] = 1.0

    for bomb in bomb_state["bombs"]:
        row, col = bomb["row"], bomb["col"]
        timer_norm = float(bomb["timer"]) / BOMB_TIMER_MAX
        channels[7, row, col] = max(channels[7, row, col], timer_norm)

    for (row, col), explode_at in bomb_state["earliest_danger"].items():
        urgency = (BOMB_TIMER_MAX + 1.0 - min(float(explode_at), BOMB_TIMER_MAX)) / BOMB_TIMER_MAX
        channels[8, row, col] = max(channels[8, row, col], urgency)

    return torch.from_numpy(channels)


def encode_aux(obs: dict, agent_id: int) -> torch.Tensor:
    """
    Scalar features: [bombs_left, radius_bonus].
    """
    players = np.asarray(obs["players"], dtype=np.float32)
    player = players[agent_id]
    bombs_left = float(player[3]) / 5.0
    radius_bonus = float(player[4]) / 4.0
    return torch.tensor([bombs_left, radius_bonus], dtype=torch.float32)


def valid_action_mask(obs: dict, agent_id: int) -> torch.Tensor:
    """
    Match the engine's validity rules and apply a safety gate for bomb placement.
    """
    players = np.asarray(obs["players"], dtype=np.int64)
    grid = np.asarray(obs["map"], dtype=np.int64)
    bomb_state = _build_bomb_state(obs)

    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)

    if agent_id < 0 or agent_id >= len(players):
        return torch.from_numpy(mask)

    row, col, alive, bombs_left, _radius_bonus = (int(v) for v in players[agent_id][:5])
    if alive == 0:
        return torch.from_numpy(mask)

    def safe_at_time(pos: tuple[int, int], step: int) -> bool:
        danger_time = bomb_state["earliest_danger"].get(pos)
        return danger_time is None or danger_time > step

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

    if bombs_left > 0 and (row, col) not in bomb_tiles and has_escape_after_placing_bomb(obs, agent_id):
        mask[ACTION_PLACE_BOMB] = True

    if not mask.any():
        mask[ACTION_STOP] = True

    return torch.from_numpy(mask)


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
    if action_mask is None:
        return logits

    if action_mask.dim() == 1:
        action_mask = action_mask.unsqueeze(0)
    action_mask = action_mask.to(device=logits.device, dtype=torch.bool)

    invalid_rows = ~action_mask.any(dim=-1)
    if invalid_rows.any():
        action_mask = action_mask.clone()
        action_mask[invalid_rows, ACTION_STOP] = True

    return logits.masked_fill(~action_mask, -1e9)


class CNNActorCritic(nn.Module):
    """
    Shared CNN backbone with actor and critic heads.
    """

    def __init__(self, num_actions: int = NUM_ACTIONS):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_out = 64 * GRID_SIZE * GRID_SIZE
        fc_in = cnn_out + 2

        self.shared_fc = nn.Sequential(
            nn.Linear(fc_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
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
