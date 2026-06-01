"""
Shared model and observation helpers for Bomberland PPO.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

NUM_CHANNELS = 9
GRID_SIZE = 13
NUM_ACTIONS = 6
BOMB_TIMER_MAX = 7.0

ACTION_STOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_UP = 3
ACTION_DOWN = 4
ACTION_PLACE_BOMB = 5

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


def encode_obs(obs: dict, agent_id: int) -> torch.Tensor:
    """
    Encode the full observation into a (9, 13, 13) float tensor.
    """
    grid = np.asarray(obs["map"], dtype=np.float32)
    players = np.asarray(obs["players"], dtype=np.float32)
    bombs = _iter_bombs(obs)

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

    for bomb in bombs:
        row, col, timer, owner = (int(v) for v in bomb[:4])
        if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
            continue

        timer = max(0, min(timer, int(BOMB_TIMER_MAX)))
        timer_norm = timer / BOMB_TIMER_MAX
        urgency = (BOMB_TIMER_MAX + 1.0 - max(timer, 1)) / BOMB_TIMER_MAX

        channels[7, row, col] = max(channels[7, row, col], timer_norm)
        channels[8, row, col] = max(channels[8, row, col], urgency)

        radius = 1
        if 0 <= owner < len(players):
            radius = 1 + int(players[owner][4])

        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            for step in range(1, radius + 1):
                nr, nc = row + dr * step, col + dc * step
                if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                    break

                cell = int(grid[nr, nc])
                if cell == 1:
                    break

                channels[8, nr, nc] = max(channels[8, nr, nc], urgency)
                if cell == 2:
                    break

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
    Match the engine's validity rules for movement and bomb placement.
    """
    players = np.asarray(obs["players"], dtype=np.int64)
    grid = np.asarray(obs["map"], dtype=np.int64)

    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    mask[ACTION_STOP] = True

    if agent_id < 0 or agent_id >= len(players):
        return torch.from_numpy(mask)

    row, col, alive, bombs_left, _radius_bonus = (int(v) for v in players[agent_id][:5])
    if alive == 0:
        return torch.from_numpy(mask)

    bomb_tiles = {
        (int(bomb[0]), int(bomb[1]))
        for bomb in _iter_bombs(obs)
    }

    for action, (dr, dc) in ACTION_DELTAS.items():
        nr, nc = row + dr, col + dc
        if not (0 < nr < GRID_SIZE - 1 and 0 < nc < GRID_SIZE - 1):
            continue
        if grid[nr, nc] in (1, 2):
            continue
        if (nr, nc) in bomb_tiles:
            continue
        mask[action] = True

    if bombs_left > 0 and (row, col) not in bomb_tiles:
        mask[ACTION_PLACE_BOMB] = True

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
