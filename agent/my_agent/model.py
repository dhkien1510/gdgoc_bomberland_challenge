"""
model.py — CNN Actor-Critic cho Bomberland PPO
Dùng chung giữa train_ppo.py và agent.py
"""
import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# Encode observation → tensor (9 channels, 13×13)
# ─────────────────────────────────────────────
# Channel 0  : grass        (map == 0)
# Channel 1  : wall         (map == 1)
# Channel 2  : box          (map == 2)
# Channel 3  : item radius  (map == 3)
# Channel 4  : item capacity(map == 4)
# Channel 5  : vị trí agent mình
# Channel 6  : vị trí các agent khác (còn sống)
# Channel 7  : vị trí bom (giá trị = timer / 7, để normalize)
# Channel 8  : vùng nguy hiểm bom (blast radius estimate)

NUM_CHANNELS = 9
GRID_SIZE    = 13


def encode_obs(obs: dict, agent_id: int) -> torch.Tensor:
    """
    Chuyển obs dict → float tensor shape (9, 13, 13).
    Không cần GPU ở đây, gọi .to(device) bên ngoài.
    """
    grid    = np.array(obs["map"],     dtype=np.float32)   # (13,13)
    players = np.array(obs["players"], dtype=np.float32)   # (4,5)
    bombs   = np.array(obs["bombs"],   dtype=np.float32)   # (N,4)

    c = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # Map channels
    c[0] = (grid == 0).astype(np.float32)
    c[1] = (grid == 1).astype(np.float32)
    c[2] = (grid == 2).astype(np.float32)
    c[3] = (grid == 3).astype(np.float32)
    c[4] = (grid == 4).astype(np.float32)

    # Agent channels
    for i, p in enumerate(players):
        r, col, alive = int(p[0]), int(p[1]), int(p[2])
        if alive and 0 <= r < GRID_SIZE and 0 <= col < GRID_SIZE:
            if i == agent_id:
                c[5, r, col] = 1.0
            else:
                c[6, r, col] = 1.0

    # Bomb channels
    if bombs.ndim == 2 and bombs.shape[0] > 0:
        for b in bombs:
            br, bc, timer, owner = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            if 0 <= br < GRID_SIZE and 0 <= bc < GRID_SIZE:
                # Channel 7: vị trí bom, normalize timer
                c[7, br, bc] = max(timer, 0) / 7.0

                # Channel 8: ước tính blast zone
                radius = 1
                if 0 <= owner < len(players):
                    radius = 1 + int(players[owner][4])
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    for step in range(1, radius + 1):
                        nr, nc = br + dr*step, bc + dc*step
                        if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                            break
                        cell = int(grid[nr, nc])
                        if cell == 1:
                            break
                        c[8, nr, nc] = max(c[8, nr, nc], max(timer, 0) / 7.0)
                        if cell == 2:
                            break

    return torch.from_numpy(c)


def encode_aux(obs: dict, agent_id: int) -> torch.Tensor:
    """
    Scalar features: [bombs_left, radius_bonus] của agent mình — shape (2,).
    Normalize về [0,1].
    """
    players = obs["players"]
    p = players[agent_id]
    bombs_left    = float(p[3]) / 5.0   # max capacity = 5
    radius_bonus  = float(p[4]) / 4.0   # max bonus = 4
    return torch.tensor([bombs_left, radius_bonus], dtype=torch.float32)


# ─────────────────────────────────────────────
# CNN Actor-Critic
# ─────────────────────────────────────────────

class CNNActorCritic(nn.Module):
    """
    Shared CNN backbone → Actor head (policy) + Critic head (value).

    Input:
        map_feat  : (B, 9, 13, 13)
        aux_feat  : (B, 2)   [bombs_left, radius_bonus]

    Output:
        logits    : (B, 6)   raw action logits (dùng Categorical)
        value     : (B, 1)   state value V(s)
    """

    def __init__(self, num_actions: int = 6):
        super().__init__()

        # CNN backbone — 3 conv layers
        self.cnn = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),           # → 64 * 13 * 13 = 10816
        )

        cnn_out = 64 * GRID_SIZE * GRID_SIZE   # 10816
        aux_dim = 2
        fc_in   = cnn_out + aux_dim            # 10818

        # Shared FC
        self.shared_fc = nn.Sequential(
            nn.Linear(fc_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Actor head
        self.actor_head  = nn.Linear(256, num_actions)

        # Critic head
        self.critic_head = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Actor head dùng gain nhỏ hơn để action distribution ban đầu gần uniform
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward(self, map_feat: torch.Tensor, aux_feat: torch.Tensor):
        cnn_out = self.cnn(map_feat)
        x = torch.cat([cnn_out, aux_feat], dim=1)
        x = self.shared_fc(x)
        logits = self.actor_head(x)
        value  = self.critic_head(x)
        return logits, value

    def get_action_and_value(self, map_feat, aux_feat, action=None):
        """
        Dùng trong training loop.
        Trả về (action, log_prob, entropy, value).
        """
        from torch.distributions import Categorical
        logits, value = self.forward(map_feat, aux_feat)
        dist  = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value

    @torch.no_grad()
    def get_action_inference(self, map_feat, aux_feat, deterministic=False):
        """
        Dùng trong act() — không tính gradient, nhanh hơn.
        """
        from torch.distributions import Categorical
        logits, _ = self.forward(map_feat, aux_feat)
        if deterministic:
            return logits.argmax(dim=-1).item()
        dist = Categorical(logits=logits)
        return dist.sample().item()