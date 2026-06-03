"""
Behavior Cloning recurrent actor components for Bomberland.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_BASE_SPEC = importlib.util.spec_from_file_location("_v6_model_v3_base_bc", _HERE / "_model_v3_base.py")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
assert _BASE_SPEC.loader is not None
_BASE_SPEC.loader.exec_module(_BASE)

AUX_DIM = _BASE.AUX_DIM
NUM_ACTIONS = _BASE.NUM_ACTIONS
NUM_CHANNELS = _BASE.NUM_CHANNELS
masked_logits = _BASE.masked_logits


class SpatialEncoderV6(nn.Module):
    """
    Preserve spatial detail longer than the v5.x PPO encoder because BC needs
    cleaner local behavior imitation before PPO fine-tuning.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 13 * 13, 512),
            nn.ReLU(),
        )

    def forward(self, map_feat: torch.Tensor) -> torch.Tensor:
        return self.net(map_feat)


class CNNLSTMActorCore(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.spatial = SpatialEncoderV6()
        self.fuse = nn.Sequential(
            nn.Linear(512 + AUX_DIM, 512),
            nn.ReLU(),
        )
        self.lstm = nn.LSTMCell(512, hidden_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

    def get_initial_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return h, c

    def encode_inputs(self, map_feat: torch.Tensor, aux_feat: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(map_feat)
        return self.fuse(torch.cat([spatial, aux_feat], dim=-1))

    def forward_step(
        self,
        map_feat: torch.Tensor,
        aux_feat: torch.Tensor,
        state=None,
        episode_start: torch.Tensor | None = None,
    ):
        batch_size = map_feat.shape[0]
        if state is None:
            state = self.get_initial_state(batch_size, map_feat.device)

        hx, cx = state
        if episode_start is not None:
            reset = episode_start.to(map_feat.device, dtype=hx.dtype).view(batch_size, 1)
            hx = hx * (1.0 - reset)
            cx = cx * (1.0 - reset)

        fused = self.encode_inputs(map_feat, aux_feat)
        hx, cx = self.lstm(fused, (hx, cx))
        return hx, (hx, cx)

    def forward_sequence(
        self,
        map_seq: torch.Tensor,
        aux_seq: torch.Tensor,
        state=None,
        episode_start_mask: torch.Tensor | None = None,
    ):
        batch_size, seq_len = map_seq.shape[:2]
        if state is None:
            state = self.get_initial_state(batch_size, map_seq.device)

        outputs = []
        hx, cx = state
        for t in range(seq_len):
            step_start = None if episode_start_mask is None else episode_start_mask[:, t]
            out, (hx, cx) = self.forward_step(
                map_seq[:, t],
                aux_seq[:, t],
                state=(hx, cx),
                episode_start=step_start,
            )
            outputs.append(out)
        return torch.stack(outputs, dim=1), (hx, cx)


class CNNLSTMBCActor(nn.Module):
    def __init__(self, num_actions: int = NUM_ACTIONS, hidden_size: int = 256):
        super().__init__()
        self.actor_core = CNNLSTMActorCore(hidden_size=hidden_size)
        self.actor_head = nn.Linear(hidden_size, num_actions)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

    def get_initial_state(self, batch_size: int, device: torch.device):
        return self.actor_core.get_initial_state(batch_size, device)

    def forward_sequence(
        self,
        map_seq: torch.Tensor,
        aux_seq: torch.Tensor,
        action_mask_seq: torch.Tensor | None = None,
        state=None,
        episode_start_mask: torch.Tensor | None = None,
    ):
        actor_out, next_state = self.actor_core.forward_sequence(
            map_seq,
            aux_seq,
            state=state,
            episode_start_mask=episode_start_mask,
        )
        logits = self.actor_head(actor_out)
        if action_mask_seq is not None:
            logits = masked_logits(logits, action_mask_seq)
        return logits, next_state

    def forward_step(
        self,
        map_feat: torch.Tensor,
        aux_feat: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        state=None,
        episode_start: torch.Tensor | None = None,
    ):
        actor_out, next_state = self.actor_core.forward_step(
            map_feat,
            aux_feat,
            state=state,
            episode_start=episode_start,
        )
        logits = self.actor_head(actor_out)
        if action_mask is not None:
            logits = masked_logits(logits, action_mask)
        return logits, next_state

    @torch.no_grad()
    def act_step(
        self,
        map_feat: torch.Tensor,
        aux_feat: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        state=None,
        deterministic: bool = True,
        episode_start: torch.Tensor | None = None,
    ):
        from torch.distributions import Categorical

        logits, next_state = self.forward_step(
            map_feat,
            aux_feat,
            action_mask=action_mask,
            state=state,
            episode_start=episode_start,
        )
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = Categorical(logits=logits).sample()
        return action, next_state


__all__ = [
    "CNNLSTMActorCore",
    "CNNLSTMBCActor",
    "SpatialEncoderV6",
]
