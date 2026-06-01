"""
V5 actor/critic-separated model for Bomberland PPO.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
sys.modules.pop("_model_base", None)
sys.modules.pop("_model_v3_base", None)

import numpy as np
import torch
import torch.nn as nn

from _model_v3_base import (
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
    NUM_ACTIONS,
    NUM_CHANNELS,
    SAFE_BOMB_HORIZON,
    build_bomb_state,
    canonicalize_obs,
    current_tile_danger_time,
    encode_aux,
    encode_obs,
    has_escape_after_placing_bomb,
    masked_logits,
    prepare_policy_inputs,
    to_canonical_action,
    to_env_action,
    valid_action_mask,
)


def _make_feature_extractor():
    return nn.Sequential(
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


def _make_head_block(input_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, 384),
        nn.ReLU(),
        nn.Linear(384, 256),
        nn.ReLU(),
    )


class CNNActorCriticV5(nn.Module):
    """
    Separate actor and critic backbones so value gradients cannot overwrite
    the policy representation.
    """

    def __init__(self, num_actions: int = NUM_ACTIONS):
        super().__init__()

        self.actor_cnn = _make_feature_extractor()
        self.critic_cnn = _make_feature_extractor()

        trunk_out = 128 * 3 * 3 + AUX_DIM
        self.actor_fc = _make_head_block(trunk_out)
        self.critic_fc = _make_head_block(trunk_out)

        self.actor_head = nn.Linear(256, num_actions)
        self.critic_head = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward_policy(self, map_feat: torch.Tensor, aux_feat: torch.Tensor):
        actor_out = self.actor_cnn(map_feat)
        actor_x = torch.cat([actor_out, aux_feat], dim=1)
        actor_x = self.actor_fc(actor_x)
        return self.actor_head(actor_x)

    def forward_value(self, map_feat: torch.Tensor, aux_feat: torch.Tensor):
        critic_out = self.critic_cnn(map_feat)
        critic_x = torch.cat([critic_out, aux_feat], dim=1)
        critic_x = self.critic_fc(critic_x)
        return self.critic_head(critic_x)

    def forward(self, map_feat: torch.Tensor, aux_feat: torch.Tensor):
        logits = self.forward_policy(map_feat, aux_feat)
        value = self.forward_value(map_feat, aux_feat)
        return logits, value

    def actor_parameters(self):
        return (
            list(self.actor_cnn.parameters())
            + list(self.actor_fc.parameters())
            + list(self.actor_head.parameters())
        )

    def critic_parameters(self):
        return (
            list(self.critic_cnn.parameters())
            + list(self.critic_fc.parameters())
            + list(self.critic_head.parameters())
        )

    def set_actor_trainable(self, trainable: bool):
        for param in self.actor_parameters():
            param.requires_grad = trainable

    def load_actor_from_checkpoint(self, checkpoint: dict | str):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location="cpu")

        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        remapped = {}

        for key, value in state_dict.items():
            if key.startswith("cnn."):
                remapped["actor_cnn." + key[len("cnn."):]] = value
            elif key.startswith("shared_fc."):
                remapped["actor_fc." + key[len("shared_fc."):]] = value
            elif key.startswith("actor_head."):
                remapped[key] = value
            elif key.startswith("actor_cnn.") or key.startswith("actor_fc."):
                remapped[key] = value

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        loaded_actor = any(key.startswith("actor_") or key.startswith("actor_head.") for key in remapped)
        return {
            "loaded_actor": loaded_actor,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }

    # Backward-compatible alias for callers migrated from v4.1.
    load_actor_from_shared_checkpoint = load_actor_from_checkpoint

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

        logits = self.forward_policy(map_feat, aux_feat)
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
    "CNNActorCriticV5",
    "GRID_SIZE",
    "MASK_WARMUP_STEPS",
    "NUM_ACTIONS",
    "NUM_CHANNELS",
    "SAFE_BOMB_HORIZON",
    "build_bomb_state",
    "canonicalize_obs",
    "current_tile_danger_time",
    "encode_aux",
    "encode_obs",
    "has_escape_after_placing_bomb",
    "masked_logits",
    "prepare_policy_inputs",
    "to_canonical_action",
    "to_env_action",
    "valid_action_mask",
]
