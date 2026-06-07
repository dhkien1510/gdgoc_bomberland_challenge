"""
Recurrent BC+PPO actor-critic for Bomberland, built on top of v5.2 features.
"""

from __future__ import annotations

import importlib.util
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

_BASE_SPEC = importlib.util.spec_from_file_location("_v6_2_model_v3_base", _HERE / "_model_v3_base.py")
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
BOMB_TIMER_MAX = _BASE.BOMB_TIMER_MAX
GRID_SIZE = _BASE.GRID_SIZE
MASK_WARMUP_STEPS = _BASE.MASK_WARMUP_STEPS
VALUE_BOMB_MASK_STEPS = _BASE.VALUE_BOMB_MASK_STEPS
NUM_ACTIONS = _BASE.NUM_ACTIONS
NUM_CHANNELS = _BASE.NUM_CHANNELS
SAFE_BOMB_HORIZON = _BASE.SAFE_BOMB_HORIZON
bfs_first_action_to_targets = _BASE.bfs_first_action_to_targets
build_bomb_state = _BASE.build_bomb_state
can_hit_enemy_if_place = _BASE.can_hit_enemy_if_place
canonicalize_obs = _BASE.canonicalize_obs
clone_obs_with_player_at = _BASE.clone_obs_with_player_at
count_boxes_if_place = _BASE.count_boxes_if_place
current_tile_danger_time = _BASE.current_tile_danger_time
encode_aux = _BASE.encode_aux
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
from bc_model import CNNLSTMActorCore


def _load_checkpoint(path_or_obj, map_location="cpu"):
    if isinstance(path_or_obj, str):
        try:
            return torch.load(path_or_obj, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path_or_obj, map_location=map_location)
    return path_or_obj


def _make_critic_encoder():
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


class RecurrentActorCriticV6(nn.Module):
    def __init__(self, num_actions: int = NUM_ACTIONS, actor_hidden_size: int = 256):
        super().__init__()
        self.actor_core = CNNLSTMActorCore(hidden_size=actor_hidden_size)
        self.actor_head = nn.Linear(actor_hidden_size, num_actions)

        self.critic_cnn = _make_critic_encoder()
        self.critic_fc = nn.Sequential(
            nn.Linear(128 * 3 * 3 + AUX_DIM, 384),
            nn.ReLU(),
            nn.Linear(384, 256),
            nn.ReLU(),
        )
        self.critic_head = nn.Linear(256, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def get_initial_actor_state(self, batch_size: int, device: torch.device):
        return self.actor_core.get_initial_state(batch_size, device)

    def actor_parameters(self):
        return list(self.actor_core.parameters()) + list(self.actor_head.parameters())

    def critic_parameters(self):
        return (
            list(self.critic_cnn.parameters())
            + list(self.critic_fc.parameters())
            + list(self.critic_head.parameters())
        )

    def set_actor_trainable(self, trainable: bool):
        for param in self.actor_parameters():
            param.requires_grad = trainable

    def forward_actor_sequence(
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

    def forward_actor_step(
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

    def forward_value_sequence(self, map_seq: torch.Tensor, aux_seq: torch.Tensor):
        batch_size, seq_len = map_seq.shape[:2]
        flat_map = map_seq.reshape(batch_size * seq_len, *map_seq.shape[2:])
        flat_aux = aux_seq.reshape(batch_size * seq_len, aux_seq.shape[-1])
        critic_out = self.critic_cnn(flat_map)
        critic_x = torch.cat([critic_out, flat_aux], dim=-1)
        critic_x = self.critic_fc(critic_x)
        values = self.critic_head(critic_x)
        return values.view(batch_size, seq_len)

    def forward_value_step(self, map_feat: torch.Tensor, aux_feat: torch.Tensor):
        critic_out = self.critic_cnn(map_feat)
        critic_x = torch.cat([critic_out, aux_feat], dim=-1)
        critic_x = self.critic_fc(critic_x)
        return self.critic_head(critic_x)

    def load_actor_from_checkpoint(self, checkpoint: dict | str):
        checkpoint = _load_checkpoint(checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        current = self.state_dict()
        remapped = {}
        for key, value in state_dict.items():
            new_key = None
            if key.startswith("actor_core.") or key.startswith("actor_head."):
                new_key = key
            elif key.startswith("model.actor_core."):
                new_key = key[len("model."):]
            elif key.startswith("model.actor_head."):
                new_key = key[len("model."):]
            if new_key is not None and new_key in current and tuple(current[new_key].shape) == tuple(value.shape):
                remapped[new_key] = value
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        loaded_actor = any(k.startswith("actor_core.") for k in remapped) and any(
            k.startswith("actor_head.") for k in remapped
        )
        return {
            "loaded_actor": loaded_actor,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }

    def get_action_and_value_sequence(
        self,
        map_seq: torch.Tensor,
        aux_seq: torch.Tensor,
        actions: torch.Tensor | None = None,
        action_mask_seq: torch.Tensor | None = None,
        episode_start_mask: torch.Tensor | None = None,
        state=None,
    ):
        from torch.distributions import Categorical

        logits, next_state = self.forward_actor_sequence(
            map_seq,
            aux_seq,
            action_mask_seq=action_mask_seq,
            state=state,
            episode_start_mask=episode_start_mask,
        )
        values = self.forward_value_sequence(map_seq, aux_seq)
        dist = Categorical(logits=logits)
        if actions is None:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_probs, entropy, values, next_state

    @torch.no_grad()
    def act_step(
        self,
        map_feat: torch.Tensor,
        aux_feat: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        state=None,
        deterministic: bool = False,
        episode_start: torch.Tensor | None = None,
    ):
        from torch.distributions import Categorical

        logits, next_state = self.forward_actor_step(
            map_feat,
            aux_feat,
            action_mask=action_mask,
            state=state,
            episode_start=episode_start,
        )
        value = self.forward_value_step(map_feat, aux_feat).squeeze(-1)
        if deterministic:
            action = logits.argmax(dim=-1)
            dist = Categorical(logits=logits)
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, next_state

    @torch.no_grad()
    def get_action_inference(
        self,
        map_feat: torch.Tensor,
        aux_feat: torch.Tensor,
        deterministic: bool = True,
        action_mask: torch.Tensor | None = None,
        state=None,
        episode_start: torch.Tensor | None = None,
    ):
        action, _, _, next_state = self.act_step(
            map_feat,
            aux_feat,
            action_mask=action_mask,
            state=state,
            deterministic=deterministic,
            episode_start=episode_start,
        )
        return (action.item() if action.numel() == 1 else action), next_state


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
    "GRID_SIZE",
    "MASK_WARMUP_STEPS",
    "VALUE_BOMB_MASK_STEPS",
    "NUM_ACTIONS",
    "NUM_CHANNELS",
    "SAFE_BOMB_HORIZON",
    "RecurrentActorCriticV6",
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
