"""
Inference entrypoint for the recurrent BC+PPO agent.
"""

from __future__ import annotations

from collections import deque
import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
sys.modules.pop("model", None)
sys.modules.pop("bc_model", None)
sys.modules.pop("_model_base", None)
sys.modules.pop("_model_v3_base", None)

from bc_model import CNNLSTMBCActor
from model import (
    ACTION_PLACE_BOMB,
    AUX_DIM,
    MASK_WARMUP_STEPS,
    NUM_CHANNELS,
    VALUE_BOMB_MASK_STEPS,
    RecurrentActorCriticV6,
    can_hit_enemy_if_place,
    count_boxes_if_place,
    nearest_valuable_bomb_spot_info,
    prepare_policy_inputs,
    to_env_action,
)


def _load_checkpoint(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_state_dict_expand_input(module: nn.Module, state_dict: dict, label: str):
    own_state = module.state_dict()
    loaded = {}

    for name, param in own_state.items():
        old = state_dict.get(name)
        if old is None:
            loaded[name] = param
            continue

        if tuple(old.shape) == tuple(param.shape):
            loaded[name] = old
            continue

        if (
            old.ndim == 2
            and param.ndim == 2
            and old.shape[0] == param.shape[0]
            and old.shape[1] < param.shape[1]
        ):
            expanded = param.clone()
            expanded[:, : old.shape[1]] = old
            expanded[:, old.shape[1] :] = 0.0
            loaded[name] = expanded
            continue

        loaded[name] = param

    module.load_state_dict(loaded, strict=True)


class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_step = 0
        self.recent_positions = deque(maxlen=12)
        self.recent_actions = deque(maxlen=12)
        self.episode_start = True

        model_candidates = [
            # _HERE / "model.pth",
            _HERE / "checkpoints/model_step1254400.pth",
        ]
        model_path = next((path for path in model_candidates if path.exists()), None)
        bc_path = _HERE / "bc_actor.pth"
        if model_path is not None:
            self.mode = "ppo"
            self.model = RecurrentActorCriticV6()
            checkpoint = _load_checkpoint(model_path, map_location=self.device)
            state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError:
                _load_state_dict_expand_input(self.model, state_dict, "ppo")
            self.state = self.model.get_initial_actor_state(1, self.device)
        else:
            self.mode = "bc"
            self.model = CNNLSTMBCActor()
            if not bc_path.exists():
                candidate_text = ", ".join(str(path) for path in model_candidates)
                raise FileNotFoundError(f"Neither [{candidate_text}] nor {bc_path} exists")
            bc_state = _load_checkpoint(bc_path, map_location=self.device)
            try:
                self.model.load_state_dict(bc_state)
            except RuntimeError:
                _load_state_dict_expand_input(self.model, bc_state, "bc")
            self.state = self.model.get_initial_state(1, self.device)

        self.model.to(self.device)
        self.model.eval()
        self._warmup()

    def _warmup(self):
        dummy_map = torch.zeros(1, NUM_CHANNELS, 13, 13, device=self.device)
        dummy_aux = torch.zeros(1, AUX_DIM, device=self.device)
        with torch.no_grad():
            if self.mode == "ppo":
                self.model.forward_actor_step(dummy_map, dummy_aux)
            else:
                self.model.forward_step(dummy_map, dummy_aux)

    def _is_looping(self) -> bool:
        if len(self.recent_positions) < 8:
            return False
        return len(set(list(self.recent_positions)[-8:])) <= 2

    def act(self, obs: dict) -> int:
        self.current_step += 1
        my = obs["players"][self.agent_id]
        pos = (int(my[0]), int(my[1]))
        self.recent_positions.append(pos)

        canonical_obs, map_feat, aux_feat, action_mask = prepare_policy_inputs(
            obs,
            self.agent_id,
            current_step=VALUE_BOMB_MASK_STEPS,
            warmup_steps=MASK_WARMUP_STEPS,
            value_bomb_mask_steps=VALUE_BOMB_MASK_STEPS,
            eval_mode=False,
        )
        map_feat = map_feat.unsqueeze(0).to(self.device)
        aux_feat = aux_feat.unsqueeze(0).to(self.device)
        action_mask = action_mask.unsqueeze(0).to(self.device)
        episode_start = torch.tensor([self.episode_start], device=self.device)

        with torch.no_grad():
            if self.mode == "ppo":
                canonical_action, self.state = self.model.get_action_inference(
                    map_feat,
                    aux_feat,
                    deterministic=True,
                    action_mask=action_mask,
                    state=self.state,
                    episode_start=episode_start,
                )
            else:
                action, self.state = self.model.act_step(
                    map_feat,
                    aux_feat,
                    action_mask=action_mask,
                    state=self.state,
                    deterministic=True,
                    episode_start=episode_start,
                )
                canonical_action = int(action.item())
        self.episode_start = False

        if self._is_looping():
            if bool(action_mask[0, ACTION_PLACE_BOMB]) and (
                can_hit_enemy_if_place(canonical_obs, self.agent_id)
                or count_boxes_if_place(canonical_obs, self.agent_id) > 0
            ):
                canonical_action = ACTION_PLACE_BOMB
            else:
                first_action, _dist, _count = nearest_valuable_bomb_spot_info(canonical_obs, self.agent_id)
                if first_action is not None and bool(action_mask[0, int(first_action)]):
                    canonical_action = int(first_action)

        self.recent_actions.append(int(canonical_action))
        return int(to_env_action(int(canonical_action), self.agent_id))
