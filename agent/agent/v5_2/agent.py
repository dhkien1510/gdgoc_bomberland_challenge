"""
Submission/inference entrypoint for version 5.2.

Keep `agent.py`, `model.py`, and `model.pth` in the same folder.
"""

from __future__ import annotations

from collections import deque
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
sys.modules.pop("model", None)
sys.modules.pop("_model_base", None)
sys.modules.pop("_model_v3_base", None)

from model import (
    ACTION_PLACE_BOMB,
    AUX_DIM,
    MASK_WARMUP_STEPS,
    NUM_CHANNELS,
    VALUE_BOMB_MASK_STEPS,
    CNNActorCriticV5_2,
    can_hit_enemy_if_place,
    canonicalize_obs,
    count_boxes_if_place,
    nearest_valuable_bomb_spot_info,
    prepare_policy_inputs,
    to_env_action,
)


class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNActorCriticV5_2(num_actions=6)
        self.current_step = 0
        self.recent_positions = deque(maxlen=12)
        self.recent_actions = deque(maxlen=12)

        ckpt_path = _HERE / "model.pth"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.model.load_state_dict(state_dict)
            print(f"[Agent {agent_id}] Loaded model from {ckpt_path}")
        else:
            print(f"[Agent {agent_id}] WARNING: model.pth not found, using random weights")

        self.model.to(self.device)
        self.model.eval()
        self._warmup()

    def _warmup(self):
        dummy_map = torch.zeros(1, NUM_CHANNELS, 13, 13, device=self.device)
        dummy_aux = torch.zeros(1, AUX_DIM, device=self.device)
        with torch.no_grad():
            self.model.forward(dummy_map, dummy_aux)

    def _is_looping(self) -> bool:
        if len(self.recent_positions) < 8:
            return False
        recent = list(self.recent_positions)
        return len(set(recent[-8:])) <= 2

    def act(self, obs: dict) -> int:
        self.current_step += 1
        my = obs["players"][self.agent_id]
        pos = (int(my[0]), int(my[1]))
        self.recent_positions.append(pos)

        canonical_obs, map_feat, aux_feat, action_mask = prepare_policy_inputs(
            obs,
            self.agent_id,
            current_step=self.current_step,
            warmup_steps=MASK_WARMUP_STEPS,
            value_bomb_mask_steps=VALUE_BOMB_MASK_STEPS,
            eval_mode=True,
        )

        canonical_action = int(
            self.model.get_action_inference(
                map_feat.unsqueeze(0).to(self.device),
                aux_feat.unsqueeze(0).to(self.device),
                deterministic=True,
                action_mask=action_mask.unsqueeze(0).to(self.device),
            )
        )

        if self._is_looping():
            if bool(action_mask[ACTION_PLACE_BOMB]) and (
                can_hit_enemy_if_place(canonical_obs, self.agent_id)
                or count_boxes_if_place(canonical_obs, self.agent_id) > 0
            ):
                canonical_action = ACTION_PLACE_BOMB
            else:
                first_action, _dist, _count = nearest_valuable_bomb_spot_info(canonical_obs, self.agent_id)
                if first_action is not None and bool(action_mask[int(first_action)]):
                    canonical_action = int(first_action)

        self.recent_actions.append(canonical_action)
        return int(to_env_action(canonical_action, self.agent_id))
