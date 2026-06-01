"""
Submission agent entrypoint.

Keep `agent.py`, `model.py`, and `model.pth` in the same folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from model import CNNActorCritic, encode_aux, encode_obs, valid_action_mask


class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CNNActorCritic(num_actions=6)
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
        dummy_map = torch.zeros(1, 9, 13, 13, device=self.device)
        dummy_aux = torch.zeros(1, 2, device=self.device)
        with torch.no_grad():
            self.model(dummy_map, dummy_aux)

    def act(self, obs: dict) -> int:
        map_feat = encode_obs(obs, self.agent_id).unsqueeze(0).to(self.device)
        aux_feat = encode_aux(obs, self.agent_id).unsqueeze(0).to(self.device)
        action_mask = valid_action_mask(obs, self.agent_id).unsqueeze(0).to(self.device)

        action = self.model.get_action_inference(
            map_feat,
            aux_feat,
            deterministic=True,
            action_mask=action_mask,
        )
        return int(action)
