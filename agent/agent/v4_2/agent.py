"""
Submission/inference entrypoint for version 4.2.

Keep `agent.py`, `model.py`, and `model.pth` in the same folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
sys.modules.pop("model", None)
sys.modules.pop("_model_base", None)
sys.modules.pop("_model_v3_base", None)

from model import AUX_DIM, MASK_WARMUP_STEPS, NUM_CHANNELS, CNNActorCriticV4_2, prepare_policy_inputs, to_env_action


class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNActorCriticV4_2(num_actions=6)

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

    def act(self, obs: dict) -> int:
        _, map_feat, aux_feat, action_mask = prepare_policy_inputs(
            obs,
            self.agent_id,
            current_step=MASK_WARMUP_STEPS,
            warmup_steps=MASK_WARMUP_STEPS,
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
        return int(to_env_action(canonical_action, self.agent_id))
