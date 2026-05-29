"""
agent.py — Nộp lên evaluation server.
Cùng thư mục phải có: model.py, model.pth
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Thêm thư mục chứa agent.py vào path để import model.py
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from model import CNNActorCritic, encode_obs, encode_aux


class Agent:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id

        # Chọn device — server dùng CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = CNNActorCritic(num_actions=6)
        ckpt_path  = _HERE / "model.pth"

        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            # Hỗ trợ cả 2 format: raw state_dict hoặc dict có key "model"
            state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            self.model.load_state_dict(state_dict)
            print(f"[Agent {agent_id}] Loaded model from {ckpt_path}")
        else:
            print(f"[Agent {agent_id}] WARNING: model.pth not found, using random weights")

        self.model.to(self.device)
        self.model.eval()

        # Warm-up để tránh timeout lần đầu
        self._warmup()

    def _warmup(self):
        dummy_map = torch.zeros(1, 9, 13, 13, device=self.device)
        dummy_aux = torch.zeros(1, 2,        device=self.device)
        with torch.no_grad():
            self.model(dummy_map, dummy_aux)

    def act(self, obs: dict) -> int:
        map_feat = encode_obs(obs, self.agent_id).unsqueeze(0).to(self.device)
        aux_feat = encode_aux(obs, self.agent_id).unsqueeze(0).to(self.device)

        action = self.model.get_action_inference(
            map_feat, aux_feat,
            deterministic=True,   # Inference: chọn action có xác suất cao nhất
        )
        return int(action)