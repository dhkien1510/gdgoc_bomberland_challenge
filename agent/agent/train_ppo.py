"""
train_ppo.py — PPO Self-Play cho Bomberland
Chạy trên Kaggle GPU (T4). Sau khi train xong, download model.pth về.

Cấu trúc thư mục cần có:
    Bomberland-GDGoC-AI-Challenge/
    ├── engine/
    ├── agent/          (các baseline agents)
    ├── agent/model.py  (copy file model.py vào đây)
    └── train_ppo.py    (file này)

Cách chạy:
    python train_ppo.py
"""

import os, sys, random, copy, time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ── Setup path ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent            # agent/agent/
ROOT  = _HERE.parent.parent                        # Bomberland-GDGoC-AI-Challenge/
sys.path.insert(0, str(_HERE))                     # để import model.py cùng thư mục
sys.path.insert(0, str(ROOT))                      # ưu tiên cao nhất — để import engine/, agent/ package

from engine.game import BomberEnv
from agent import SimpleRuleAgent, SmarterRuleAgent, GeniusRuleAgent, BoxFarmerAgent, TacticalRuleAgent
from model import CNNActorCritic, encode_obs, encode_aux

# ── Hyperparameters ───────────────────────────────────────────────────────────
CFG = {
    # PPO core
    "lr":               2.5e-4,
    "gamma":            0.99,
    "gae_lambda":       0.95,
    "clip_eps":         0.2,
    "value_coef":       0.5,
    "entropy_coef":     0.05,
    "max_grad_norm":    0.5,
    "ppo_epochs":       4,
    "batch_size":       256,

    # Rollout
    "n_steps":          2048,       # steps thu thập mỗi vòng (4 agents × 512)
    "num_envs":         1,          # số env song song (Kaggle dùng 1 để đơn giản)

    # Training duration
    "total_steps":      10_000_000,

    # Self-play pool
    "selfplay_prob":    0.5,        # 50% đấu với version cũ, 50% đấu baseline
    "pool_size":        10,         # giữ 10 checkpoint cũ

    # Checkpoint
    "save_every":       200_000,    # lưu mỗi 200k steps
    "ckpt_dir":         "checkpoints",

    # Reward shaping weights
    "r_kill":           10.0,
    "r_survive_step":   0.0001,
    "r_box_destroy":    1.0,
    "r_item_collect":   2.0,
    "r_death":         -5.0,
    "r_win":            20.0,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Reward Shaping ─────────────────────────────────────────────────────────────

def compute_reward(prev_obs, obs, agent_id, done, is_winner):
    """
    Tính shaped reward từ sự thay đổi giữa 2 obs liên tiếp.
    """
    reward = 0.0
    p_prev = prev_obs["players"]
    p_curr = obs["players"]

    # Còn sống → nhận reward nhỏ mỗi step để khuyến khích sống lâu
    if p_curr[agent_id][2] == 1:
        reward += CFG["r_survive_step"]
    else:
        # Vừa chết
        reward += CFG["r_death"]

    # Đếm kill: agent khác chết so với bước trước
    for i in range(4):
        if i == agent_id:
            continue
        if p_prev[i][2] == 1 and p_curr[i][2] == 0:
            reward += CFG["r_kill"]

    # Box bị phá (map thay đổi từ 2 → khác)
    prev_map = np.array(prev_obs["map"])
    curr_map = np.array(obs["map"])
    boxes_destroyed = int(np.sum((prev_map == 2) & (curr_map != 2)))
    reward += boxes_destroyed * CFG["r_box_destroy"]

    # Thu thập item: radius hoặc capacity tăng
    prev_r = int(p_prev[agent_id][4])
    curr_r = int(p_curr[agent_id][4])
    prev_c = int(p_prev[agent_id][3])
    curr_c = int(p_curr[agent_id][3])
    if curr_r > prev_r or curr_c > prev_c:
        reward += CFG["r_item_collect"]

    # Thắng
    if done and is_winner:
        reward += CFG["r_win"]

    return reward


# ── Opponent Pool (Self-Play) ──────────────────────────────────────────────────

class OpponentPool:
    """
    Giữ một pool các checkpoint cũ và baseline agents.
    Mỗi episode, sample ngẫu nhiên 3 opponents.
    """

    def __init__(self, model_factory, pool_size=10):
        self.pool_size    = pool_size
        self.model_factory = model_factory
        self.checkpoints  = deque(maxlen=pool_size)   # list of state_dicts
        self.baselines    = [
            SimpleRuleAgent, SmarterRuleAgent,
            GeniusRuleAgent, BoxFarmerAgent, TacticalRuleAgent,
        ]

    def add_checkpoint(self, state_dict):
        self.checkpoints.append(copy.deepcopy(state_dict))

    def get_opponent(self, agent_id: int):
        """
        Trả về một opponent agent (rule-based hoặc model-based).
        """
        use_selfplay = (
            len(self.checkpoints) > 0
            and random.random() < CFG["selfplay_prob"]
        )
        if use_selfplay:
            ckpt = random.choice(self.checkpoints)
            model = self.model_factory()
            model.load_state_dict(ckpt)
            model.to(DEVICE)
            model.eval()
            return ModelOpponent(model, agent_id)
        else:
            cls = random.choice(self.baselines)
            return cls(agent_id)


class ModelOpponent:
    """Wrapper để model cũ act như một baseline agent."""
    def __init__(self, model, agent_id):
        self.model    = model
        self.agent_id = agent_id

    def act(self, obs):
        map_feat = encode_obs(obs, self.agent_id).unsqueeze(0).to(DEVICE)
        aux_feat = encode_aux(obs, self.agent_id).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.model.get_action_inference(map_feat, aux_feat, deterministic=False)
        return action


# ── Rollout Buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, n_steps, device):
        self.n_steps = n_steps
        self.device  = device
        self.reset()

    def reset(self):
        self.map_feats  = []
        self.aux_feats  = []
        self.actions    = []
        self.log_probs  = []
        self.rewards    = []
        self.values     = []
        self.dones      = []

    def push(self, map_feat, aux_feat, action, log_prob, reward, value, done):
        self.map_feats.append(map_feat)
        self.aux_feats.append(aux_feat)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value, gamma, gae_lambda):
        """Tính Generalized Advantage Estimation."""
        rewards  = np.array(self.rewards,  dtype=np.float32)
        values   = np.array(self.values,   dtype=np.float32)
        dones    = np.array(self.dones,    dtype=np.float32)

        advantages = np.zeros_like(rewards)
        last_gae   = 0.0

        for t in reversed(range(len(rewards))):
            next_val   = last_value if t == len(rewards) - 1 else values[t + 1]
            next_done  = dones[t]
            delta      = rewards[t] + gamma * next_val * (1 - next_done) - values[t]
            last_gae   = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def get_tensors(self):
        return (
            torch.stack(self.map_feats).to(self.device),
            torch.stack(self.aux_feats).to(self.device),
            torch.tensor(self.actions,  dtype=torch.long,  device=self.device),
            torch.tensor(self.log_probs,dtype=torch.float32,device=self.device),
        )


# ── PPO Update ────────────────────────────────────────────────────────────────

def ppo_update(model, optimizer, buffer, last_value):
    advantages, returns = buffer.compute_gae(
        last_value, CFG["gamma"], CFG["gae_lambda"]
    )

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    map_feats, aux_feats, actions, old_log_probs = buffer.get_tensors()
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
    returns_t    = torch.tensor(returns,    dtype=torch.float32, device=DEVICE)

    n = len(buffer.actions)
    indices = np.arange(n)

    total_pg_loss  = 0.0
    total_val_loss = 0.0
    total_entropy  = 0.0
    num_updates    = 0

    for _ in range(CFG["ppo_epochs"]):
        np.random.shuffle(indices)

        for start in range(0, n, CFG["batch_size"]):
            end = min(start + CFG["batch_size"], n)
            idx = indices[start:end]

            mb_map  = map_feats[idx]
            mb_aux  = aux_feats[idx]
            mb_act  = actions[idx]
            mb_adv  = advantages_t[idx]
            mb_ret  = returns_t[idx]
            mb_old_lp = old_log_probs[idx]

            _, new_log_probs, entropy, new_values = model.get_action_and_value(
                mb_map, mb_aux, mb_act
            )
            new_values = new_values.squeeze(-1)

            # Probability ratio
            ratio = torch.exp(new_log_probs - mb_old_lp)

            # Clipped surrogate loss
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CFG["clip_eps"], 1 + CFG["clip_eps"])
            pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            val_loss = 0.5 * ((new_values - mb_ret) ** 2).mean()

            # Total loss
            loss = (
                pg_loss
                + CFG["value_coef"] * val_loss
                - CFG["entropy_coef"] * entropy.mean()
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
            optimizer.step()

            total_pg_loss  += pg_loss.item()
            total_val_loss += val_loss.item()
            total_entropy  += entropy.mean().item()
            num_updates    += 1

    return {
        "pg_loss":  total_pg_loss  / max(num_updates, 1),
        "val_loss": total_val_loss / max(num_updates, 1),
        "entropy":  total_entropy  / max(num_updates, 1),
    }


# ── Main Training Loop ────────────────────────────────────────────────────────

def train():
    os.makedirs(CFG["ckpt_dir"], exist_ok=True)

    # Model & optimizer
    model     = CNNActorCritic(num_actions=6).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"], eps=1e-5)

    # Learning rate annealing
    def lr_lambda(step):
        frac = 1.0 - step / CFG["total_steps"]
        return max(frac, 0.01)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    # Self-play pool
    pool = OpponentPool(model_factory=lambda: CNNActorCritic(num_actions=6))

    # Buffer
    buffer = RolloutBuffer(CFG["n_steps"], DEVICE)

    # Env
    env = BomberEnv(max_steps=500)

    # State tracking
    global_step  = 0
    episode      = 0
    best_win_rate = 0.0
    win_history  = deque(maxlen=100)

    # Khởi tạo episode
    obs         = env.reset()
    agent_id    = 0   # Ta train agent 0 (góc top-left)
    opponents   = [pool.get_opponent(i) for i in range(1, 4)]
    prev_obs    = None

    print(f"\n{'='*60}")
    print(f"Training PPO Agent — Bomberland Self-Play")
    print(f"Device: {DEVICE} | Total steps: {CFG['total_steps']:,}")
    print(f"{'='*60}\n")

    t_start = time.time()

    while global_step < CFG["total_steps"]:
        buffer.reset()
        model.eval()

        # ── Rollout phase ──────────────────────────────────────
        for _ in range(CFG["n_steps"]):
            global_step += 1

            map_feat = encode_obs(obs, agent_id).unsqueeze(0).to(DEVICE)
            aux_feat = encode_aux(obs, agent_id).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(map_feat, aux_feat)

            # Opponents act
            actions = [0] * 4
            actions[agent_id] = action.item()
            for opp_idx, opp in enumerate(opponents):
                opp_agent_id = opp_idx + 1
                try:
                    actions[opp_agent_id] = opp.act(obs)
                except Exception:
                    actions[opp_agent_id] = 0

            prev_obs = {
                "map":     np.array(obs["map"],     copy=True),
                "players": np.array(obs["players"], copy=True),
                "bombs":   np.array(obs["bombs"],   copy=True),
            }

            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated

            # Xác định winner
            alive = [bool(obs["players"][i][2]) for i in range(4)]
            survivors = [i for i, a in enumerate(alive) if a]
            is_winner = done and len(survivors) == 1 and survivors[0] == agent_id

            reward = compute_reward(prev_obs, obs, agent_id, done, is_winner)

            buffer.push(
                map_feat.squeeze(0).cpu(),
                aux_feat.squeeze(0).cpu(),
                action.item(),
                log_prob.item(),
                reward,
                value.item(),
                float(done),
            )

            if done:
                episode += 1
                win_history.append(1.0 if is_winner else 0.0)

                # Reset episode
                obs       = env.reset()
                opponents = [pool.get_opponent(i) for i in range(1, 4)]
                prev_obs  = None

        # ── Update phase ───────────────────────────────────────
        model.train()

        # Last value cho GAE
        with torch.no_grad():
            map_feat_last = encode_obs(obs, agent_id).unsqueeze(0).to(DEVICE)
            aux_feat_last = encode_aux(obs, agent_id).unsqueeze(0).to(DEVICE)
            _, _, _, last_val = model.get_action_and_value(map_feat_last, aux_feat_last)
            last_value = last_val.item() if obs["players"][agent_id][2] else 0.0

        stats = ppo_update(model, optimizer, buffer, last_value)

        # ── Logging ────────────────────────────────────────────
        win_rate = np.mean(win_history) if win_history else 0.0
        elapsed  = time.time() - t_start
        sps      = global_step / max(elapsed, 1)

        print(
            f"Step {global_step:>8,} | Ep {episode:>5} | "
            f"WinRate {win_rate:.2%} | "
            f"PG {stats['pg_loss']:+.4f} | "
            f"Val {stats['val_loss']:.4f} | "
            f"Ent {stats['entropy']:.3f} | "
            f"SPS {sps:.0f}"
        )

        # ── Save checkpoint ────────────────────────────────────
        if global_step % CFG["save_every"] == 0:
            ckpt_path = Path(CFG["ckpt_dir"]) / f"model_step{global_step}.pth"
            torch.save({
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "global_step": global_step,
                "win_rate":    win_rate,
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path}")

            # Thêm vào self-play pool
            pool.add_checkpoint(model.state_dict())

            # Lưu best model riêng
            if win_rate >= best_win_rate:
                best_win_rate = win_rate
                torch.save(model.state_dict(), "model.pth")
                print(f"  → New best model! Win rate: {win_rate:.2%}")

    # Lưu lần cuối
    torch.save(model.state_dict(), "model.pth")
    print(f"\nTraining done! Final model saved to model.pth")
    print(f"Copy model.pth vào thư mục agent/ rồi zip lại để nộp.")


if __name__ == "__main__":
    train() 