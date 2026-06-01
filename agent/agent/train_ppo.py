"""
PPO self-play training for Bomberland.
"""

from __future__ import annotations

import copy
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import (
    BoxFarmerAgent,
    GeniusRuleAgent,
    RandomAgent,
    SimpleRuleAgent,
    SmarterRuleAgent,
    TacticalRuleAgent,
)
from engine.game import BomberEnv
from model import CNNActorCritic, encode_aux, encode_obs, valid_action_mask

CFG = {
    "lr": 2.5e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "ppo_epochs": 4,
    "batch_size": 256,
    "n_steps": 2048,
    "total_steps": 10_000_000,
    "selfplay_prob": 0.5,
    "pool_size": 10,
    "save_every": 200_000,
    "ckpt_dir": "checkpoints",
    "r_kill": 10.0,
    "r_survive_step": 0.0001,
    "r_box_destroy": 1.0,
    "r_item_collect": 2.0,
    "r_death": -5.0,
    "rank_rewards": {
        0: 20.0,
        1: 5.0,
        2: -5.0,
        3: -20.0,
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def clone_obs(obs: dict) -> dict:
    return {
        "map": np.array(obs["map"], copy=True),
        "players": np.array(obs["players"], copy=True),
        "bombs": np.array(obs["bombs"], copy=True),
    }


def clone_stats(player) -> dict:
    return dict(player.stats)


def sync_linear_lr(optimizer: optim.Optimizer, global_step: int):
    frac = max(1.0 - (global_step / CFG["total_steps"]), 0.01)
    for group in optimizer.param_groups:
        group["lr"] = CFG["lr"] * frac


def record_deaths(players, alive_mask, death_order):
    deaths = []
    for i, player in enumerate(players):
        if alive_mask[i] and not player.alive:
            alive_mask[i] = False
            deaths.append(i)
    if deaths:
        death_order.append(deaths)


def compute_competition_ranks(players, death_order, alive_mask):
    groups = [list(group) for group in death_order]

    alive_players = []
    for i, player in enumerate(players):
        if alive_mask[i] and player.alive:
            alive_players.append(i)

    if alive_players:
        def get_stats(player_id):
            stats = players[player_id].stats
            return (
                stats["kills"],
                stats["boxes"],
                stats["items"],
                stats["bombs"],
            )

        alive_players.sort(key=get_stats, reverse=True)
        tied_groups = []
        current_group = [alive_players[0]]
        current_stats = get_stats(alive_players[0])

        for player_id in alive_players[1:]:
            stats = get_stats(player_id)
            if stats == current_stats:
                current_group.append(player_id)
            else:
                tied_groups.append(current_group)
                current_group = [player_id]
                current_stats = stats
        tied_groups.append(current_group)

        groups.extend(reversed(tied_groups))

    ranks = [0] * len(players)
    for rank, group in enumerate(reversed(groups)):
        for player_id in group:
            ranks[player_id] = rank
    return ranks


def compute_reward(prev_obs, obs, prev_stats, curr_stats, agent_id, done, rank):
    reward = 0.0

    was_alive = bool(prev_obs["players"][agent_id][2])
    is_alive = bool(obs["players"][agent_id][2])

    if is_alive:
        reward += CFG["r_survive_step"]
    if was_alive and not is_alive:
        reward += CFG["r_death"]

    reward += (curr_stats["kills"] - prev_stats["kills"]) * CFG["r_kill"]
    reward += (curr_stats["boxes"] - prev_stats["boxes"]) * CFG["r_box_destroy"]
    reward += (curr_stats["items"] - prev_stats["items"]) * CFG["r_item_collect"]

    if done and rank is not None:
        reward += CFG["rank_rewards"].get(rank, 0.0)

    return reward


class OpponentPool:
    """
    Keep old checkpoints plus rule-based baselines for curriculum self-play.
    """

    def __init__(self, model_factory, pool_size=10):
        self.pool_size = pool_size
        self.model_factory = model_factory
        self.checkpoints = deque(maxlen=pool_size)
        self.easy_bots = [RandomAgent, SimpleRuleAgent]
        self.medium_bots = [SimpleRuleAgent, SmarterRuleAgent, BoxFarmerAgent]
        self.hard_bots = [SmarterRuleAgent, GeniusRuleAgent, TacticalRuleAgent]

    def add_checkpoint(self, state_dict):
        self.checkpoints.append(copy.deepcopy(state_dict))

    def get_opponent(self, agent_id: int, current_step: int = 0):
        use_selfplay = (
            len(self.checkpoints) > 0
            and random.random() < CFG["selfplay_prob"]
        )
        if use_selfplay:
            checkpoint = random.choice(self.checkpoints)
            model = self.model_factory()
            model.load_state_dict(checkpoint)
            model.to(DEVICE)
            model.eval()
            return ModelOpponent(model, agent_id)

        if current_step < 3_000_000:
            cls = random.choice(self.easy_bots)
        elif current_step < 6_000_000:
            cls = random.choice(self.medium_bots)
        else:
            cls = random.choice(self.hard_bots)
        return cls(agent_id)


class ModelOpponent:
    def __init__(self, model, agent_id):
        self.model = model
        self.agent_id = agent_id

    def act(self, obs):
        map_feat = encode_obs(obs, self.agent_id).unsqueeze(0).to(DEVICE)
        aux_feat = encode_aux(obs, self.agent_id).unsqueeze(0).to(DEVICE)
        action_mask = valid_action_mask(obs, self.agent_id).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return self.model.get_action_inference(
                map_feat,
                aux_feat,
                deterministic=False,
                action_mask=action_mask,
            )


class RolloutBuffer:
    def __init__(self, n_steps, device):
        self.n_steps = n_steps
        self.device = device
        self.reset()

    def reset(self):
        self.map_feats = []
        self.aux_feats = []
        self.action_masks = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def push(self, map_feat, aux_feat, action_mask, action, log_prob, reward, value, done):
        self.map_feats.append(map_feat)
        self.aux_feats.append(aux_feat)
        self.action_masks.append(action_mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value, gamma, gae_lambda):
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def get_tensors(self):
        return (
            torch.stack(self.map_feats).to(self.device),
            torch.stack(self.aux_feats).to(self.device),
            torch.stack(self.action_masks).to(self.device),
            torch.tensor(self.actions, dtype=torch.long, device=self.device),
            torch.tensor(self.log_probs, dtype=torch.float32, device=self.device),
        )


def ppo_update(model, optimizer, buffer, last_value):
    advantages, returns = buffer.compute_gae(
        last_value,
        CFG["gamma"],
        CFG["gae_lambda"],
    )

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    map_feats, aux_feats, action_masks, actions, old_log_probs = buffer.get_tensors()
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

    n = len(buffer.actions)
    indices = np.arange(n)

    total_pg_loss = 0.0
    total_val_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    for _ in range(CFG["ppo_epochs"]):
        np.random.shuffle(indices)

        for start in range(0, n, CFG["batch_size"]):
            end = min(start + CFG["batch_size"], n)
            idx = indices[start:end]

            mb_map = map_feats[idx]
            mb_aux = aux_feats[idx]
            mb_mask = action_masks[idx]
            mb_act = actions[idx]
            mb_adv = advantages_t[idx]
            mb_ret = returns_t[idx]
            mb_old_lp = old_log_probs[idx]

            _, new_log_probs, entropy, new_values = model.get_action_and_value(
                mb_map,
                mb_aux,
                mb_act,
                action_mask=mb_mask,
            )
            new_values = new_values.squeeze(-1)

            ratio = torch.exp(new_log_probs - mb_old_lp)
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CFG["clip_eps"], 1 + CFG["clip_eps"])
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            val_loss = 0.5 * ((new_values - mb_ret) ** 2).mean()

            loss = (
                pg_loss
                + CFG["value_coef"] * val_loss
                - CFG["entropy_coef"] * entropy.mean()
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
            optimizer.step()

            total_pg_loss += pg_loss.item()
            total_val_loss += val_loss.item()
            total_entropy += entropy.mean().item()
            num_updates += 1

    return {
        "pg_loss": total_pg_loss / max(num_updates, 1),
        "val_loss": total_val_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
    }


def start_episode(env, pool, global_step):
    obs = env.reset()
    agent_id = random.randint(0, 3)
    opponents = {
        player_id: pool.get_opponent(player_id, global_step)
        for player_id in range(4)
        if player_id != agent_id
    }
    alive_mask = [bool(player.alive) for player in env.players]
    death_order = []
    return obs, agent_id, opponents, alive_mask, death_order


def train():
    os.makedirs(CFG["ckpt_dir"], exist_ok=True)

    model = CNNActorCritic(num_actions=6).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"], eps=1e-5)

    pool = OpponentPool(model_factory=lambda: CNNActorCritic(num_actions=6))
    buffer = RolloutBuffer(CFG["n_steps"], DEVICE)
    env = BomberEnv(max_steps=500)

    global_step = 0
    episode = 0
    best_win_rate = 0.0
    last_save_step = 0
    win_history = deque(maxlen=100)

    obs, agent_id, opponents, alive_mask, death_order = start_episode(env, pool, global_step)

    print(f"\n{'=' * 60}")
    print("Training PPO Agent - Bomberland Self-Play")
    print(f"Device: {DEVICE} | Total steps: {CFG['total_steps']:,}")
    print(f"{'=' * 60}\n")

    t_start = time.time()

    while global_step < CFG["total_steps"]:
        buffer.reset()
        model.eval()

        for _ in range(CFG["n_steps"]):
            global_step += 1

            map_feat = encode_obs(obs, agent_id).unsqueeze(0).to(DEVICE)
            aux_feat = encode_aux(obs, agent_id).unsqueeze(0).to(DEVICE)
            action_mask = valid_action_mask(obs, agent_id).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(
                    map_feat,
                    aux_feat,
                    action_mask=action_mask,
                )

            actions = [0] * 4
            actions[agent_id] = int(action.item())
            for opponent_id, opponent in opponents.items():
                try:
                    actions[opponent_id] = int(opponent.act(obs))
                except Exception:
                    actions[opponent_id] = 0

            prev_obs = clone_obs(obs)
            prev_stats = clone_stats(env.players[agent_id])

            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated

            record_deaths(env.players, alive_mask, death_order)

            agent_rank = None
            if done:
                ranks = compute_competition_ranks(env.players, death_order, alive_mask)
                agent_rank = ranks[agent_id]

            curr_stats = clone_stats(env.players[agent_id])
            reward = compute_reward(
                prev_obs,
                obs,
                prev_stats,
                curr_stats,
                agent_id,
                done,
                agent_rank,
            )

            buffer.push(
                map_feat.squeeze(0).cpu(),
                aux_feat.squeeze(0).cpu(),
                action_mask.squeeze(0).cpu(),
                action.item(),
                log_prob.item(),
                reward,
                value.item(),
                float(done),
            )

            if done:
                episode += 1
                win_history.append(1.0 if agent_rank == 0 else 0.0)
                obs, agent_id, opponents, alive_mask, death_order = start_episode(env, pool, global_step)

        model.train()
        sync_linear_lr(optimizer, global_step)

        with torch.no_grad():
            map_feat_last = encode_obs(obs, agent_id).unsqueeze(0).to(DEVICE)
            aux_feat_last = encode_aux(obs, agent_id).unsqueeze(0).to(DEVICE)
            action_mask_last = valid_action_mask(obs, agent_id).unsqueeze(0).to(DEVICE)
            _, _, _, last_val = model.get_action_and_value(
                map_feat_last,
                aux_feat_last,
                action_mask=action_mask_last,
            )
            last_value = last_val.item()

        stats = ppo_update(model, optimizer, buffer, last_value)

        win_rate = np.mean(win_history) if win_history else 0.0
        elapsed = time.time() - t_start
        sps = global_step / max(elapsed, 1.0)

        print(
            f"Step {global_step:>8,} | Ep {episode:>5} | "
            f"WinRate {win_rate:.2%} | "
            f"PG {stats['pg_loss']:+.4f} | "
            f"Val {stats['val_loss']:.4f} | "
            f"Ent {stats['entropy']:.3f} | "
            f"SPS {sps:.0f}"
        )

        if global_step - last_save_step >= CFG["save_every"]:
            last_save_step = global_step
            ckpt_path = Path(CFG["ckpt_dir"]) / f"model_step{global_step}.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "win_rate": win_rate,
                },
                ckpt_path,
            )
            print(f"  -> Saved checkpoint: {ckpt_path}")

            pool.add_checkpoint(model.state_dict())

            if win_rate >= best_win_rate:
                best_win_rate = win_rate
                torch.save(model.state_dict(), _HERE / "model.pth")
                print(f"  -> New best model! Win rate: {win_rate:.2%}")

    torch.save(model.state_dict(), _HERE / "model.pth")
    print("\nTraining done! Final model saved to model.pth")


if __name__ == "__main__":
    train()
