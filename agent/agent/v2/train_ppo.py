"""
Improved PPO self-play training for Bomberland.
"""

from __future__ import annotations

import copy
import math
import os
import random
import sys
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent.parent
BASELINE_DIR = ROOT / "agent"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
sys.modules.pop("model", None)

from model import (
    MASK_WARMUP_STEPS,
    CNNActorCriticV2,
    build_bomb_state,
    current_tile_danger_time,
    prepare_policy_inputs,
    to_env_action,
)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(BASELINE_DIR) not in sys.path:
    sys.path.append(str(BASELINE_DIR))

from box_farmer_agent import BoxFarmerAgent
from genius_rule_agent import GeniusRuleAgent
from random_agent import RandomAgent
from simple_rule_agent import SimpleRuleAgent
from smarter_rule_agent import SmarterRuleAgent
from tactical_rule_agent import TacticalRuleAgent
from engine.game import BomberEnv

CFG = {
    "lr": 2.5e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.015,
    "max_grad_norm": 0.5,
    "ppo_epochs": 4,
    "batch_size": 256,
    "n_steps": 1024,
    "total_steps": 2_000_000,
    "mask_warmup_steps": MASK_WARMUP_STEPS,
    "selfplay_prob": 0.35,
    "pool_size": 8,
    "save_every": 100_000,
    "ckpt_dir": str(_HERE / "checkpoints"),
    "step_penalty": -0.01,
    "r_death": -0.5,
    "r_kill": 1.0,
    "r_box_destroy": 0.1,
    "r_item_collect": 0.1,
    "r_move_closer": 0.002,
    "r_move_away": -0.002,
    "r_best_enemy_dist": 0.1,
    "r_in_danger": -0.000666,
    "r_escape_danger": 0.002,
    "rank_rewards": {
        0: 1.0,
        1: 0.25,
        2: -0.25,
        3: -1.0,
    },
    "reward_clip": 2.0,
    "eval_easy_medium_matches": 30,
    "eval_hard_matches": 30,
    "eval_seed_base": 17_291,
    "holdout_eval_easy_medium_matches": 15,
    "holdout_eval_hard_matches": 15,
    "holdout_eval_seed_base": 91_337,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANK_TO_POINTS = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}
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
    frac = max(1.0 - (global_step / CFG["total_steps"]), 0.05)
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


def nearest_enemy_distance(obs: dict, agent_id: int) -> float:
    players = np.asarray(obs["players"], dtype=np.int64)
    my_row, my_col = int(players[agent_id][0]), int(players[agent_id][1])
    dists = [
        abs(my_row - int(player[0])) + abs(my_col - int(player[1]))
        for i, player in enumerate(players)
        if i != agent_id and int(player[2]) == 1
    ]
    return float(min(dists)) if dists else 0.0


def in_danger(obs: dict, agent_id: int) -> bool:
    bomb_state = build_bomb_state(obs)
    danger_time = current_tile_danger_time(obs, agent_id, bomb_state)
    return danger_time is not None


def make_episode_context(obs: dict, agent_id: int) -> dict:
    return {
        "best_enemy_dist": nearest_enemy_distance(obs, agent_id),
    }


def compute_reward(prev_obs, obs, prev_stats, curr_stats, agent_id, done, rank, episode_ctx):
    reward = CFG["step_penalty"]

    was_alive = bool(prev_obs["players"][agent_id][2])
    is_alive = bool(obs["players"][agent_id][2])

    if was_alive and not is_alive:
        reward += CFG["r_death"]

    reward += (curr_stats["kills"] - prev_stats["kills"]) * CFG["r_kill"]
    reward += (curr_stats["boxes"] - prev_stats["boxes"]) * CFG["r_box_destroy"]
    reward += (curr_stats["items"] - prev_stats["items"]) * CFG["r_item_collect"]

    prev_enemy_dist = nearest_enemy_distance(prev_obs, agent_id)
    curr_enemy_dist = nearest_enemy_distance(obs, agent_id)
    if prev_enemy_dist > 0.0 and curr_enemy_dist > 0.0:
        if curr_enemy_dist < prev_enemy_dist:
            reward += CFG["r_move_closer"]
        elif curr_enemy_dist > prev_enemy_dist:
            reward += CFG["r_move_away"]

    best_enemy_dist = episode_ctx["best_enemy_dist"]
    if curr_enemy_dist > 0.0 and curr_enemy_dist < best_enemy_dist:
        reward += CFG["r_best_enemy_dist"]
        episode_ctx["best_enemy_dist"] = curr_enemy_dist
    else:
        episode_ctx["best_enemy_dist"] = min(best_enemy_dist, curr_enemy_dist) if curr_enemy_dist > 0.0 else best_enemy_dist

    prev_in_danger = in_danger(prev_obs, agent_id)
    curr_in_danger = in_danger(obs, agent_id)
    if curr_in_danger:
        reward += CFG["r_in_danger"]
    if prev_in_danger and not curr_in_danger:
        reward += CFG["r_escape_danger"]

    if done and rank is not None:
        reward += CFG["rank_rewards"].get(rank, 0.0)

    return float(np.clip(reward, -CFG["reward_clip"], CFG["reward_clip"]))


@contextmanager
def temporary_random_seed(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def model_action(model, obs, agent_id: int, deterministic: bool, current_step: int, eval_mode: bool):
    _, map_feat, aux_feat, action_mask = prepare_policy_inputs(
        obs,
        agent_id,
        current_step=current_step,
        warmup_steps=CFG["mask_warmup_steps"],
        eval_mode=eval_mode,
    )
    map_feat = map_feat.unsqueeze(0).to(DEVICE)
    aux_feat = aux_feat.unsqueeze(0).to(DEVICE)
    action_mask = action_mask.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        canonical_action = int(
            model.get_action_inference(
                map_feat,
                aux_feat,
                deterministic=deterministic,
                action_mask=action_mask,
            )
        )
    env_action = to_env_action(canonical_action, agent_id)
    return env_action, canonical_action


def eval_mask_mode(current_step: int) -> bool:
    # Keep eval behavior aligned with training during the early exploration
    # phase, then switch both to the strict safe-bomb mask.
    return current_step >= CFG["mask_warmup_steps"]


def run_eval_match(model, opponent_classes, seed: int, agent_id: int, current_step: int):
    with temporary_random_seed(seed):
        env = BomberEnv(max_steps=500, seed=seed)
        obs = env.reset(seed=seed)

        opponent_rng = random.Random(seed ^ 0x5A5A)
        opponents = {
            player_id: opponent_rng.choice(opponent_classes)(player_id)
            for player_id in range(4)
            if player_id != agent_id
        }

        alive_mask = [bool(player.alive) for player in env.players]
        death_order = []

        terminated = False
        truncated = False
        while not (terminated or truncated):
            actions = [0] * 4
            if env.players[agent_id].alive:
                actions[agent_id], _ = model_action(
                    model,
                    obs,
                    agent_id,
                    deterministic=True,
                    current_step=current_step,
                    eval_mode=eval_mask_mode(current_step),
                )

            for opponent_id, opponent in opponents.items():
                if env.players[opponent_id].alive:
                    actions[opponent_id] = int(opponent.act(obs))

            obs, terminated, truncated = env.step(actions)
            record_deaths(env.players, alive_mask, death_order)

        ranks = compute_competition_ranks(env.players, death_order, alive_mask)
        rank = ranks[agent_id]
        first_group_size = sum(1 for r in ranks if r == 0)
        return {
            "rank": rank,
            "points": RANK_TO_POINTS[rank],
            "first": float(rank == 0),
            "unique_first": float(rank == 0 and first_group_size == 1),
            "shared_first": float(rank == 0 and first_group_size > 1),
        }


def evaluate_suite(model, opponent_classes, num_matches: int, seed_offset: int, seed_base: int, current_step: int):
    suite_rng = random.Random(seed_base + seed_offset)
    total_points = 0.0
    total_first = 0.0
    total_unique_first = 0.0
    total_shared_first = 0.0
    total_rank = 0.0

    for match_idx in range(num_matches):
        seed = seed_base + seed_offset + match_idx
        agent_id = suite_rng.randrange(4)
        result = run_eval_match(model, opponent_classes, seed=seed, agent_id=agent_id, current_step=current_step)
        total_points += result["points"]
        total_first += result["first"]
        total_unique_first += result["unique_first"]
        total_shared_first += result["shared_first"]
        total_rank += result["rank"]

    return {
        "matches": num_matches,
        "avg_points": total_points / max(num_matches, 1),
        "first_rate": total_first / max(num_matches, 1),
        "unique_first_rate": total_unique_first / max(num_matches, 1),
        "shared_first_rate": total_shared_first / max(num_matches, 1),
        "avg_rank": total_rank / max(num_matches, 1),
        "total_points": total_points,
    }


def _evaluate_policy_with_seed_base(model, seed_base: int, easy_medium_matches: int, hard_matches: int, current_step: int):
    easy_medium = evaluate_suite(
        model,
        opponent_classes=[SimpleRuleAgent, SmarterRuleAgent, BoxFarmerAgent],
        num_matches=easy_medium_matches,
        seed_offset=0,
        seed_base=seed_base,
        current_step=current_step,
    )
    hard = evaluate_suite(
        model,
        opponent_classes=[GeniusRuleAgent, TacticalRuleAgent],
        num_matches=hard_matches,
        seed_offset=10_000,
        seed_base=seed_base,
        current_step=current_step,
    )
    total_matches = easy_medium["matches"] + hard["matches"]
    total_points = easy_medium["total_points"] + hard["total_points"]
    return {
        "score": total_points / max(total_matches, 1),
        "easy_medium": easy_medium,
        "hard": hard,
    }


def evaluate_policy(model, current_step: int):
    dev = _evaluate_policy_with_seed_base(
        model,
        seed_base=CFG["eval_seed_base"],
        easy_medium_matches=CFG["eval_easy_medium_matches"],
        hard_matches=CFG["eval_hard_matches"],
        current_step=current_step,
    )
    holdout = _evaluate_policy_with_seed_base(
        model,
        seed_base=CFG["holdout_eval_seed_base"],
        easy_medium_matches=CFG["holdout_eval_easy_medium_matches"],
        hard_matches=CFG["holdout_eval_hard_matches"],
        current_step=current_step,
    )
    return {"dev": dev, "holdout": holdout}


class OpponentPool:
    def __init__(self, model_factory, pool_size=8):
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
            return ModelOpponent(model, agent_id, current_step)

        if current_step < CFG["mask_warmup_steps"]:
            cls = random.choice(self.easy_bots)
        elif current_step < 1_100_000:
            cls = random.choice(self.medium_bots)
        else:
            cls = random.choice(self.hard_bots)
        return cls(agent_id)


class ModelOpponent:
    def __init__(self, model, agent_id, current_step: int):
        self.model = model
        self.agent_id = agent_id
        self.current_step = current_step

    def act(self, obs):
        env_action, _ = model_action(
            self.model,
            obs,
            self.agent_id,
            deterministic=False,
            current_step=self.current_step,
            eval_mode=eval_mask_mode(self.current_step),
        )
        return env_action


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
    episode_ctx = make_episode_context(obs, agent_id)
    return obs, agent_id, opponents, alive_mask, death_order, episode_ctx


def train():
    os.makedirs(CFG["ckpt_dir"], exist_ok=True)

    model = CNNActorCriticV2(num_actions=6).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"], eps=1e-5)

    pool = OpponentPool(model_factory=lambda: CNNActorCriticV2(num_actions=6))
    buffer = RolloutBuffer(CFG["n_steps"], DEVICE)
    env = BomberEnv(max_steps=500)

    global_step = 0
    episode = 0
    best_eval_score = float("-inf")
    last_save_step = 0
    first_history = deque(maxlen=100)

    obs, agent_id, opponents, alive_mask, death_order, episode_ctx = start_episode(env, pool, global_step)

    print(f"\n{'=' * 60}")
    print("Training PPO Agent V2 - Bomberland Self-Play")
    print(f"Device: {DEVICE} | Total steps: {CFG['total_steps']:,}")
    print(f"{'=' * 60}\n")

    t_start = time.time()

    while global_step < CFG["total_steps"]:
        buffer.reset()
        model.eval()

        for _ in range(CFG["n_steps"]):
            global_step += 1

            _, map_feat, aux_feat, action_mask = prepare_policy_inputs(
                obs,
                agent_id,
                current_step=global_step,
                warmup_steps=CFG["mask_warmup_steps"],
                eval_mode=False,
            )
            map_feat_batch = map_feat.unsqueeze(0).to(DEVICE)
            aux_feat_batch = aux_feat.unsqueeze(0).to(DEVICE)
            action_mask_batch = action_mask.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                canonical_action, log_prob, _, value = model.get_action_and_value(
                    map_feat_batch,
                    aux_feat_batch,
                    action_mask=action_mask_batch,
                )

            env_action = to_env_action(int(canonical_action.item()), agent_id)

            actions = [0] * 4
            actions[agent_id] = env_action
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
                episode_ctx,
            )

            buffer.push(
                map_feat.cpu(),
                aux_feat.cpu(),
                action_mask.cpu(),
                int(canonical_action.item()),
                float(log_prob.item()),
                reward,
                float(value.item()),
                float(done),
            )

            if done:
                episode += 1
                first_history.append(1.0 if agent_rank == 0 else 0.0)
                obs, agent_id, opponents, alive_mask, death_order, episode_ctx = start_episode(
                    env,
                    pool,
                    global_step,
                )

        model.train()
        sync_linear_lr(optimizer, global_step)

        with torch.no_grad():
            _, map_feat_last, aux_feat_last, action_mask_last = prepare_policy_inputs(
                obs,
                agent_id,
                current_step=global_step,
                warmup_steps=CFG["mask_warmup_steps"],
                eval_mode=False,
            )
            _, _, _, last_val = model.get_action_and_value(
                map_feat_last.unsqueeze(0).to(DEVICE),
                aux_feat_last.unsqueeze(0).to(DEVICE),
                action_mask=action_mask_last.unsqueeze(0).to(DEVICE),
            )
            last_value = float(last_val.item())

        stats = ppo_update(model, optimizer, buffer, last_value)

        first_rate = np.mean(first_history) if first_history else 0.0
        elapsed = time.time() - t_start
        sps = global_step / max(elapsed, 1.0)

        print(
            f"Step {global_step:>8,} | Ep {episode:>5} | "
            f"FirstRate {first_rate:.2%} | "
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
                    "first_rate": first_rate,
                },
                ckpt_path,
            )
            print(f"  -> Saved checkpoint: {ckpt_path}")

            pool.add_checkpoint(model.state_dict())
            eval_stats = evaluate_policy(model, current_step=global_step)
            dev_stats = eval_stats["dev"]
            holdout_stats = eval_stats["holdout"]

            print(
                "  -> Dev "
                f"score {dev_stats['score']:.3f} | "
                f"EM unique/shared {dev_stats['easy_medium']['unique_first_rate']:.2%}/{dev_stats['easy_medium']['shared_first_rate']:.2%} | "
                f"Hard unique/shared {dev_stats['hard']['unique_first_rate']:.2%}/{dev_stats['hard']['shared_first_rate']:.2%}"
            )
            print(
                "  -> Holdout "
                f"score {holdout_stats['score']:.3f} | "
                f"EM unique/shared {holdout_stats['easy_medium']['unique_first_rate']:.2%}/{holdout_stats['easy_medium']['shared_first_rate']:.2%} | "
                f"Hard unique/shared {holdout_stats['hard']['unique_first_rate']:.2%}/{holdout_stats['hard']['shared_first_rate']:.2%}"
            )

            if dev_stats["score"] >= best_eval_score:
                best_eval_score = dev_stats["score"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "global_step": global_step,
                        "eval": eval_stats,
                    },
                    _HERE / "model.pth",
                )
                print(f"  -> New best v2 model! Dev score: {best_eval_score:.3f}")

    last_model_path = _HERE / "model_last.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "global_step": global_step,
        },
        last_model_path,
    )

    if best_eval_score == float("-inf"):
        torch.save({"model": model.state_dict(), "global_step": global_step}, _HERE / "model.pth")

    print(f"\nTraining done! Best eval model saved to {_HERE / 'model.pth'}")
    print(f"Latest weights saved to {last_model_path}")


if __name__ == "__main__":
    train()
