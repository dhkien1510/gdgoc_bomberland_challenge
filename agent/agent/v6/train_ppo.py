"""
Recurrent PPO fine-tuning for the v6 BC+PPO agent.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
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
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
sys.modules.pop("model", None)
sys.modules.pop("_model_base", None)
sys.modules.pop("_model_v3_base", None)
sys.modules.pop("_train_base", None)

import _train_base as base
from bc_model import CNNLSTMBCActor

_MODEL_SPEC = importlib.util.spec_from_file_location("_v6_model_train_ppo", _HERE / "model.py")
_MODEL = importlib.util.module_from_spec(_MODEL_SPEC)
assert _MODEL_SPEC.loader is not None
_MODEL_SPEC.loader.exec_module(_MODEL)

RecurrentActorCriticV6 = _MODEL.RecurrentActorCriticV6
VALUE_BOMB_MASK_STEPS = _MODEL.VALUE_BOMB_MASK_STEPS
prepare_policy_inputs = _MODEL.prepare_policy_inputs
to_env_action = _MODEL.to_env_action

CFG = copy.deepcopy(base.CFG)
CFG.update(
    {
        "entropy_coef": 0.001,
        "n_steps": 1024,
        "num_envs": 1,
        "seq_len": 64,
        "batch_size_sequences": 16,
        "ppo_epochs": 4,
        "bc_coef": 0.1,
        "actor_core_lr": 5.0e-5,
        "actor_head_lr": 1.0e-4,
        "critic_lr": base.CFG["lr"],
        "initial_actor_warmup_steps": 100_000,
        "stage_actor_warmup_steps": 15_000,
        "actor_init_candidates": [
            str(_HERE / "bc_actor.pth"),
            str(_HERE / "model.pth"),
            str(_HERE.parent / "v5_2" / "model.pth"),
        ],
        "ckpt_dir": str(_HERE / "checkpoints"),
    }
)

DEVICE = base.DEVICE
print(f"Using device: {DEVICE}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=CFG["total_steps"])
    parser.add_argument("--save_every", type=int, default=CFG["save_every"])
    parser.add_argument("--n_steps", type=int, default=CFG["n_steps"])
    parser.add_argument("--num_envs", type=int, default=CFG["num_envs"])
    parser.add_argument("--seq_len", type=int, default=CFG["seq_len"])
    parser.add_argument("--batch_size_sequences", type=int, default=CFG["batch_size_sequences"])
    parser.add_argument("--ppo_epochs", type=int, default=CFG["ppo_epochs"])
    parser.add_argument("--bc_coef", type=float, default=CFG["bc_coef"])
    parser.add_argument("--entropy_coef", type=float, default=CFG["entropy_coef"])
    parser.add_argument("--actor_core_lr", type=float, default=CFG["actor_core_lr"])
    parser.add_argument("--actor_head_lr", type=float, default=CFG["actor_head_lr"])
    parser.add_argument("--critic_lr", type=float, default=CFG["critic_lr"])
    parser.add_argument("--initial_actor_warmup_steps", type=int, default=CFG["initial_actor_warmup_steps"])
    parser.add_argument("--stage_actor_warmup_steps", type=int, default=CFG["stage_actor_warmup_steps"])
    parser.add_argument("--eval_easy_medium_matches", type=int, default=CFG["eval_easy_medium_matches"])
    parser.add_argument("--eval_hard_matches", type=int, default=CFG["eval_hard_matches"])
    parser.add_argument("--ckpt_dir", type=str, default=CFG["ckpt_dir"])
    parser.add_argument("--actor_init_path", type=str, default="")
    parser.add_argument("--bc_reference_path", type=str, default=str(_HERE / "bc_actor.pth"))
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--resume_global_step_override", type=int, default=-1)
    parser.add_argument("--eval_only", type=int, default=0)
    return parser.parse_args()


def apply_cli_overrides(args):
    CFG["total_steps"] = int(args.total_steps)
    CFG["save_every"] = int(args.save_every)
    CFG["n_steps"] = int(args.n_steps)
    CFG["num_envs"] = int(args.num_envs)
    CFG["seq_len"] = int(args.seq_len)
    CFG["batch_size_sequences"] = int(args.batch_size_sequences)
    CFG["ppo_epochs"] = int(args.ppo_epochs)
    CFG["bc_coef"] = float(args.bc_coef)
    CFG["entropy_coef"] = float(args.entropy_coef)
    CFG["actor_core_lr"] = float(args.actor_core_lr)
    CFG["actor_head_lr"] = float(args.actor_head_lr)
    CFG["critic_lr"] = float(args.critic_lr)
    CFG["initial_actor_warmup_steps"] = int(args.initial_actor_warmup_steps)
    CFG["stage_actor_warmup_steps"] = int(args.stage_actor_warmup_steps)
    CFG["eval_easy_medium_matches"] = int(args.eval_easy_medium_matches)
    CFG["eval_hard_matches"] = int(args.eval_hard_matches)
    CFG["ckpt_dir"] = str(args.ckpt_dir)
    CFG["actor_init_path"] = str(args.actor_init_path).strip()
    CFG["bc_reference_path"] = str(args.bc_reference_path).strip()
    CFG["resume_checkpoint"] = str(args.resume_checkpoint).strip()
    CFG["resume_global_step_override"] = int(args.resume_global_step_override)
    CFG["eval_only"] = bool(int(args.eval_only))
    if CFG["initial_actor_warmup_steps"] >= CFG["total_steps"]:
        print(
            "Warning: initial_actor_warmup_steps >= total_steps; "
            "the actor may stay frozen for the whole run."
        )
    if CFG["save_every"] > CFG["total_steps"]:
        print("Warning: save_every > total_steps; only the final model may be saved.")
    if CFG["num_envs"] <= 0:
        raise ValueError("num_envs must be >= 1")
    if CFG["n_steps"] % CFG["num_envs"] != 0:
        raise ValueError("n_steps must be divisible by num_envs for vectorized rollout")


def _load_checkpoint(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _try_load_optimizer_state(optimizer: optim.Optimizer, optimizer_state) -> bool:
    if optimizer_state is None:
        return False
    saved_groups = optimizer_state.get("param_groups")
    if not isinstance(saved_groups, list):
        return False
    current_groups = optimizer.param_groups
    if len(saved_groups) != len(current_groups):
        return False
    try:
        optimizer.load_state_dict(optimizer_state)
    except ValueError:
        return False
    return True


def sync_group_lrs(optimizer: optim.Optimizer, global_step: int):
    frac = max(1.0 - (global_step / CFG["total_steps"]), 0.05)
    for group in optimizer.param_groups:
        base_lr = float(group.get("base_lr", CFG["critic_lr"]))
        group["lr"] = base_lr * frac


class SequenceRolloutBuffer:
    def __init__(self, num_envs: int):
        self.num_envs = int(num_envs)
        self.reset()

    def reset(self):
        self.map_feats = [[] for _ in range(self.num_envs)]
        self.aux_feats = [[] for _ in range(self.num_envs)]
        self.action_masks = [[] for _ in range(self.num_envs)]
        self.actor_h = [[] for _ in range(self.num_envs)]
        self.actor_c = [[] for _ in range(self.num_envs)]
        self.actions = [[] for _ in range(self.num_envs)]
        self.bc_teacher_actions = [[] for _ in range(self.num_envs)]
        self.log_probs = [[] for _ in range(self.num_envs)]
        self.rewards = [[] for _ in range(self.num_envs)]
        self.values = [[] for _ in range(self.num_envs)]
        self.dones = [[] for _ in range(self.num_envs)]
        self.episode_starts = [[] for _ in range(self.num_envs)]

    def push(
        self,
        env_idx,
        map_feat,
        aux_feat,
        action_mask,
        actor_state,
        action,
        bc_teacher_action,
        log_prob,
        reward,
        value,
        done,
        episode_start,
    ):
        env_idx = int(env_idx)
        self.map_feats[env_idx].append(map_feat.clone())
        self.aux_feats[env_idx].append(aux_feat.clone())
        self.action_masks[env_idx].append(action_mask.clone())
        self.actor_h[env_idx].append(actor_state[0].detach().cpu().clone())
        self.actor_c[env_idx].append(actor_state[1].detach().cpu().clone())
        self.actions[env_idx].append(int(action))
        self.bc_teacher_actions[env_idx].append(int(bc_teacher_action))
        self.log_probs[env_idx].append(float(log_prob))
        self.rewards[env_idx].append(float(reward))
        self.values[env_idx].append(float(value))
        self.dones[env_idx].append(float(done))
        self.episode_starts[env_idx].append(float(episode_start))

    def compute_gae(self, last_values, gamma, gae_lambda):
        advantages = []
        returns = []
        for env_idx in range(self.num_envs):
            env_rewards = self.rewards[env_idx]
            env_values = self.values[env_idx]
            env_dones = self.dones[env_idx]
            env_advantages = np.zeros(len(env_rewards), dtype=np.float32)
            last_gae = 0.0
            next_value = float(last_values[env_idx])
            for t in reversed(range(len(env_rewards))):
                nonterminal = 1.0 - env_dones[t]
                delta = env_rewards[t] + gamma * next_value * nonterminal - env_values[t]
                last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
                env_advantages[t] = last_gae
                next_value = env_values[t]
            env_returns = env_advantages + np.asarray(env_values, dtype=np.float32)
            advantages.append(env_advantages)
            returns.append(env_returns)
        return advantages, returns

    def make_windows(self, seq_len: int):
        windows = []
        for env_idx in range(self.num_envs):
            env_len = len(self.actions[env_idx])
            for start in range(0, env_len, seq_len):
                windows.append((env_idx, start, min(start + seq_len, env_len)))
        return windows

    def build_sequence_batch(self, windows, advantages, returns, device):
        batch_size = len(windows)
        seq_len = max(end - start for _env_idx, start, end in windows)
        sample_env_idx = next((idx for idx in range(self.num_envs) if self.map_feats[idx]), None)
        if sample_env_idx is None:
            raise RuntimeError("Cannot build a sequence batch from an empty rollout buffer")
        map_shape = self.map_feats[sample_env_idx][0].shape
        aux_dim = self.aux_feats[sample_env_idx][0].shape[0]
        action_dim = self.action_masks[sample_env_idx][0].shape[0]

        map_batch = torch.zeros((batch_size, seq_len, *map_shape), dtype=torch.float32, device=device)
        aux_batch = torch.zeros((batch_size, seq_len, aux_dim), dtype=torch.float32, device=device)
        mask_batch = torch.zeros((batch_size, seq_len, action_dim), dtype=torch.bool, device=device)
        action_batch = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        bc_teacher_action_batch = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        old_log_prob_batch = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        adv_batch = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        ret_batch = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        loss_mask = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        episode_start_batch = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        hidden_size = self.actor_h[sample_env_idx][0].shape[0]
        init_h_batch = torch.zeros((batch_size, hidden_size), dtype=torch.float32, device=device)
        init_c_batch = torch.zeros((batch_size, hidden_size), dtype=torch.float32, device=device)

        for batch_idx, (env_idx, start, end) in enumerate(windows):
            length = end - start
            init_h_batch[batch_idx] = self.actor_h[env_idx][start]
            init_c_batch[batch_idx] = self.actor_c[env_idx][start]
            for t in range(length):
                src = start + t
                map_batch[batch_idx, t] = self.map_feats[env_idx][src]
                aux_batch[batch_idx, t] = self.aux_feats[env_idx][src]
                mask_batch[batch_idx, t] = self.action_masks[env_idx][src]
                action_batch[batch_idx, t] = self.actions[env_idx][src]
                bc_teacher_action_batch[batch_idx, t] = self.bc_teacher_actions[env_idx][src]
                old_log_prob_batch[batch_idx, t] = self.log_probs[env_idx][src]
                adv_batch[batch_idx, t] = float(advantages[env_idx][src])
                ret_batch[batch_idx, t] = float(returns[env_idx][src])
                loss_mask[batch_idx, t] = 1.0
                episode_start_batch[batch_idx, t] = self.episode_starts[env_idx][src]

        return {
            "map_feats": map_batch,
            "aux_feats": aux_batch,
            "action_masks": mask_batch,
            "init_h": init_h_batch,
            "init_c": init_c_batch,
            "actions": action_batch,
            "bc_teacher_actions": bc_teacher_action_batch,
            "old_log_probs": old_log_prob_batch,
            "advantages": adv_batch,
            "returns": ret_batch,
            "loss_mask": loss_mask,
            "episode_starts": episode_start_batch,
        }


class ActiveOpponentPoolV6(base.ActiveOpponentPool):
    def _load_checkpoint_opponent(self, checkpoint, agent_id: int, current_step: int):
        model = self.model_factory()
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        return ModelOpponentV6(model, agent_id, current_step)


class ModelOpponentV6:
    def __init__(self, model, agent_id, current_step: int):
        self.model = model
        self.agent_id = int(agent_id)
        self.current_step = int(current_step)
        self.state = model.get_initial_actor_state(1, DEVICE)
        self.episode_start = True

    def act(self, obs):
        env_action, _canonical_action, self.state = model_action(
            self.model,
            obs,
            self.agent_id,
            deterministic=False,
            current_step=self.current_step,
            state=self.state,
            episode_start=self.episode_start,
        )
        self.episode_start = False
        return env_action


def resolve_actor_init_path() -> Path | None:
    explicit = str(CFG.get("actor_init_path", "")).strip()
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    for candidate in CFG["actor_init_candidates"]:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def resolve_resume_checkpoint_path() -> Path | None:
    explicit = str(CFG.get("resume_checkpoint", "")).strip()
    if not explicit:
        return None
    path = Path(explicit)
    return path if path.exists() else None


def initial_actor_warmup_active(global_step: int, actor_initialized: bool) -> bool:
    return actor_initialized and global_step < CFG["initial_actor_warmup_steps"]


def stage_actor_warmup_active(global_step: int, stage_warmup_until: int) -> bool:
    return global_step < stage_warmup_until


def actor_trainable_now(global_step: int, actor_initialized: bool, stage_warmup_until: int) -> bool:
    return not (
        initial_actor_warmup_active(global_step, actor_initialized)
        or stage_actor_warmup_active(global_step, stage_warmup_until)
    )


def model_action(model, obs, agent_id: int, deterministic: bool, current_step: int, state, episode_start: bool):
    _, map_feat, aux_feat, action_mask = prepare_policy_inputs(
        obs,
        agent_id,
        current_step=current_step,
        warmup_steps=CFG["mask_warmup_steps"],
        value_bomb_mask_steps=VALUE_BOMB_MASK_STEPS,
        eval_mode=False,
    )
    map_feat = map_feat.unsqueeze(0).to(DEVICE)
    aux_feat = aux_feat.unsqueeze(0).to(DEVICE)
    action_mask = action_mask.unsqueeze(0).to(DEVICE)
    episode_start_tensor = torch.tensor([float(episode_start)], device=DEVICE)

    with torch.no_grad():
        action, next_state = model.get_action_inference(
            map_feat,
            aux_feat,
            deterministic=deterministic,
            action_mask=action_mask,
            state=state,
            episode_start=episode_start_tensor,
        )
    env_action = to_env_action(int(action), agent_id)
    return env_action, int(action), next_state


def run_eval_match(model, opponent_classes, seed: int, agent_id: int, current_step: int):
    with base.temporary_random_seed(seed):
        env = base.BomberEnv(max_steps=500, seed=seed)
        obs = env.reset(seed=seed)

        opponent_rng = random.Random(seed ^ 0x5A5A)
        opponents = {
            player_id: opponent_rng.choice(opponent_classes)(player_id)
            for player_id in range(4)
            if player_id != agent_id
        }
        actor_state = model.get_initial_actor_state(1, DEVICE)
        episode_start = True

        alive_mask = [bool(player.alive) for player in env.players]
        death_order = []
        episode_ctx = base.make_episode_context(obs, agent_id)
        done = False

        while not done:
            actions = [0] * 4
            if env.players[agent_id].alive:
                actions[agent_id], canonical_action, actor_state = model_action(
                    model,
                    obs,
                    agent_id,
                    deterministic=True,
                    current_step=current_step,
                    state=actor_state,
                    episode_start=episode_start,
                )
            else:
                canonical_action = 0
            episode_start = False

            for opponent_id, opponent in opponents.items():
                if env.players[opponent_id].alive:
                    actions[opponent_id] = int(opponent.act(obs))

            prev_obs = base.clone_obs(obs)
            prev_stats = base.clone_stats(env.players[agent_id])
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            base.record_deaths(env.players, alive_mask, death_order)
            curr_stats = base.clone_stats(env.players[agent_id])
            base.compute_reward(
                prev_obs,
                obs,
                prev_stats,
                curr_stats,
                agent_id,
                done,
                None,
                episode_ctx,
                base.get_stage(current_step),
                canonical_action=canonical_action,
            )

        ranks = base.compute_competition_ranks(env.players, death_order, alive_mask)
        rank = ranks[agent_id]
        first_group_size = sum(1 for r in ranks if r == 0)
        final_stats = base.clone_stats(env.players[agent_id])
        episode_metrics = base.summarize_episode_metrics(episode_ctx, final_stats)
        return {
            "rank": rank,
            "points": base.RANK_TO_POINTS[rank],
            "first": float(rank == 0),
            "unique_first": float(rank == 0 and first_group_size == 1),
            "shared_first": float(rank == 0 and first_group_size > 1),
            "bombs": episode_metrics["bombs_per_episode"],
            "kills": episode_metrics["kills_per_episode"],
            "boxes": episode_metrics["boxes_per_episode"],
            "items": episode_metrics["items_per_episode"],
            "valuable_bomb_ratio": episode_metrics["valuable_bomb_ratio"],
            "repeat_position_rate": episode_metrics["repeat_position_rate"],
            "danger_steps": episode_metrics["danger_steps_per_episode"],
        }


def evaluate_suite(model, opponent_classes, num_matches: int, seed_offset: int, seed_base: int, current_step: int):
    suite_rng = random.Random(seed_base + seed_offset)
    rank_counts = [0, 0, 0, 0]
    totals = {
        "points": 0.0,
        "first": 0.0,
        "unique_first": 0.0,
        "shared_first": 0.0,
        "rank": 0.0,
        "bombs": 0.0,
        "kills": 0.0,
        "boxes": 0.0,
        "items": 0.0,
        "valuable_bomb_ratio": 0.0,
        "repeat_position_rate": 0.0,
        "danger_steps": 0.0,
    }
    for match_idx in range(num_matches):
        seed = seed_base + seed_offset + match_idx
        agent_id = suite_rng.randrange(4)
        result = run_eval_match(model, opponent_classes, seed=seed, agent_id=agent_id, current_step=current_step)
        if 0 <= int(result["rank"]) < len(rank_counts):
            rank_counts[int(result["rank"])] += 1
        for key in totals:
            if key in result:
                totals[key] += result[key]
    denom = max(num_matches, 1)
    return {
        "matches": num_matches,
        "avg_points": totals["points"] / denom,
        "first_rate": totals["first"] / denom,
        "unique_first_rate": totals["unique_first"] / denom,
        "shared_first_rate": totals["shared_first"] / denom,
        "avg_rank": totals["rank"] / denom,
        "avg_bombs": totals["bombs"] / denom,
        "avg_kills": totals["kills"] / denom,
        "avg_boxes": totals["boxes"] / denom,
        "avg_items": totals["items"] / denom,
        "valuable_bomb_ratio": totals["valuable_bomb_ratio"] / denom,
        "repeat_position_rate": totals["repeat_position_rate"] / denom,
        "danger_steps": totals["danger_steps"] / denom,
        "total_points": totals["points"],
        "rank0_rate": rank_counts[0] / denom,
        "rank1_rate": rank_counts[1] / denom,
        "rank2_rate": rank_counts[2] / denom,
        "rank3_rate": rank_counts[3] / denom,
    }


def evaluate_policy(model, current_step: int):
    easy_medium = evaluate_suite(
        model,
        opponent_classes=[base.SimpleRuleAgent, base.SmarterRuleAgent, base.BoxFarmerAgent],
        num_matches=CFG["eval_easy_medium_matches"],
        seed_offset=0,
        seed_base=CFG["eval_seed_base"],
        current_step=current_step,
    )
    hard = evaluate_suite(
        model,
        opponent_classes=[base.GeniusRuleAgent, base.TacticalRuleAgent],
        num_matches=CFG["eval_hard_matches"],
        seed_offset=10_000,
        seed_base=CFG["eval_seed_base"],
        current_step=current_step,
    )
    total_matches = easy_medium["matches"] + hard["matches"]
    total_points = easy_medium["total_points"] + hard["total_points"]
    total_first = easy_medium["first_rate"] * easy_medium["matches"] + hard["first_rate"] * hard["matches"]
    total_unique_first = (
        easy_medium["unique_first_rate"] * easy_medium["matches"]
        + hard["unique_first_rate"] * hard["matches"]
    )
    total_shared_first = (
        easy_medium["shared_first_rate"] * easy_medium["matches"]
        + hard["shared_first_rate"] * hard["matches"]
    )
    total_rank = easy_medium["avg_rank"] * easy_medium["matches"] + hard["avg_rank"] * hard["matches"]
    total_rank0 = easy_medium["rank0_rate"] * easy_medium["matches"] + hard["rank0_rate"] * hard["matches"]
    total_rank1 = easy_medium["rank1_rate"] * easy_medium["matches"] + hard["rank1_rate"] * hard["matches"]
    total_rank2 = easy_medium["rank2_rate"] * easy_medium["matches"] + hard["rank2_rate"] * hard["matches"]
    total_rank3 = easy_medium["rank3_rate"] * easy_medium["matches"] + hard["rank3_rate"] * hard["matches"]
    return {
        "score": total_points / max(total_matches, 1),
        "avg_rank": total_rank / max(total_matches, 1),
        "first_rate": total_first / max(total_matches, 1),
        "unique_first_rate": total_unique_first / max(total_matches, 1),
        "shared_first_rate": total_shared_first / max(total_matches, 1),
        "rank0_rate": total_rank0 / max(total_matches, 1),
        "rank1_rate": total_rank1 / max(total_matches, 1),
        "rank2_rate": total_rank2 / max(total_matches, 1),
        "rank3_rate": total_rank3 / max(total_matches, 1),
        "easy_medium": easy_medium,
        "hard": hard,
    }


def ppo_update(
    model,
    optimizer,
    buffer: SequenceRolloutBuffer,
    last_values,
    actor_trainable: bool,
    bc_coef: float,
):
    advantages, returns = buffer.compute_gae(last_values, CFG["gamma"], CFG["gae_lambda"])
    flat_advantages = np.concatenate([env_adv for env_adv in advantages if len(env_adv) > 0], axis=0)
    flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
    offset = 0
    norm_advantages = []
    for env_adv in advantages:
        env_len = len(env_adv)
        norm_advantages.append(flat_advantages[offset:offset + env_len].copy())
        offset += env_len
    advantages = norm_advantages
    windows = buffer.make_windows(CFG["seq_len"])

    total_pg_loss = 0.0
    total_val_loss = 0.0
    total_bc_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    total_clip_fraction = 0.0
    total_ratio_mean = 0.0
    total_ratio_std = 0.0
    total_explained_variance = 0.0
    num_updates = 0

    for _ in range(CFG["ppo_epochs"]):
        random.shuffle(windows)
        for start in range(0, len(windows), CFG["batch_size_sequences"]):
            batch_windows = windows[start:start + CFG["batch_size_sequences"]]
            batch = buffer.build_sequence_batch(batch_windows, advantages, returns, DEVICE)
            _, new_log_probs, entropy, new_values, _ = model.get_action_and_value_sequence(
                batch["map_feats"],
                batch["aux_feats"],
                actions=batch["actions"],
                action_mask_seq=batch["action_masks"],
                episode_start_mask=batch["episode_starts"],
                state=(batch["init_h"], batch["init_c"]),
            )

            valid = batch["loss_mask"] > 0
            valid_returns = batch["returns"][valid]
            valid_values = new_values[valid]
            returns_var = torch.var(valid_returns, unbiased=False)
            if float(returns_var.item()) > 1e-8:
                explained_variance = 1.0 - torch.var(valid_returns - valid_values, unbiased=False) / returns_var
            else:
                explained_variance = torch.zeros((), device=DEVICE)
            if actor_trainable:
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                ratio_valid = ratio[valid]
                pg_loss1 = -batch["advantages"] * ratio
                pg_loss2 = -batch["advantages"] * torch.clamp(ratio, 1 - CFG["clip_eps"], 1 + CFG["clip_eps"])
                pg_loss = torch.max(pg_loss1, pg_loss2)[valid].mean()
                entropy_term = CFG["entropy_coef"] * entropy[valid].mean()
                approx_kl = (batch["old_log_probs"][valid] - new_log_probs[valid]).mean()
                clip_fraction = ((ratio_valid - 1.0).abs() > CFG["clip_eps"]).float().mean()
                ratio_mean = ratio_valid.mean()
                ratio_std = ratio_valid.std(unbiased=False)
                flat_logits, _ = model.forward_actor_sequence(
                    batch["map_feats"],
                    batch["aux_feats"],
                    action_mask_seq=batch["action_masks"],
                    episode_start_mask=batch["episode_starts"],
                    state=(batch["init_h"], batch["init_c"]),
                )
                flat_valid = valid.view(-1)
                bc_loss = F.cross_entropy(
                    flat_logits.view(-1, flat_logits.shape[-1])[flat_valid],
                    batch["bc_teacher_actions"].view(-1)[flat_valid],
                )
            else:
                pg_loss = torch.zeros((), device=DEVICE)
                entropy_term = torch.zeros((), device=DEVICE)
                bc_loss = torch.zeros((), device=DEVICE)
                approx_kl = torch.zeros((), device=DEVICE)
                clip_fraction = torch.zeros((), device=DEVICE)
                ratio_mean = torch.zeros((), device=DEVICE)
                ratio_std = torch.zeros((), device=DEVICE)

            val_loss = 0.5 * ((new_values - batch["returns"]) ** 2)[valid].mean()
            loss = pg_loss + CFG["value_coef"] * val_loss + bc_coef * bc_loss - entropy_term

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
            optimizer.step()

            total_pg_loss += float(pg_loss.item())
            total_val_loss += float(val_loss.item())
            total_bc_loss += float(bc_loss.item())
            total_entropy += float(entropy[valid].mean().item())
            total_approx_kl += float(approx_kl.item())
            total_clip_fraction += float(clip_fraction.item())
            total_ratio_mean += float(ratio_mean.item())
            total_ratio_std += float(ratio_std.item())
            total_explained_variance += float(explained_variance.item())
            num_updates += 1

    return {
        "pg_loss": total_pg_loss / max(num_updates, 1),
        "val_loss": total_val_loss / max(num_updates, 1),
        "bc_loss": total_bc_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
        "approx_kl": total_approx_kl / max(num_updates, 1),
        "clip_fraction": total_clip_fraction / max(num_updates, 1),
        "ratio_mean": total_ratio_mean / max(num_updates, 1),
        "ratio_std": total_ratio_std / max(num_updates, 1),
        "explained_variance": total_explained_variance / max(num_updates, 1),
    }


def train():
    os.makedirs(CFG["ckpt_dir"], exist_ok=True)
    num_envs = CFG["num_envs"]
    rollout_env_steps = CFG["n_steps"] // num_envs

    model = RecurrentActorCriticV6().to(DEVICE)
    actor_init_path = resolve_actor_init_path()
    resume_checkpoint_path = resolve_resume_checkpoint_path()
    actor_initialized = False
    bc_reference_path = Path(str(CFG.get("bc_reference_path", _HERE / "bc_actor.pth")))
    bc_reference_model = None
    if resume_checkpoint_path is None:
        if actor_init_path is not None:
            result = model.load_actor_from_checkpoint(str(actor_init_path))
            actor_initialized = bool(result["loaded_actor"])
            print(f"Actor init from {actor_init_path}: {actor_initialized}")
        else:
            print("Actor init checkpoint not found, starting PPO actor from scratch")
    else:
        print(f"Resume checkpoint requested: {resume_checkpoint_path}")
    if bc_reference_path.exists():
        checkpoint = _load_checkpoint(bc_reference_path, map_location=DEVICE)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        bc_reference_model = CNNLSTMBCActor().to(DEVICE)
        bc_reference_model.load_state_dict(state_dict)
        bc_reference_model.eval()
        print(f"Loaded frozen BC reference from {bc_reference_path}")
    else:
        print("BC reference checkpoint not found, disabling BC regularization during PPO")

    optimizer = optim.Adam(
        [
            {
                "params": list(model.actor_core.parameters()),
                "lr": CFG["actor_core_lr"],
                "base_lr": CFG["actor_core_lr"],
                "name": "actor_core",
            },
            {
                "params": list(model.actor_head.parameters()),
                "lr": CFG["actor_head_lr"],
                "base_lr": CFG["actor_head_lr"],
                "name": "actor_head",
            },
            {
                "params": list(model.critic_parameters()),
                "lr": CFG["critic_lr"],
                "base_lr": CFG["critic_lr"],
                "name": "critic",
            },
        ],
        eps=1e-5,
    )
    global_step = 0
    best_eval_score = float("-inf")
    stage_warmup_until = 0
    last_save_step = 0
    if resume_checkpoint_path is not None:
        resume_payload = _load_checkpoint(resume_checkpoint_path, map_location=DEVICE)
        if not isinstance(resume_payload, dict) or "model" not in resume_payload:
            raise RuntimeError(
                f"Resume checkpoint {resume_checkpoint_path} does not contain a full PPO checkpoint payload"
            )
        model.load_state_dict(resume_payload["model"])
        optimizer_state = resume_payload.get("optimizer")
        optimizer_loaded = _try_load_optimizer_state(optimizer, optimizer_state)
        global_step = int(resume_payload.get("global_step", 0))
        if CFG.get("resume_global_step_override", -1) >= 0:
            global_step = int(CFG["resume_global_step_override"])
        actor_initialized = bool(resume_payload.get("actor_initialized", True))
        stage_warmup_until = int(resume_payload.get("stage_warmup_until", 0))
        if CFG.get("resume_global_step_override", -1) >= 0:
            stage_warmup_until = 0
        best_eval_score = float(
            resume_payload.get(
                "best_eval_score",
                resume_payload.get("eval", {}).get("score", float("-inf")),
            )
        )
        last_save_step = global_step
        print(
            f"Resumed PPO from step {global_step:,} | "
            f"stage {resume_payload.get('stage', base.get_stage(global_step)['name'])} | "
            f"actor_initialized {actor_initialized}"
        )
        if CFG.get("resume_global_step_override", -1) >= 0:
            print(
                f"Resume global_step override applied: {CFG['resume_global_step_override']:,} | "
                f"effective stage {base.get_stage(global_step)['name']}"
            )
        if optimizer_state is None:
            print("Resume checkpoint had no optimizer state; continuing with a fresh optimizer")
        elif not optimizer_loaded:
            print(
                "Resume checkpoint optimizer state is incompatible with the current optimizer groups; "
                "continuing with a fresh optimizer"
            )

    model.set_actor_trainable(actor_trainable_now(global_step, actor_initialized, stage_warmup_until))
    pool = ActiveOpponentPoolV6(
        model_factory=lambda: RecurrentActorCriticV6(),
        recent_size=CFG["pool_size_recent"],
        best_size=CFG["pool_size_best"],
    )
    buffer = SequenceRolloutBuffer(num_envs=num_envs)
    envs = [base.BomberEnv(max_steps=500) for _ in range(num_envs)]

    episode = 0
    first_history = deque(maxlen=100)
    bombs_per_episode_history = deque(maxlen=100)
    valuable_bomb_ratio_history = deque(maxlen=100)
    useless_bomb_ratio_history = deque(maxlen=100)
    no_escape_bomb_ratio_history = deque(maxlen=100)
    danger_steps_history = deque(maxlen=100)
    unique_tiles_history = deque(maxlen=100)
    repeat_position_rate_history = deque(maxlen=100)

    env_obs = []
    env_agent_ids = []
    env_opponents = []
    env_alive_masks = []
    env_death_orders = []
    env_episode_ctxs = []
    env_episode_starts = [True for _ in range(num_envs)]
    for env in envs:
        obs, agent_id, opponents, alive_mask, death_order, episode_ctx = base.start_episode(env, pool, global_step)
        env_obs.append(obs)
        env_agent_ids.append(agent_id)
        env_opponents.append(opponents)
        env_alive_masks.append(alive_mask)
        env_death_orders.append(death_order)
        env_episode_ctxs.append(episode_ctx)
    actor_h, actor_c = model.get_initial_actor_state(num_envs, DEVICE)
    bc_reference_state = None if bc_reference_model is None else bc_reference_model.get_initial_state(num_envs, DEVICE)
    actor_trainable = actor_trainable_now(global_step, actor_initialized, stage_warmup_until)
    active_stage_name = base.get_stage(global_step)["name"]
    t_start = time.time()

    if CFG.get("eval_only", False):
        model.eval()
        eval_stats = evaluate_policy(model, current_step=global_step)
        print(
            f"Eval only | score {eval_stats['score']:.3f} | AvgRank {eval_stats['avg_rank']:.2f} | "
            f"R0/1/2/3 {eval_stats['rank0_rate']:.0%}/{eval_stats['rank1_rate']:.0%}/"
            f"{eval_stats['rank2_rate']:.0%}/{eval_stats['rank3_rate']:.0%} | "
            f"UF {eval_stats['unique_first_rate']:.0%}"
        )
        print(
            f"  EM score/rank {eval_stats['easy_medium']['avg_points']:.3f}/{eval_stats['easy_medium']['avg_rank']:.2f} | "
            f"R0/1/2/3 {eval_stats['easy_medium']['rank0_rate']:.0%}/{eval_stats['easy_medium']['rank1_rate']:.0%}/"
            f"{eval_stats['easy_medium']['rank2_rate']:.0%}/{eval_stats['easy_medium']['rank3_rate']:.0%} | "
            f"VB/Repeat {eval_stats['easy_medium']['valuable_bomb_ratio']:.2%}/{eval_stats['easy_medium']['repeat_position_rate']:.2%}"
        )
        print(
            f"  Hard score/rank {eval_stats['hard']['avg_points']:.3f}/{eval_stats['hard']['avg_rank']:.2f} | "
            f"R0/1/2/3 {eval_stats['hard']['rank0_rate']:.0%}/{eval_stats['hard']['rank1_rate']:.0%}/"
            f"{eval_stats['hard']['rank2_rate']:.0%}/{eval_stats['hard']['rank3_rate']:.0%} | "
            f"VB/Repeat {eval_stats['hard']['valuable_bomb_ratio']:.2%}/{eval_stats['hard']['repeat_position_rate']:.2%}"
        )
        return

    print(f"\n{'=' * 60}")
    print("Training Recurrent PPO Agent V6 - BC Initialized")
    print(
        f"Device: {DEVICE} | Total steps: {CFG['total_steps']:,} | "
        f"n_steps: {CFG['n_steps']:,} | seq_len: {CFG['seq_len']} | batch seq: {CFG['batch_size_sequences']}"
    )
    print(f"{'=' * 60}\n")

    while global_step < CFG["total_steps"]:
        buffer.reset()
        model.eval()
        stage = base.get_stage(global_step)

        for _ in range(rollout_env_steps):
            stage = base.get_stage(global_step)
            if stage["name"] != active_stage_name:
                prev_stage_name = active_stage_name
                active_stage_name = stage["name"]
                stage_warmup_until = max(stage_warmup_until, global_step + CFG["stage_actor_warmup_steps"])
                print(
                    f"Stage changed {prev_stage_name} -> {active_stage_name} at step {global_step:,}; "
                    f"critic-only warm-up until step {stage_warmup_until:,}"
                )

            should_train_actor = actor_trainable_now(global_step, actor_initialized, stage_warmup_until)
            if should_train_actor != actor_trainable:
                actor_trainable = should_train_actor
                model.set_actor_trainable(actor_trainable)
                print(f"Actor trainable switched to {actor_trainable} at step {global_step:,}")

            step_current = global_step + 1
            map_feats_cpu = []
            aux_feats_cpu = []
            action_masks_cpu = []
            for env_idx in range(num_envs):
                _canonical_obs, map_feat, aux_feat, action_mask = prepare_policy_inputs(
                    env_obs[env_idx],
                    env_agent_ids[env_idx],
                    current_step=step_current,
                    warmup_steps=CFG["mask_warmup_steps"],
                    value_bomb_mask_steps=VALUE_BOMB_MASK_STEPS,
                    eval_mode=False,
                )
                map_feats_cpu.append(map_feat)
                aux_feats_cpu.append(aux_feat)
                action_masks_cpu.append(action_mask)
            map_feat_batch = torch.stack(map_feats_cpu, dim=0).to(DEVICE)
            aux_feat_batch = torch.stack(aux_feats_cpu, dim=0).to(DEVICE)
            action_mask_batch = torch.stack(action_masks_cpu, dim=0).to(DEVICE)
            episode_start_tensor = torch.tensor(env_episode_starts, device=DEVICE, dtype=torch.float32)

            with torch.no_grad():
                rollout_actor_state = (actor_h.clone(), actor_c.clone())
                action, log_prob, value, next_actor_state = model.act_step(
                    map_feat_batch,
                    aux_feat_batch,
                    action_mask=action_mask_batch,
                    state=(actor_h, actor_c),
                    deterministic=False,
                    episode_start=episode_start_tensor,
                )
                if bc_reference_model is not None:
                    bc_teacher_action, bc_reference_state = bc_reference_model.act_step(
                        map_feat_batch,
                        aux_feat_batch,
                        action_mask=action_mask_batch,
                        state=bc_reference_state,
                        deterministic=True,
                        episode_start=episode_start_tensor,
                    )
                if bc_reference_model is None:
                    bc_teacher_action = action.clone()

            next_actor_h, next_actor_c = next_actor_state
            next_bc_h = None if bc_reference_state is None else bc_reference_state[0]
            next_bc_c = None if bc_reference_state is None else bc_reference_state[1]

            for env_idx in range(num_envs):
                canonical_action = int(action[env_idx].item())
                agent_id = env_agent_ids[env_idx]
                env_action = to_env_action(canonical_action, agent_id)
                actions = [0] * 4
                actions[agent_id] = env_action
                for opponent_id, opponent in env_opponents[env_idx].items():
                    try:
                        actions[opponent_id] = int(opponent.act(env_obs[env_idx]))
                    except Exception:
                        actions[opponent_id] = 0

                prev_obs = base.clone_obs(env_obs[env_idx])
                prev_stats = base.clone_stats(envs[env_idx].players[agent_id])
                env_obs[env_idx], terminated, truncated = envs[env_idx].step(actions)
                done = terminated or truncated
                base.record_deaths(envs[env_idx].players, env_alive_masks[env_idx], env_death_orders[env_idx])

                agent_rank = None
                if done:
                    ranks = base.compute_competition_ranks(
                        envs[env_idx].players,
                        env_death_orders[env_idx],
                        env_alive_masks[env_idx],
                    )
                    agent_rank = ranks[agent_id]

                curr_stats = base.clone_stats(envs[env_idx].players[agent_id])
                reward = base.compute_reward(
                    prev_obs,
                    env_obs[env_idx],
                    prev_stats,
                    curr_stats,
                    agent_id,
                    done,
                    agent_rank,
                    env_episode_ctxs[env_idx],
                    stage,
                    canonical_action=canonical_action,
                )

                buffer.push(
                    env_idx,
                    map_feats_cpu[env_idx],
                    aux_feats_cpu[env_idx],
                    action_masks_cpu[env_idx],
                    (
                        rollout_actor_state[0][env_idx].detach().cpu().clone(),
                        rollout_actor_state[1][env_idx].detach().cpu().clone(),
                    ),
                    canonical_action,
                    int(bc_teacher_action[env_idx].item()),
                    float(log_prob[env_idx].item()),
                    reward,
                    float(value[env_idx].item()),
                    float(done),
                    float(env_episode_starts[env_idx]),
                )

                env_episode_starts[env_idx] = False

                if done:
                    episode += 1
                    first_history.append(1.0 if agent_rank == 0 else 0.0)
                    episode_metrics = base.summarize_episode_metrics(env_episode_ctxs[env_idx], curr_stats)
                    bombs_per_episode_history.append(episode_metrics["bombs_per_episode"])
                    valuable_bomb_ratio_history.append(episode_metrics["valuable_bomb_ratio"])
                    useless_bomb_ratio_history.append(episode_metrics["useless_bomb_ratio"])
                    no_escape_bomb_ratio_history.append(episode_metrics["no_escape_bomb_ratio"])
                    danger_steps_history.append(episode_metrics["danger_steps_per_episode"])
                    unique_tiles_history.append(episode_metrics["unique_tiles_visited"])
                    repeat_position_rate_history.append(episode_metrics["repeat_position_rate"])

                    (
                        env_obs[env_idx],
                        env_agent_ids[env_idx],
                        env_opponents[env_idx],
                        env_alive_masks[env_idx],
                        env_death_orders[env_idx],
                        env_episode_ctxs[env_idx],
                    ) = base.start_episode(envs[env_idx], pool, global_step)
                    next_actor_h[env_idx].zero_()
                    next_actor_c[env_idx].zero_()
                    if next_bc_h is not None and next_bc_c is not None:
                        next_bc_h[env_idx].zero_()
                        next_bc_c[env_idx].zero_()
                    env_episode_starts[env_idx] = True

            actor_h, actor_c = next_actor_h, next_actor_c
            if next_bc_h is not None and next_bc_c is not None:
                bc_reference_state = (next_bc_h, next_bc_c)
            global_step += num_envs

        model.train()
        model.set_actor_trainable(actor_trainable)
        sync_group_lrs(optimizer, global_step)

        with torch.no_grad():
            last_map_feats = []
            last_aux_feats = []
            for env_idx in range(num_envs):
                _, map_feat_last, aux_feat_last, _action_mask_last = prepare_policy_inputs(
                    env_obs[env_idx],
                    env_agent_ids[env_idx],
                    current_step=global_step,
                    warmup_steps=CFG["mask_warmup_steps"],
                    value_bomb_mask_steps=VALUE_BOMB_MASK_STEPS,
                    eval_mode=False,
                )
                last_map_feats.append(map_feat_last)
                last_aux_feats.append(aux_feat_last)
            last_values = (
                model.forward_value_step(
                    torch.stack(last_map_feats, dim=0).to(DEVICE),
                    torch.stack(last_aux_feats, dim=0).to(DEVICE),
                )
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
            )

        stats = ppo_update(
            model,
            optimizer,
            buffer,
            last_values,
            actor_trainable=actor_trainable,
            bc_coef=CFG["bc_coef"] if bc_reference_model is not None else 0.0,
        )

        first_rate = np.mean(first_history) if first_history else 0.0
        bombs_per_episode = np.mean(bombs_per_episode_history) if bombs_per_episode_history else 0.0
        valuable_bomb_ratio = np.mean(valuable_bomb_ratio_history) if valuable_bomb_ratio_history else 0.0
        useless_bomb_ratio = np.mean(useless_bomb_ratio_history) if useless_bomb_ratio_history else 0.0
        no_escape_bomb_ratio = np.mean(no_escape_bomb_ratio_history) if no_escape_bomb_ratio_history else 0.0
        danger_steps_per_episode = np.mean(danger_steps_history) if danger_steps_history else 0.0
        unique_tiles_visited = np.mean(unique_tiles_history) if unique_tiles_history else 0.0
        repeat_position_rate = np.mean(repeat_position_rate_history) if repeat_position_rate_history else 0.0
        elapsed = time.time() - t_start
        sps = global_step / max(elapsed, 1.0)

        print(
            f"Step {global_step:>8,} | Ep {episode:>5} | Stage {stage['name']} | "
            f"ActorTrainable {actor_trainable} | FirstRate {first_rate:.2%} | "
            f"Bombs {bombs_per_episode:.2f} | VB {valuable_bomb_ratio:.2%} | UB {useless_bomb_ratio:.2%} | "
            f"NEB {no_escape_bomb_ratio:.2%} | Danger {danger_steps_per_episode:.1f} | "
            f"Tiles {unique_tiles_visited:.1f} | Repeat {repeat_position_rate:.2%} | "
            f"PG {stats['pg_loss']:+.4f} | Val {stats['val_loss']:.4f} | BC {stats['bc_loss']:.4f} | "
            f"Ent {stats['entropy']:.3f} | KL {stats['approx_kl']:.4f} | "
            f"Clip {stats['clip_fraction']:.2%} | Ratio {stats['ratio_mean']:.3f}+/-{stats['ratio_std']:.3f} | "
            f"EV {stats['explained_variance']:.3f} | SPS {sps:.0f}"
        )

        if global_step - last_save_step >= CFG["save_every"]:
            last_save_step = global_step
            ckpt_path = Path(CFG["ckpt_dir"]) / f"model_step{global_step}.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "stage": stage["name"],
                    "actor_initialized": actor_initialized,
                    "stage_warmup_until": stage_warmup_until,
                    "best_eval_score": best_eval_score,
                },
                ckpt_path,
            )
            print(f"  -> Saved checkpoint: {ckpt_path}")

            pool.add_recent_checkpoint(model.state_dict())
            eval_stats = evaluate_policy(model, current_step=global_step)
            print(
                f"  -> Eval score {eval_stats['score']:.3f} | AvgRank {eval_stats['avg_rank']:.2f} | "
                f"R0/1/2/3 {eval_stats['rank0_rate']:.0%}/{eval_stats['rank1_rate']:.0%}/"
                f"{eval_stats['rank2_rate']:.0%}/{eval_stats['rank3_rate']:.0%} | "
                f"UF {eval_stats['unique_first_rate']:.0%}"
            )
            print(
                f"     EM score/rank {eval_stats['easy_medium']['avg_points']:.3f}/{eval_stats['easy_medium']['avg_rank']:.2f} | "
                f"R0/1/2/3 {eval_stats['easy_medium']['rank0_rate']:.0%}/{eval_stats['easy_medium']['rank1_rate']:.0%}/"
                f"{eval_stats['easy_medium']['rank2_rate']:.0%}/{eval_stats['easy_medium']['rank3_rate']:.0%} | "
                f"VB/Repeat {eval_stats['easy_medium']['valuable_bomb_ratio']:.2%}/{eval_stats['easy_medium']['repeat_position_rate']:.2%}"
            )
            print(
                f"     Hard score/rank {eval_stats['hard']['avg_points']:.3f}/{eval_stats['hard']['avg_rank']:.2f} | "
                f"R0/1/2/3 {eval_stats['hard']['rank0_rate']:.0%}/{eval_stats['hard']['rank1_rate']:.0%}/"
                f"{eval_stats['hard']['rank2_rate']:.0%}/{eval_stats['hard']['rank3_rate']:.0%} | "
                f"VB/Repeat {eval_stats['hard']['valuable_bomb_ratio']:.2%}/{eval_stats['hard']['repeat_position_rate']:.2%}"
            )

            if eval_stats["score"] >= best_eval_score:
                best_eval_score = eval_stats["score"]
                pool.add_best_checkpoint(model.state_dict())
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                        "stage": stage["name"],
                        "actor_initialized": actor_initialized,
                        "stage_warmup_until": stage_warmup_until,
                        "best_eval_score": best_eval_score,
                        "eval": eval_stats,
                    },
                    _HERE / "model.pth",
                )
                print(f"  -> New best v6 model! Eval score: {best_eval_score:.3f}")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "actor_initialized": actor_initialized,
            "stage_warmup_until": stage_warmup_until,
            "best_eval_score": best_eval_score,
        },
        _HERE / "model_last.pth",
    )


if __name__ == "__main__":
    args = parse_args()
    apply_cli_overrides(args)
    train()
