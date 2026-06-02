"""
Recurrent PPO fine-tuning for the v6 BC+PPO agent.
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
from model import RecurrentActorCriticV6, VALUE_BOMB_MASK_STEPS, prepare_policy_inputs, to_env_action

CFG = copy.deepcopy(base.CFG)
CFG.update(
    {
        "entropy_coef": 0.001,
        "n_steps": 1024,
        "seq_len": 64,
        "batch_size_sequences": 16,
        "ppo_epochs": 4,
        "bc_coef": 0.1,
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


def _load_checkpoint(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class SequenceRolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.map_feats = []
        self.aux_feats = []
        self.action_masks = []
        self.actor_h = []
        self.actor_c = []
        self.actions = []
        self.bc_teacher_actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.episode_starts = []

    def push(
        self,
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
        self.map_feats.append(map_feat.clone())
        self.aux_feats.append(aux_feat.clone())
        self.action_masks.append(action_mask.clone())
        self.actor_h.append(actor_state[0].detach().cpu().squeeze(0).clone())
        self.actor_c.append(actor_state[1].detach().cpu().squeeze(0).clone())
        self.actions.append(int(action))
        self.bc_teacher_actions.append(int(bc_teacher_action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(float(done))
        self.episode_starts.append(float(episode_start))

    def compute_gae(self, last_value, gamma, gae_lambda):
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        last_gae = 0.0
        next_value = float(last_value)
        for t in reversed(range(len(self.rewards))):
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
            next_value = self.values[t]
        returns = advantages + np.asarray(self.values, dtype=np.float32)
        return advantages, returns

    def make_windows(self, seq_len: int):
        return [(start, min(start + seq_len, len(self.actions))) for start in range(0, len(self.actions), seq_len)]

    def build_sequence_batch(self, windows, advantages, returns, device):
        batch_size = len(windows)
        seq_len = max(end - start for start, end in windows)
        map_shape = self.map_feats[0].shape
        aux_dim = self.aux_feats[0].shape[0]
        action_dim = self.action_masks[0].shape[0]

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
        hidden_size = self.actor_h[0].shape[0]
        init_h_batch = torch.zeros((batch_size, hidden_size), dtype=torch.float32, device=device)
        init_c_batch = torch.zeros((batch_size, hidden_size), dtype=torch.float32, device=device)

        for batch_idx, (start, end) in enumerate(windows):
            length = end - start
            init_h_batch[batch_idx] = self.actor_h[start]
            init_c_batch[batch_idx] = self.actor_c[start]
            for t in range(length):
                src = start + t
                map_batch[batch_idx, t] = self.map_feats[src]
                aux_batch[batch_idx, t] = self.aux_feats[src]
                mask_batch[batch_idx, t] = self.action_masks[src]
                action_batch[batch_idx, t] = self.actions[src]
                bc_teacher_action_batch[batch_idx, t] = self.bc_teacher_actions[src]
                old_log_prob_batch[batch_idx, t] = self.log_probs[src]
                adv_batch[batch_idx, t] = advantages[src]
                ret_batch[batch_idx, t] = returns[src]
                loss_mask[batch_idx, t] = 1.0
                episode_start_batch[batch_idx, t] = self.episode_starts[src]

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
    for candidate in CFG["actor_init_candidates"]:
        path = Path(candidate)
        if path.exists():
            return path
    return None


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
    return {
        "score": total_points / max(total_matches, 1),
        "easy_medium": easy_medium,
        "hard": hard,
    }


def ppo_update(
    model,
    optimizer,
    buffer: SequenceRolloutBuffer,
    last_value: float,
    actor_trainable: bool,
    bc_coef: float,
):
    advantages, returns = buffer.compute_gae(last_value, CFG["gamma"], CFG["gae_lambda"])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    windows = buffer.make_windows(CFG["seq_len"])

    total_pg_loss = 0.0
    total_val_loss = 0.0
    total_bc_loss = 0.0
    total_entropy = 0.0
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
            if actor_trainable:
                ratio = torch.exp(new_log_probs - batch["old_log_probs"])
                pg_loss1 = -batch["advantages"] * ratio
                pg_loss2 = -batch["advantages"] * torch.clamp(ratio, 1 - CFG["clip_eps"], 1 + CFG["clip_eps"])
                pg_loss = torch.max(pg_loss1, pg_loss2)[valid].mean()
                entropy_term = CFG["entropy_coef"] * entropy[valid].mean()
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
            num_updates += 1

    return {
        "pg_loss": total_pg_loss / max(num_updates, 1),
        "val_loss": total_val_loss / max(num_updates, 1),
        "bc_loss": total_bc_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
    }


def train():
    os.makedirs(CFG["ckpt_dir"], exist_ok=True)

    model = RecurrentActorCriticV6().to(DEVICE)
    actor_init_path = resolve_actor_init_path()
    actor_initialized = False
    bc_reference_path = _HERE / "bc_actor.pth"
    bc_reference_model = None
    if actor_init_path is not None:
        result = model.load_actor_from_checkpoint(str(actor_init_path))
        actor_initialized = bool(result["loaded_actor"])
        print(f"Actor init from {actor_init_path}: {actor_initialized}")
    else:
        print("Actor init checkpoint not found, starting PPO actor from scratch")
    if bc_reference_path.exists():
        checkpoint = _load_checkpoint(bc_reference_path, map_location=DEVICE)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        bc_reference_model = CNNLSTMBCActor().to(DEVICE)
        bc_reference_model.load_state_dict(state_dict)
        bc_reference_model.eval()
        print(f"Loaded frozen BC reference from {bc_reference_path}")
    else:
        print("BC reference checkpoint not found, disabling BC regularization during PPO")

    stage_warmup_until = 0
    model.set_actor_trainable(actor_trainable_now(0, actor_initialized, stage_warmup_until))
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"], eps=1e-5)
    pool = ActiveOpponentPoolV6(
        model_factory=lambda: RecurrentActorCriticV6(),
        recent_size=CFG["pool_size_recent"],
        best_size=CFG["pool_size_best"],
    )
    buffer = SequenceRolloutBuffer()
    env = base.BomberEnv(max_steps=500)

    global_step = 0
    episode = 0
    best_eval_score = float("-inf")
    last_save_step = 0
    first_history = deque(maxlen=100)
    bombs_per_episode_history = deque(maxlen=100)
    valuable_bomb_ratio_history = deque(maxlen=100)
    useless_bomb_ratio_history = deque(maxlen=100)
    no_escape_bomb_ratio_history = deque(maxlen=100)
    danger_steps_history = deque(maxlen=100)
    unique_tiles_history = deque(maxlen=100)
    repeat_position_rate_history = deque(maxlen=100)

    obs, agent_id, opponents, alive_mask, death_order, episode_ctx = base.start_episode(env, pool, global_step)
    actor_state = model.get_initial_actor_state(1, DEVICE)
    bc_reference_state = None if bc_reference_model is None else bc_reference_model.get_initial_state(1, DEVICE)
    episode_start = True
    actor_trainable = actor_trainable_now(global_step, actor_initialized, stage_warmup_until)
    active_stage_name = base.get_stage(global_step)["name"]
    t_start = time.time()

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

        for _ in range(CFG["n_steps"]):
            global_step += 1
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

            _, map_feat, aux_feat, action_mask = prepare_policy_inputs(
                obs,
                agent_id,
                current_step=global_step,
                warmup_steps=CFG["mask_warmup_steps"],
                value_bomb_mask_steps=VALUE_BOMB_MASK_STEPS,
                eval_mode=False,
            )
            map_feat_batch = map_feat.unsqueeze(0).to(DEVICE)
            aux_feat_batch = aux_feat.unsqueeze(0).to(DEVICE)
            action_mask_batch = action_mask.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                rollout_actor_state = actor_state
                action, log_prob, value, next_actor_state = model.act_step(
                    map_feat_batch,
                    aux_feat_batch,
                    action_mask=action_mask_batch,
                    state=actor_state,
                    deterministic=False,
                    episode_start=torch.tensor([float(episode_start)], device=DEVICE),
                )
                if bc_reference_model is not None:
                    bc_teacher_action, bc_reference_state = bc_reference_model.act_step(
                        map_feat_batch,
                        aux_feat_batch,
                        action_mask=action_mask_batch,
                        state=bc_reference_state,
                        deterministic=True,
                        episode_start=torch.tensor([float(episode_start)], device=DEVICE),
                    )
                    bc_teacher_action = int(bc_teacher_action.item())
                else:
                    bc_teacher_action = int(action.item())

            canonical_action = int(action.item())
            env_action = to_env_action(canonical_action, agent_id)
            actions = [0] * 4
            actions[agent_id] = env_action
            for opponent_id, opponent in opponents.items():
                try:
                    actions[opponent_id] = int(opponent.act(obs))
                except Exception:
                    actions[opponent_id] = 0

            prev_obs = base.clone_obs(obs)
            prev_stats = base.clone_stats(env.players[agent_id])
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            base.record_deaths(env.players, alive_mask, death_order)

            agent_rank = None
            if done:
                ranks = base.compute_competition_ranks(env.players, death_order, alive_mask)
                agent_rank = ranks[agent_id]

            curr_stats = base.clone_stats(env.players[agent_id])
            reward = base.compute_reward(
                prev_obs,
                obs,
                prev_stats,
                curr_stats,
                agent_id,
                done,
                agent_rank,
                episode_ctx,
                stage,
                canonical_action=canonical_action,
            )

            buffer.push(
                map_feat.cpu(),
                aux_feat.cpu(),
                action_mask.cpu(),
                rollout_actor_state,
                canonical_action,
                bc_teacher_action,
                float(log_prob.item()),
                reward,
                float(value.item()),
                float(done),
                float(episode_start),
            )

            actor_state = next_actor_state
            episode_start = False

            if done:
                episode += 1
                first_history.append(1.0 if agent_rank == 0 else 0.0)
                episode_metrics = base.summarize_episode_metrics(episode_ctx, curr_stats)
                bombs_per_episode_history.append(episode_metrics["bombs_per_episode"])
                valuable_bomb_ratio_history.append(episode_metrics["valuable_bomb_ratio"])
                useless_bomb_ratio_history.append(episode_metrics["useless_bomb_ratio"])
                no_escape_bomb_ratio_history.append(episode_metrics["no_escape_bomb_ratio"])
                danger_steps_history.append(episode_metrics["danger_steps_per_episode"])
                unique_tiles_history.append(episode_metrics["unique_tiles_visited"])
                repeat_position_rate_history.append(episode_metrics["repeat_position_rate"])

                obs, agent_id, opponents, alive_mask, death_order, episode_ctx = base.start_episode(env, pool, global_step)
                actor_state = model.get_initial_actor_state(1, DEVICE)
                if bc_reference_model is not None:
                    bc_reference_state = bc_reference_model.get_initial_state(1, DEVICE)
                episode_start = True

        model.train()
        model.set_actor_trainable(actor_trainable)
        base.sync_linear_lr(optimizer, global_step)

        with torch.no_grad():
            _, map_feat_last, aux_feat_last, _action_mask_last = prepare_policy_inputs(
                obs,
                agent_id,
                current_step=global_step,
                warmup_steps=CFG["mask_warmup_steps"],
                value_bomb_mask_steps=VALUE_BOMB_MASK_STEPS,
                eval_mode=False,
            )
            last_value = float(
                model.forward_value_step(
                    map_feat_last.unsqueeze(0).to(DEVICE),
                    aux_feat_last.unsqueeze(0).to(DEVICE),
                ).item()
            )

        stats = ppo_update(
            model,
            optimizer,
            buffer,
            last_value,
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
            f"Ent {stats['entropy']:.3f} | SPS {sps:.0f}"
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
                },
                ckpt_path,
            )
            print(f"  -> Saved checkpoint: {ckpt_path}")

            pool.add_recent_checkpoint(model.state_dict())
            eval_stats = evaluate_policy(model, current_step=global_step)
            print(
                f"  -> Eval score {eval_stats['score']:.3f} | "
                f"EM VB/Repeat {eval_stats['easy_medium']['valuable_bomb_ratio']:.2%}/{eval_stats['easy_medium']['repeat_position_rate']:.2%} | "
                f"Hard VB/Repeat {eval_stats['hard']['valuable_bomb_ratio']:.2%}/{eval_stats['hard']['repeat_position_rate']:.2%}"
            )

            if eval_stats["score"] >= best_eval_score:
                best_eval_score = eval_stats["score"]
                pool.add_best_checkpoint(model.state_dict())
                torch.save(
                    {
                        "model": model.state_dict(),
                        "global_step": global_step,
                        "stage": stage["name"],
                        "actor_initialized": actor_initialized,
                        "stage_warmup_until": stage_warmup_until,
                        "eval": eval_stats,
                    },
                    _HERE / "model.pth",
                )
                print(f"  -> New best v6 model! Eval score: {best_eval_score:.3f}")

    torch.save(
        {
            "model": model.state_dict(),
            "global_step": global_step,
            "actor_initialized": actor_initialized,
            "stage_warmup_until": stage_warmup_until,
        },
        _HERE / "model_last.pth",
    )


if __name__ == "__main__":
    train()
