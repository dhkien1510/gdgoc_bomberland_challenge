"""
V4.1 PPO training: V3 curriculum with separate actor/critic networks.
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
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
sys.modules.pop("model", None)
sys.modules.pop("_model_base", None)
sys.modules.pop("_model_v3_base", None)
sys.modules.pop("_train_base", None)

import _train_base as base
from model import CNNActorCriticV4_1, MASK_WARMUP_STEPS, prepare_policy_inputs, to_env_action

CFG = copy.deepcopy(base.CFG)
CFG.update(
    {
        "entropy_coef": 0.0,
        "actor_warmup_steps": 100_000,
        "actor_init_path": str(_HERE.parent / "v3" / "model.pth"),
        "ckpt_dir": str(_HERE / "checkpoints"),
    }
)

DEVICE = base.DEVICE
print(f"Using device: {DEVICE}")


class ActiveOpponentPoolV4_1(base.ActiveOpponentPool):
    def _load_checkpoint_opponent(self, checkpoint, agent_id: int, current_step: int):
        model = self.model_factory()
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        return ModelOpponentV4_1(model, agent_id, current_step)


class ModelOpponentV4_1:
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
            eval_mode=base.mask_eval_mode(self.current_step),
        )
        return env_action


def actor_warmup_active(global_step: int, actor_initialized: bool) -> bool:
    return actor_initialized and global_step < CFG["actor_warmup_steps"]


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
                    eval_mode=base.mask_eval_mode(current_step),
                )

            for opponent_id, opponent in opponents.items():
                if env.players[opponent_id].alive:
                    actions[opponent_id] = int(opponent.act(obs))

            obs, terminated, truncated = env.step(actions)
            base.record_deaths(env.players, alive_mask, death_order)

        ranks = base.compute_competition_ranks(env.players, death_order, alive_mask)
        rank = ranks[agent_id]
        first_group_size = sum(1 for r in ranks if r == 0)
        return {
            "rank": rank,
            "points": base.RANK_TO_POINTS[rank],
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
        opponent_classes=[base.SimpleRuleAgent, base.SmarterRuleAgent, base.BoxFarmerAgent],
        num_matches=easy_medium_matches,
        seed_offset=0,
        seed_base=seed_base,
        current_step=current_step,
    )
    hard = evaluate_suite(
        model,
        opponent_classes=[base.GeniusRuleAgent, base.TacticalRuleAgent],
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


def ppo_update(model, optimizer, buffer, last_value, actor_trainable: bool):
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

            if actor_trainable:
                ratio = torch.exp(new_log_probs - mb_old_lp)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CFG["clip_eps"], 1 + CFG["clip_eps"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_term = CFG["entropy_coef"] * entropy.mean()
            else:
                pg_loss = torch.zeros((), device=DEVICE)
                entropy_term = torch.zeros((), device=DEVICE)

            val_loss = 0.5 * ((new_values - mb_ret) ** 2).mean()
            loss = pg_loss + CFG["value_coef"] * val_loss - entropy_term

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
            optimizer.step()

            total_pg_loss += float(pg_loss.item())
            total_val_loss += float(val_loss.item())
            total_entropy += float(entropy.mean().item())
            num_updates += 1

    return {
        "pg_loss": total_pg_loss / max(num_updates, 1),
        "val_loss": total_val_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
    }


def train():
    os.makedirs(CFG["ckpt_dir"], exist_ok=True)

    model = CNNActorCriticV4_1(num_actions=6).to(DEVICE)
    actor_init_path = Path(CFG["actor_init_path"])
    actor_initialized = False

    if actor_init_path.exists():
        result = model.load_actor_from_shared_checkpoint(str(actor_init_path))
        actor_initialized = bool(result["loaded_actor"])
        print(f"Actor init from {actor_init_path}: {actor_initialized}")
    else:
        print(f"Actor init checkpoint not found at {actor_init_path}, skipping actor-only warm-up")

    model.set_actor_trainable(not actor_warmup_active(0, actor_initialized))
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"], eps=1e-5)

    pool = ActiveOpponentPoolV4_1(
        model_factory=lambda: CNNActorCriticV4_1(num_actions=6),
        recent_size=CFG["pool_size_recent"],
        best_size=CFG["pool_size_best"],
    )
    buffer = base.RolloutBuffer(CFG["n_steps"], DEVICE)
    env = base.BomberEnv(max_steps=500)

    global_step = 0
    episode = 0
    best_eval_score = float("-inf")
    last_save_step = 0
    first_history = deque(maxlen=100)

    obs, agent_id, opponents, alive_mask, death_order, episode_ctx = base.start_episode(env, pool, global_step)

    print(f"\n{'=' * 60}")
    print("Training PPO Agent V4.1 - Separate Actor/Critic")
    print(f"Device: {DEVICE} | Total steps: {CFG['total_steps']:,}")
    print(f"{'=' * 60}\n")

    t_start = time.time()
    actor_trainable = not actor_warmup_active(global_step, actor_initialized)

    while global_step < CFG["total_steps"]:
        buffer.reset()
        model.eval()
        stage = base.get_stage(global_step)

        for _ in range(CFG["n_steps"]):
            global_step += 1
            stage = base.get_stage(global_step)

            should_train_actor = not actor_warmup_active(global_step, actor_initialized)
            if should_train_actor != actor_trainable:
                actor_trainable = should_train_actor
                model.set_actor_trainable(actor_trainable)
                print(f"Actor trainable switched to {actor_trainable} at step {global_step:,}")

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
                obs, agent_id, opponents, alive_mask, death_order, episode_ctx = base.start_episode(
                    env,
                    pool,
                    global_step,
                )

        model.train()
        model.set_actor_trainable(actor_trainable)
        base.sync_linear_lr(optimizer, global_step)

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

        stats = ppo_update(model, optimizer, buffer, last_value, actor_trainable=actor_trainable)

        first_rate = np.mean(first_history) if first_history else 0.0
        elapsed = time.time() - t_start
        sps = global_step / max(elapsed, 1.0)

        print(
            f"Step {global_step:>8,} | Ep {episode:>5} | "
            f"Stage {stage['name']} | ActorTrainable {actor_trainable} | "
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
                    "stage": stage["name"],
                    "first_rate": first_rate,
                    "actor_initialized": actor_initialized,
                },
                ckpt_path,
            )
            print(f"  -> Saved checkpoint: {ckpt_path}")

            pool.add_recent_checkpoint(model.state_dict())
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
                pool.add_best_checkpoint(model.state_dict())
                torch.save(
                    {
                        "model": model.state_dict(),
                        "global_step": global_step,
                        "stage": stage["name"],
                        "actor_initialized": actor_initialized,
                        "eval": eval_stats,
                    },
                    _HERE / "model.pth",
                )
                print(f"  -> New best v4.1 model! Dev score: {best_eval_score:.3f}")

    last_model_path = _HERE / "model_last.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "global_step": global_step,
            "actor_initialized": actor_initialized,
        },
        last_model_path,
    )

    if best_eval_score == float("-inf"):
        torch.save(
            {"model": model.state_dict(), "global_step": global_step, "actor_initialized": actor_initialized},
            _HERE / "model.pth",
        )

    print(f"\nTraining done! Best eval model saved to {_HERE / 'model.pth'}")
    print(f"Latest weights saved to {last_model_path}")


if __name__ == "__main__":
    train()
