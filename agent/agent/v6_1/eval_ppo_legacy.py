"""
Stable PPO checkpoint benchmark for v6_1 using the older easy/hard pools.
"""

from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import random
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent.parent
BASELINE_DIR = ROOT / "agent"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))
sys.modules.pop("model", None)
sys.modules.pop("_model_base", None)
sys.modules.pop("_model_v3_base", None)
sys.modules.pop("_train_base", None)

_MODEL_SPEC = importlib.util.spec_from_file_location("_v6_1_model_eval_legacy", _HERE / "model.py")
_MODEL = importlib.util.module_from_spec(_MODEL_SPEC)
assert _MODEL_SPEC.loader is not None
_MODEL_SPEC.loader.exec_module(_MODEL)

from box_farmer_agent import BoxFarmerAgent
from genius_rule_agent import GeniusRuleAgent
from random_agent import RandomAgent
from simple_rule_agent import SimpleRuleAgent
from smarter_rule_agent import SmarterRuleAgent
from tactical_rule_agent import TacticalRuleAgent
from engine.game import BomberEnv
import _train_base as base
from v6_submission_agent import V6SubmissionAgent

ACTION_PLACE_BOMB = _MODEL.ACTION_PLACE_BOMB
MASK_WARMUP_STEPS = _MODEL.MASK_WARMUP_STEPS
VALUE_BOMB_MASK_STEPS = _MODEL.VALUE_BOMB_MASK_STEPS
RecurrentActorCriticV6 = _MODEL.RecurrentActorCriticV6
can_hit_enemy_if_place = _MODEL.can_hit_enemy_if_place
count_boxes_if_place = _MODEL.count_boxes_if_place
nearest_valuable_bomb_spot_info = _MODEL.nearest_valuable_bomb_spot_info
prepare_policy_inputs = _MODEL.prepare_policy_inputs
to_env_action = _MODEL.to_env_action


def _load_checkpoint(path: str | Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class PPOAgentEval:
    def __init__(
        self,
        checkpoint_path: str | Path,
        agent_id: int,
        current_step: int = VALUE_BOMB_MASK_STEPS,
        deterministic: bool = True,
    ):  
        self.agent_id = int(agent_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_step = int(current_step)
        self.recent_positions = deque(maxlen=12)
        self.recent_actions = deque(maxlen=12)
        self.local_loop_steps = 0
        self.loop_breaker_triggers = 0
        self.model = RecurrentActorCriticV6()
        checkpoint = _load_checkpoint(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.state = self.model.get_initial_actor_state(1, self.device)
        self.episode_start = True
        self.deterministic = bool(deterministic)

    def reset(self):
        self.state = self.model.get_initial_actor_state(1, self.device)
        self.episode_start = True
        self.recent_positions.clear()
        self.recent_actions.clear()
        self.local_loop_steps = 0
        self.loop_breaker_triggers = 0

    def _is_looping(self) -> bool:
        if len(self.recent_positions) < 8:
            return False
        return len(set(list(self.recent_positions)[-8:])) <= 2

    def act(self, obs):
        self.current_step += 1
        my = obs["players"][self.agent_id]
        pos = (int(my[0]), int(my[1]))
        self.recent_positions.append(pos)
        is_looping = self._is_looping()
        if is_looping:
            self.local_loop_steps += 1

        canonical_obs, map_feat, aux_feat, action_mask = prepare_policy_inputs(
            obs,
            self.agent_id,
            current_step=max(self.current_step, VALUE_BOMB_MASK_STEPS),
            warmup_steps=MASK_WARMUP_STEPS,
            value_bomb_mask_steps=VALUE_BOMB_MASK_STEPS,
            eval_mode=False,
        )

        with torch.no_grad():
            action, self.state = self.model.get_action_inference(
                map_feat.unsqueeze(0).to(self.device),
                aux_feat.unsqueeze(0).to(self.device),
                deterministic=self.deterministic,
                action_mask=action_mask.unsqueeze(0).to(self.device),
                state=self.state,
                episode_start=torch.tensor([self.episode_start], device=self.device),
            )
        canonical_action = int(action)
        self.episode_start = False

        if is_looping:
            self.loop_breaker_triggers += 1
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
        return to_env_action(canonical_action, self.agent_id)

    def summarize_eval_metrics(self) -> dict:
        steps = max(self.current_step, 1)
        return {
            "repeat_position_rate": float(self.local_loop_steps) / steps,
            "loop_breaker_rate": float(self.loop_breaker_triggers) / steps,
            "loop_breaker_triggers": float(self.loop_breaker_triggers),
        }


def opponent_pool(name: str, suite_name: str = "legacy"):
    if name == "easy":
        return [SimpleRuleAgent, SmarterRuleAgent, BoxFarmerAgent]
    if name == "hard":
        if suite_name == "v6hard":
            return [V6SubmissionAgent, TacticalRuleAgent]
        return [GeniusRuleAgent, TacticalRuleAgent]
    if name == "mixed":
        if suite_name == "v6hard":
            return [SimpleRuleAgent, SmarterRuleAgent, BoxFarmerAgent, V6SubmissionAgent, TacticalRuleAgent]
        return [SimpleRuleAgent, SmarterRuleAgent, BoxFarmerAgent, GeniusRuleAgent, TacticalRuleAgent]
    return [RandomAgent, SimpleRuleAgent, BoxFarmerAgent, SmarterRuleAgent, TacticalRuleAgent]

def run_match(
    checkpoint_path: str,
    pool_name: str,
    num_matches: int,
    seed_base: int,
    stage_name: str,
    seed_offset: int = 0,
    suite_name: str = "legacy",
):
    total_points = 0.0
    total_first = 0.0
    total_unique_first = 0.0
    total_shared_first = 0.0
    total_rank = 0.0
    total_bombs = 0.0
    total_boxes = 0.0
    total_items = 0.0
    total_kills = 0.0
    total_danger = 0.0
    total_no_escape = 0.0
    total_tiles = 0.0
    total_repeat = 0.0
    total_revisit = 0.0
    total_loop_break = 0.0
    total_vb = 0.0
    opponents_pool = opponent_pool(pool_name, suite_name=suite_name)
    reward_stage = base.get_stage_by_name(stage_name)

    suite_rng = random.Random(seed_base + seed_offset)
    for match_idx in range(num_matches):
        seed = seed_base + seed_offset + match_idx
        rng = random.Random(seed)
        env = BomberEnv(max_steps=500, seed=seed)
        agent_id = suite_rng.randrange(4)
        obs = env.reset(seed=seed)

        learner = PPOAgentEval(checkpoint_path, agent_id)
        opponents = {
            player_id: rng.choice(opponents_pool)(player_id)
            for player_id in range(4)
            if player_id != agent_id
        }
        alive_mask = [bool(player.alive) for player in env.players]
        death_order = []
        episode_ctx = base.make_episode_context(obs, agent_id)

        done = False
        while not done:
            actions = [0] * 4
            if env.players[agent_id].alive:
                actions[agent_id] = learner.act(obs)
            for opponent_id, opponent in opponents.items():
                if env.players[opponent_id].alive:
                    actions[opponent_id] = int(opponent.act(obs))

            prev_obs = base.clone_obs(obs)
            prev_stats = base.clone_stats(env.players[agent_id])
            prev_all_stats = [base.clone_stats(player) for player in env.players]
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            base.record_deaths(env.players, alive_mask, death_order)
            curr_stats = base.clone_stats(env.players[agent_id])
            curr_all_stats = [base.clone_stats(player) for player in env.players]
            base.compute_reward(
                prev_obs,
                obs,
                prev_stats,
                curr_stats,
                agent_id,
                done,
                None,
                episode_ctx,
                reward_stage,
                canonical_action=None,
                all_prev_stats=prev_all_stats,
                all_curr_stats=curr_all_stats,
                current_step=env.current_step,
            )

        ranks = base.compute_competition_ranks(env.players, death_order, alive_mask)
        first_group_size = sum(1 for rank in ranks if rank == 0)
        final_stats = base.clone_stats(env.players[agent_id])
        metrics = base.summarize_episode_metrics(episode_ctx, final_stats)
        eval_metrics = learner.summarize_eval_metrics()

        total_points += base.RANK_TO_POINTS[ranks[agent_id]]
        total_first += float(ranks[agent_id] == 0)
        total_unique_first += float(ranks[agent_id] == 0 and first_group_size == 1)
        total_shared_first += float(ranks[agent_id] == 0 and first_group_size > 1)
        total_rank += float(ranks[agent_id])
        total_bombs += metrics["bombs_per_episode"]
        total_boxes += metrics["boxes_per_episode"]
        total_items += metrics["items_per_episode"]
        total_kills += metrics["kills_per_episode"]
        total_danger += metrics["danger_steps_per_episode"]
        total_no_escape += metrics["no_escape_bomb_ratio"]
        total_tiles += metrics["unique_tiles_visited"]
        total_repeat += eval_metrics["repeat_position_rate"]
        total_revisit += metrics["repeat_position_rate"]
        total_loop_break += eval_metrics["loop_breaker_rate"]
        total_vb += metrics["valuable_bomb_ratio"]

    denom = max(num_matches, 1)
    return {
        "matches": int(num_matches),
        "avg_points": total_points / denom,
        "avg_rank": total_rank / denom,
        "first_rate": total_first / denom,
        "unique_first_rate": total_unique_first / denom,
        "shared_first_rate": total_shared_first / denom,
        "bombs": total_bombs / denom,
        "boxes": total_boxes / denom,
        "items": total_items / denom,
        "kills": total_kills / denom,
        "valuable_bomb_ratio": total_vb / denom,
        "no_escape_bomb_ratio": total_no_escape / denom,
        "danger_steps": total_danger / denom,
        "tiles": total_tiles / denom,
        "repeat_position_rate": total_repeat / denom,
        "revisit_rate": total_revisit / denom,
        "loop_break_rate": total_loop_break / denom,
    }


def print_pool_summary(label: str, stats: dict):
    print(
        f"Pool {label} | points {stats['avg_points']:.3f} | first {stats['first_rate']:.2%} "
        f"(unique {stats['unique_first_rate']:.2%}, shared {stats['shared_first_rate']:.2%}) | "
        f"rank {stats['avg_rank']:.2f} | bombs {stats['bombs']:.2f} | "
        f"boxes {stats['boxes']:.2f} | items {stats['items']:.2f} | "
        f"kills {stats['kills']:.2f} | vb {stats['valuable_bomb_ratio']:.2%} | "
        f"no_escape {stats['no_escape_bomb_ratio']:.2%} | danger {stats['danger_steps']:.1f} | "
        f"tiles {stats['tiles']:.1f} | repeat {stats['repeat_position_rate']:.2%} | "
        f"revisit {stats['revisit_rate']:.2%} | loop_break {stats['loop_break_rate']:.2%}"
    )


def run_suite(
    checkpoint_path: str,
    easy_matches: int,
    hard_matches: int,
    seed_base: int,
    stage_name: str,
    suite_name: str = "legacy",
):
    easy = run_match(
        checkpoint_path=checkpoint_path,
        pool_name="easy",
        num_matches=easy_matches,
        seed_base=seed_base,
        stage_name=stage_name,
        seed_offset=0,
        suite_name=suite_name,
    )
    hard = run_match(
        checkpoint_path=checkpoint_path,
        pool_name="hard",
        num_matches=hard_matches,
        seed_base=seed_base,
        stage_name=stage_name,
        seed_offset=10_000,
        suite_name=suite_name,
    )
    total_matches = max(easy["matches"] + hard["matches"], 1)
    total_points = easy["avg_points"] * easy["matches"] + hard["avg_points"] * hard["matches"]
    total_rank = easy["avg_rank"] * easy["matches"] + hard["avg_rank"] * hard["matches"]
    total_first = easy["first_rate"] * easy["matches"] + hard["first_rate"] * hard["matches"]
    total_unique_first = easy["unique_first_rate"] * easy["matches"] + hard["unique_first_rate"] * hard["matches"]
    total_shared_first = easy["shared_first_rate"] * easy["matches"] + hard["shared_first_rate"] * hard["matches"]
    print(
        f"Eval suite ({suite_name}) | score {total_points / total_matches:.3f} | AvgRank {total_rank / total_matches:.2f} | "
        f"First {total_first / total_matches:.2%} | UF {total_unique_first / total_matches:.2%} | "
        f"SF {total_shared_first / total_matches:.2%}"
    )
    print_pool_summary("easy", easy)
    print_pool_summary("hard", hard)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pool", choices=["easy", "hard", "mixed", "suite"], default="suite")
    parser.add_argument("--suite", choices=["legacy", "v6hard"], default="legacy")
    parser.add_argument("--num_matches", type=int, default=20)
    parser.add_argument("--easy_matches", type=int, default=50)
    parser.add_argument("--hard_matches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stage_name",
        choices=[
            "farm_box_safe",
            "pressure_mid",
            "survive_hard",
            "selfplay_hard",
            "resource_control",
            "late_game_closer",
            "easy_item_taken",
        ],
        default="resource_control",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.pool == "suite":
        run_suite(
            checkpoint_path=args.checkpoint,
            easy_matches=args.easy_matches,
            hard_matches=args.hard_matches,
            seed_base=args.seed,
            stage_name=args.stage_name,
            suite_name=args.suite,
        )
    else:
        stats = run_match(
            args.checkpoint,
            args.pool,
            args.num_matches,
            args.seed,
            args.stage_name,
            suite_name=args.suite,
        )
        print_pool_summary(args.pool, stats)
