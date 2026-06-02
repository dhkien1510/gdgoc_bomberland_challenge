# V6 Training Instructions

This document describes the recommended workflow to train the `v6` agent:

1. collect BC data
2. analyze dataset quality
3. train the BC actor
4. evaluate BC behavior
5. collect DAgger-lite data
6. fine-tune with recurrent PPO

All commands below assume you run from the repo root:

```powershell
cd D:\other\CNTT\Bomberland-GDGoC-AI-Challenge
```

## Files

Main `v6` entrypoints:

- `agent/agent/v6/collect_dataset.py`
- `agent/agent/v6/analyze_dataset.py`
- `agent/agent/v6/train_bc.py`
- `agent/agent/v6/eval_bc.py`
- `agent/agent/v6/train_ppo.py`
- `agent/agent/v6/agent.py`

Artifacts:

- `agent/agent/v6/bc_actor.pth`: best BC actor
- `agent/agent/v6/model.pth`: best PPO model
- `agent/agent/v6/model_last.pth`: latest PPO model
- `agent/agent/v6/checkpoints/`: PPO checkpoints

## Phase 1: Smoke BC Dataset

Goal: validate the pipeline before collecting a large dataset.

Collect a small dataset for 4 scenario blocks:

```powershell
python agent/agent/v6/collect_dataset.py --num_episodes 200 --scenario farm --output_dir agent/agent/v6/bc_data_smoke/farm
python agent/agent/v6/collect_dataset.py --num_episodes 200 --scenario survive --output_dir agent/agent/v6/bc_data_smoke/survive
python agent/agent/v6/collect_dataset.py --num_episodes 200 --scenario pressure --output_dir agent/agent/v6/bc_data_smoke/pressure
python agent/agent/v6/collect_dataset.py --num_episodes 200 --scenario selfplay --output_dir agent/agent/v6/bc_data_smoke/selfplay
```

Analyze each block:

```powershell
python agent/agent/v6/analyze_dataset.py --data_dir agent/agent/v6/bc_data_smoke/farm
python agent/agent/v6/analyze_dataset.py --data_dir agent/agent/v6/bc_data_smoke/survive
python agent/agent/v6/analyze_dataset.py --data_dir agent/agent/v6/bc_data_smoke/pressure
python agent/agent/v6/analyze_dataset.py --data_dir agent/agent/v6/bc_data_smoke/selfplay
```

What to look for:

- `PLACE_BOMB` should usually stay around `5% - 15%`
- `danger states` should not be too low
- `valuable bomb states` should be meaningful, especially in `pressure`
- no obvious sign that almost everything is only movement

## Phase 2: Train BC Directly From Multiple Scenario Folders

`train_bc.py` can now read multiple `--data_dir` values and sample them using explicit `--scenario_weight`.

This is the preferred way to keep `selfplay` from dominating the dataset just because it has more samples.

## Phase 3: Train BC Smoke

Train a small BC model:

```powershell
python agent/agent/v6/train_bc.py `
  --data_dir `
    agent/agent/v6/bc_data_smoke/farm `
    agent/agent/v6/bc_data_smoke/survive `
    agent/agent/v6/bc_data_smoke/pressure `
    agent/agent/v6/bc_data_smoke/selfplay `
  --scenario_weight 0.17 0.28 0.30 0.25 `
  --epochs 5 --batch_size 32 --seq_len 64 --stride 32
```

BC training logs already report:

- `val_loss`
- `accuracy`
- `bomb_precision`
- `bomb_recall`
- `danger_accuracy`
- `valuable_accuracy`
- `illegal_pre_mask_rate`
- confusion matrix `6x6`

Minimum sanity checks before moving on:

- `bomb_recall` should not be extremely low
- `illegal_pre_mask_rate` should stay low
- confusion matrix should not collapse almost everything into movement-only behavior

## Phase 4: Evaluate BC Behavior

Run rollout evaluation instead of relying only on offline accuracy:

```powershell
python agent/agent/v6/eval_bc.py --checkpoint agent/agent/v6/bc_actor.pth --pool easy --num_matches 20
python agent/agent/v6/eval_bc.py --checkpoint agent/agent/v6/bc_actor.pth --pool hard --num_matches 20
python agent/agent/v6/eval_bc.py --checkpoint agent/agent/v6/bc_actor.pth --pool mixed --num_matches 20
```

Key behavior metrics:

- `bombs`
- `boxes`
- `items`
- `kills`
- `valuable_bomb_ratio`
- `no_escape`
- `danger`
- `tiles`
- `repeat`

Practical readiness checks before PPO:

- `bombs_per_episode >= 1.5`
- `no_escape_bomb_ratio < 5%`
- `repeat_position_rate < 25%`
- agent visibly places bombs, breaks boxes, escapes danger, and does not loop badly

Also run a direct local match:

```powershell
python scripts/participant/run_local_match.py --agent_paths agent/agent/v6 RandomAgent SimpleRuleAgent SmarterRuleAgent --num_episodes 10 --max_steps 500 --seed 42
```

Optional visual check:

```powershell
python scripts/participant/run_local_match.py --agent_paths agent/agent/v6 RandomAgent SimpleRuleAgent SmarterRuleAgent --num_episodes 3 --max_steps 500 --seed 42 --visualize true --autoplay true
```

## Phase 5: Main BC Dataset

Once smoke is stable, collect the main dataset:

```powershell
python agent/agent/v6/collect_dataset.py --num_episodes 3000 --scenario farm --output_dir agent/agent/v6/bc_data_main/farm
python agent/agent/v6/collect_dataset.py --num_episodes 3000 --scenario survive --output_dir agent/agent/v6/bc_data_main/survive
python agent/agent/v6/collect_dataset.py --num_episodes 2500 --scenario pressure --output_dir agent/agent/v6/bc_data_main/pressure
python agent/agent/v6/collect_dataset.py --num_episodes 2500 --scenario selfplay --output_dir agent/agent/v6/bc_data_main/selfplay
python agent/agent/v6/collect_dataset.py --num_episodes 2000 --scenario late --output_dir agent/agent/v6/bc_data_main/late
```

## Phase 6: Train BC Main

```powershell
python agent/agent/v6/train_bc.py `
  --data_dir `
    agent/agent/v6/bc_data_main/farm `
    agent/agent/v6/bc_data_main/survive `
    agent/agent/v6/bc_data_main/pressure `
    agent/agent/v6/bc_data_main/selfplay `
    agent/agent/v6/bc_data_main/late `
  --scenario_weight 0.16 0.27 0.27 0.22 0.08 `
  --epochs 10 --batch_size 64 --seq_len 64 --stride 32
```

Re-evaluate:

```powershell
python agent/agent/v6/eval_bc.py --checkpoint agent/agent/v6/bc_actor.pth --pool mixed --num_matches 30
```

Do not move to PPO unless BC behavior is already reasonable.

## Phase 7: DAgger-lite

After you have a decent `bc_actor.pth`, collect DAgger-lite data:

```powershell
python agent/agent/v6/collect_dataset.py --num_episodes 2000 --scenario dagger --bc_policy_ckpt agent/agent/v6/bc_actor.pth --output_dir agent/agent/v6/bc_data_main/dagger
```

Then include the new dagger folder as one more `--data_dir` and train BC again:

```powershell
python agent/agent/v6/train_bc.py `
  --data_dir `
    agent/agent/v6/bc_data_main/farm `
    agent/agent/v6/bc_data_main/survive `
    agent/agent/v6/bc_data_main/pressure `
    agent/agent/v6/bc_data_main/selfplay `
    agent/agent/v6/bc_data_main/late `
    agent/agent/v6/bc_data_main/dagger `
  --scenario_weight 0.15 0.24 0.25 0.20 0.06 0.10 `
  --epochs 10 --batch_size 64 --seq_len 64 --stride 32
```

The idea is:

- learner policy creates some off-distribution states
- tactical teacher still provides the target action labels
- this reduces BC distribution drift before PPO

## Phase 8: PPO Fine-Tuning

Only start PPO when BC already handles the basics:

- places bombs regularly
- escapes after bombing
- breaks boxes
- does not loop badly
- survives danger better than a random walk

Then run:

```powershell
python agent/agent/v6/train_ppo.py
```

Current PPO design:

- recurrent actor-critic
- actor initialized from `agent/agent/v6/bc_actor.pth`
- correct hidden-state replay for PPO chunks
- BC regularization term during PPO to reduce policy forgetting
- curriculum and tactical reward shaping inherited from `v5.2/v5.1`

## Notes on Scenarios

- `farm`: early farm, boxes, items, basic bombing
- `survive`: danger handling, bomb avoidance, survival pressure
- `pressure`: enemy chase and valuable bomb situations
- `selfplay`: one `TacticalRuleAgent` teacher seat against a mixed opponent block of `GeniusRuleAgent` `45%`, `SmarterRuleAgent` `35%`, and `SimpleRuleAgent` `20%`
- `late`: late-game and harder tactical states
- `dagger`: learner policy acts part of the time, teacher still labels

## Common Failure Signs

Do not continue to PPO if BC still shows these:

- high overall accuracy but almost no bombs in rollout
- obvious left-right or up-down loops
- frequent no-escape bombs
- repeated early deaths
- confusion matrix collapsed heavily toward movement-only outputs

## Recommended Order

```text
1. Smoke collect
2. Smoke analyze
3. Smoke BC train
4. Smoke BC eval
5. Main collect
6. Main BC train
7. DAgger-lite collect
8. BC retrain
9. PPO fine-tune
```

## Quick Start Summary

If you want the shortest viable path:

```powershell
python agent/agent/v6/collect_dataset.py --num_episodes 200 --scenario farm --output_dir agent/agent/v6/bc_data_smoke/farm
python agent/agent/v6/collect_dataset.py --num_episodes 200 --scenario survive --output_dir agent/agent/v6/bc_data_smoke/survive
python agent/agent/v6/collect_dataset.py --num_episodes 200 --scenario pressure --output_dir agent/agent/v6/bc_data_smoke/pressure
python agent/agent/v6/collect_dataset.py --num_episodes 200 --scenario selfplay --output_dir agent/agent/v6/bc_data_smoke/selfplay
python agent/agent/v6/train_bc.py --data_dir agent/agent/v6/bc_data_smoke/farm agent/agent/v6/bc_data_smoke/survive agent/agent/v6/bc_data_smoke/pressure agent/agent/v6/bc_data_smoke/selfplay --scenario_weight 0.17 0.28 0.30 0.25 --epochs 5 --batch_size 32
python agent/agent/v6/eval_bc.py --checkpoint agent/agent/v6/bc_actor.pth --pool mixed --num_matches 20
python agent/agent/v6/train_ppo.py
```

Do not skip the BC evaluation step. PPO is much more likely to help when BC already has usable behavior.
