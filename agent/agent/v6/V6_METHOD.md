# V6 Method Overview

This document summarizes the `v6` agent clearly from end to end:

1. overall goal of `v6`
2. BC + PPO LSTM pipeline
3. BC dataset design
4. BC training design
5. PPO fine-tuning design
6. important functions
7. `v6` folder structure and role of each file

---

## 1. Goal of V6

`v6` is a two-stage learning pipeline:

1. train a recurrent policy with **Behavior Cloning (BC)**
2. fine-tune that policy with **recurrent PPO**

The main idea is:

- BC gives the agent strong basic behavior quickly
- PPO then improves tactical decisions with reward-driven learning

Compared to `v5.x`, `v6` is the first version that fully introduces:

- expert dataset collection
- supervised imitation learning
- recurrent actor with LSTM
- recurrent PPO with hidden-state replay
- BC regularization during PPO to reduce forgetting

---

## 2. End-to-End Pipeline

The full `v6` workflow is:

```text
Rule agents -> BC dataset shards -> BC actor (CNN + LSTM) -> BC rollout evaluation
-> optional DAgger-lite data -> recurrent PPO fine-tuning -> final submission agent
```

More concretely:

1. `collect_dataset.py`
   - runs Bomberland matches with rule-based teachers/opponents
   - stores expert state-action pairs into `.npz` shards

2. `analyze_dataset.py`
   - checks action distribution, danger ratio, valuable bomb ratio, and step distribution

3. `train_bc.py`
   - trains a recurrent actor to imitate teacher actions

4. `eval_bc.py`
   - runs the BC actor in real matches before PPO

5. `collect_dataset.py --scenario dagger`
   - collects DAgger-lite data from learner-induced states with teacher labels

6. `train_ppo.py`
   - loads `bc_actor.pth`
   - fine-tunes with recurrent PPO

7. `agent.py`
   - inference entrypoint
   - loads `model.pth` if PPO is available
   - otherwise falls back to `bc_actor.pth`

---

## 3. BC Dataset Design

### 3.1 BC Data Philosophy

BC in `v6` is not trained on random gameplay.

It is trained on:

- strong heuristic rule-agent behavior
- canonicalized observations
- canonicalized actions
- explicit action masks
- tactical auxiliary features

The goal is to teach:

- movement
- box farming
- item collection
- danger escape
- useful bomb placement

before PPO starts changing policy.

### 3.2 Dataset Scenarios

BC dataset collection is scenario-based.

Current scenarios in `collect_dataset.py`:

- `farm`
  - basic movement, box breaking, item collection
- `survive`
  - danger handling and bomb avoidance
- `pressure`
  - more enemy pressure and more valuable bomb states
- `selfplay`
  - one `TacticalRuleAgent` teacher against a mixed opponent pool:
    - `GeniusRuleAgent 45%`
    - `SmarterRuleAgent 35%`
    - `SimpleRuleAgent 20%`
- `late`
  - later and harder tactical states
- `dagger`
  - learner acts part of the time, teacher still supplies labels

### 3.3 Teacher Design

The main teacher is `TacticalRuleAgent`.

Why:

- better than `Random` and `Simple`
- more stable than pure learner self-play
- gives useful tactical bomb placement and pathing behavior

Opponent diversity is used so the teacher does not only produce one narrow kind of state.

### 3.4 Canonical Observation and Canonical Action

This is one of the most important design decisions in `v6`.

`prepare_policy_inputs(...)` converts the environment state into a canonical perspective:

- each controlled agent sees itself as if it were in the same reference corner
- observations are flipped by `agent_id`
- actions are also remapped into canonical action space

This allows one policy to learn consistent behavior across all four spawn corners.

Important functions:

- `canonicalize_obs(...)`
- `to_canonical_action(...)`
- `to_env_action(...)`
- `prepare_policy_inputs(...)`

### 3.5 BC Mask Semantics

BC should not use the strict late-stage PPO tactical mask.

After fixes, BC collection uses a **permissive BC mask**:

- action mask should represent mostly physical validity and basic safety
- it should not over-constrain the teacher labels using late-stage tactical PPO rules

This matters because `train_bc.py` uses invalid-action regularization. If the stored mask is too strict, BC may learn to suppress bombs or other actions incorrectly.

### 3.6 What Each BC Sample Stores

Each dataset shard stores:

- `map_feats`
- `aux_feats`
- `action_masks`
- `actions`
- `dones`
- `agent_ids`
- `episode_ids`
- `steps`
- `scenario_ids`
- `danger_times`
- `valuable_states`
- `can_escape_if_place`

This is enough for:

- supervised action learning
- danger-focused diagnostics
- bomb-quality diagnostics
- sequence reconstruction by episode

### 3.7 Sequence Dataset

`bc_dataset.py` converts raw per-step arrays into sequence samples for the LSTM.

Key design:

- sequences of length `seq_len`
- sliding windows with `stride`
- grouped by episode
- each sequence tagged as:
  - `normal`
  - `danger`
  - `bomb`

Important fix already applied:

- `episode_starts[0] = sample.start == 0`

This avoids incorrectly resetting the LSTM at the start of every chunk.

---

## 4. BC Training Design

### 4.1 BC Model

The BC model is in `bc_model.py`.

Architecture:

```text
map features -> CNN spatial encoder
aux features -> concatenated with CNN output
fused features -> LSTMCell sequence core
hidden state -> actor head -> 6 action logits
```

Main classes:

- `SpatialEncoderV6`
- `CNNLSTMActorCore`
- `CNNLSTMBCActor`

### 4.2 Why LSTM

Bomberland is full-state, but short temporal memory still helps with:

- repeated local pathing choices
- bomb timing context
- short escape behavior
- sequential tactical commitment

`v6` therefore uses recurrent BC instead of purely feed-forward BC.

### 4.3 Multi-Directory Training

`train_bc.py` supports:

- multiple `--data_dir`
- multiple `--scenario_weight`

This means scenario balance is controlled directly in training, not by manually merging shards.

### 4.4 Current BC Objective

The current BC objective is:

```text
loss = raw_ce_coef * raw_ce
     + masked_ce_coef * masked_ce
     + illegal_action_coef * invalid_action_mass
```

Where:

- `raw_ce`
  - cross-entropy on raw logits
  - forces the policy itself to learn the expert action

- `masked_ce`
  - cross-entropy on masked logits
  - preserves alignment with valid action constraints

- `invalid_action_mass`
  - softmax probability mass placed on invalid actions
  - penalizes raw illegal-action preference without the old unstable margin formulation

This replaced the earlier approach that relied too heavily on masked logits and the old `illegal_action_margin_loss`.

### 4.5 Why the BC Objective Was Changed

Earlier BC versions had two major issues:

1. CE trained mostly on masked logits
   - the model could survive by depending on the mask instead of learning a good raw policy

2. `illegal_action_margin_loss`
   - could accidentally reinforce the model's favorite wrong valid action

The current objective was designed to reduce both issues.

### 4.6 BC Metrics

`train_bc.py` now logs more than plain accuracy:

- `accuracy`
- `raw_accuracy`
- `bomb_precision`
- `bomb_recall`
- `pred_bomb_rate`
- `true_bomb_rate`
- `danger_accuracy`
- `valuable_accuracy`
- `illegal_pre_mask_rate`
- `invalid_action_mass`
- `confusion_matrix`

These are critical because total accuracy alone is not enough in Bomberland.

### 4.7 Overfit Debug Mode

`train_bc.py` supports:

```text
--overfit_sequences N
```

Purpose:

- limit each scenario to a small number of train sequences
- test whether the model can overfit a tiny subset
- catch bugs in loss, masking, sequence handling, or labels

Important recent fixes:

- overfit mode now evaluates on the actual overfit train subset
- overfit mode disables the weighted sampler and uses direct shuffle

This makes overfit debugging meaningful.

---

## 5. BC Evaluation Before PPO

`eval_bc.py` runs the BC actor in real matches.

This is necessary because offline accuracy can be misleading.

BC is considered useful only if rollout behavior is already reasonable:

- places bombs regularly
- breaks boxes
- escapes after bombing
- does not loop badly
- does not die immediately in danger

Important rollout metrics:

- `bombs_per_episode`
- `boxes_per_episode`
- `items_per_episode`
- `kills_per_episode`
- `valuable_bomb_ratio`
- `no_escape_bomb_ratio`
- `danger_steps_per_episode`
- `unique_tiles_visited`
- `repeat_position_rate`

---

## 6. DAgger-lite Design

`collect_dataset.py --scenario dagger` is a lightweight DAgger stage.

Mechanism:

- learner BC policy acts part of the time
- teacher still provides labels
- learner-induced off-distribution states are recorded

This reduces distribution shift before PPO.

It is not full DAgger, but it captures the main benefit:

- the learner sees states caused by its own imperfect behavior
- but still gets expert labels

---

## 7. PPO Fine-Tuning Design

### 7.1 PPO Model

The PPO model is in `model.py`.

Main class:

- `RecurrentActorCriticV6`

Architecture:

- actor:
  - `CNNLSTMActorCore` from BC model
  - actor head
- critic:
  - separate CNN + MLP value network

So the actor is recurrent, and the critic is separate.

### 7.2 Why Separate Actor/Critic

This inherits the lessons from `v4.1` and later versions:

- critic gradients should not destroy the actor representation
- actor should preserve BC knowledge better

### 7.3 PPO Initialization

`train_ppo.py` initializes the actor from `bc_actor.pth`.

This means PPO starts from:

- already-good movement
- already-good danger response
- already-good bomb basics

instead of starting from random policy.

### 7.4 Recurrent PPO Buffer

PPO is recurrent, so the rollout buffer must store sequence state correctly.

The recurrent PPO pipeline in `train_ppo.py` includes:

- action sequences
- log-probs
- rewards
- values
- dones
- action masks
- hidden states at each step / chunk start

Important fix already applied:

- PPO recomputes logits with the correct hidden state from rollout chunks
- not from zero hidden state

This matters for stable PPO ratios.

### 7.5 BC Regularization During PPO

PPO in `v6` also includes BC regularization:

- a frozen BC reference policy is loaded
- PPO actor is softly regularized toward BC behavior

Purpose:

- reduce policy forgetting
- prevent PPO from quickly destroying useful BC behavior

### 7.6 Curriculum and Reward

PPO in `v6` inherits the tactical curriculum and reward shaping lineage from `v5.2`.

That includes:

- phase-based curriculum
- tactical bomb rewards
- danger penalties
- loop penalties
- rank-aware competition rewards

So `v6` changes the learning pipeline more than the game reward philosophy.

---

## 8. Important Functions and What They Do

### In preprocessing / features

- `prepare_policy_inputs(...)`
  - main entrypoint to build canonical observation, map features, aux features, and mask

- `canonicalize_obs(...)`
  - flips board state into canonical perspective

- `to_canonical_action(...)`
  - maps environment action to canonical action

- `to_env_action(...)`
  - maps canonical action back to engine action

- `encode_obs(...)`
  - builds spatial map tensor

- `encode_aux(...)`
  - builds auxiliary tactical vector

### In danger / tactical reasoning

- `build_bomb_state(...)`
  - builds bomb timing and blast information

- `current_tile_danger_time(...)`
  - time until current tile becomes dangerous

- `has_escape_after_placing_bomb(...)`
  - safety check for bomb placement

- `count_boxes_if_place(...)`
  - number of boxes that would be hit by a bomb

- `can_hit_enemy_if_place(...)`
  - whether a bomb could hit an enemy

- `nearest_valuable_bomb_spot_info(...)`
  - tactical navigation target for strong bomb spots

- `valid_action_mask(...)`
  - builds action mask given current state and safety rules

### In BC dataset / sequence handling

- `load_bc_shards(...)`
  - load and merge `.npz` shard files

- `split_episode_keys(...)`
  - split episodes into train/val/test groups

- `BCSequenceDataset`
  - sequence dataset for LSTM BC training

### In BC loss / training

- `ce_from_logits(...)`
  - CE helper on sequence logits

- `mask_logits(...)`
  - apply action mask to logits

- `invalid_action_mass_loss(...)`
  - penalize probability mass on invalid actions

- `evaluate_loader(...)`
  - BC validation and diagnostics

### In PPO recurrent training

- `SequenceRolloutBuffer`
  - recurrent rollout storage

- `ppo_update(...)`
  - PPO update with recurrent state replay and BC regularization

---

## 9. V6 Folder Structure

Current important files in `agent/agent/v6`:

### Core runtime

- `agent.py`
  - inference entrypoint
  - loads PPO model if available, otherwise BC actor

- `model.py`
  - recurrent PPO actor-critic

- `train_ppo.py`
  - recurrent PPO training script

### BC pipeline

- `collect_dataset.py`
  - collect BC dataset shards from scenario-based expert matches

- `analyze_dataset.py`
  - dataset statistics and sanity checks

- `bc_dataset.py`
  - sequence dataset builder for BC

- `bc_model.py`
  - CNN + LSTM BC actor model

- `train_bc.py`
  - BC training script

- `eval_bc.py`
  - BC rollout evaluation

- `export_bc_agent.py`
  - helper to export a BC-only agent package/checkpoint if needed

### Shared compatibility / reward helpers

- `_model_base.py`
  - shared feature, mask, tactical reasoning, and canonicalization helpers

- `_model_v3_base.py`
  - compatibility exports for the v3/v5.x stack

- `_train_base.py`
  - reward shaping, eval helpers, episode metrics, ranking helpers

### Documentation / notes

- `intructions.md`
  - practical training commands and workflow

- `V6_METHOD.md`
  - this high-level technical explanation

- `log.md`
  - optional user log / experiment notes

### Artifacts generated by training

- `bc_actor.pth`
  - best BC checkpoint

- `model.pth`
  - best PPO checkpoint

- `model_last.pth`
  - latest PPO checkpoint

- `checkpoints/`
  - intermediate PPO checkpoints

- `bc_data_smoke*/`, `bc_data_main*/`
  - BC datasets

---

## 10. Practical Training Order

Recommended order:

```text
1. collect smoke BC data
2. analyze smoke BC data
3. run BC overfit debug
4. run smoke BC training
5. evaluate BC in rollout
6. collect main BC data
7. train main BC
8. collect DAgger-lite data
9. retrain BC
10. fine-tune PPO
```

The key rule is:

- do not move to PPO just because BC loss decreases
- move to PPO only when BC rollout behavior is already usable

---

## 11. Current Known Debug Priorities

The most important debugging priorities for `v6` BC are:

1. verify BC can truly overfit a tiny train subset
2. ensure bomb prediction does not collapse to:
   - bomb spam
   - or zero-bomb policy
3. ensure action masks used in BC are permissive enough
4. ensure canonical action mapping is consistent
5. ensure rollout behavior, not just offline accuracy, improves

If BC overfit still fails after the current fixes, the next most likely suspects are:

- deeper canonical action / perspective mismatch
- insufficient recurrent burn-in for LSTM chunks
- remaining label ambiguity that requires extra auxiliary supervision or simplification

---

## 12. Summary

`v6` is a recurrent imitation-learning-plus-RL stack:

- BC teaches the agent basic competent Bomberland behavior
- DAgger-lite reduces distribution shift
- recurrent PPO improves tactics while preserving BC knowledge

The success of `v6` depends most on:

- correct canonicalization
- good BC mask semantics
- stable recurrent sequence handling
- strong BC rollout behavior before PPO

That is the core method behind the entire `v6` design.
