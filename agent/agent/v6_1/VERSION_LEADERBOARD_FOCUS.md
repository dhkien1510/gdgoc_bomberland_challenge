# v6_1 Leaderboard-Focused Update

## Version intent

This update shifts `v6_1` a bit closer to the real leaderboard objective:

- avoid early `rank 3`
- stabilize `top 2`
- win timeout tie-breaks more often
- convert `1v1` into `rank 0` more reliably

## What changed

### Reward shaping

Updated in [agent/agent/v6_1/_train_base.py](/d:/other/CNTT/Bomberland-GDGoC-AI-Challenge/agent/agent/v6_1/_train_base.py):

- Added survival transition rewards:
  - `r_survive_to_top3`
  - `r_survive_to_top2`
- Added timeout rank shaping:
  - `timeout_rank_rewards`
- Increased separation between `rank 1` and `rank 2`
  - especially in `resource_control`, `late_game_closer`, `selfplay_hard`
- Increased FFA late-game emphasis on:
  - `box` tie-break
  - `item` tie-break
- Reduced pressure to over-commit in FFA by making disengagement slightly less bad

### Metrics

Training now tracks:

- `Top3`
- `Top2`

These should help you see whether the policy is getting more leaderboard-stable, not just more aggressive.

## Recommended training CLI

### Continue from latest submission-style checkpoint

```powershell
python agent/agent/v6_1/train_ppo.py `
  --resume_checkpoint agent/agent/v6_1/checkpoint_bank/submission_latest/model_step1356800.pth `
  --bc_reference_path agent/agent/v6_1/bc_actor.pth `
  --external_opponent_prob 0.35 `
  --external_opponent_paths "agent/agent/v6_1/checkpoint_bank/leaderboard_best/model_step1331200.pth;agent/agent/v6/v6_submission/model.pth" `
  --force_stage late_game_closer `
  --total_steps 1510400 `
  --save_every 25600 `
  --eval_easy_medium_matches 5 `
  --eval_hard_matches 5 `
  --eval_workers 1 `
  --num_envs 4 `
  --n_steps 1024
```

Notes:

- `1510400` means roughly `+153,600` steps from `1356800`.
- This run is aimed at improving `top 2` stability and late-game conversion.

### Continue from leaderboard-best checkpoint and train against stronger pool

```powershell
python agent/agent/v6_1/train_ppo.py `
  --resume_checkpoint agent/agent/v6_1/checkpoint_bank/leaderboard_best/model_step1331200.pth `
  --bc_reference_path agent/agent/v6_1/bc_actor.pth `
  --external_opponent_prob 0.40 `
  --external_opponent_paths "agent/agent/v6_1/checkpoint_bank/submission_latest/model_step1356800.pth;agent/agent/v6/v6_submission/model.pth" `
  --force_stage late_game_closer `
  --total_steps 1484800 `
  --save_every 25600 `
  --eval_easy_medium_matches 5 `
  --eval_hard_matches 5 `
  --eval_workers 1 `
  --num_envs 4 `
  --n_steps 1024
```

## Evaluation CLI

### Current train-time protocol

```powershell
python agent/agent/v6_1/train_ppo.py `
  --resume_checkpoint agent/agent/v6_1/checkpoint_bank/submission_latest/model_step1356800.pth `
  --bc_reference_path agent/agent/v6_1/bc_actor.pth `
  --eval_only 1 `
  --eval_easy_medium_matches 50 `
  --eval_hard_matches 50
```

Current eval pools in `train_ppo.py`:

- `easy_medium`: `SimpleRuleAgent`, `SmarterRuleAgent`, `BoxFarmerAgent`
- `hard`: `V6SubmissionAgent`, `TacticalRuleAgent`

### Legacy benchmark protocol

```powershell
python agent/agent/v6_1/eval_ppo_legacy.py `
  --checkpoint agent/agent/v6_1/checkpoint_bank/leaderboard_best/model_step1331200.pth `
  --pool suite `
  --suite legacy `
  --easy_matches 50 `
  --hard_matches 50 `
  --seed 42 `
  --stage_name resource_control
```

### Stronger benchmark protocol

```powershell
python agent/agent/v6_1/eval_ppo_legacy.py `
  --checkpoint agent/agent/v6_1/checkpoint_bank/submission_latest/model_step1356800.pth `
  --pool suite `
  --suite v6hard `
  --easy_matches 50 `
  --hard_matches 50 `
  --seed 42 `
  --stage_name resource_control
```

## How to read improvement

Prefer this order:

1. `AvgRank`
2. `rank3_rate`
3. `rank2_rate`
4. `rank0_rate`
5. timeout loss breakdown
6. item/capacity/radius deltas

If a checkpoint wins a bit less often but:

- dies late instead of early
- reaches `top 2` more often
- loses fewer timeout tie-breaks

then it can still be better for leaderboard climbing.
