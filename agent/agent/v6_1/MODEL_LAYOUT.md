# v6_1 Model Layout

## Purpose

This folder is organized so that three workflows stay separate:

1. `submission/`
Contains the exact files used to package and submit the current competition agent.

2. `runtime_local/`
Contains the model used by `agent/agent/v6_1/agent.py` for local matches and visual tests.

3. `checkpoint_bank/`
Contains curated strong checkpoints kept stable for benchmarking, training opponents, and manual promotion.

## Important paths

- Current submission model:
  - `submission/model.pth`
  - currently sourced from step `1356800`

- Current local runtime model:
  - `runtime_local/model.pth`

- Curated strong checkpoints:
  - `checkpoint_bank/submission_latest/model_step1356800.pth`
  - `checkpoint_bank/leaderboard_best/model_step1331200.pth`

## Raw training outputs

- `checkpoints/`
General-purpose training outputs from the default PPO flow.

- `checkpoints_v6_1_fight_with_harder_enemy/`
Training outputs from the harder-opponent branch.

These raw folders are not treated as stable runtime targets. Promote a checkpoint into
`checkpoint_bank/` or `runtime_local/` when you want to keep using it intentionally.

## Recommended workflow

1. Train into one of the raw checkpoint folders.
2. Benchmark promising checkpoints with `eval_ppo_legacy.py`.
3. If a checkpoint is worth keeping:
   - copy it into `checkpoint_bank/...`
4. If you want it for local play:
   - copy it into `runtime_local/model.pth`
5. If you want it as the new submission:
   - copy it into `submission/model.pth`
