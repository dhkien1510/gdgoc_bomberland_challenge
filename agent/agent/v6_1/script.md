python scripts/participant/run_local_match.py `
  --agent_paths GeniusRuleAgent BoxFarmerAgent TacticalRuleAgent agent/agent/v6_1 `
  --num_episodes 1 `
  --max_steps 500 `
  --visualize 1

python agent/agent/v6_1/eval_ppo_legacy.py `
  --checkpoint agent/agent/v6_1/checkpoint_bank/leaderboard_best/model_step1331200.pth `
  --pool suite `
  --suite legacy `
  --easy_matches 50 `
  --hard_matches 50 `
  --seed 42 `
  --stage_name resource_control

python agent/agent/v6_1/eval_ppo_legacy.py `
  --checkpoint agent/agent/v6_1/checkpoint_bank/submission_latest/model_step1356800.pth `
  --pool suite `
  --suite v6hard `
  --easy_matches 50 `
  --hard_matches 50 `
  --seed 42 `
  --stage_name resource_control

python agent/agent/v6_1/train_ppo.py `
  --resume_checkpoint agent/agent/v6_1/checkpoint_bank/submission_latest/model_step1356800.pth `
  --bc_reference_path agent/agent/v6_1/bc_actor.pth `
  --external_opponent_prob 0.35 `
  --external_opponent_paths "agent/agent/v6_1/checkpoint_bank/leaderboard_best/model_step1331200.pth;agent/agent/v6/v6_submission/model.pth" `
  --force_stage late_game_closer `
  --total_steps 1459200 `
  --save_every 25600 `
  --eval_easy_medium_matches 5 `
  --eval_hard_matches 5 `
  --eval_workers 1 `
  --num_envs 4 `
  --n_steps 1024

python agent/agent/v6_1/train_ppo.py `
  --resume_checkpoint agent/agent/v6/checkpoints/phase_item_control_easy/model_step2528800.pth `
  --bc_reference_path "" `
  --bc_coef 0.0 `
  --resume_global_step_override 0 `
  --eval_only 1 `
  --eval_easy_medium_matches 50 `
  --eval_hard_matches 50

python agent/agent/v6_1/train_ppo.py `
  --resume_checkpoint agent/agent/v6/checkpoints/phase_item_control_easy/model_step2528800.pth `
  --bc_reference_path "" `
  --bc_coef 0.0 `
  --resume_global_step_override 0 `
  --force_stage easy_item_taken `
  --total_steps 1800000 `
  --save_every 25000 `
  --eval_easy_medium_matches 5 `
  --eval_hard_matches 5 `
  --eval_workers 4 `
  --num_envs 4 `
  --n_steps 1024
