python scripts/participant/run_local_match.py `
  --agent_paths GeniusRuleAgent BoxFarmerAgent TacticalRuleAgent agent/agent/v6_1 `
  --num_episodes 1 `
  --max_steps 500 `
  --visualize 1

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
