python scripts\participant\run_local_match.py `
--agent_paths GeniusRuleAgent BoxFarmerAgent TacticalRuleAgent agent\agent\v6_1\ `
--num_episodes 1 `
--max_steps 500 `
--visualize 1 

python agent/agent/v6_1/train_ppo.py `                                
--resume_checkpoint agent/agent/v6_1/checkpoints/model_step1331200.pth `           
--bc_reference_path agent/agent/v6_1/bc_actor.pth `
--eval_only 1 `
--eval_easy_medium_matches 50 `
--eval_hard_matches 50

python agent/agent/v6/train_ppo.py `                                                                                                               
>>   --resume_checkpoint agent/agent/v6/checkpoints/model_step1914400.pth `             
>>   --bc_reference_path agent/agent/v6/bc_actor.pth `
>>   --force_stage phase_resource_control `
>>   --total_steps 3_000_000 `
>>   --save_every 25000 `
>>   --eval_easy_medium_matches 5 `
>>   --eval_hard_matches 5 `
>>   --eval_workers 4 `
>>   --num_envs 4 `
>>   --n_steps 1024

phase_eassy_control
2_682_400: 2.6 - 3 - 2.2
2_631_200: 2.5 - 2.6 - 2.4
2_554_400 - eval;2.5 - 2.6 - 2.4
2_528_800 - eval:2.7 - 3 - 2.4
2_605_600 - 2.4 - 2.4 -2.4


![alt text](image-1.png)


![alt text](image.png)

Step 2,682,400 | Ep   813 | Stage phase_item_control_easy | ActorTrainable True | FirstRate 46.00% | Bombs 16.43 | VB 51.02% | UB 0.00% | NEB 0.00% | Danger 60.9 | Tiles 48.0 | Repeat 75.71% | ItemOpp 23.8 | ItemTake 11.26% | ItemIgn 34.58% | I/Box 0.39 | B/Bo 0.53 | PG -0.0008 | Val 0.2192 | BC 0.3265 | Ent 0.272 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.873 | SPS 466 | T r/u/e/c 17.1/1.6/0.0/0.0s
  -> Saved checkpoint: /workspace/gdgoc_bomberland_challenge/agent/agent/v6/checkpoints/model_step2682400.pth
  -> Eval score 2.600 | AvgRank 0.40 | R0/1/2/3 70%/20%/10%/0% | UF 60%
     EM score/rank 3.000/0.00 | R0/1/2/3 100%/0%/0%/0% | VB/Repeat 71.66%/76.98%
     Hard score/rank 2.200/0.80 | R0/1/2/3 40%/40%/20%/0% | VB/Repeat 74.58%/80.28%
     Diff K/B/I/Bp +0.40/-1.40/-5.30/+5.60 | Cap/Radius -2.00/-1.10 | Timeout loss K/B/I/B 0.00%/10.00%/0.00%/0.00%
     ItemOpp 28.4 | Take 10.61% | Ignore 36.78% | I/Box 0.57 | Box/Bomb 0.28
     TIMING rollout 17.1s | update 1.6s | checkpoint 0.1s | eval 74.3s

Step 2,682,400 | Ep   813 | Stage phase_item_control_easy | ActorTrainable True | FirstRate 46.00% | Bombs 16.43 | VB 51.02% | UB 0.00% | NEB 0.00% | Danger 60.9 | Tiles 48.0 | Repeat 75.71% | ItemOpp 23.8 | ItemTake 11.26% | ItemIgn 34.58% | I/Box 0.39 | B/Bo 0.53 | PG -0.0008 | Val 0.2192 | BC 0.3265 | Ent 0.272 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.873 | SPS 466 | T r/u/e/c 17.1/1.6/0.0/0.0s
  -> Saved checkpoint: /workspace/gdgoc_bomberland_challenge/agent/agent/v6/checkpoints/model_step2682400.pth
  -> Eval score 2.600 | AvgRank 0.40 | R0/1/2/3 70%/20%/10%/0% | UF 60%
     EM score/rank 3.000/0.00 | R0/1/2/3 100%/0%/0%/0% | VB/Repeat 71.66%/76.98%
     Hard score/rank 2.200/0.80 | R0/1/2/3 40%/40%/20%/0% | VB/Repeat 74.58%/80.28%
     Diff K/B/I/Bp +0.40/-1.40/-5.30/+5.60 | Cap/Radius -2.00/-1.10 | Timeout loss K/B/I/B 0.00%/10.00%/0.00%/0.00%
     ItemOpp 28.4 | Take 10.61% | Ignore 36.78% | I/Box 0.57 | Box/Bomb 0.28
     TIMING rollout 17.1s | update 1.6s | checkpoint 0.1s | eval 74.3s
  
(aic_gdgoc) PS D:\other\CNTT\Bomberland-GDGoC-AI-Challenge> python agent/agent/v6_1/train_ppo.py `
>> --resume_checkpoint agent/agent/v6_1/checkpoints/model_step1126400.pth `
>> --bc_reference_path agent/agent/v6_1/bc_actor.pth `
>>  --eval_only 1 `       
>> --eval_easy_medium_matches 50 `
>> --eval_hard_matches 50   
Using device: cpu
Using device: cpu
Using device: cpu
Resume checkpoint requested: agent\agent\v6_1\checkpoints\model_step1126400.pth
BC reference strict load failed; trying expanded-input load: Error(s) in loading state_dict for CNNLSTMBCActor:
        size mismatch for actor_core.fuse.0.weight: copying a param with shape torch.Size([512, 530]) from checkpoint, the shape in current model is torch.Size([512, 547]).
[bc_reference] Expanded input weight: actor_core.fuse.0.weight (512, 530) -> (512, 547)
Loaded frozen BC reference from agent\agent\v6_1\bc_actor.pth
Resumed PPO from step 1,126,400 | stage pressure_kill | actor_initialized True
Eval only | score 2.420 | AvgRank 0.58 | R0/1/2/3 65%/18%/11%/6% | UF 62%
  EM score/rank 2.380/0.62 | R0/1/2/3 64%/20%/6%/10% | VB/Repeat 53.10%/74.56%
  Hard score/rank 2.460/0.54 | R0/1/2/3 66%/16%/16%/2% | VB/Repeat 62.57%/77.39%
  Our K/B/I/Bp 0.85/7.08/4.15/24.38 | BestEnemy K/B/I/Bp 0.62/6.37/5.50/15.77
  Diff K/B/I/Bp +0.23/+0.71/-1.35/+8.61 | Cap/Radius diff -0.15/-0.64
  Timeout 10.00% | win 4.00% | loss 6.00% | loss_by K/B/I/B 2.00%/2.00%/0.00%/0.00%
  ItemOpp 21.4 | Take 18.85% | Ignore 16.53% | I/Box 0.58 | Box/Bomb 0.46
(aic_gdgoc) PS D:\other\CNTT\Bomberland-GDGoC-AI-Challenge> python agent/agent/v6_1/train_ppo.py `
>> --resume_checkpoint agent/agent/v6_1/checkpoints/model_step1152000.pth `
>> --bc_reference_path agent/agent/v6_1/bc_actor.pth `
>>  --eval_only 1 `
>> --eval_easy_medium_matches 50 `
>> --eval_hard_matches 50 
Using device: cpu
Using device: cpu
Using device: cpu
Resume checkpoint requested: agent\agent\v6_1\checkpoints\model_step1152000.pth
BC reference strict load failed; trying expanded-input load: Error(s) in loading state_dict for CNNLSTMBCActor:
        size mismatch for actor_core.fuse.0.weight: copying a param with shape torch.Size([512, 530]) from checkpoint, the shape in current model is torch.Size([512, 547]).
[bc_reference] Expanded input weight: actor_core.fuse.0.weight (512, 530) -> (512, 547)
Loaded frozen BC reference from agent\agent\v6_1\bc_actor.pth
Resumed PPO from step 1,152,000 | stage pressure_kill | actor_initialized True
Eval only | score 2.320 | AvgRank 0.68 | R0/1/2/3 60%/20%/12%/8% | UF 58%
  EM score/rank 2.400/0.60 | R0/1/2/3 66%/14%/14%/6% | VB/Repeat 49.58%/73.94%
  Hard score/rank 2.240/0.76 | R0/1/2/3 54%/26%/10%/10% | VB/Repeat 62.58%/79.17%
  Our K/B/I/Bp 0.70/6.80/3.67/22.41 | BestEnemy K/B/I/Bp 0.56/6.37/5.50/15.37
  Diff K/B/I/Bp +0.14/+0.43/-1.83/+7.04 | Cap/Radius diff -0.46/-0.73
  Timeout 15.00% | win 3.00% | loss 12.00% | loss_by K/B/I/B 4.00%/4.00%/1.00%/0.00%
  ItemOpp 23.5 | Take 16.97% | Ignore 17.76% | I/Box 0.53 | Box/Bomb 0.47
(aic_gdgoc) PS D:\other\CNTT\Bomberland-GDGoC-AI-Challenge> python agent/agent/v6/train_ppo.py `  
>> --resume_checkpoint agent/agent/v6/submission/model.pth `               
>> --bc_reference_path agent/agent/v6/bc_actor.pth `  
>>  --eval_only 1 `
>> --eval_easy_medium_matches 50 `
>> --eval_hard_matches 50 
Using device: cpu
Using device: cpu
Using device: cpu
Resume checkpoint requested: agent\agent\v6\submission\model.pth
Loaded frozen BC reference from agent\agent\v6\bc_actor.pth
Resumed PPO from step 2,528,800 | stage phase_item_control_easy | actor_initialized True
Eval only | score 2.460 | AvgRank 0.54 | R0/1/2/3 62%/27%/6%/5% | UF 60%
  EM score/rank 2.580/0.42 | R0/1/2/3 70%/22%/4%/4% | VB/Repeat 56.68%/74.61%
  Hard score/rank 2.340/0.66 | R0/1/2/3 54%/32%/8%/6% | VB/Repeat 58.54%/78.65%
  Our K/B/I/Bp 0.51/5.94/2.31/17.34 | BestEnemy K/B/I/Bp 0.83/6.87/6.87/17.82
  Diff K/B/I/Bp -0.32/-0.93/-4.56/-0.48 | Cap/Radius diff -1.38/-1.62
  Timeout 6.00% | win 4.00% | loss 2.00% | loss_by K/B/I/B 0.00%/1.00%/0.00%/0.00%
  ItemOpp 21.0 | Take 11.82% | Ignore 32.68% | I/Box 0.37 | Box/Bomb 0.46
(aic_gdgoc) PS D:\other\CNTT\Bomberland-GDGoC-AI-Challenge> 

Eval only | score 2.740 | AvgRank 0.26 | R0/1/2/3 86%/6%/4%/4% | UF 85%
  EM score/rank 2.860/0.14 | R0/1/2/3 92%/4%/2%/2% | VB/Repeat 58.71%/71.36%
  Hard score/rank 2.620/0.38 | R0/1/2/3 80%/8%/6%/6% | VB/Repeat 67.77%/75.16%
  Our K/B/I/Bp 1.21/7.40/4.11/25.78 | BestEnemy K/B/I/Bp 0.54/6.34/5.05/14.84
  Diff K/B/I/Bp +0.67/+1.06/-0.94/+10.94 | Cap/Radius diff -0.52/-0.18
  Timeout 6.00% | win 3.00% | loss 3.00% | loss_by K/B/I/B 0.00%/2.00%/1.00%/0.00%
  ItemOpp 24.2 | Take 18.35% | Ignore 17.34% | I/Box 0.59 | Box/Bomb 0.42


Resume checkpoint requested: /kaggle/input/datasets/kininhhng/bomberman-checkpoints-ppo/new_checkpoins/model_step1280000.pth
BC reference checkpoint not found, disabling BC regularization during PPO
Resumed PPO from step 0 | stage resource_control | actor_initialized True
Resume global_step override applied: 0 | effective stage farm_box_safe
Eval only | score 2.290 | AvgRank 0.71 | R0/1/2/3 60%/15%/19%/6% | UF 51%
  EM score/rank 2.400/0.60 | R0/1/2/3 64%/16%/16%/4% | VB/Repeat 53.45%/74.16%
  Hard score/rank 2.180/0.82 | R0/1/2/3 56%/14%/22%/8% | VB/Repeat 62.17%/78.30%
  Our K/B/I/Bp 0.71/7.58/4.38/25.78 | BestEnemy K/B/I/Bp 0.72/6.02/5.44/17.25
  Diff K/B/I/Bp -0.01/+1.56/-1.06/+8.53 | Cap/Radius diff -0.28/-0.38
  Timeout 12.00% | win 3.00% | loss 9.00% | loss_by K/B/I/B 3.00%/1.00%/0.00%/0.00%
  ItemOpp 25.1 | Take 18.73% | Ignore 15.72% | I/Box 0.59 | Box/Bomb 0.44