python scripts\participant\run_local_match.py `
--agent_paths GeniusRuleAgent TacticalRuleAgent TacticalRuleAgent agent\agent\v6 `
--num_episodes 1 `
--max_steps 500 `
--visualize 1 

python agent/agent/v6/train_ppo.py `                                
--resume_checkpoint agent/agent/v6/checkpoints/model_step760832.pth `           
--bc_reference_path agent/agent/v6/bc_actor.pth `
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

Step 2,210,336 | Ep  1055 | Stage phase_resource_control | ActorTrainable True | FirstRate 37.00% | Bombs 17.79 | VB 57.89% | UB 0.00% | NEB 0.00% | Danger 74.8 | Tiles 49.0 | Repeat 81.28% | PG -0.0037 | Val 0.2858 | BC 0.5845 | Ent 0.444 | KL 0.0025 | Clip 0.95% | Ratio 0.998+/-0.035 | EV 0.730 | SPS 158 | T r/u/e/c 35.7/13.1/0.0/0.0s
Step 2,211,360 | Ep  1059 | Stage phase_resource_control | ActorTrainable True | FirstRate 37.00% | Bombs 17.68 | VB 57.93% | UB 0.00% | NEB 0.00% | Danger 74.0 | Tiles 48.9 | Repeat 81.26% | PG -0.0026 | Val 0.1892 | BC 0.6470 | Ent 0.520 | KL -0.0018 | Clip 0.54% | Ratio 1.003+/-0.032 | EV 0.238 | SPS 157 | T r/u/e/c 49.1/13.1/0.0/0.0s
Step 2,212,384 | Ep  1062 | Stage phase_resource_control | ActorTrainable True | FirstRate 37.00% | Bombs 17.60 | VB 57.84% | UB 0.00% | NEB 0.00% | Danger 73.4 | Tiles 48.6 | Repeat 81.26% | PG -0.0019 | Val 0.2489 | BC 0.5689 | Ent 0.388 | KL -0.0000 | Clip 0.12% | Ratio 1.000+/-0.019 | EV 0.798 | SPS 157 | T r/u/e/c 37.2/13.2/0.0/0.0s
Step 2,213,408 | Ep  1065 | Stage phase_resource_control | ActorTrainable True | FirstRate 37.00% | Bombs 17.85 | VB 57.84% | UB 0.00% | NEB 0.00% | Danger 74.5 | Tiles 49.2 | Repeat 81.10% | PG -0.0017 | Val 0.2098 | BC 0.2558 | Ent 0.224 | KL 0.0007 | Clip 0.02% | Ratio 0.999+/-0.011 | EV 0.876 | SPS 156 | T r/u/e/c 18.8/13.1/0.0/0.0s
Step 2,214,432 | Ep  1070 | Stage phase_resource_control | ActorTrainable True | FirstRate 36.00% | Bombs 17.18 | VB 56.30% | UB 0.00% | NEB 0.00% | Danger 72.4 | Tiles 48.5 | Repeat 81.25% | PG -0.0020 | Val 0.2784 | BC 0.5098 | Ent 0.338 | KL 0.0004 | Clip 0.10% | Ratio 1.000+/-0.016 | EV 0.817 | SPS 156 | T r/u/e/c 30.6/13.1/0.0/0.0s
Step 2,215,456 | Ep  1073 | Stage phase_resource_control | ActorTrainable True | FirstRate 36.00% | Bombs 16.89 | VB 55.56% | UB 0.00% | NEB 0.00% | Danger 71.2 | Tiles 47.7 | Repeat 81.46% | PG -0.0017 | Val 0.1577 | BC 0.5033 | Ent 0.379 | KL -0.0002 | Clip 0.02% | Ratio 1.000+/-0.014 | EV 0.891 | SPS 155 | T r/u/e/c 34.6/13.2/0.0/0.0s
Step 2,216,480 | Ep  1077 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 16.96 | VB 55.23% | UB 0.00% | NEB 0.00% | Danger 71.8 | Tiles 48.3 | Repeat 81.21% | PG -0.0022 | Val 0.1695 | BC 0.5196 | Ent 0.386 | KL 0.0008 | Clip 0.12% | Ratio 1.000+/-0.024 | EV 0.864 | SPS 155 | T r/u/e/c 28.5/13.1/0.0/0.0s
Step 2,217,504 | Ep  1080 | Stage phase_resource_control | ActorTrainable True | FirstRate 35.00% | Bombs 17.18 | VB 55.35% | UB 0.00% | NEB 0.00% | Danger 72.0 | Tiles 48.9 | Repeat 80.98% | PG -0.0026 | Val 0.0795 | BC 0.6756 | Ent 0.532 | KL -0.0001 | Clip 0.95% | Ratio 1.001+/-0.039 | EV 0.439 | SPS 155 | T r/u/e/c 35.6/13.2/0.0/0.0s
Step 2,218,528 | Ep  1083 | Stage phase_resource_control | ActorTrainable True | FirstRate 35.00% | Bombs 17.68 | VB 56.08% | UB 0.00% | NEB 0.00% | Danger 74.3 | Tiles 49.4 | Repeat 81.29% | PG -0.0025 | Val 0.1940 | BC 0.6952 | Ent 0.493 | KL 0.0003 | Clip 0.44% | Ratio 1.000+/-0.030 | EV 0.675 | SPS 154 | T r/u/e/c 35.0/13.3/0.0/0.0s
Step 2,219,552 | Ep  1085 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 17.29 | VB 55.96% | UB 0.00% | NEB 0.00% | Danger 73.4 | Tiles 48.7 | Repeat 81.31% | PG -0.0027 | Val 0.1103 | BC 0.5806 | Ent 0.452 | KL -0.0004 | Clip 0.39% | Ratio 1.001+/-0.029 | EV 0.893 | SPS 154 | T r/u/e/c 34.8/13.3/0.0/0.0s
Step 2,220,576 | Ep  1088 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 17.43 | VB 56.76% | UB 0.00% | NEB 0.00% | Danger 74.8 | Tiles 49.4 | Repeat 81.24% | PG -0.0038 | Val 0.0942 | BC 0.7088 | Ent 0.494 | KL 0.0010 | Clip 0.59% | Ratio 1.000+/-0.031 | EV 0.563 | SPS 153 | T r/u/e/c 36.5/13.2/0.0/0.0s
Step 2,221,600 | Ep  1092 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 17.85 | VB 57.55% | UB 0.00% | NEB 0.00% | Danger 75.4 | Tiles 49.2 | Repeat 81.22% | PG -0.0021 | Val 0.2809 | BC 0.7295 | Ent 0.455 | KL -0.0004 | Clip 0.17% | Ratio 1.001+/-0.027 | EV 0.773 | SPS 153 | T r/u/e/c 34.6/13.1/0.0/0.0s
  -> Saved checkpoint: D:\other\CNTT\Bomberland-GDGoC-AI-Challenge\agent\agent\v6\checkpoints\model_step2221600.pth
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
  -> Eval score 2.300 | AvgRank 0.70 | R0/1/2/3 50%/30%/20%/0% | UF 50%
     EM score/rank 2.200/0.80 | R0/1/2/3 40%/40%/20%/0% | VB/Repeat 57.14%/79.61%
     Hard score/rank 2.400/0.60 | R0/1/2/3 60%/20%/20%/0% | VB/Repeat 74.01%/78.97%
     Diff K/B/I/Bp +0.10/+1.00/-4.10/+6.60 | Cap/Radius -0.60/-1.20 | Timeout loss K/B/I/B 10.00%/0.00%/0.00%/0.00%
     TIMING rollout 34.6s | update 13.1s | checkpoint 0.1s | eval 51.9s
Step 2,222,624 | Ep  1095 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 17.84 | VB 57.44% | UB 0.00% | NEB 0.00% | Danger 75.5 | Tiles 48.6 | Repeat 81.55% | PG -0.0016 | Val 0.1360 | BC 0.6434 | Ent 0.446 | KL 0.0003 | Clip 0.02% | Ratio 1.000+/-0.020 | EV 0.773 | SPS 152 | T r/u/e/c 46.3/13.5/0.0/0.0s
Step 2,223,648 | Ep  1101 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 17.78 | VB 58.17% | UB 0.00% | NEB 0.00% | Danger 75.4 | Tiles 49.1 | Repeat 81.09% | PG -0.0021 | Val 0.3711 | BC 0.6796 | Ent 0.462 | KL 0.0006 | Clip 0.02% | Ratio 1.000+/-0.014 | EV 0.743 | SPS 151 | T r/u/e/c 36.9/13.6/0.0/0.0s
Step 2,224,672 | Ep  1105 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 17.79 | VB 58.35% | UB 0.00% | NEB 0.00% | Danger 75.8 | Tiles 48.9 | Repeat 81.55% | PG -0.0020 | Val 0.1002 | BC 0.4520 | Ent 0.303 | KL -0.0002 | Clip 0.02% | Ratio 1.000+/-0.010 | EV 0.940 | SPS 151 | T r/u/e/c 25.8/13.6/0.0/0.0s
Step 2,225,696 | Ep  1107 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 17.83 | VB 58.34% | UB 0.00% | NEB 0.00% | Danger 76.3 | Tiles 49.2 | Repeat 81.48% | PG -0.0017 | Val 0.1533 | BC 0.7026 | Ent 0.536 | KL 0.0006 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.813 | SPS 151 | T r/u/e/c 28.3/13.3/0.0/0.0s
Step 2,226,720 | Ep  1111 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 17.68 | VB 58.39% | UB 0.00% | NEB 0.00% | Danger 75.9 | Tiles 48.6 | Repeat 81.76% | PG -0.0032 | Val 0.2542 | BC 0.5267 | Ent 0.371 | KL 0.0003 | Clip 0.05% | Ratio 1.000+/-0.015 | EV 0.838 | SPS 150 | T r/u/e/c 38.1/13.5/0.0/0.0s
Step 2,227,744 | Ep  1115 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 18.06 | VB 58.62% | UB 0.00% | NEB 0.00% | Danger 76.6 | Tiles 48.2 | Repeat 81.83% | PG -0.0010 | Val 0.3127 | BC 0.5366 | Ent 0.376 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.795 | SPS 150 | T r/u/e/c 35.3/13.5/0.0/0.0s
Step 2,228,768 | Ep  1118 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 18.27 | VB 59.41% | UB 0.00% | NEB 0.00% | Danger 76.7 | Tiles 48.7 | Repeat 81.47% | PG -0.0016 | Val 0.3113 | BC 0.6182 | Ent 0.444 | KL 0.0000 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.733 | SPS 149 | T r/u/e/c 40.6/13.5/0.0/0.0s
Step 2,229,792 | Ep  1122 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 18.35 | VB 58.34% | UB 0.00% | NEB 0.00% | Danger 76.9 | Tiles 48.4 | Repeat 81.77% | PG -0.0033 | Val 0.2139 | BC 0.3993 | Ent 0.305 | KL 0.0011 | Clip 0.20% | Ratio 0.999+/-0.017 | EV 0.871 | SPS 149 | T r/u/e/c 22.8/13.5/0.0/0.0s
Step 2,230,816 | Ep  1127 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 18.39 | VB 59.21% | UB 0.00% | NEB 0.00% | Danger 77.0 | Tiles 48.6 | Repeat 81.40% | PG -0.0018 | Val 0.2838 | BC 0.4891 | Ent 0.350 | KL 0.0002 | Clip 0.02% | Ratio 1.000+/-0.011 | EV 0.815 | SPS 149 | T r/u/e/c 28.1/13.3/0.0/0.0s
Step 2,231,840 | Ep  1129 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 17.91 | VB 58.91% | UB 0.00% | NEB 0.00% | Danger 75.0 | Tiles 47.9 | Repeat 81.46% | PG -0.0019 | Val 0.1926 | BC 0.3742 | Ent 0.294 | KL 0.0006 | Clip 0.02% | Ratio 1.000+/-0.013 | EV 0.903 | SPS 148 | T r/u/e/c 21.1/13.6/0.0/0.0s
Step 2,232,864 | Ep  1134 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 17.90 | VB 59.62% | UB 0.00% | NEB 0.00% | Danger 77.8 | Tiles 48.2 | Repeat 81.53% | PG -0.0026 | Val 0.2026 | BC 0.3699 | Ent 0.296 | KL -0.0005 | Clip 0.07% | Ratio 1.001+/-0.015 | EV 0.890 | SPS 148 | T r/u/e/c 25.1/13.6/0.0/0.0s
Step 2,233,888 | Ep  1136 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 17.81 | VB 59.20% | UB 0.00% | NEB 0.00% | Danger 77.2 | Tiles 47.5 | Repeat 81.90% | PG -0.0022 | Val 0.1720 | BC 0.5700 | Ent 0.389 | KL -0.0001 | Clip 0.07% | Ratio 1.000+/-0.019 | EV 0.887 | SPS 148 | T r/u/e/c 33.6/13.4/0.0/0.0s
Step 2,234,912 | Ep  1141 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 18.56 | VB 60.65% | UB 0.00% | NEB 0.00% | Danger 78.5 | Tiles 48.7 | Repeat 81.50% | PG -0.0010 | Val 0.1474 | BC 0.7315 | Ent 0.518 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.016 | EV 0.519 | SPS 147 | T r/u/e/c 38.2/13.8/0.0/0.0s
Step 2,235,936 | Ep  1146 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 18.50 | VB 60.85% | UB 0.00% | NEB 0.00% | Danger 78.0 | Tiles 48.0 | Repeat 81.55% | PG -0.0024 | Val 0.3378 | BC 0.3864 | Ent 0.301 | KL 0.0005 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.785 | SPS 147 | T r/u/e/c 24.0/13.7/0.0/0.0s
Step 2,236,960 | Ep  1148 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 18.28 | VB 60.48% | UB 0.00% | NEB 0.00% | Danger 77.3 | Tiles 47.3 | Repeat 81.89% | PG -0.0011 | Val 0.1206 | BC 0.3675 | Ent 0.241 | KL 0.0002 | Clip 0.02% | Ratio 1.000+/-0.011 | EV 0.927 | SPS 147 | T r/u/e/c 16.9/13.5/0.0/0.0s
Step 2,237,984 | Ep  1154 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 17.87 | VB 59.25% | UB 0.00% | NEB 0.00% | Danger 74.4 | Tiles 45.7 | Repeat 82.13% | PG -0.0014 | Val 0.3434 | BC 0.5733 | Ent 0.398 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.764 | SPS 146 | T r/u/e/c 40.1/13.6/0.0/0.0s
Step 2,239,008 | Ep  1159 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 17.65 | VB 57.82% | UB 0.00% | NEB 0.00% | Danger 73.5 | Tiles 44.8 | Repeat 82.29% | PG -0.0021 | Val 0.2181 | BC 0.3234 | Ent 0.238 | KL -0.0003 | Clip 0.02% | Ratio 1.000+/-0.012 | EV 0.888 | SPS 146 | T r/u/e/c 26.9/13.7/0.0/0.0s
Step 2,240,032 | Ep  1162 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 17.78 | VB 58.04% | UB 0.00% | NEB 0.00% | Danger 74.1 | Tiles 45.3 | Repeat 82.06% | PG -0.0015 | Val 0.1660 | BC 0.6432 | Ent 0.500 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.013 | EV 0.546 | SPS 145 | T r/u/e/c 42.6/13.7/0.0/0.0s
Step 2,241,056 | Ep  1166 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 18.00 | VB 58.51% | UB 0.00% | NEB 0.00% | Danger 74.9 | Tiles 46.0 | Repeat 81.87% | PG -0.0032 | Val 0.2028 | BC 0.5018 | Ent 0.374 | KL 0.0002 | Clip 0.07% | Ratio 1.000+/-0.014 | EV 0.875 | SPS 145 | T r/u/e/c 40.4/13.4/0.0/0.0s
Step 2,242,080 | Ep  1171 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 18.50 | VB 59.27% | UB 0.00% | NEB 0.00% | Danger 75.8 | Tiles 46.2 | Repeat 81.76% | PG -0.0032 | Val 0.3005 | BC 0.5717 | Ent 0.421 | KL 0.0006 | Clip 0.17% | Ratio 1.000+/-0.016 | EV 0.716 | SPS 145 | T r/u/e/c 39.3/13.5/0.0/0.0s
Step 2,243,104 | Ep  1174 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 18.69 | VB 59.74% | UB 0.00% | NEB 0.00% | Danger 76.1 | Tiles 46.5 | Repeat 81.65% | PG -0.0021 | Val 0.1842 | BC 0.7195 | Ent 0.514 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.013 | EV 0.495 | SPS 144 | T r/u/e/c 39.1/13.4/0.0/0.0s
Step 2,244,128 | Ep  1177 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 18.74 | VB 59.76% | UB 0.00% | NEB 0.00% | Danger 76.5 | Tiles 46.7 | Repeat 81.68% | PG -0.0017 | Val 0.1879 | BC 0.7100 | Ent 0.452 | KL 0.0006 | Clip 0.00% | Ratio 1.000+/-0.014 | EV 0.820 | SPS 144 | T r/u/e/c 36.9/13.5/0.0/0.0s
Step 2,245,152 | Ep  1181 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 18.81 | VB 59.84% | UB 0.00% | NEB 0.00% | Danger 76.6 | Tiles 46.0 | Repeat 81.95% | PG -0.0017 | Val 0.1842 | BC 0.6758 | Ent 0.505 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.575 | SPS 143 | T r/u/e/c 34.8/13.6/0.0/0.0s
Step 2,246,176 | Ep  1183 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 18.75 | VB 59.82% | UB 0.00% | NEB 0.00% | Danger 76.8 | Tiles 45.8 | Repeat 82.00% | PG -0.0012 | Val 0.3420 | BC 0.2641 | Ent 0.218 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.778 | SPS 143 | T r/u/e/c 18.1/13.4/0.0/0.0s
Step 2,247,200 | Ep  1188 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 18.09 | VB 58.56% | UB 0.00% | NEB 0.00% | Danger 73.1 | Tiles 44.1 | Repeat 82.21% | PG -0.0018 | Val 0.3363 | BC 0.4580 | Ent 0.372 | KL 0.0005 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.794 | SPS 143 | T r/u/e/c 34.7/13.6/0.0/0.0s
  -> Saved checkpoint: D:\other\CNTT\Bomberland-GDGoC-AI-Challenge\agent\agent\v6\checkpoints\model_step2247200.pth
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
  -> Eval score 2.100 | AvgRank 0.90 | R0/1/2/3 50%/20%/20%/10% | UF 50%
     EM score/rank 2.400/0.60 | R0/1/2/3 60%/20%/20%/0% | VB/Repeat 69.30%/80.86%
     Hard score/rank 1.800/1.20 | R0/1/2/3 40%/20%/20%/20% | VB/Repeat 70.97%/87.05%
     Diff K/B/I/Bp +0.10/-0.40/-6.30/+1.70 | Cap/Radius -1.50/-0.90 | Timeout loss K/B/I/B 10.00%/10.00%/10.00%/0.00%
     TIMING rollout 34.7s | update 13.6s | checkpoint 0.1s | eval 62.5s
Step 2,248,224 | Ep  1194 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 17.45 | VB 57.28% | UB 0.00% | NEB 0.00% | Danger 71.3 | Tiles 43.6 | Repeat 81.94% | PG -0.0019 | Val 0.2597 | BC 0.6305 | Ent 0.450 | KL 0.0009 | Clip 0.00% | Ratio 0.999+/-0.013 | EV 0.613 | SPS 142 | T r/u/e/c 33.9/13.5/0.0/0.0s
Step 2,249,248 | Ep  1197 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 17.07 | VB 57.13% | UB 0.00% | NEB 0.00% | Danger 70.4 | Tiles 43.0 | Repeat 81.98% | PG -0.0011 | Val 0.1812 | BC 0.5273 | Ent 0.424 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.861 | SPS 142 | T r/u/e/c 26.5/13.6/0.0/0.0s
Step 2,250,272 | Ep  1201 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 17.12 | VB 57.06% | UB 0.00% | NEB 0.00% | Danger 70.8 | Tiles 42.9 | Repeat 82.04% | PG -0.0034 | Val 0.1833 | BC 0.4240 | Ent 0.362 | KL -0.0006 | Clip 0.05% | Ratio 1.001+/-0.017 | EV 0.889 | SPS 141 | T r/u/e/c 25.1/13.6/0.0/0.0s
Step 2,251,296 | Ep  1205 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 17.05 | VB 56.23% | UB 0.00% | NEB 0.00% | Danger 70.1 | Tiles 42.7 | Repeat 82.02% | PG -0.0012 | Val 0.3032 | BC 0.4262 | Ent 0.332 | KL 0.0010 | Clip 0.00% | Ratio 0.999+/-0.015 | EV 0.820 | SPS 141 | T r/u/e/c 29.4/13.5/0.0/0.0s
Step 2,252,320 | Ep  1208 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 17.01 | VB 55.69% | UB 0.00% | NEB 0.00% | Danger 70.0 | Tiles 42.5 | Repeat 82.04% | PG -0.0021 | Val 0.1314 | BC 0.4490 | Ent 0.350 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.920 | SPS 141 | T r/u/e/c 30.2/13.7/0.0/0.0s
Step 2,253,344 | Ep  1213 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 16.66 | VB 55.73% | UB 0.00% | NEB 0.00% | Danger 69.3 | Tiles 42.6 | Repeat 81.72% | PG -0.0011 | Val 0.1376 | BC 0.6386 | Ent 0.486 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.787 | SPS 140 | T r/u/e/c 37.0/13.6/0.0/0.0s
Step 2,254,368 | Ep  1217 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 16.97 | VB 56.81% | UB 0.00% | NEB 0.00% | Danger 70.2 | Tiles 43.5 | Repeat 81.36% | PG -0.0015 | Val 0.2939 | BC 0.6169 | Ent 0.451 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.470 | SPS 140 | T r/u/e/c 38.9/13.8/0.0/0.0s
Step 2,255,392 | Ep  1222 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 16.74 | VB 57.30% | UB 0.00% | NEB 0.00% | Danger 69.6 | Tiles 43.3 | Repeat 81.26% | PG -0.0010 | Val 0.1215 | BC 0.5447 | Ent 0.337 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.919 | SPS 140 | T r/u/e/c 34.3/13.7/0.0/0.0s
Step 2,256,416 | Ep  1227 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 16.79 | VB 56.64% | UB 0.00% | NEB 0.00% | Danger 69.3 | Tiles 43.1 | Repeat 81.19% | PG -0.0026 | Val 0.2288 | BC 0.6591 | Ent 0.462 | KL 0.0006 | Clip 0.05% | Ratio 1.000+/-0.017 | EV 0.733 | SPS 139 | T r/u/e/c 40.1/13.6/0.0/0.0s
Step 2,257,440 | Ep  1230 | Stage phase_resource_control | ActorTrainable True | FirstRate 35.00% | Bombs 17.32 | VB 57.43% | UB 0.00% | NEB 0.00% | Danger 70.9 | Tiles 44.1 | Repeat 81.11% | PG -0.0028 | Val 0.1106 | BC 0.5944 | Ent 0.455 | KL 0.0008 | Clip 0.12% | Ratio 0.999+/-0.019 | EV 0.834 | SPS 139 | T r/u/e/c 37.6/13.9/0.0/0.0s
Step 2,258,464 | Ep  1234 | Stage phase_resource_control | ActorTrainable True | FirstRate 36.00% | Bombs 17.45 | VB 57.61% | UB 0.00% | NEB 0.00% | Danger 68.6 | Tiles 44.5 | Repeat 80.83% | PG -0.0019 | Val 0.2542 | BC 0.5591 | Ent 0.385 | KL 0.0004 | Clip 0.02% | Ratio 1.000+/-0.018 | EV 0.776 | SPS 138 | T r/u/e/c 36.8/13.6/0.0/0.0s
Step 2,259,488 | Ep  1237 | Stage phase_resource_control | ActorTrainable True | FirstRate 37.00% | Bombs 17.77 | VB 57.97% | UB 0.00% | NEB 0.00% | Danger 69.7 | Tiles 45.3 | Repeat 80.71% | PG -0.0030 | Val 0.2177 | BC 0.5322 | Ent 0.378 | KL 0.0012 | Clip 0.12% | Ratio 0.999+/-0.023 | EV 0.821 | SPS 138 | T r/u/e/c 41.0/13.6/0.0/0.0s
Step 2,260,512 | Ep  1240 | Stage phase_resource_control | ActorTrainable True | FirstRate 36.00% | Bombs 17.50 | VB 57.68% | UB 0.00% | NEB 0.00% | Danger 69.2 | Tiles 45.3 | Repeat 80.93% | PG -0.0023 | Val 0.2155 | BC 0.3597 | Ent 0.277 | KL 0.0008 | Clip 0.27% | Ratio 1.000+/-0.023 | EV 0.885 | SPS 138 | T r/u/e/c 27.5/13.6/0.0/0.0s
Step 2,261,536 | Ep  1244 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 17.30 | VB 57.10% | UB 0.00% | NEB 0.00% | Danger 68.5 | Tiles 44.8 | Repeat 81.34% | PG -0.0016 | Val 0.2218 | BC 0.4084 | Ent 0.281 | KL -0.0001 | Clip 0.02% | Ratio 1.000+/-0.012 | EV 0.836 | SPS 137 | T r/u/e/c 31.7/13.5/0.0/0.0s
Step 2,262,560 | Ep  1248 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 17.30 | VB 56.69% | UB 0.00% | NEB 0.00% | Danger 68.3 | Tiles 45.5 | Repeat 80.70% | PG -0.0019 | Val 0.2734 | BC 0.5098 | Ent 0.389 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.806 | SPS 137 | T r/u/e/c 34.6/13.8/0.0/0.0s
Step 2,263,584 | Ep  1252 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 17.72 | VB 57.12% | UB 0.00% | NEB 0.00% | Danger 70.0 | Tiles 45.7 | Repeat 80.84% | PG -0.0010 | Val 0.1358 | BC 0.5600 | Ent 0.422 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.006 | EV 0.851 | SPS 137 | T r/u/e/c 34.4/13.4/0.0/0.0s
Step 2,264,608 | Ep  1255 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 17.48 | VB 56.73% | UB 0.00% | NEB 0.00% | Danger 69.2 | Tiles 45.6 | Repeat 80.72% | PG -0.0016 | Val 0.3329 | BC 0.3978 | Ent 0.211 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.822 | SPS 136 | T r/u/e/c 35.8/13.5/0.0/0.0s
Step 2,265,632 | Ep  1258 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 17.62 | VB 57.58% | UB 0.00% | NEB 0.00% | Danger 70.6 | Tiles 46.1 | Repeat 81.03% | PG -0.0020 | Val 0.1936 | BC 0.2148 | Ent 0.139 | KL -0.0006 | Clip 0.00% | Ratio 1.001+/-0.009 | EV 0.874 | SPS 136 | T r/u/e/c 25.5/13.4/0.0/0.0s
Step 2,266,656 | Ep  1263 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 18.23 | VB 57.49% | UB 0.00% | NEB 0.00% | Danger 72.6 | Tiles 46.2 | Repeat 81.31% | PG -0.0022 | Val 0.4660 | BC 0.4765 | Ent 0.367 | KL 0.0008 | Clip 0.07% | Ratio 0.999+/-0.018 | EV 0.752 | SPS 136 | T r/u/e/c 33.9/13.4/0.0/0.0s
Step 2,267,680 | Ep  1268 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 17.15 | VB 55.91% | UB 0.00% | NEB 0.00% | Danger 69.1 | Tiles 44.9 | Repeat 81.31% | PG -0.0028 | Val 0.1395 | BC 0.5401 | Ent 0.375 | KL 0.0002 | Clip 0.05% | Ratio 1.000+/-0.019 | EV 0.911 | SPS 135 | T r/u/e/c 34.4/13.4/0.0/0.0s
Step 2,268,704 | Ep  1271 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 17.37 | VB 56.44% | UB 0.00% | NEB 0.00% | Danger 69.7 | Tiles 45.5 | Repeat 80.91% | PG -0.0012 | Val 0.2186 | BC 0.6269 | Ent 0.439 | KL -0.0003 | Clip 0.00% | Ratio 1.000+/-0.014 | EV 0.793 | SPS 135 | T r/u/e/c 40.2/13.5/0.0/0.0s
Step 2,269,728 | Ep  1274 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 18.07 | VB 56.88% | UB 0.00% | NEB 0.00% | Danger 72.0 | Tiles 46.0 | Repeat 80.98% | PG -0.0019 | Val 0.3640 | BC 0.4651 | Ent 0.339 | KL -0.0007 | Clip 0.02% | Ratio 1.001+/-0.010 | EV 0.797 | SPS 135 | T r/u/e/c 27.9/13.5/0.0/0.0s
Step 2,270,752 | Ep  1279 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 17.54 | VB 55.95% | UB 0.00% | NEB 0.00% | Danger 70.2 | Tiles 44.4 | Repeat 81.13% | PG -0.0012 | Val 0.2753 | BC 0.4435 | Ent 0.339 | KL 0.0009 | Clip 0.00% | Ratio 0.999+/-0.011 | EV 0.812 | SPS 135 | T r/u/e/c 20.3/13.5/0.0/0.0s
Step 2,271,776 | Ep  1282 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 17.57 | VB 55.91% | UB 0.00% | NEB 0.00% | Danger 70.2 | Tiles 44.6 | Repeat 81.11% | PG -0.0020 | Val 0.2804 | BC 0.3368 | Ent 0.271 | KL 0.0006 | Clip 0.05% | Ratio 0.999+/-0.010 | EV 0.834 | SPS 134 | T r/u/e/c 22.2/13.4/0.0/0.0s
Step 2,272,800 | Ep  1284 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 17.45 | VB 55.27% | UB 0.00% | NEB 0.00% | Danger 70.2 | Tiles 44.4 | Repeat 81.20% | PG -0.0019 | Val 0.1423 | BC 0.3134 | Ent 0.252 | KL 0.0005 | Clip 0.00% | Ratio 1.000+/-0.015 | EV 0.899 | SPS 134 | T r/u/e/c 19.7/13.5/0.0/0.0s
  -> Saved checkpoint: D:\other\CNTT\Bomberland-GDGoC-AI-Challenge\agent\agent\v6\checkpoints\model_step2272800.pth
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
  -> Eval score 2.500 | AvgRank 0.50 | R0/1/2/3 60%/30%/10%/0% | UF 60%
     EM score/rank 2.400/0.60 | R0/1/2/3 40%/60%/0%/0% | VB/Repeat 60.67%/67.42%
     Hard score/rank 2.600/0.40 | R0/1/2/3 80%/0%/20%/0% | VB/Repeat 71.63%/79.56%
     Diff K/B/I/Bp -0.10/-0.30/-3.90/+1.10 | Cap/Radius -2.10/-1.80 | Timeout loss K/B/I/B 0.00%/0.00%/0.00%/0.00%
     TIMING rollout 19.7s | update 13.5s | checkpoint 0.1s | eval 43.2s
Step 2,273,824 | Ep  1289 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 17.59 | VB 55.17% | UB 0.00% | NEB 0.00% | Danger 70.3 | Tiles 45.2 | Repeat 80.92% | PG -0.0018 | Val 0.4661 | BC 0.3037 | Ent 0.219 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.014 | EV 0.739 | SPS 134 | T r/u/e/c 20.3/13.5/0.0/0.0s
Step 2,274,848 | Ep  1295 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 17.22 | VB 54.73% | UB 0.00% | NEB 0.00% | Danger 69.3 | Tiles 44.8 | Repeat 81.00% | PG -0.0024 | Val 0.1462 | BC 0.6824 | Ent 0.466 | KL 0.0004 | Clip 0.05% | Ratio 1.000+/-0.021 | EV 0.851 | SPS 133 | T r/u/e/c 35.0/13.1/0.0/0.0s
Step 2,275,872 | Ep  1297 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 17.26 | VB 54.56% | UB 0.00% | NEB 0.00% | Danger 69.1 | Tiles 45.1 | Repeat 80.87% | PG -0.0013 | Val 0.1820 | BC 0.5718 | Ent 0.421 | KL 0.0000 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.849 | SPS 133 | T r/u/e/c 35.8/13.3/0.0/0.0s
Step 2,276,896 | Ep  1300 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 17.76 | VB 55.38% | UB 0.00% | NEB 0.00% | Danger 70.8 | Tiles 45.4 | Repeat 81.18% | PG -0.0015 | Val 0.2347 | BC 0.3631 | Ent 0.238 | KL 0.0001 | Clip 0.05% | Ratio 1.000+/-0.010 | EV 0.873 | SPS 133 | T r/u/e/c 31.9/13.1/0.0/0.0s
Step 2,277,920 | Ep  1303 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 17.84 | VB 55.01% | UB 0.00% | NEB 0.00% | Danger 70.6 | Tiles 45.4 | Repeat 81.18% | PG -0.0019 | Val 0.3218 | BC 0.6347 | Ent 0.397 | KL 0.0001 | Clip 0.02% | Ratio 1.000+/-0.017 | EV 0.830 | SPS 132 | T r/u/e/c 31.2/13.1/0.0/0.0s
Step 2,278,944 | Ep  1306 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 17.70 | VB 55.31% | UB 0.00% | NEB 0.00% | Danger 71.3 | Tiles 45.1 | Repeat 81.36% | PG -0.0018 | Val 0.2029 | BC 0.3829 | Ent 0.288 | KL 0.0002 | Clip 0.12% | Ratio 1.000+/-0.019 | EV 0.858 | SPS 132 | T r/u/e/c 25.5/13.2/0.0/0.0s
Step 2,279,968 | Ep  1310 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 17.71 | VB 55.54% | UB 0.00% | NEB 0.00% | Danger 71.1 | Tiles 44.9 | Repeat 81.60% | PG -0.0022 | Val 0.3987 | BC 0.6050 | Ent 0.477 | KL 0.0021 | Clip 0.24% | Ratio 0.998+/-0.026 | EV 0.589 | SPS 132 | T r/u/e/c 39.8/13.0/0.0/0.0s
Step 2,280,992 | Ep  1314 | Stage phase_resource_control | ActorTrainable True | FirstRate 26.00% | Bombs 17.83 | VB 55.63% | UB 0.00% | NEB 0.00% | Danger 71.7 | Tiles 45.2 | Repeat 81.84% | PG -0.0022 | Val 0.2545 | BC 0.4239 | Ent 0.378 | KL 0.0016 | Clip 0.17% | Ratio 0.999+/-0.021 | EV 0.834 | SPS 132 | T r/u/e/c 22.6/13.2/0.0/0.0s
Step 2,282,016 | Ep  1319 | Stage phase_resource_control | ActorTrainable True | FirstRate 26.00% | Bombs 17.30 | VB 53.89% | UB 0.00% | NEB 0.00% | Danger 70.9 | Tiles 44.1 | Repeat 81.87% | PG -0.0011 | Val 0.1326 | BC 0.5380 | Ent 0.445 | KL 0.0007 | Clip 0.24% | Ratio 1.000+/-0.021 | EV 0.913 | SPS 131 | T r/u/e/c 27.0/13.2/0.0/0.0s
Step 2,283,040 | Ep  1324 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 17.54 | VB 54.62% | UB 0.00% | NEB 0.00% | Danger 71.6 | Tiles 44.4 | Repeat 81.81% | PG -0.0025 | Val 0.4255 | BC 0.6328 | Ent 0.493 | KL 0.0010 | Clip 0.17% | Ratio 0.999+/-0.018 | EV 0.565 | SPS 131 | T r/u/e/c 32.3/13.2/0.0/0.0s
Step 2,284,064 | Ep  1330 | Stage phase_resource_control | ActorTrainable True | FirstRate 26.00% | Bombs 16.79 | VB 54.06% | UB 0.00% | NEB 0.00% | Danger 69.2 | Tiles 43.3 | Repeat 81.87% | PG -0.0009 | Val 0.2261 | BC 0.5568 | Ent 0.451 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.736 | SPS 131 | T r/u/e/c 33.8/13.2/0.0/0.0s
Step 2,285,088 | Ep  1331 | Stage phase_resource_control | ActorTrainable True | FirstRate 26.00% | Bombs 16.76 | VB 54.11% | UB 0.00% | NEB 0.00% | Danger 69.0 | Tiles 43.2 | Repeat 81.87% | PG -0.0019 | Val 0.0720 | BC 0.6410 | Ent 0.500 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.589 | SPS 130 | T r/u/e/c 39.4/13.1/0.0/0.0s
Step 2,286,112 | Ep  1335 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 17.38 | VB 53.94% | UB 0.00% | NEB 0.00% | Danger 71.2 | Tiles 43.8 | Repeat 81.98% | PG -0.0022 | Val 0.2173 | BC 0.5971 | Ent 0.439 | KL 0.0001 | Clip 0.07% | Ratio 1.000+/-0.017 | EV 0.818 | SPS 130 | T r/u/e/c 35.9/13.1/0.0/0.0s
Step 2,287,136 | Ep  1339 | Stage phase_resource_control | ActorTrainable True | FirstRate 26.00% | Bombs 17.31 | VB 54.37% | UB 0.00% | NEB 0.00% | Danger 71.6 | Tiles 44.0 | Repeat 81.84% | PG -0.0020 | Val 0.1678 | BC 0.5436 | Ent 0.431 | KL 0.0000 | Clip 0.07% | Ratio 1.000+/-0.017 | EV 0.827 | SPS 130 | T r/u/e/c 36.8/13.5/0.0/0.0s
Step 2,288,160 | Ep  1344 | Stage phase_resource_control | ActorTrainable True | FirstRate 25.00% | Bombs 17.43 | VB 54.39% | UB 0.00% | NEB 0.00% | Danger 71.8 | Tiles 43.9 | Repeat 81.82% | PG -0.0027 | Val 0.3255 | BC 0.4053 | Ent 0.320 | KL -0.0005 | Clip 0.00% | Ratio 1.001+/-0.017 | EV 0.808 | SPS 130 | T r/u/e/c 30.6/13.1/0.0/0.0s
Step 2,289,184 | Ep  1346 | Stage phase_resource_control | ActorTrainable True | FirstRate 25.00% | Bombs 17.17 | VB 53.87% | UB 0.00% | NEB 0.00% | Danger 71.0 | Tiles 43.5 | Repeat 81.97% | PG -0.0010 | Val 0.3499 | BC 0.4304 | Ent 0.358 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.015 | EV 0.775 | SPS 129 | T r/u/e/c 25.2/13.1/0.0/0.0s
Step 2,290,208 | Ep  1350 | Stage phase_resource_control | ActorTrainable True | FirstRate 25.00% | Bombs 17.43 | VB 54.03% | UB 0.00% | NEB 0.00% | Danger 72.9 | Tiles 43.7 | Repeat 82.32% | PG -0.0011 | Val 0.1417 | BC 0.5285 | Ent 0.439 | KL 0.0007 | Clip 0.02% | Ratio 0.999+/-0.010 | EV 0.852 | SPS 129 | T r/u/e/c 24.5/13.1/0.0/0.0s
Step 2,291,232 | Ep  1353 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 17.19 | VB 54.20% | UB 0.00% | NEB 0.00% | Danger 72.5 | Tiles 43.8 | Repeat 82.21% | PG -0.0019 | Val 0.2930 | BC 0.3993 | Ent 0.298 | KL -0.0003 | Clip 0.10% | Ratio 1.000+/-0.013 | EV 0.817 | SPS 129 | T r/u/e/c 27.2/13.1/0.0/0.0s
Step 2,292,256 | Ep  1357 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 16.93 | VB 53.41% | UB 0.00% | NEB 0.00% | Danger 71.5 | Tiles 43.0 | Repeat 82.39% | PG -0.0017 | Val 0.2993 | BC 0.4360 | Ent 0.330 | KL -0.0008 | Clip 0.02% | Ratio 1.001+/-0.014 | EV 0.773 | SPS 129 | T r/u/e/c 25.8/13.1/0.0/0.0s
Step 2,293,280 | Ep  1359 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 16.88 | VB 53.65% | UB 0.00% | NEB 0.00% | Danger 70.9 | Tiles 43.3 | Repeat 82.20% | PG -0.0008 | Val 0.1123 | BC 0.3158 | Ent 0.280 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.929 | SPS 128 | T r/u/e/c 21.6/13.1/0.0/0.0s
Step 2,294,304 | Ep  1363 | Stage phase_resource_control | ActorTrainable True | FirstRate 22.00% | Bombs 16.73 | VB 54.21% | UB 0.00% | NEB 0.00% | Danger 71.2 | Tiles 43.8 | Repeat 82.27% | PG -0.0023 | Val 0.2176 | BC 0.5561 | Ent 0.418 | KL 0.0008 | Clip 0.00% | Ratio 0.999+/-0.014 | EV 0.791 | SPS 128 | T r/u/e/c 34.7/13.0/0.0/0.0s
Step 2,295,328 | Ep  1368 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 16.67 | VB 54.62% | UB 0.00% | NEB 0.00% | Danger 71.0 | Tiles 43.9 | Repeat 82.28% | PG -0.0016 | Val 0.4100 | BC 0.4600 | Ent 0.322 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.749 | SPS 128 | T r/u/e/c 25.7/13.3/0.0/0.0s
Step 2,296,352 | Ep  1372 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 16.42 | VB 53.89% | UB 0.00% | NEB 0.00% | Danger 70.4 | Tiles 43.1 | Repeat 82.80% | PG -0.0016 | Val 0.2633 | BC 0.3007 | Ent 0.227 | KL 0.0001 | Clip 0.05% | Ratio 1.000+/-0.011 | EV 0.801 | SPS 128 | T r/u/e/c 22.0/13.3/0.0/0.0s
Step 2,297,376 | Ep  1377 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 15.64 | VB 53.88% | UB 0.00% | NEB 0.00% | Danger 68.2 | Tiles 42.8 | Repeat 82.55% | PG -0.0014 | Val 0.2655 | BC 0.5906 | Ent 0.442 | KL 0.0002 | Clip 0.05% | Ratio 1.000+/-0.019 | EV 0.776 | SPS 127 | T r/u/e/c 44.8/13.2/0.0/0.0s
Step 2,298,400 | Ep  1381 | Stage phase_resource_control | ActorTrainable True | FirstRate 22.00% | Bombs 15.41 | VB 53.88% | UB 0.00% | NEB 0.00% | Danger 67.6 | Tiles 42.7 | Repeat 82.34% | PG -0.0007 | Val 0.1154 | BC 0.7051 | Ent 0.546 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.010 | EV -0.065 | SPS 127 | T r/u/e/c 30.9/13.2/0.0/0.0s
  -> Saved checkpoint: D:\other\CNTT\Bomberland-GDGoC-AI-Challenge\agent\agent\v6\checkpoints\model_step2298400.pth
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
  -> Eval score 2.100 | AvgRank 0.90 | R0/1/2/3 40%/30%/30%/0% | UF 40%
     EM score/rank 1.800/1.20 | R0/1/2/3 20%/40%/40%/0% | VB/Repeat 70.75%/82.65%
     Hard score/rank 2.400/0.60 | R0/1/2/3 60%/20%/20%/0% | VB/Repeat 63.14%/83.68%
     Diff K/B/I/Bp +0.10/-0.50/-4.60/-9.40 | Cap/Radius -1.40/-1.20 | Timeout loss K/B/I/B 0.00%/10.00%/0.00%/0.00%
     TIMING rollout 30.9s | update 13.2s | checkpoint 0.1s | eval 52.0s
Step 2,299,424 | Ep  1384 | Stage phase_resource_control | ActorTrainable True | FirstRate 23.00% | Bombs 16.05 | VB 55.47% | UB 0.00% | NEB 0.00% | Danger 69.2 | Tiles 44.2 | Repeat 81.90% | PG -0.0018 | Val 0.1761 | BC 0.4887 | Ent 0.429 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.816 | SPS 127 | T r/u/e/c 32.2/14.0/0.0/0.0s
Step 2,300,448 | Ep  1388 | Stage phase_resource_control | ActorTrainable True | FirstRate 22.00% | Bombs 15.91 | VB 54.90% | UB 0.00% | NEB 0.00% | Danger 68.8 | Tiles 43.6 | Repeat 82.22% | PG -0.0012 | Val 0.3371 | BC 0.4280 | Ent 0.317 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.005 | EV 0.713 | SPS 126 | T r/u/e/c 29.8/13.4/0.0/0.0s
Step 2,301,472 | Ep  1392 | Stage phase_resource_control | ActorTrainable True | FirstRate 23.00% | Bombs 16.51 | VB 56.61% | UB 0.00% | NEB 0.00% | Danger 71.3 | Tiles 45.3 | Repeat 81.82% | PG -0.0019 | Val 0.1910 | BC 0.5881 | Ent 0.466 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.014 | EV 0.866 | SPS 126 | T r/u/e/c 37.8/13.4/0.0/0.0s
Step 2,302,496 | Ep  1394 | Stage phase_resource_control | ActorTrainable True | FirstRate 23.00% | Bombs 16.41 | VB 56.71% | UB 0.00% | NEB 0.00% | Danger 71.1 | Tiles 44.8 | Repeat 82.10% | PG -0.0022 | Val 0.2706 | BC 0.4058 | Ent 0.319 | KL 0.0005 | Clip 0.02% | Ratio 1.000+/-0.017 | EV 0.793 | SPS 126 | T r/u/e/c 29.1/13.5/0.0/0.0s
Step 2,303,520 | Ep  1398 | Stage phase_resource_control | ActorTrainable True | FirstRate 21.00% | Bombs 16.00 | VB 56.62% | UB 0.00% | NEB 0.00% | Danger 71.2 | Tiles 44.7 | Repeat 82.69% | PG -0.0012 | Val 0.1204 | BC 0.3387 | Ent 0.295 | KL -0.0005 | Clip 0.00% | Ratio 1.001+/-0.012 | EV 0.882 | SPS 126 | T r/u/e/c 27.1/13.6/0.0/0.0s
Step 2,304,544 | Ep  1399 | Stage phase_resource_control | ActorTrainable True | FirstRate 21.00% | Bombs 15.53 | VB 56.48% | UB 0.00% | NEB 0.00% | Danger 70.0 | Tiles 44.2 | Repeat 82.78% | PG -0.0013 | Val 0.0888 | BC 0.5016 | Ent 0.434 | KL -0.0002 | Clip 0.05% | Ratio 1.000+/-0.012 | EV 0.917 | SPS 125 | T r/u/e/c 38.7/13.5/0.0/0.0s
Step 2,305,568 | Ep  1403 | Stage phase_resource_control | ActorTrainable True | FirstRate 23.00% | Bombs 15.90 | VB 57.00% | UB 0.00% | NEB 0.00% | Danger 71.9 | Tiles 45.0 | Repeat 82.56% | PG -0.0032 | Val 0.0951 | BC 0.7664 | Ent 0.561 | KL 0.0003 | Clip 0.49% | Ratio 1.000+/-0.025 | EV 0.610 | SPS 125 | T r/u/e/c 39.5/13.4/0.0/0.0s
Step 2,306,592 | Ep  1405 | Stage phase_resource_control | ActorTrainable True | FirstRate 25.00% | Bombs 16.01 | VB 57.15% | UB 0.00% | NEB 0.00% | Danger 71.6 | Tiles 45.5 | Repeat 82.27% | PG -0.0026 | Val 0.1473 | BC 0.5691 | Ent 0.439 | KL 0.0005 | Clip 0.37% | Ratio 1.000+/-0.026 | EV 0.881 | SPS 125 | T r/u/e/c 30.9/13.4/0.0/0.0s
Step 2,307,616 | Ep  1410 | Stage phase_resource_control | ActorTrainable True | FirstRate 26.00% | Bombs 16.11 | VB 57.06% | UB 0.00% | NEB 0.00% | Danger 72.0 | Tiles 45.7 | Repeat 82.40% | PG -0.0012 | Val 0.2676 | BC 0.5114 | Ent 0.403 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.015 | EV 0.783 | SPS 124 | T r/u/e/c 29.7/13.5/0.0/0.0s
Step 2,308,640 | Ep  1414 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 16.36 | VB 57.85% | UB 0.00% | NEB 0.00% | Danger 72.8 | Tiles 46.1 | Repeat 82.15% | PG -0.0009 | Val 0.2387 | BC 0.7258 | Ent 0.508 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.255 | SPS 124 | T r/u/e/c 43.8/13.5/0.0/0.0s
Step 2,309,664 | Ep  1418 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 16.56 | VB 58.02% | UB 0.00% | NEB 0.00% | Danger 73.7 | Tiles 46.1 | Repeat 82.24% | PG -0.0015 | Val 0.1922 | BC 0.7066 | Ent 0.430 | KL -0.0004 | Clip 0.00% | Ratio 1.001+/-0.015 | EV 0.806 | SPS 124 | T r/u/e/c 34.8/13.6/0.0/0.0s
Step 2,310,688 | Ep  1423 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 16.65 | VB 58.32% | UB 0.00% | NEB 0.00% | Danger 73.1 | Tiles 46.5 | Repeat 82.00% | PG -0.0019 | Val 0.2392 | BC 0.4849 | Ent 0.370 | KL 0.0008 | Clip 0.02% | Ratio 0.999+/-0.015 | EV 0.852 | SPS 124 | T r/u/e/c 25.9/13.6/0.0/0.0s
Step 2,311,712 | Ep  1427 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 16.54 | VB 58.58% | UB 0.00% | NEB 0.00% | Danger 72.7 | Tiles 46.6 | Repeat 81.94% | PG -0.0019 | Val 0.3661 | BC 0.4589 | Ent 0.344 | KL 0.0003 | Clip 0.10% | Ratio 1.000+/-0.013 | EV 0.737 | SPS 123 | T r/u/e/c 31.5/13.4/0.0/0.0s
Step 2,312,736 | Ep  1431 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 16.46 | VB 58.54% | UB 0.00% | NEB 0.00% | Danger 72.5 | Tiles 45.8 | Repeat 82.46% | PG -0.0020 | Val 0.1786 | BC 0.2295 | Ent 0.196 | KL 0.0001 | Clip 0.10% | Ratio 1.000+/-0.011 | EV 0.891 | SPS 123 | T r/u/e/c 28.7/13.8/0.0/0.0s
Step 2,313,760 | Ep  1434 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 15.76 | VB 57.81% | UB 0.00% | NEB 0.00% | Danger 69.7 | Tiles 44.9 | Repeat 82.26% | PG -0.0013 | Val 0.0719 | BC 0.5561 | Ent 0.404 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.013 | EV 0.941 | SPS 123 | T r/u/e/c 34.1/13.7/0.0/0.0s
Step 2,314,784 | Ep  1437 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 15.74 | VB 58.29% | UB 0.00% | NEB 0.00% | Danger 70.3 | Tiles 45.0 | Repeat 82.30% | PG -0.0015 | Val 0.3277 | BC 0.4653 | Ent 0.365 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.848 | SPS 123 | T r/u/e/c 36.9/13.5/0.0/0.0s
Step 2,315,808 | Ep  1443 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 16.12 | VB 58.71% | UB 0.00% | NEB 0.00% | Danger 71.6 | Tiles 45.7 | Repeat 81.68% | PG -0.0020 | Val 0.1594 | BC 0.5643 | Ent 0.458 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.893 | SPS 122 | T r/u/e/c 33.6/13.5/0.0/0.0s
Step 2,316,832 | Ep  1446 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 15.88 | VB 59.01% | UB 0.00% | NEB 0.00% | Danger 70.2 | Tiles 46.0 | Repeat 81.27% | PG -0.0018 | Val 0.3087 | BC 0.4551 | Ent 0.359 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.853 | SPS 122 | T r/u/e/c 33.4/13.7/0.0/0.0s
Step 2,317,856 | Ep  1450 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 15.73 | VB 58.56% | UB 0.00% | NEB 0.00% | Danger 69.2 | Tiles 45.9 | Repeat 81.14% | PG -0.0019 | Val 0.2442 | BC 0.3548 | Ent 0.332 | KL 0.0010 | Clip 0.39% | Ratio 0.999+/-0.022 | EV 0.869 | SPS 122 | T r/u/e/c 24.6/13.5/0.0/0.0s
Step 2,318,880 | Ep  1453 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 15.37 | VB 58.08% | UB 0.00% | NEB 0.00% | Danger 67.8 | Tiles 44.6 | Repeat 81.51% | PG -0.0018 | Val 0.2302 | BC 0.3867 | Ent 0.312 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.871 | SPS 122 | T r/u/e/c 23.0/13.5/0.0/0.0s
Step 2,319,904 | Ep  1459 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 15.64 | VB 58.98% | UB 0.00% | NEB 0.00% | Danger 68.3 | Tiles 45.3 | Repeat 81.24% | PG -0.0013 | Val 0.3309 | BC 0.6005 | Ent 0.484 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.695 | SPS 122 | T r/u/e/c 37.1/13.3/0.0/0.0s
Step 2,320,928 | Ep  1461 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 15.59 | VB 58.84% | UB 0.00% | NEB 0.00% | Danger 68.2 | Tiles 45.1 | Repeat 81.05% | PG -0.0018 | Val 0.2427 | BC 0.5289 | Ent 0.428 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.013 | EV 0.811 | SPS 121 | T r/u/e/c 32.2/13.5/0.0/0.0s
Step 2,321,952 | Ep  1465 | Stage phase_resource_control | ActorTrainable True | FirstRate 36.00% | Bombs 16.05 | VB 59.25% | UB 0.00% | NEB 0.00% | Danger 69.8 | Tiles 45.8 | Repeat 81.04% | PG -0.0020 | Val 0.1123 | BC 0.6108 | Ent 0.494 | KL 0.0006 | Clip 0.05% | Ratio 1.000+/-0.017 | EV 0.842 | SPS 121 | T r/u/e/c 44.0/13.6/0.0/0.0s
Step 2,322,976 | Ep  1469 | Stage phase_resource_control | ActorTrainable True | FirstRate 36.00% | Bombs 16.53 | VB 60.60% | UB 0.00% | NEB 0.00% | Danger 72.0 | Tiles 46.5 | Repeat 80.94% | PG -0.0022 | Val 0.2590 | BC 0.6435 | Ent 0.529 | KL 0.0003 | Clip 0.49% | Ratio 1.000+/-0.026 | EV 0.612 | SPS 121 | T r/u/e/c 41.4/13.5/0.0/0.0s
Step 2,324,000 | Ep  1474 | Stage phase_resource_control | ActorTrainable True | FirstRate 35.00% | Bombs 16.42 | VB 61.25% | UB 0.00% | NEB 0.00% | Danger 71.8 | Tiles 46.6 | Repeat 80.76% | PG -0.0020 | Val 0.1515 | BC 0.4432 | Ent 0.375 | KL 0.0007 | Clip 0.22% | Ratio 1.000+/-0.021 | EV 0.869 | SPS 120 | T r/u/e/c 35.9/13.4/0.0/0.0s
  -> Saved checkpoint: D:\other\CNTT\Bomberland-GDGoC-AI-Challenge\agent\agent\v6\checkpoints\model_step2324000.pth
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
  -> Eval score 2.300 | AvgRank 0.70 | R0/1/2/3 50%/30%/20%/0% | UF 40%
     EM score/rank 2.600/0.40 | R0/1/2/3 80%/0%/20%/0% | VB/Repeat 55.29%/74.80%
     Hard score/rank 2.000/1.00 | R0/1/2/3 20%/60%/20%/0% | VB/Repeat 68.20%/84.31%
     Diff K/B/I/Bp -0.60/-2.00/-3.50/-11.40 | Cap/Radius -1.00/-1.90 | Timeout loss K/B/I/B 10.00%/0.00%/0.00%/0.00%
     TIMING rollout 35.9s | update 13.4s | checkpoint 0.1s | eval 40.8s
Step 2,325,024 | Ep  1475 | Stage phase_resource_control | ActorTrainable True | FirstRate 35.00% | Bombs 16.54 | VB 61.56% | UB 0.00% | NEB 0.00% | Danger 72.6 | Tiles 47.2 | Repeat 80.54% | PG -0.0014 | Val 0.3026 | BC 0.5092 | Ent 0.412 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.840 | SPS 120 | T r/u/e/c 41.1/13.1/0.0/0.0s
Step 2,326,048 | Ep  1481 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 16.65 | VB 60.92% | UB 0.00% | NEB 0.00% | Danger 74.4 | Tiles 47.5 | Repeat 81.03% | PG -0.0020 | Val 0.1638 | BC 0.6747 | Ent 0.512 | KL 0.0008 | Clip 0.00% | Ratio 0.999+/-0.018 | EV 0.675 | SPS 120 | T r/u/e/c 41.0/13.1/0.0/0.0s
Step 2,327,072 | Ep  1484 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 16.09 | VB 60.19% | UB 0.00% | NEB 0.00% | Danger 72.9 | Tiles 46.5 | Repeat 81.21% | PG -0.0023 | Val 0.2756 | BC 0.6542 | Ent 0.477 | KL 0.0013 | Clip 0.20% | Ratio 0.999+/-0.025 | EV 0.770 | SPS 119 | T r/u/e/c 37.5/13.1/0.0/0.0s
Step 2,328,096 | Ep  1486 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 16.24 | VB 60.73% | UB 0.00% | NEB 0.00% | Danger 73.4 | Tiles 46.9 | Repeat 81.02% | PG -0.0014 | Val 0.3355 | BC 0.4007 | Ent 0.326 | KL -0.0015 | Clip 0.39% | Ratio 1.002+/-0.023 | EV 0.842 | SPS 119 | T r/u/e/c 29.2/13.1/0.0/0.0s
Step 2,329,120 | Ep  1489 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 16.26 | VB 60.67% | UB 0.00% | NEB 0.00% | Danger 73.8 | Tiles 46.9 | Repeat 81.07% | PG -0.0021 | Val 0.1030 | BC 0.5085 | Ent 0.434 | KL 0.0004 | Clip 0.02% | Ratio 1.000+/-0.014 | EV 0.922 | SPS 119 | T r/u/e/c 29.6/13.1/0.0/0.0s
Step 2,330,144 | Ep  1493 | Stage phase_resource_control | ActorTrainable True | FirstRate 35.00% | Bombs 16.09 | VB 60.38% | UB 0.00% | NEB 0.00% | Danger 73.5 | Tiles 46.9 | Repeat 81.15% | PG -0.0020 | Val 0.2365 | BC 0.5697 | Ent 0.488 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.740 | SPS 119 | T r/u/e/c 42.1/13.1/0.0/0.0s
Step 2,331,168 | Ep  1497 | Stage phase_resource_control | ActorTrainable True | FirstRate 37.00% | Bombs 16.20 | VB 60.43% | UB 0.00% | NEB 0.00% | Danger 73.7 | Tiles 47.3 | Repeat 80.92% | PG -0.0008 | Val 0.1278 | BC 0.4907 | Ent 0.369 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.006 | EV 0.938 | SPS 118 | T r/u/e/c 39.5/13.1/0.0/0.0s
Step 2,332,192 | Ep  1502 | Stage phase_resource_control | ActorTrainable True | FirstRate 40.00% | Bombs 16.31 | VB 59.75% | UB 0.00% | NEB 0.00% | Danger 73.2 | Tiles 46.7 | Repeat 80.66% | PG -0.0017 | Val 0.1470 | BC 0.6813 | Ent 0.487 | KL 0.0006 | Clip 0.00% | Ratio 1.000+/-0.013 | EV 0.517 | SPS 118 | T r/u/e/c 42.5/13.3/0.0/0.0s
Step 2,333,216 | Ep  1505 | Stage phase_resource_control | ActorTrainable True | FirstRate 39.00% | Bombs 16.05 | VB 58.97% | UB 0.00% | NEB 0.00% | Danger 72.1 | Tiles 46.4 | Repeat 80.47% | PG -0.0027 | Val 0.1388 | BC 0.5694 | Ent 0.460 | KL 0.0007 | Clip 0.05% | Ratio 0.999+/-0.015 | EV 0.850 | SPS 118 | T r/u/e/c 35.2/13.2/0.0/0.0s
Step 2,334,240 | Ep  1509 | Stage phase_resource_control | ActorTrainable True | FirstRate 39.00% | Bombs 16.28 | VB 59.10% | UB 0.00% | NEB 0.00% | Danger 72.2 | Tiles 46.6 | Repeat 80.38% | PG -0.0025 | Val 0.2802 | BC 0.6392 | Ent 0.472 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.746 | SPS 118 | T r/u/e/c 43.7/13.2/0.0/0.0s
Step 2,335,264 | Ep  1514 | Stage phase_resource_control | ActorTrainable True | FirstRate 36.00% | Bombs 15.66 | VB 57.79% | UB 0.00% | NEB 0.00% | Danger 70.4 | Tiles 45.8 | Repeat 80.54% | PG -0.0021 | Val 0.2125 | BC 0.5097 | Ent 0.393 | KL -0.0004 | Clip 0.00% | Ratio 1.001+/-0.011 | EV 0.790 | SPS 117 | T r/u/e/c 27.3/13.2/0.0/0.0s
Step 2,336,288 | Ep  1518 | Stage phase_resource_control | ActorTrainable True | FirstRate 35.00% | Bombs 15.47 | VB 58.46% | UB 0.00% | NEB 0.00% | Danger 70.4 | Tiles 45.9 | Repeat 80.79% | PG -0.0015 | Val 0.3499 | BC 0.5360 | Ent 0.370 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.773 | SPS 117 | T r/u/e/c 20.4/13.2/0.0/0.0s
Step 2,337,312 | Ep  1521 | Stage phase_resource_control | ActorTrainable True | FirstRate 35.00% | Bombs 15.30 | VB 58.65% | UB 0.00% | NEB 0.00% | Danger 69.4 | Tiles 46.0 | Repeat 80.69% | PG -0.0014 | Val 0.1767 | BC 0.4152 | Ent 0.357 | KL -0.0003 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.892 | SPS 117 | T r/u/e/c 29.0/13.2/0.0/0.0s
Step 2,338,336 | Ep  1527 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 15.50 | VB 58.09% | UB 0.00% | NEB 0.00% | Danger 70.5 | Tiles 45.5 | Repeat 81.03% | PG -0.0018 | Val 0.3672 | BC 0.5107 | Ent 0.394 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.754 | SPS 117 | T r/u/e/c 30.2/13.0/0.0/0.0s
Step 2,339,360 | Ep  1531 | Stage phase_resource_control | ActorTrainable True | FirstRate 36.00% | Bombs 15.71 | VB 58.47% | UB 0.00% | NEB 0.00% | Danger 71.5 | Tiles 46.4 | Repeat 80.50% | PG -0.0021 | Val 0.2214 | BC 0.5847 | Ent 0.517 | KL -0.0002 | Clip 0.02% | Ratio 1.000+/-0.009 | EV 0.690 | SPS 117 | T r/u/e/c 39.0/13.1/0.0/0.0s
Step 2,340,384 | Ep  1534 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 15.82 | VB 58.48% | UB 0.00% | NEB 0.00% | Danger 72.1 | Tiles 46.8 | Repeat 80.69% | PG -0.0012 | Val 0.1500 | BC 0.5094 | Ent 0.451 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.878 | SPS 116 | T r/u/e/c 40.9/13.3/0.0/0.0s
Step 2,341,408 | Ep  1538 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 15.46 | VB 57.41% | UB 0.00% | NEB 0.00% | Danger 70.0 | Tiles 46.4 | Repeat 80.60% | PG -0.0017 | Val 0.3750 | BC 0.4000 | Ent 0.313 | KL 0.0012 | Clip 0.02% | Ratio 0.999+/-0.013 | EV 0.733 | SPS 116 | T r/u/e/c 31.2/13.1/0.0/0.0s
Step 2,342,432 | Ep  1542 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 15.42 | VB 57.48% | UB 0.00% | NEB 0.00% | Danger 69.3 | Tiles 46.3 | Repeat 80.65% | PG -0.0011 | Val 0.3046 | BC 0.2350 | Ent 0.196 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.779 | SPS 116 | T r/u/e/c 17.2/13.2/0.0/0.0s
Step 2,343,456 | Ep  1546 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 15.25 | VB 56.41% | UB 0.00% | NEB 0.00% | Danger 69.4 | Tiles 45.5 | Repeat 81.40% | PG -0.0017 | Val 0.4103 | BC 0.3890 | Ent 0.323 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.740 | SPS 116 | T r/u/e/c 33.0/13.1/0.0/0.0s
Step 2,344,480 | Ep  1550 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 15.23 | VB 56.86% | UB 0.00% | NEB 0.00% | Danger 70.1 | Tiles 45.1 | Repeat 81.73% | PG -0.0008 | Val 0.1913 | BC 0.4744 | Ent 0.367 | KL -0.0004 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.874 | SPS 116 | T r/u/e/c 25.8/13.2/0.0/0.0s
Step 2,345,504 | Ep  1551 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 15.38 | VB 56.81% | UB 0.00% | NEB 0.00% | Danger 70.5 | Tiles 45.4 | Repeat 81.46% | PG -0.0016 | Val 0.2590 | BC 0.3978 | Ent 0.353 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.859 | SPS 115 | T r/u/e/c 30.4/13.2/0.0/0.0s
Step 2,346,528 | Ep  1555 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 15.65 | VB 56.73% | UB 0.00% | NEB 0.00% | Danger 72.8 | Tiles 46.0 | Repeat 81.48% | PG -0.0012 | Val 0.1912 | BC 0.5100 | Ent 0.429 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.013 | EV 0.850 | SPS 115 | T r/u/e/c 34.4/13.1/0.0/0.0s
Step 2,347,552 | Ep  1559 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 15.93 | VB 57.92% | UB 0.00% | NEB 0.00% | Danger 74.6 | Tiles 46.6 | Repeat 81.65% | PG -0.0015 | Val 0.3429 | BC 0.4818 | Ent 0.349 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.760 | SPS 115 | T r/u/e/c 33.9/13.1/0.0/0.0s
Step 2,348,576 | Ep  1562 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 15.75 | VB 57.31% | UB 0.00% | NEB 0.00% | Danger 74.0 | Tiles 46.1 | Repeat 81.81% | PG -0.0020 | Val 0.2228 | BC 0.5937 | Ent 0.475 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.016 | EV 0.800 | SPS 115 | T r/u/e/c 36.7/13.1/0.0/0.0s
Step 2,349,600 | Ep  1566 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 15.46 | VB 56.90% | UB 0.00% | NEB 0.00% | Danger 73.4 | Tiles 46.1 | Repeat 81.81% | PG -0.0026 | Val 0.1282 | BC 0.3193 | Ent 0.314 | KL -0.0001 | Clip 0.15% | Ratio 1.000+/-0.017 | EV 0.912 | SPS 115 | T r/u/e/c 29.8/13.2/0.0/0.0s
  -> Saved checkpoint: D:\other\CNTT\Bomberland-GDGoC-AI-Challenge\agent\agent\v6\checkpoints\model_step2349600.pth
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
  -> Eval score 2.500 | AvgRank 0.50 | R0/1/2/3 60%/30%/10%/0% | UF 60%
     EM score/rank 2.400/0.60 | R0/1/2/3 60%/20%/20%/0% | VB/Repeat 65.65%/79.07%
     Hard score/rank 2.600/0.40 | R0/1/2/3 60%/40%/0%/0% | VB/Repeat 72.18%/76.70%
     Diff K/B/I/Bp +0.20/-1.10/-4.00/-8.80 | Cap/Radius -0.80/-1.50 | Timeout loss K/B/I/B 0.00%/0.00%/0.00%/0.00%
     TIMING rollout 29.8s | update 13.2s | checkpoint 0.1s | eval 58.5s
Step 2,350,624 | Ep  1570 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 15.21 | VB 55.91% | UB 0.00% | NEB 0.00% | Danger 72.4 | Tiles 45.4 | Repeat 82.26% | PG -0.0019 | Val 0.1756 | BC 0.5458 | Ent 0.436 | KL -0.0002 | Clip 0.05% | Ratio 1.000+/-0.017 | EV 0.817 | SPS 114 | T r/u/e/c 43.8/13.8/0.0/0.0s
Step 2,351,648 | Ep  1574 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 15.18 | VB 55.52% | UB 0.00% | NEB 0.00% | Danger 72.1 | Tiles 45.5 | Repeat 82.01% | PG -0.0018 | Val 0.2877 | BC 0.6017 | Ent 0.541 | KL 0.0001 | Clip 0.10% | Ratio 1.000+/-0.020 | EV -0.080 | SPS 114 | T r/u/e/c 44.4/13.7/0.0/0.0s
Step 2,352,672 | Ep  1577 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 15.65 | VB 55.99% | UB 0.00% | NEB 0.00% | Danger 72.9 | Tiles 46.1 | Repeat 81.92% | PG -0.0011 | Val 0.2335 | BC 0.5185 | Ent 0.473 | KL -0.0004 | Clip 0.05% | Ratio 1.001+/-0.017 | EV 0.800 | SPS 113 | T r/u/e/c 41.1/13.8/0.0/0.0s
Step 2,353,696 | Ep  1581 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 15.37 | VB 55.59% | UB 0.00% | NEB 0.00% | Danger 70.9 | Tiles 45.8 | Repeat 81.91% | PG -0.0024 | Val 0.1104 | BC 0.6163 | Ent 0.503 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.013 | EV 0.804 | SPS 113 | T r/u/e/c 37.8/13.6/0.0/0.0s
Step 2,354,720 | Ep  1584 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 15.46 | VB 55.72% | UB 0.00% | NEB 0.00% | Danger 70.7 | Tiles 46.1 | Repeat 81.51% | PG -0.0003 | Val 0.1558 | BC 0.7450 | Ent 0.608 | KL 0.0000 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.498 | SPS 113 | T r/u/e/c 39.0/13.6/0.0/0.0s
Step 2,355,744 | Ep  1589 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 15.44 | VB 55.84% | UB 0.00% | NEB 0.00% | Danger 70.7 | Tiles 46.7 | Repeat 81.22% | PG -0.0025 | Val 0.1937 | BC 0.6147 | Ent 0.475 | KL -0.0003 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.755 | SPS 113 | T r/u/e/c 38.3/13.5/0.0/0.0s
Step 2,356,768 | Ep  1593 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 15.21 | VB 55.01% | UB 0.00% | NEB 0.00% | Danger 69.8 | Tiles 46.1 | Repeat 81.27% | PG -0.0016 | Val 0.3901 | BC 0.4806 | Ent 0.371 | KL -0.0002 | Clip 0.10% | Ratio 1.000+/-0.014 | EV 0.682 | SPS 112 | T r/u/e/c 39.6/13.6/0.0/0.0s
Step 2,357,792 | Ep  1596 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 15.20 | VB 55.24% | UB 0.00% | NEB 0.00% | Danger 70.3 | Tiles 45.7 | Repeat 81.59% | PG -0.0011 | Val 0.1885 | BC 0.3504 | Ent 0.268 | KL 0.0001 | Clip 0.02% | Ratio 1.000+/-0.008 | EV 0.865 | SPS 112 | T r/u/e/c 34.8/14.0/0.0/0.0s
Step 2,358,816 | Ep  1599 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 15.58 | VB 56.25% | UB 0.00% | NEB 0.00% | Danger 72.2 | Tiles 46.8 | Repeat 81.41% | PG -0.0017 | Val 0.0982 | BC 0.5718 | Ent 0.517 | KL 0.0006 | Clip 0.02% | Ratio 1.000+/-0.014 | EV 0.371 | SPS 112 | T r/u/e/c 35.8/13.9/0.0/0.0s
Step 2,359,840 | Ep  1602 | Stage phase_resource_control | ActorTrainable True | FirstRate 26.00% | Bombs 15.77 | VB 56.72% | UB 0.00% | NEB 0.00% | Danger 73.4 | Tiles 47.6 | Repeat 81.47% | PG -0.0015 | Val 0.1594 | BC 0.5392 | Ent 0.510 | KL 0.0005 | Clip 0.05% | Ratio 1.000+/-0.016 | EV 0.669 | SPS 112 | T r/u/e/c 32.8/13.5/0.0/0.0s
Step 2,360,864 | Ep  1603 | Stage phase_resource_control | ActorTrainable True | FirstRate 25.00% | Bombs 15.82 | VB 56.70% | UB 0.00% | NEB 0.00% | Danger 73.5 | Tiles 47.7 | Repeat 81.47% | PG -0.0015 | Val 0.0414 | BC 0.4350 | Ent 0.428 | KL 0.0005 | Clip 0.02% | Ratio 1.000+/-0.013 | EV 0.970 | SPS 112 | T r/u/e/c 33.3/13.4/0.0/0.0s
Step 2,361,888 | Ep  1606 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 16.15 | VB 57.07% | UB 0.00% | NEB 0.00% | Danger 75.8 | Tiles 48.4 | Repeat 81.87% | PG -0.0024 | Val 0.1339 | BC 0.4885 | Ent 0.422 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.910 | SPS 111 | T r/u/e/c 36.0/13.4/0.0/0.0s
Step 2,362,912 | Ep  1610 | Stage phase_resource_control | ActorTrainable True | FirstRate 23.00% | Bombs 15.79 | VB 55.67% | UB 0.00% | NEB 0.00% | Danger 75.3 | Tiles 47.2 | Repeat 82.13% | PG -0.0012 | Val 0.2426 | BC 0.3417 | Ent 0.253 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.835 | SPS 111 | T r/u/e/c 27.5/13.5/0.0/0.0s
Step 2,363,936 | Ep  1612 | Stage phase_resource_control | ActorTrainable True | FirstRate 22.00% | Bombs 15.61 | VB 55.74% | UB 0.00% | NEB 0.00% | Danger 75.0 | Tiles 46.8 | Repeat 82.44% | PG -0.0017 | Val 0.2669 | BC 0.3620 | Ent 0.274 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.752 | SPS 111 | T r/u/e/c 26.4/13.6/0.0/0.0s
Step 2,364,960 | Ep  1616 | Stage phase_resource_control | ActorTrainable True | FirstRate 23.00% | Bombs 15.70 | VB 56.78% | UB 0.00% | NEB 0.00% | Danger 75.5 | Tiles 47.2 | Repeat 82.25% | PG -0.0011 | Val 0.3037 | BC 0.4868 | Ent 0.371 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.781 | SPS 111 | T r/u/e/c 25.7/13.9/0.0/0.0s
Step 2,365,984 | Ep  1619 | Stage phase_resource_control | ActorTrainable True | FirstRate 22.00% | Bombs 15.66 | VB 56.60% | UB 0.00% | NEB 0.00% | Danger 75.6 | Tiles 47.0 | Repeat 82.51% | PG -0.0019 | Val 0.2159 | BC 0.4636 | Ent 0.410 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.015 | EV 0.775 | SPS 111 | T r/u/e/c 31.0/13.5/0.0/0.0s
Step 2,367,008 | Ep  1622 | Stage phase_resource_control | ActorTrainable True | FirstRate 21.00% | Bombs 15.90 | VB 56.84% | UB 0.00% | NEB 0.00% | Danger 76.8 | Tiles 48.2 | Repeat 82.34% | PG -0.0021 | Val 0.2022 | BC 0.3792 | Ent 0.351 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.017 | EV 0.894 | SPS 111 | T r/u/e/c 30.1/13.7/0.0/0.0s
Step 2,368,032 | Ep  1627 | Stage phase_resource_control | ActorTrainable True | FirstRate 21.00% | Bombs 15.61 | VB 57.07% | UB 0.00% | NEB 0.00% | Danger 76.3 | Tiles 48.0 | Repeat 82.23% | PG -0.0024 | Val 0.2014 | BC 0.3618 | Ent 0.309 | KL 0.0000 | Clip 0.12% | Ratio 1.000+/-0.020 | EV 0.824 | SPS 110 | T r/u/e/c 22.7/13.5/0.0/0.0s
Step 2,369,056 | Ep  1633 | Stage phase_resource_control | ActorTrainable True | FirstRate 19.00% | Bombs 15.21 | VB 56.44% | UB 0.00% | NEB 0.00% | Danger 74.9 | Tiles 47.1 | Repeat 82.24% | PG -0.0023 | Val 0.2593 | BC 0.4205 | Ent 0.344 | KL 0.0005 | Clip 0.10% | Ratio 1.000+/-0.020 | EV 0.819 | SPS 110 | T r/u/e/c 22.4/13.5/0.0/0.0s
Step 2,370,080 | Ep  1637 | Stage phase_resource_control | ActorTrainable True | FirstRate 19.00% | Bombs 14.79 | VB 54.72% | UB 0.00% | NEB 0.00% | Danger 74.7 | Tiles 45.8 | Repeat 82.74% | PG -0.0016 | Val 0.3539 | BC 0.2374 | Ent 0.173 | KL 0.0007 | Clip 0.02% | Ratio 0.999+/-0.013 | EV 0.658 | SPS 110 | T r/u/e/c 22.7/13.8/0.0/0.0s
Step 2,371,104 | Ep  1641 | Stage phase_resource_control | ActorTrainable True | FirstRate 19.00% | Bombs 14.65 | VB 53.41% | UB 0.00% | NEB 0.00% | Danger 74.3 | Tiles 45.1 | Repeat 82.99% | PG -0.0010 | Val 0.4075 | BC 0.4073 | Ent 0.298 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.672 | SPS 110 | T r/u/e/c 27.3/13.7/0.0/0.0s
Step 2,372,128 | Ep  1644 | Stage phase_resource_control | ActorTrainable True | FirstRate 18.00% | Bombs 14.76 | VB 54.65% | UB 0.00% | NEB 0.00% | Danger 74.9 | Tiles 45.4 | Repeat 83.01% | PG -0.0017 | Val 0.2697 | BC 0.1954 | Ent 0.150 | KL 0.0002 | Clip 0.02% | Ratio 1.000+/-0.010 | EV 0.676 | SPS 110 | T r/u/e/c 22.7/13.6/0.0/0.0s
Step 2,373,152 | Ep  1646 | Stage phase_resource_control | ActorTrainable True | FirstRate 18.00% | Bombs 14.86 | VB 54.80% | UB 0.00% | NEB 0.00% | Danger 75.2 | Tiles 45.8 | Repeat 83.04% | PG -0.0030 | Val 0.2668 | BC 0.5054 | Ent 0.423 | KL 0.0013 | Clip 0.12% | Ratio 0.999+/-0.022 | EV 0.703 | SPS 110 | T r/u/e/c 33.0/13.6/0.0/0.0s
Step 2,374,176 | Ep  1653 | Stage phase_resource_control | ActorTrainable True | FirstRate 16.00% | Bombs 15.40 | VB 55.07% | UB 0.00% | NEB 0.00% | Danger 75.3 | Tiles 46.3 | Repeat 82.99% | PG -0.0021 | Val 0.2578 | BC 0.4667 | Ent 0.378 | KL 0.0008 | Clip 0.05% | Ratio 1.000+/-0.019 | EV 0.796 | SPS 110 | T r/u/e/c 34.5/13.7/0.0/0.0s
Step 2,375,200 | Ep  1655 | Stage phase_resource_control | ActorTrainable True | FirstRate 15.00% | Bombs 15.09 | VB 54.45% | UB 0.00% | NEB 0.00% | Danger 72.7 | Tiles 45.8 | Repeat 82.77% | PG -0.0011 | Val 0.2108 | BC 0.5000 | Ent 0.395 | KL -0.0003 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.833 | SPS 109 | T r/u/e/c 29.7/13.5/0.0/0.0s
  -> Saved checkpoint: D:\other\CNTT\Bomberland-GDGoC-AI-Challenge\agent\agent\v6\checkpoints\model_step2375200.pth
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
  -> Eval score 2.400 | AvgRank 0.60 | R0/1/2/3 60%/20%/20%/0% | UF 60%
     EM score/rank 2.400/0.60 | R0/1/2/3 60%/20%/20%/0% | VB/Repeat 40.89%/73.81%
     Hard score/rank 2.400/0.60 | R0/1/2/3 60%/20%/20%/0% | VB/Repeat 64.75%/78.10%
     Diff K/B/I/Bp -0.20/-0.70/-4.50/-11.30 | Cap/Radius -1.50/-1.20 | Timeout loss K/B/I/B 0.00%/10.00%/0.00%/0.00%
     TIMING rollout 29.7s | update 13.5s | checkpoint 0.1s | eval 44.2s
Step 2,376,224 | Ep  1658 | Stage phase_resource_control | ActorTrainable True | FirstRate 16.00% | Bombs 15.18 | VB 53.73% | UB 0.00% | NEB 0.00% | Danger 73.0 | Tiles 45.8 | Repeat 82.78% | PG -0.0015 | Val 0.1508 | BC 0.6095 | Ent 0.518 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.729 | SPS 109 | T r/u/e/c 33.6/13.2/0.0/0.0s
Step 2,377,248 | Ep  1664 | Stage phase_resource_control | ActorTrainable True | FirstRate 18.00% | Bombs 14.68 | VB 53.20% | UB 0.00% | NEB 0.00% | Danger 70.6 | Tiles 45.5 | Repeat 82.47% | PG -0.0020 | Val 0.2634 | BC 0.5166 | Ent 0.410 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.772 | SPS 109 | T r/u/e/c 33.1/13.1/0.0/0.0s
Step 2,378,272 | Ep  1664 | Stage phase_resource_control | ActorTrainable True | FirstRate 18.00% | Bombs 14.68 | VB 53.20% | UB 0.00% | NEB 0.00% | Danger 70.6 | Tiles 45.5 | Repeat 82.47% | PG -0.0022 | Val 0.1890 | BC 0.3980 | Ent 0.323 | KL 0.0005 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.882 | SPS 109 | T r/u/e/c 37.4/13.1/0.0/0.0s
Step 2,379,296 | Ep  1669 | Stage phase_resource_control | ActorTrainable True | FirstRate 18.00% | Bombs 15.64 | VB 54.68% | UB 0.00% | NEB 0.00% | Danger 73.6 | Tiles 46.8 | Repeat 82.28% | PG -0.0014 | Val 0.1011 | BC 0.3763 | Ent 0.321 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.929 | SPS 108 | T r/u/e/c 28.9/13.1/0.0/0.0s
Step 2,380,320 | Ep  1672 | Stage phase_resource_control | ActorTrainable True | FirstRate 18.00% | Bombs 15.47 | VB 53.76% | UB 0.00% | NEB 0.00% | Danger 72.4 | Tiles 46.5 | Repeat 82.30% | PG -0.0014 | Val 0.0670 | BC 0.6385 | Ent 0.495 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.854 | SPS 108 | T r/u/e/c 35.4/13.3/0.0/0.0s
Step 2,381,344 | Ep  1678 | Stage phase_resource_control | ActorTrainable True | FirstRate 17.00% | Bombs 15.24 | VB 53.80% | UB 0.00% | NEB 0.00% | Danger 71.3 | Tiles 45.8 | Repeat 82.24% | PG -0.0017 | Val 0.2634 | BC 0.6674 | Ent 0.515 | KL 0.0007 | Clip 0.00% | Ratio 0.999+/-0.011 | EV 0.357 | SPS 108 | T r/u/e/c 40.0/13.1/0.0/0.0s
Step 2,382,368 | Ep  1681 | Stage phase_resource_control | ActorTrainable True | FirstRate 17.00% | Bombs 15.46 | VB 54.04% | UB 0.00% | NEB 0.00% | Danger 72.6 | Tiles 46.0 | Repeat 82.51% | PG -0.0018 | Val 0.2110 | BC 0.5191 | Ent 0.399 | KL -0.0002 | Clip 0.12% | Ratio 1.000+/-0.017 | EV 0.843 | SPS 108 | T r/u/e/c 41.7/13.0/0.0/0.0s
Step 2,383,392 | Ep  1683 | Stage phase_resource_control | ActorTrainable True | FirstRate 17.00% | Bombs 15.76 | VB 54.28% | UB 0.00% | NEB 0.00% | Danger 73.6 | Tiles 46.0 | Repeat 82.64% | PG -0.0007 | Val 0.2254 | BC 0.5781 | Ent 0.475 | KL 0.0005 | Clip 0.02% | Ratio 1.000+/-0.015 | EV 0.716 | SPS 108 | T r/u/e/c 42.9/13.1/0.0/0.0s
Step 2,384,416 | Ep  1688 | Stage phase_resource_control | ActorTrainable True | FirstRate 16.00% | Bombs 15.90 | VB 53.30% | UB 0.00% | NEB 0.00% | Danger 73.4 | Tiles 45.5 | Repeat 83.02% | PG -0.0019 | Val 0.1982 | BC 0.4569 | Ent 0.401 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.849 | SPS 107 | T r/u/e/c 36.6/13.1/0.0/0.0s
Step 2,385,440 | Ep  1690 | Stage phase_resource_control | ActorTrainable True | FirstRate 16.00% | Bombs 15.85 | VB 52.99% | UB 0.00% | NEB 0.00% | Danger 73.6 | Tiles 45.5 | Repeat 83.10% | PG -0.0008 | Val 0.2886 | BC 0.3766 | Ent 0.322 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.741 | SPS 107 | T r/u/e/c 26.7/13.1/0.0/0.0s
Step 2,386,464 | Ep  1694 | Stage phase_resource_control | ActorTrainable True | FirstRate 17.00% | Bombs 16.18 | VB 53.37% | UB 0.00% | NEB 0.00% | Danger 74.4 | Tiles 46.2 | Repeat 83.03% | PG -0.0011 | Val 0.2339 | BC 0.4201 | Ent 0.335 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.006 | EV 0.829 | SPS 107 | T r/u/e/c 32.3/13.1/0.0/0.0s
Step 2,387,488 | Ep  1697 | Stage phase_resource_control | ActorTrainable True | FirstRate 16.00% | Bombs 16.03 | VB 52.98% | UB 0.00% | NEB 0.00% | Danger 73.2 | Tiles 45.7 | Repeat 82.97% | PG -0.0014 | Val 0.1414 | BC 0.5101 | Ent 0.404 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.885 | SPS 107 | T r/u/e/c 30.4/13.2/0.0/0.0s
Step 2,388,512 | Ep  1702 | Stage phase_resource_control | ActorTrainable True | FirstRate 16.00% | Bombs 15.56 | VB 52.28% | UB 0.00% | NEB 0.00% | Danger 71.0 | Tiles 44.9 | Repeat 82.94% | PG -0.0010 | Val 0.1694 | BC 0.5560 | Ent 0.520 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.620 | SPS 107 | T r/u/e/c 29.2/13.1/0.0/0.0s
Step 2,389,536 | Ep  1704 | Stage phase_resource_control | ActorTrainable True | FirstRate 17.00% | Bombs 14.95 | VB 51.54% | UB 0.00% | NEB 0.00% | Danger 68.7 | Tiles 44.0 | Repeat 82.76% | PG -0.0016 | Val 0.0815 | BC 0.4797 | Ent 0.429 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.918 | SPS 107 | T r/u/e/c 26.7/13.3/0.0/0.0s
Step 2,390,560 | Ep  1707 | Stage phase_resource_control | ActorTrainable True | FirstRate 17.00% | Bombs 15.18 | VB 52.27% | UB 0.00% | NEB 0.00% | Danger 69.4 | Tiles 44.8 | Repeat 82.64% | PG -0.0026 | Val 0.0493 | BC 0.6207 | Ent 0.523 | KL 0.0006 | Clip 0.02% | Ratio 1.000+/-0.017 | EV 0.832 | SPS 106 | T r/u/e/c 41.7/13.2/0.0/0.0s
Step 2,391,584 | Ep  1711 | Stage phase_resource_control | ActorTrainable True | FirstRate 20.00% | Bombs 15.32 | VB 52.75% | UB 0.00% | NEB 0.00% | Danger 68.7 | Tiles 45.9 | Repeat 82.05% | PG -0.0020 | Val 0.1105 | BC 0.4964 | Ent 0.400 | KL 0.0001 | Clip 0.07% | Ratio 1.000+/-0.021 | EV 0.925 | SPS 106 | T r/u/e/c 35.0/13.3/0.0/0.0s
Step 2,392,608 | Ep  1716 | Stage phase_resource_control | ActorTrainable True | FirstRate 20.00% | Bombs 15.43 | VB 51.79% | UB 0.00% | NEB 0.00% | Danger 68.1 | Tiles 46.0 | Repeat 82.01% | PG -0.0009 | Val 0.4507 | BC 0.3045 | Ent 0.247 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.651 | SPS 106 | T r/u/e/c 21.4/13.3/0.0/0.0s
Step 2,393,632 | Ep  1719 | Stage phase_resource_control | ActorTrainable True | FirstRate 21.00% | Bombs 15.53 | VB 51.93% | UB 0.00% | NEB 0.00% | Danger 68.4 | Tiles 46.3 | Repeat 81.63% | PG -0.0013 | Val 0.2596 | BC 0.4878 | Ent 0.394 | KL 0.0000 | Clip 0.00% | Ratio 1.000+/-0.006 | EV 0.814 | SPS 106 | T r/u/e/c 33.1/13.1/0.0/0.0s
Step 2,394,656 | Ep  1725 | Stage phase_resource_control | ActorTrainable True | FirstRate 20.00% | Bombs 15.36 | VB 50.93% | UB 0.00% | NEB 0.00% | Danger 66.5 | Tiles 45.3 | Repeat 81.67% | PG -0.0015 | Val 0.2854 | BC 0.5759 | Ent 0.441 | KL 0.0007 | Clip 0.00% | Ratio 0.999+/-0.012 | EV 0.635 | SPS 106 | T r/u/e/c 34.5/13.2/0.0/0.0s
Step 2,395,680 | Ep  1728 | Stage phase_resource_control | ActorTrainable True | FirstRate 20.00% | Bombs 15.54 | VB 51.26% | UB 0.00% | NEB 0.00% | Danger 67.2 | Tiles 45.8 | Repeat 81.40% | PG -0.0018 | Val 0.2787 | BC 0.5669 | Ent 0.439 | KL -0.0003 | Clip 0.02% | Ratio 1.000+/-0.015 | EV 0.729 | SPS 106 | T r/u/e/c 31.1/13.1/0.0/0.0s
Step 2,396,704 | Ep  1731 | Stage phase_resource_control | ActorTrainable True | FirstRate 20.00% | Bombs 15.68 | VB 51.09% | UB 0.00% | NEB 0.00% | Danger 68.0 | Tiles 46.3 | Repeat 81.53% | PG -0.0016 | Val 0.2648 | BC 0.3234 | Ent 0.293 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.774 | SPS 105 | T r/u/e/c 24.3/13.5/0.0/0.0s
Step 2,397,728 | Ep  1735 | Stage phase_resource_control | ActorTrainable True | FirstRate 20.00% | Bombs 15.96 | VB 52.43% | UB 0.00% | NEB 0.00% | Danger 69.0 | Tiles 47.3 | Repeat 81.41% | PG -0.0013 | Val 0.2338 | BC 0.4432 | Ent 0.367 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.787 | SPS 105 | T r/u/e/c 41.6/13.5/0.0/0.0s
Step 2,398,752 | Ep  1739 | Stage phase_resource_control | ActorTrainable True | FirstRate 21.00% | Bombs 16.18 | VB 53.83% | UB 0.00% | NEB 0.00% | Danger 69.4 | Tiles 48.1 | Repeat 81.08% | PG -0.0007 | Val 0.1333 | BC 0.5143 | Ent 0.484 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.814 | SPS 105 | T r/u/e/c 44.1/13.1/0.0/0.0s
Step 2,399,776 | Ep  1742 | Stage phase_resource_control | ActorTrainable True | FirstRate 21.00% | Bombs 16.96 | VB 55.68% | UB 0.00% | NEB 0.00% | Danger 71.6 | Tiles 49.5 | Repeat 80.85% | PG -0.0024 | Val 0.3212 | BC 0.4120 | Ent 0.327 | KL -0.0009 | Clip 0.10% | Ratio 1.001+/-0.015 | EV 0.772 | SPS 105 | T r/u/e/c 42.4/13.1/0.0/0.0s
Step 2,400,800 | Ep  1745 | Stage phase_resource_control | ActorTrainable True | FirstRate 21.00% | Bombs 17.05 | VB 54.84% | UB 0.00% | NEB 0.00% | Danger 72.5 | Tiles 49.3 | Repeat 80.95% | PG -0.0018 | Val 0.1042 | BC 0.2815 | Ent 0.246 | KL 0.0005 | Clip 0.02% | Ratio 1.000+/-0.006 | EV 0.948 | SPS 105 | T r/u/e/c 28.5/13.1/0.0/0.0s
  -> Saved checkpoint: D:\other\CNTT\Bomberland-GDGoC-AI-Challenge\agent\agent\v6\checkpoints\model_step2400800.pth
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
  -> Eval score 2.100 | AvgRank 0.90 | R0/1/2/3 40%/40%/10%/10% | UF 30%
     EM score/rank 2.000/1.00 | R0/1/2/3 40%/40%/0%/20% | VB/Repeat 59.39%/79.00%
     Hard score/rank 2.200/0.80 | R0/1/2/3 40%/40%/20%/0% | VB/Repeat 72.12%/79.01%
     Diff K/B/I/Bp -0.30/+0.70/-2.80/+5.90 | Cap/Radius -1.20/-1.00 | Timeout loss K/B/I/B 10.00%/10.00%/0.00%/0.00%
     TIMING rollout 28.5s | update 13.1s | checkpoint 0.1s | eval 61.1s
Step 2,401,824 | Ep  1747 | Stage phase_resource_control | ActorTrainable True | FirstRate 21.00% | Bombs 16.70 | VB 54.51% | UB 0.00% | NEB 0.00% | Danger 71.3 | Tiles 49.1 | Repeat 81.00% | PG -0.0020 | Val 0.2396 | BC 0.6685 | Ent 0.491 | KL 0.0005 | Clip 0.00% | Ratio 1.000+/-0.014 | EV 0.662 | SPS 104 | T r/u/e/c 41.8/13.8/0.0/0.0s
Step 2,402,848 | Ep  1751 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 16.75 | VB 54.75% | UB 0.00% | NEB 0.00% | Danger 71.8 | Tiles 49.5 | Repeat 80.93% | PG -0.0020 | Val 0.1415 | BC 0.5743 | Ent 0.480 | KL 0.0003 | Clip 0.10% | Ratio 1.000+/-0.020 | EV 0.879 | SPS 104 | T r/u/e/c 36.8/13.3/0.0/0.0s
Step 2,403,872 | Ep  1756 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 16.97 | VB 55.32% | UB 0.00% | NEB 0.00% | Danger 73.4 | Tiles 50.6 | Repeat 80.69% | PG -0.0023 | Val 0.1773 | BC 0.6209 | Ent 0.571 | KL 0.0005 | Clip 0.00% | Ratio 1.000+/-0.017 | EV 0.415 | SPS 104 | T r/u/e/c 35.7/13.6/0.0/0.0s
Step 2,404,896 | Ep  1759 | Stage phase_resource_control | ActorTrainable True | FirstRate 26.00% | Bombs 17.06 | VB 55.30% | UB 0.00% | NEB 0.00% | Danger 73.4 | Tiles 50.4 | Repeat 80.72% | PG -0.0010 | Val 0.1670 | BC 0.4068 | Ent 0.352 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.879 | SPS 104 | T r/u/e/c 23.5/13.6/0.0/0.0s
Step 2,405,920 | Ep  1766 | Stage phase_resource_control | ActorTrainable True | FirstRate 23.00% | Bombs 16.61 | VB 55.54% | UB 0.00% | NEB 0.00% | Danger 72.8 | Tiles 49.7 | Repeat 80.39% | PG -0.0015 | Val 0.2685 | BC 0.5571 | Ent 0.403 | KL -0.0003 | Clip 0.00% | Ratio 1.000+/-0.006 | EV 0.771 | SPS 103 | T r/u/e/c 28.1/13.5/0.0/0.0s
Step 2,406,944 | Ep  1771 | Stage phase_resource_control | ActorTrainable True | FirstRate 23.00% | Bombs 16.45 | VB 55.47% | UB 0.00% | NEB 0.00% | Danger 72.4 | Tiles 49.4 | Repeat 80.03% | PG -0.0016 | Val 0.3919 | BC 0.5887 | Ent 0.425 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.459 | SPS 103 | T r/u/e/c 35.2/13.6/0.0/0.0s
Step 2,407,968 | Ep  1776 | Stage phase_resource_control | ActorTrainable True | FirstRate 23.00% | Bombs 16.34 | VB 54.88% | UB 0.00% | NEB 0.00% | Danger 71.5 | Tiles 48.1 | Repeat 80.20% | PG -0.0013 | Val 0.2463 | BC 0.5941 | Ent 0.472 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.633 | SPS 103 | T r/u/e/c 39.6/13.7/0.0/0.0s
Step 2,408,992 | Ep  1780 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 16.74 | VB 55.11% | UB 0.00% | NEB 0.00% | Danger 72.7 | Tiles 48.6 | Repeat 80.08% | PG -0.0016 | Val 0.2159 | BC 0.6193 | Ent 0.457 | KL -0.0000 | Clip 0.02% | Ratio 1.000+/-0.015 | EV 0.684 | SPS 103 | T r/u/e/c 37.4/13.3/0.0/0.0s
Step 2,410,016 | Ep  1783 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 16.55 | VB 55.10% | UB 0.00% | NEB 0.00% | Danger 72.2 | Tiles 48.6 | Repeat 79.97% | PG -0.0014 | Val 0.2378 | BC 0.5320 | Ent 0.463 | KL -0.0006 | Clip 0.00% | Ratio 1.001+/-0.013 | EV 0.799 | SPS 103 | T r/u/e/c 34.9/13.2/0.0/0.0s
Step 2,411,040 | Ep  1788 | Stage phase_resource_control | ActorTrainable True | FirstRate 23.00% | Bombs 16.30 | VB 54.86% | UB 0.00% | NEB 0.00% | Danger 71.1 | Tiles 48.2 | Repeat 79.91% | PG -0.0022 | Val 0.1773 | BC 0.5318 | Ent 0.378 | KL 0.0009 | Clip 0.02% | Ratio 0.999+/-0.016 | EV 0.822 | SPS 103 | T r/u/e/c 34.2/13.2/0.0/0.0s
Step 2,412,064 | Ep  1793 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 16.38 | VB 55.64% | UB 0.00% | NEB 0.00% | Danger 71.2 | Tiles 47.3 | Repeat 80.09% | PG -0.0006 | Val 0.2561 | BC 0.5469 | Ent 0.413 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.820 | SPS 102 | T r/u/e/c 31.0/13.2/0.0/0.0s
Step 2,413,088 | Ep  1798 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 16.29 | VB 55.40% | UB 0.00% | NEB 0.00% | Danger 70.0 | Tiles 46.9 | Repeat 79.98% | PG -0.0019 | Val 0.2165 | BC 0.7062 | Ent 0.559 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.579 | SPS 102 | T r/u/e/c 41.2/13.1/0.0/0.0s
Step 2,414,112 | Ep  1801 | Stage phase_resource_control | ActorTrainable True | FirstRate 25.00% | Bombs 16.07 | VB 54.88% | UB 0.00% | NEB 0.00% | Danger 69.2 | Tiles 45.8 | Repeat 80.37% | PG -0.0020 | Val 0.2050 | BC 0.4515 | Ent 0.335 | KL 0.0011 | Clip 0.02% | Ratio 0.999+/-0.015 | EV 0.832 | SPS 102 | T r/u/e/c 34.1/13.1/0.0/0.0s
Step 2,415,136 | Ep  1804 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 16.31 | VB 54.97% | UB 0.00% | NEB 0.00% | Danger 69.7 | Tiles 46.1 | Repeat 80.55% | PG -0.0010 | Val 0.2352 | BC 0.3174 | Ent 0.238 | KL -0.0003 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.819 | SPS 102 | T r/u/e/c 32.7/13.1/0.0/0.0s
Step 2,416,160 | Ep  1809 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 15.97 | VB 54.52% | UB 0.00% | NEB 0.00% | Danger 68.4 | Tiles 44.8 | Repeat 80.81% | PG -0.0013 | Val 0.1333 | BC 0.5492 | Ent 0.438 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.849 | SPS 102 | T r/u/e/c 42.8/13.2/0.0/0.0s
Step 2,417,184 | Ep  1813 | Stage phase_resource_control | ActorTrainable True | FirstRate 22.00% | Bombs 16.00 | VB 55.10% | UB 0.00% | NEB 0.00% | Danger 68.5 | Tiles 44.7 | Repeat 80.82% | PG -0.0014 | Val 0.2446 | BC 0.6325 | Ent 0.463 | KL 0.0000 | Clip 0.00% | Ratio 1.000+/-0.013 | EV 0.513 | SPS 102 | T r/u/e/c 35.9/13.1/0.0/0.0s
Step 2,418,208 | Ep  1817 | Stage phase_resource_control | ActorTrainable True | FirstRate 22.00% | Bombs 16.63 | VB 56.13% | UB 0.00% | NEB 0.00% | Danger 70.5 | Tiles 45.6 | Repeat 80.62% | PG -0.0015 | Val 0.0860 | BC 0.5412 | Ent 0.502 | KL -0.0004 | Clip 0.00% | Ratio 1.001+/-0.014 | EV -0.385 | SPS 101 | T r/u/e/c 38.0/13.1/0.0/0.0s
Step 2,419,232 | Ep  1819 | Stage phase_resource_control | ActorTrainable True | FirstRate 21.00% | Bombs 16.57 | VB 56.35% | UB 0.00% | NEB 0.00% | Danger 70.0 | Tiles 45.4 | Repeat 80.60% | PG -0.0020 | Val 0.0799 | BC 0.6610 | Ent 0.499 | KL 0.0007 | Clip 0.00% | Ratio 0.999+/-0.011 | EV 0.293 | SPS 101 | T r/u/e/c 42.6/13.1/0.0/0.0s
Step 2,420,256 | Ep  1824 | Stage phase_resource_control | ActorTrainable True | FirstRate 21.00% | Bombs 17.45 | VB 57.39% | UB 0.00% | NEB 0.00% | Danger 73.0 | Tiles 46.1 | Repeat 80.72% | PG -0.0017 | Val 0.3399 | BC 0.5704 | Ent 0.446 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.724 | SPS 101 | T r/u/e/c 40.6/13.0/0.0/0.0s
Step 2,421,280 | Ep  1829 | Stage phase_resource_control | ActorTrainable True | FirstRate 22.00% | Bombs 17.93 | VB 57.86% | UB 0.00% | NEB 0.00% | Danger 74.0 | Tiles 46.0 | Repeat 81.10% | PG -0.0011 | Val 0.2180 | BC 0.4902 | Ent 0.364 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.870 | SPS 101 | T r/u/e/c 31.5/13.1/0.0/0.0s
Step 2,422,304 | Ep  1834 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 18.13 | VB 57.59% | UB 0.00% | NEB 0.00% | Danger 74.0 | Tiles 45.8 | Repeat 80.82% | PG -0.0021 | Val 0.1806 | BC 0.6517 | Ent 0.544 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.014 | EV 0.347 | SPS 101 | T r/u/e/c 38.9/13.1/0.0/0.0s
Step 2,423,328 | Ep  1837 | Stage phase_resource_control | ActorTrainable True | FirstRate 25.00% | Bombs 17.98 | VB 57.49% | UB 0.00% | NEB 0.00% | Danger 73.3 | Tiles 45.4 | Repeat 80.65% | PG -0.0017 | Val 0.1574 | BC 0.7270 | Ent 0.521 | KL 0.0005 | Clip 0.00% | Ratio 1.000+/-0.017 | EV 0.547 | SPS 101 | T r/u/e/c 36.2/13.2/0.0/0.0s
Step 2,424,352 | Ep  1841 | Stage phase_resource_control | ActorTrainable True | FirstRate 26.00% | Bombs 18.05 | VB 57.41% | UB 0.00% | NEB 0.00% | Danger 73.8 | Tiles 45.3 | Repeat 80.82% | PG -0.0008 | Val 0.1688 | BC 0.5817 | Ent 0.475 | KL 0.0002 | Clip 0.05% | Ratio 1.000+/-0.014 | EV 0.803 | SPS 100 | T r/u/e/c 28.9/13.1/0.0/0.0s
Step 2,425,376 | Ep  1844 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 17.65 | VB 57.24% | UB 0.00% | NEB 0.00% | Danger 72.3 | Tiles 45.0 | Repeat 80.65% | PG -0.0011 | Val 0.1722 | BC 0.3344 | Ent 0.303 | KL -0.0006 | Clip 0.00% | Ratio 1.001+/-0.011 | EV 0.890 | SPS 100 | T r/u/e/c 21.3/13.1/0.0/0.0s
Step 2,426,400 | Ep  1847 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 17.92 | VB 57.84% | UB 0.00% | NEB 0.00% | Danger 73.7 | Tiles 45.6 | Repeat 80.54% | PG -0.0016 | Val 0.2038 | BC 0.5797 | Ent 0.433 | KL 0.0000 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.853 | SPS 100 | T r/u/e/c 33.4/13.3/0.0/0.0s
  -> Saved checkpoint: D:\other\CNTT\Bomberland-GDGoC-AI-Challenge\agent\agent\v6\checkpoints\model_step2426400.pth
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
  -> Eval score 2.300 | AvgRank 0.70 | R0/1/2/3 60%/20%/10%/10% | UF 60%
     EM score/rank 2.000/1.00 | R0/1/2/3 40%/40%/0%/20% | VB/Repeat 74.67%/80.01%
     Hard score/rank 2.600/0.40 | R0/1/2/3 80%/0%/20%/0% | VB/Repeat 66.54%/80.29%
     Diff K/B/I/Bp -0.20/+0.00/-2.90/+8.00 | Cap/Radius -0.60/-1.30 | Timeout loss K/B/I/B 0.00%/0.00%/0.00%/0.00%
     TIMING rollout 33.4s | update 13.3s | checkpoint 0.1s | eval 63.5s
Step 2,427,424 | Ep  1851 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 17.71 | VB 57.50% | UB 0.00% | NEB 0.00% | Danger 73.3 | Tiles 44.6 | Repeat 80.91% | PG -0.0015 | Val 0.1527 | BC 0.5040 | Ent 0.403 | KL 0.0011 | Clip 0.00% | Ratio 0.999+/-0.016 | EV 0.912 | SPS 100 | T r/u/e/c 37.4/14.1/0.0/0.0s
Step 2,428,448 | Ep  1854 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 17.52 | VB 57.62% | UB 0.00% | NEB 0.00% | Danger 71.7 | Tiles 43.3 | Repeat 81.11% | PG -0.0006 | Val 0.1859 | BC 0.6273 | Ent 0.559 | KL 0.0007 | Clip 0.00% | Ratio 0.999+/-0.012 | EV 0.566 | SPS 100 | T r/u/e/c 44.8/13.8/0.0/0.0s
Step 2,429,472 | Ep  1858 | Stage phase_resource_control | ActorTrainable True | FirstRate 23.00% | Bombs 17.40 | VB 58.05% | UB 0.00% | NEB 0.00% | Danger 72.0 | Tiles 43.7 | Repeat 80.96% | PG -0.0018 | Val 0.2522 | BC 0.3184 | Ent 0.296 | KL 0.0004 | Clip 0.05% | Ratio 1.000+/-0.011 | EV 0.849 | SPS 99 | T r/u/e/c 32.0/13.7/0.0/0.0s
Step 2,430,496 | Ep  1862 | Stage phase_resource_control | ActorTrainable True | FirstRate 24.00% | Bombs 17.85 | VB 58.72% | UB 0.00% | NEB 0.00% | Danger 74.3 | Tiles 44.4 | Repeat 81.28% | PG -0.0011 | Val 0.2414 | BC 0.4741 | Ent 0.478 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.781 | SPS 99 | T r/u/e/c 31.2/13.7/0.0/0.0s
Step 2,431,520 | Ep  1866 | Stage phase_resource_control | ActorTrainable True | FirstRate 25.00% | Bombs 17.71 | VB 57.53% | UB 0.00% | NEB 0.00% | Danger 72.7 | Tiles 44.2 | Repeat 81.38% | PG -0.0011 | Val 0.1714 | BC 0.6176 | Ent 0.515 | KL 0.0006 | Clip 0.00% | Ratio 0.999+/-0.011 | EV 0.677 | SPS 99 | T r/u/e/c 33.0/13.6/0.0/0.0s
Step 2,432,544 | Ep  1870 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 17.74 | VB 57.87% | UB 0.00% | NEB 0.00% | Danger 73.7 | Tiles 44.8 | Repeat 81.67% | PG -0.0018 | Val 0.2906 | BC 0.4243 | Ent 0.319 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.799 | SPS 99 | T r/u/e/c 27.3/13.8/0.0/0.0s
Step 2,433,568 | Ep  1873 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 17.43 | VB 57.23% | UB 0.00% | NEB 0.00% | Danger 72.2 | Tiles 44.5 | Repeat 81.91% | PG -0.0015 | Val 0.1068 | BC 0.4873 | Ent 0.420 | KL 0.0000 | Clip 0.02% | Ratio 1.000+/-0.012 | EV 0.921 | SPS 99 | T r/u/e/c 24.4/13.6/0.0/0.0s
Step 2,434,592 | Ep  1878 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 16.88 | VB 56.73% | UB 0.00% | NEB 0.00% | Danger 71.1 | Tiles 44.6 | Repeat 81.84% | PG -0.0017 | Val 0.1567 | BC 0.6189 | Ent 0.567 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.013 | EV 0.159 | SPS 99 | T r/u/e/c 36.3/13.6/0.0/0.0s
Step 2,435,616 | Ep  1882 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 17.25 | VB 56.92% | UB 0.00% | NEB 0.00% | Danger 72.5 | Tiles 45.1 | Repeat 81.88% | PG -0.0022 | Val 0.1057 | BC 0.6585 | Ent 0.573 | KL -0.0001 | Clip 0.10% | Ratio 1.000+/-0.019 | EV 0.595 | SPS 99 | T r/u/e/c 38.5/13.5/0.0/0.0s
Step 2,436,640 | Ep  1883 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 17.06 | VB 56.81% | UB 0.00% | NEB 0.00% | Danger 71.6 | Tiles 44.9 | Repeat 81.87% | PG -0.0021 | Val 0.2410 | BC 0.5193 | Ent 0.449 | KL 0.0003 | Clip 0.05% | Ratio 1.000+/-0.017 | EV 0.773 | SPS 98 | T r/u/e/c 31.7/13.3/0.0/0.0s
Step 2,437,664 | Ep  1887 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 17.48 | VB 57.41% | UB 0.00% | NEB 0.00% | Danger 73.6 | Tiles 45.6 | Repeat 82.04% | PG -0.0014 | Val 0.3012 | BC 0.5696 | Ent 0.487 | KL -0.0004 | Clip 0.00% | Ratio 1.001+/-0.015 | EV 0.671 | SPS 98 | T r/u/e/c 41.3/13.7/0.0/0.0s
Step 2,438,688 | Ep  1892 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 17.19 | VB 57.06% | UB 0.00% | NEB 0.00% | Danger 73.9 | Tiles 45.9 | Repeat 81.99% | PG -0.0013 | Val 0.1980 | BC 0.5686 | Ent 0.434 | KL 0.0001 | Clip 0.02% | Ratio 1.000+/-0.016 | EV 0.848 | SPS 98 | T r/u/e/c 32.4/13.4/0.0/0.0s
Step 2,439,712 | Ep  1895 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 17.33 | VB 57.43% | UB 0.00% | NEB 0.00% | Danger 74.0 | Tiles 46.7 | Repeat 81.71% | PG -0.0017 | Val 0.2288 | BC 0.6161 | Ent 0.537 | KL 0.0016 | Clip 0.22% | Ratio 0.999+/-0.027 | EV 0.711 | SPS 98 | T r/u/e/c 38.5/13.6/0.0/0.0s
Step 2,440,736 | Ep  1898 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 17.45 | VB 57.31% | UB 0.00% | NEB 0.00% | Danger 74.5 | Tiles 46.9 | Repeat 81.87% | PG -0.0013 | Val 0.2172 | BC 0.5117 | Ent 0.425 | KL 0.0008 | Clip 0.20% | Ratio 1.000+/-0.022 | EV 0.857 | SPS 98 | T r/u/e/c 31.6/13.7/0.0/0.0s
Step 2,441,760 | Ep  1901 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 17.96 | VB 58.26% | UB 0.00% | NEB 0.00% | Danger 76.1 | Tiles 48.3 | Repeat 81.76% | PG -0.0015 | Val 0.2393 | BC 0.4516 | Ent 0.368 | KL -0.0003 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.857 | SPS 98 | T r/u/e/c 25.8/13.5/0.0/0.0s
Step 2,442,784 | Ep  1904 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 17.63 | VB 57.69% | UB 0.00% | NEB 0.00% | Danger 75.6 | Tiles 48.2 | Repeat 81.66% | PG -0.0013 | Val 0.0664 | BC 0.5834 | Ent 0.412 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.939 | SPS 98 | T r/u/e/c 33.0/13.5/0.0/0.0s
Step 2,443,808 | Ep  1908 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 18.25 | VB 58.39% | UB 0.00% | NEB 0.00% | Danger 77.1 | Tiles 48.8 | Repeat 81.42% | PG -0.0023 | Val 0.2221 | BC 0.5673 | Ent 0.471 | KL 0.0001 | Clip 0.05% | Ratio 1.000+/-0.015 | EV 0.822 | SPS 97 | T r/u/e/c 27.1/13.6/0.0/0.0s
Step 2,444,832 | Ep  1912 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 18.09 | VB 57.99% | UB 0.00% | NEB 0.00% | Danger 76.6 | Tiles 48.1 | Repeat 81.72% | PG -0.0014 | Val 0.2325 | BC 0.5554 | Ent 0.481 | KL 0.0001 | Clip 0.02% | Ratio 1.000+/-0.017 | EV 0.657 | SPS 97 | T r/u/e/c 36.2/13.6/0.0/0.0s
Step 2,445,856 | Ep  1916 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 18.11 | VB 58.25% | UB 0.00% | NEB 0.00% | Danger 77.3 | Tiles 48.0 | Repeat 81.95% | PG -0.0016 | Val 0.1566 | BC 0.5804 | Ent 0.506 | KL 0.0010 | Clip 0.00% | Ratio 0.999+/-0.014 | EV 0.835 | SPS 97 | T r/u/e/c 36.6/13.6/0.0/0.0s
Step 2,446,880 | Ep  1921 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 17.17 | VB 56.76% | UB 0.00% | NEB 0.00% | Danger 74.7 | Tiles 47.0 | Repeat 82.33% | PG -0.0013 | Val 0.3111 | BC 0.4050 | Ent 0.306 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.801 | SPS 97 | T r/u/e/c 36.8/13.6/0.0/0.0s
Step 2,447,904 | Ep  1923 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 16.69 | VB 56.27% | UB 0.00% | NEB 0.00% | Danger 73.4 | Tiles 46.8 | Repeat 82.24% | PG -0.0017 | Val 0.0780 | BC 0.5853 | Ent 0.461 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.015 | EV 0.928 | SPS 97 | T r/u/e/c 34.4/13.7/0.0/0.0s
Step 2,448,928 | Ep  1929 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 16.68 | VB 56.82% | UB 0.00% | NEB 0.00% | Danger 74.2 | Tiles 47.7 | Repeat 81.98% | PG -0.0016 | Val 0.2005 | BC 0.6728 | Ent 0.502 | KL 0.0006 | Clip 0.02% | Ratio 1.000+/-0.015 | EV 0.744 | SPS 97 | T r/u/e/c 28.8/13.7/0.0/0.0s
Step 2,449,952 | Ep  1930 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 16.73 | VB 56.87% | UB 0.00% | NEB 0.00% | Danger 74.4 | Tiles 47.7 | Repeat 81.99% | PG -0.0013 | Val 0.2885 | BC 0.7373 | Ent 0.502 | KL -0.0004 | Clip 0.00% | Ratio 1.001+/-0.014 | EV 0.677 | SPS 97 | T r/u/e/c 40.9/13.7/0.0/0.0s
Step 2,450,976 | Ep  1935 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 17.18 | VB 57.62% | UB 0.00% | NEB 0.00% | Danger 75.6 | Tiles 48.5 | Repeat 82.16% | PG -0.0015 | Val 0.1701 | BC 0.5389 | Ent 0.415 | KL -0.0001 | Clip 0.02% | Ratio 1.000+/-0.015 | EV 0.876 | SPS 96 | T r/u/e/c 37.9/13.6/0.0/0.0s
Step 2,452,000 | Ep  1938 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 17.00 | VB 57.33% | UB 0.00% | NEB 0.00% | Danger 74.7 | Tiles 48.2 | Repeat 82.39% | PG -0.0016 | Val 0.2965 | BC 0.5971 | Ent 0.375 | KL 0.0007 | Clip 0.00% | Ratio 0.999+/-0.014 | EV 0.721 | SPS 96 | T r/u/e/c 40.6/13.6/0.0/0.0s
  -> Saved checkpoint: D:\other\CNTT\Bomberland-GDGoC-AI-Challenge\agent\agent\v6\checkpoints\model_step2452000.pth
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
  -> Eval score 2.000 | AvgRank 1.00 | R0/1/2/3 50%/10%/30%/10% | UF 50%
     EM score/rank 2.200/0.80 | R0/1/2/3 60%/0%/40%/0% | VB/Repeat 62.73%/81.20%
     Hard score/rank 1.800/1.20 | R0/1/2/3 40%/20%/20%/20% | VB/Repeat 53.68%/84.44%
     Diff K/B/I/Bp -0.40/-0.40/-5.00/-5.40 | Cap/Radius -1.30/-1.80 | Timeout loss K/B/I/B 20.00%/0.00%/0.00%/0.00%
     TIMING rollout 40.6s | update 13.6s | checkpoint 0.1s | eval 40.7s
Step 2,453,024 | Ep  1943 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 16.84 | VB 57.86% | UB 0.00% | NEB 0.00% | Danger 75.2 | Tiles 48.4 | Repeat 82.33% | PG -0.0008 | Val 0.1195 | BC 0.4850 | Ent 0.398 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.015 | EV 0.914 | SPS 96 | T r/u/e/c 40.8/13.1/0.0/0.0s
Step 2,454,048 | Ep  1948 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 16.62 | VB 56.79% | UB 0.00% | NEB 0.00% | Danger 72.9 | Tiles 47.6 | Repeat 82.00% | PG -0.0010 | Val 0.4535 | BC 0.5438 | Ent 0.415 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.620 | SPS 96 | T r/u/e/c 46.3/13.0/0.0/0.0s
Step 2,455,072 | Ep  1953 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 16.61 | VB 55.61% | UB 0.00% | NEB 0.00% | Danger 72.9 | Tiles 47.8 | Repeat 82.05% | PG -0.0016 | Val 0.2561 | BC 0.3560 | Ent 0.293 | KL 0.0002 | Clip 0.12% | Ratio 1.000+/-0.014 | EV 0.851 | SPS 96 | T r/u/e/c 29.3/13.1/0.0/0.0s
Step 2,456,096 | Ep  1958 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 16.60 | VB 55.11% | UB 0.00% | NEB 0.00% | Danger 72.0 | Tiles 48.0 | Repeat 81.45% | PG -0.0016 | Val 0.3247 | BC 0.6393 | Ent 0.428 | KL 0.0008 | Clip 0.37% | Ratio 1.000+/-0.022 | EV 0.634 | SPS 95 | T r/u/e/c 41.0/13.0/0.0/0.0s
Step 2,457,120 | Ep  1961 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 16.38 | VB 54.18% | UB 0.00% | NEB 0.00% | Danger 70.2 | Tiles 47.6 | Repeat 81.32% | PG -0.0005 | Val 0.1443 | BC 0.5506 | Ent 0.470 | KL -0.0014 | Clip 0.00% | Ratio 1.002+/-0.018 | EV 0.771 | SPS 95 | T r/u/e/c 38.7/13.2/0.0/0.0s
Step 2,458,144 | Ep  1965 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 16.65 | VB 55.94% | UB 0.00% | NEB 0.00% | Danger 71.7 | Tiles 48.6 | Repeat 81.26% | PG -0.0014 | Val 0.3094 | BC 0.6033 | Ent 0.465 | KL 0.0005 | Clip 0.15% | Ratio 1.000+/-0.017 | EV 0.725 | SPS 95 | T r/u/e/c 35.3/13.1/0.0/0.0s
Step 2,459,168 | Ep  1969 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 16.57 | VB 55.20% | UB 0.00% | NEB 0.00% | Danger 71.1 | Tiles 47.9 | Repeat 81.20% | PG -0.0010 | Val 0.2697 | BC 0.3404 | Ent 0.240 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.005 | EV 0.737 | SPS 95 | T r/u/e/c 24.0/13.1/0.0/0.0s
Step 2,460,192 | Ep  1974 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 16.53 | VB 55.18% | UB 0.00% | NEB 0.00% | Danger 71.2 | Tiles 47.9 | Repeat 81.08% | PG -0.0018 | Val 0.1951 | BC 0.4628 | Ent 0.421 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.816 | SPS 95 | T r/u/e/c 28.6/13.2/0.0/0.0s
Step 2,461,216 | Ep  1978 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 16.47 | VB 54.92% | UB 0.00% | NEB 0.00% | Danger 71.5 | Tiles 47.4 | Repeat 81.54% | PG -0.0012 | Val 0.2531 | BC 0.2824 | Ent 0.248 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.005 | EV 0.799 | SPS 95 | T r/u/e/c 25.2/13.1/0.0/0.0s
Step 2,462,240 | Ep  1981 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 16.37 | VB 55.17% | UB 0.00% | NEB 0.00% | Danger 71.6 | Tiles 47.1 | Repeat 81.73% | PG -0.0012 | Val 0.1274 | BC 0.4703 | Ent 0.426 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.884 | SPS 95 | T r/u/e/c 30.9/13.0/0.0/0.0s
Step 2,463,264 | Ep  1986 | Stage phase_resource_control | ActorTrainable True | FirstRate 26.00% | Bombs 15.57 | VB 53.46% | UB 0.00% | NEB 0.00% | Danger 68.3 | Tiles 45.5 | Repeat 81.49% | PG -0.0014 | Val 0.4371 | BC 0.5341 | Ent 0.446 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.006 | EV 0.481 | SPS 95 | T r/u/e/c 38.5/13.2/0.0/0.0s
Step 2,464,288 | Ep  1990 | Stage phase_resource_control | ActorTrainable True | FirstRate 26.00% | Bombs 15.48 | VB 54.53% | UB 0.00% | NEB 0.00% | Danger 68.5 | Tiles 45.9 | Repeat 81.40% | PG -0.0012 | Val 0.2096 | BC 0.4848 | Ent 0.400 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.006 | EV 0.824 | SPS 94 | T r/u/e/c 26.0/13.2/0.0/0.0s
Step 2,465,312 | Ep  1992 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 15.93 | VB 55.32% | UB 0.00% | NEB 0.00% | Danger 69.2 | Tiles 46.7 | Repeat 81.25% | PG -0.0016 | Val 0.1374 | BC 0.5306 | Ent 0.444 | KL -0.0005 | Clip 0.00% | Ratio 1.001+/-0.009 | EV 0.903 | SPS 94 | T r/u/e/c 26.4/13.3/0.0/0.0s
Step 2,466,336 | Ep  1996 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 16.22 | VB 55.36% | UB 0.00% | NEB 0.00% | Danger 70.9 | Tiles 46.6 | Repeat 81.46% | PG -0.0016 | Val 0.1528 | BC 0.4833 | Ent 0.397 | KL 0.0006 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.873 | SPS 94 | T r/u/e/c 35.6/13.2/0.0/0.0s
Step 2,467,360 | Ep  2000 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 15.76 | VB 54.52% | UB 0.00% | NEB 0.00% | Danger 68.8 | Tiles 45.8 | Repeat 81.33% | PG -0.0013 | Val 0.1962 | BC 0.4967 | Ent 0.361 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.871 | SPS 94 | T r/u/e/c 36.1/13.2/0.0/0.0s
Step 2,468,384 | Ep  2002 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 15.71 | VB 54.45% | UB 0.00% | NEB 0.00% | Danger 68.9 | Tiles 45.3 | Repeat 81.56% | PG -0.0014 | Val 0.1543 | BC 0.5059 | Ent 0.432 | KL 0.0000 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.819 | SPS 94 | T r/u/e/c 30.6/13.2/0.0/0.0s
Step 2,469,408 | Ep  2006 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 16.01 | VB 54.55% | UB 0.00% | NEB 0.00% | Danger 70.3 | Tiles 45.9 | Repeat 81.48% | PG -0.0017 | Val 0.0931 | BC 0.5043 | Ent 0.402 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.926 | SPS 94 | T r/u/e/c 25.1/13.1/0.0/0.0s
Step 2,470,432 | Ep  2011 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 15.60 | VB 53.51% | UB 0.00% | NEB 0.00% | Danger 69.1 | Tiles 45.6 | Repeat 81.47% | PG -0.0010 | Val 0.3129 | BC 0.4591 | Ent 0.371 | KL 0.0005 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.680 | SPS 94 | T r/u/e/c 23.8/13.1/0.0/0.0s
Step 2,471,456 | Ep  2012 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 15.55 | VB 53.23% | UB 0.00% | NEB 0.00% | Danger 68.9 | Tiles 45.4 | Repeat 81.66% | PG -0.0013 | Val 0.1976 | BC 0.4644 | Ent 0.372 | KL -0.0008 | Clip 0.00% | Ratio 1.001+/-0.011 | EV 0.893 | SPS 94 | T r/u/e/c 28.5/13.1/0.0/0.0s
Step 2,472,480 | Ep  2017 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 15.70 | VB 52.91% | UB 0.00% | NEB 0.00% | Danger 68.9 | Tiles 45.4 | Repeat 81.50% | PG -0.0018 | Val 0.2031 | BC 0.4457 | Ent 0.316 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.014 | EV 0.878 | SPS 94 | T r/u/e/c 27.6/13.1/0.0/0.0s
Step 2,473,504 | Ep  2022 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 15.96 | VB 53.09% | UB 0.00% | NEB 0.00% | Danger 68.4 | Tiles 45.6 | Repeat 81.07% | PG -0.0017 | Val 0.2152 | BC 0.4476 | Ent 0.337 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.857 | SPS 93 | T r/u/e/c 24.8/13.1/0.0/0.0s
Step 2,474,528 | Ep  2024 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 15.51 | VB 52.35% | UB 0.00% | NEB 0.00% | Danger 67.0 | Tiles 44.8 | Repeat 81.24% | PG -0.0017 | Val 0.2764 | BC 0.2239 | Ent 0.197 | KL -0.0003 | Clip 0.02% | Ratio 1.000+/-0.010 | EV 0.814 | SPS 93 | T r/u/e/c 22.0/13.1/0.0/0.0s
Step 2,475,552 | Ep  2028 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 15.37 | VB 51.42% | UB 0.00% | NEB 0.00% | Danger 67.1 | Tiles 44.6 | Repeat 81.47% | PG -0.0016 | Val 0.1487 | BC 0.4732 | Ent 0.342 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.909 | SPS 93 | T r/u/e/c 32.7/13.1/0.0/0.0s
Step 2,476,576 | Ep  2029 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 15.46 | VB 51.68% | UB 0.00% | NEB 0.00% | Danger 67.4 | Tiles 44.5 | Repeat 81.61% | PG -0.0019 | Val 0.2018 | BC 0.3369 | Ent 0.229 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.868 | SPS 93 | T r/u/e/c 31.4/13.1/0.0/0.0s
Step 2,477,600 | Ep  2034 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 14.95 | VB 50.77% | UB 0.00% | NEB 0.00% | Danger 66.4 | Tiles 43.7 | Repeat 81.76% | PG -0.0011 | Val 0.2095 | BC 0.4063 | Ent 0.291 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.848 | SPS 93 | T r/u/e/c 30.5/13.1/0.0/0.0s
  -> Saved checkpoint: D:\other\CNTT\Bomberland-GDGoC-AI-Challenge\agent\agent\v6\checkpoints\model_step2477600.pth
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
  -> Eval score 1.800 | AvgRank 1.20 | R0/1/2/3 40%/10%/40%/10% | UF 40%
     EM score/rank 1.600/1.40 | R0/1/2/3 20%/20%/60%/0% | VB/Repeat 63.87%/79.01%
     Hard score/rank 2.000/1.00 | R0/1/2/3 60%/0%/20%/20% | VB/Repeat 67.65%/85.31%
     Diff K/B/I/Bp -0.10/-0.10/-5.30/-0.90 | Cap/Radius -1.10/-1.90 | Timeout loss K/B/I/B 10.00%/10.00%/0.00%/0.00%
     TIMING rollout 30.5s | update 13.1s | checkpoint 0.1s | eval 60.0s
Step 2,478,624 | Ep  2037 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 15.29 | VB 51.23% | UB 0.00% | NEB 0.00% | Danger 67.5 | Tiles 44.1 | Repeat 81.68% | PG -0.0013 | Val 0.1860 | BC 0.5209 | Ent 0.384 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.866 | SPS 93 | T r/u/e/c 32.8/13.8/0.0/0.0s
Step 2,479,648 | Ep  2042 | Stage phase_resource_control | ActorTrainable True | FirstRate 25.00% | Bombs 15.45 | VB 50.79% | UB 0.00% | NEB 0.00% | Danger 67.7 | Tiles 44.2 | Repeat 81.73% | PG -0.0011 | Val 0.2156 | BC 0.6092 | Ent 0.444 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.687 | SPS 93 | T r/u/e/c 29.4/13.4/0.0/0.0s
Step 2,480,672 | Ep  2047 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 15.52 | VB 51.23% | UB 0.00% | NEB 0.00% | Danger 67.5 | Tiles 44.7 | Repeat 81.44% | PG -0.0010 | Val 0.2625 | BC 0.5480 | Ent 0.424 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.777 | SPS 92 | T r/u/e/c 40.1/13.5/0.0/0.0s
Step 2,481,696 | Ep  2050 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 15.68 | VB 52.33% | UB 0.00% | NEB 0.00% | Danger 68.1 | Tiles 45.0 | Repeat 81.34% | PG -0.0011 | Val 0.2944 | BC 0.5999 | Ent 0.496 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.716 | SPS 92 | T r/u/e/c 30.4/13.5/0.0/0.0s
Step 2,482,720 | Ep  2053 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 15.73 | VB 52.44% | UB 0.00% | NEB 0.00% | Danger 68.5 | Tiles 45.2 | Repeat 81.16% | PG -0.0013 | Val 0.1571 | BC 0.6456 | Ent 0.513 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.015 | EV 0.175 | SPS 92 | T r/u/e/c 37.0/13.5/0.0/0.0s
Step 2,483,744 | Ep  2056 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 16.49 | VB 53.67% | UB 0.00% | NEB 0.00% | Danger 71.6 | Tiles 46.4 | Repeat 81.36% | PG -0.0010 | Val 0.1891 | BC 0.7347 | Ent 0.490 | KL -0.0010 | Clip 0.12% | Ratio 1.001+/-0.019 | EV 0.659 | SPS 92 | T r/u/e/c 46.8/13.7/0.0/0.0s
Step 2,484,768 | Ep  2061 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 16.91 | VB 55.14% | UB 0.00% | NEB 0.00% | Danger 73.1 | Tiles 46.5 | Repeat 81.79% | PG -0.0001 | Val 0.1965 | BC 0.5661 | Ent 0.417 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.868 | SPS 92 | T r/u/e/c 36.3/13.5/0.0/0.0s
Step 2,485,792 | Ep  2064 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 16.40 | VB 53.86% | UB 0.00% | NEB 0.00% | Danger 71.3 | Tiles 45.2 | Repeat 82.02% | PG -0.0015 | Val 0.3589 | BC 0.4138 | Ent 0.346 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.006 | EV 0.752 | SPS 92 | T r/u/e/c 32.7/13.5/0.0/0.0s
Step 2,486,816 | Ep  2067 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 16.76 | VB 54.78% | UB 0.00% | NEB 0.00% | Danger 73.0 | Tiles 46.1 | Repeat 82.06% | PG -0.0021 | Val 0.2175 | BC 0.4744 | Ent 0.463 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.806 | SPS 92 | T r/u/e/c 32.8/13.4/0.0/0.0s
Step 2,487,840 | Ep  2071 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 17.02 | VB 56.28% | UB 0.00% | NEB 0.00% | Danger 73.9 | Tiles 46.7 | Repeat 81.77% | PG -0.0020 | Val 0.2489 | BC 0.3924 | Ent 0.298 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.016 | EV 0.856 | SPS 92 | T r/u/e/c 36.5/13.5/0.0/0.0s
Step 2,488,864 | Ep  2078 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 17.06 | VB 55.74% | UB 0.00% | NEB 0.00% | Danger 73.7 | Tiles 46.0 | Repeat 81.82% | PG -0.0015 | Val 0.1762 | BC 0.5644 | Ent 0.461 | KL 0.0006 | Clip 0.22% | Ratio 1.000+/-0.022 | EV 0.830 | SPS 91 | T r/u/e/c 35.0/13.5/0.0/0.0s
Step 2,489,888 | Ep  2083 | Stage phase_resource_control | ActorTrainable True | FirstRate 37.00% | Bombs 17.21 | VB 55.54% | UB 0.00% | NEB 0.00% | Danger 73.4 | Tiles 45.9 | Repeat 81.50% | PG -0.0001 | Val 0.3030 | BC 0.5915 | Ent 0.494 | KL -0.0003 | Clip 0.00% | Ratio 1.000+/-0.013 | EV 0.751 | SPS 91 | T r/u/e/c 33.6/13.5/0.0/0.0s
Step 2,490,912 | Ep  2088 | Stage phase_resource_control | ActorTrainable True | FirstRate 39.00% | Bombs 17.32 | VB 56.48% | UB 0.00% | NEB 0.00% | Danger 73.0 | Tiles 45.5 | Repeat 81.62% | PG -0.0009 | Val 0.1888 | BC 0.5881 | Ent 0.473 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.818 | SPS 91 | T r/u/e/c 34.1/13.4/0.0/0.0s
Step 2,491,936 | Ep  2090 | Stage phase_resource_control | ActorTrainable True | FirstRate 41.00% | Bombs 17.26 | VB 56.18% | UB 0.00% | NEB 0.00% | Danger 72.5 | Tiles 45.8 | Repeat 81.25% | PG -0.0011 | Val 0.2614 | BC 0.4770 | Ent 0.396 | KL -0.0003 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.805 | SPS 91 | T r/u/e/c 31.9/13.6/0.0/0.0s
Step 2,492,960 | Ep  2094 | Stage phase_resource_control | ActorTrainable True | FirstRate 38.00% | Bombs 17.11 | VB 55.76% | UB 0.00% | NEB 0.00% | Danger 72.4 | Tiles 45.3 | Repeat 81.41% | PG -0.0012 | Val 0.2886 | BC 0.3617 | Ent 0.263 | KL -0.0003 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.844 | SPS 91 | T r/u/e/c 29.1/13.8/0.0/0.0s
Step 2,493,984 | Ep  2100 | Stage phase_resource_control | ActorTrainable True | FirstRate 37.00% | Bombs 16.85 | VB 55.20% | UB 0.00% | NEB 0.00% | Danger 71.5 | Tiles 44.5 | Repeat 81.78% | PG -0.0013 | Val 0.2443 | BC 0.5510 | Ent 0.409 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.014 | EV 0.803 | SPS 91 | T r/u/e/c 40.9/13.8/0.0/0.0s
Step 2,495,008 | Ep  2102 | Stage phase_resource_control | ActorTrainable True | FirstRate 37.00% | Bombs 16.83 | VB 55.62% | UB 0.00% | NEB 0.00% | Danger 71.2 | Tiles 44.6 | Repeat 81.65% | PG -0.0021 | Val 0.2420 | BC 0.4080 | Ent 0.383 | KL 0.0005 | Clip 0.05% | Ratio 1.000+/-0.016 | EV 0.859 | SPS 91 | T r/u/e/c 35.5/13.5/0.0/0.0s
Step 2,496,032 | Ep  2106 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 16.23 | VB 55.13% | UB 0.00% | NEB 0.00% | Danger 69.3 | Tiles 43.6 | Repeat 81.94% | PG -0.0021 | Val 0.2416 | BC 0.4071 | Ent 0.314 | KL 0.0008 | Clip 0.02% | Ratio 0.999+/-0.011 | EV 0.827 | SPS 91 | T r/u/e/c 37.1/13.6/0.0/0.0s
Step 2,497,056 | Ep  2108 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 16.43 | VB 55.38% | UB 0.00% | NEB 0.00% | Danger 69.7 | Tiles 44.5 | Repeat 81.66% | PG -0.0017 | Val 0.2123 | BC 0.2019 | Ent 0.147 | KL -0.0003 | Clip 0.02% | Ratio 1.000+/-0.007 | EV 0.847 | SPS 90 | T r/u/e/c 22.6/13.6/0.0/0.0s
Step 2,498,080 | Ep  2113 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 16.46 | VB 55.51% | UB 0.00% | NEB 0.00% | Danger 70.5 | Tiles 44.3 | Repeat 81.51% | PG -0.0011 | Val 0.1800 | BC 0.4361 | Ent 0.389 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.843 | SPS 90 | T r/u/e/c 27.3/13.5/0.0/0.0s
Step 2,499,104 | Ep  2117 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 15.81 | VB 54.53% | UB 0.00% | NEB 0.00% | Danger 68.4 | Tiles 43.8 | Repeat 81.42% | PG -0.0012 | Val 0.2438 | BC 0.4264 | Ent 0.309 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.853 | SPS 90 | T r/u/e/c 25.3/13.5/0.0/0.0s
Step 2,500,128 | Ep  2121 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 16.17 | VB 56.12% | UB 0.00% | NEB 0.00% | Danger 71.0 | Tiles 44.4 | Repeat 81.52% | PG -0.0018 | Val 0.2078 | BC 0.5487 | Ent 0.416 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.837 | SPS 90 | T r/u/e/c 38.3/13.4/0.0/0.0s
Step 2,501,152 | Ep  2124 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 16.21 | VB 56.82% | UB 0.00% | NEB 0.00% | Danger 71.1 | Tiles 44.5 | Repeat 81.48% | PG -0.0013 | Val 0.1982 | BC 0.3213 | Ent 0.274 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.855 | SPS 90 | T r/u/e/c 26.8/13.4/0.0/0.0s
Step 2,502,176 | Ep  2127 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 15.95 | VB 57.35% | UB 0.00% | NEB 0.00% | Danger 70.2 | Tiles 44.5 | Repeat 81.51% | PG -0.0013 | Val 0.1070 | BC 0.3573 | Ent 0.286 | KL 0.0005 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.917 | SPS 90 | T r/u/e/c 24.5/13.6/0.0/0.0s
Step 2,503,200 | Ep  2129 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 15.89 | VB 57.64% | UB 0.00% | NEB 0.00% | Danger 69.8 | Tiles 44.5 | Repeat 81.60% | PG -0.0009 | Val 0.2166 | BC 0.2593 | Ent 0.199 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.834 | SPS 90 | T r/u/e/c 18.2/13.6/0.0/0.0s
  -> Saved checkpoint: D:\other\CNTT\Bomberland-GDGoC-AI-Challenge\agent\agent\v6\checkpoints\model_step2503200.pth
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
  -> Eval score 2.200 | AvgRank 0.80 | R0/1/2/3 60%/0%/40%/0% | UF 60%
     EM score/rank 2.200/0.80 | R0/1/2/3 60%/0%/40%/0% | VB/Repeat 46.70%/73.94%
     Hard score/rank 2.200/0.80 | R0/1/2/3 60%/0%/40%/0% | VB/Repeat 70.95%/85.71%
     Diff K/B/I/Bp -0.70/+1.50/-1.60/-1.40 | Cap/Radius -0.30/-0.80 | Timeout loss K/B/I/B 20.00%/0.00%/0.00%/0.00%
     TIMING rollout 18.2s | update 13.6s | checkpoint 0.1s | eval 50.5s
Step 2,504,224 | Ep  2133 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 15.56 | VB 56.92% | UB 0.00% | NEB 0.00% | Danger 69.0 | Tiles 43.6 | Repeat 81.81% | PG -0.0016 | Val 0.2797 | BC 0.4181 | Ent 0.321 | KL -0.0005 | Clip 0.00% | Ratio 1.001+/-0.013 | EV 0.854 | SPS 90 | T r/u/e/c 30.1/13.8/0.0/0.0s
Step 2,505,248 | Ep  2138 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 15.42 | VB 56.96% | UB 0.00% | NEB 0.00% | Danger 68.3 | Tiles 43.5 | Repeat 81.77% | PG -0.0007 | Val 0.3021 | BC 0.4396 | Ent 0.383 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.802 | SPS 90 | T r/u/e/c 31.9/13.7/0.0/0.0s
Step 2,506,272 | Ep  2140 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 15.28 | VB 56.35% | UB 0.00% | NEB 0.00% | Danger 67.9 | Tiles 43.4 | Repeat 81.82% | PG -0.0015 | Val 0.1524 | BC 0.5627 | Ent 0.454 | KL -0.0003 | Clip 0.00% | Ratio 1.000+/-0.013 | EV 0.908 | SPS 89 | T r/u/e/c 36.2/13.7/0.0/0.0s
Step 2,507,296 | Ep  2144 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 15.44 | VB 55.82% | UB 0.00% | NEB 0.00% | Danger 68.3 | Tiles 43.7 | Repeat 82.04% | PG -0.0009 | Val 0.1797 | BC 0.5432 | Ent 0.417 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.863 | SPS 89 | T r/u/e/c 45.6/13.6/0.0/0.0s
Step 2,508,320 | Ep  2144 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 15.44 | VB 55.82% | UB 0.00% | NEB 0.00% | Danger 68.3 | Tiles 43.7 | Repeat 82.04% | PG -0.0009 | Val 0.1816 | BC 0.3382 | Ent 0.218 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.900 | SPS 89 | T r/u/e/c 35.0/13.5/0.0/0.0s
Step 2,509,344 | Ep  2150 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 15.24 | VB 55.95% | UB 0.00% | NEB 0.00% | Danger 67.9 | Tiles 43.0 | Repeat 82.62% | PG -0.0012 | Val 0.2267 | BC 0.4924 | Ent 0.376 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.840 | SPS 89 | T r/u/e/c 31.5/13.5/0.0/0.0s
Step 2,510,368 | Ep  2152 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 15.57 | VB 56.44% | UB 0.00% | NEB 0.00% | Danger 68.8 | Tiles 43.5 | Repeat 82.68% | PG -0.0015 | Val 0.1501 | BC 0.5622 | Ent 0.487 | KL -0.0005 | Clip 0.00% | Ratio 1.001+/-0.015 | EV 0.856 | SPS 89 | T r/u/e/c 35.3/13.5/0.0/0.0s
Step 2,511,392 | Ep  2156 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 15.15 | VB 56.04% | UB 0.00% | NEB 0.00% | Danger 67.7 | Tiles 43.1 | Repeat 82.90% | PG -0.0015 | Val 0.1164 | BC 0.5204 | Ent 0.471 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.013 | EV 0.903 | SPS 89 | T r/u/e/c 33.1/13.7/0.0/0.0s
Step 2,512,416 | Ep  2158 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 15.01 | VB 55.41% | UB 0.00% | NEB 0.00% | Danger 66.9 | Tiles 43.1 | Repeat 82.63% | PG -0.0009 | Val 0.1005 | BC 0.6461 | Ent 0.571 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.168 | SPS 89 | T r/u/e/c 39.7/13.4/0.0/0.0s
Step 2,513,440 | Ep  2161 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 14.81 | VB 55.37% | UB 0.00% | NEB 0.00% | Danger 66.9 | Tiles 43.4 | Repeat 82.68% | PG -0.0013 | Val 0.2157 | BC 0.4398 | Ent 0.371 | KL 0.0000 | Clip 0.02% | Ratio 1.000+/-0.011 | EV 0.838 | SPS 89 | T r/u/e/c 30.0/13.4/0.0/0.0s
Step 2,514,464 | Ep  2165 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 15.20 | VB 56.72% | UB 0.00% | NEB 0.00% | Danger 68.7 | Tiles 44.9 | Repeat 82.15% | PG -0.0021 | Val 0.2519 | BC 0.5299 | Ent 0.410 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.014 | EV 0.870 | SPS 88 | T r/u/e/c 42.9/13.4/0.0/0.0s
Step 2,515,488 | Ep  2168 | Stage phase_resource_control | ActorTrainable True | FirstRate 31.00% | Bombs 14.98 | VB 55.95% | UB 0.00% | NEB 0.00% | Danger 67.1 | Tiles 44.2 | Repeat 82.28% | PG -0.0011 | Val 0.1396 | BC 0.5473 | Ent 0.455 | KL 0.0004 | Clip 0.02% | Ratio 1.000+/-0.016 | EV 0.836 | SPS 88 | T r/u/e/c 41.9/13.5/0.0/0.0s
Step 2,516,512 | Ep  2175 | Stage phase_resource_control | ActorTrainable True | FirstRate 33.00% | Bombs 15.07 | VB 55.55% | UB 0.00% | NEB 0.00% | Danger 66.3 | Tiles 44.9 | Repeat 81.67% | PG -0.0016 | Val 0.3325 | BC 0.6135 | Ent 0.480 | KL -0.0000 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.691 | SPS 88 | T r/u/e/c 38.9/13.6/0.0/0.0s
Step 2,517,536 | Ep  2177 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 14.94 | VB 55.79% | UB 0.00% | NEB 0.00% | Danger 66.0 | Tiles 44.5 | Repeat 81.83% | PG -0.0009 | Val 0.3016 | BC 0.3644 | Ent 0.269 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.824 | SPS 88 | T r/u/e/c 25.6/13.4/0.0/0.0s
Step 2,518,560 | Ep  2181 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 15.41 | VB 56.59% | UB 0.00% | NEB 0.00% | Danger 69.2 | Tiles 45.4 | Repeat 82.20% | PG -0.0011 | Val 0.1939 | BC 0.2223 | Ent 0.186 | KL -0.0004 | Clip 0.02% | Ratio 1.000+/-0.009 | EV 0.892 | SPS 88 | T r/u/e/c 24.8/13.5/0.0/0.0s
Step 2,519,584 | Ep  2185 | Stage phase_resource_control | ActorTrainable True | FirstRate 29.00% | Bombs 14.99 | VB 54.51% | UB 0.00% | NEB 0.00% | Danger 67.6 | Tiles 44.5 | Repeat 82.39% | PG -0.0016 | Val 0.4949 | BC 0.3953 | Ent 0.310 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.009 | EV 0.693 | SPS 88 | T r/u/e/c 28.0/13.7/0.0/0.0s
Step 2,520,608 | Ep  2189 | Stage phase_resource_control | ActorTrainable True | FirstRate 26.00% | Bombs 14.71 | VB 53.35% | UB 0.00% | NEB 0.00% | Danger 67.1 | Tiles 44.0 | Repeat 83.10% | PG -0.0007 | Val 0.1223 | BC 0.3294 | Ent 0.262 | KL -0.0001 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.926 | SPS 88 | T r/u/e/c 29.3/13.7/0.0/0.0s
Step 2,521,632 | Ep  2194 | Stage phase_resource_control | ActorTrainable True | FirstRate 27.00% | Bombs 14.46 | VB 53.18% | UB 0.00% | NEB 0.00% | Danger 65.8 | Tiles 43.6 | Repeat 83.07% | PG -0.0012 | Val 0.2865 | BC 0.5240 | Ent 0.413 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.824 | SPS 88 | T r/u/e/c 35.0/13.7/0.0/0.0s
Step 2,522,656 | Ep  2197 | Stage phase_resource_control | ActorTrainable True | FirstRate 26.00% | Bombs 14.27 | VB 52.35% | UB 0.00% | NEB 0.00% | Danger 65.2 | Tiles 43.2 | Repeat 82.94% | PG -0.0009 | Val 0.3142 | BC 0.4577 | Ent 0.383 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.006 | EV 0.786 | SPS 88 | T r/u/e/c 32.3/13.7/0.0/0.0s
Step 2,523,680 | Ep  2202 | Stage phase_resource_control | ActorTrainable True | FirstRate 28.00% | Bombs 14.71 | VB 52.83% | UB 0.00% | NEB 0.00% | Danger 66.9 | Tiles 44.4 | Repeat 82.59% | PG -0.0012 | Val 0.2137 | BC 0.6135 | Ent 0.476 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.008 | EV 0.611 | SPS 87 | T r/u/e/c 37.5/13.4/0.0/0.0s
Step 2,524,704 | Ep  2204 | Stage phase_resource_control | ActorTrainable True | FirstRate 30.00% | Bombs 14.79 | VB 53.41% | UB 0.00% | NEB 0.00% | Danger 67.7 | Tiles 44.8 | Repeat 82.56% | PG -0.0023 | Val 0.0607 | BC 0.7243 | Ent 0.511 | KL 0.0007 | Clip 0.05% | Ratio 1.000+/-0.016 | EV 0.557 | SPS 87 | T r/u/e/c 45.7/13.7/0.0/0.0s
Step 2,525,728 | Ep  2209 | Stage phase_resource_control | ActorTrainable True | FirstRate 32.00% | Bombs 15.11 | VB 54.32% | UB 0.00% | NEB 0.00% | Danger 68.4 | Tiles 45.0 | Repeat 82.31% | PG -0.0018 | Val 0.2394 | BC 0.6902 | Ent 0.507 | KL 0.0000 | Clip 0.00% | Ratio 1.000+/-0.019 | EV 0.396 | SPS 87 | T r/u/e/c 36.8/13.8/0.0/0.0s
Step 2,526,752 | Ep  2212 | Stage phase_resource_control | ActorTrainable True | FirstRate 34.00% | Bombs 15.20 | VB 54.26% | UB 0.00% | NEB 0.00% | Danger 68.1 | Tiles 45.2 | Repeat 82.02% | PG -0.0012 | Val 0.2632 | BC 0.5538 | Ent 0.442 | KL 0.0004 | Clip 0.00% | Ratio 1.000+/-0.018 | EV 0.751 | SPS 87 | T r/u/e/c 41.8/13.4/0.0/0.0s
Step 2,527,776 | Ep  2216 | Stage phase_resource_control | ActorTrainable True | FirstRate 35.00% | Bombs 15.83 | VB 54.77% | UB 0.00% | NEB 0.00% | Danger 70.2 | Tiles 46.0 | Repeat 82.45% | PG -0.0007 | Val 0.1346 | BC 0.5934 | Ent 0.470 | KL 0.0000 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.854 | SPS 87 | T r/u/e/c 46.6/13.3/0.0/0.0s
Step 2,528,800 | Ep  2221 | Stage phase_resource_control | ActorTrainable True | FirstRate 37.00% | Bombs 16.21 | VB 55.45% | UB 0.00% | NEB 0.00% | Danger 70.5 | Tiles 46.9 | Repeat 82.00% | PG -0.0014 | Val 0.2918 | BC 0.6756 | Ent 0.483 | KL 0.0000 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.697 | SPS 87 | T r/u/e/c 38.9/14.4/0.0/0.0s
  -> Saved checkpoint: D:\other\CNTT\Bomberland-GDGoC-AI-Challenge\agent\agent\v6\checkpoints\model_step2528800.pth
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
Using device: cpu
  -> Eval score 2.300 | AvgRank 0.70 | R0/1/2/3 60%/10%/30%/0% | UF 60%
     EM score/rank 2.200/0.80 | R0/1/2/3 60%/0%/40%/0% | VB/Repeat 58.16%/79.56%
     Hard score/rank 2.400/0.60 | R0/1/2/3 60%/20%/20%/0% | VB/Repeat 64.30%/79.95%
     Diff K/B/I/Bp +0.10/-0.60/-2.90/-8.90 | Cap/Radius -0.40/-1.90 | Timeout loss K/B/I/B 10.00%/10.00%/0.00%/0.00%
     TIMING rollout 38.9s | update 14.4s | checkpoint 0.1s | eval 42.4s
Step 2,529,824 | Ep  2224 | Stage phase_resource_control | ActorTrainable True | FirstRate 37.00% | Bombs 16.12 | VB 54.06% | UB 0.00% | NEB 0.00% | Danger 70.1 | Tiles 46.6 | Repeat 82.12% | PG -0.0018 | Val 0.3061 | BC 0.4780 | Ent 0.360 | KL -0.0003 | Clip 0.00% | Ratio 1.000+/-0.012 | EV 0.831 | SPS 87 | T r/u/e/c 21.7/13.1/0.0/0.0s
Step 2,530,848 | Ep  2230 | Stage phase_resource_control | ActorTrainable True | FirstRate 38.00% | Bombs 16.64 | VB 53.55% | UB 0.00% | NEB 0.00% | Danger 71.5 | Tiles 47.0 | Repeat 81.58% | PG -0.0008 | Val 0.1233 | BC 0.5791 | Ent 0.502 | KL -0.0005 | Clip 0.00% | Ratio 1.001+/-0.012 | EV 0.831 | SPS 87 | T r/u/e/c 25.1/13.2/0.0/0.0s
Step 2,531,872 | Ep  2233 | Stage phase_resource_control | ActorTrainable True | FirstRate 39.00% | Bombs 16.93 | VB 54.23% | UB 0.00% | NEB 0.00% | Danger 71.3 | Tiles 47.6 | Repeat 81.20% | PG -0.0009 | Val 0.2539 | BC 0.6394 | Ent 0.483 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.010 | EV 0.719 | SPS 86 | T r/u/e/c 32.0/13.3/0.0/0.0s
Step 2,532,896 | Ep  2236 | Stage phase_resource_control | ActorTrainable True | FirstRate 39.00% | Bombs 16.84 | VB 54.13% | UB 0.00% | NEB 0.00% | Danger 71.0 | Tiles 47.2 | Repeat 81.35% | PG -0.0011 | Val 0.2467 | BC 0.4563 | Ent 0.325 | KL 0.0002 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.832 | SPS 86 | T r/u/e/c 36.5/13.1/0.0/0.0s
Step 2,533,920 | Ep  2240 | Stage phase_resource_control | ActorTrainable True | FirstRate 39.00% | Bombs 17.08 | VB 54.51% | UB 0.00% | NEB 0.00% | Danger 72.8 | Tiles 47.6 | Repeat 81.41% | PG -0.0014 | Val 0.2221 | BC 0.5125 | Ent 0.414 | KL -0.0002 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.830 | SPS 86 | T r/u/e/c 33.3/13.5/0.0/0.0s
Step 2,534,944 | Ep  2244 | Stage phase_resource_control | ActorTrainable True | FirstRate 38.00% | Bombs 17.00 | VB 55.23% | UB 0.00% | NEB 0.00% | Danger 72.5 | Tiles 47.1 | Repeat 81.60% | PG -0.0009 | Val 0.2638 | BC 0.4933 | Ent 0.378 | KL 0.0001 | Clip 0.00% | Ratio 1.000+/-0.011 | EV 0.836 | SPS 86 | T r/u/e/c 26.6/13.5/0.0/0.0s
Step 2,535,968 | Ep  2246 | Stage phase_resource_control | ActorTrainable True | FirstRate 37.00% | Bombs 16.48 | VB 54.10% | UB 0.00% | NEB 0.00% | Danger 71.3 | Tiles 46.5 | Repeat 81.66% | PG -0.0014 | Val 0.1308 | BC 0.5122 | Ent 0.382 | KL 0.0003 | Clip 0.00% | Ratio 1.000+/-0.007 | EV 0.879 | SPS 86 | T r/u/e/c 29.8/13.9/0.0/0.0s
