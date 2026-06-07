# v6_2 BC Pipeline

## 1. Convert public replay manifest to BC shards

```powershell
python agent/agent/v6_2/prepare_public_bc_dataset.py `
  --manifest data/public_replays/agent_2f875942_top50/manifest.csv `
  --output_dir agent/agent/v6_2/bc_data_public_top1 `
  --agent_id 2f875942-8faf-4f9c-8008-9b6acf48d3e0 `
  --episodes_per_shard 100
```

## 2. Analyze dataset

```powershell
python agent/agent/v6_2/analyze_dataset.py `
  --data_dir agent/agent/v6_2/bc_data_public_top1
```

## 3. Train BC

```powershell
python agent/agent/v6_2/train_bc.py `
  --data_dir agent/agent/v6_2/bc_data_public_top1 `
  --output agent/agent/v6_2/bc_actor_public_top1.pth `
  --epochs 24 `
  --batch_size 64 `
  --seq_len 64 `
  --stride 16 `
  --bomb_weight 1.25
```

## 4. Evaluate BC

```powershell
python agent/agent/v6_2/eval_bc.py `
  --checkpoint agent/agent/v6_2/bc_actor_public_top1.pth `
  --pool hard `
  --num_matches 20
```

## 5. Export actor checkpoint if needed

```powershell
python agent/agent/v6_2/export_bc_agent.py `
  --input agent/agent/v6_2/bc_actor_public_top1.pth `
  --output agent/agent/v6_2/bc_actor.pth
```
