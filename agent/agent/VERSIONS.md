# Version Layout

Each model version now has its own folder under `agent/agent/`:

- `v1/`
- `v2/`
- `v3/`
- `v4_1/`

Each folder is intended to own its own artifacts:

- `agent.py`: inference entrypoint for that version
- `model.py`: model/helpers for that version
- `train_ppo.py`: training script for that version
- optional internal helpers prefixed with `_`: only support files for that
  version, not cross-version entrypoints
- `model.pth`: best checkpoint for that version
- `model_last.pth`: latest checkpoint for that version
- `checkpoints/`: intermediate checkpoints for that version

Typical usage:

```powershell
python agent/agent/v3/train_ppo.py
```

After training on Kaggle/Colab, copy the exported weight back into the matching
folder as `model.pth`.
