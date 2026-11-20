# Models
This folder establishes checkpoints for all trained models used in this project. It should be listed on gitignore.
## fusion/
`audio.pt`, `video.pt`, `score.pt`: the residual autoencoders of `maestro/fusion/`, written by
`src/training/train_fusion.py` (plain tensors and dicts, loaded with `weights_only=True`). `MAESTRO_MODELS` moves this folder.
