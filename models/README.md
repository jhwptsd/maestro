# Models
This folder establishes checkpoints for all trained models used in this project. It should be listed on gitignore.
## fusion/
`audio.pt`, `video.pt`, `score.pt`: the residual autoencoders of `maestro/fusion/`, written by
`src/training/train_fusion.py` (plain tensors and dicts, loaded with `weights_only=True`). `MAESTRO_MODELS` moves the models folder. Train them with `python src/training/train_fusion.py` (see the root README); `analyze()` uses them automatically when they exist.

Other weights are not kept here: VioPose and its weights live in `third_party/` (`src/scripts/setup_viopose.py`), the optional CLAP
model is cached by `transformers` (the Hugging Face cache) on first use, and torchcrepe ships its own.
