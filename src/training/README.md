# Training
Scripts to train the models in `maestro/`; checkpoints are written to `models/` (see `models/README.md`).

## Fusion encoders (`maestro/fusion/`)
`train_fusion.py {audio,video,score} PATH...` trains one residual autoencoder (see `maestro/fusion/residual_autoencoder.py`
for the idea: the decoder also receives the analysis features, so the latent holds what the analysis misses).
`fusion_data.py` reads the data (and lists suggested public datasets and their licences). Needs `pip install -e ".[fusion]"`.

    python src/training/train_fusion.py score --music21 --epochs 20          # bundled scores, no download
    python src/training/train_fusion.py audio recordings/ --cache cache/     # any violin audio or video
    python src/training/train_fusion.py video poses/                         # pose.json files or (frames, 18, 3) mm arrays
