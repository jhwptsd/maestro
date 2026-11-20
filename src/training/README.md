# Training
Scripts that train the models in `maestro/` from data. Data lives in `data/`, trained models in `models/` (see the README at
the repository root, "Train the models", for the whole picture). Install the extras first: `pip install -e ".[training]"`
(or `./setup.sh --training` / `.\setup.ps1 -Training`); the video dataset also needs ffmpeg and VioPose.

| file | what it does |
|---|---|
| `install_data.py` | `list` what is installed; `audio` (MusicNet violin recordings); `video` (Creative Commons videos -> VioPose poses); `ingest NAME audio\|video FOLDER` (data you obtained yourself) |
| `datasets.py` | the installers and the `data/` layout: resumable checksum-verified downloads, per-recording train/val/test partitions, manifests with source, licence and attribution |
| `fusion_data.py` | turns files into training windows with Maestro's own analysis (so training sees the same side information as inference); caches audio windows in `data/cache/`; lists suggested public datasets |
| `train_fusion.py` | trains one fusion encoder and writes `models/fusion/<audio\|video\|score>.pt` |

## Workflow
    python src/training/install_data.py audio                      # data/audio/musicnet/{train,val,test}/ (large download)
    python src/training/install_data.py video --count 60           # data/video/youtube-cc/{train,val,test}/ (+ audio/youtube-cc)
    python src/training/train_fusion.py audio --dataset musicnet --epochs 30
    python src/training/train_fusion.py video --dataset youtube-cc --epochs 40
    python src/training/train_fusion.py score --music21 --epochs 40      # the scores bundled with music21, nothing to install
    python src/training/train_fusion.py audio my_recordings/ --cache cache/     # your own files instead of a dataset

`--dataset` trains on the dataset's train partition, picks the best epoch on val and scores test once at the end (stored in the
checkpoint). Useful options: `--z-dim`, `--beta` (KL weight), `--leak` (z / analysis cross-covariance weight), `--batch`, `--lr`,
`--limit`, `--device`, `--out`. After training, `maestro.analyze()` uses the checkpoints automatically (`perf.embeddings`).
Request-only datasets (VioDat, SyncViolinist, MOSA) are brought in with `install_data.py ingest`: convert them to wav or
`(frames, 18, 3)` millimetre `.npy` arrays in `pose_io.VIOPOSE_JOINTS` order first.
