# Data
Training data, all of it git-ignored. Installed and partitioned by `src/training/install_data.py` (move this folder with the
`MAESTRO_DATA` environment variable); trained models go to `models/` (`models/fusion/audio.pt`, `video.pt`, `score.pt`). The
whole workflow (install, train, use) is in the root README under "Train the models" and in `src/training/README.md`.

    downloads/<dataset>/               archives and videos as downloaded (safe to delete once installed)
    audio/<dataset>/train|val|test/    wav files;                 manifest.json: id, split, source, licence, attribution
    video/<dataset>/train|val|test/    pose files (Maestro pose JSON, or (frames, 18, 3) mm .npy); manifest.json likewise
    cache/audio/                       analysed audio windows (slow to compute, so cached)

Recordings are split by a hash of their id (80 / 10 / 10), never by window, so a recording is in exactly one split and
re-running an install never moves it. Datasets (details and licences in `src/training/datasets.py`):

| dataset | for | how |
|---|---|---|
| `musicnet` | audio | `install_data.py audio`: the violin recordings of MusicNet (Zenodo, CC BY 4.0), downloaded and extracted automatically |
| `youtube-cc` | video, audio | `install_data.py video`: Creative Commons violin videos (yt-dlp), poses estimated with VioPose |
| VioDat, SyncViolinist, MOSA | video, audio | need a request form and are research-only: `install_data.py ingest NAME audio\|video FOLDER` |

The project's earlier note on MOSA (Music Motion with Semantic Annotation, Huang et al. 2024): its full data on Zenodo
(record 11393449) is released on request, 742 solo piano and violin performances with audio and motion capture.
