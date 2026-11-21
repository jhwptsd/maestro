# Tests
Tests for every major component of this project, and every individual neural network. Run them from the repository root:

    pip install -e ".[crepe,fusion,dev]"
    pytest                                                        # configured in pyproject.toml
    PYTHONPATH=src python -m unittest discover -s src/tests       # the same, without pytest

No network and no downloads are needed: synthetic performances stand in for recordings, a fake VioPose checkout stands in for the
pose model, and a local HTTP server with range support stands in for Zenodo.

| file | covers |
|---|---|
| `test_audio.py`, `test_crepe_backend.py`, `test_pitch_speed.py` | pitch backends, transcription, onsets |
| `test_device.py` | device choice (any GPU, else CPU), CREPE through ONNX Runtime (matches PyTorch, falls back cleanly) |
| `test_ffmpeg.py` | finding ffmpeg/ffprobe and the readable error when they are missing |
| `test_score.py`, `test_preprocessing.py`, `test_denoise.py` | score parsing (repeats, dynamics), file conversion, denoising |
| `test_alignment_assessment.py`, `test_assessment_events.py` | alignment, the assessment modules and their events, causes and synthesis |
| `test_vision.py`, `test_viopose_setup.py` | pose, motion features, visual pitch, the VioPose runner and its installer |
| `test_gui.py`, `test_viewer.py` | the player page data, the server (range requests), the score viewer |
| `test_pipeline.py` | end to end through `analyze()` |
| `test_fusion.py` | the residual-autoencoder idea (known information leaves z), the three models, embeddings, training |
| `test_datasets.py` | dataset installation, partitions, ingest, training from `data/` and saving to `models/` |

`test_fusion.py` and `test_datasets.py` need torch and are skipped without it. One test in `test_gui.py` and one in `test_viewer.py`
drive the pages in headless Chromium and are skipped unless `node` and `playwright` are installed. Helpers: `helpers.py`
(synthetic audio/video/MusicXML), `minuet.py` (a realistic Minuet), `pose_fixture.py` (synthetic violinists, a fake VioPose),
`noise_fixture.py`.
