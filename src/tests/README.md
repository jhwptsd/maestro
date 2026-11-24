# Tests
Tests for every major component of this project, and every individual neural network. Run them from the repository root:

    pip install -e ".[crepe,dev]"
    pytest                                                        # configured in pyproject.toml
    PYTHONPATH=src python -m unittest discover -s src/tests       # the same, without pytest

No network and no downloads are needed: synthetic performances stand in for recordings and a fake VioPose checkout stands in for the
pose model.

| file | covers |
|---|---|
| `test_audio.py`, `test_crepe_backend.py`, `test_pitch_speed.py` | pitch backends, transcription, onsets |
| `test_device.py` | device choice (any GPU, else CPU), CREPE through ONNX Runtime (matches PyTorch, falls back cleanly) |
| `test_feedback.py` | the LLM layer: the local Ollama model, OpenRouter and Groq (against local fakes), retries and model fallback, OpenRouter sign-in, saved settings, the digest, the generator and its bar check, the chat, the server endpoints and their cross-site protection, the feedback card and chat in Chromium |
| `test_ffmpeg.py` | finding ffmpeg/ffprobe and the readable error when they are missing |
| `test_score.py`, `test_preprocessing.py`, `test_denoise.py` | score parsing (repeats, dynamics), file conversion, denoising |
| `test_alignment_assessment.py`, `test_assessment_events.py` | alignment, the assessment modules and their events, causes and synthesis |
| `test_vision.py`, `test_viopose_setup.py` | pose, motion features, visual pitch, the VioPose runner and its installer |
| `test_gui.py`, `test_viewer.py` | the player page data, the server (range requests, background analysis and its progress), the progress bar of the start page in Chromium, the score viewer |
| `test_pipeline.py` | end to end through `analyze()` |

One test in `test_gui.py` and one in `test_viewer.py`
drive the pages in headless Chromium and are skipped unless `node` and `playwright` are installed. Helpers: `helpers.py`
(synthetic audio/video/MusicXML), `minuet.py` (a realistic Minuet), `pose_fixture.py` (synthetic violinists, a fake VioPose),
`noise_fixture.py`.
