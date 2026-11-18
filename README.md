# maestro
[WIP] An app that takes a video of a violin performance alongside a score and provides personalized feedback.

## Audio pitch backends

Maestro supports interchangeable audio backends behind one `PitchTrack` interface:

| Backend | Select with | Output | Optional install |
|---|---|---|---|
| CREPE (default) | `crepe` | Frame-wise F0 and periodicity confidence | `pip install -e ".[crepe]"` |
| NumPy YIN | `yin` | Frame-wise F0 and YIN confidence | No extra dependency |
| Librosa pYIN | `pyin` | Probabilistic F0 and voiced probability with temporal decoding | `pip install -e ".[pyin]"` |
| Spotify Basic Pitch | `basic_pitch` | Polyphonic note events; dominant-note projection for frame plots | `pip install -e ".[basic-pitch]"` |

Install the default backend and development tools with:

```bash
pip install -e ".[crepe,dev]"
```

To install all four implementations:

```bash
pip install -e ".[audio,dev]"
```

Install the PyTorch build appropriate for the machine before installing torchcrepe when GPU support is needed. CREPE automatically selects CUDA when available; pass `pitch_options={"device": "cpu"}` to `analyze()` to force CPU inference.

The CREPE model exposed by torchcrepe has a maximum F0 of approximately 2006 Hz. This excludes some of the violin's highest fundamentals. Use `pyin` or `yin` with a higher `fmax` for those passages, for example `analyze(..., pitch_backend="pyin", pitch_options={"fmax": 4200})`.

Basic Pitch is an event-based polyphonic transcriber. It can preserve overlapping notes such as double stops, but its note-event MIDI pitches are semitone-quantized. Do not use Basic Pitch alone as the source of cents-level intonation assessment; use CREPE, pYIN or YIN for that purpose. Its frame F0 output is a strongest-active-note projection, while the full list of note events remains available to transcription and alignment.

### Choosing a backend

```python
from maestro import analyze

# CREPE is the default.
performance = analyze("performance.mp4", "score.musicxml")

# Compare with the NumPy implementation without installing optional ML packages.
performance_yin = analyze("performance.mp4", "score.musicxml", pitch_backend="yin")

# pYIN supports a higher F0 range than torchcrepe.
performance_pyin = analyze(
    "performance.mp4",
    "score.musicxml",
    pitch_backend="pyin",
    pitch_options={"fmax": 4200},
)

# Basic Pitch can retain simultaneous note events.
performance_polyphonic = analyze(
    "performance.mp4", "score.musicxml", pitch_backend="basic_pitch"
)
```

The lower-level interface is `maestro.audio.pitch.estimate_pitch()`. It returns a `PitchTrack` with `times`, `f0`, `confidence`, `backend`, and optional `notes`. The old `estimate_f0()` function remains available and still returns the three-array tuple `(times, f0, confidence)`.

Maestro estimates pitch once per performance and shares the result between note transcription and frame-feature extraction, avoiding the previous duplicate inference pass.

## Running

Requires `ffmpeg` on PATH (and optionally Audiveris for PDF/image scores).

`.\setup.ps1` (Windows, ffmpeg via winget) or `./setup.sh` (Linux/macOS) installs ffmpeg and git if missing, the package
and VioPose for the movement analysis (see below); add `-SkipViopose` / `--skip-viopose` to leave VioPose out.

```bash
pip install -e ".[crepe,dev]"
pytest
python src/scripts/analyze.py performance.mp4 score.musicxml out/ --pitch-backend crepe
python src/scripts/analyze.py performance.mp4 score.musicxml out-yin/ --pitch-backend yin
```

## Movement analysis (VioPose)

Bow movement, arm angles, posture and the pitch implied by the left hand's position come from a 3D pose that
[VioPose](https://github.com/SeongJong-Yoo/VioPose) (MIT) estimates from the video. The pose is generated and used
automatically: upload a video and a score and the movement sections appear in the report; nothing else is submitted or
configured, and the generated pose is also saved as `pose.json` next to the other outputs.

VioPose pins TensorFlow 2.13 (Python 3.9), so the setup scripts install it separately: cloned into `third_party/`
with its own Python 3.9 environment (conda, uv, or a `python3.9` venv; uv is installed if none is found), its requirements
(several GB) and its model weights (Google Drive). `python src/scripts/setup_viopose.py` does just that step, is safe to
re-run (finished steps are skipped) and takes `--dry-run`, `--skip-weights`, `--python PATH` (an existing environment)
and `--dir`. The result is recorded in `third_party/viopose.json`; the variables `VIOPOSE_DIR` and `VIOPOSE_PYTHON`
override it, and `VIOPOSE_AUDIO=1` also gives VioPose the recording's audio. Without VioPose the audio analysis works as
before and the report says how to install it. Details and the visual pitch method are in `src/docs/README.md`.

## Status

Implemented: audio extraction, selectable pitch/transcription backends, score conversion, score-audio alignment, pitch-deviation and timing assessment, and score visualization. Movement analysis from VioPose poses (bowing, posture, visual pitch) is implemented and independent of the audio analysis; combining the two, multimodal fusion and LLM feedback remain in development.
