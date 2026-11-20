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
override it, and `VIOPOSE_AUDIO=0` runs VioPose without the recording's audio (by default its audio model is used). Without VioPose the audio analysis works as
before and the report says how to install it. Details and the visual pitch method are in `src/docs/README.md`.

## Assessment events and root causes

`maestro/assessment/` turns the aligned audio, video and score into **events**: one JSON object per problem, in a single
shared form (`assessment/events.py`): `module`, `category`, `start`/`end` (s), `measure`/`beat`, `severity` (0-1), measured
`value` vs `expected` (+ `unit`), and `causes`: cause slugs with a confidence and the measurements they rest on. The
events are purely quantitative (what, when, why as numbers and slugs, no prose): putting them into words is the job of the
feedback stage.

| file | events | causes come from |
|---|---|---|
| `intonation.py` | sharp / flat / wrong note | global tuning offset, shift overshoot (the leap into the note), drift over time, register, left hand in the video |
| `rhythm.py` | early / late notes, tempo | rests, leaps, fast notes, rushing/dragging neighbours, bow change lagging the onset |
| `expression.py` | flat dynamics, missing hairpins, vibrato (absent / rate / width / uneven), staccato and legato marks | bow speed (constant vs not used for loudness), stroke length, vibrato measurements |
| `tone.py` | noisy, slow attack, unsteady level, too bright / dull | bow speed at that moment (pressure vs skating), bow wobble |
| `bowing.py` | speed outliers, jerky strokes, unbalanced up/down bows, bow change off the onset | stroke statistics; strokes are also labelled detache / legato / staccato |
| `posture.py` | uneven shoulders, head movement, drift | persistent tilt = setup, transient = leaning, leaps = shift compensation, steady drift = fatigue |
| `synthesize.py` | merges all of the above | `assessment["synthesis"]`: events with ids, ranked causes, per-bar summary, cross-skill links, scores |

**The report** (`python src/scripts/serve.py`, or `analyze.py` which writes `report.html` with the media beside it) is a
player: the recording plays with a skeleton of the estimated pose beside it (front and side view; VioPose gives 3D joints
without a camera calibration, so the pose cannot be drawn over the picture itself), and live charts trace out as it plays:
pitch against the written notes, loudness, timing error per note, bow position and speed, shoulder tilt, elbow angles and
left-hand position, each shading the problems of its own skills. Below are a score per skill, a timeline of problem areas,
one button per bar coloured by its worst problem, and a sortable event table (click a row to jump there). Clicking
anywhere on the timeline seeks. The score viewer (written vs played notation) is folded away at the bottom.
`analyze.py` also writes `assessment.json`, the full synthesized event report.

**Neural models.** No public pretrained violin tone/expression grader exists (Giraldo et al. 2019 trained on unreleased
TELMI recordings; Dalmazzo & Ramirez 2019 needs wearable sensors; VioTech 2025 builds on MERT with its own annotations).
The optional `neural="clap"` / `--neural clap` option runs the pretrained LAION CLAP audio-text model
(`pip install -e ".[neural]"`, ~600 MB download) on 6 s windows of the recording against descriptions such as "a warm, smooth
violin tone" vs "a harsh, scratchy violin tone". Its probabilities become extra events, explained by the measured features
inside each window, never by the network alone. It is zero-shot and coarse: treat it as a second opinion. A backend that
returns `p_first(windows, sr, prompts)` can be swapped in (`assessment/neural.py`); trained heads (e.g. on a frozen MERT
encoder with the VioTech labels) would plug in the same way.

## Fusion encoders: learning what the analysis misses

`maestro/fusion/` holds three autoencoders, one per modality (`audio_encoder.py`, `video_encoder.py`, `score_encoder.py`) built
on one idea (`residual_autoencoder.py`): the **decoder also receives the analysis features** (pitch, loudness, bow position,
key, ... everything the rule-based modules already measure), so the small latent `z` (16 numbers per window) is pushed to hold
only what they do not explain. A KL term and a cross-covariance penalty between `z` and the analysis features keep it from
copying them; `leakage_r2` measures how much can still be read back (a test checks that an ordinary autoencoder keeps the known
factors in `z` and this one moves them out while keeping the hidden ones).

| model | x (encoded) | side information (given to the decoder) | z should hold |
|---|---|---|---|
| audio | 1 s log-mel patch | pitch, voicing, loudness, centroid, onsets per frame | timbre, bow noise and texture beyond those |
| video | 2 s of 3D pose, body-size and position independent | bow position/speed, hand distance, elbow angles, shoulder tilt | posture, wrist/finger style, carriage of the bow arm |
| score | 16 notes as tokens | pitch-class profile, range, density, intervals, key, metre | note order, contour, motifs, rhythmic pattern |

Training needs no labels (`pip install -e ".[training]"`): `python src/training/train_fusion.py score --music21` uses the
scores bundled with music21; audio takes any folder of violin recordings or videos, video takes `pose.json` files or
(frames, 18, 3) motion-capture arrays. `src/training/fusion_data.py` lists suggested public data and licences (MusicNet,
URMP, VioDat, SyncViolinist, PDMX, the Violin MIDI dataset ...). Checkpoints go to `models/fusion/`; when they exist
`analyze()` stores `perf.embeddings` (one `z` per window on the performance's clock, plus a reconstruction error that is high where
the playing is unlike the training data even given its pitch and loudness). Combining the three streams (`fusion.py`) and
prediction heads are still to do.

## Status

Implemented: audio extraction, selectable pitch/transcription backends, score conversion, score-audio alignment, pitch-deviation and timing assessment, and score visualization. Movement analysis from VioPose poses (bowing, posture, visual pitch) is implemented; VioPose and the visual pitch use the audio by default (`visual_audio=False` turns that off) and the assessment events combine both; multimodal fusion and LLM feedback remain in development.
