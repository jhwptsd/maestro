# maestro
[WIP] An app that takes a video of a violin performance alongside a score and provides personalized feedback: it aligns the
audio, video and score, measures pitch, timing, expression, tone, bowing and posture, and reports every problem as a
quantitative event (what, when, why) in a player with live charts.

Contents: [Install](#install) · [Use the app](#use-the-app) · [Train the models](#train-the-models) · [Folders](#folders) ·
[Tests](#tests) · [Status](#status)

## Install

**Prerequisites** (programs, not pip packages): Python 3.10+, `ffmpeg` (required), `git` (to fetch VioPose), Audiveris
(optional: PDF and image scores; MusicXML and MIDI scores work without it).

**One command** installs everything the app needs, including those programs where it can:

```bash
.\setup.ps1 [-SkipViopose] [-Training] [-Neural] [-NoVenv]     # Windows (ffmpeg, Audiveris, git through winget)
./setup.sh  [--skip-viopose] [--training] [--neural] [--no-venv] # Linux / macOS (ffmpeg, git through apt or brew)
```

It installs Maestro into its own virtual environment (`.venv` in the folder, or `%LOCALAPPDATA%\maestro\venv` on Windows when the
folder is inside OneDrive; `-NoVenv` / `--no-venv` skips this), with the default CREPE pitch backend and torch
(`pip install -e ".[crepe,fusion,dev]"`), and VioPose
(see [Movement analysis](#movement-analysis-viopose); a large download, skipped with `-SkipViopose`). `-Training`
adds what training needs (music21, yt-dlp) and `-Neural` adds `transformers` for the optional CLAP judgement. Install the
CUDA build of PyTorch into that environment if you want GPU acceleration. **Activate the environment in every new terminal**
before using Maestro (the script prints the command: `& ".venv\Scripts\Activate.ps1"` on Windows, `source .venv/bin/activate` elsewhere). If VioPose does not finish, everything else still works; re-run
`python src/scripts/setup_viopose.py` (safe to re-run, finished steps are skipped).

*Troubleshooting:* `The system cannot find the file specified` from `ffprobe`/`ffmpeg` (or a message "ffprobe was not found") means ffmpeg is installed but
not on the PATH of the terminal Maestro runs in: winget only updates the PATH of *new* terminals, so open a new terminal (and reload VS
Code). Maestro also looks in the usual install places and in the PATH as stored in the Windows registry, so a restart of the
server is usually enough. `pip ... WinError 2 / 5 ... Scripts\xxx.exe -> xxx.exe.deleteme` means pip tried to replace a package that is
already in your global Python (typically an older torch) and could not. The virtual environment avoids it; if you install by hand,
use one too (`python -m venv .venv`), or delete the leftover file it names and any `~orch` folder in `site-packages`.

**By hand**, pick the extras you need (`pyproject.toml` is the source of truth; `requirements.txt` lists the default set):

| extra | adds | needed for |
|---|---|---|
| *(core)* | numpy, scipy | everything (the YIN pitch backend needs nothing more) |
| `crepe` | torch, torchcrepe, librosa | the default pitch backend |
| `pyin`, `basic-pitch`, `audio` | librosa / basic-pitch / all four backends | the other pitch backends |
| `fusion` | torch | running the trained fusion encoders (`perf.embeddings`) |
| `training` | torch, music21, yt-dlp | training the models and installing their datasets |
| `neural` | torch, transformers | the optional CLAP tone/expression judgement |
| `dev` | pytest | running the tests |

```bash
pip install -e ".[crepe,fusion,dev]"          # what the setup scripts install
pip install -e ".[audio,training,neural,dev]" # everything except VioPose (which has its own environment)
```

## Use the app

Normal use needs no training: the rule-based analysis works as soon as the package is installed. Trained fusion encoders
and the CLAP judgement are optional extras on top.

**Web GUI**: `python src/scripts/serve.py [PORT]` (or `maestro-gui`), open http://127.0.0.1:8000, upload a recording (video or
audio) and the score. The report is a player: the recording plays with a skeleton of the estimated pose beside it (front and
side view; VioPose gives 3D joints without a camera calibration, so the pose cannot be drawn over the picture itself), and live
charts trace out as it plays: pitch against the written notes, loudness, timing error per note, bow position and speed,
shoulder tilt, elbow angles and left-hand position, each shading the problems of its own skills. Below are a score per
skill, a timeline of problem areas, one button per bar coloured by its worst problem, and a sortable event table (click a
row to jump there; clicking the timeline seeks). The score viewer (written vs played notation) is folded away at the bottom.

**Command line**: `python src/scripts/analyze.py performance.mp4 score.musicxml out/ [--pitch-backend crepe|yin|pyin|basic_pitch]
[--neural clap]` writes `out/report.html` (open it next to the media it saved there), `out/assessment.json` (all events) and the
intermediate files (`pose.json`, converted video, ...).

**Python**:

```python
from maestro import analyze
perf = analyze("performance.mp4", "score.musicxml", out_dir="out/")
perf.assessment["synthesis"]      # every event, ranked causes, per-bar summary, scores (plain JSON)
perf.embeddings                   # learned per-window descriptions, if trained models exist (see Train the models)
```

`analyze(video_path, score_path, out_dir=None, denoise=True, pitch_backend="crepe", pitch_options=None, pose_file=None,
neural=None, visual_audio=True, embeddings=None)`:
`denoise` cleans audio, video and score first (vibrato and bow noise are kept); `pose_file` (a saved `pose.json`) skips VioPose;
`neural="clap"` adds the CLAP judgement; `visual_audio=False` keeps the visual models audio-free; `embeddings` is `None`
(use trained encoders if they exist), `True` (require them) or `False`. Errors for unusable input are
`maestro.utils.errors.InputError`; a score that does not match the recording raises its subclass `MismatchError`.

### Audio pitch backends

Maestro supports interchangeable audio backends behind one `PitchTrack` interface:

| Backend | Select with | Output | Optional install |
|---|---|---|---|
| CREPE (default) | `crepe` | Frame-wise F0 and periodicity confidence | `pip install -e ".[crepe]"` |
| NumPy YIN | `yin` | Frame-wise F0 and YIN confidence | No extra dependency |
| Librosa pYIN | `pyin` | Probabilistic F0 and voiced probability with temporal decoding | `pip install -e ".[pyin]"` |
| Spotify Basic Pitch | `basic_pitch` | Polyphonic note events; dominant-note projection for frame plots | `pip install -e ".[basic-pitch]"` |

CREPE automatically selects CUDA when available; pass `pitch_options={"device": "cpu"}` to force CPU inference. The CREPE model
exposed by torchcrepe has a maximum F0 of about 2006 Hz, which excludes some of the violin's highest fundamentals: use `pyin` or
`yin` with a higher `fmax` for those passages, e.g. `analyze(..., pitch_backend="pyin", pitch_options={"fmax": 4200})`. Basic
Pitch preserves overlapping notes such as double stops, but its note pitches are semitone-quantized, so do not use it alone
for cents-level intonation. `python src/scripts/bench_pitch.py` times every installed backend on your hardware. Pitch is
estimated once per performance and shared between note transcription and frame features. The lower-level interface is
`maestro.audio.pitch.estimate_pitch()`, which returns a `PitchTrack` (`times`, `f0`, `confidence`, `backend`, optional `notes`).

### Movement analysis (VioPose)

Bow movement, arm angles, posture and the pitch implied by the left hand's position come from a 3D pose that
[VioPose](https://github.com/SeongJong-Yoo/VioPose) (MIT) estimates from the video, automatically, for every analysis that has
a video; the pose is saved as `pose.json`. VioPose pins TensorFlow 2.13 (Python 3.9), so the setup installs it separately: cloned
into `third_party/` with its own Python 3.9 environment (conda, uv, or a `python3.9` venv; uv is installed if none is found),
the interpreter is verified to be a working Python 3.9 with pip (a wrong or broken existing environment is rebuilt; if conda, uv and `py -3.9` fail, uv fetches Python 3.9 itself), only the packages `inference.py` really imports (pip resolves the rest for your platform), gdown, and its weights (Google
Drive). `python src/scripts/setup_viopose.py` does just that step and takes `--dry-run`, `--skip-weights`, `--python PATH` (an
existing environment) and `--dir`. The result is recorded in `third_party/viopose.json`; `VIOPOSE_DIR` and `VIOPOSE_PYTHON`
override it, `VIOPOSE_AUDIO=0` runs VioPose without the recording's audio (by default its audio model is used, falling back to
the video-only weights). Without VioPose the audio analysis works as before and the report omits the movement parts. Details
and the visual pitch method are in `src/docs/README.md`.

### Assessment events and root causes

`maestro/assessment/` turns the aligned audio, video and score into **events**: one JSON object per problem, in a single
shared form (`assessment/events.py`): `module`, `category`, `start`/`end` (s), `measure`/`beat`, `severity` (0-1), measured
`value` vs `expected` (+ `unit`), and `causes`: cause slugs with a confidence and the measurements they rest on. The
events are purely quantitative (what, when, why as numbers and slugs, no prose): putting them into words is the job of the
feedback stage. Thresholds scale with the passage (the written gap to the neighbouring notes), timing is judged against the
local tempo so rubato is not an error, and tone and bowing limits are relative to the player's own norms.

| file | events | causes come from |
|---|---|---|
| `intonation.py` | sharp / flat / wrong note | global tuning offset, shift overshoot (the leap into the note), drift over time, register, left hand in the video |
| `rhythm.py` | early / late notes, tempo, tempo changes | rests, leaps, fast notes, bow change lagging the onset, possible misalignment in fast runs |
| `expression.py` | flat dynamics, missing hairpins, vibrato (absent / rate / width / uneven), staccato and legato marks | bow speed (constant vs not used for loudness), stroke length, vibrato measurements |
| `tone.py` | noisy, slow attack, unsteady level, too bright / dull | bow speed at that moment (pressure vs skating), bow wobble |
| `bowing.py` | speed outliers, jerky strokes, unbalanced up/down bows, bow change off the onset | stroke statistics; strokes are also labelled detache / legato / staccato |
| `posture.py` | uneven shoulders, head movement, drift | persistent tilt = setup, transient = leaning, leaps = shift compensation, steady drift = fatigue |
| `synthesize.py` | merges all of the above | `assessment["synthesis"]`: events with ids, ranked causes, per-bar summary, cross-skill links, scores |

**Optional neural judgement.** No public pretrained violin tone/expression grader exists. `neural="clap"` / `--neural clap`
(`pip install -e ".[neural]"`, ~600 MB download on first use) runs the pretrained LAION CLAP audio-text model on 6 s windows against
descriptions such as "a warm, smooth violin tone" vs "a harsh, scratchy violin tone". Its probabilities become extra events whose
causes come from the measured features inside each window. It is zero-shot and coarse: a second opinion. A backend that returns
`p_first(windows, sr, prompts)` can be swapped in (`assessment/neural.py`).

## Train the models

Training is optional and separate from normal use. The models that learn from data are the three **fusion encoders**
(`maestro/fusion/`), residual autoencoders for audio, video and score. The **decoder also receives the analysis features**
(pitch, loudness, bow position, key, ... everything the rule-based modules already measure), so the small latent `z`
(16 numbers per window) is pushed to hold only what they do not explain. A KL term and a cross-covariance penalty between `z`
and the analysis features keep it from copying them; `leakage_r2` measures how much can still be read back.

| model | x (encoded) | side information (given to the decoder) | z should hold |
|---|---|---|---|
| audio | 1 s log-mel patch | pitch, voicing, loudness, centroid, onsets per frame | timbre, bow noise and texture beyond those |
| video | 2 s of 3D pose, body-size and position independent | bow position/speed, hand distance, elbow angles, shoulder tilt | posture, wrist/finger style, carriage of the bow arm |
| score | 16 notes as tokens | pitch-class profile, range, density, intervals, key, metre | note order, contour, motifs, rhythmic pattern |

Everything is unsupervised (no labels). Where things live: **data in `data/`, trained models in `models/`**.

**1. Install the training extras**: `./setup.sh --training` / `.\setup.ps1 -Training`, or `pip install -e ".[training]"`. The
video dataset also needs ffmpeg and a working VioPose (the normal setup).

**2. Install a dataset** (`python src/training/install_data.py list` shows what is installed):

| dataset | for | command |
|---|---|---|
| `musicnet` | audio | `python src/training/install_data.py audio`: the violin recordings of MusicNet (Zenodo, CC BY 4.0). Downloads the large archive (resumable, checksum-verified), keeps only violin `.wav` files, honours MusicNet's own test folder |
| `youtube-cc` | video (and audio) | `python src/training/install_data.py video [--count 60] [--query "..."]`: Creative Commons violin videos found with yt-dlp, each run through VioPose to a `pose.json`; uploader, title and URL kept for attribution |
| the music21 corpus | score | nothing to install (`pip install music21`): `train_fusion.py score --music21` |
| VioDat, SyncViolinist, MOSA | video, audio | need a request form and are research-only: request access, convert to wav or `(frames, 18, 3)` mm `.npy` arrays, then `install_data.py ingest NAME audio\|video FOLDER` |

Data is partitioned per recording (hash of its id: 80 % train, 10 % val, 10 % test; MusicNet's own test folder is the test
split) into `data/<audio|video>/<dataset>/{train,val,test}/` with a `manifest.json` (source, licence, attribution); downloads
sit in `data/downloads/` (safe to delete) and analysed audio windows are cached in `data/cache/`. `MAESTRO_DATA` moves the folder.

**3. Train**:

```bash
python src/training/train_fusion.py audio --dataset musicnet --epochs 30
python src/training/train_fusion.py video --dataset youtube-cc --epochs 40
python src/training/train_fusion.py score --music21 --epochs 40
python src/training/train_fusion.py audio recordings/ --cache cache/      # or any folder of your own files
```

Options: `--z-dim`, `--beta` (KL weight), `--leak` (cross-covariance weight), `--batch`, `--lr`, `--val`, `--limit`, `--device`,
`--out`. Each epoch prints the validation reconstruction loss and `leakage` (lower = `z` repeats less of what the analysis knows).
The best epoch is saved to `models/fusion/<audio|video|score>.pt` (override the folder with `--out` or `MAESTRO_MODELS`), with
the dataset, validation scores and, when a dataset has a test split, the test scores scored once at the end. `src/training/fusion_data.py` has
the loaders and the list of suggested public datasets; `src/training/datasets.py` the installers.

**4. Use the trained models**: nothing to configure. When checkpoints exist in `models/fusion/` (and torch is installed),
`analyze()` stores `perf.embeddings`: per modality, one `z` per window on the performance's clock plus a reconstruction
error that is high where the playing is unlike the training data even given its pitch and loudness. Combining the three streams
(`fusion/fusion.py`) and prediction heads are still to do.

## Folders

| folder | contents |
|---|---|
| `src/maestro/` | the package: `preprocessing`, `audio`, `score`, `alignment`, `vision`, `assessment`, `fusion`, `gui`, `utils` (see the header comment of each file) |
| `src/scripts/` | command-line tools: `analyze.py`, `serve.py`, `setup_viopose.py`, `convert_viopose.py`, `bench_pitch.py`, `tune_denoise.py` |
| `src/training/` | `install_data.py`, `train_fusion.py`, and their data code |
| `src/tests/` | the tests (`pytest`) |
| `src/docs/` | longer documentation (visual analysis, VioPose, visual pitch) |
| `data/` | datasets, partitions, caches (git-ignored; see `data/README.md`) |
| `models/` | trained models (git-ignored; see `models/README.md`) |
| `third_party/` | VioPose and its environment (git-ignored) |

## Tests

`pytest` (or `PYTHONPATH=src python -m unittest discover -s src/tests`) runs everything without network access or downloads:
datasets are served by a local stand-in server and VioPose by a fake. The fusion and dataset tests need torch (`.[fusion]`) and
are skipped without it; one test drives the report page in headless Chromium and is skipped without `node` + `playwright`.

## Status

Implemented: audio extraction, selectable pitch/transcription backends, score conversion, score-audio alignment, assessment of
pitch, timing, expression, tone, bowing and posture as quantitative events with causes, denoising, the player GUI, VioPose
movement analysis, and the three trained-from-data fusion encoders with their dataset installers and training code.
Multimodal fusion (combining the encoder outputs), prediction heads and LLM feedback remain in development.
