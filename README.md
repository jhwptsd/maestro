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
.\setup.ps1 [-SkipViopose] [-Neural] [-NoVenv] [-NoGpu]     # Windows (ffmpeg, Audiveris, git through winget)
./setup.sh  [--skip-viopose] [--neural] [--no-venv] # Linux / macOS (ffmpeg, git through apt or brew)
```

It installs Maestro into its own virtual environment (`.venv` in the folder, or `%LOCALAPPDATA%\maestro\venv` on Windows when the
folder is inside OneDrive; `-NoVenv` / `--no-venv` skips this), with the default CREPE pitch backend and torch
(`pip install -e ".[crepe,dev]"`), and VioPose
(see [Movement analysis](#movement-analysis-viopose); a large download, skipped with `-SkipViopose`). `-Neural`
adds `transformers` for the optional CLAP judgement. Install the
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
| `crepe` | torch, torchcrepe, librosa, onnx, onnxruntime | the default pitch backend (ONNX Runtime runs its network about twice as fast as PyTorch on the CPU) |
| `directml` | onnxruntime-directml | Windows: runs that network on any DirectX 12 GPU (AMD, Intel, NVIDIA). Replaces `onnxruntime` (`pip uninstall onnxruntime` first; `setup.ps1` does this for you when it finds a GPU) |
| `pyin`, `basic-pitch`, `audio` | librosa / basic-pitch / all four backends | the other pitch backends |
| `neural` | torch, transformers | the optional CLAP tone/expression judgement |
| `dev` | pytest | running the tests |

```bash
pip install -e ".[crepe,dev]"                 # what the setup scripts install
pip install -e ".[audio,neural,dev]"          # everything except VioPose (which has its own environment)
```

## Use the app

The rule-based analysis works as soon as the package is installed; the CLAP judgement is an optional extra on top.

**Web GUI**: `python src/scripts/serve.py [PORT]` (or `maestro-gui`), open http://127.0.0.1:8000, upload a recording (video or
audio) and the score. A progress bar follows the upload and then the analysis stage by stage (converting the video, finding the pitch,
waiting for the body-movement analysis, which is the slowest step, ...; how far a stage has got is not known, so the bar eases
towards the end of the current stage). The report is a player: the recording plays with a skeleton of the estimated pose beside it (front and
side view; VioPose gives 3D joints without a camera calibration, so the pose cannot be drawn over the picture itself), and live
charts trace out as it plays: pitch against the written notes, loudness, timing error per note, bow position and speed,
shoulder tilt, elbow angles and left-hand position, each shading the problems of its own skills. Below are a score per
skill, a timeline of problem areas, one button per bar coloured by its worst problem, and a sortable event table (click a
row to jump there; clicking the timeline seeks). The score viewer (written vs played notation) is folded away at the bottom.

**Command line**: `python src/scripts/analyze.py performance.mp4 score.musicxml out/ [--pitch-backend crepe|yin|pyin|basic_pitch]
[--neural clap] [--no-feedback] [--model NAME]` writes `out/report.html` (open it next to the media it saved there), `out/assessment.json` (all events) and the
intermediate files (`pose.json`, converted video, ...).

**Python**:

```python
from maestro import analyze
perf = analyze("performance.mp4", "score.musicxml", out_dir="out/")
perf.assessment["synthesis"]      # every event, ranked causes, per-bar summary, scores (plain JSON)
```

`analyze(video_path, score_path, out_dir=None, denoise=True, pitch_backend="crepe", pitch_options=None, pose_file=None,
neural=None, visual_audio=True, progress=None)`:
`denoise` cleans audio, video and score first (vibrato and bow noise are kept); `pose_file` (a saved `pose.json`) skips VioPose;
`neural="clap"` adds the CLAP judgement; `visual_audio=False` keeps the visual models audio-free; `progress` is called with the name of each stage as it starts. Errors for unusable input are
`maestro.utils.errors.InputError`; a score that does not match the recording raises its subclass `MismatchError`.

### Speed and GPUs

`analyze()` prints nothing, but `perf.timings` (and `analyze.py`, and the server's log) report the seconds per stage, so you can see
where the time goes. What runs where (`maestro/utils/device.py`; `MAESTRO_DEVICE=cpu|cuda:0|mps|xpu|onnx` overrides the automatic choice):

| work | runs on | notes |
|---|---|---|
| CREPE pitch network (the main neural cost of the audio analysis) | a PyTorch GPU (NVIDIA CUDA, AMD ROCm on Linux, Intel XPU, Apple MPS), else ONNX Runtime on a GPU (**DirectML: any AMD, Intel or NVIDIA GPU on Windows**), else ONNX Runtime on the CPU, else PyTorch on the CPU | with a GPU the larger, more accurate CREPE model is used; on the CPU the tiny one. The network is exported once to `models/onnx/` (about 10 s, 90 MB for the full model); a GPU path that fails falls back to the CPU with a warning. `audio_features["pitch_runtime"]` says which ran |
| CLAP (optional) | the best PyTorch accelerator, else the CPU | PyTorch cannot use an AMD or Intel GPU on Windows (ROCm PyTorch is Linux-only for most cards), so on such a machine it stays on the CPU |
| pose (VioPose: TensorFlow 2.13 and MediaPipe) | the CPU | TensorFlow has had no GPU support on native Windows since 2.10, and MediaPipe's pose model runs on the CPU here. It is the slowest step (minutes), so it runs **in the background while the audio is analysed** and is cancelled if the analysis fails |
| ffmpeg conversion, alignment, assessment | the CPU | |

A faster CPU speeds up the pose and conversion steps; a GPU speeds up the pitch network. `MAESTRO_DEVICE=cpu` is the escape hatch if a GPU
driver misbehaves.

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

### Written feedback and chat (a small local model)

The analysis gives numbers; a small language model running **on your own computer** turns them into feedback a student can use: a short
summary, priorities (the bars involved, what is probably going wrong, a concrete practice suggestion each) and what is going well, and then
answers follow-up questions in a chat (`maestro/feedback/`). There is no account, key or internet use: it runs through
[Ollama](https://ollama.com/download) (free; or `setup.ps1 -LocalModel` / `./setup.sh --local-model`). Start Ollama and download a model once
(a button on the start page with a progress bar, or `ollama pull llama3.2:3b`); a model you already have (e.g. `qwen3.5:4b`) is used as it is.

- **Models**: the ones offered are small 4-bit models of 1 to 4 billion parameters (Llama 3.2 1B/3B, Qwen 3.5 4B, Qwen 2.5 1.5B, Gemma 2 2B,
  Phi 3.5 mini), light enough for a laptop and for a recent phone. For them Maestro sends a compact summary (the 10 worst events, 5 worst bars),
  asks for JSON, switches "thinking" off (thinking models would otherwise return nothing) and sets an 8192-token context (Ollama's default
  would silently cut the prompt). Expect plainer feedback than from a large hosted model. `OLLAMA_HOST` points Maestro at an Ollama on another
  machine. Maestro itself is a computer app; it does not run on an iPhone.
- **Feedback is written automatically** when a report is opened (one call per analysis; the answer is kept, and a failure shows a readable
  message with a retry button). **Chat**: below the feedback, ask anything about your playing ("why is my intonation sharp in bar 12?"); the
  model sees the same digest and the feedback, plus the conversation (up to 20 questions per analysis; "New conversation" clears it). From the
  command line `analyze.py` also writes `feedback.json` when Ollama is running (`--model M` picks a model, `--no-feedback` skips it).
- **What the model sees**: a digest of numbers and what you type in the chat, never the recording, the video, the score or file names; the
  report page shows the digest ("What is sent to the model"). It is told to use only those measurements, and every bar it cites in the
  feedback is checked against the bars the analysis flagged: an invented bar is removed and a warning shown. Its text is displayed as
  plain text only.
- The server only accepts these requests from its own page (Host and Origin are checked and bodies must be JSON), so another website
  cannot drive them.

`feedback/ollama.py` is the client, `feedback/prompts.py` holds the system prompt and the digest, `feedback/generator.py` the call, parsing
and bar check, `feedback/chat.py` the conversation, `feedback/settings.py` the saved model choice (`llm.json` in the user config folder;
`MAESTRO_CONFIG` moves it). Offline evaluation against teachers' comments (`feedback/evidence.py`, `feedback/validate.py`) is still an outline.

### Professional reference (`maestro/reference/`)

Compare your playing with a professional's, without treating the professional as an answer key. Give a recording (an audio or video file, or a
YouTube link) in the start page ("Optional: compare with a professional recording": a link or a file; the progress bar then has an extra step, and the report gets a
"Compared with a professional" card with the kinds, the soloist's choices and what was left out), with `analyze.py ... --reference FILE_OR_URL`, or `analyze(..., reference=...)`:

1. **Load** (`load_video.py`): any file ffmpeg reads, or a YouTube video (needs `pip install yt-dlp`; at most 720 p, one video, no playlists;
   recordings over an hour are refused). Respect the copyright and terms of the source: the download stays in the analysis folder.
2. **Clip** (`process.py`): the recording is aligned with the score and cut, audio and video, to **the bars you played** (plus a short lead-in and tail),
   so only that part is processed. A recording that does not contain those bars is refused with a message (the rest of your analysis is unaffected).
3. **Isolate** (`isolate.py`): score-informed harmonic masking keeps the soloist's pitch and its partials (the score and the alignment say what the
   soloist plays when) and drops accompaniment and noise; `--separate` runs [Demucs](https://github.com/facebookresearch/demucs) first (`pip install demucs`)
   to remove drums, bass and vocals. Where the soloist cannot be heard although the score says they play, the stretch is a **gap**, not a measurement.
   Limit: an orchestra note on the soloist's own partials cannot be separated, and the first pass assumes the soloist is the loudest voice.
4. **Video gaps** (`gaps.py`): the pose of the clip comes from VioPose as for you (or `--reference-pose pose.json`). Camera cuts (ffmpeg's scene detector),
   lost pose and a clearly different framing (a close-up) become gaps; movement is measured only between them and never interpolated across one.
5. **Baseline** (`baseline.py`): the soloist is measured exactly as you are (per note: cents from equal temperament, duration, articulation, loudness,
   attack, vibrato; per bar: tempo against the marking, loudness, bow changes and speed). Notes and bars inside a gap carry no numbers.
6. **Subjective choices are marked.** Every category has a *nature*: `objective` (pitch, attack: the target is the same for everyone) or `interpretive`
   (tempo, rubato, dynamics, articulation, vibrato, bowing, tone colour). The soloist's visible decisions are listed in `choices` with their numbers
   (`tempo_choice`, `local_tempo_shift`, `dynamic_shaping`, `vibrato_use`, `played_short`, `bowing_choice`, `expressive_intonation` for a note more than
   12 cents from equal temperament).
7. **The comparison is differentials, as events, like everything else** (`assessment/reference.py`): the professional is the baseline, and each way your take
   differs from it is an event of the module `reference` in the common format (`tempo_vs_reference`, `dynamics_vs_reference`, ... with bar, beat, unit and, as
   its `value`, the differential: you minus the soloist; `expected` is 0). No averages or tallies are reported. Its cause is always
   `soloist_interpretive_choice` (evidence: the differential, its relative size, both values and, when it applies, the soloist's own `soloist_choice`).
   The differences are emphasised (severity 0.25 to 0.55, their own slots in the digest, the card at the top of the report) but stay below a clear note
   error, the feedback model is told to name them yet put wrong notes, intonation, rhythm and tone first, and the module has no score and is left out of
   the problem bars and leading causes. Accuracy (pitch, attack) is judged by the intonation and tone modules. The events go through the synthesis into
   the player's event list and into the digest that the feedback model reads, exactly like the other modules' (it gets its own slots among the events,
   plus the soloist's choices and the gaps), and it is told never to call such a difference a mistake. `reference.json` (command line) has the baseline.

Not yet: playing the reference alongside the student in the player, portamento and fingering/bowing-direction detection.

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

## Folders

| folder | contents |
|---|---|
| `src/maestro/` | the package: `preprocessing`, `audio`, `score`, `alignment`, `vision`, `assessment`, `reference`, `feedback`, `fusion` (outlined, not implemented yet), `gui`, `utils` (see the header comment of each file) |
| `src/scripts/` | command-line tools: `analyze.py`, `serve.py`, `setup_viopose.py`, `convert_viopose.py`, `bench_pitch.py`, `tune_denoise.py` |
| `src/training/` | (reserved for training scripts) |
| `src/tests/` | the tests (`pytest`) |
| `src/docs/` | longer documentation (visual analysis, VioPose, visual pitch) |
| `data/` | training data (git-ignored; see `data/README.md`) |
| `models/` | trained and exported models (git-ignored; see `models/README.md`) |
| `third_party/` | VioPose and its environment (git-ignored) |

## Tests

`pytest` (or `PYTHONPATH=src python -m unittest discover -s src/tests`) runs everything without network access or downloads
(VioPose is replaced by a fake). One test drives the report page in headless Chromium and is skipped without `node` + `playwright`.

## Status

Implemented: audio extraction, selectable pitch/transcription backends, score conversion, score-audio alignment, assessment of
pitch, timing, expression, tone, bowing and posture as quantitative events with causes, denoising, the player GUI, VioPose
movement analysis, GPU-aware pitch estimation, written feedback and chat from a local model, and the professional reference comparison. The `fusion/` folder (learned
encoders, multimodal fusion, prediction heads) and the offline evaluation of the feedback against teachers' comments are outlined but not
implemented yet. The professional reference (`reference/`) works from the GUI, the command line and `analyze()`.
