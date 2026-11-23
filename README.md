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
[--neural clap] [--feedback NAME] [--no-feedback] [--model NAME]` writes `out/report.html` (open it next to the media it saved there), `out/assessment.json` (all events) and the
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

### Written feedback and chat (local model, OpenRouter or Groq)

The analysis gives numbers; an AI model turns them into feedback a student can use: a short summary, up to three priorities (the bars
involved, what is probably going wrong, one concrete practice suggestion each) and what is going well, and then answers follow-up
questions in a chat (`maestro/feedback/`). It is optional, uses **your own** connection, and runs on services with **free tiers**:

| way to connect | service | cost |
|---|---|---|
| **Sign in with OpenRouter** (OAuth, no key to copy) | OpenRouter: you log in there and Maestro receives an API key for your own account (`feedback/oauth.py`) | free models (`:free`) |
| **Local model** (Ollama) | a small model on your own computer: no account, no key, works offline, nothing is sent anywhere | free |
| paste an **API key** | [OpenRouter](https://openrouter.ai/keys) or [Groq](https://console.groq.com/keys) | both have free tiers with rate limits |

(Claude, ChatGPT and Gemini were tried and dropped: their app subscriptions cannot be linked to other apps, their APIs are billed by use, and a free Gemini key
was often "overloaded".) Free models are shared and sometimes busy or rate limited, so a busy model is
retried once and then replaced by up to two other models (a free model only ever by another free one, so a fallback never spends money);
the model that worked is remembered and the feedback notes the swap.

- **Local model.** Install [Ollama](https://ollama.com/download) (or `setup.ps1 -LocalModel` / `./setup.sh --local-model`), start it, choose
  "Local model" on the start page, press **Connect to Ollama**, then pick a model and **Download and use** (real progress bar). The models
  offered are small, 4-bit models of 1 to 4 billion parameters (Llama 3.2 1B/3B, Qwen 2.5 1.5B, Gemma 2 2B, Phi 3.5 mini): the same ones run
  on a recent phone. To suit them Maestro sends a compact summary (the 10 worst events, 5 worst bars), asks for JSON mode and sets an 8192-token
  context (Ollama's default would silently cut the prompt). Expect plainer feedback than from the big hosted models; the bar numbers are still
  checked against the analysis. Note that Maestro itself is a computer app: it does not run on an iPhone (that would need it packaged as a
  separate app); `OLLAMA_HOST` points it at an Ollama on another machine.
- **Connect** on the start page of the GUI ("Written feedback from an AI"): sign in or paste the key (it is checked by listing the models it
  can use, and only then saved), pick a model, connect both if you like and choose which one to use. Keys are kept in your user config
  folder (`%APPDATA%\maestro\llm.json`, `~/.config/maestro/llm.json`; readable by you only, `MAESTRO_CONFIG` moves it) and are only ever
  shown masked. `OPENROUTER_API_KEY` and `GROQ_API_KEY` work too. The default model is chosen from what your key can actually use.
- **Feedback is written automatically** when a report is opened and a provider is connected (one call per analysis; the answer is kept,
  and a failure shows the provider's message with a retry button). **Chat**: below the feedback, ask anything about your playing ("why is
  my intonation sharp in bar 12?"); the model sees the same digest and the feedback, plus the conversation (up to 20 questions per
  analysis; "New conversation" clears it). From the command line `analyze.py` also writes `feedback.json` when a provider is connected
  (`--feedback NAME [--model M]` picks one, `--no-feedback` skips it).
- **What is sent**: a digest of numbers (scores, the 30 most severe problems with bar, time, size and likely causes, the bars with most
  problems) and what you type in the chat, never the recording, the video, the score or file names; the report page shows the digest
  ("What is sent to the AI provider"). The model is told to use only those measurements, and every bar it cites in the feedback is checked
  against the bars the analysis flagged: an invented bar is removed and a warning shown. Its text is displayed as plain text only. Free
  tiers may use prompts to improve the provider's products; the digest is only numbers.
- The server only accepts these requests from its own page (Host and Origin are checked and bodies must be JSON), so another
  website cannot spend your quota or read or change your keys.

The services are called over plain HTTPS with the OpenAI chat-completions format (`feedback/providers.py`, no vendor SDK);
`feedback/prompts.py` holds the system prompt and the digest, `feedback/generator.py` the call, parsing and bar check,
`feedback/chat.py` the conversation. Offline evaluation against teachers' comments (`feedback/evidence.py`, `feedback/validate.py`) is
still an outline.

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
| `src/maestro/` | the package: `preprocessing`, `audio`, `score`, `alignment`, `vision`, `assessment`, `feedback`, `fusion` (outlined, not implemented yet), `gui`, `utils` (see the header comment of each file) |
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
movement analysis, GPU-aware pitch estimation and written feedback and chat from OpenRouter or Groq (free tiers). The `fusion/` folder (learned
encoders, multimodal fusion, prediction heads), the professional reference (`reference/`) and the offline evaluation of the feedback
against teachers' comments are outlined but not implemented yet.
