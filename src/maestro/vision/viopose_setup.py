# Install and locate VioPose (new file).
#
# `setup_viopose()` (run by setup.ps1 / setup.sh, or `python src/scripts/setup_viopose.py`) puts a working VioPose next
# to the project, in third_party/ (git-ignored):
#   1. git clone github.com/SeongJong-Yoo/VioPose            -> third_party/VioPose
#   2. a Python 3.9 environment for it (it pins TensorFlow 2.13, mediapipe, OpenCV: they cannot share Maestro's
#      environment) via conda, else uv, else a python3.9 venv; if none of those exists, `pip install uv` (uv fetches
#      Python 3.9 itself)                                     -> third_party/viopose-env
#   3. pip install -r VioPose/requirements.txt in that environment (several GB; skipped if already installed)
#   4. the model weights, by carrying out VioPose's own download.sh in Python (cd / mkdir / gdown / unzip / rm), so it
#      works on Windows too and follows upstream if the file ids change; the demo data it also fetches is skipped
#   5. third_party/viopose.json {"dir", "python"}: what `load_config()` reads, so vision/pose_estimation.py finds it.
# Every step is skipped when already done, so re-running only repairs what is missing. `load_config()` also honours the
# environment variables VIOPOSE_DIR / VIOPOSE_PYTHON, which win over the file, and falls back to the default checkout.
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/SeongJong-Yoo/VioPose"
ROOT = Path(__file__).resolve().parents[3]              # the maestro repository (editable install)
THIRD_PARTY = ROOT / "third_party"
DEFAULT_DIR = THIRD_PARTY / "VioPose"
CONFIG_PATH = THIRD_PARTY / "viopose.json"
PYTHON_VERSION = "3.9"                                   # what VioPose's pinned requirements need


class SetupError(RuntimeError):
    """A VioPose setup step cannot be completed (the message says what to install or fix)."""


def load_config():
    """Where VioPose is: {"dir": str or None, "python": str or None}. Environment variables beat third_party/viopose.json,
    which beats the default checkout (used only if it exists). `dir` is None when VioPose is not installed."""
    saved = {}
    if CONFIG_PATH.is_file():
        try:
            saved = json.loads(CONFIG_PATH.read_text())
        except ValueError:
            pass
    directory = os.environ.get("VIOPOSE_DIR") or saved.get("dir") or (str(DEFAULT_DIR) if DEFAULT_DIR.is_dir() else None)
    return {"dir": directory, "python": os.environ.get("VIOPOSE_PYTHON") or saved.get("python")}


def write_config(directory, python):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({"dir": str(directory), "python": str(python)}, indent=2))


def env_python(env_dir):
    """The interpreter inside a virtual/conda environment directory."""
    env_dir = Path(env_dir)
    return env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def environment_command(env_dir, which=shutil.which, run=subprocess.run):
    """Command that creates a Python 3.9 environment at env_dir, from whatever is installed: conda, else uv, else a
    python3.9 (or the Windows `py -3.9`) venv. Raises SetupError listing the options if none is available."""
    if which("conda"):
        return ["conda", "create", "-p", str(env_dir), f"python={PYTHON_VERSION}", "-y"]
    if which("uv"):
        return ["uv", "venv", "--seed", "--python", PYTHON_VERSION, str(env_dir)]      # --seed installs pip
    candidates = [["py", f"-{PYTHON_VERSION}"]] if sys.platform == "win32" else [[f"python{PYTHON_VERSION}"]]
    for cmd in candidates:
        if which(cmd[0]) and run(cmd + ["-c", "import sys"], capture_output=True).returncode == 0:
            return cmd + ["-m", "venv", str(env_dir)]
    raise SetupError(f"VioPose needs Python {PYTHON_VERSION}. Install Miniconda (conda), uv, or Python {PYTHON_VERSION}, "
                     "then re-run the setup.")


def parse_download_script(text):
    """VioPose's download.sh -> [(command, args)] for the commands it uses: cd, mkdir, gdown, unzip, rm."""
    actions = []
    for line in text.splitlines():
        words = shlex.split(line.strip(), comments=True)
        if words and words[0] in ("cd", "mkdir", "gdown", "unzip", "rm"):
            actions.append((words[0], [w for w in words[1:] if not w.startswith("-")]))
    return actions


def run_download_script(text, root, python, run, skip_demo=True):
    """Carry out download.sh inside `root` (the VioPose checkout), using `python -m gdown` for the Google Drive
    downloads. Commands run in the demo-data folder are skipped when skip_demo (Maestro supplies its own video)."""
    root, cwd = Path(root).resolve(), Path(root).resolve()
    demo = root / "data"
    for command, args in parse_download_script(text):
        if command == "cd":
            cwd = (cwd / args[0]).resolve()
        elif command == "mkdir":
            (cwd / args[0]).mkdir(parents=True, exist_ok=True)
        elif skip_demo and (cwd == demo or demo in cwd.parents):
            continue
        elif command == "gdown":
            run([str(python), "-m", "gdown", *args], cwd=cwd)
        elif command == "unzip":
            with zipfile.ZipFile(cwd / args[0]) as z:
                z.extractall(cwd)
        elif command == "rm":
            (cwd / args[0]).unlink(missing_ok=True)


def setup_viopose(directory=None, python=None, skip_weights=False, dry_run=False, run=None, log=print):
    """Install VioPose (see the module comment) -> the config written. `python` uses an existing environment's
    interpreter instead of creating one; dry_run only reports the steps; `run(cmd, cwd=None)` runs a command
    (tests replace it). Raises SetupError, or subprocess.CalledProcessError if a command fails."""
    directory = Path(directory or DEFAULT_DIR).resolve()
    env_dir = directory.parent / "viopose-env"
    marker = lambda name: directory.parent / f".viopose-{name}"      # "this step finished" files, outside the clone

    def execute(cmd, cwd=None):
        log("  $ " + " ".join(str(c) for c in cmd))
        if not dry_run:
            (run or (lambda c, cwd=None: subprocess.run(c, cwd=cwd, check=True)))(cmd, cwd=cwd)

    log(f"[1/5] VioPose source in {directory}")
    if (directory / "inference.py").is_file():
        log("  already cloned")
    else:
        directory.parent.mkdir(parents=True, exist_ok=True)
        execute(["git", "clone", "--depth", "1", REPO_URL, str(directory)])

    log(f"[2/5] Python {PYTHON_VERSION} environment")
    if python:
        interpreter = Path(python)
        log(f"  using {interpreter}")
    else:
        interpreter = env_python(env_dir)
        if interpreter.is_file():
            log(f"  already created: {env_dir}")
        else:
            try:
                command = environment_command(env_dir)
            except SetupError:
                log(f"  no conda, uv or Python {PYTHON_VERSION} found: installing uv, which can fetch Python {PYTHON_VERSION} itself")
                execute([sys.executable, "-m", "pip", "install", "uv"])
                command = [sys.executable, "-m", "uv", "venv", "--seed", "--python", PYTHON_VERSION, str(env_dir)]
            execute(command)

    log("[3/5] VioPose requirements (TensorFlow, mediapipe, OpenCV: several GB)")
    requirements = directory / "requirements.txt"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest() if requirements.is_file() else ""
    if digest and marker("requirements").is_file() and marker("requirements").read_text() == digest:
        log("  already installed")
    else:
        execute([str(interpreter), "-m", "pip", "install", "-r", str(requirements)], cwd=directory)
        if not dry_run:
            marker("requirements").write_text(digest)

    log("[4/5] Model weights")
    if skip_weights:
        log("  skipped (--skip-weights)")
    elif marker("weights").is_file():
        log("  already downloaded")
    else:
        script = directory / "download.sh"
        if not dry_run:
            run_download_script(script.read_text(), directory, interpreter, lambda c, cwd=None: execute(c, cwd))
            marker("weights").write_text("ok")
        else:
            for command, args in parse_download_script(script.read_text() if script.is_file() else ""):
                log(f"  (would run) {command} {' '.join(args)}")

    log("[5/5] Configuration")
    if not dry_run:
        write_config(directory, interpreter)
        problems = [p for p in (directory / "Logs" / "VioPose", directory / "Logs" / "VioPose_wo_audio") if not p.is_dir()]
        if not skip_weights and problems:
            raise SetupError(f"Setup finished but the weights are missing: {', '.join(map(str, problems))}")
    log(f"  {'(would write) ' if dry_run else ''}{CONFIG_PATH}")
    return load_config()
