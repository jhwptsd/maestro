# Install and locate VioPose (new file).
#
# `setup_viopose()` (run by setup.ps1 / setup.sh, or `python src/scripts/setup_viopose.py`) puts a working VioPose next
# to the project, in third_party/ (git-ignored):
#   1. git clone github.com/SeongJong-Yoo/VioPose            -> third_party/VioPose
#   2. a Python 3.9 environment for it (it pins TensorFlow 2.13, mediapipe, OpenCV: they cannot share Maestro's
#      environment) via conda, else uv, else a python3.9 venv; if none of those exists, `pip install uv` (uv fetches
#      Python 3.9 itself)                                     -> third_party/viopose-env
#      The interpreter is then CHECKED (it must run, be Python 3.9 and have pip); an existing environment that fails the check
#      (left over from another Python, or broken) is deleted and rebuilt, and if the first way gives something unusable uv
#      is tried, so a wrong Python is reported here and not later as a wall of failed imports.
#   3. pip install what inference.py really imports, at the versions VioPose's requirements.txt pins (several GB;
#      skipped if already installed). requirements.txt is an exported environment of ~140 pins, many Linux/CUDA-only
#      (tensorrt, triton, jaxlib 0.4.13, wurlitzer ...), so only the pins of the imported packages plus the few that
#      keep TensorFlow consistent are installed (COMPAT_PINS) and pip resolves everything else for the platform. The
#      result is accepted only if each of inference.py's imports works in the environment (checked one by one)
#   4. gdown (which download.sh needs) and the model weights, by carrying out VioPose's own download.sh in Python (cd /
#      mkdir / gdown / unzip / rm), so it works on Windows too and follows upstream if the file ids change; the demo data
#      it also fetches is skipped
#   5. third_party/viopose.json {"dir", "python"}: what `load_config()` reads, so vision/pose_estimation.py finds it.
# Every step is skipped when already done, so re-running only repairs what is missing. `load_config()` also honours the
# environment variables VIOPOSE_DIR / VIOPOSE_PYTHON, which win over the file, and falls back to the default checkout.
import ast
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
# pins in VioPose's requirements.txt that none of its code imports (checked against its sources); several have no
# wheels on Windows or other platforms and would stop the whole install
# Module -> pip distribution where they differ, and extra pins that keep TensorFlow 2.13 working (all else is resolved)
DISTRIBUTIONS = {"cv2": {"opencv-python", "opencv-contrib-python", "opencv-python-headless"}}
COMPAT_PINS = {"keras", "protobuf", "numpy", "tensorflow-estimator", "tensorflow-hub"}


# gdown has no __main__ module (`python -m gdown` fails), so its command-line entry point is called directly
GDOWN_CODE = "from gdown.cli import main; main()"


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
    """The interpreter inside a virtual/conda environment directory: venvs keep it in Scripts/ (Windows) or bin/, conda
    environments on Windows keep it at the top; whichever exists (else the venv location)."""
    env_dir = Path(env_dir)
    candidates = [env_dir / "Scripts" / "python.exe", env_dir / "python.exe"] if sys.platform == "win32" else [env_dir / "bin" / "python"]
    return next((c for c in candidates if c.is_file()), candidates[0])


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


def inspect_interpreter(interpreter, timeout=120):
    """What an interpreter is: {"version": (major, minor) or None if it cannot be run, "pip": whether `-m pip` works,
    "error": what went wrong running it (first line), or ""}."""
    try:
        probe = subprocess.run([str(interpreter), "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
                               capture_output=True, text=True, timeout=timeout)
        if probe.returncode != 0:
            lines = (probe.stderr or probe.stdout).strip().splitlines()
            return {"version": None, "pip": False, "error": f"exit {probe.returncode}" + (f": {lines[-1]}" if lines else "")}
        pip = subprocess.run([str(interpreter), "-m", "pip", "--version"], capture_output=True, timeout=timeout)
        return {"version": tuple(int(x) for x in probe.stdout.split()), "pip": pip.returncode == 0, "error": ""}
    except (OSError, subprocess.TimeoutExpired, ValueError) as e:
        return {"version": None, "pip": False, "error": f"{type(e).__name__}: {e}"}


def environment_problem(interpreter):
    """None if `interpreter` is a working Python PYTHON_VERSION with pip; else what is wrong, in words."""
    info = inspect_interpreter(interpreter)
    if info["version"] is None:
        return f"its interpreter {interpreter} cannot be run ({info.get('error') or 'unknown error'})"
    if "%d.%d" % info["version"] != PYTHON_VERSION:
        return "it is Python %d.%d" % info["version"]
    return None if info["pip"] else "pip does not work in it"


def find_uv_python(version=PYTHON_VERSION):
    """Path of the uv-managed Python `version` (after `uv python install`), or None."""
    try:
        out = subprocess.run([sys.executable, "-m", "uv", "python", "find", version], capture_output=True, text=True, timeout=120)
        path = out.stdout.strip().splitlines()[-1] if out.returncode == 0 and out.stdout.strip() else ""
        return path if path and Path(path).is_file() else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def create_environment(env_dir, execute, log, verify=True):
    """Create the Python PYTHON_VERSION environment at env_dir: the first way that is available (conda, uv, python3.9, `py
    -3.9`), then, if that fails or gives something that is not a working Python 3.9 (checked), uv installed with pip: first
    uv's own Python 3.9 with the standard `venv` module (no launcher shim), then `uv venv`. Raises SetupError if nothing
    works. With verify=False (a dry run) only the first way is shown."""
    first = None
    try:
        first = environment_command(env_dir)
    except SetupError:
        pass
    uv_ready = []

    def need_uv():                                              # pip install uv, once, only if a uv attempt is reached
        if not uv_ready:
            log(f"  installing uv, which can fetch Python {PYTHON_VERSION} itself")
            execute([sys.executable, "-m", "pip", "install", "uv"])
            uv_ready.append(True)

    def uv_managed():
        need_uv()
        execute([sys.executable, "-m", "uv", "python", "install", PYTHON_VERSION])
        found = find_uv_python() if verify else "<uv python>"
        return [found, "-m", "venv", str(env_dir)] if found else None

    def uv_venv():
        need_uv()
        return [sys.executable, "-m", "uv", "venv", "--seed", "--python", PYTHON_VERSION, str(env_dir)]

    attempts = ([lambda: first] if first else []) + [uv_managed, uv_venv]
    problem = "no way to create it worked"
    for attempt in attempts:
        try:
            command = attempt()
            if command is None:
                problem = "uv could not provide Python " + PYTHON_VERSION
                continue
            execute(command)
        except subprocess.CalledProcessError as e:
            problem = f"`{' '.join(str(c) for c in e.cmd[:4])} ...` failed (exit {e.returncode})"
        else:
            if not verify:
                return
            problem = environment_problem(env_python(env_dir))
            if problem is None:
                return
        log(f"  the environment is not usable: {problem}")
        shutil.rmtree(env_dir, ignore_errors=True)
    raise SetupError(f"Could not create a working Python {PYTHON_VERSION} environment at {env_dir} ({problem}). Install Miniconda, "
                     f"or Python {PYTHON_VERSION} from python.org, and re-run the setup.")


def core_requirements(text, imports):
    """requirements.txt text + the import statements of inference.py -> only the pin lines of those modules' packages (and
    COMPAT_PINS); pip chooses the rest."""
    wanted = set(COMPAT_PINS)
    for statement in imports:
        module = ast.parse(statement).body[0]
        module = (module.names[0].name if isinstance(module, ast.Import) else module.module).split(".")[0]
        wanted |= DISTRIBUTIONS.get(module, {module.lower().replace("_", "-")})
    lines = [line for line in text.splitlines()
             if line.split("==")[0].split("[")[0].strip().lower().replace("_", "-") in wanted]
    return "\n".join(lines) + "\n"


def runtime_imports(root, entry="inference.py"):
    """Third-party import statements that `entry` runs at module level, following the checkout's own modules (absolute and
    relative imports, packages with or without __init__.py). Imports inside functions do not count: inference.py only
    imports moviepy when it renders a visualisation video, which Maestro never asks for. -> sorted statement texts,
    e.g. `from tensorflow import keras` (checking whole statements catches broken submodules, not just missing packages)."""
    root, stdlib = Path(root), sys.stdlib_module_names

    def local(parts, base):                                    # dotted name -> file in the checkout (longest prefix)
        for n in range(len(parts), 0, -1):
            path = base.joinpath(*parts[:n])
            for candidate in (path.with_suffix(".py"), path / "__init__.py"):
                if candidate.is_file():
                    return candidate

    is_local = lambda name: (root / name).is_dir() or (root / f"{name}.py").is_file()      # the checkout's own names
    seen, todo, third = set(), [root / entry], set()          # third = import statements
    while todo:
        file = todo.pop()
        if file in seen:
            continue
        seen.add(file)
        for node in ast.parse(file.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Import):
                targets = [(a.name.split("."), root) for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = root if not node.level else file.parents[node.level - 1]
                module = node.module.split(".") if node.module else []
                targets = [(module, base)] + [(module + [a.name], base) for a in node.names]
            else:
                continue
            for parts, base in targets:
                found = local(parts, base) if parts else None
                if found:
                    todo.append(found)
                elif parts and not getattr(node, "level", 0) and parts[0] not in stdlib and not is_local(parts[0]):
                    third.add(ast.unparse(node))
    return sorted(third)


def parse_download_script(text):
    """VioPose's download.sh -> [(command, args)] for the commands it uses: cd, mkdir, gdown, unzip, rm."""
    actions = []
    for line in text.splitlines():
        words = shlex.split(line.strip(), comments=True)
        if words and words[0] in ("cd", "mkdir", "gdown", "unzip", "rm"):
            actions.append((words[0], [w for w in words[1:] if not w.startswith("-")]))
    return actions


def run_download_script(text, root, python, run, skip_demo=True):
    """Carry out download.sh inside `root` (the VioPose checkout), using gdown's command line for the Google Drive
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
            run([str(python), "-c", GDOWN_CODE, *args], cwd=cwd)
        elif command == "unzip":
            with zipfile.ZipFile(cwd / args[0]) as z:
                z.extractall(cwd)
        elif command == "rm":
            (cwd / args[0]).unlink(missing_ok=True)


def install_requirements(interpreter, requirements_file, text, execute, log):
    """pip install -r the file; if that fails, retry the pins one at a time (each with its dependencies) and keep
    what installs. -> the pins that could not be installed."""
    try:
        execute([str(interpreter), "-m", "pip", "install", "-r", str(requirements_file)])
        return []
    except subprocess.CalledProcessError:
        log("  pip could not install the whole list; retrying one pin at a time and keeping what works")
    failed = []
    for pin in (line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")):
        try:
            execute([str(interpreter), "-m", "pip", "install", pin])
        except subprocess.CalledProcessError:
            failed.append(pin)
            log(f"  could not install {pin}")
    return failed


def check_imports(interpreter, imports, failed_pins, execute):
    """Run each import statement in the VioPose environment; raise SetupError naming those that fail."""
    broken = []
    for statement in imports:
        try:
            execute([str(interpreter), "-c", statement])
        except subprocess.CalledProcessError:
            broken.append(statement)
    if broken:
        raise SetupError("The VioPose environment is incomplete: these imports fail: " + "; ".join(broken)
                         + (f" (pins that would not install: {', '.join(failed_pins)})" if failed_pins else "")
                         + ". See the output above; the environment's Python must be 3.9.")


def repair_tensorflow(interpreter, text, execute, log):
    """On Windows `tensorflow` is a stub whose real files come from `tensorflow-intel`; reinstalling or removing one
    of them (e.g. in an earlier failed setup) leaves a TensorFlow with files missing ("No module named
    tensorflow.compat"). Reinstall tensorflow-intel at the pinned version. -> True if that was attempted."""
    pin = next((l.strip() for l in text.splitlines() if l.lower().startswith("tensorflow==")), None)
    if sys.platform != "win32" or not pin:
        return False
    log("  TensorFlow looks damaged; reinstalling tensorflow-intel")
    execute([str(interpreter), "-m", "pip", "install", "--force-reinstall", "--no-deps",
             pin.replace("tensorflow==", "tensorflow-intel==")])
    return True


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
        problem = None if dry_run else environment_problem(interpreter)
        if problem:
            raise SetupError(f"--python {interpreter} cannot be used for VioPose: {problem}. It must be a working Python "
                             f"{PYTHON_VERSION} with pip.")
    else:
        interpreter = env_python(env_dir)
        exists = interpreter.is_file()
        problem = environment_problem(interpreter) if exists and not dry_run else None
        if exists and problem is None:
            log(f"  already created: {env_dir}")
        else:
            if exists:                                          # left over from an earlier run on another Python, or broken
                log(f"  the existing environment {env_dir} is not usable ({problem}); rebuilding it")
                shutil.rmtree(env_dir, ignore_errors=True)
                marker("requirements").unlink(missing_ok=True)   # nothing is installed in the new one
            create_environment(env_dir, execute, log, verify=not dry_run)
            interpreter = env_python(env_dir)                   # where it ended up (conda keeps python.exe at the top)

    log("[3/5] VioPose requirements (TensorFlow, mediapipe, OpenCV: several GB)")
    original = directory / "requirements.txt"
    modules = runtime_imports(directory) if (directory / "inference.py").is_file() else []
    wanted = core_requirements(original.read_text(), modules) if original.is_file() else ""
    digest = hashlib.sha256((wanted + "\n".join(modules)).encode()).hexdigest()   # re-check when the imports change
    if wanted and marker("requirements").is_file() and marker("requirements").read_text() == digest:
        log("  already installed")
    else:
        filtered = directory.parent / "viopose-requirements.txt"
        if not dry_run:
            filtered.parent.mkdir(parents=True, exist_ok=True)
            filtered.write_text(wanted)
        failed = install_requirements(interpreter, filtered, wanted, execute, log)
        if not dry_run:
            try:
                check_imports(interpreter, modules, failed, execute)
            except SetupError as broken:
                try:
                    repaired = repair_tensorflow(interpreter, wanted, execute, log)
                except subprocess.CalledProcessError:           # the repair itself failed: the first error is the useful one
                    repaired = False
                if not repaired:
                    raise broken
                check_imports(interpreter, modules, failed, execute)
            marker("requirements").write_text(digest)

    log("[4/5] Model weights")
    if skip_weights:
        log("  skipped (--skip-weights)")
    elif marker("weights").is_file():
        log("  already downloaded")
    else:
        script = directory / "download.sh"
        if not dry_run:
            # download.sh calls gdown, which inference.py never imports, so the requirements step above does not install it
            gdown = next((l.strip() for l in original.read_text().splitlines() if l.lower().startswith("gdown")), "gdown") \
                if original.is_file() else "gdown"
            execute([str(interpreter), "-m", "pip", "install", gdown])
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
