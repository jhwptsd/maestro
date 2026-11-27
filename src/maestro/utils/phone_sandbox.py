# Run Maestro under the restrictions of an iPhone, on any computer (new file).
#
# The surest way to learn what cannot run on iOS is to take it away and see what breaks. Inside `phone_sandbox()`:
#   - importing a package that has no iOS build (PyTorch, ONNX Runtime, librosa, numba, TensorFlow, OpenCV ...) raises ImportError, as on the phone;
#   - starting another program is impossible (iOS has no fork/exec): subprocess, os.system and multiprocessing raise OSError;
#   - shutil.which finds nothing, so ffmpeg, ffprobe, Audiveris and Ollama's tools are "not installed".
# Every attempt is recorded in `sandbox.attempts` even when the code catches the error (an optional import that falls back is a feature that
# degrades on the phone, which is just what we want to know), each with the file and line that tried. The tests run the whole analysis inside it;
# `python -m maestro.utils.phone_sandbox` runs a demonstration and prints what was tried. It cannot know what Apple itself forbids (memory limits,
# background time, store rules) or whether a package that is allowed really has an iOS wheel: utils/selftest.py on a device, and src/scripts/ios_audit.py, cover those.
import contextlib
import importlib.abc
import importlib.util
import multiprocessing
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

NO_IOS_BUILD = ("torch", "torchaudio", "torchcrepe", "onnxruntime", "onnx", "librosa", "numba", "llvmlite", "tensorflow", "keras", "cv2", "transformers", "basic_pitch",
                "resampy", "soundfile", "audioread", "mediapipe", "sklearn", "pandas", "matplotlib")
HERE = str(Path(__file__).resolve())


class Sandbox:
    def __init__(self, blocked_modules=NO_IOS_BUILD):
        self.blocked, self.attempts = tuple(blocked_modules), []

    def record(self, kind, what):
        """Note an attempt with the first stack frame outside this file."""
        frame = next((f for f in reversed(traceback.extract_stack()) if f.filename != HERE and "importlib" not in f.filename), None)
        self.attempts.append({"kind": kind, "what": what, "where": f"{Path(frame.filename).name}:{frame.lineno}" if frame else "?"})

    def tried(self, kind=None):
        """The distinct things attempted (optionally of one kind: "import", "program")."""
        return sorted({a["what"] for a in self.attempts if kind in (None, a["kind"])})


class _Blocker(importlib.abc.MetaPathFinder):
    def __init__(self, sandbox):
        self.sandbox = sandbox

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.sandbox.blocked:
            self.sandbox.record("import", name.split(".")[0])
            raise ImportError(f"No module named {name.split('.')[0]!r} (not available on iOS)", name=name)
        return None


@contextlib.contextmanager
def phone_sandbox(blocked_modules=NO_IOS_BUILD):
    """`with phone_sandbox() as box:` run code as it would run on an iPhone; `box.attempts` says what it tried that a phone cannot do."""
    box = Sandbox(blocked_modules)
    saved_modules = {n: m for n, m in sys.modules.items() if n.split(".")[0] in box.blocked}
    for name in saved_modules:                                    # a package already imported here must look uninstalled too
        del sys.modules[name]
    finder = _Blocker(box)
    sys.meta_path.insert(0, finder)

    def refuse(kind):
        def fail(*args, **kwargs):
            what = args[0] if args else kwargs.get("args", "")
            box.record("program", str(what[0] if isinstance(what, (list, tuple)) and what else what)[:80])
            raise OSError(f"{kind} is not available on iOS: a phone app cannot start other programs")
        return fail

    patches = [(subprocess, name, refuse("subprocess")) for name in ("run", "Popen", "call", "check_call", "check_output")]
    patches += [(os, "system", refuse("os.system")), (multiprocessing.Process, "start", refuse("multiprocessing"))]
    patches.append((shutil, "which", lambda cmd, *a, **k: (box.record("program", f"which {cmd}"), None)[1]))
    from maestro.utils import ffmpeg
    real_find_spec = importlib.util.find_spec
    patches.append((importlib.util, "find_spec", lambda name, package=None: None if name.split(".")[0] in box.blocked else real_find_spec(name, package)))   # "is it installed?": no
    originals = [(obj, name, getattr(obj, name)) for obj, name, _ in patches] + [(ffmpeg, "_cache", dict(ffmpeg._cache))]
    for obj, name, fake in patches:
        setattr(obj, name, fake)
    ffmpeg._cache.clear()
    real_found = ffmpeg._found
    ffmpeg._found = lambda name, extra: (box.record("program", name), None)[1]        # (the search of utils/ffmpeg.py: nothing is installed)
    try:
        yield box
    finally:
        ffmpeg._found = real_found
        for obj, name, original in originals:
            setattr(obj, name, original)
        sys.meta_path.remove(finder)
        sys.modules.update(saved_modules)


if __name__ == "__main__":
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from maestro.utils import capabilities
    with phone_sandbox() as box:
        print("capabilities:", capabilities.detect())
        for module in ("torch", "librosa"):
            try:
                __import__(module)
            except ImportError as e:
                print("import", module, "->", e)
    print("attempted:", box.attempts)
