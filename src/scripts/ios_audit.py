"""Which parts of Maestro cannot run on iOS, found from the code itself (new file).

    python src/scripts/ios_audit.py                 # what the code needs, by feature (offline)
    python src/scripts/ios_audit.py --online        # + ask the package indexes whether each package has an iOS wheel
    python src/scripts/ios_audit.py --online --flet # + Flet's own index (pypi.flet.dev), which is where `flet build ios` gets its wheels
    python src/scripts/ios_audit.py --json

Three questions, answered per feature (FEATURES maps source files to what the user sees):
  1. Which third-party packages does it import, and are they needed (imported at the top of a file) or optional (imported inside a function or a try)?
  2. Does it start another program (subprocess, os.system, shutil.which)? iOS apps cannot, so ffmpeg, Audiveris, VioPose and Ollama are out.
  3. With --online: does each package have a wheel for iOS? pip itself is asked, for the platform tag of an iPhone (ios_13_0_arm64_iphoneos): a pure-Python
     wheel (py3-none-any) works, a compiled one needs an iOS build, and none means the build fails ("flet build" does not compile from source).
The verdict of a feature is "runs" (nothing blocked), "degrades" (only optional parts are blocked) or "blocked" (a needed package or program is not available).
What this cannot know: whether a package that has a wheel works well, and what Apple restricts (memory, background time, store rules): utils/selftest.py on a real
device and the phone sandbox (utils/phone_sandbox.py, run by the tests) cover those.
"""
import argparse
import ast
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "maestro"
IOS_TAG = "ios_13_0_arm64_iphoneos"
FLET_INDEX = "https://pypi.flet.dev/simple"
PROGRAM_CALLS = {("subprocess", name) for name in ("run", "Popen", "call", "check_call", "check_output")} | {("os", "system"), ("shutil", "which")}

# What the user sees -> the source files behind it (path prefixes under src/maestro/)
FEATURES = {
    "Reading a recording (WAV)": ["preprocessing/audio.py"],
    "Reading other audio and video (ffmpeg)": ["utils/convert_video_type.py", "utils/ffmpeg.py", "preprocessing/video.py"],
    "Pitch: YIN": ["audio/backends/yin.py", "audio/pitch.py", "audio/backends/base.py", "audio/backends/__init__.py"],
    "Pitch: CREPE": ["audio/backends/crepe.py", "audio/backends/crepe_onnx.py", "utils/device.py"],
    "Pitch: pYIN and Basic Pitch": ["audio/backends/pyin.py", "audio/backends/basic_pitch.py"],
    "Notes, alignment and the score (MusicXML, MIDI)": ["audio/", "alignment/", "score/musicxml.py", "score/score_representation.py", "score/repeats.py",
                                                      "score/symbolic_features.py", "utils/midi.py", "utils/convert_file_type.py", "preprocessing/score.py",
                                                      "preprocessing/denoise.py"],
    "Intonation, rhythm, expression, tone": ["assessment/"],
    "Scanning a score (reading photos and PDFs)": ["score/omr.py", "score/scan.py"],
    "Body movement (VioPose)": ["vision/"],
    "Tone and expression model (CLAP)": ["assessment/neural.py"],
    "Professional reference": ["reference/"],
    "Written feedback (the writing model)": ["feedback/"],
    "App screens and camera": ["app/"],
}
# Imports that name a different pip distribution
DISTRIBUTION = {"cv2": "opencv-python", "PIL": "Pillow", "sklearn": "scikit-learn", "yaml": "PyYAML", "skimage": "scikit-image", "basic_pitch": "basic-pitch",
                "flet_video": "flet-video", "flet_camera": "flet-camera", "flet_charts": "flet-charts", "flet_audio_recorder": "flet-audio-recorder",
                "flet_permission_handler": "flet-permission-handler", "flet_audio": "flet-audio"}


def feature_of(relative):
    """The most specific feature whose prefix matches a file (longest prefix wins)."""
    best = max(((len(p), name) for name, prefixes in FEATURES.items() for p in prefixes if relative.startswith(p)), default=None)
    return best[1] if best else None


def scan_file(path, relative):
    """-> ([(module, line, needed)], [(call, line)]): third-party imports (needed = at the top of the file, not in a function or try) and program calls."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports, programs = [], []

    def visit(node, needed):
        for child in ast.iter_child_nodes(node):
            inner = needed and not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Try, ast.ClassDef))
            if isinstance(child, ast.Import):
                imports.extend((a.name.split(".")[0], child.lineno, needed) for a in child.names)
            elif isinstance(child, ast.ImportFrom) and child.module and not child.level:
                imports.append((child.module.split(".")[0], child.lineno, needed))
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and isinstance(child.func.value, ast.Name):
                if (child.func.value.id, child.func.attr) in PROGRAM_CALLS:
                    programs.append((f"{child.func.value.id}.{child.func.attr}", child.lineno))
            visit(child, inner if not isinstance(child, ast.ClassDef) else needed)
    visit(tree, True)
    return ([i for i in imports if i[0] not in sys.stdlib_module_names and i[0] not in ("maestro", "__future__")], programs)


def audit(package=PACKAGE):
    """{feature: {"packages": {module: {"needed": bool, "files": [...]}}, "programs": [...]}} for every file of the package."""
    out = {}
    for path in sorted(Path(package).rglob("*.py")):
        relative = path.relative_to(package).as_posix()
        feature = feature_of(relative)
        if feature is None:
            continue
        imports, programs = scan_file(path, relative)
        entry = out.setdefault(feature, {"packages": {}, "programs": []})
        for module, line, needed in imports:
            info = entry["packages"].setdefault(module, {"needed": False, "files": []})
            info["needed"] = info["needed"] or needed
            info["files"].append(f"{relative}:{line}")
        entry["programs"] += [f"{call} ({relative}:{line})" for call, line in programs]
    return out


def distribution(module):
    return DISTRIBUTION.get(module) or next(iter(importlib.metadata.packages_distributions().get(module, [])), module)


def wheel_status(dist, flet=False):
    """Ask pip: "pure" (a py3-none-any wheel), "ios" (a wheel for an iPhone), or "none"; "unknown" if the index could not be reached."""
    with tempfile.TemporaryDirectory() as tmp:
        base = [sys.executable, "-m", "pip", "download", dist, "--no-deps", "--only-binary=:all:", "-d", tmp, "--disable-pip-version-check", "-q"]
        index = ["--index-url", FLET_INDEX, "--extra-index-url", "https://pypi.org/simple"] if flet else []
        ios = subprocess.run(base + index + ["--platform", IOS_TAG, "--python-version", "3.13", "--implementation", "cp", "--abi", "cp313"], capture_output=True, text=True)
        if ios.returncode == 0:
            name = next(Path(tmp).iterdir()).name
            return "pure" if name.endswith("-none-any.whl") else "ios"
        if "Could not find a version" in ios.stderr:
            return "none"
        return "unknown"


def verdicts(report, status=None):
    """Add a verdict to every feature: runs / degrades / blocked (+ why). `status`: {distribution: pure | ios | none | unknown} when known."""
    for feature, entry in report.items():
        blocked, optional = [], []
        for module, info in entry["packages"].items():
            state = (status or {}).get(distribution(module), "unknown")
            if state in ("none",):
                (blocked if info["needed"] else optional).append(f"{distribution(module)} has no iOS wheel")
        if entry["programs"]:
            (optional if feature not in ("Reading other audio and video (ffmpeg)", "Scanning a score (reading photos and PDFs)", "Body movement (VioPose)") else blocked)\
                .append("starts another program: " + ", ".join(sorted({p.split(" ")[0] for p in entry["programs"]})))
        entry["verdict"] = "blocked" if blocked else "degrades" if optional else "runs" if status else "unchecked"
        entry["why"] = blocked + optional
    return report


def render(report, status):
    lines = ["| feature | verdict | needs (top-level) | optional | starts programs |", "|---|---|---|---|---|"]
    for feature in FEATURES:
        if feature not in report:
            continue
        e = report[feature]
        mark = lambda module: f"{distribution(module)}" + (f" [{status[distribution(module)]}]" if status and distribution(module) in status else "")
        needed = ", ".join(sorted(mark(m) for m, i in e["packages"].items() if i["needed"])) or "-"
        optional = ", ".join(sorted(mark(m) for m, i in e["packages"].items() if not i["needed"])) or "-"
        lines.append(f"| {feature} | {e['verdict']} | {needed} | {optional} | {len({p.split(' ')[0] for p in e['programs']}) or '-'} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--online", action="store_true", help="ask pip whether each package has an iOS wheel")
    ap.add_argument("--flet", action="store_true", help="with --online: also use Flet's package index")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    report = audit()
    status = None
    if args.online:
        dists = sorted({distribution(m) for e in report.values() for m in e["packages"] if not m.startswith("flet")} | {"numpy", "scipy"})
        status = {d: wheel_status(d, args.flet) for d in dists}
    verdicts(report, status)
    print(json.dumps({"features": report, "packages": status}, indent=1) if args.json else render(report, status))


if __name__ == "__main__":
    main()
