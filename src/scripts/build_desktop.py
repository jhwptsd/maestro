"""Build the desktop app for this computer: export the models, then `flet build`.

    python src/scripts/build_desktop.py [windows|macos|linux] [--dry-run]

Windows builds on Windows, macOS on a Mac, Linux on Linux (Flutter builds for the platform it runs on). Needs the Flutter SDK (https://docs.flutter.dev/get-started/install)
and `pip install -e ".[app]"`; the model export also needs torch, torchcrepe and onnx. The result is in build/<platform>/. See the README, "The app".
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOST = {"win32": "windows", "darwin": "macos", "linux": "linux"}


def plan(target):
    """The commands to run, in order, for `target` (a list of argument lists)."""
    return [[sys.executable, str(ROOT / "src" / "scripts" / "export_crepe_onnx.py")], ["flet", "build", target, "--project", "Maestro", "--yes"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target", nargs="?", default=HOST.get(sys.platform), choices=("windows", "macos", "linux"))
    ap.add_argument("--dry-run", action="store_true", help="print the commands instead of running them")
    args = ap.parse_args(argv)
    if args.target != HOST.get(sys.platform):
        ap.error(f"{args.target} apps are built on {args.target}: Flutter builds for the platform it runs on (this is {HOST.get(sys.platform, sys.platform)}).")
    for command in plan(args.target):
        print("$", " ".join(command))
        if args.dry_run:
            continue
        if command[0] == "flet" and not shutil.which("flet"):
            sys.exit("flet was not found: pip install -e '.[app]' and install the Flutter SDK (https://docs.flutter.dev/get-started/install).")
        if subprocess.run(command, cwd=ROOT, env=None).returncode:
            sys.exit(f"stopped: {' '.join(command[:3])} failed")


if __name__ == "__main__":
    main()
