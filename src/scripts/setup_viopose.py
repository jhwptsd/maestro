"""Install VioPose (3D pose estimation) for Maestro's movement analysis.

Usage: python src/scripts/setup_viopose.py [--dir DIR] [--python PATH] [--skip-weights] [--dry-run]

Clones VioPose into third_party/, creates its own Python 3.9 environment (conda, uv or python3.9), installs its
requirements, downloads the weights and records where everything is (third_party/viopose.json). Safe to re-run: finished
steps are skipped. setup.ps1 / setup.sh call this; see maestro/vision/viopose_setup.py for the details.
"""
import argparse
import sys

from maestro.vision.viopose_setup import SetupError, setup_viopose

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", help="where to put the VioPose checkout (default third_party/VioPose)")
    ap.add_argument("--python", help="use this existing VioPose environment's interpreter instead of creating one")
    ap.add_argument("--skip-weights", action="store_true", help="do not download the model weights")
    ap.add_argument("--dry-run", action="store_true", help="only print what would be done")
    args = ap.parse_args()
    try:
        config = setup_viopose(args.dir, args.python, args.skip_weights, args.dry_run)
    except SetupError as e:
        sys.exit(f"VioPose setup: {e}")
    print(f"VioPose ready: {config['dir']}" if not args.dry_run else "dry run complete")
