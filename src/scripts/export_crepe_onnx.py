"""Export the CREPE networks to ONNX into src/assets/models/, so the installed app can run CREPE without PyTorch.

    python src/scripts/export_crepe_onnx.py [tiny] [full]        # needs torch, torchcrepe and onnx (a build machine, not the installed app)

The installed desktop app contains no PyTorch: it runs the network through ONNX Runtime (audio/backends/crepe_numpy.py) and looks for the exported files in the
models folder and then in src/assets/models, which `flet build` packs into the app. Run this before `flet build` (build_desktop.py does).
"""
import sys
from pathlib import Path

from maestro.audio.backends import crepe_onnx

TARGET = Path(__file__).resolve().parents[1] / "assets" / "models"


def main(capacities=("tiny", "full")):
    TARGET.mkdir(parents=True, exist_ok=True)
    for capacity in capacities:
        print("exported", crepe_onnx.export(capacity, TARGET / f"crepe-{capacity}.onnx"))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("tiny", "full"))
