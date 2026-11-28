# Where trained models and downloaded models are kept (new file).
#
# models/ at the repository root holds everything trained or exported by Maestro (models/onnx/*.onnx, and trained models if any are added);
# the MAESTRO_MODELS environment variable moves it. 
import os
from pathlib import Path


def models_root():
    """The models folder: $MAESTRO_MODELS, else the installed app's own storage (Flet sets FLET_APP_STORAGE_DATA: the folder inside an installed app is
    read-only, this one is not), else <repo>/models."""
    if os.environ.get("MAESTRO_MODELS"):
        return Path(os.environ["MAESTRO_MODELS"])
    if os.environ.get("FLET_APP_STORAGE_DATA"):
        return Path(os.environ["FLET_APP_STORAGE_DATA"]) / "models"
    return Path(__file__).resolve().parents[3] / "models"


def bundled_models():
    """Models shipped inside the installed app: src/assets/models next to the maestro package (empty in a development checkout)."""
    return Path(__file__).resolve().parents[2] / "assets" / "models"
