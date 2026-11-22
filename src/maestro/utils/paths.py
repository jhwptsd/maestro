# Where trained models and downloaded models are kept (new file).
#
# models/ at the repository root holds everything trained or exported by Maestro (models/onnx/*.onnx, and trained models if any are added);
# the MAESTRO_MODELS environment variable moves it. 
import os
from pathlib import Path


def models_root():
    """The models folder: $MAESTRO_MODELS, else <repo>/models."""
    return Path(os.environ.get("MAESTRO_MODELS") or Path(__file__).resolve().parents[3] / "models")
