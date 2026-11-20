# Embed a Performance with the trained fusion encoders (new file).
#
# Loads whichever of models/fusion/{audio,video,score}.pt exist (written by src/training/train_fusion.py; the folder can be
# moved with the MAESTRO_MODELS environment variable) and runs each encoder on its modality of the Performance:
#   audio  <- perf.audio + perf.audio_features        (fusion/audio_encoder.py)
#   video  <- perf.pose  + perf.motion_features       (fusion/video_encoder.py)
#   score  <- perf.score + perf.score_features + alignment   (fusion/score_encoder.py)
# Result, also stored in perf.embeddings and plain JSON: {"audio": {"t", "window_s", "z", "recon_error"}, "video": {...},
# "score": {"t", "notes", "z", "recon_error"}}, one z (16 numbers) per window on the performance's clock. A modality with
# no checkpoint or no input (no pose, say) is simply absent. maestro.analyze(embeddings=...) calls this; fusion/fusion.py
# is where the three streams are to be combined.
import os
from pathlib import Path

from maestro.fusion import audio_encoder, score_encoder, video_encoder
from maestro.fusion.residual_autoencoder import load_checkpoint

MODULES = {"audio": audio_encoder, "video": video_encoder, "score": score_encoder}
BUILDERS = {kind: module.build for kind, module in MODULES.items()}


def models_dir():
    """Folder with the checkpoints: $MAESTRO_MODELS/fusion, else <repo>/models/fusion."""
    base = os.environ.get("MAESTRO_MODELS") or Path(__file__).resolve().parents[3] / "models"
    return Path(base) / "fusion"


def available(directory=None):
    """Kinds that have a checkpoint in `directory`."""
    directory = Path(directory or models_dir())
    return [kind for kind in MODULES if (directory / f"{kind}.pt").is_file()]


def _inputs(perf):
    """Which modalities the Performance has the inputs for."""
    return {"audio": perf.audio is not None and bool(perf.audio_features), "video": perf.pose is not None,
            "score": bool(perf.score_features) and perf.score is not None and bool(perf.alignment)}


def embed_performance(perf, directory=None, device="cpu"):
    """Run every available encoder on `perf` -> the dict above (also stored in perf.embeddings)."""
    directory, have = Path(directory or models_dir()), _inputs(perf)
    out = {}
    for kind in available(directory):
        if have[kind]:
            model, _ = load_checkpoint(directory / f"{kind}.pt", BUILDERS, device)
            out[kind] = MODULES[kind].embed(perf, model, device)
    perf.embeddings = out
    return out
