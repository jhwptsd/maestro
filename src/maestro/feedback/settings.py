# Which local model writes the feedback (rewritten: there is nothing else to set up).
#
# The chosen model name is kept per user in `llm.json` in the config folder ($MAESTRO_CONFIG, else %APPDATA%\maestro on Windows,
# ~/.config/maestro elsewhere). Without a choice the best installed model is used (ollama.pick_model). gui/server.py sets it when a model
# is downloaded or selected; feedback/generator.py reads it.
import json
import os
import sys
import tempfile
from pathlib import Path


def config_dir():
    if os.environ.get("MAESTRO_CONFIG"):
        return Path(os.environ["MAESTRO_CONFIG"])
    base = os.environ.get("APPDATA") if sys.platform == "win32" else os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base or Path.home()) / "maestro"


def get_model():
    """The saved model name, or None (a missing, corrupt or old-format file counts as nothing saved)."""
    try:
        model = json.loads((config_dir() / "llm.json").read_text(encoding="utf-8")).get("model")
    except (OSError, ValueError, AttributeError):
        return None
    return model if isinstance(model, str) and model else None


def set_model(model):
    folder = config_dir()
    folder.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")           # write aside, then replace: never a half-written file
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"model": model}, f)
        os.replace(tmp, folder / "llm.json")
    finally:
        Path(tmp).unlink(missing_ok=True)
