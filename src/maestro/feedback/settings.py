# Which LLM provider the user connected, with which key and model (new file).
#
# Stored per user, not in the project: `llm.json` in the config folder ($MAESTRO_CONFIG, else %APPDATA%\maestro on Windows,
# ~/.config/maestro elsewhere), written readable by the user only where the OS supports that. An environment variable
# (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY / GOOGLE_API_KEY) is used when no key was saved. Keys never go into the
# report page, the logs or the digest sent to the model; the GUI only ever shows `masked(key)`. gui/server.py edits these settings
# for the "connect" panel, and feedback/generator.py reads them.
import json
import os
import sys
import tempfile
from pathlib import Path

from maestro.feedback import providers


def config_dir():
    if os.environ.get("MAESTRO_CONFIG"):
        return Path(os.environ["MAESTRO_CONFIG"])
    base = os.environ.get("APPDATA") if sys.platform == "win32" else os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base or Path.home()) / "maestro"


def _path():
    return config_dir() / "llm.json"


def load():
    """{"provider": chosen provider or None, "keys": {provider: key}, "models": {provider: model}}."""
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {"provider": data.get("provider"), "keys": dict(data.get("keys", {})), "models": dict(data.get("models", {}))}


def _save(data):
    folder = config_dir()
    folder.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")           # write aside, then replace: never a half-written key file
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        if sys.platform != "win32":
            os.chmod(tmp, 0o600)
        os.replace(tmp, _path())
    finally:
        Path(tmp).unlink(missing_ok=True)


def get_key(provider):
    """The saved key for `provider`, else the first of its environment variables that is set, else None."""
    saved = load()["keys"].get(provider)
    if saved:
        return saved
    return next((os.environ[v] for v in providers.get(provider).env_vars if os.environ.get(v)), None)


def set_key(provider, key, model=None):
    """Save a key (and make this the chosen provider). `model` defaults to the provider's default."""
    providers.get(provider)
    data = load()
    data["keys"][provider] = key.strip()
    data["models"].setdefault(provider, model or providers.PROVIDERS[provider].default_model)
    if model:
        data["models"][provider] = model
    data["provider"] = provider
    _save(data)


def choose(provider, model=None):
    """Choose which connected provider (and model) writes the feedback."""
    providers.get(provider)
    data = load()
    if not (data["keys"].get(provider) or get_key(provider)):
        raise providers.LLMError(f"{providers.PROVIDERS[provider].label} is not connected yet.")
    data["provider"] = provider
    if model:
        data["models"][provider] = model
    _save(data)


def remove_key(provider):
    data = load()
    data["keys"].pop(provider, None)
    data["models"].pop(provider, None)
    if data["provider"] == provider:
        data["provider"] = next(iter(data["keys"]), None)
    _save(data)


def masked(key):
    """A key as it may be shown: its first 3 and last 4 characters (nothing for short keys)."""
    return f"{key[:3]}...{key[-4:]}" if key and len(key) >= 16 else ("set" if key else "")


def status():
    """What the GUI shows: per provider {label, connected, key (masked), source, model, key_url}, and the chosen provider."""
    data = load()
    out = {}
    for name, p in providers.PROVIDERS.items():
        key, saved = get_key(name), data["keys"].get(name)
        out[name] = {"label": p.label, "connected": bool(key), "key": masked(key), "source": "saved" if saved else ("environment" if key else ""),
                     "model": data["models"].get(name) or p.default_model, "key_url": p.key_url}
    chosen = data["provider"] if data["provider"] and out[data["provider"]]["connected"] else next((n for n, s in out.items() if s["connected"]), None)
    return {"provider": chosen, "providers": out}
