# The local language model behind the written feedback and chat (replaces providers.py; standard library only).
#
# Maestro writes its feedback with a small model running on this computer through Ollama (https://ollama.com, free): no account, no key,
# no internet once a model is downloaded, and nothing leaves the computer. The suggested models are small 4-bit models of 1 to 4 billion
# parameters, light enough for a laptop and for a recent phone. Ollama's own API is used (not the OpenAI-style one) because it lets us
# set the context window (its default of 2048 tokens would silently cut the prompt), ask for JSON-only output and switch "thinking" off.
# Because the models are small, the digest sent to them is made compact (prompts.compact). Used by generator.py, chat.py and gui/server.py.
# Errors are `LLMError`s whose message is fit to show to the user; `.status` carries the HTTP status if there was one. `OLLAMA_HOST`
# moves the server (Ollama's own setting); tests point `BASE_URL` at a fake.
import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = "llama3.2:3b"
PREFER = ("llama3.2", "qwen", "gemma", "phi3", "llama3", "mistral")    # substrings of installed model names, best first, when choosing one
SUGGESTED = (                                                          # offered for download; all small enough for phones as well
    {"id": "llama3.2:1b", "label": "Llama 3.2 1B (1.3 GB; fits any recent phone, fastest, simplest feedback)"},
    {"id": "llama3.2:3b", "label": "Llama 3.2 3B (2 GB; phones with 6 GB+ of memory, best balance)"},
    {"id": "qwen3.5:4b", "label": "Qwen 3.5 4B (about 3 GB; phones with 8 GB of memory)"},
    {"id": "qwen2.5:1.5b", "label": "Qwen 2.5 1.5B (1 GB; small and good at following instructions)"},
    {"id": "gemma2:2b", "label": "Gemma 2 2B (1.6 GB)"},
    {"id": "phi3.5", "label": "Phi 3.5 mini 3.8B (2.2 GB; strong for its size)"},
)
CONTEXT = 8192                     # tokens: the compact digest, the prompt rules and the answer fit comfortably
TIMEOUT = 900                      # seconds: a small model on a CPU writes slowly, and the first call also loads it
BASE_URL = None                    # overrides the address (tests); else OLLAMA_HOST, else localhost
NOT_RUNNING = "Ollama is not running at {url}. Install it from https://ollama.com/download (free), start it, and try again."


class LLMError(RuntimeError):
    """A call to the local model failed; the message is written for the user."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def base_url():
    """http://localhost:11434, or what OLLAMA_HOST says."""
    if BASE_URL:
        return BASE_URL
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if host:
        return host.rstrip("/") if "://" in host else f"http://{host}"
    return "http://localhost:11434"


def _call(path, body=None, timeout=TIMEOUT):
    """GET (no body) or POST (JSON body) to Ollama -> parsed JSON; failures become LLMError."""
    request = urllib.request.Request(base_url() + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8", "replace")).get("error")
        except (ValueError, AttributeError):
            detail = None
        raise LLMError(detail or f"Ollama answered with HTTP {e.code}.", e.code) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise LLMError(NOT_RUNNING.format(url=base_url())) from None
    except ValueError:
        raise LLMError("Ollama sent an answer that is not valid JSON.") from None


def list_models():
    """The models already downloaded to Ollama (embedding models left out): [{"id", "label"}]."""
    data = _call("/api/tags", timeout=30)
    return [{"id": m["name"], "label": f"{m['name']} ({m.get('size', 0) / 1e9:.1f} GB)"} for m in data.get("models", [])
            if "embed" not in m["name"].lower()]


def pick_model(models, exclude=()):
    """The model to use among `models` ([{"id"}], minus `exclude`): the default if installed, else the first of the preferred kinds,
    else the first one; None if there is none."""
    ids = [m["id"] for m in models if m["id"] not in exclude]
    if DEFAULT_MODEL in ids:
        return DEFAULT_MODEL
    for word in PREFER:
        found = [i for i in ids if word in i]
        if found:
            return found[0]
    return ids[0] if ids else None


def complete(model, system, messages, max_tokens=2000, json_mode=False):
    """One answer from `model` -> {"text", "usage": {"input", "output"}}. `messages`: [{"role": "user" | "assistant", "content"}]."""
    body = {"model": model, "stream": False, "messages": [{"role": "system", "content": system}, *messages],
            "options": {"num_ctx": CONTEXT, "num_predict": max_tokens, "temperature": 0.4}}
    if json_mode:
        body["format"] = "json"                                # Ollama then only produces valid JSON
    # "Thinking" models (Qwen 3.x, ...) would spend the whole token budget reasoning and return an empty answer: switch that off.
    # (An Ollama or model without the option rejects it: then ask again without.)
    try:
        data = _call("/api/chat", {**body, "think": False})
    except LLMError as e:
        if e.status != 400 or "think" not in str(e).lower():
            raise
        data = _call("/api/chat", body)
    text = (data.get("message") or {}).get("content", "")
    if not text:
        raise LLMError("The local model returned no answer. Try a different model, or update Ollama (https://ollama.com/download).")
    return {"text": text, "usage": {"input": data.get("prompt_eval_count", 0), "output": data.get("eval_count", 0)}}


def complete_with_fallback(model, system, messages, max_tokens=2000, json_mode=False):
    """`complete`, but if `model` is not installed (404) another installed model is used. -> (reply, model used, note or None)."""
    try:
        return complete(model, system, messages, max_tokens, json_mode), model, None
    except LLMError as e:
        if e.status != 404:
            raise
        other = pick_model(list_models(), exclude={model})
        if other is None:
            raise LLMError(f"The model {model} is not downloaded, and no other is: press Download and use on the start page "
                           f"(or run: ollama pull {DEFAULT_MODEL}).") from None
        return complete(other, system, messages, max_tokens, json_mode), other, f"model {model} is not downloaded, so {other} was used"


def pull(model, on_progress):
    """Download `model` into Ollama, calling on_progress(fraction 0-1 of the biggest piece, text) as it goes (blocking; streamed)."""
    request = urllib.request.Request(f"{base_url()}/api/pull", data=json.dumps({"name": model, "stream": True}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            best = 0.0
            for line in response:
                info = json.loads(line)
                if info.get("error"):
                    raise LLMError(f"Ollama could not download {model}: {info['error']}")
                if info.get("total"):
                    best = max(best, info.get("completed", 0) / info["total"])      # layers come one after another: never go back
                    on_progress(best, f"Downloading {model}: {info.get('completed', 0) / 1e9:.1f} of {info['total'] / 1e9:.1f} GB")
                else:
                    on_progress(best, str(info.get("status", "")))
    except urllib.error.HTTPError as e:
        raise LLMError(f"Ollama could not download {model} (HTTP {e.code}).", e.code) from None
    except (urllib.error.URLError, OSError):
        raise LLMError(NOT_RUNNING.format(url=base_url())) from None
