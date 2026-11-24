# The LLM providers behind one interface (new file; standard library only).
#
# OpenRouter and Groq, both with free tiers and both speaking the OpenAI chat-completions API, and a local small model through Ollama
# (no account, no internet, nothing leaves the computer): each `Provider` can `list_models(key)` and
# `complete(key, model, system, messages)` over plain HTTPS with urllib, so no vendor SDK is needed. A key comes from the provider's
# console, or, for OpenRouter, from a sign-in (feedback/oauth.py, gui/server.py). (Claude, ChatGPT and Gemini were tried and dropped: their
# app subscriptions cannot be linked, and a free Gemini key was often overloaded.) Free models are shared and get busy, so
# `complete_with_fallback` retries a busy model once and then tries other models. Used by feedback/generator.py and feedback/chat.py.
# Errors are `LLMError`s whose message is fit to show to the user (bad key, quota, unknown model, offline ...); `.status` carries the HTTP
# status if there was one. `base_url` can be overridden (tests point it at a local server).
import json
import os
import re
import time
import urllib.error
import urllib.request

TIMEOUT = 120                      # seconds for one request: a long answer from a large model takes a while
BUSY = (429, 502, 503, 504)        # statuses that mean 'busy or rate limited': worth a retry and then another model
RETRY_DELAY = 2.0                  # seconds before the retry of a busy model
MAX_ALTERNATIVES = 2               # other models tried after the chosen one failed


class LLMError(RuntimeError):
    """A provider call failed; the message is written for the user."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def http_json(url, headers, body=None, timeout=TIMEOUT):
    """GET (no body) or POST (JSON body) -> parsed JSON; HTTP and network failures become LLMError."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers},
                                     method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMError(_http_message(e), e.code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"Could not reach the provider ({getattr(e, 'reason', e)}). Check your internet connection.") from None
    except ValueError:
        raise LLMError("The provider sent an answer that is not valid JSON.") from None


def _http_message(error):
    """Readable text for an HTTP error: the provider's own message when it has one, plus what the status usually means."""
    try:
        detail = json.loads(error.read().decode("utf-8", "replace"))
        detail = detail.get("error", detail) if isinstance(detail, dict) else detail
        detail = detail.get("message") if isinstance(detail, dict) else str(detail)
    except (ValueError, AttributeError):
        detail = None
    meaning = {400: "the request was rejected", 401: "the API key is not valid", 403: "the API key is not allowed to do this",
               404: "that model was not found for this key", 429: "rate limit or quota exceeded (check the account's billing)",
               500: "the provider had an error", 503: "the provider is overloaded, try again"}.get(error.code, f"HTTP {error.code}")
    return f"{meaning[0].upper()}{meaning[1:]}" + (f" ({detail})" if detail else "") + "."


class Provider:
    """A service with the OpenAI chat-completions API. Subclasses give the name, console links, base URL and model preferences."""
    name = label = key_url = default_model = base_url = ""
    env_vars = ()
    prefer = ()                        # substrings of model ids preferred when choosing a default, in order
    needs_key = True                   # False for a local model: nothing to paste
    small_context = False              # True when the model can only take a short prompt: the digest is made compact (prompts.compact)
    timeout = TIMEOUT                  # seconds for one request
    _NOT_CHAT = ("embed", "whisper", "tts", "dall-e", "image", "moderation", "realtime", "transcribe", "audio", "rerank", "guard")

    def _headers(self, key):
        return {"Authorization": f"Bearer {key}"}

    def list_models(self, key):
        """[{"id", "label"}] of the models this key can use for text generation (embedding, speech and image models left out)."""
        data = http_json(f"{self.base_url}/v1/models", self._headers(key), timeout=self.timeout)
        ids = sorted(m["id"] for m in data.get("data", []))
        return [{"id": i, "label": i + (" (free)" if i.endswith(":free") else "")} for i in ids if not any(w in i.lower() for w in self._NOT_CHAT)]

    def complete(self, key, model, system, messages, max_tokens=2000, json_mode=False):
        """`messages`: [{"role": "user" | "assistant", "content": text}, ...], starting and ending with the user.
        -> {"text": the answer, "usage": {"input": tokens, "output": tokens} (0 if not reported)}. (`json_mode` is accepted for the callers'
        sake; the prompt asks for JSON, since not every model on these services supports a response-format option.)"""
        body = {"model": model, "messages": [{"role": "system", "content": system}, *messages], "max_tokens": max_tokens}
        data = http_json(f"{self.base_url}/v1/chat/completions", self._headers(key), body, timeout=self.timeout)
        usage = data.get("usage") or {}
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError("The provider returned no answer.") from None
        return {"text": text, "usage": {"input": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0)}}

    def pick_model(self, models, exclude=()):
        """The model to use among `models` ([{"id"}], minus `exclude`): the provider's default if offered, else the first of the preferred
        kinds, else the first one; None if there is none."""
        ids = [m["id"] for m in models if m["id"] not in exclude]
        if self.default_model in ids:
            return self.default_model
        for word in self.prefer:
            found = [i for i in ids if word in i]
            if found:
                return found[0]
        return ids[0] if ids else None


class OpenRouter(Provider):
    """OpenRouter: many models (some free) through one API; the key can come from a sign-in (OAuth PKCE)."""
    name, label = "openrouter", "OpenRouter (free models available)"
    key_url = "https://openrouter.ai/keys"
    env_vars = ("OPENROUTER_API_KEY",)
    default_model = "meta-llama/llama-3.3-70b-instruct:free"
    base_url = "https://openrouter.ai/api"
    prefer = ("llama-3.3-70b-instruct:free", "gemini", "deepseek", "qwen", "mistral", ":free")

    def list_models(self, key):
        return sorted(super().list_models(key), key=lambda m: not m["id"].endswith(":free"))               # free ones first

    def pick_model(self, models, exclude=()):
        """A free model (the free tier is the point of using this); only if there is none, the best of the rest."""
        free = [m for m in models if m["id"].endswith(":free")]
        return super().pick_model(free, exclude) or super().pick_model(models, exclude)


class Groq(Provider):
    name, label = "groq", "Groq (free tier)"
    key_url = "https://console.groq.com/keys"
    env_vars = ("GROQ_API_KEY",)
    default_model = "llama-3.3-70b-versatile"
    base_url = "https://api.groq.com/openai"
    prefer = ("llama-3.3-70b", "70b", "llama")


class Local(Provider):
    """A small model running on this computer through Ollama (https://ollama.com, free): no account, no key, no internet once the model
    is downloaded, and nothing is sent anywhere. It talks to Ollama's own API rather than the OpenAI-style one because that lets us set the
    context window: Ollama's default (2048 tokens) would silently cut the prompt. The suggested models are small enough (about 1 to 4
    billion parameters, 4-bit) to run on a phone too, which also keeps them light for a laptop, so the digest sent to them is made
    compact (prompts.compact) and the answer is asked for in JSON mode."""
    name, label = "local", "Local model (Ollama: private, no internet)"
    key_url = "https://ollama.com/download"
    env_vars = ()
    default_model = "llama3.2:3b"
    prefer = ("llama3.2", "qwen", "gemma", "phi3", "llama3", "mistral")
    needs_key = False
    small_context = True
    timeout = 900                      # a small model on a CPU writes slowly, and the first call also loads it
    CONTEXT = 8192                     # tokens: the compact digest, the prompt rules and the answer fit comfortably
    SUGGESTED = (                      # (id, what it is): small enough for phones as well
        {"id": "llama3.2:1b", "label": "Llama 3.2 1B (1.3 GB; fits any recent phone, fastest, simplest feedback)"},
        {"id": "llama3.2:3b", "label": "Llama 3.2 3B (2 GB; phones with 6 GB+ of memory, best balance)"},
        {"id": "qwen3.5:4b", "label": "Qwen 3.5 4B (about 3 GB; phones with 8 GB of memory)"},
        {"id": "qwen2.5:1.5b", "label": "Qwen 2.5 1.5B (1 GB; small and good at following instructions)"},
        {"id": "gemma2:2b", "label": "Gemma 2 2B (1.6 GB)"},
        {"id": "phi3.5", "label": "Phi 3.5 mini 3.8B (2.2 GB; strong for its size)"},
    )

    def __init__(self):
        self._override = None

    @property
    def base_url(self):
        """http://localhost:11434, or what OLLAMA_HOST says (Ollama's own setting), or what was set for tests."""
        if self._override:
            return self._override
        host = os.environ.get("OLLAMA_HOST", "").strip()
        if host:
            return host.rstrip("/") if "://" in host else f"http://{host}"
        return "http://localhost:11434"

    @base_url.setter
    def base_url(self, value):
        self._override = value

    @base_url.deleter
    def base_url(self):
        self._override = None

    def _call(self, path, body=None, timeout=None):
        try:
            return http_json(f"{self.base_url}{path}", {}, body, timeout or self.timeout)
        except LLMError as e:
            if e.status is None and "Could not reach" in str(e):
                raise LLMError(f"Ollama is not running at {self.base_url}. Install it from https://ollama.com/download (free), start it, "
                               "and try again.") from None
            raise

    def list_models(self, key=None):
        """The models already downloaded to Ollama (embedding models left out)."""
        data = self._call("/api/tags", timeout=30)
        return [{"id": m["name"], "label": f"{m['name']} ({m.get('size', 0) / 1e9:.1f} GB)"} for m in data.get("models", [])
                if "embed" not in m["name"].lower()]

    def complete(self, key, model, system, messages, max_tokens=2000, json_mode=False):
        body = {"model": model, "stream": False, "messages": [{"role": "system", "content": system}, *messages],
                "options": {"num_ctx": self.CONTEXT, "num_predict": max_tokens, "temperature": 0.4}}
        if json_mode:
            body["format"] = "json"                                # Ollama then only produces valid JSON
        # "Thinking" models (Qwen 3.x, ...) would spend the whole token budget reasoning and return an empty answer: switch that off.
        # (An Ollama or model without the option rejects it: then ask again without.)
        try:
            data = self._call("/api/chat", {**body, "think": False})
        except LLMError as e:
            if e.status != 400 or "think" not in str(e).lower():
                raise
            data = self._call("/api/chat", body)
        text = (data.get("message") or {}).get("content", "")
        if not text:
            raise LLMError("The local model returned no answer. Try a different model, or update Ollama (https://ollama.com/download).")
        return {"text": text, "usage": {"input": data.get("prompt_eval_count", 0), "output": data.get("eval_count", 0)}}

    def pull(self, model, on_progress):
        """Download `model` into Ollama, calling on_progress(fraction 0-1 of the biggest piece, text) as it goes (blocking; streamed)."""
        import urllib.error
        import urllib.request
        request = urllib.request.Request(f"{self.base_url}/api/pull", data=json.dumps({"name": model, "stream": True}).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
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
            raise LLMError(_http_message(e), e.code) from None
        except (urllib.error.URLError, OSError):
            raise LLMError(f"Ollama is not running at {self.base_url}. Install it from https://ollama.com/download (free), start it, "
                           "and try again.") from None


PROVIDERS = {p.name: p for p in (Local(), OpenRouter(), Groq())}


def get(name, base_url=None):
    """The provider called `name` ("local", "openrouter" or "groq"), optionally with another base URL (tests, proxies)."""
    if name not in PROVIDERS:
        raise LLMError(f"Unknown provider {name!r}: choose one of {', '.join(PROVIDERS)}.")
    provider = PROVIDERS[name]
    if base_url:
        provider = type(provider)()
        provider.base_url = re.sub(r"/v1$", "", base_url.rstrip("/"))      # (the paths of the OpenAI-style services start with /v1)
    return provider


def complete_with_fallback(provider, key, model, system, messages, max_tokens=2000, json_mode=False):
    """provider.complete, made robust against what free models do:
      - busy or rate limited (429, 502, 503, 504): wait RETRY_DELAY seconds and retry once, then try other models;
      - unknown to the account (404: retired or never available to this key): try other models;
      - any other error (bad key, bad request, offline) is raised as it is.
    At most MAX_ALTERNATIVES other models are tried, best first (a free model is only replaced by another free one, so a fallback never
    spends money). -> (reply, model used, note or None)."""
    def attempt(m):
        return provider.complete(key, m, system, messages, max_tokens, json_mode)

    try:
        return attempt(model), model, None
    except LLMError as first:
        if first.status not in (404, *BUSY):
            raise
        error = first
        if first.status in BUSY:
            time.sleep(RETRY_DELAY)
            try:
                return attempt(model), model, "the model was busy; the second try worked"
            except LLMError as again:
                if again.status not in (404, *BUSY):
                    raise
                error = again
        tried, models = {model}, provider.list_models(key)
        if model.endswith(":free"):                                  # never turn a free request into a paid one
            models = [m for m in models if m["id"].endswith(":free")]
        for _ in range(MAX_ALTERNATIVES):
            alternative = provider.pick_model(models, exclude=tried)
            if alternative is None:
                break
            tried.add(alternative)
            try:
                reply = attempt(alternative)
            except LLMError as failed:
                if failed.status not in (404, *BUSY):
                    raise
                error = failed
                continue
            why = "busy" if first.status in BUSY else "not available to this key"
            return reply, alternative, f"model {model} was {why}, so {alternative} was used"
        raise error
