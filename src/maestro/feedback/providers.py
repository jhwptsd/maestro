# The LLM providers behind one interface (new file; standard library only).
#
# OpenRouter and Groq, both with free tiers and both speaking the OpenAI chat-completions API: each `Provider` can `list_models(key)` and
# `complete(key, model, system, messages)` over plain HTTPS with urllib, so no vendor SDK is needed. A key comes from the provider's
# console, or, for OpenRouter, from a sign-in (feedback/oauth.py, gui/server.py). (Claude, ChatGPT and Gemini were tried and dropped: their
# app subscriptions cannot be linked, and a free Gemini key was often overloaded.) Free models are shared and get busy, so
# `complete_with_fallback` retries a busy model once and then tries other models. Used by feedback/generator.py and feedback/chat.py.
# Errors are `LLMError`s whose message is fit to show to the user (bad key, quota, unknown model, offline ...); `.status` carries the HTTP
# status if there was one. `base_url` can be overridden (tests point it at a local server).
import json
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
    _NOT_CHAT = ("embed", "whisper", "tts", "dall-e", "image", "moderation", "realtime", "transcribe", "audio", "rerank", "guard")

    def _headers(self, key):
        return {"Authorization": f"Bearer {key}"}

    def list_models(self, key):
        """[{"id", "label"}] of the models this key can use for text generation (embedding, speech and image models left out)."""
        data = http_json(f"{self.base_url}/v1/models", self._headers(key))
        ids = sorted(m["id"] for m in data.get("data", []))
        return [{"id": i, "label": i + (" (free)" if i.endswith(":free") else "")} for i in ids if not any(w in i.lower() for w in self._NOT_CHAT)]

    def complete(self, key, model, system, messages, max_tokens=2000, json_mode=False):
        """`messages`: [{"role": "user" | "assistant", "content": text}, ...], starting and ending with the user.
        -> {"text": the answer, "usage": {"input": tokens, "output": tokens} (0 if not reported)}. (`json_mode` is accepted for the callers'
        sake; the prompt asks for JSON, since not every model on these services supports a response-format option.)"""
        body = {"model": model, "messages": [{"role": "system", "content": system}, *messages], "max_tokens": max_tokens}
        data = http_json(f"{self.base_url}/v1/chat/completions", self._headers(key), body)
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


PROVIDERS = {p.name: p for p in (OpenRouter(), Groq())}


def get(name, base_url=None):
    """The provider called `name` ("openrouter" or "groq"), optionally with another base URL (tests, proxies)."""
    if name not in PROVIDERS:
        raise LLMError(f"Unknown provider {name!r}: choose one of {', '.join(PROVIDERS)}.")
    provider = PROVIDERS[name]
    if base_url:
        provider = type(provider)()
        provider.base_url = re.sub(r"/v1$", "", base_url.rstrip("/"))      # the paths we call start with /v1 already
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
