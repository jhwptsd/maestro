# The three LLM providers behind one interface (new file; standard library only).
#
# Anthropic (Claude), OpenAI (ChatGPT models) and Google (Gemini): each `Provider` can `list_models(key)` and
# `complete(key, model, system, user)`, over plain HTTPS with urllib, so no vendor SDK is needed. They need an API key from
# the provider's developer console; a Claude.ai, ChatGPT or Gemini app subscription is a different product and cannot be linked
# (none of the three offers third-party sign-in to it), so the user pastes a key (see feedback/settings.py, gui/server.py).
# Used by feedback/generator.py. Errors are `LLMError`s whose message is fit to show to the user (bad key, quota, offline ...).
# `base_url` can be overridden (tests point it at a local server).
import json
import urllib.error
import urllib.request

TIMEOUT = 120                      # seconds for one request: a long answer from a large model takes a while


class LLMError(RuntimeError):
    """A provider call failed; the message is written for the user."""


def http_json(url, headers, body=None, timeout=TIMEOUT):
    """GET (no body) or POST (JSON body) -> parsed JSON; HTTP and network failures become LLMError."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers},
                                     method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMError(_http_message(e)) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"Could not reach the provider ({getattr(e, 'reason', e)}). Check your internet connection.") from None
    except ValueError:
        raise LLMError("The provider sent an answer that is not valid JSON.") from None


def _http_message(error):
    """Readable text for an HTTP error: the provider's own message when it has one, plus what the status usually means."""
    try:
        detail = json.loads(error.read().decode("utf-8", "replace"))
        detail = detail.get("error", detail)
        detail = detail.get("message") if isinstance(detail, dict) else str(detail)
    except (ValueError, AttributeError):
        detail = None
    meaning = {400: "the request was rejected", 401: "the API key is not valid", 403: "the API key is not allowed to do this",
               404: "that model was not found for this key", 429: "rate limit or quota exceeded (check the account's billing)",
               500: "the provider had an error", 503: "the provider is overloaded, try again"}.get(error.code, f"HTTP {error.code}")
    return f"{meaning[0].upper()}{meaning[1:]}" + (f" ({detail})" if detail else "") + "."


class Provider:
    name = label = key_url = default_model = base_url = ""
    env_vars = ()

    def list_models(self, key):
        """[{"id", "label"}] of the models this key can use for text generation."""
        raise NotImplementedError

    def complete(self, key, model, system, user, max_tokens=2000, json_mode=False):
        """-> {"text": the answer, "usage": {"input": tokens, "output": tokens} (0 if not reported)}."""
        raise NotImplementedError


class Anthropic(Provider):
    name, label = "anthropic", "Claude (Anthropic)"
    key_url = "https://console.anthropic.com/settings/keys"
    env_vars = ("ANTHROPIC_API_KEY",)
    default_model = "claude-sonnet-5-5"
    base_url = "https://api.anthropic.com"

    def _headers(self, key):
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}

    def list_models(self, key):
        data = http_json(f"{self.base_url}/v1/models?limit=100", self._headers(key))
        return [{"id": m["id"], "label": m.get("display_name") or m["id"]} for m in data.get("data", [])]

    def complete(self, key, model, system, user, max_tokens=2000, json_mode=False):
        data = http_json(f"{self.base_url}/v1/messages", self._headers(key),
                         {"model": model, "max_tokens": max_tokens, "system": system,
                          "messages": [{"role": "user", "content": user}]})
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage", {})
        return {"text": text, "usage": {"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0)}}


class OpenAI(Provider):
    name, label = "openai", "ChatGPT (OpenAI)"
    key_url = "https://platform.openai.com/api-keys"
    env_vars = ("OPENAI_API_KEY",)
    default_model = "gpt-4o"
    base_url = "https://api.openai.com"
    _NOT_CHAT = ("embedding", "whisper", "tts", "dall-e", "image", "moderation", "realtime", "transcribe", "audio", "search",
                 "davinci", "babbage", "instruct", "codex")

    def _headers(self, key):
        return {"Authorization": f"Bearer {key}"}

    def list_models(self, key):
        data = http_json(f"{self.base_url}/v1/models", self._headers(key))
        ids = sorted(m["id"] for m in data.get("data", []))
        chat = [i for i in ids if i.startswith(("gpt", "chatgpt", "o1", "o3", "o4", "o5")) and not any(w in i for w in self._NOT_CHAT)]
        return [{"id": i, "label": i} for i in chat]

    def complete(self, key, model, system, user, max_tokens=2000, json_mode=False):
        body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_completion_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = http_json(f"{self.base_url}/v1/chat/completions", self._headers(key), body)
        usage = data.get("usage", {})
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise LLMError("The provider returned no answer.") from None
        return {"text": text, "usage": {"input": usage.get("prompt_tokens", 0), "output": usage.get("completion_tokens", 0)}}


class Gemini(Provider):
    name, label = "gemini", "Gemini (Google)"
    key_url = "https://aistudio.google.com/apikey"
    env_vars = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    default_model = "gemini-2.5-flash"
    base_url = "https://generativelanguage.googleapis.com"

    def _headers(self, key):
        return {"x-goog-api-key": key}

    def list_models(self, key):
        data = http_json(f"{self.base_url}/v1beta/models?pageSize=200", self._headers(key))
        return [{"id": m["name"].split("/", 1)[-1], "label": m.get("displayName") or m["name"]} for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", []) and m["name"].split("/", 1)[-1].startswith("gemini")]

    def complete(self, key, model, system, user, max_tokens=2000, json_mode=False):
        config = {"maxOutputTokens": max_tokens}
        if json_mode:
            config["responseMimeType"] = "application/json"
        data = http_json(f"{self.base_url}/v1beta/models/{model}:generateContent", self._headers(key),
                         {"systemInstruction": {"parts": [{"text": system}]},
                          "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": config})
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            reason = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMError("Gemini returned no answer" + (f" (blocked: {reason})." if reason else ".")) from None
        usage = data.get("usageMetadata", {})
        return {"text": "".join(p.get("text", "") for p in parts),
                "usage": {"input": usage.get("promptTokenCount", 0), "output": usage.get("candidatesTokenCount", 0)}}


PROVIDERS = {p.name: p for p in (Anthropic(), OpenAI(), Gemini())}


def get(name, base_url=None):
    """The provider called `name` ("anthropic", "openai", "gemini"), optionally with another base URL (tests, proxies)."""
    if name not in PROVIDERS:
        raise LLMError(f"Unknown provider {name!r}: choose one of {', '.join(PROVIDERS)}.")
    provider = PROVIDERS[name]
    if base_url:
        provider = type(provider)()
        provider.base_url = base_url.rstrip("/")
    return provider
