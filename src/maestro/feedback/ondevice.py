# A writing model that runs on the phone itself (new file): the seam where a native engine plugs in.
#
# Ollama cannot run on an iPhone, so the written feedback there needs an engine that does: Apple's Foundation Models framework (iOS 26, Apple Intelligence
# devices), a llama.cpp or MLX build of a small open model, or MediaPipe's LLM inference. Those are Swift or C++ and reach Python only through a Flet extension
# (a Flutter plugin with a Python control). Nothing here is that engine; it is the contract it must meet and everything on the Python side that does not depend on it:
#   Engine         what an engine provides: available(), models(), generate(model, system, messages, max_tokens) -> text (blocking, called off the interface thread)
#   register()     the app registers the engine it has (none on a computer: Ollama is used there)
#   CallbackEngine an Engine for an engine reached through an async call (a Flet control's method call): it runs the call on the app's event loop and waits
# feedback/ollama.py asks this module for models first, so a registered engine shows up as "device:<id>" next to (or instead of) Ollama's, is preferred by
# `pick_model` (no server, no download), and `complete` hands its prompts here; the generator, the chat and the settings screen need no change.
import asyncio

PREFIX = "device:"
_engine = None


class Engine:
    """What a native engine implements."""
    name = "this device"

    def available(self):
        """True when the model can run now (the device supports it, its weights are present)."""
        raise NotImplementedError

    def models(self):
        """[{"id", "label"}] of the models it offers."""
        raise NotImplementedError

    def generate(self, model, system, messages, max_tokens):
        """One answer as text. `messages`: [{"role": "user" | "assistant", "content"}]. Raise RuntimeError with a sentence for the user if it fails."""
        raise NotImplementedError


class CallbackEngine(Engine):
    """An engine reached by an async call `invoke(method, arguments) -> value` (a Flet extension control's method call), with the methods
    "status" -> {"available": bool, "models": [{"id", "label"}]} and "generate" {model, system, messages, max_tokens} -> {"text"}. `loop` is the
    event loop the call must run on (the page's); generate() blocks the calling worker thread until the answer is back."""

    def __init__(self, invoke, loop, name="this device", timeout=600):
        self._invoke, self._loop, self.name, self._timeout = invoke, loop, name, timeout
        self._status = {"available": False, "models": []}

    def _call(self, method, arguments):
        return asyncio.run_coroutine_threadsafe(self._invoke(method, arguments), self._loop).result(self._timeout)

    async def refresh(self):
        """Ask the engine what it has (call once at start, on the event loop)."""
        self._status = await self._invoke("status", {})

    def available(self):
        return bool(self._status.get("available"))

    def models(self):
        return list(self._status.get("models", []))

    def generate(self, model, system, messages, max_tokens):
        return self._call("generate", {"model": model, "system": system, "messages": messages, "max_tokens": max_tokens})["text"]


def register(engine):
    global _engine
    _engine = engine


def unregister():
    register(None)


def engine():
    """The registered engine if it can run now, else None."""
    return _engine if _engine is not None and _engine.available() else None


def list_models():
    """The engine's models as {"id": "device:<id>", "label"} (empty without an engine)."""
    e = engine()
    return [{"id": PREFIX + m["id"], "label": f"{m['label']} - on this device"} for m in e.models()] if e else []


def complete(model, system, messages, max_tokens):
    """The answer of the engine's model (`model` with its "device:" prefix) -> {"text", "usage"}; ollama.LLMError (readable) if it fails."""
    from maestro.feedback.ollama import LLMError
    e = engine()
    if e is None:
        raise LLMError("The model on this device is not available right now.")
    try:
        text = e.generate(model[len(PREFIX):], system, messages, max_tokens)
    except Exception as exc:                                    # an engine's failure is the user's to read, never a crash
        raise LLMError(f"The model on this device could not answer ({exc}).") from None
    if not text or not text.strip():
        raise LLMError("The model on this device returned no answer.")
    return {"text": text, "usage": {"input": 0, "output": 0}}
