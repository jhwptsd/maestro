# Generates written feedback from an LLM (rewritten from the outline).
#
# `generate_feedback(perf)` builds the digest of the analysis (prompts.build_digest), sends it with the system prompt to the
# provider the user connected (providers.py, settings.py: OpenRouter or Groq, through the user's own key or sign-in) and stores the
# result in Performance.feedback:
#   {"provider", "model", "summary", "priorities": [{"title", "bars", "why", "practice"}], "strengths": [...],
#    "warnings": [...], "usage": {"input", "output"}}
# Only the digest leaves the machine (numbers, bar numbers, short labels). The model's JSON is parsed leniently, and every bar it
# cites is checked against the bars that actually have problems in the digest: an invented bar is dropped from the list and noted in
# "warnings", so the feedback cannot point the student at a place the analysis never flagged. gui/server.py calls this on request
# (the analysis itself never contacts a provider), as does `analyze.py --feedback`.
import json
import re

from maestro.feedback import prompts, providers, settings

MAX_TOKENS = 1500


def _parse(text):
    """The model's answer -> dict, tolerating markdown fences and prose around the JSON; None if there is no JSON object."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    text = fenced.group(1) if fenced else text
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def known_bars(digest):
    """Bar numbers the digest mentions (events and the worst-bar list): the only ones feedback may cite."""
    return ({e["bar"] for e in digest["problem_events"] if e["bar"] is not None}
            | {b["bar"] for b in digest["bars_with_most_problems"]})


def shape(answer, digest):
    """The model's parsed answer -> the feedback dict (strings trimmed, at most 3 priorities, unknown bars removed + warned)."""
    allowed, warnings = known_bars(digest), []
    priorities = []
    for p in (answer.get("priorities") or [])[:3]:
        if not isinstance(p, dict):
            continue
        bars = [b for b in (p.get("bars") or []) if isinstance(b, int) and not isinstance(b, bool)]
        invented = sorted(set(bars) - allowed)
        if invented:
            warnings.append(f"removed bar(s) {invented} cited under '{str(p.get('title', ''))[:40]}': the analysis flagged no problem there")
        priorities.append({"title": str(p.get("title", "")).strip(), "bars": [b for b in bars if b in allowed],
                           "why": str(p.get("why", "")).strip(), "practice": str(p.get("practice", "")).strip()})
    strengths = [str(s).strip() for s in (answer.get("strengths") or []) if str(s).strip()][:3]
    return {"summary": str(answer.get("summary", "")).strip(), "priorities": priorities, "strengths": strengths, "warnings": warnings}


def resolve(provider=None, model=None, key=None, base_url=None):
    """The provider to call -> (Provider, key, model, name). Anything not given comes from the saved settings; LLMError (readable)
    if nothing is connected or the model is unknown."""
    chosen = provider or settings.status()["provider"]
    if not chosen:
        raise providers.LLMError("No AI provider is connected: add an API key for Groq or OpenRouter (or sign in to OpenRouter) first.")
    p = providers.get(chosen, base_url)
    key = key or settings.get_key(chosen)
    if not key:
        raise providers.LLMError(f"{p.label} is not connected: add its API key first.")
    model = model or settings.load()["models"].get(chosen) or p.default_model
    if not model:
        raise providers.LLMError(f"Choose a model for {p.label} first.")
    return p, key, model, chosen


def remember_model(chosen, model):
    """Save `model` as the one to use for `chosen` after a fallback (silently nothing if that provider is not saved)."""
    try:
        settings.choose(chosen, model)
    except providers.LLMError:
        pass


def feedback_from_digest(digest, provider=None, model=None, key=None, base_url=None):
    """Ask the provider for feedback on a digest -> the feedback dict. provider/model/key default to the saved settings.
    If the model is busy or not available to the key another one is tried and remembered (noted in "warnings").
    Raises providers.LLMError (readable message) if no provider is connected or the call fails."""
    p, key, model, chosen = resolve(provider, model, key, base_url)
    reply, used, note = providers.complete_with_fallback(p, key, model, prompts.SYSTEM,
                                                         [{"role": "user", "content": prompts.user_message(digest)}], MAX_TOKENS, json_mode=True)
    if note:
        remember_model(chosen, used)
    answer = _parse(reply["text"])
    if answer is None:                                           # not JSON: show the text as it is rather than nothing
        result = {"summary": reply["text"].strip(), "priorities": [], "strengths": [],
                  "warnings": ["the model did not answer in the expected format; its text is shown as it came"]}
    else:
        result = shape(answer, digest)
    if note:
        result["warnings"].append(note)
    return {"provider": chosen, "model": used, **result, "usage": reply["usage"]}


def generate_feedback(perf, **options):
    """Feedback for a Performance (after analyze()) -> the feedback dict, also stored in perf.feedback."""
    perf.feedback = feedback_from_digest(prompts.build_digest(perf), **options)
    return perf.feedback
