# Generates written feedback from an LLM (rewritten from the outline).
#
# `generate_feedback(perf)` builds the digest of the analysis (prompts.build_digest), sends it with the system prompt to the
# local model (ollama.py, settings.py) and stores the result in Performance.feedback:
#   {"model", "summary", "priorities": [{"title", "bars", "why", "practice"}], "strengths": [...],
#    "warnings": [...], "usage": {"input", "output"}}
# Only the digest goes to the model (numbers, bar numbers, short labels), and the model runs on this computer. The model's JSON is parsed leniently, and every bar it
# cites is checked against the bars that actually have problems in the digest: an invented bar is dropped from the list and noted in
# "warnings", so the feedback cannot point the student at a place the analysis never flagged. gui/server.py calls this on request
# (the analysis itself never calls the model), as does `analyze.py --feedback`.
import json
import re

from maestro.feedback import ollama, prompts, settings

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


def resolve(model=None):
    """The model to use: the one given, else the saved one, else the best installed. LLMError (readable) if Ollama is not running or
    nothing is downloaded yet."""
    model = model or settings.get_model() or ollama.pick_model(ollama.list_models())
    if not model:
        raise ollama.LLMError("No local model is downloaded yet: open the start page, \"Written feedback\", and press Download and use "
                              f"(or run: ollama pull {ollama.DEFAULT_MODEL}).")
    return model


def feedback_from_digest(digest, model=None):
    """Ask the local model for feedback on a digest -> the feedback dict. If the model is not installed another one is used (noted in
    "warnings"). Raises ollama.LLMError (readable message) if Ollama is not running or the call fails."""
    reply, used, note = ollama.complete_with_fallback(resolve(model), prompts.SYSTEM,
                                                      [{"role": "user", "content": prompts.user_message(prompts.compact(digest))}],
                                                      MAX_TOKENS, json_mode=True)
    answer = _parse(reply["text"])
    if answer is None:                                           # not JSON: show the text as it is rather than nothing
        result = {"summary": reply["text"].strip(), "priorities": [], "strengths": [],
                  "warnings": ["the model did not answer in the expected format; its text is shown as it came"]}
    else:
        result = shape(answer, digest)
    if note:
        result["warnings"].append(note)
    return {"model": used, **result, "usage": reply["usage"]}


def generate_feedback(perf, **options):
    """Feedback for a Performance (after analyze()) -> the feedback dict, also stored in perf.feedback."""
    perf.feedback = feedback_from_digest(prompts.build_digest(perf), **options)
    return perf.feedback
