# Generates the written feedback from the local model (rewritten as flowing prose).
#
# `generate_feedback(perf)` builds the digest of the analysis (prompts.build_digest), sends it with the system prompt to the local model
# (ollama.py, settings.py) and stores the result in Performance.feedback:
#   {"model", "text" (what a teacher would say: paragraphs separated by blank lines, no lists), "warnings": [...], "usage": {"input", "output"}}
# Only the digest goes to the model (numbers, bar numbers, short labels), and the model runs on this computer. The text is free prose, so the only
# check is on the bars it names: `check_bars` warns about a bar that is outside the stretch the student played or that the analysis flagged
# nothing in, so the feedback cannot send the student to a place the analysis never looked at. gui/server.py calls this on request (the
# analysis itself never calls the model), as does `analyze.py`.
import re

from maestro.feedback import ollama, prompts, settings

MAX_TOKENS = 900
BAR_MENTION = re.compile(r"\b(?:bars?|measures?)\s+((?:\d+\s*(?:-|\u2013|to|and|&|,)?\s*)+)", re.I)


def known_bars(digest):
    """Bar numbers the digest mentions (events and the worst-bar list) and the ends of the stretch played: the only ones feedback may cite."""
    played = digest["piece"]
    return ({e["bar"] for e in digest["problem_events"] if e["bar"] is not None}
            | {b["bar"] for b in digest["bars_with_most_problems"]}
            | {int(played[k]) for k in ("bars_played_from", "bars_played_to") if played.get(k) is not None}
            | set(digest.get("reference", {}).get("bars", [])))


def check_bars(text, digest):
    """Warnings for every bar number the prose names that the analysis did not flag (empty when it names none or only known ones)."""
    allowed = known_bars(digest)
    named = {int(n) for m in BAR_MENTION.finditer(text) for n in re.findall(r"\d+", m.group(1))}
    unknown = sorted(named - allowed)
    return [f"the text names bar(s) {unknown}, where the analysis flagged nothing: check them against the player"] if unknown else []


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
                                                      [{"role": "user", "content": prompts.user_message(prompts.compact(digest))}], MAX_TOKENS)
    text = reply["text"].strip()
    return {"model": used, "text": text, "warnings": check_bars(text, digest) + ([note] if note else []), "usage": reply["usage"]}


def generate_feedback(perf, **options):
    """Feedback for a Performance (after analyze()) -> the feedback dict, also stored in perf.feedback."""
    perf.feedback = feedback_from_digest(prompts.build_digest(perf), **options)
    return perf.feedback
