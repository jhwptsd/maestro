# Prompts for the feedback model, and the digest of the analysis they carry (rewritten from the outline).
#
# `build_digest(perf)` condenses Performance.assessment (the synthesized events of assessment/synthesize.py and each module's
# summary) into a small JSON-able dict: the only thing about the performance that is sent to the model, so it contains
# numbers, bar numbers and short labels but no audio, video, score file, file names or personal data. `SYSTEM` tells the model
# what it is and the rules it must follow; `user_message` wraps a digest. Used by feedback/generator.py.
import json

SYSTEM = """You are an experienced, encouraging violin teacher writing feedback for a student, from measurements of one recording of \
their playing compared with the score. The measurements are given as JSON.

Rules:
- Use ONLY the measurements given. Never invent a bar, time, number, fault or strength that is not in them. If something was not \
measured (see "unavailable" and "limits"), do not comment on it except to say it was not assessed.
- Refer to places by bar number (and beat when given). Times are seconds in the recording.
- "severity" runs 0 (trivial) to 1 (severe). "causes" are hypotheses: each has a confidence from 0 to 1; say "probably" or \
"may be" unless it is above 0.7, and never present a cause as certain. "weight" in top_causes adds up severity x confidence over \
the events a cause leads: a high weight is a habit that explains many problems.
- Choose the problems that matter most (high severity, many events, or a cause that explains many events), \
one practice suggestion each that a student can do today (a concrete exercise, with the tempo or bar range), not generic advice.
- Mention genuine strengths from the scores and the absence of problems (a skill with a high score and few events).
- Speak to the student ("you"), warmly but honestly, in plain language; no jargon without a short explanation; no grades or \
percentages unless they help. Do not mention JSON, "scores", "events" or "the data".
- Keep the whole answer short: a student should read it in about a minute.
- Do not summarize the data or simply repeat what you are told: analyze it and look for probable causes and solutions.
- Focus more on how to fix the error than what the error is: recommend exercises or ways to practice.
- Events with skill "reference" record how the student differs from a professional's recording of the same bars: "value" is the DIFFERENCE (the \
student minus the professional, in "unit"; "expected" is 0), so a large value is a clear difference. When such events exist, the feedback MUST \
compare the student with the professional recording: name it "the professional recording" in the summary, and make at least one priority about \
the biggest differences (the bars, what differs, and a practice such as listening to the professional's version of those bars and trying their reading). \
They are not faults: their cause "soloist_interpretive_choice" says the professional made a musical decision (tempo, dynamics, articulation, vibrato, \
bowing, expressive intonation), so present a difference as another reading to try, never as a mistake. Wrong notes, intonation, rhythm and tone \
problems still come first among the priorities. "reference.soloist_choices" lists the professional's decisions in general, and "reference.gaps" what \
could not be measured (say nothing about those).

Reply with JSON only, no markdown fences, exactly in this shape:
{"summary": "2-3 sentences overall",
 "priorities": [{"title": "short", "bars": [bar numbers the problem is in], "why": "what is going wrong and the likely cause", \
"practice": "what to do"}],
 "strengths": ["short sentence", ...]}"""

MAX_EVENTS = 30                    # events listed in the digest (the most severe); the rest are only counted
MAX_BARS, MAX_CAUSES, MAX_LINKS, MAX_CHOICES, MAX_REFERENCE_EVENTS = 10, 6, 5, 8, 8


def _round(x, digits=2):
    return round(x, digits) if isinstance(x, float) else x


def _clean(d):
    """A flat dict of the scalars of `d` (floats rounded, None dropped); one level of nested dicts is flattened as key_subkey."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update({f"{k}_{k2}": _round(v2) for k2, v2 in v.items() if v2 is not None and not isinstance(v2, (dict, list))})
        elif v is not None and not isinstance(v, list):
            out[k] = _round(v)
    return out


def build_digest(perf):
    """The analysis of `perf` as a compact dict (see the module comment); needs perf.assessment["synthesis"]."""
    a, synthesis = perf.assessment, perf.assessment["synthesis"]
    events = synthesis["events"]
    section, rhythm = a.get("section") or {}, a.get("rhythm") or {}
    counts = {}
    for e in events:
        counts.setdefault(e["module"], {}).setdefault(e["category"], 0)
        counts[e["module"]][e["category"]] += 1
    by_severity = lambda es: sorted(es, key=lambda e: -e["severity"])
    top = (by_severity([e for e in events if e["module"] != "reference"])[:MAX_EVENTS]       # (the comparison events are informational: they get
           + by_severity([e for e in events if e["module"] == "reference"])[:MAX_REFERENCE_EVENTS])   # their own slots, or they would never rank)
    bars = sorted(synthesis["by_measure"].items(), key=lambda kv: -kv[1]["max_severity"])[:MAX_BARS]
    digest = {
        "piece": _clean({"bars_played_from": section.get("first_measure"), "bars_played_to": section.get("last_measure"),
                         "share_of_score_played": section.get("coverage"), "seconds": len(perf.audio["samples"]) / perf.audio["sr"],
                         "tempo_vs_written": rhythm.get("tempo_ratio")}),
        "scores_0_to_100": {k: _round(v, 0) for k, v in synthesis["scores"].items() if v is not None},
        "summaries": {m: _clean(a[m].get("summary") or {})
                      for m in ("intonation", "rhythm", "expression", "tone", "bowing", "posture") if a.get(m) and a[m].get("available", True)},
        "problem_events": [{"skill": e["module"], "kind": e["category"], "bar": e["measure"], "beat": e["beat"],
                            "time": _round(e["start"], 1), "severity": _round(e["severity"]),
                            "value": _round(e["value"]), "expected": _round(e["expected"]), "unit": e["unit"],
                            "causes": [[c["cause"], _round(c["confidence"])] for c in e["causes"][:2]]} for e in top],
        "event_counts": counts,
        "bars_with_most_problems": [{"bar": int(m), "worst_severity": _round(v["max_severity"]), "skills": v["modules"]} for m, v in bars],
        "top_causes": [{"cause": c["cause"], "events": c["count"], "weight": _round(c["weight"]), "skills": c["modules"]}
                       for c in synthesis["causes"][:MAX_CAUSES]],
        "simultaneous_problems": [{"time": _round(l["start"], 1), "skills": l["modules"], "kinds": l["categories"]}
                                  for l in synthesis["links"][:MAX_LINKS]],
        "unavailable": synthesis["unavailable"],
        "limits": [],
    }
    ref = (a.get("reference") or {}).get("summary") or {}
    if ref.get("clip"):
        digest["reference"] = {"bars": [ref["clip"]["first_measure"], ref["clip"]["last_measure"]],
                               "soloist_choices": [_clean({k: c.get(k) for k in ("slug", "measure", "value", "unit")}) for c in ref["choices"][:MAX_CHOICES]],
                               "gaps": {"notes_not_audible": ref["gaps"]["notes"], "bars_without_video": ref["gaps"]["bars_without_video"],
                                        "video_seconds_lost": _round(sum(g["end"] - g["start"] for g in ref["gaps"]["video"]), 1)}}
    status = a.get("visual_status") or {}
    if not status.get("available", False):
        digest["limits"].append("no video analysis (bowing, posture, left hand): " + str(status.get("reason", "not available")))
    if not events:
        digest["limits"].append("no problems were detected at all")
    digest["problem_events_shown"] = f"{len(top)} most severe of {len(events)}"
    return digest


COMPACT_EVENTS, COMPACT_BARS, COMPACT_CAUSES, COMPACT_REFERENCE_EVENTS = 10, 5, 4, 6


def compact(digest):
    """A much shorter digest for small local models (short context, easily distracted): the most severe events with one cause each,
    the worst bars and main causes, the scores and the limits; the per-skill summaries and simultaneity links are dropped."""
    return {"piece": digest["piece"], "scores_0_to_100": digest["scores_0_to_100"],
            "problem_events": [{**e, "causes": e["causes"][:1]} for e in
                               [e for e in digest["problem_events"] if e["skill"] != "reference"][:COMPACT_EVENTS]
                               + [e for e in digest["problem_events"] if e["skill"] == "reference"][:COMPACT_REFERENCE_EVENTS]],
            "event_counts": {m: sum(c.values()) for m, c in digest["event_counts"].items()},
            "bars_with_most_problems": digest["bars_with_most_problems"][:COMPACT_BARS],
            "top_causes": digest["top_causes"][:COMPACT_CAUSES], "unavailable": digest["unavailable"], "limits": digest["limits"],
            **({"reference": digest["reference"]} if "reference" in digest else {})}


def user_message(digest):
    """The user turn: the digest as JSON, with the instruction to answer in the required shape."""
    compared = any(e.get("skill") == "reference" for e in digest.get("problem_events", []))
    reminder = (" The measurements include comparison events with the professional recording: say how the student differs from the professional "
                "recording, in the summary and in one priority (after the wrong-note problems)." if compared else "")
    return ("Measurements of the student's recording:\n" + json.dumps(digest, separators=(",", ":"), ensure_ascii=False) +
            "\n\nWrite the feedback now, as the JSON described." + reminder)
