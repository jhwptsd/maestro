# Synthesize every assessment into one JSON report (new file).
#
# Input: Performance.assessment after assessment/{intonation,rhythm,expression,tone,bowing,posture}.py have run (each a
# {"module", "available", "score", "summary", "events"} dict, see assessment/events.py). Output, stored as
# assessment["synthesis"] (plain JSON types, safe for json.dumps):
#   {"version", "scores": {module: 0-100 or None, "overall"}, "n_events",
#    "events": [every event, sorted by start, each with a unique "id"],
#    "causes": [{"cause", "count", "weight", "modules", "events": [ids]}, ... most weighty first]   (weight = sum of
#               severity x confidence over the events the cause is the leading explanation of)
#    "by_measure": {"12": {"events": n, "max_severity", "modules": [...]}, ...},
#    "links": [{"events": [id, id], "modules", "start", "end", "hypothesis"}]    events from different modules that
#               overlap in time, with what their combination suggests,
#    "unavailable": {module: reason}}
# gui/report.py draws it and feedback/generator.py is meant to read it, so the schema is the contract between them.
import numpy as np

from maestro.assessment.events import jsonable

MODULES = ("intonation", "rhythm", "expression", "tone", "bowing", "posture")
WEIGHTS = {"intonation": 0.25, "rhythm": 0.25, "expression": 0.2, "tone": 0.15, "bowing": 0.1, "posture": 0.05}
LINK_MARGIN_S = 0.5            # events this close in time (gap between their spans) are linked
LINK_MIN_SEVERITY = 0.4        # only substantial events are linked, otherwise every pair would be
HYPOTHESES = {                 # what two modules failing at the same moment suggests
    ("bowing", "rhythm"): "The bow arm's timing is the likely source of the timing fault.",
    ("bowing", "tone"): "The bow motion (speed, length or direction change) matches the tone fault: bow technique is "
                        "the likely cause.",
    ("bowing", "expression"): "The bow motion matches the expressive fault: shaping needs a different bow speed or weight.",
    ("intonation", "posture"): "Unsteady posture coincides with out-of-tune notes: tension or movement may disturb the left hand.",
    ("posture", "rhythm"): "Body movement coincides with a timing fault: the player may be pushing the tempo with the body.",
    ("intonation", "rhythm"): "Pitch and timing faults coincide: the passage is probably too hard at this tempo.",
    ("intonation", "tone"): "Pitch and tone faults coincide: the left hand and the bow are not coordinated here.",
    ("rhythm", "tone"): "Timing and tone faults coincide: unprepared bow changes may cause both.",
}


def _gap(a, b):
    """Seconds between two events' spans (0 when they overlap)."""
    return max(0.0, max(a["start"], b["start"]) - min(a["end"], b["end"]))


def _links(events):
    strong = [e for e in events if e["severity"] >= LINK_MIN_SEVERITY]
    links = []
    for i, a in enumerate(strong):
        for b in strong[i + 1:]:
            if b["start"] - a["end"] > LINK_MARGIN_S:          # sorted by start: nothing later can be close
                break
            if a["module"] != b["module"] and _gap(a, b) <= LINK_MARGIN_S:
                pair = tuple(sorted((a["module"], b["module"])))
                links.append({"events": [a["id"], b["id"]], "modules": list(pair),
                              "start": min(a["start"], b["start"]), "end": max(a["end"], b["end"]),
                              "hypothesis": HYPOTHESES.get(pair, f"{pair[0]} and {pair[1]} faults coincide.")})
    return links


def synthesize_assessment(assessment):
    """Merge the per-module results of `assessment` (Performance.assessment) into the single JSON report above."""
    events, scores, unavailable = [], {}, {}
    for name in MODULES:
        res = assessment.get(name)
        if not res:
            unavailable[name] = "not run"
        elif not res.get("available", True):
            unavailable[name] = (res.get("summary") or {}).get("reason") or res.get("reason", "unavailable")
        else:
            scores[name] = res["score"]
            events += res["events"]
    events.sort(key=lambda e: (e["start"], e["module"]))
    for k, e in enumerate(events):
        e["id"] = f"e{k:04d}"

    have = {m: s for m, s in scores.items() if s is not None}
    overall = (sum(WEIGHTS[m] * s for m, s in have.items()) / sum(WEIGHTS[m] for m in have)) if have else None
    scores = {**{m: scores.get(m) for m in MODULES if m in scores}, "overall": None if overall is None else round(overall, 1)}

    causes = {}
    for e in events:
        if e["causes"]:
            top = e["causes"][0]
            c = causes.setdefault(top["cause"], {"cause": top["cause"], "count": 0, "weight": 0.0, "modules": set(), "events": []})
            c["count"] += 1
            c["weight"] += e["severity"] * top["confidence"]
            c["modules"].add(e["module"])
            c["events"].append(e["id"])
    ranked = sorted(causes.values(), key=lambda c: -c["weight"])
    for c in ranked:
        c["weight"], c["modules"] = round(c["weight"], 3), sorted(c["modules"])

    by_measure = {}
    for e in events:
        if e["measure"] is not None:
            m = by_measure.setdefault(str(e["measure"]), {"events": 0, "max_severity": 0.0, "modules": set()})
            m["events"] += 1
            m["max_severity"] = max(m["max_severity"], e["severity"])
            m["modules"].add(e["module"])
    for m in by_measure.values():
        m["modules"] = sorted(m["modules"])

    return jsonable({"version": 1, "scores": scores, "n_events": len(events), "events": events, "causes": ranked,
                     "by_measure": by_measure, "links": _links(events), "unavailable": unavailable})
