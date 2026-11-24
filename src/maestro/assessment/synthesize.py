# Synthesize every assessment into one JSON report (new file).
#
# Input: Performance.assessment after assessment/{intonation,rhythm,expression,tone,bowing,posture,reference}.py have run (each a
# {"module", "available", "score", "summary", "events"} dict, see assessment/events.py). Output, stored as
# assessment["synthesis"] (plain JSON types, safe for json.dumps):
#   {"version", "scores": {module: 0-100 or None, "overall"}, "n_events",
#    "events": [every event, sorted by start, each with a unique "id"],
#    "causes": [{"cause", "count", "weight", "modules", "events": [ids]}, ... most weighty first]   (weight = sum of
#               severity x confidence over the events the cause is the leading one of)
#    "by_measure": {"12": {"events": n, "max_severity", "modules": [...]}, ...},
#    "links": [{"events": [id, id], "modules": [a, b], "start", "end", "categories": [a, b], "causes": [a, b]}]
#               events from different modules that overlap in time (the leading cause of each, for the feedback stage),
#    "unavailable": {module: reason}}
# gui/report.py draws it and feedback/generator.py is meant to read it, so the schema is the contract between them.
import numpy as np

from maestro.assessment.events import jsonable

MODULES = ("intonation", "rhythm", "expression", "tone", "bowing", "posture", "reference")
UNSCORED = ("reference",)      # modules whose events inform but are not faults (assessment/reference.py): no score, not counted among problem bars or causes
WEIGHTS = {"intonation": 0.25, "rhythm": 0.25, "expression": 0.2, "tone": 0.15, "bowing": 0.1, "posture": 0.05}
LINK_MARGIN_S = 0.5            # events this close in time (gap between their spans) are linked
LINK_MIN_SEVERITY = 0.4        # only substantial events are linked, otherwise every pair would be


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
                links.append({"events": [a["id"], b["id"]], "modules": [a["module"], b["module"]],
                              "start": min(a["start"], b["start"]), "end": max(a["end"], b["end"]),
                              "categories": [a["category"], b["category"]],
                              "causes": [a["causes"][0]["cause"] if a["causes"] else None,
                                         b["causes"][0]["cause"] if b["causes"] else None]})
    return links


def synthesize_assessment(assessment):
    """Merge the per-module results of `assessment` (Performance.assessment) into the single JSON report above."""
    events, scores, unavailable = [], {}, {}
    for name in MODULES:
        res = assessment.get(name)
        if not res:
            if name not in UNSCORED:                             # (an optional module that was not asked for is not "unavailable")
                unavailable[name] = "not run"
        elif not res.get("available", True):
            unavailable[name] = (res.get("summary") or {}).get("reason") or res.get("reason", "unavailable")
        else:
            if name not in UNSCORED:
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
        if e["causes"] and e["module"] not in UNSCORED:
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
        if e["measure"] is not None and e["module"] not in UNSCORED:
            m = by_measure.setdefault(str(e["measure"]), {"events": 0, "max_severity": 0.0, "modules": set()})
            m["events"] += 1
            m["max_severity"] = max(m["max_severity"], e["severity"])
            m["modules"].add(e["module"])
    for m in by_measure.values():
        m["modules"] = sorted(m["modules"])

    return jsonable({"version": 1, "scores": scores, "n_events": len(events), "events": events, "causes": ranked,
                     "by_measure": by_measure, "links": _links(events), "unavailable": unavailable})
