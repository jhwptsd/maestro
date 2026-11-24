# The comparison with a professional recording as assessment events (new file).
#
# reference/compare.py measures the student against the reference soloist; this turns the differences into events in the common format
# (assessment/events.py), the same shape as every other module's, so synthesize.py, the player and the feedback model all take the comparison the
# way they take everything else: what (category, the student's value against the soloist's), when (start/end, bar, beat) and why (cause slugs
# with the measurements they rest on).
#
# What a professional does is not an answer key, so these events are informational, never faults:
#   - only interpretive categories become events (tempo, dynamics, articulation, vibrato, bowing, and a pitch where the soloist chose expressive
#     intonation). Accuracy (pitch, attack) is already judged against absolute targets by the intonation and tone modules, and the comparison
#     table keeps the accuracy figures.
#   - the leading cause of every event is "soloist_interpretive_choice" (confidence 1: it is a fact that this is the soloist's decision, not a
#     guess), its evidence the two values and, when the soloist's own choice list names it (reference/baseline.reference_choices), that slug.
#   - the severity is the size of the difference scaled to at most MAX_SEVERITY, so these never outrank real problems; the module has no score
#     (synthesize.UNSCORED) and its events are not counted among the problem bars or the leading causes.
# Differences smaller than MIN_DIFFERENCE are not events. Gaps stay gaps: nothing is compared where the soloist could not be measured.
from maestro.assessment.events import cause, make_event, module_result, ramp, unavailable

MAX_SEVERITY = 0.3
MIN_DIFFERENCE = {"tempo": 0.08, "dynamics": 3.0, "articulation": 0.25, "vibrato": 8.0, "bowing": 1.0, "pitch": 10.0}   # in each row's unit; tempo relative
SCALE = 4.0                      # a difference this many times the minimum has the full severity


def _size(row):
    """How big a row's difference is, in units of its minimum (tempo as a relative difference)."""
    delta = abs(row["delta"])
    if row["category"] == "tempo":
        delta /= row["reference"] or 1.0
    return delta / MIN_DIFFERENCE[row["category"]]


def assess_reference(comparison, ctx):
    """The comparison (reference/compare.compare_to_reference) -> a module result with the events described above, or an unavailable one."""
    if not comparison.get("available"):
        return unavailable("reference", comparison.get("reason", "no reference"))
    named = {}                                                   # the soloist's own choices by (category, bar or note): evidence for the cause
    for c in comparison["reference_choices"]:
        named[(c["category"], c.get("measure"), c.get("score_idx"))] = c["slug"]
    events = []
    for row in comparison["differences"]:
        category = row["category"]
        if category not in MIN_DIFFERENCE or (category == "pitch" and not row["reference_choice"]):
            continue                                             # accuracy is judged elsewhere; only the soloist's choices are listed here
        size = _size(row)
        if size < 1.0:
            continue
        slug = named.get((category, row.get("measure"), row.get("score_idx"))) or (named.get((category, None, None)) if category != "pitch" else None)
        evidence = {"student": row["student"], "soloist": row["reference"], "difference": row["delta"]}
        if slug:
            evidence["soloist_choice"] = slug
        name = "expressive_intonation" if category == "pitch" else category
        events.append(make_event("reference", f"{name}_vs_reference", row["start"], row["end"], MAX_SEVERITY * ramp(size, 1.0, SCALE),
                                 [cause("soloist_interpretive_choice", 1.0, **evidence)], ctx, score_idx=row.get("score_idx"),
                                 value=row["student"], expected=row["reference"], unit=row["unit"]))
    summary = {"source": {k: comparison["source"][k] for k in ("origin", "title")}, "clip": comparison["clip"], "by_category": comparison["summary"],
               "gaps": comparison["gaps"], "choices": comparison["reference_choices"]}
    result = module_result("reference", events, len(comparison["differences"]), summary)
    result["score"] = None                                       # a professional's reading is not a standard to be scored against
    return result
