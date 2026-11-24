# Compare a student's performance with the reference baseline (rewritten from the outline).
#
# `compare_to_reference(perf)` measures the student the same way the reference was measured (reference/baseline.profile) and sets the two side
# by side, note by note and bar by bar, for assessment/ and feedback/ to cite. Nothing here is a grade: every difference carries its nature
# (baseline.NATURE) and whether the reference soloist visibly chose it:
#   objective     pitch accuracy and attack: the target is the same for everyone. If the reference itself departs from equal temperament by more
#                 than baseline.EXPRESSIVE_CENTS on a note, that note's pitch difference is listed as interpretive (expressive intonation), since
#                 the professional's value is then a choice and not the target.
#   interpretive  tempo, rubato, dynamics, articulation, vibrato, bowing and tone colour: differences are a different musical reading, to be
#                 cited as "the soloist chose X; you chose Y", never as errors.
# Gaps stay gaps: a note or bar the reference soloist could not be heard in, or a bar without usable video, is counted under "gaps" and not
# compared. Result (assessment/reference.py turns it into events in the common event format, stored in perf.assessment["reference"]):
#   {"available": True, "source", "clip", "differences": [{"category", "nature", "measure", "score_idx", "student", "reference", "delta", "unit",
#     "start", "end" (the student's note or bar), "reference_choice": bool}], "summary": {category: {"nature", "compared", "median_delta", "unit"}}, "gaps": {"notes", "bars_without_video",
#     "audio", "video"}, "reference_choices": [...]}, or {"available": False, "reason"}.
import numpy as np

from maestro.assessment.events import Context, jsonable
from maestro.reference import baseline
from maestro.reference.baseline import NATURE

UNITS = {"pitch": "cents", "attack": "seconds", "tempo": "x_marked_tempo", "dynamics": "db", "articulation": "share_of_gap_filled",
         "vibrato": "cents_extent", "bowing": "bow_changes", "rubato": "x_usual_tempo"}


def student_profile(perf):
    """The student's profile (same measurements as the reference's); the bow strokes come from the student's own pose analysis."""
    feats = perf.audio_features
    return baseline.profile(perf.score_features, feats["notes"], perf.alignment, perf.audio["samples"], perf.audio["sr"], feats["frames"],
                            Context(perf).strokes())


def _row(category, delta, student, reference, span, nature=None, reference_choice=False, **where):
    """One difference; `span` = (start, end) seconds in the student's performance (the note or the bar it concerns)."""
    return {"category": category, "nature": nature or NATURE[category], "student": student, "reference": reference, "delta": delta,
            "unit": UNITS[category], "reference_choice": reference_choice, "start": span[0], "end": span[1], **where}


def compare_profiles(student, ref_profile, choices):
    """Differences between two profiles -> (rows, gap counts)."""
    flagged = {c["score_idx"] for c in choices if c["slug"] == "expressive_intonation"}
    rows, skipped = [], 0
    for s, ref in ref_profile["notes"].items():
        mine = student["notes"].get(s)
        if mine is None or "gap" in mine:
            continue                                           # the student did not play it: assessment/ already reports that
        if "gap" in ref:
            skipped += 1
            continue
        choice, span = s in flagged, (mine["onset"], mine["onset"] + mine["duration"])
        rows.append(_row("pitch", mine["cents"] - ref["cents"], mine["cents"], ref["cents"], span, "interpretive" if choice else "objective", choice, score_idx=s))
        if mine.get("attack_s") is not None and ref.get("attack_s") is not None:
            rows.append(_row("attack", mine["attack_s"] - ref["attack_s"], mine["attack_s"], ref["attack_s"], span, score_idx=s))
        if mine.get("articulation") is not None and ref.get("articulation") is not None:
            rows.append(_row("articulation", mine["articulation"] - ref["articulation"], mine["articulation"], ref["articulation"], span, score_idx=s))
        mv, rv = mine["vibrato"]["extent_cents"], ref["vibrato"]["extent_cents"]
        if mv is not None and rv is not None and (mine["vibrato"]["present"] or ref["vibrato"]["present"]):
            rows.append(_row("vibrato", mv - rv, mv, rv, span, score_idx=s))
    bars_without_video = []
    for number, ref in ref_profile["measures"].items():
        mine = student["measures"].get(number)
        if mine is None:
            continue
        span = (mine["start"], mine["end"])
        if ref["tempo_ratio"] and mine["tempo_ratio"]:
            rows.append(_row("tempo", mine["tempo_ratio"] - ref["tempo_ratio"], mine["tempo_ratio"], ref["tempo_ratio"], span, measure=number))
        if ref["loudness_db"] is not None and mine["loudness_db"] is not None and student["median_loudness_db"] is not None:
            a, b = mine["loudness_db"] - student["median_loudness_db"], ref["loudness_db"] - ref_profile["median_loudness_db"]
            rows.append(_row("dynamics", a - b, a, b, span, measure=number))                     # shape within the piece: levels are not comparable across recordings
        if ref["bow_changes"] is None:
            bars_without_video.append(number)
        elif mine["bow_changes"] is not None:
            rows.append(_row("bowing", mine["bow_changes"] - ref["bow_changes"], mine["bow_changes"], ref["bow_changes"], span, measure=number))
    return rows, skipped, bars_without_video


def compare_to_reference(perf):
    """The student's differences from the reference baseline (see the module comment) -> the comparison dict, or {"available": False, "reason"}.
    assessment/reference.py turns it into events in the common format."""
    ref = perf.reference
    if ref is None:
        return {"available": False, "reason": "no reference was given"}
    rows, skipped, no_video = compare_profiles(student_profile(perf), ref["profile"], ref["choices"])
    summary = {}
    for category in sorted({r["category"] for r in rows}):
        deltas = [r["delta"] for r in rows if r["category"] == category]
        summary[category] = {"nature": NATURE[category], "compared": len(deltas), "median_delta": float(np.median(deltas)), "unit": UNITS[category]}
    return jsonable({"available": True, "source": ref["source"], "clip": ref["clip"], "differences": rows, "summary": summary,
                     "gaps": {"notes": skipped, "bars_without_video": no_video, **ref["gaps"]}, "reference_choices": ref["choices"]})
