# Evaluate intonation based on processed data
#
# Compares aligned pitch (audio/pitch.py -> audio/transcription.py) against score pitch
# (score/symbolic_features.py), matched by alignment/align_audio_score.py. Result goes to
# Performance.assessment["intonation"], which feedback/generator.py, gui/ and assessment/synthesize.py read.
#
# Returns the common event format of assessment/events.py: one event per note that is out of tune (> CENTS_OK) or
# wrong, each with its probable causes. Causes use only measurements from the earlier analysis: the player's overall
# tuning offset, the interval leading into the note (shifts), drift of the deviation over the performance, the register,
# and, when a pose exists, whether the left hand's position from the video (vision/visual_pitch.py) agrees with
# the note. The legacy keys (notes, n_wrong_notes, tuning_offset_cents, mean_abs_cents) are kept for charts.
import numpy as np

from maestro.assessment.events import Context, cause, make_event, module_result, ramp
from maestro.utils.audio_math import cents_between

CENTS_GOOD, CENTS_OK = 10, 25      # |pitch error| in cents: in tune / a bit off / out of tune (shared by every chart)
CENTS_SEVERE = 50                  # severity 1.0 from here
LEAP_SEMITONES = 3                 # an interval at least this large into the note counts as a shift / leap
HIGH_MIDI = 81                     # A5: 4th position and above on the E string, where finger spacing is tight


def _drift(notes, ctx):
    """Slope of the deviation over the performance in cents per minute, and how much of the variance it explains."""
    ok = [(ctx.notes[n["perf_idx"]]["onset"], n["cents"]) for n in notes if not n["wrong_note"]]
    t, c = np.array(ok).T if ok else (np.zeros(0), np.zeros(0))
    if len(ok) < 8 or np.ptp(c) < 1e-9:                        # too few notes, or no deviation to explain
        return 0.0, 0.0
    slope = np.polyfit(t, c, 1)[0]
    return float(slope * 60), float(np.corrcoef(t, c)[0, 1] ** 2)


def _visual_disagreement(ctx, score_note, time):
    """Semitones between the note and the pitch the video says the left hand is at (nearest string), or None.
    The hand estimate is coarse (about a semitone), so only a large mismatch is evidence."""
    vp = ctx.assessment.get("visual_pitch")
    if not vp or not vp.get("available"):
        return None
    t = np.asarray(vp["times"])
    cand = np.asarray(vp["candidates"])[int(np.argmin(np.abs(t - time)))]
    return float(np.min(np.abs(cand - score_note["midi_pitch"])))


def _causes(note, ctx, offset, drift, drift_r2, preceding_leap):
    s, cents = ctx.score[note["score_idx"]], note["cents"]
    out = []
    if offset is not None and abs(offset) >= 12 and np.sign(offset) == np.sign(cents):
        out.append(cause("global_tuning_offset",
                         f"The whole performance sits {abs(offset):.0f} cents {'sharp' if offset > 0 else 'flat'}: "
                         "the instrument or reference pitch is probably off, not this finger.",
                         min(1, abs(offset) / abs(cents)), tuning_offset_cents=offset))
    if preceding_leap is not None and abs(preceding_leap) >= LEAP_SEMITONES and np.sign(preceding_leap) == np.sign(cents):
        out.append(cause("shift_overshoot",
                         f"The note follows a {abs(preceding_leap):.0f}-semitone {'up' if preceding_leap > 0 else 'down'} "
                         f"leap and lands {'sharp' if cents > 0 else 'flat'}: the shift overshot its target.",
                         0.55, leap_semitones=preceding_leap))
    elif preceding_leap is not None and abs(preceding_leap) >= LEAP_SEMITONES:
        out.append(cause("shift_undershoot",
                         f"The note follows a {abs(preceding_leap):.0f}-semitone leap and lands on the other side of "
                         "the target than the leap's direction: the shift stopped short or the hand corrected late.",
                         0.35, leap_semitones=preceding_leap))
    if abs(drift_r2) > 0.15 and abs(drift) > 10:
        out.append(cause("pitch_drift",
                         f"Pitch drifts {drift:+.0f} cents per minute over the performance (strings settling, or "
                         "the hand losing its reference).", min(0.6, 2 * drift_r2), cents_per_minute=drift, r2=drift_r2))
    if s["midi_pitch"] >= HIGH_MIDI:
        out.append(cause("high_position", "The note is in a high position where the finger spacing is very small.",
                         0.4, midi=s["midi_pitch"]))
    gap = _visual_disagreement(ctx, s, ctx.notes[note["perf_idx"]]["onset"])
    if gap is not None and gap >= 2.0:
        out.append(cause("hand_position_mismatch",
                         f"The left hand in the video is {gap:.1f} semitones away from where this note lies on any "
                         "string.", min(0.7, gap / 5), visual_gap_semitones=gap))
    if not out:
        out.append(cause("finger_placement", "No other pattern explains it: the finger was placed slightly off "
                         "the correct spot.", 0.3))
    return out


def assess_intonation(score_notes, perf_notes, alignment, wrong_note_semitones=1.0, ctx=None):
    """Per-note pitch deviation in cents (+ sharp / - flat), a summary and events with causes.

    Matches further than `wrong_note_semitones` from the written pitch are flagged "wrong_note"
    (almost certainly a wrong or mis-transcribed note, not an intonation error) and excluded from the
    summary. `tuning_offset_cents` (median deviation) reveals a consistently sharp/flat player;
    `mean_abs_cents` is the average absolute deviation of the rest. `ctx` (events.Context) adds audio/pose evidence.
    """
    ctx = ctx or Context.from_parts(score_notes, perf_notes, alignment)
    notes = []
    for a in alignment:
        if a["perf_idx"] is None:
            continue
        s, p = score_notes[a["score_idx"]], perf_notes[a["perf_idx"]]
        cents = float(cents_between(p["hz"], s["hz"]))
        notes.append({"score_idx": a["score_idx"], "perf_idx": a["perf_idx"], "expected_hz": s["hz"],
                      "actual_hz": p["hz"], "cents": cents, "wrong_note": abs(cents) > wrong_note_semitones * 100})
    ok = np.array([n["cents"] for n in notes if not n["wrong_note"]])
    offset = float(np.median(ok)) if ok.size else None
    drift, drift_r2 = _drift(notes, ctx)

    events = []
    for n in notes:
        s, p = score_notes[n["score_idx"]], perf_notes[n["perf_idx"]]
        if n["wrong_note"]:
            events.append(make_event("intonation", "wrong_note", p["onset"], p.get("offset", p["onset"]), 1.0,
                                     f"Played {p['midi']:.1f} instead of {s['midi_pitch']} ({s.get('pitch', s['midi_pitch'])}).",
                                     [cause("wrong_note", "The pitch is over a semitone from the written note: a wrong "
                                            "finger or string, or a misread note.", 0.6)],
                                     ctx, n["score_idx"], n["perf_idx"], n["cents"], 0, "cents"))
        elif abs(n["cents"]) > CENTS_OK:
            leap = (s["midi_pitch"] - score_notes[n["score_idx"] - 1]["midi_pitch"]) if n["score_idx"] > 0 else None
            events.append(make_event(
                "intonation", "sharp" if n["cents"] > 0 else "flat", p["onset"], p.get("offset", p["onset"]),
                ramp(abs(n["cents"]), CENTS_OK - 15, CENTS_SEVERE) * 0.8 + 0.2,
                f"{s.get('pitch', s['midi_pitch'])} is {abs(n['cents']):.0f} cents {'sharp' if n['cents'] > 0 else 'flat'}.",
                _causes(n, ctx, offset, drift, drift_r2, leap), ctx, n["score_idx"], n["perf_idx"],
                n["cents"], 0, "cents"))
    summary = {"n_notes": len(notes), "n_wrong_notes": sum(n["wrong_note"] for n in notes),
               "tuning_offset_cents": offset, "mean_abs_cents": float(np.abs(ok).mean()) if ok.size else None,
               "drift_cents_per_minute": drift}
    result = module_result("intonation", events, len(notes), summary)
    result.update({"notes": notes, "n_wrong_notes": summary["n_wrong_notes"], "tuning_offset_cents": offset,
                   "mean_abs_cents": summary["mean_abs_cents"]})
    return result
