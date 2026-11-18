# Evaluate rhythm based on processed data
#
# Compares aligned onsets (audio/onset_detection.py -> audio/transcription.py) against score
# timing (score/score_representation.py), matched by alignment/align_audio_score.py. Writes to
# Performance.assessment["rhythm"] in the common event format of assessment/events.py: one event per note whose onset
# is more than MS_OK off the steady pulse, plus one for a tempo far from the written one, each with probable causes
# read from the score (rests, leaps, short notes), the neighbouring errors (rushing/dragging) and, if a pose exists, the
# bow stroke boundary nearest the onset (vision/motion_features.py). Legacy keys (notes, tempo_ratio, ...) are kept.
import numpy as np

from maestro.assessment.events import Context, cause, make_event, module_result, ramp

MS_GOOD, MS_OK = 40, 100            # |timing error| in milliseconds: on time / a bit off / off (shared by every chart)
MS_SEVERE = 250                     # severity 1.0 from here
TEMPO_OFF = 0.15                    # tempo this far from the written one (as a ratio) is reported
FAST_NOTE_QUARTERS = 0.5            # notes this short or shorter are "fast passage" notes
LEAP_SEMITONES = 5                  # a leap this large into the note may delay it (shift / string crossing)
TREND_NOTES = 4                     # neighbours on each side that are compared to see rushing or dragging
BOW_LAG_S = 0.08                    # the bow change lagging the onset by this much is itself a timing fault


def _causes(k, errors, items, score, ctx, onset):
    """Causes for the k-th matched note (error `errors[k]` seconds, + late / - early)."""
    err = errors[k]
    si = items[k]
    s, out = score[si], []
    prev = score[si - 1] if si > 0 else None
    if prev is not None and s["onset"] - prev["offset"] > 0.1 and err > 0:
        out.append(cause("late_after_rest", f"The note comes after a {s['onset'] - prev['offset']:.2f}s rest and "
                         "enters late: the entry was not prepared.", 0.5, rest_seconds=s["onset"] - prev["offset"]))
    if prev is not None and abs(s["midi_pitch"] - prev["midi_pitch"]) >= LEAP_SEMITONES and err > 0:
        out.append(cause("late_after_leap", f"A {abs(s['midi_pitch'] - prev['midi_pitch'])}-semitone leap leads into "
                         "the note: the shift or string crossing took time.", 0.5,
                         leap_semitones=abs(s["midi_pitch"] - prev["midi_pitch"])))
    if s["duration"] <= FAST_NOTE_QUARTERS:
        out.append(cause("fast_passage", "Short notes: finger and bow coordination is tightest here.", 0.35,
                         duration_quarters=s["duration"]))
    lo, hi = max(0, k - TREND_NOTES), min(len(errors), k + TREND_NOTES + 1)
    window = errors[lo:hi]
    if len(window) >= 4 and np.sign(np.mean(window)) == np.sign(err) and abs(np.mean(window)) > MS_GOOD / 1000:
        out.append(cause("rushing" if err < 0 else "dragging",
                         f"The neighbouring notes are also {'early' if err < 0 else 'late'} by "
                         f"{abs(np.mean(window)) * 1000:.0f} ms on average: the passage {'rushes' if err < 0 else 'drags'}.",
                         0.6, neighbour_mean_error_s=float(np.mean(window))))
    boundary = _nearest_bow_change(ctx, onset)
    if boundary is not None and boundary - onset > BOW_LAG_S and abs(boundary - onset) < 0.3:
        out.append(cause("late_bow_change", f"The bow changes direction {boundary - onset:.2f}s after the note "
                         "sounds: the bow arm is behind the left hand.", 0.5, bow_change_lag_s=boundary - onset))
    if not out:
        out.append(cause("uneven_pulse", "No local pattern: the pulse is simply uneven here.", 0.3))
    return out


def _nearest_bow_change(ctx, time):
    """Time of the bow direction change closest to `time` (video), or None without a pose."""
    ends = [s["end"] for s in ctx.strokes()]
    return min(ends, key=lambda t: abs(t - time)) if ends else None


def assess_rhythm(score_notes, perf_notes, alignment, wrong_note_semitones=1.0, ctx=None):
    """Per-note timing error in seconds (+ late / - early), a summary and events with causes.

    The student's overall tempo and start offset are removed first by a least-squares line
    performed_onset = a * score_onset + b over the matched notes (wrong notes excluded), so the error
    measures *unevenness* rather than playing slower or faster than the score's tempo.
    `tempo_ratio` = 1/a (1.0 = as written, 0.8 = 20% slower), reported as its own event when far from 1.
    """
    ctx = ctx or Context.from_parts(score_notes, perf_notes, alignment)
    pairs = []
    for a in alignment:
        if a["perf_idx"] is None:
            continue
        s, p = score_notes[a["score_idx"]], perf_notes[a["perf_idx"]]
        if abs(p["midi"] - s["midi_pitch"]) <= wrong_note_semitones:
            pairs.append((a["score_idx"], a["perf_idx"], s["onset"], p["onset"]))
    if len(pairs) < 2:
        return {**module_result("rhythm", [], 0, {}, available=False), "notes": [], "tempo_ratio": None,
                "mean_abs_error": None, "std_error": None}

    so, po = np.array([p[2] for p in pairs]), np.array([p[3] for p in pairs])
    slope, intercept = np.polyfit(so, po, 1)
    err = po - (slope * so + intercept)
    items = [p[0] for p in pairs]
    events = []
    for k, (si, pi, _, onset) in enumerate(pairs):
        if abs(err[k]) * 1000 > MS_OK:
            events.append(make_event(
                "rhythm", "late" if err[k] > 0 else "early", onset, onset, 0.2 + 0.8 * ramp(abs(err[k]) * 1000, MS_OK - 60, MS_SEVERE),
                f"Onset is {abs(err[k]) * 1000:.0f} ms {'late' if err[k] > 0 else 'early'}.",
                _causes(k, err, items, score_notes, ctx, onset), ctx, si, pi, float(err[k]), 0.0, "s"))
    ratio = float(1 / slope)
    if abs(ratio - 1) > TEMPO_OFF:
        events.append(make_event("rhythm", "tempo", float(po[0]), float(po[-1]), ramp(abs(ratio - 1), TEMPO_OFF, 0.5),
                                 f"Played at {ratio:.0%} of the written tempo.",
                                 [cause("tempo_choice", "The whole piece is taken slower/faster than written (practice "
                                        "tempo, or a technical limit at the written tempo).", 0.5, tempo_ratio=ratio)],
                                 ctx, value=ratio, expected=1.0, unit="ratio"))
    summary = {"n_notes": len(pairs), "tempo_ratio": ratio, "mean_abs_error": float(np.abs(err).mean()),
               "std_error": float(err.std())}
    result = module_result("rhythm", events, len(pairs), summary)
    result.update({"notes": [{"score_idx": si, "perf_idx": pi, "error": float(e)} for (si, pi, _, _), e in zip(pairs, err)],
                   "tempo_ratio": ratio, "mean_abs_error": summary["mean_abs_error"], "std_error": summary["std_error"]})
    return result
