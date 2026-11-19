# Evaluate bowing based on processed data
#
# Uses right-hand motion from vision/ aligned to score articulations. Writes to
# Performance.assessment.
# Use Bowing Gestures Classification in Violin Performance: A Machine Learning Approach (Dalmazzo & Ramírez, 2019)
#
# Input: the bow strokes found by vision/motion_features.py (right-hand travel along the bow axis). Output:
# assessment["bowing"] in the common event format of assessment/events.py. The summary metrics use the pose alone
# (how steady and balanced the bow arm is); the events additionally use the aligned notes (via events.Context):
#   uneven_bow_speed / bow_speed_wobble   a stroke much faster/slower than the player's norm, or jerky inside
#   unbalanced_bow_use                     down- and up-bows travel different distances (running out of bow)
#   bow_change_early / bow_change_late     a bow direction change that does not coincide with the note onset it belongs to
# Each stroke also gets a technique label (detache / legato / staccato) from the notes it carries: the Dalmazzo &
# Ramirez (2019) gesture classes, here by rule since their model needs wearable sensors and is not released.
# Evenness metrics are coefficients of variation (spread / mean: 0 = perfectly even), so they are unit-free.
# Stroke length varies with note length in real music, so evenness is judged on bow *speed* per stroke.
import numpy as np

from maestro.assessment.events import Context, cause, make_event, module_result, ramp

SPEED_Z = 4.0            # robust z-score of a stroke's mean speed beyond which it is flagged (note lengths vary, so speeds do)
WOBBLE_CV = 0.6          # speed CV inside a stroke beyond which it is jerky, and above the player's own norm (WOBBLE_MADS)
WOBBLE_MADS = 3.0
BALANCE_MIN = 0.7        # shorter / longer mean stroke length across directions below this is unbalanced
SYNC_WINDOW_S = 0.25     # a bow change this close to an onset (and mutually nearest, see Context.bow_change_pairs) is its bow change
SYNC_LAG_S = 0.12        # ... and counts as misaligned when it is off by more than this after the constant video/audio offset is removed


def _cv(values):
    values = np.asarray(values, dtype=float)
    return float(values.std() / values.mean()) if len(values) >= 2 and values.mean() > 0 else None


def _technique(stroke, ctx):
    """detache / legato / staccato from the performed notes that start inside the stroke (None if there are none)."""
    inside = [ctx.notes[p] for _, p in ctx.matches if stroke["start"] - 0.05 <= ctx.notes[p]["onset"] <= stroke["end"]]
    if not inside:
        return None
    if len(inside) > 1:
        return "legato"
    n = inside[0]
    slot = stroke["end"] - stroke["start"]
    return "staccato" if "offset" in n and (n["offset"] - n["onset"]) < 0.5 * slot else "detache"


def _events(strokes, motion, ctx):
    events, fps = [], motion["fps"]
    speeds = np.array([s["mean_speed"] for s in strokes])
    med = np.median(speeds)
    mad = max(float(np.median(np.abs(speeds - med)) * 1.4826), 0.05 * med, 1e-9)
    speed = np.abs(motion["bow"]["v"])
    cvs = []                                                    # speed CV inside each stroke, to find the player's norm
    for s in strokes:
        a, b = int(s["start"] * fps), int(s["end"] * fps)
        inner = speed[a + (b - a) // 5:b - (b - a) // 5]
        cvs.append(inner.std() / inner.mean() if len(inner) >= 6 and inner.mean() > 0 else np.nan)
    wobble_limit = max(WOBBLE_CV, float(np.nanmedian(cvs) + WOBBLE_MADS * 1.4826 * np.nanmedian(np.abs(cvs - np.nanmedian(cvs))))) \
        if not np.isnan(cvs).all() else WOBBLE_CV
    for k, s in enumerate(strokes):
        z = (s["mean_speed"] - med) / mad
        if abs(z) > SPEED_Z:
            fast = z > 0
            events.append(make_event(
                "bowing", "bow_too_fast" if fast else "bow_too_slow", s["start"], s["end"],
                ramp(abs(z), SPEED_Z, 2 * SPEED_Z) * 0.7 + 0.2, f"Stroke speed is {s['mean_speed'] / med:.1f}x the player's median.",
                [cause("note_length_mismatch" if fast else "bow_hoarding",
                       "The bow is used up quickly (fast stroke): more bow than the note needs." if fast else
                       "Very little bow is used (slow stroke): the stroke stalls or the note is long.", 0.4,
                       speed_ratio=float(s["mean_speed"] / med))], ctx, value=float(s["mean_speed"]), expected=float(med), unit="mm/s"))
        if cvs[k] > wobble_limit:
            cv = float(cvs[k])
            events.append(make_event("bowing", "bow_speed_wobble", s["start"], s["end"], ramp(cv, wobble_limit, 1.5) * 0.7 + 0.2,
                                     f"The bow speed varies strongly inside the stroke (CV {cv:.2f}).",
                                     [cause("jerky_bow_arm", "The bow arm accelerates and stops within the stroke: tension "
                                            "in the arm or an unprepared bow change.", 0.5, speed_cv=cv)],
                                     ctx, value=cv, expected=WOBBLE_CV, unit="CV"))
    lengths = {d: [s["length"] for s in strokes if s["dir"] == d] for d in (1, -1)}
    if all(lengths.values()):
        balance = min(np.mean(lengths[1]), np.mean(lengths[-1])) / max(np.mean(lengths[1]), np.mean(lengths[-1]))
        if balance < BALANCE_MIN:
            short = "up" if np.mean(lengths[-1]) < np.mean(lengths[1]) else "down"
            events.append(make_event("bowing", "unbalanced_bow_use", strokes[0]["start"], strokes[-1]["end"],
                                     ramp(balance, BALANCE_MIN, 0.3) * 0.6 + 0.2,
                                     f"{short.capitalize()}-bows are {balance:.0%} as long as the other direction.",
                                     [cause("running_out_of_bow", f"{short.capitalize()}-bows travel less: the bow "
                                            "drifts toward one end and strokes get cut short.", 0.5, balance=float(balance))],
                                     ctx, value=float(balance), expected=1.0, unit="length ratio"))
    offset = ctx.bow_sync_offset()
    for p, raw in ctx.bow_change_pairs().items():
        onset, lag = ctx.notes[p]["onset"], raw - offset
        if abs(lag) > SYNC_LAG_S:
            late = lag > 0
            events.append(make_event(
                "bowing", "bow_change_late" if late else "bow_change_early", onset, onset,
                ramp(abs(lag), SYNC_LAG_S, SYNC_WINDOW_S) * 0.7 + 0.2,
                f"The bow changes {abs(lag) * 1000:.0f} ms {'after' if late else 'before'} the note sounds.",
                [cause("bow_behind_fingers" if late else "bow_ahead_of_fingers",
                       "The bow arm lags the left hand." if late else "The bow arm anticipates the left hand "
                       "(the old note is cut short or the new one is pre-sounded).", 0.5, lag_s=lag)],
                ctx, None, p, lag, 0.0, "s"))
    return events


def assess_bowing(motion, ctx=None):
    """Bow strokes -> consistency summary and events ({"available": False, "reason"} if fewer than two strokes).

    `ctx` (events.Context) adds the note-aligned events and technique labels; without it only the pose-only
    ones (speed, wobble, balance) are produced."""
    bow = motion.get("bow")
    strokes = bow["strokes"] if bow else []
    if len(strokes) < 2:
        return {"available": False, "reason": "fewer than two bow strokes found (right hand not tracked or not moving)",
                "module": "bowing", "score": None, "summary": {}, "events": []}
    ctx = ctx or Context.from_parts([], [], [])
    fps = motion["fps"]
    lengths = np.array([s["length"] for s in strokes])
    durations = np.array([s["end"] - s["start"] for s in strokes])
    speed = np.abs(bow["v"])
    within = []                                # speed variation inside a stroke, away from its turnarounds
    for s in strokes:
        a, b = int(s["start"] * fps), int(s["end"] * fps)
        middle = speed[a + (b - a) // 5:b - (b - a) // 5]
        if len(middle) >= 3 and middle.mean() > 0:
            within.append(middle.std() / middle.mean())
    by_direction = [np.mean([s["length"] for s in strokes if s["dir"] == d]) for d in (1, -1) if any(s["dir"] == d for s in strokes)]
    events = _events(strokes, motion, ctx)
    techniques = [_technique(st, ctx) for st in strokes]
    summary = {"n_strokes": len(strokes), "strokes_per_second": float(len(strokes) / (strokes[-1]["end"] - strokes[0]["start"])),
               "mean_stroke_length": float(lengths.mean()), "stroke_length_cv": _cv(lengths),
               "stroke_duration_cv": _cv(durations), "mean_speed": float(np.mean([s["mean_speed"] for s in strokes])),
               "stroke_speed_cv": _cv([s["mean_speed"] for s in strokes]),
               "speed_cv_within_stroke": float(np.mean(within)) if within else None,
               "direction_balance": float(min(by_direction) / max(by_direction)) if len(by_direction) == 2 else None,
               "direction_source": bow["axis_source"], "audio_video_offset_s": ctx.bow_sync_offset(),
               "techniques": {t: techniques.count(t) for t in ("detache", "legato", "staccato")}}
    result = module_result("bowing", sorted(events, key=lambda e: e["start"]), len(strokes), summary)
    result.update({"available": True, "strokes": strokes, "stroke_techniques": techniques, **{k: v for k, v in summary.items() if k != "techniques"}})
    return result
