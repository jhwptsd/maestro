# Evaluate tone based on processed data
#
# Uses audio/tone_quality.py per aligned note (attack clarity, level stability, brightness, harmonic-to-noise) and,
# optionally, the neural windows of assessment/neural.py. Writes to Performance.assessment["tone"] in the common event
# format of assessment/events.py. The dimensions follow Giraldo et al. (2019), "Automatic Assessment of Tone Quality in
# Violin Music Performance" (their models are trained on TELMI recordings that are not public, so thresholds here are
# measured against the player's own median instead of a learned model). A note is flagged when it departs from the
# player's typical sound; causes come from the bow speed / stroke length at that moment (vision/motion_features.py,
# via Context.stroke_at) because on the violin tone is mostly made by the right hand: speed, weight and contact point.
import numpy as np

from maestro.assessment.events import Context, cause, make_event, module_result, ramp
from maestro.audio.tone_quality import note_tone

MIN_STABILITY_S = 0.3                # the level of a shorter note has too few frames to be called unsteady
MIN_NOTE_S = 0.15                    # shorter notes have too few frames for stable descriptors
HARMONIC_FLOOR = 0.55                # absolute: below this share of harmonic energy the tone is noisy
HARMONIC_DROP = 0.2                  # relative: this far below the player's median is also noisy
ATTACK_SLOW_S = 0.2                  # attack slower than this on a note of >= 0.4 s (and slow for this player) is unclear
STABILITY_FLOOR = 0.5                # level stability (1 = steady) below this (and low for this player) is an unsteady bow
OUTLIER_MADS = 2.5                   # ... and this many MADs beyond the player's median: shaping a note is not a fault
BRIGHT_Z = 3.0                       # brightness more than this many MADs from the median
NEURAL_BAD = 0.35                    # p(good tone) below this flags a window


def _mad(x):
    return float(np.median(np.abs(x - np.median(x))) * 1.4826) or 1e-9


def _bow_state(ctx, onset):
    """(speed relative to the player's median stroke speed, stroke) at `onset`, or (None, None) without a pose."""
    stroke = ctx.stroke_at(onset)
    strokes = ctx.strokes()
    if not stroke or len(strokes) < 4:
        return None, None
    return stroke["mean_speed"] / (np.median([s["mean_speed"] for s in strokes]) + 1e-12), stroke


def assess_tone(score_notes, perf_notes, alignment, ctx=None):
    """Per-note tone events with causes + a summary of the median descriptors. Needs audio in `ctx`."""
    ctx = ctx or Context.from_parts(score_notes, perf_notes, alignment)
    if ctx.y is None:
        return {"module": "tone", "available": False, "score": None,
                "summary": {"reason": "no audio samples in the context"}, "events": []}
    tones = {}
    for i, p in ctx.matches:
        n = ctx.notes[p]
        if n.get("offset", 0) - n["onset"] >= MIN_NOTE_S:
            t = note_tone(ctx.y, ctx.sr, n["onset"], n["offset"], n["hz"])
            if t:
                tones[i] = t
    if len(tones) < 3:
        return {"module": "tone", "available": False, "score": None,
                "summary": {"reason": "fewer than three notes long enough to measure"}, "events": []}
    med = {k: float(np.median([t[k] for t in tones.values()])) for k in ("harmonic_ratio", "brightness", "attack_s")}
    mad_bright = _mad([t["brightness"] for t in tones.values()])
    att = np.array([t["attack_s"] for t in tones.values()])
    stab = np.array([t["dynamic_stability"] for t in tones.values()])
    slow_limit = max(ATTACK_SLOW_S, float(np.median(att) + OUTLIER_MADS * _mad(att)))        # relative to this player
    unsteady_limit = min(STABILITY_FLOOR, float(np.median(stab) - OUTLIER_MADS * max(_mad(stab), 0.05)))
    events = []
    for i, t in tones.items():
        p = ctx._by_score[i]
        n = ctx.notes[p]
        span = (n["onset"], n["offset"])
        speed, _ = _bow_state(ctx, n["onset"])
        make = lambda cat, sev, causes, val, exp, unit: events.append(
            make_event("tone", cat, *span, sev, causes, ctx, i, p, val, exp, unit))
        if t["harmonic_ratio"] < HARMONIC_FLOOR or t["harmonic_ratio"] < med["harmonic_ratio"] - HARMONIC_DROP:
            causes = [cause("too_much_bow_pressure", 0.6, bow_speed_ratio=speed)] if speed is not None and speed < 0.8 else \
                     [cause("bow_too_fast_or_light", 0.6, bow_speed_ratio=speed)] if speed is not None and speed > 1.25 else \
                     [cause("bow_contact_unclear", 0.35, bow_speed_ratio=speed)]
            make("noisy_tone", ramp(t["harmonic_ratio"], HARMONIC_FLOOR + 0.1, 0.2) * 0.8 + 0.2,
                 causes, t["harmonic_ratio"], med["harmonic_ratio"], "harmonic ratio")
        if t["attack_s"] > slow_limit and n["offset"] - n["onset"] >= 0.4:
            make("slow_attack", ramp(t["attack_s"], slow_limit, slow_limit + 0.2) * 0.7 + 0.2,
                 [cause("bow_not_prepared", 0.45, attack_s=t["attack_s"])],
                 t["attack_s"], slow_limit, "s")
        if t["dynamic_stability"] < unsteady_limit and n["offset"] - n["onset"] >= MIN_STABILITY_S:
            wobble = _bow_wobble(ctx, *span)
            make("unsteady_level", ramp(t["dynamic_stability"], unsteady_limit, 0.1) * 0.7 + 0.2,
                 [wobble] if wobble else [cause("uneven_bow", 0.4,
                                                stability=t["dynamic_stability"])],
                 t["dynamic_stability"], unsteady_limit, "stability")
        z = (t["brightness"] - med["brightness"]) / mad_bright
        if abs(z) > BRIGHT_Z and abs(t["brightness"] - med["brightness"]) > 1.5:
            bright = z > 0
            make("too_bright" if bright else "too_dull", ramp(abs(z), BRIGHT_Z, 2 * BRIGHT_Z) * 0.6 + 0.2,
                 [cause("bow_near_bridge" if bright else "bow_over_fingerboard", 0.4, z=float(z))],
                 t["brightness"], med["brightness"], "harmonic numbers")
    events += _neural(ctx, tones, med)
    summary = {"n_notes": len(tones), "median": med,
               "scratchy_share": sum(e["category"] == "noisy_tone" for e in events) / len(tones)}
    return module_result("tone", sorted(events, key=lambda e: e["start"]), len(tones), summary)


def _bow_wobble(ctx, start, end):
    """Cause from bow speed variation inside the stroke carrying the note (needs the pose)."""
    bow, fps = (ctx.motion or {}).get("bow"), (ctx.motion or {}).get("fps")
    if not bow or not fps:
        return None
    speed = np.abs(bow["v"])[int(start * fps):int(end * fps)]
    if len(speed) < 6 or speed.mean() <= 0:
        return None
    cv = float(speed.std() / speed.mean())
    if cv > 0.5:
        return cause("bow_speed_wobble", 0.6,
                     speed_cv=cv)


def _neural(ctx, tones, med):
    """Windows the neural model hears as poor tone, with the measured descriptors inside them as evidence."""
    events = []
    for w in ctx.neural.get("tone", []):
        if w["p_good"] >= NEURAL_BAD:
            continue
        inside = [t for i, t in tones.items() if w["start"] <= ctx.notes[ctx._by_score[i]]["onset"] <= w["end"]]
        hr = float(np.mean([t["harmonic_ratio"] for t in inside])) if inside else None
        causes = ([cause("measured_noise",
                         0.5, harmonic_ratio=hr)] if hr is not None and hr < med["harmonic_ratio"] - 0.05 else
                  [cause("unmeasured", 0.2)])
        events.append(make_event("tone", "neural_harsh_tone", w["start"], w["end"], ramp(w["p_good"], NEURAL_BAD, 0.05), causes, ctx,
                                 None, None, w["p_good"], NEURAL_BAD, "p(good tone)"))
    return events
