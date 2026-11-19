# Evaluate expression based on processed data
#
# Uses dynamics (RMS, audio/tone_quality.py), vibrato (audio/vibrato.py), note length vs articulation and, optionally,
# the neural judgement of assessment/neural.py, against the score's markings (Note.dynamic, .hairpin, .articulation from
# score/musicxml.py). Writes to Performance.assessment["expression"] in the common event format of assessment/events.py:
# how far each aspect is from what the score asks for ("severity"), and the likely cause, taken from the bow motion
# (vision/) when a pose exists:
#   dynamics      loudness per note does not follow the written p..f marks (a level is judged against the player's own
#                 dB-per-step scale, since absolute loudness depends on the microphone), flat or wrong-way hairpins
#   vibrato       missing on long notes, too slow/fast, too wide/narrow, or uneven
#   articulation  staccato/accent/legato marks that are not played as written (note length vs the gap to the next note)
#   expressiveness  windows the neural model finds mechanical (only with neural enabled)
import numpy as np

from maestro.assessment.events import Context, cause, make_event, module_result, ramp
from maestro.audio.tone_quality import note_tone
from maestro.audio.vibrato import note_vibrato

LEVELS = {"ppp": 0, "pp": 1, "p": 2, "mp": 3, "mf": 4, "f": 5, "ff": 6, "fff": 7}
MIN_DB_PER_LEVEL = 2.0           # a player whose dynamic marks sound less than this far apart per step plays flat
HAIRPIN_MIN_DB = 1.0             # a crescendo / diminuendo should change the loudness at least this much
SUSTAINED_S = 0.6                # notes at least this long are expected to carry vibrato
VIBRATO_RATE_HZ = (4.5, 8.0)     # comfortable violin vibrato rates
VIBRATO_EXTENT_CENTS = (12, 45)  # half peak-to-peak
STACCATO_RATIO, LEGATO_RATIO = 0.6, 0.85     # sounding length / gap to the next onset
NEURAL_BAD = 0.35                # p(expressive) below this marks a window as mechanical


def _bow_cause(ctx, start, end, what):
    """Cause hypothesis from the bow strokes overlapping [start, end]: is the bow speed (hence loudness/colour) varying?"""
    speeds = [s["mean_speed"] for s in ctx.strokes()]
    inside = [s for s in ctx.strokes() if s["end"] >= start and s["start"] <= end]
    if len(speeds) < 4 or not inside:
        return None
    cv = float(np.std(speeds) / (np.mean(speeds) + 1e-12))
    local = float(np.mean([s["mean_speed"] for s in inside]) / (np.mean(speeds) + 1e-12))
    if cv < 0.15:
        return cause("constant_bow_speed", 0.55, bow_speed_cv=cv)
    return cause("bow_weight_not_varied", 0.4, bow_speed_cv=cv, local_speed_ratio=local)


def _dynamics(ctx, tones):
    """Dynamics events: marks against measured loudness, then hairpins."""
    events, marked = [], [(i, LEVELS[ctx.score[i]["dynamic"]], t["loudness_db"]) for i, t in tones.items()
                          if ctx.score[i].get("dynamic") in LEVELS]
    levels = sorted({m[1] for m in marked})
    fit = None
    if len(levels) >= 2:
        slope, intercept = np.polyfit([m[1] for m in marked], [m[2] for m in marked], 1)
        fit = (slope, intercept)
        if slope < MIN_DB_PER_LEVEL:
            first, last = (min(ctx.notes[ctx._by_score[i]]["onset"] for i, _, _ in marked),
                           max(ctx.notes[ctx._by_score[i]]["onset"] for i, _, _ in marked))
            causes = [c for c in (_bow_cause(ctx, first, last, "loudness"),) if c] or \
                     [cause("limited_dynamic_range", 0.4)]
            events.append(make_event("expression", "flat_dynamics", first, last, ramp(slope, MIN_DB_PER_LEVEL, 0.0), causes, ctx,
                                     value=float(slope), expected=MIN_DB_PER_LEVEL, unit="dB per step"))
        else:                                              # notes far from the player's own dB-per-step line
            resid = np.array([m[2] - (slope * m[1] + intercept) for m in marked])
            spread = max(float(resid.std()), 1.0)
            for (i, _, db), r in zip(marked, resid):
                if abs(r) > 2 * spread:
                    p = ctx.played(i)
                    events.append(make_event(
                        "expression", "dynamic_too_loud" if r > 0 else "dynamic_too_soft", p["onset"],
                        p.get("offset", p["onset"]), ramp(abs(r), 2 * spread, 4 * spread) * 0.8 + 0.2,
                        [c for c in (_bow_cause(ctx, p["onset"], p.get("offset", p["onset"]), "loudness"),) if c] or
                        [cause("dynamic_misjudged", 0.3)],
                        ctx, i, ctx._by_score[i], float(r), 0.0, "dB"))
    runs, current = [], []                                 # consecutive notes inside the same hairpin
    for i in sorted(tones):
        h = ctx.score[i].get("hairpin")
        if current and h == ctx.score[current[-1]].get("hairpin") and h:
            current.append(i)
        else:
            if len(current) >= 3:
                runs.append(current)
            current = [i] if h else []
    if len(current) >= 3:
        runs.append(current)
    for run in runs:
        h = ctx.score[run[0]]["hairpin"]
        db = np.array([tones[i]["loudness_db"] for i in run])
        change = float(np.polyfit(np.arange(len(run)), db, 1)[0] * (len(run) - 1))     # dB over the whole hairpin
        want = 1 if h == "crescendo" else -1
        if want * change < HAIRPIN_MIN_DB:
            a, b = ctx.played(run[0]), ctx.played(run[-1])
            events.append(make_event(
                "expression", f"{h}_missing", a["onset"], b.get("offset", b["onset"]), ramp(want * change, HAIRPIN_MIN_DB, -3),
                [c for c in (_bow_cause(ctx, a["onset"], b["onset"], "the swell"),) if c] or
                [cause("hairpin_ignored", 0.4)],
                ctx, run[0], ctx._by_score[run[0]], change, want * HAIRPIN_MIN_DB, "dB"))
    return events, (None if fit is None else {"db_per_level": float(fit[0])})


def _vibrato(ctx, notes):
    """Vibrato events on sustained notes, and the vibrato summary."""
    stats = [(i, note_vibrato(ctx.frame_times, ctx.f0, ctx.notes[p]["onset"], ctx.notes[p].get("offset", ctx.notes[p]["onset"])))
             for i, p in ctx.matches if ctx.notes[p].get("offset", 0) - ctx.notes[p]["onset"] >= SUSTAINED_S]
    if not stats:
        return [], {"sustained_notes": 0}
    share = sum(v["present"] for _, v in stats) / len(stats)
    events = []
    if share < 0.2:                                          # the player does not use vibrato: one finding, not a flood
        a, b = ctx.played(stats[0][0]), ctx.played(stats[-1][0])
        events.append(make_event("expression", "no_vibrato", a["onset"], b.get("offset", b["onset"]), 0.3,
                                 [cause("vibrato_not_used", 0.5, share=share)],
                                 ctx, value=share, expected=1.0, unit="share of sustained notes"))
    else:
        for i, v in stats:
            p = ctx.played(i)
            span = (p["onset"], p.get("offset", p["onset"]))
            if not v["present"]:
                events.append(make_event("expression", "missing_vibrato", *span, 0.25,
                                         [cause("vibrato_dropped", 0.3)],
                                         ctx, i, ctx._by_score[i], 0.0, 1.0, "vibrato present"))
                continue
            issues = []
            if not VIBRATO_RATE_HZ[0] <= v["rate_hz"] <= VIBRATO_RATE_HZ[1]:
                slow = v["rate_hz"] < VIBRATO_RATE_HZ[0]
                issues.append(("vibrato_slow" if slow else "vibrato_fast",
                               cause("wide_slow_arm_motion" if slow else "tense_fast_vibrato", 0.4,
                                     rate_hz=v["rate_hz"]), v["rate_hz"], VIBRATO_RATE_HZ[0 if slow else 1], "Hz"))
            if v["extent_cents"] > VIBRATO_EXTENT_CENTS[1]:
                issues.append(("vibrato_wide", cause("exaggerated_vibrato", 0.4, extent_cents=v["extent_cents"]),
                               v["extent_cents"], VIBRATO_EXTENT_CENTS[1], "cents"))
            elif v["extent_cents"] < VIBRATO_EXTENT_CENTS[0]:
                issues.append(("vibrato_narrow",
                               cause("timid_vibrato", 0.35,
                                     extent_cents=v["extent_cents"]), v["extent_cents"], VIBRATO_EXTENT_CENTS[0], "cents"))
            if v["regularity"] < 0.4:
                issues.append(("vibrato_uneven",
                               cause("unsteady_vibrato", 0.4,
                                     regularity=v["regularity"]), v["regularity"], 0.4, "regularity"))
            for category, c, value, expected, unit in issues:
                events.append(make_event("expression", category, *span, 0.4, [c], ctx, i, ctx._by_score[i],
                                         value, expected, unit))
    rates = [v["rate_hz"] for _, v in stats if v["present"]]
    return events, {"sustained_notes": len(stats), "vibrato_share": share,
                    "mean_rate_hz": float(np.mean(rates)) if rates else None,
                    "mean_extent_cents": float(np.mean([v["extent_cents"] for _, v in stats if v["present"]])) if rates else None}


def _articulation(ctx):
    """Articulation events: sounding length / gap to the next onset against staccato / legato marks."""
    events, pairs = [], sorted(ctx.matches, key=lambda m: ctx.notes[m[1]]["onset"])
    for (i, p), (_, q) in zip(pairs, pairs[1:]):
        a, nxt = ctx.notes[p], ctx.notes[q]
        gap = nxt["onset"] - a["onset"]
        if gap <= 0.05 or "offset" not in a:
            continue
        ratio = (a["offset"] - a["onset"]) / gap
        art = ctx.score[i].get("articulation")
        stroke = ctx.stroke_at(a["onset"])
        if art == "staccato" and ratio > STACCATO_RATIO:
            c = [cause("bow_not_stopped", 0.5, length_ratio=ratio)]
            if stroke and ctx.strokes():
                longest = np.percentile([s["length"] for s in ctx.strokes()], 75)
                if stroke["length"] > longest:
                    c.insert(0, cause("long_stroke_for_staccato", 0.6, stroke_length=stroke["length"]))
            events.append(make_event("expression", "staccato_too_long", a["onset"], a["offset"],
                                     ramp(ratio, STACCATO_RATIO, 1.0),
                                     c, ctx, i, p, ratio, STACCATO_RATIO, "sounding length / slot"))
        elif art in (None, "tenuto") and ratio < 0.4 and ctx.score[i]["duration"] >= 0.5:
            events.append(make_event("expression", "notes_detached", a["onset"], a["offset"],
                                     ramp(ratio, 0.4, 0.15) * 0.7 + 0.1,
                                     [cause("bow_stopped_between_notes", 0.4,
                                            length_ratio=ratio)], ctx, i, p, ratio, LEGATO_RATIO, "sounding length / slot"))
    return events


def _neural(ctx, tones, vib):
    """Windows the neural model hears as mechanical, explained by what was measured inside them."""
    events = []
    for w in ctx.neural.get("expression", []):
        if w["p_good"] >= NEURAL_BAD:
            continue
        inside = [i for i, p in ctx.matches if w["start"] <= ctx.notes[p]["onset"] <= w["end"]]
        causes = []
        db = [tones[i]["loudness_db"] for i in inside if i in tones]
        if len(db) >= 4 and np.ptp(db) < 4:
            causes.append(cause("flat_dynamics_in_window", 0.5,
                                range_db=float(np.ptp(db))))
        if vib.get("vibrato_share") is not None and vib["vibrato_share"] < 0.3:
            causes.append(cause("little_vibrato", 0.4, share=vib["vibrato_share"]))
        events.append(make_event("expression", "mechanical_sound", w["start"], w["end"], ramp(w["p_good"], NEURAL_BAD, 0.05),
                                 causes or [cause("unmeasured", 0.2)], ctx, None, None, w["p_good"],
                                 NEURAL_BAD, "p(expressive)"))
    return events


def assess_expression(score_notes, perf_notes, alignment, ctx=None):
    """Dynamics, vibrato and articulation events plus a summary. Needs the audio in `ctx` (events.Context(perf));
    without it the module reports itself unavailable."""
    ctx = ctx or Context.from_parts(score_notes, perf_notes, alignment)
    if ctx.y is None:
        return {"module": "expression", "available": False, "score": None,
                "summary": {"reason": "no audio samples in the context"}, "events": []}
    tones = {}
    for i, p in ctx.matches:
        n = ctx.notes[p]
        t = note_tone(ctx.y, ctx.sr, n["onset"], n.get("offset", n["onset"]), n["hz"])
        if t:
            tones[i] = t
    dyn_events, dyn = _dynamics(ctx, tones)
    vib_events, vib = _vibrato(ctx, tones)
    events = dyn_events + vib_events + _articulation(ctx) + _neural(ctx, tones, vib)
    n_units = max(len(ctx.matches), 1)
    return module_result("expression", sorted(events, key=lambda e: e["start"]), n_units,
                         {"dynamics": dyn, "vibrato": vib, "n_notes": len(ctx.matches)})
