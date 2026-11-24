# The numbers of a performance as a profile, and the reference's profile as a baseline with its interpretive choices marked (new file).
#
# `profile(...)` measures one performance (the student's or the reference's, the same way for both so they can be compared fairly): per written
# note (score index) the pitch deviation from equal temperament, duration, how much of the gap to the next note it fills (articulation), loudness,
# attack, brightness, vibrato; per measure the tempo against the marking, the loudness and the bow changes and speed. Notes the soloist is not
# audible in carry {"gap": ...} instead of numbers, and measures without usable video carry no bowing numbers: gaps stay gaps.
#
# A professional is not an answer key. What a professional does falls in two kinds, and every comparison says which:
#   objective     playing accurately is the same for everyone: pitch within a few cents of the target, a clean attack.
#   interpretive  the soloist's own musical decisions: the tempo, rubato, dynamic shaping, articulation, vibrato use, bowing, tone colour, and
#                 intonation that departs from equal temperament on purpose (expressive or melodic intonation). Differing from these is a different
#                 interpretation, not an error.
# `NATURE` fixes the kind of each category, and `reference_choices` lists the decisions this soloist visibly made (with their numbers), so the
# comparison and the feedback can cite them as choices, never as targets.
import numpy as np

from maestro.assessment.events import jsonable
from maestro.audio.tone_quality import note_tone
from maestro.audio.vibrato import note_vibrato

NATURE = {"pitch": "objective", "attack": "objective", "tempo": "interpretive", "rubato": "interpretive", "dynamics": "interpretive",
          "articulation": "interpretive", "vibrato": "interpretive", "bowing": "interpretive", "tone_color": "interpretive"}
EXPRESSIVE_CENTS = 12.0          # a reference note this far from equal temperament is read as expressive intonation (or a slip), not as the target
RUBATO = 0.08                    # a measure whose tempo differs this much (relative) from the soloist's usual one is a deliberate shift
DYNAMIC_DB = 4.0                 # a measure this much louder or softer than the clip's median is dynamic shaping
SHORT_NOTE = 0.6                 # a note filling less than this share of the gap to the next one is played short (detache, staccato)
LONG_NOTE_S = 0.4                # vibrato statistics use notes at least this long
MIN_NOTE_PRESENCE = 0.6          # share of a note's frames that must hold the soloist for it to be measured


def _pairs(score, alignment):
    """[(score_idx, perf_idx)] of the matched notes, in score order."""
    return sorted(((a["score_idx"], a["perf_idx"]) for a in alignment if a["perf_idx"] is not None), key=lambda m: score[m[0]]["onset"])


def profile(score, notes, alignment, y, sr, frames, strokes=(), presence=None):
    """The measurements of one performance -> {"notes": {score_idx: {...} | {"gap"}}, "measures": {measure: {...}}, "tempo_ratio" (median
    over the piece; 1 = the marked tempo, 2 = twice as fast), "median_loudness_db"} (integer keys; reference_summary makes it JSON). `strokes`: bow strokes with times in this performance (empty:
    no video, so no bowing numbers). `presence(start, end)` -> 0-1 gives the share of the span in which the soloist is audible (None: always)."""
    times = np.array([f["time"] for f in frames])
    f0 = np.array([f["estimated_pitch"] for f in frames])
    pairs = _pairs(score, alignment)
    out = {}
    for k, (s, p) in enumerate(pairs):
        n, written = notes[p], score[s]
        if presence is not None and presence(n["onset"], n["offset"]) < MIN_NOTE_PRESENCE:
            out[s] = {"gap": "soloist_not_audible"}
            continue
        tone = note_tone(y, sr, n["onset"], n["offset"], n["hz"]) or {}
        entry = {"cents": 100 * (n["midi"] - written["midi_pitch"]), "onset": n["onset"], "duration": n["offset"] - n["onset"],
                 "loudness_db": tone.get("loudness_db"), "attack_s": tone.get("attack_s"), "brightness": tone.get("brightness"),
                 "vibrato": note_vibrato(times, f0, n["onset"], n["offset"])}
        if k + 1 < len(pairs):                                     # how much of the gap to the next note this one fills, and the pace
            nxt_s, nxt_p = pairs[k + 1]
            gap, written_gap = notes[nxt_p]["onset"] - n["onset"], score[nxt_s]["onset"] - written["onset"]
            if gap > 0.05:
                entry["articulation"] = entry["duration"] / gap
            if written_gap > 0.05 and gap > 0.05 and score[nxt_s]["measure"] == written["measure"]:
                entry["tempo_ratio"] = written_gap / gap
        out[s] = entry
    measures = {}
    for s, e in out.items():
        m = measures.setdefault(score[s]["measure"], {"notes": [], "tempo": [], "loudness": []})
        m["notes"].append(s)
        m["tempo"] += [e["tempo_ratio"]] if "tempo_ratio" in e else []
        m["loudness"] += [e["loudness_db"]] if e.get("loudness_db") is not None else []
    all_tempo = [e["tempo_ratio"] for e in out.values() if "tempo_ratio" in e]
    all_loud = [e["loudness_db"] for e in out.values() if e.get("loudness_db") is not None]
    summary = {}
    for number, m in measures.items():
        matched = [notes[dict(pairs)[s]] for s in m["notes"]]
        start, end = min(n["onset"] for n in matched), max(n["offset"] for n in matched)
        overlapping = [st for st in strokes if st["end"] > start and st["start"] < end]
        summary[number] = {"tempo_ratio": float(np.median(m["tempo"])) if m["tempo"] else None,
                           "loudness_db": float(np.mean(m["loudness"])) if m["loudness"] else None,
                           "dynamic": next((score[s].get("dynamic") for s in m["notes"] if score[s].get("dynamic")), None),
                           "start": start, "end": end,
                           "bow_changes": sum(start < st["start"] < end for st in strokes) if overlapping else None,
                           "bow_speed": float(np.mean([st["mean_speed"] for st in overlapping])) if overlapping else None}
    return {"notes": out, "measures": summary, "tempo_ratio": float(np.median(all_tempo)) if all_tempo else None,
            "median_loudness_db": float(np.median(all_loud)) if all_loud else None}


def reference_choices(prof, score):
    """The decisions the reference soloist visibly made, from its profile -> [{"category", "slug", "measure" | "score_idx" | None, "value",
    "expected", "unit", "nature": "interpretive"}]: always interpretive, and only numbers (what the soloist did, against the neutral reading)."""
    choice = lambda category, slug, value, expected, unit, **where: {"category": category, "slug": slug, "value": value, "expected": expected,
                                                                      "unit": unit, "nature": NATURE[category], **where}
    out, tempo = [], prof["tempo_ratio"]
    if tempo is not None:
        out.append(choice("tempo", "tempo_choice", tempo, 1.0, "x_marked_tempo"))
    for number, m in prof["measures"].items():
        if tempo and m["tempo_ratio"] and abs(m["tempo_ratio"] / tempo - 1) > RUBATO:
            out.append(choice("rubato", "local_tempo_shift", m["tempo_ratio"] / tempo, 1.0, "x_usual_tempo", measure=number))
        if prof["median_loudness_db"] is not None and m["loudness_db"] is not None and abs(m["loudness_db"] - prof["median_loudness_db"]) > DYNAMIC_DB:
            out.append(choice("dynamics", "dynamic_shaping", m["loudness_db"] - prof["median_loudness_db"], 0.0, "db_from_median", measure=number))
    long_notes = [e for e in prof["notes"].values() if "gap" not in e and e["duration"] >= LONG_NOTE_S and e["vibrato"]["rate_hz"] is not None]
    if long_notes:
        with_vibrato = [e["vibrato"] for e in long_notes if e["vibrato"]["present"]]
        out.append(choice("vibrato", "vibrato_use", len(with_vibrato) / len(long_notes), None, "share_of_long_notes",
                          rate_hz=float(np.median([v["rate_hz"] for v in with_vibrato])) if with_vibrato else None,
                          extent_cents=float(np.median([v["extent_cents"] for v in with_vibrato])) if with_vibrato else None))
    for s, e in prof["notes"].items():
        if "gap" in e:
            continue
        if abs(e["cents"]) > EXPRESSIVE_CENTS:
            out.append(choice("pitch", "expressive_intonation", e["cents"], 0.0, "cents_from_equal_temperament", score_idx=s))
        if e.get("articulation") is not None and e["articulation"] < SHORT_NOTE and not score[s].get("articulation"):
            out.append(choice("articulation", "played_short", e["articulation"], None, "share_of_gap_filled", score_idx=s))
    changes = [m["bow_changes"] for m in prof["measures"].values() if m["bow_changes"] is not None]
    if changes:
        out.append(choice("bowing", "bowing_choice", float(np.median(changes)), None, "bow_changes_per_measure"))
    return jsonable(out)
