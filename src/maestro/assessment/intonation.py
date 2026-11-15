# Evaluate intonation based on processed data
#
# Compares aligned pitch (audio/pitch.py -> audio/transcription.py) against score pitch
# (score/symbolic_features.py), matched by alignment/align_audio_score.py. Result goes to
# Performance.assessment["intonation"], which feedback/generator.py reads.
import numpy as np

from maestro.utils.audio_math import cents_between


def assess_intonation(score_notes, perf_notes, alignment, wrong_note_semitones=1.0):
    """Per-note pitch deviation in cents (+ sharp / - flat) and a summary.

    Matches further than `wrong_note_semitones` from the written pitch are flagged "wrong_note"
    (almost certainly a wrong or mis-transcribed note, not an intonation error) and excluded from the
    summary. `tuning_offset_cents` (median deviation) reveals a consistently sharp/flat player;
    `mean_abs_cents` is the average absolute deviation of the rest.
    """
    notes = []
    for a in alignment:
        if a["perf_idx"] is None:
            continue
        s, p = score_notes[a["score_idx"]], perf_notes[a["perf_idx"]]
        cents = float(cents_between(p["hz"], s["hz"]))
        notes.append({"score_idx": a["score_idx"], "perf_idx": a["perf_idx"], "expected_hz": s["hz"],
                      "actual_hz": p["hz"], "cents": cents, "wrong_note": abs(cents) > wrong_note_semitones * 100})
    ok = np.array([n["cents"] for n in notes if not n["wrong_note"]])
    return {"notes": notes,
            "n_wrong_notes": sum(n["wrong_note"] for n in notes),
            "tuning_offset_cents": float(np.median(ok)) if ok.size else None,
            "mean_abs_cents": float(np.abs(ok).mean()) if ok.size else None}
