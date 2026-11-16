# Evaluate rhythm based on processed data
#
# Compares aligned onsets (audio/onset_detection.py -> audio/transcription.py) against score
# timing (score/score_representation.py), matched by alignment/align_audio_score.py. Writes to
# Performance.assessment["rhythm"].
import numpy as np

MS_GOOD, MS_OK = 40, 100            # |timing error| in milliseconds: on time / a bit off / off (shared by every chart)


def assess_rhythm(score_notes, perf_notes, alignment, wrong_note_semitones=1.0):
    """Per-note timing error in seconds (+ late / - early) and a summary.

    The student's overall tempo and start offset are removed first by a least-squares line
    performed_onset = a * score_onset + b over the matched notes (wrong notes excluded), so the error
    measures *unevenness* rather than playing slower or faster than the score's tempo.
    `tempo_ratio` = 1/a (1.0 = as written, 0.8 = 20% slower).
    """
    pairs = []
    for a in alignment:
        if a["perf_idx"] is None:
            continue
        s, p = score_notes[a["score_idx"]], perf_notes[a["perf_idx"]]
        if abs(p["midi"] - s["midi_pitch"]) <= wrong_note_semitones:
            pairs.append((a["score_idx"], a["perf_idx"], s["onset"], p["onset"]))
    if len(pairs) < 2:
        return {"notes": [], "tempo_ratio": None, "mean_abs_error": None, "std_error": None}

    so, po = np.array([p[2] for p in pairs]), np.array([p[3] for p in pairs])
    slope, intercept = np.polyfit(so, po, 1)
    err = po - (slope * so + intercept)
    return {"notes": [{"score_idx": si, "perf_idx": pi, "error": float(e)} for (si, pi, _, _), e in zip(pairs, err)],
            "tempo_ratio": float(1 / slope),
            "mean_abs_error": float(np.abs(err).mean()),
            "std_error": float(err.std())}
