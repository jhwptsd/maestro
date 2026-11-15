# Take in processed audio and processed score and align them
#
# Uses dtw.py to match the transcribed performance notes (audio/transcription.py) to the score
# notes (score/symbolic_features.py); the mapping is written into Performance.alignment.
# Required by assessment/intonation.py, rhythm.py (and later tone.py, expression.py).
#
# Alignment is note-level (not frame-level): robust to tempo changes and cheap, but a missed or
# extra note shows up as a poor match rather than an explicit gap.
import numpy as np

from maestro.alignment.dtw import dtw


def align_audio_score(score_notes, perf_notes, position_weight=0.5, max_semitones=12.0):
    """Match each score note to one performed note.

    score_notes: dicts with "midi_pitch" (score/symbolic_features.py output)
    perf_notes:  dicts with "midi"        (audio/transcription.py output)
    Cost = pitch distance (semitones, capped, scaled to [0, 1]) + `position_weight` x difference in
    relative position in the piece, which disambiguates repeated pitches.
    Returns one entry per score note: {"score_idx", "perf_idx", "cost"} ("perf_idx" None if the
    performance has no notes). When DTW maps several performed notes to one score note, the
    cheapest is kept.
    """
    n, m = len(score_notes), len(perf_notes)
    if n == 0 or m == 0:
        return [{"score_idx": i, "perf_idx": None, "cost": None} for i in range(n)]
    sp = np.array([s["midi_pitch"] for s in score_notes], dtype=float)
    pp = np.array([p["midi"] for p in perf_notes], dtype=float)
    pos_s = np.arange(n) / max(n - 1, 1)
    pos_p = np.arange(m) / max(m - 1, 1)
    cost = (np.minimum(np.abs(sp[:, None] - pp[None, :]), max_semitones) / max_semitones
            + position_weight * np.abs(pos_s[:, None] - pos_p[None, :]))
    path, _ = dtw(cost)
    best = {}
    for i, j in path:
        if i not in best or cost[i, j] < cost[i, best[i]]:
            best[i] = j
    return [{"score_idx": i, "perf_idx": int(best[i]), "cost": float(cost[i, best[i]])} for i in range(n)]
