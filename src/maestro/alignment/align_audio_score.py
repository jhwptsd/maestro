# Take in processed audio and processed score and align them
#
# Uses dtw.py to match the transcribed performance notes (audio/transcription.py) to the score
# notes (score/symbolic_features.py); the mapping is written into Performance.alignment.
# Required by assessment/intonation.py, rhythm.py (and later tone.py, expression.py).
#
# Multiple stops: score notes sharing an onset form one *event*. DTW aligns events (not individual
# notes), and the performed notes it maps to an event are then assigned to the chord's pitches.
# Because the transcriber is monophonic, chord notes that were not heard stay unmatched
# (perf_idx None) and are skipped by assessment/.
#
# Alignment is note-level (not frame-level): robust to tempo changes and cheap, but a missed or
# extra note shows up as a poor match rather than an explicit gap.
import numpy as np

from maestro.alignment.dtw import dtw


def score_events(score_notes, tol=1e-3):
    """Group score notes with (nearly) equal onset -> list of index lists (chords = len > 1)."""
    groups = []
    for i, n in enumerate(score_notes):
        if groups and abs(n["onset"] - score_notes[groups[-1][0]]["onset"]) < tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def align_audio_score(score_notes, perf_notes, position_weight=0.5, max_semitones=12.0,
                      chord_tolerance=1.5):
    """Match each score note to at most one performed note.

    score_notes: dicts with "midi_pitch", "onset" (score/symbolic_features.py output), sorted by onset
    perf_notes:  dicts with "midi"                (audio/transcription.py output)
    Event cost = distance (semitones, capped, scaled to [0, 1]) from the performed note to the
    *closest* pitch of the event + `position_weight` x difference in relative position in the piece
    (disambiguates repeated pitches).
    Returns one entry per score note: {"score_idx", "perf_idx", "cost"}; "perf_idx" is None if the
    performance has no notes or, for a chord note, nothing within `chord_tolerance` semitones was
    heard. Single notes always get the cheapest performed note DTW mapped to them (so wrong notes
    surface in assessment/ as large deviations); a performed note can be shared by successive
    score notes only if DTW stretched it over them.
    """
    n, m = len(score_notes), len(perf_notes)
    none = lambda i: {"score_idx": i, "perf_idx": None, "cost": None}
    if n == 0 or m == 0:
        return [none(i) for i in range(n)]
    events = score_events(score_notes)
    pp = np.array([p["midi"] for p in perf_notes], dtype=float)
    # distance from every performed note to every pitch of every event, per (event, note)
    dist = np.stack([np.abs(np.array([score_notes[i]["midi_pitch"] for i in ev], dtype=float)[:, None]
                            - pp[None, :]).min(axis=0) for ev in events])
    pos_e = np.arange(len(events)) / max(len(events) - 1, 1)
    pos_p = np.arange(m) / max(m - 1, 1)
    cost = (np.minimum(dist, max_semitones) / max_semitones
            + position_weight * np.abs(pos_e[:, None] - pos_p[None, :]))
    path, _ = dtw(cost)

    candidates = {}                      # event -> performed notes DTW mapped to it
    for e, j in path:
        candidates.setdefault(e, []).append(j)

    out = [none(i) for i in range(n)]
    for e, ev in enumerate(events):
        cands = candidates[e]
        if len(ev) == 1:                 # single note: cheapest candidate
            j = min(cands, key=lambda j: cost[e, j])
            out[ev[0]] = {"score_idx": ev[0], "perf_idx": int(j), "cost": float(cost[e, j])}
            continue
        # chord: greedily pair closest (pitch, performed note) first, one-to-one, within tolerance
        pairs = sorted((abs(score_notes[i]["midi_pitch"] - pp[j]), i, j) for i in ev for j in cands)
        used_i, used_j = set(), set()
        for d, i, j in pairs:
            if d <= chord_tolerance and i not in used_i and j not in used_j:
                used_i.add(i), used_j.add(j)
                out[i] = {"score_idx": i, "perf_idx": int(j), "cost": float(cost[e, j])}
    return out
