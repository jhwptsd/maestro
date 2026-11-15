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
# Partial performances: by default the alignment is open at both ends of the *score* (alignment/dtw.py
# open_begin/open_end), so a recording that starts halfway down the page, stops early, or covers
# only a middle section is matched to exactly that stretch of the score. Score notes outside it get
# perf_idx None. Pass partial=False to force start-to-start and end-to-end matching.
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


def _release_shared(score_notes, perf_notes, out):
    """A performed note can account for only one written note. When DTW gave one to several (a written note
    was skipped next to a repeated pitch, or two written notes were played as one), keep it with the
    claimant whose timing fits the rest of the performance best and mark the others unmatched (missed)."""
    claims = {}
    for e in out:
        if e["perf_idx"] is not None:
            claims.setdefault(e["perf_idx"], []).append(e["score_idx"])
    shared = {j: ids for j, ids in claims.items() if len(ids) > 1}
    if not shared:
        return
    firm = [(score_notes[ids[0]]["onset"], perf_notes[j]["onset"]) for j, ids in claims.items() if len(ids) == 1]
    if len(firm) >= 2:                                   # overall tempo/offset from the unambiguous matches
        a, b = np.polyfit(*zip(*firm), 1)
    else:
        a, b = 1.0, 0.0
    for j, ids in shared.items():
        keep = min(ids, key=lambda i: abs(a * score_notes[i]["onset"] + b - perf_notes[j]["onset"]))
        for i in ids:
            if i != keep:
                out[i] = {"score_idx": i, "perf_idx": None, "cost": None}


def _pair_leftovers(score_notes, perf_notes, out):
    """A written note nobody played next to a played note nobody explains, inside the same gap between two
    firm matches, is one wrong note (not one missed plus one extra): pair them, in order. Whole events
    only: chord notes that were simply not heard stay unmatched."""
    matched = [e for e in out if e["perf_idx"] is not None]
    if len(matched) < 2:
        return
    used = {e["perf_idx"] for e in matched}
    heard_onsets = {round(score_notes[e["score_idx"]]["onset"], 6) for e in matched}
    for a, b in zip(matched[:-1], matched[1:]):
        gap_notes = [i for i in range(a["score_idx"] + 1, b["score_idx"])
                     if out[i]["perf_idx"] is None and round(score_notes[i]["onset"], 6) not in heard_onsets]
        t0, t1 = perf_notes[a["perf_idx"]]["onset"], perf_notes[b["perf_idx"]]["onset"]
        gap_perf = [j for j, p in enumerate(perf_notes) if j not in used and t0 < p["onset"] < t1]
        for i, j in zip(gap_notes, gap_perf):
            dist = abs(score_notes[i]["midi_pitch"] - perf_notes[j]["midi"])
            out[i] = {"score_idx": i, "perf_idx": j, "cost": min(dist, 12.0) / 12.0}
            used.add(j)


def align_audio_score(score_notes, perf_notes, position_weight=0.5, max_semitones=12.0,
                      chord_tolerance=1.5, partial=True):
    """Match each score note to at most one performed note.

    score_notes: dicts with "midi_pitch", "onset" (score/symbolic_features.py output), sorted by onset
    perf_notes:  dicts with "midi"                (audio/transcription.py output)
    Event cost = distance (semitones, capped, scaled to [0, 1]) from the performed note to the
    *closest* pitch of the event + `position_weight` x difference in relative position in the piece
    (disambiguates repeated pitches; only used when partial=False, since a partial performance does not
    span the whole piece). With partial=True ties (e.g. a phrase that occurs twice) resolve to the first
    occurrence and the longest stretch (see dtw.py).
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
    cost = np.minimum(dist, max_semitones) / max_semitones
    if not partial:
        cost = cost + position_weight * np.abs(pos_e[:, None] - pos_p[None, :])
    path, _ = dtw(cost, open_begin=partial, open_end=partial)

    candidates = {}                      # event -> performed notes DTW mapped to it
    for e, j in path:
        candidates.setdefault(e, []).append(j)

    out = [none(i) for i in range(n)]
    for e, ev in enumerate(events):
        if e not in candidates:          # outside the stretch of the score that was played
            continue
        cands = candidates[e]
        if len(ev) == 1:                 # single note: cheapest candidate
            j = min(cands, key=lambda j: (round(cost[e, j], 2), j))   # near-equal cost: the earlier played note
            out[ev[0]] = {"score_idx": ev[0], "perf_idx": int(j), "cost": float(cost[e, j])}
            continue
        # chord: greedily pair closest (pitch, performed note) first, one-to-one, within tolerance
        pairs = sorted((abs(score_notes[i]["midi_pitch"] - pp[j]), i, j) for i in ev for j in cands)
        used_i, used_j = set(), set()
        for d, i, j in pairs:
            if d <= chord_tolerance and i not in used_i and j not in used_j:
                used_i.add(i), used_j.add(j)
                out[i] = {"score_idx": i, "perf_idx": int(j), "cost": float(cost[e, j])}
    _release_shared(score_notes, perf_notes, out)
    _pair_leftovers(score_notes, perf_notes, out)
    return out
