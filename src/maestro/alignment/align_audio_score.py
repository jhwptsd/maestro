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


def _entry(score_idx, perf_idx=None, cost=None):
    """One alignment result: written note `score_idx` matched to performed note `perf_idx` (None = missed)."""
    return {"score_idx": score_idx, "perf_idx": perf_idx, "cost": cost}


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
                out[i] = _entry(i)


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
            out[i] = _entry(i, j, min(dist, 12.0) / 12.0)
            used.add(j)


def align_audio_score(score_notes, perf_notes, position_weight=0.5, max_semitones=12.0,
                      chord_tolerance=1.5, partial=True):
    """Match each score note to at most one performed note.

    score_notes: dicts with "midi_pitch", "onset" (score/symbolic_features.py output), sorted by onset
    perf_notes:  dicts with "midi", "onset"       (audio/transcription.py output)
    Returns one {"score_idx", "perf_idx", "cost"} per score note ("perf_idx" None = not matched).

    Event cost = distance (semitones, capped, scaled to [0, 1]) from the performed note to the *closest*
    pitch of the event. With partial=False the cost also gets `position_weight` x the difference in relative
    position in the piece, which helps tell repeated pitches apart when the whole piece was played; with
    partial=True the performance may cover only part of the score, so ties (e.g. a phrase that occurs twice)
    resolve to the first occurrence and the longest stretch instead (see dtw.py).

    After DTW:
    - a single note takes the cheapest performed note mapped to its event (so a wrong note shows up in
      assessment/ as a large deviation); a chord pairs the heard notes to its pitches within `chord_tolerance`
      semitones, and chord notes nobody heard stay unmatched;
    - a performed note serves one written note only (`_release_shared`), and a missed note next to an
      unexplained performed note becomes one wrong note (`_pair_leftovers`).
    """
    n, m = len(score_notes), len(perf_notes)
    out = [_entry(i) for i in range(n)]
    if n == 0 or m == 0:
        return out
    events = score_events(score_notes)
    perf_pitch = np.array([p["midi"] for p in perf_notes], dtype=float)
    # distance from every performed note to the closest pitch of every event: (events, performed notes)
    dist = np.stack([np.abs(np.array([score_notes[i]["midi_pitch"] for i in ev], dtype=float)[:, None]
                            - perf_pitch[None, :]).min(axis=0) for ev in events])
    cost = np.minimum(dist, max_semitones) / max_semitones
    if not partial:
        pos_e = np.arange(len(events)) / max(len(events) - 1, 1)
        pos_p = np.arange(m) / max(m - 1, 1)
        cost = cost + position_weight * np.abs(pos_e[:, None] - pos_p[None, :])
    path, _ = dtw(cost, open_begin=partial, open_end=partial)

    candidates = {}                                   # event -> performed notes DTW mapped to it
    for e, j in path:
        candidates.setdefault(e, []).append(j)
    for e, ev in enumerate(events):
        cands = candidates.get(e)
        if cands is None:                             # outside the stretch of the score that was played
            continue
        if len(ev) == 1:                              # near-equal cost: prefer the earlier played note
            j = min(cands, key=lambda j: (round(cost[e, j], 2), j))
            out[ev[0]] = _entry(ev[0], int(j), float(cost[e, j]))
            continue
        # chord: pair the closest (pitch, performed note) first, one-to-one, within tolerance
        pairs = sorted((abs(score_notes[i]["midi_pitch"] - perf_pitch[j]), i, j) for i in ev for j in cands)
        used_i, used_j = set(), set()
        for d, i, j in pairs:
            if d <= chord_tolerance and i not in used_i and j not in used_j:
                used_i.add(i), used_j.add(j)
                out[i] = _entry(i, int(j), float(cost[e, j]))
    _release_shared(score_notes, perf_notes, out)
    _pair_leftovers(score_notes, perf_notes, out)
    return out
