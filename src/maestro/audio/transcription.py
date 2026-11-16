# Convert frame-wise pitch or model note-events into note events.
#
# Frame-based trackers (CREPE, YIN and pYIN) use Maestro's vibrato-aware segmentation
# and loudness-based bow-attack detection. Basic Pitch supplies note events directly and
# can retain simultaneous pitches (e.g. violin double stops).
import bisect
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from maestro.audio.onset_detection import detect_onsets
from maestro.audio.pitch import DEFAULT_PITCH_BACKEND, PitchTrack, estimate_pitch
from maestro.utils.audio_math import hz_to_midi


def _rolling_median(x, win):
    """NaN-aware rolling median (odd win); NaNs in the input stay NaN."""
    pad = np.pad(x, win // 2, constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(sliding_window_view(pad, win), axis=1)
    out[np.isnan(x)] = np.nan
    return out


def _cut_by_pitch(sm, max_dev, persist):
    """Split NaN-free pitch runs into stable-pitch frame ranges.

    A run is cut where the smoothed pitch has stayed more than `max_dev` semitones from the running
    median of the current note for `persist` frames. The running median is kept in a sorted list
    (insertion is a cheap memmove), so each step is O(1) instead of re-sorting the whole note.
    """
    vals = sm.tolist()
    segs, n, i = [], len(vals), 0
    while i < n:
        if vals[i] != vals[i]:                      # NaN: unvoiced
            i += 1
            continue
        start, j = i, i + 1
        buf = [vals[i]]                             # sorted values of the current note so far
        while j < n and vals[j] == vals[j]:
            half = len(buf) // 2
            centre = buf[half] if len(buf) % 2 else (buf[half - 1] + buf[half]) / 2
            ahead = vals[j:j + persist]
            if len(ahead) == persist and all(v == v and abs(v - centre) > max_dev for v in ahead):
                break
            bisect.insort(buf, vals[j])
            j += 1
        segs.append((start, j))
        i = j
    return segs


def transcribe(y, sr, hop_length=256, min_conf=0.5, min_dur=0.07, max_dev=0.6,
               vibrato_win=0.12, dropout=0.06, snap=0.08,
               backend=DEFAULT_PITCH_BACKEND, pitch_track=None, pitch_options=None):
    """Audio -> note dictionaries with onset, offset, hz and fractional MIDI pitch.

    Select the pitch implementation with backend: ``crepe`` (default), ``yin``,
    ``pyin`` or ``basic_pitch``. Pass a previously computed PitchTrack to avoid running
    inference twice in a pipeline. pitch_options is forwarded to the selected backend.

    Basic Pitch is event-based and can return overlapping notes. Its MIDI-note estimates
    are semitone-quantized, so it should not be used as the sole source for cents-level
    intonation assessment.
    """
    track = pitch_track or estimate_pitch(
        y, sr, backend=backend, hop_length=hop_length, **(pitch_options or {})
    )
    if track.notes is not None:
        return [dict(note) for note in track.notes]

    onsets, _ = detect_onsets(y, sr, hop_length=hop_length)
    t, f0, conf = track.times, track.f0, track.confidence
    dt = hop_length / sr
    n = len(t)
    voiced = (conf >= min_conf) & (f0 > 0)
    midi = np.full(n, np.nan)
    midi[voiced] = hz_to_midi(f0[voiced])
    win = max(1, int(vibrato_win / dt)) | 1
    sm = _rolling_median(midi, win)
    onset_frames = np.clip(np.round(onsets / dt).astype(int), 0, max(n - 1, 0))

    def onset_between(a, b):
        return bool(np.any((onset_frames > a) & (onset_frames < b)))

    segs = _cut_by_pitch(sm, max_dev, persist=3)
    merged = []
    for start, end in segs:
        if merged:
            previous_start, previous_end = merged[-1]
            same = abs(np.nanmedian(sm[previous_start:previous_end]) - np.nanmedian(sm[start:end])) < max_dev
            if same and (start - previous_end) * dt <= dropout and not onset_between(previous_end - 2, start + 2):
                merged[-1] = (previous_start, end)
                continue
        merged.append((start, end))

    pieces = []
    for start, end in merged:
        cuts = [start] + [frame for frame in onset_frames
                          if start + min_dur / dt <= frame <= end - min_dur / dt] + [end]
        pieces += list(zip(cuts[:-1], cuts[1:]))

    notes = []
    for start, end in pieces:
        if (end - start) * dt < min_dur:
            continue
        core = slice(start + (end - start) // 5, end - (end - start) // 5) if end - start >= 8 else slice(start, end)
        valid = voiced[core]
        if not valid.any():
            continue
        hz = float(np.median(f0[core][valid]))
        onset = float(t[start])
        if len(onsets):
            nearest = onsets[np.argmin(np.abs(onsets - onset))]
            if abs(nearest - onset) <= snap:
                onset = float(nearest)
        notes.append({
            "onset": onset,
            "offset": float(t[end - 1]),
            "hz": hz,
            "midi": float(hz_to_midi(hz)),
        })
    return notes
