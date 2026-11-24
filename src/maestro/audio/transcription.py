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
from maestro.audio.pitch import DEFAULT_PITCH_BACKEND, estimate_pitch
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


def _merge_dropouts(segs, sm, onset_frames, max_dev, max_gap_frames):
    """Re-join neighbouring pieces of the same pitch separated by a short tracker dropout, unless a bow
    onset lies between them (that is a repeated note, not one note)."""
    merged = []
    for start, end in segs:
        if merged:
            prev_start, prev_end = merged[-1]
            same_pitch = abs(np.nanmedian(sm[prev_start:prev_end]) - np.nanmedian(sm[start:end])) < max_dev
            rebowed = np.any((onset_frames > prev_end - 2) & (onset_frames < start + 2))
            if same_pitch and start - prev_end <= max_gap_frames and not rebowed:
                merged[-1] = (prev_start, end)
                continue
        merged.append((start, end))
    return merged


def _split_at_onsets(segs, onset_frames, min_frames):
    """Split a piece wherever a bow onset falls inside it (leaving at least `min_frames` on each side)."""
    pieces = []
    for start, end in segs:
        cuts = [start] + [f for f in onset_frames if start + min_frames <= f <= end - min_frames] + [end]
        pieces += zip(cuts[:-1], cuts[1:])
    return pieces


def transcribe(y, sr, hop_length=256, min_conf=0.5, min_dur=0.07, max_dev=0.6,
               vibrato_win=0.12, dropout=0.06, snap=0.08,
               backend=DEFAULT_PITCH_BACKEND, pitch_track=None, pitch_options=None):
    """Audio -> note dictionaries with onset, offset, hz and fractional MIDI pitch.

    Select the pitch implementation with backend: ``crepe`` (default), ``yin``,
    ``pyin`` or ``basic_pitch``. Pass a previously computed PitchTrack to avoid running
    inference twice in a pipeline. pitch_options is forwarded to the selected backend.

    Frame trackers are segmented into notes in five steps (parameters in seconds unless noted):
    1. keep confident frames (>= min_conf) and smooth their pitch with a median filter of `vibrato_win`,
       so vibrato does not look like a note change;
    2. cut where the smoothed pitch stays more than `max_dev` semitones from the current note's median;
    3. re-join same-pitch pieces split by a tracker dropout of up to `dropout`, unless a bow onset is between;
    4. split pieces at bow onsets (a repeated pitch); drop anything shorter than `min_dur`;
    5. a note's pitch is the median f0 of its middle 60%; its onset snaps to a detected bow onset within `snap`.

    Basic Pitch is event-based and can return overlapping notes. Its MIDI-note estimates
    are semitone-quantized, so it should not be used as the sole source for cents-level
    intonation assessment.
    """
    track = pitch_track or estimate_pitch(
        y, sr, backend=backend, hop_length=hop_length, **(pitch_options or {})
    )
    if track.notes is not None:
        return [dict(note) for note in track.notes]

    times, f0 = track.times, track.f0
    dt = hop_length / sr
    onsets, _ = detect_onsets(y, sr, hop_length=hop_length)
    onset_frames = np.clip(np.round(onsets / dt).astype(int), 0, max(len(times) - 1, 0))

    voiced = (track.confidence >= min_conf) & (f0 > 0)
    midi = np.full(len(times), np.nan)
    midi[voiced] = hz_to_midi(f0[voiced])
    smooth = _rolling_median(midi, max(1, int(vibrato_win / dt)) | 1)         # odd window length

    segments = _cut_by_pitch(smooth, max_dev, persist=3)
    segments = _merge_dropouts(segments, smooth, onset_frames, max_dev, dropout / dt)
    pieces = _split_at_onsets(segments, onset_frames, min_dur / dt)

    notes = []
    for start, end in pieces:
        if (end - start) * dt < min_dur:
            continue
        trim = (end - start) // 5 if end - start >= 8 else 0                  # skip the glide into/out of the note
        core = slice(start + trim, end - trim)
        if not voiced[core].any():
            continue
        hz = float(np.median(f0[core][voiced[core]]))
        onset = float(times[start])
        if len(onsets):
            nearest = onsets[np.argmin(np.abs(onsets - onset))]
            if abs(nearest - onset) <= snap:
                onset = float(nearest)
        notes.append({"onset": onset, "offset": float(times[end - 1]), "hz": hz, "midi": float(hz_to_midi(hz))})
    return notes
