# Perform pitch integration
#
# Combines pitch.py and onset_detection.py outputs into note events (pitch + onset + duration),
# comparable to score notes in alignment/align_audio_score.py.
#
# Assumes a monophonic line (double stops are not transcribed correctly).
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from maestro.audio.onset_detection import detect_onsets
from maestro.audio.pitch import estimate_f0
from maestro.utils.audio_math import hz_to_midi


def _rolling_median(x, win=5):
    """NaN-aware rolling median; NaNs in the input stay NaN."""
    pad = np.pad(x, win // 2, constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN windows
        out = np.nanmedian(sliding_window_view(pad, win), axis=1)
    out[np.isnan(x)] = np.nan
    return out


def transcribe(y, sr, hop_length=256, min_conf=0.8, min_dur=0.06, max_dev=0.7, snap=0.06):
    """Audio -> list of notes {"onset", "offset", "hz", "midi"} (seconds / Hz / fractional MIDI).

    A note ends on silence, a pitch jump > `max_dev` semitones (vibrato is ~0.3, so it is tolerated),
    or a detected onset (re-articulated repeated pitch). Notes shorter than `min_dur` are dropped.
    Start times snap to a detected onset within `snap` seconds, since the pitch tracker alone
    is only accurate to about half a frame window.
    """
    t, f0, conf = estimate_f0(y, sr, hop_length=hop_length)
    onsets, _ = detect_onsets(y, sr, hop_length=hop_length)
    n = len(t)
    voiced = (conf >= min_conf) & (f0 > 0)
    midi = np.full(n, np.nan)
    midi[voiced] = hz_to_midi(f0[voiced])
    sm = _rolling_median(midi)
    onset_frames = set(np.round(onsets * sr / hop_length).astype(int))
    dt = hop_length / sr

    segs, start = [], None
    for i in range(n):
        if np.isnan(sm[i]):
            if start is not None:
                segs.append((start, i))
            start = None
        elif start is None:
            start = i
        else:
            onset_here = any(k in onset_frames for k in (i - 1, i, i + 1)) and (i - start) * dt > min_dur
            if abs(sm[i] - np.median(sm[start:i])) > max_dev or onset_here:
                segs.append((start, i))
                start = i
    if start is not None:
        segs.append((start, n))

    notes = []
    for s, e in segs:
        if (e - s) * dt < min_dur:
            continue
        hz = float(np.median(f0[s:e][voiced[s:e]]))
        on = t[s]
        if len(onsets):
            near = onsets[np.argmin(np.abs(onsets - on))]
            on = near if abs(near - on) <= snap else on
        notes.append({"onset": float(on), "offset": float(t[e - 1]), "hz": hz, "midi": float(hz_to_midi(hz))})
    return notes
