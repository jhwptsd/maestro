# Perform pitch integration
#
# Combines pitch.py and onset_detection.py outputs into note events (pitch + onset + duration),
# comparable to score notes in alignment/align_audio_score.py.
#
# Assumes a monophonic line (double stops are not transcribed correctly).
#
# Method (designed for real violin playing, not clean synthetic tones):
#   1. Track f0 (pitch.py); keep confident frames.
#   2. Smooth the pitch with a median filter slightly shorter than a vibrato cycle, so vibrato
#      (+-30-50 cents) does not look like note changes, then cut where the smoothed pitch moves more
#      than `max_dev` semitones from the current note's centre for a few frames (legato changes).
#   3. Re-join pieces of the same pitch separated by short pitch-tracker dropouts, unless a loudness
#      onset (onset_detection.py) lies between them (a re-bowed repeated pitch).
#   4. Split pieces that contain an onset in the middle (repeated pitch with a bow change).
#   5. Drop pieces shorter than `min_dur` (glides between notes, noise).
# A note's pitch is the median f0 of its middle part, which skips the glide into and out of it.
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from maestro.audio.onset_detection import detect_onsets
from maestro.audio.pitch import estimate_f0
from maestro.utils.audio_math import hz_to_midi


def _rolling_median(x, win):
    """NaN-aware rolling median (odd `win`); NaNs in the input stay NaN."""
    pad = np.pad(x, win // 2, constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN windows
        out = np.nanmedian(sliding_window_view(pad, win), axis=1)
    out[np.isnan(x)] = np.nan
    return out


def _cut_by_pitch(sm, max_dev, persist):
    """Split NaN-free runs of the smoothed pitch `sm` into (start, end) frame ranges of steady pitch."""
    segs, n, i = [], len(sm), 0
    while i < n:
        if np.isnan(sm[i]):
            i += 1
            continue
        start, j = i, i + 1
        while j < n and not np.isnan(sm[j]):
            centre = np.median(sm[start:j])
            ahead = sm[j:j + persist]
            if len(ahead) == persist and not np.isnan(ahead).any() and np.all(np.abs(ahead - centre) > max_dev):
                break                                      # pitch really moved: new note starts at j
            j += 1
        segs.append((start, j))
        i = j
    return segs


def transcribe(y, sr, hop_length=256, min_conf=0.8, min_dur=0.07, max_dev=0.6, vibrato_win=0.12,
               dropout=0.06, snap=0.08):
    """Audio -> list of notes {"onset", "offset", "hz", "midi"} (seconds / Hz / fractional MIDI).

    min_dur: shortest note kept (s); max_dev: semitones of (vibrato-smoothed) pitch change that start a
    new note; vibrato_win: median-filter length (s); dropout: longest pitch-tracking gap (s) bridged
    inside one note; snap: how close (s) a detected onset must be to a note start to set its time.
    """
    t, f0, conf = estimate_f0(y, sr, hop_length=hop_length)
    onsets, _ = detect_onsets(y, sr, hop_length=hop_length)
    dt = hop_length / sr
    n = len(t)
    voiced = (conf >= min_conf) & (f0 > 0)
    midi = np.full(n, np.nan)
    midi[voiced] = hz_to_midi(f0[voiced])
    win = int(vibrato_win / dt) | 1                        # odd number of frames
    sm = _rolling_median(midi, win)
    onset_frames = np.round(onsets / dt).astype(int)

    def onset_between(a, b):
        """True if a loudness onset lies strictly inside frames (a, b)."""
        return bool(np.any((onset_frames > a) & (onset_frames < b)))

    # 2. cut by pitch, 3. re-join same-pitch pieces split by tracker dropouts
    segs = _cut_by_pitch(sm, max_dev, persist=3)
    merged = []
    for s, e in segs:
        if merged:
            ps, pe = merged[-1]
            same = abs(np.nanmedian(sm[ps:pe]) - np.nanmedian(sm[s:e])) < max_dev
            if same and (s - pe) * dt <= dropout and not onset_between(pe - 2, s + 2):
                merged[-1] = (ps, e)
                continue
        merged.append((s, e))

    # 4. split at onsets inside a piece (needs `min_dur` of note on both sides)
    pieces = []
    for s, e in merged:
        cuts = [s] + [f for f in onset_frames if s + min_dur / dt <= f <= e - min_dur / dt] + [e]
        pieces += list(zip(cuts[:-1], cuts[1:]))

    # 5. build notes
    notes = []
    for s, e in pieces:
        if (e - s) * dt < min_dur:
            continue
        core = slice(s + (e - s) // 5, e - (e - s) // 5) if e - s >= 8 else slice(s, e)
        ok = voiced[core]
        if not ok.any():
            continue
        hz = float(np.median(f0[core][ok]))
        on = t[s]
        if len(onsets):
            near = onsets[np.argmin(np.abs(onsets - on))]
            on = near if abs(near - on) <= snap else on
        notes.append({"onset": float(on), "offset": float(t[e - 1]), "hz": hz, "midi": float(hz_to_midi(hz))})
    return notes
