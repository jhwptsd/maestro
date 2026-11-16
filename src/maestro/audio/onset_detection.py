# Detect the onset of a note
#
# Called by audio/features.py. Onset times feed audio/transcription.py and
# assessment/rhythm.py (via features).
#
# Method: loudness rise. A bow change makes the loudness dip and then rise again, so an onset is a
# rise of >= `threshold` x 20 dB out of the quietest point of the previous ~90 ms. This is
# deliberately *not* spectral flux: vibrato, room reverb and bow noise move spectral energy
# around constantly and made flux fire several times inside a single held note. Loudness changes
# only a little with vibrato (~1 dB) but a lot at a real re-articulation. Legato (slurred) note
# changes have no dip, so they are not onsets here; audio/transcription.py finds them from pitch.
# Frames use the same grid as audio/pitch.py (utils/audio_math.frame_signal), so indices match.
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from maestro.utils.audio_math import frame_rms

RISE_WINDOW = 8        # frames (~90 ms at hop 256 / 22.05 kHz) searched for the preceding trough
DB_SCALE = 20.0        # a rise of this many dB gives strength 1.0


def _loudness_db(y, n_fft, hop_length):
    """Frame loudness in dB, floored 60 dB below the loudest frame (so silence is a finite trough)."""
    rms = frame_rms(y, n_fft, hop_length)
    return 20 * np.log10(np.maximum(rms, 1e-3 * (rms.max() if rms.size else 1.0)) + 1e-12)


def _rise_db(db):
    """Per frame: loudness above the quietest of the previous RISE_WINDOW frames (0 if not above)."""
    pad = np.pad(db, (RISE_WINDOW, 0), mode="edge")
    trough = sliding_window_view(pad, RISE_WINDOW).min(axis=1)[:len(db)]
    return np.maximum(db - trough, 0), trough


def onset_strength(y, sr, n_fft=1024, hop_length=256):
    """Onset strength per frame in [0, 1]: loudness rise out of the recent trough / 20 dB."""
    return np.clip(_rise_db(_loudness_db(y, n_fft, hop_length))[0] / DB_SCALE, 0, 1)


def detect_onsets(y, sr, hop_length=256, threshold=0.3, min_gap=0.08, n_fft=1024):
    """Return (onset times [s], strength envelope).

    An onset needs a rise of at least threshold x 20 dB (default 6 dB) and is at least `min_gap`
    seconds after the previous one. Its time is the half-rise point of the attack, not the end of it.
    """
    db = _loudness_db(y, n_fft, hop_length)
    rise, trough = _rise_db(db)
    s = np.clip(rise / DB_SCALE, 0, 1)
    w = max(1, int(min_gap * sr / hop_length))
    # candidates: first frame of a local maximum (rises strictly, then is not exceeded within `w` frames)
    ahead = sliding_window_view(np.pad(s, (0, w), constant_values=-np.inf), w + 1).max(axis=1)
    candidates = np.flatnonzero((s >= threshold) & (np.diff(s, prepend=np.inf) > 0) & (s >= ahead))
    times, last = [], -10 * w
    for i in candidates:
        if i - last < w:                                  # too close to the previous onset
            continue
        j = i                                             # walk back to the half-rise point of the attack
        while j > 0 and db[j - 1] > trough[i] + 0.5 * rise[i]:
            j -= 1
        times.append(j * hop_length / sr)
        last = i
    return np.array(times), s
