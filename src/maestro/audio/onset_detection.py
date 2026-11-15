# Detect the onset of a note
#
# Called by audio/features.py. Onset times feed audio/transcription.py and
# assessment/rhythym.py (via features).
#
# Method: spectral flux (positive change of log-magnitude STFT) + local-maximum peak picking.
# Frames use the same grid as audio/pitch.py (utils/audio_math.frame_signal), so indices match.
import numpy as np

from maestro.utils.audio_math import frame_signal


def onset_strength(y, sr, n_fft=1024, hop_length=256):
    """Onset strength envelope normalised to max 1, one value per frame."""
    frames = frame_signal(y, n_fft, hop_length)
    mag = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1))
    rms = np.sqrt((frames ** 2).mean(axis=1))
    logm = np.log1p(100.0 * mag)
    flux = np.zeros(len(logm))
    flux[1:] = np.maximum(np.diff(logm, axis=0), 0).sum(axis=1)
    # Onsets must rise out of a loudness trough (>1.25x the quietest of the last ~90 ms): releases and
    # codec ripple inside sustained notes change the spectrum too but are not articulations. Legato
    # note changes have no trough; audio/transcription.py splits those by pitch instead.
    trough = np.lib.stride_tricks.sliding_window_view(np.pad(rms, (8, 0), constant_values=np.inf), 8)[:-1].min(axis=1)
    flux[1:] *= rms[1:] > 1.25 * trough[1:]
    return flux / flux.max() if flux.max() > 0 else flux


def detect_onsets(y, sr, hop_length=256, threshold=0.15, min_gap=0.05):
    """Return (onset times [s], strength envelope). A peak must be the local maximum within
    +-`min_gap` seconds and exceed `threshold` x the strongest onset."""
    s = onset_strength(y, sr, hop_length=hop_length)
    w = max(1, int(min_gap * sr / hop_length))
    peaks = [i for i in range(len(s))
             if s[i] > threshold and s[i] == s[max(0, i - w):i + w + 1].max()]
    return np.array(peaks, dtype=float) * hop_length / sr, s
