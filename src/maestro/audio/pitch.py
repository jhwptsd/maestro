# Use CREPE to estimate fundamental-frequency (f0) and confidence from audio
#
# Called by audio/features.py. Its f0 output feeds assessment/intonation.py (via features) and
# audio/vibrato.py, audio/transcription.py.
# Optionally use Spotify Basic Pitch
# Use Librosa
#
# NOTE: CREPE needs TensorFlow, which is not a dependency yet, so this uses the classic YIN
# algorithm (de Cheveigné & Kawahara, 2002) in pure numpy. The interface
# (times, f0, confidence) is what CREPE returns, so a CREPE backend can replace `estimate_f0` later.
import numpy as np

from maestro.utils.audio_math import frame_signal


def estimate_f0(y, sr, frame_length=1024, hop_length=256, fmin=180.0, fmax=3000.0,
                threshold=0.15, rms_floor=0.02):
    """Frame-wise f0 estimate -> (times [s], f0 [Hz], confidence in [0, 1]).

    Defaults cover the violin range (G3 = 196 Hz up). Frames quieter than `rms_floor` x the loudest
    frame get confidence 0 (silence). f0 is always returned; callers decide voicing by confidence.
    """
    tau_min = max(2, int(sr // fmax))
    tau_max = int(np.ceil(sr / fmin))
    w = frame_length
    h = w - tau_max                              # fixed integration window of the difference function
    nfft = 1 << int(np.ceil(np.log2(w + h)))     # FFT size for linear cross-correlation

    frames = frame_signal(y, w, hop_length)
    n = len(frames)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    f0, conf = np.zeros(n), np.zeros(n)
    taus = np.arange(tau_max + 1)

    for c in range(0, n, 256):                   # chunked to bound memory
        fr = frames[c:c + 256]
        # YIN difference d(tau) = sum_j (x[j] - x[j+tau])^2 = e0 + e_tau - 2 r(tau); r via FFT
        r = np.fft.irfft(np.conj(np.fft.rfft(fr[:, :h], nfft)) * np.fft.rfft(fr, nfft), nfft)[:, :tau_max + 1]
        cs = np.concatenate([np.zeros((len(fr), 1)), np.cumsum(fr ** 2, axis=1)], axis=1)
        d = np.maximum(cs[:, h:h + 1] + (cs[:, h + taus] - cs[:, taus]) - 2 * r, 0)
        # cumulative-mean-normalised difference removes the bias towards small lags
        cmnd = np.ones_like(d)
        cmnd[:, 1:] = d[:, 1:] * taus[1:] / (np.cumsum(d[:, 1:], axis=1) + 1e-12)
        for k, row in enumerate(cmnd):
            tau, cf = _pick(row, tau_min, threshold)
            f0[c + k], conf[c + k] = sr / tau, cf

    conf[rms <= rms_floor * rms.max()] = 0.0
    return np.arange(n) * hop_length / sr, f0, conf


def _pick(row, tau_min, threshold):
    """Choose the period: first dip below `threshold` (down to its minimum), else global minimum."""
    seg = row[tau_min:]
    below = np.flatnonzero(seg < threshold)
    if below.size:
        k = below[0]
        while k + 1 < len(seg) and seg[k + 1] < seg[k]:
            k += 1
    else:
        k = int(np.argmin(seg))
    tau = k + tau_min
    conf = 1.0 - row[tau]
    if tau + 1 < len(row):                       # parabolic interpolation for sub-sample precision
        a, b, c = row[tau - 1], row[tau], row[tau + 1]
        den = a - 2 * b + c
        if den > 0:
            tau = tau + float(np.clip(0.5 * (a - c) / den, -1, 1))
    return tau, float(np.clip(conf, 0, 1))
