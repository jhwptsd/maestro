"""NumPy implementation of the YIN fundamental-frequency estimator.

Speed notes (the results are numerically identical to the straightforward implementation):
 - The cross-correlation needs only lags 0..tau_max, so a transform length of the next power of two
   >= frame_length is enough (no wrap-around reaches those lags); the textbook zero-padding to
   frame_length + window doubled the FFT size and the FFT time.
 - Frames are read through a strided view instead of materialising an (n_frames x frame_length) copy.
 - Period picking (first trough under the threshold, descend to its minimum, parabolic interpolation)
   is vectorised over all frames of a chunk instead of looping in Python per frame.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from maestro.audio.backends.base import PitchTrack, frame_times

CHUNK = 256  # frames per FFT batch: large enough to amortise call overhead, small enough to stay in cache


def _pick(cmnd, tau_min, threshold):
    """Vectorised period choice for a batch of CMND rows -> (tau (float, sub-sample), confidence).

    Per row: the first lag >= tau_min whose CMND is under `threshold`, followed down to the bottom of that
    trough; if none is under the threshold, the global minimum. Then parabolic interpolation around it.
    """
    seg = cmnd[:, tau_min:]
    n, width = seg.shape
    rows = np.arange(n)
    below = seg < threshold
    has = below.any(axis=1)
    k = np.where(has, below.argmax(axis=1), seg.argmin(axis=1))
    # descend from the first dip while the curve keeps falling: stop at the first k' >= k with seg[k'+1] >= seg[k']
    rising = np.zeros((n, width), dtype=bool)
    rising[:, :-1] = seg[:, 1:] >= seg[:, :-1]
    rising &= np.arange(width)[None, :] >= k[:, None]
    stops = rising.any(axis=1)
    k = np.where(has, np.where(stops, rising.argmax(axis=1), width - 1), k)
    tau = k + tau_min
    confidence = np.clip(1.0 - cmnd[rows, tau], 0.0, 1.0)
    interior = (tau > 0) & (tau < cmnd.shape[1] - 1)
    t = np.where(interior, tau, 1)                                   # safe index where the neighbours do not exist
    a, b, c = cmnd[rows, t - 1], cmnd[rows, t], cmnd[rows, t + 1]
    den = a - 2 * b + c
    shift = np.where(interior & (den > 0), np.clip(0.5 * (a - c) / np.where(den > 0, den, 1.0), -1, 1), 0.0)
    return tau + shift, confidence


def estimate(y, sr, frame_length=1024, hop_length=256, fmin=None, fmax=None,
             threshold=0.15, rms_floor=0.02, **_):
    """Return a PitchTrack using a vectorized, FFT-based YIN implementation."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    fmin = 180.0 if fmin is None else float(fmin)
    fmax = 3000.0 if fmax is None else float(fmax)
    if sr <= 0 or not 0 < fmin < fmax <= sr / 2:
        raise ValueError("YIN requires 0 < fmin < fmax <= Nyquist.")
    tau_min = max(2, int(sr // fmax))
    tau_max = int(np.ceil(sr / fmin))
    if frame_length <= tau_max:
        raise ValueError("YIN frame_length must exceed the maximum period implied by fmin.")

    # centred frames, the same convention as utils.audio_math.frame_signal, but as a view (no copy)
    padded = np.pad(y, frame_length // 2)
    frames = sliding_window_view(padded, frame_length)[::hop_length]
    n_frames = 1 + (len(padded) - frame_length) // hop_length
    frames = frames[:n_frames]

    f0 = np.zeros(n_frames, dtype=float)
    confidence = np.zeros(n_frames, dtype=float)
    rms = np.zeros(n_frames, dtype=float)
    taus = np.arange(tau_max + 1)
    h = frame_length - tau_max                                    # fixed integration window of the difference function
    nfft = 1 << int(np.ceil(np.log2(frame_length)))               # >= frame_length is enough for lags 0..tau_max

    for start in range(0, n_frames, CHUNK):
        chunk = np.ascontiguousarray(frames[start:start + CHUNK])
        end = start + len(chunk)
        sq = chunk ** 2
        cumulative = np.concatenate([np.zeros((len(chunk), 1)), np.cumsum(sq, axis=1)], axis=1)
        rms[start:end] = np.sqrt(cumulative[:, -1] / frame_length)
        correlation = np.fft.irfft(
            np.conj(np.fft.rfft(chunk[:, :h], nfft)) * np.fft.rfft(chunk, nfft), nfft
        )[:, :tau_max + 1]
        difference = np.maximum(
            cumulative[:, h:h + 1] + (cumulative[:, h + taus] - cumulative[:, taus]) - 2 * correlation, 0)
        cmnd = np.ones_like(difference)
        cmnd[:, 1:] = difference[:, 1:] * taus[1:] / (np.cumsum(difference[:, 1:], axis=1) + 1e-12)
        period, score = _pick(cmnd, tau_min, threshold)
        f0[start:end] = sr / period
        confidence[start:end] = score

    if rms.size:
        confidence[rms <= rms_floor * rms.max()] = 0.0
        f0[confidence == 0.0] = 0.0
    return PitchTrack(frame_times(len(y), sr, hop_length), f0, confidence, "yin")
