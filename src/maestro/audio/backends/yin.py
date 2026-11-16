"""NumPy implementation of the YIN fundamental-frequency estimator."""
import numpy as np

from maestro.audio.backends.base import PitchTrack, frame_times
from maestro.utils.audio_math import frame_signal


def _pick(row, tau_min, threshold):
    """Choose the first trough below threshold, or the global minimum."""
    seg = row[tau_min:]
    below = np.flatnonzero(seg < threshold)
    if below.size:
        k = below[0]
        while k + 1 < len(seg) and seg[k + 1] < seg[k]:
            k += 1
    else:
        k = int(np.argmin(seg))
    tau = k + tau_min
    confidence = 1.0 - row[tau]
    if 0 < tau < len(row) - 1:
        a, b, c = row[tau - 1], row[tau], row[tau + 1]
        denominator = a - 2 * b + c
        if denominator > 0:
            tau += float(np.clip(0.5 * (a - c) / denominator, -1, 1))
    return tau, float(np.clip(confidence, 0, 1))


def estimate(y, sr, frame_length=1024, hop_length=256, fmin=None, fmax=None,
             threshold=0.15, rms_floor=0.02, **_):
    """Return a PitchTrack using the original vectorized, FFT-based YIN implementation."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    fmin = 180.0 if fmin is None else float(fmin)
    fmax = 3000.0 if fmax is None else float(fmax)
    if sr <= 0 or not 0 < fmin < fmax <= sr / 2:
        raise ValueError("YIN requires 0 < fmin < fmax <= Nyquist.")
    tau_min = max(2, int(sr // fmax))
    tau_max = int(np.ceil(sr / fmin))
    if frame_length <= tau_max:
        raise ValueError("YIN frame_length must exceed the maximum period implied by fmin.")

    frames = frame_signal(y, frame_length, hop_length)
    n_frames = len(frames)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    f0 = np.zeros(n_frames, dtype=float)
    confidence = np.zeros(n_frames, dtype=float)
    taus = np.arange(tau_max + 1)
    h = frame_length - tau_max
    nfft = 1 << int(np.ceil(np.log2(frame_length + h)))

    for start in range(0, n_frames, 256):
        chunk = frames[start:start + 256]
        correlation = np.fft.irfft(
            np.conj(np.fft.rfft(chunk[:, :h], nfft))
            * np.fft.rfft(chunk, nfft),
            nfft,
        )[:, :tau_max + 1]
        cumulative = np.concatenate(
            [np.zeros((len(chunk), 1)), np.cumsum(chunk ** 2, axis=1)], axis=1
        )
        difference = np.maximum(
            cumulative[:, h:h + 1]
            + (cumulative[:, h + taus] - cumulative[:, taus])
            - 2 * correlation,
            0,
        )
        cmnd = np.ones_like(difference)
        cmnd[:, 1:] = difference[:, 1:] * taus[1:] / (
            np.cumsum(difference[:, 1:], axis=1) + 1e-12
        )
        for offset, row in enumerate(cmnd):
            period, score = _pick(row, tau_min, threshold)
            f0[start + offset] = sr / period
            confidence[start + offset] = score

    if rms.size:
        confidence[rms <= rms_floor * rms.max()] = 0.0
        f0[confidence == 0.0] = 0.0
    return PitchTrack(frame_times(len(y), sr, hop_length), f0, confidence, "yin")
