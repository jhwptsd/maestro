"""CREPE without PyTorch: framing, decoding and the network through ONNX Runtime, all in numpy (new file; used by audio/backends/crepe.py).

torchcrepe is a thin layer around the network: it frames the 16 kHz audio, normalises each frame, runs the network, masks the pitch bins outside the
allowed range and decodes. Maestro already replaced its decoder (a Viterbi path refined to sub-bin precision, no dither), and the network runs through ONNX
Runtime (crepe_onnx.py). This file does the rest in numpy, so a computer that has onnxruntime and the exported network needs neither PyTorch nor librosa (whose
Viterbi was the other dependency): the installed desktop app is hundreds of megabytes smaller, and the same code is what a phone with an ONNX Runtime would use.
The tests check it against torchcrepe frame by frame. Constants are torchcrepe's.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SAMPLE_RATE = 16000          # CREPE's input rate
WINDOW = 1024                # samples per frame
PITCH_BINS = 360
CENTS_PER_BIN = 20
CENTS_OFFSET = 1997.3794084376191        # cents of pitch bin 0
MAX_FMAX = 2006.0            # the highest pitch the network covers (Hz)
REFINE_BINS = 4              # bins either side of the Viterbi path used for the weighted average
CHUNK = 2048                 # frames normalised and sent to the network at a time (bounds memory)
_transition = None


def transition_matrix():
    """Viterbi transition matrix between the 360 bins: a triangle of width 12 bins, each row summing to 1 (torchcrepe's)."""
    global _transition
    if _transition is None:
        bins = np.arange(PITCH_BINS)
        t = np.maximum(12 - np.abs(bins[:, None] - bins[None, :]), 0).astype(float)
        _transition = t / t.sum(axis=1, keepdims=True)
    return _transition


def frame_count(n_samples, hop):
    return 1 + n_samples // hop


def frames(audio, hop, start=0, stop=None):
    """Normalised network input for frames start..stop of `audio` (16 kHz mono): windows of 1024 samples every `hop`, the audio zero-padded by half a
    window at both ends, each window mean-centred and divided by its (unbiased) standard deviation, at least 1e-10, exactly as torchcrepe does."""
    total = frame_count(len(audio), hop)
    stop = total if stop is None else min(stop, total)
    padded = np.pad(np.asarray(audio, np.float32), (WINDOW // 2, WINDOW // 2))
    windows = sliding_window_view(padded, WINDOW)[::hop][start:stop].astype(np.float32)          # (frames, 1024), a copy
    windows -= windows.mean(axis=1, keepdims=True)
    windows /= np.maximum(1e-10, windows.std(axis=1, ddof=1, keepdims=True))
    return windows


def bin_range(fmin, fmax):
    """(first bin, one past the last bin) allowed for pitches fmin..fmax Hz."""
    def to_bin(f, rounding):
        return int(rounding((1200 * np.log2(f / 10.0) - CENTS_OFFSET) / CENTS_PER_BIN))
    return to_bin(fmin, np.floor), to_bin(fmax, np.ceil)


def viterbi(prob, transition):
    """Most likely state path through `prob` (states, frames) with `transition[i, j]` = P(j next | i now), uniform start: librosa.sequence.viterbi in numpy."""
    tiny = np.finfo(prob.dtype).tiny
    log_trans, log_prob = np.log(transition + tiny), np.log(prob + tiny)
    states, steps = prob.shape
    value = log_prob[:, 0] - np.log(states)
    pointer = np.zeros((steps, states), dtype=np.int64)
    for t in range(1, steps):
        scores = value[:, None] + log_trans                                  # [from, to]
        pointer[t] = scores.argmax(axis=0)
        value = log_prob[:, t] + scores[pointer[t], np.arange(states)]
    path = np.empty(steps, dtype=np.int64)
    path[-1] = value.argmax()
    for t in range(steps - 1, 0, -1):
        path[t - 1] = pointer[t, path[t]]
    return path


def decode(probabilities, lowest, highest):
    """Network output (frames, 360) -> (bins, frequency in Hz, periodicity) per frame. Bins outside lowest..highest are ignored; the Viterbi path runs over the
    outputs normalised across bins, and each frame is refined to sub-bin precision by a weighted average of the bins around the path (no dither)."""
    probs = np.asarray(probabilities, dtype=np.float64)
    valid = np.zeros(PITCH_BINS, bool)
    valid[max(lowest, 0):min(highest, PITCH_BINS)] = True
    weights = np.where(valid, probs, 0.0).clip(min=0)
    emission = np.where(valid, weights + 1e-6, 0.0)
    emission /= np.maximum(emission.sum(axis=1, keepdims=True), 1e-12)
    bins = viterbi(emission.T, transition_matrix())
    pos = bins[:, None] + np.arange(-REFINE_BINS, REFINE_BINS + 1)
    inside = (pos >= 0) & (pos < PITCH_BINS)
    w = np.take_along_axis(weights, pos.clip(0, PITCH_BINS - 1), axis=1) * inside
    total = w.sum(axis=1)
    refined = np.where(total > 0, (w * pos).sum(axis=1) / np.maximum(total, 1e-9), bins.astype(float))
    frequency = 10 * 2 ** ((CENTS_PER_BIN * refined + CENTS_OFFSET) / 1200)
    return bins, frequency, probs[np.arange(len(bins)), bins]


def predict(audio, hop, fmin, fmax, network):
    """(f0 Hz, periodicity) per CREPE frame of 16 kHz mono `audio`. `network(frames (N, 1024)) -> (N, 360)` runs the model (crepe_onnx.make_run)."""
    total = frame_count(len(audio), hop)
    out = np.concatenate([network(frames(audio, hop, i, i + CHUNK)) for i in range(0, total, CHUNK)])
    _, f0, periodicity = decode(out, *bin_range(fmin, fmax))
    return f0, periodicity
