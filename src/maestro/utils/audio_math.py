# Small numeric helpers shared by audio/, score/, alignment/ and assessment/.
# Keeping them here ensures every module uses the same pitch convention (A4 = 440 Hz, equal temperament)
# and the same framing grid, so frame indices line up between pitch, onset and feature extraction.
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def hz_to_midi(hz):
    """Frequency (Hz) -> fractional MIDI note number."""
    return 69.0 + 12.0 * np.log2(np.asarray(hz, dtype=float) / 440.0)


def midi_to_hz(midi):
    """MIDI note number (may be fractional) -> frequency (Hz)."""
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


def cents_between(hz, ref_hz):
    """Signed deviation of `hz` from `ref_hz` in cents (+ = sharp, - = flat)."""
    return 1200.0 * np.log2(np.asarray(hz, dtype=float) / np.asarray(ref_hz, dtype=float))


def frame_signal(y, frame_length, hop_length):
    """Split `y` into centred, overlapping frames -> array (n_frames, frame_length).

    Frame i is centred on sample i*hop_length (zero padded at the edges), so n_frames = 1 + len(y)//hop
    and frame time = i*hop/sr. audio/pitch.py and audio/onset_detection.py both use this grid.

    The result is a read-only *view* of one padded copy of the signal, not an (n_frames x frame_length)
    copy: with 75% overlap that is 4x less memory and avoids building a huge index array. Do arithmetic
    on it (squares, windowing, FFTs all return new arrays) but do not write into it.
    """
    y = np.pad(np.asarray(y, dtype=float), frame_length // 2)
    n = 1 + (len(y) - frame_length) // hop_length
    return sliding_window_view(y, frame_length)[::hop_length][:n]


def frame_rms(y, frame_length, hop_length, chunk=512):
    """RMS of every frame of `frame_signal(y, ...)`, without ever materialising frame-sized temporaries.

    Sums squares chunk by chunk over the strided frame view (einsum); shared by onset detection, feature
    extraction and the pitch backends, which all need loudness on the same frame grid.
    """
    frames = frame_signal(y, frame_length, hop_length)
    out = np.empty(len(frames))
    for start in range(0, len(frames), chunk):
        block = frames[start:start + chunk]
        out[start:start + chunk] = np.einsum("ij,ij->i", block, block)
    return np.sqrt(out / frame_length)
