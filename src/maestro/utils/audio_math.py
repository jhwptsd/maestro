# Small numeric helpers shared by audio/, score/, alignment/ and assessment/.
# Keeping them here ensures every module uses the same pitch convention (A4 = 440 Hz, equal temperament)
# and the same framing grid, so frame indices line up between pitch, onset and feature extraction.
import numpy as np


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
    """
    y = np.pad(np.asarray(y, dtype=float), frame_length // 2)
    n = 1 + (len(y) - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n)[:, None]
    return y[idx]
