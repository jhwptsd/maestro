"""Shared result type and frame helpers for pitch backends."""
from dataclasses import dataclass

import numpy as np


@dataclass
class PitchTrack:
    """Backend-independent frame pitches plus optional note events.

    times, f0 and confidence are one-dimensional frame-aligned arrays. confidence is
    normalized to [0, 1]. notes is None for frame-only trackers and a list (possibly
    empty) for event-based trackers such as Basic Pitch.
    """

    times: np.ndarray
    f0: np.ndarray
    confidence: np.ndarray
    backend: str
    notes: list[dict] | None = None
    runtime: str = ""                 # what ran the network, e.g. "onnx:DmlExecutionProvider" (informational)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float).reshape(-1)
        self.f0 = np.asarray(self.f0, dtype=float).reshape(-1)
        self.confidence = np.asarray(self.confidence, dtype=float).reshape(-1)
        if not (len(self.times) == len(self.f0) == len(self.confidence)):
            raise ValueError("PitchTrack times, f0 and confidence must have equal lengths.")
        self.f0 = np.nan_to_num(self.f0, nan=0.0, posinf=0.0, neginf=0.0)
        self.confidence = np.clip(
            np.nan_to_num(self.confidence, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0
        )

    def __iter__(self):
        """Preserve the historical three-array unpacking API."""
        yield self.times
        yield self.f0
        yield self.confidence


def frame_times(length, sr, hop_length):
    """Centered frame timestamps matching Maestro's existing framing convention."""
    n_frames = 1 + int(length) // int(hop_length)
    return np.arange(n_frames, dtype=float) * hop_length / sr
