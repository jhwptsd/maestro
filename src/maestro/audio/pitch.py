"""Public pitch-backend interface.

The backend names are ``crepe`` (default), ``yin``, ``pyin`` and ``basic_pitch``.
Optional dependencies are loaded only when the corresponding backend is selected.
"""
from importlib import import_module

from maestro.audio.backends.base import PitchTrack

DEFAULT_PITCH_BACKEND = "crepe"
PITCH_BACKENDS = ("yin", "pyin", "crepe", "basic_pitch")


def list_pitch_backends():
    """Return canonical backend identifiers accepted by estimate_pitch()."""
    return list(PITCH_BACKENDS)


def estimate_pitch(y, sr, backend=DEFAULT_PITCH_BACKEND, frame_length=None,
                   hop_length=256, fmin=None, fmax=None, **backend_options):
    """Estimate pitch using the selected backend and return a common PitchTrack.

    Args:
        y: Mono waveform as floating-point samples.
        sr: Waveform sample rate in Hz.
        backend: One of ``crepe``, ``yin``, ``pyin`` or ``basic_pitch``.
        frame_length: Analysis frame length used by frame-based backends.
        hop_length: Output frame spacing in samples.
        fmin/fmax: Optional backend frequency limits in Hz. Each backend has
            violin-oriented defaults; CREPE is limited to about 2006 Hz.
        **backend_options: Options specific to the selected backend. Examples
            include ``model``, ``device`` and ``batch_size`` for CREPE, or
            onset/frame thresholds for Basic Pitch.

    Basic Pitch is an event-oriented polyphonic transcriber. Its PitchTrack.notes
    contains all detected notes, while f0/confidence are a dominant-note projection
    for existing frame-oriented visualizations and feature consumers.
    """
    name = str(backend).strip().lower().replace("-", "_")
    if name not in PITCH_BACKENDS:
        choices = ", ".join(PITCH_BACKENDS)
        raise ValueError(f"Unknown pitch backend {backend!r}. Choose one of: {choices}.")

    module = import_module(f"maestro.audio.backends.{name}")
    call_options = dict(
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        **backend_options,
    )
    if frame_length is not None:
        call_options["frame_length"] = frame_length
    track = module.estimate(y, sr, **call_options)
    if not isinstance(track, PitchTrack):
        raise TypeError(f"Pitch backend {name!r} did not return a PitchTrack.")
    return track


def estimate_f0(y, sr, frame_length=None, hop_length=256, fmin=None, fmax=None,
                backend=DEFAULT_PITCH_BACKEND, **backend_options):
    """Compatibility wrapper returning ``(times, f0, confidence)`` arrays.

    New code should use estimate_pitch() to retain backend metadata and Basic Pitch
    note events. The tuple return preserves the original Maestro call convention.
    """
    track = estimate_pitch(
        y,
        sr,
        backend=backend,
        frame_length=frame_length,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        **backend_options,
    )
    return track.times, track.f0, track.confidence


__all__ = [
    "DEFAULT_PITCH_BACKEND",
    "PITCH_BACKENDS",
    "PitchTrack",
    "estimate_f0",
    "estimate_pitch",
    "list_pitch_backends",
]
