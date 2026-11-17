"""Librosa pYIN backend (probabilistic YIN with temporal decoding)."""
import numpy as np

from maestro.utils.audio_math import frame_rms
from maestro.audio.backends.base import PitchTrack, frame_times


def estimate(y, sr, frame_length=2048, hop_length=256, fmin=None, fmax=None,
             rms_floor_db=60.0, **options):
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "The pYIN backend requires librosa. Install it with: pip install -e '.[pyin]'"
        ) from exc

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    fmin = 175.0 if fmin is None else float(fmin)
    fmax = 4200.0 if fmax is None else float(fmax)
    if not 0 < fmin < fmax <= sr / 2:
        raise ValueError("pYIN requires 0 < fmin < fmax <= Nyquist.")

    f0, voiced, voiced_probability = librosa.pyin(
        y,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
        fill_na=np.nan,
        **options,
    )
    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    confidence = np.nan_to_num(voiced_probability, nan=0.0, posinf=0.0, neginf=0.0)
    confidence = np.where(voiced, confidence, 0.0)

    rms = frame_rms(y, frame_length, hop_length)
    if rms.size and rms.max() > 0:
        floor = rms.max() * (10.0 ** (-float(rms_floor_db) / 20.0))
        active = rms > floor
        n = min(len(active), len(f0))
        confidence[:n] = np.where(active[:n], confidence[:n], 0.0)
        f0[:n] = np.where(active[:n], f0[:n], 0.0)

    n = min(len(f0), len(confidence))
    times = frame_times(len(y), sr, hop_length)[:n]
    return PitchTrack(times, f0[:n], confidence[:n], "pyin")
