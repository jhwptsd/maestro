# Audio feature extraction and analysis.
import numpy as np

from maestro.audio.onset_detection import onset_strength
from maestro.audio.pitch import DEFAULT_PITCH_BACKEND, estimate_pitch
from maestro.utils.audio_math import frame_signal


def estimate_tempo(strength, sr, hop_length, bpm_range=(40, 220)):
    """Global tempo (BPM) from autocorrelation of the onset-strength envelope."""
    s = strength - strength.mean()
    if not s.any():
        return 0.0
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    lo, hi = (int(60 * sr / hop_length / b) for b in reversed(bpm_range))
    if hi >= len(ac) or lo >= hi:
        return 0.0
    return float(60 * sr / hop_length / (lo + np.argmax(ac[lo:hi + 1])))


def extract_features(y, sr, hop_length=256, frame_length=1024,
                     backend=DEFAULT_PITCH_BACKEND, pitch_track=None, pitch_options=None):
    """Extract aligned frame features, reusing a precomputed PitchTrack when supplied."""
    track = pitch_track or estimate_pitch(
        y,
        sr,
        backend=backend,
        frame_length=frame_length,
        hop_length=hop_length,
        **(pitch_options or {}),
    )
    strength = onset_strength(y, sr, n_fft=frame_length, hop_length=hop_length)
    frames = frame_signal(y, frame_length, hop_length)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    mag = np.abs(np.fft.rfft(frames * np.hanning(frame_length), axis=1))
    freqs = np.fft.rfftfreq(frame_length, 1 / sr)
    centroid = (mag * freqs).sum(axis=1) / (mag.sum(axis=1) + 1e-12)
    tempo = estimate_tempo(strength, sr, hop_length)

    n = min(len(track.times), len(track.f0), len(track.confidence), len(rms), len(centroid), len(strength))
    return [{
        "time": float(track.times[i]),
        "pitch_confidence": float(track.confidence[i]),
        "rms_energy": float(rms[i]),
        "spectral_centroid": float(centroid[i]),
        "onset_probability": float(strength[i]),
        "estimated_pitch": float(track.f0[i]) if track.confidence[i] > 0 else 0.0,
        "estimated_tempo": tempo,
    } for i in range(n)]
