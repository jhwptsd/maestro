# Audio feature extraction and analysis
#
# Aggregates the outputs of pitch.py and onset_detection.py (vibrato.py and tone_quality.py are
# not implemented yet) into the common per-frame JSON below, stored in Performance.audio_features.
# Consumed by alignment/*, assessment/*, and fusion/audio_encoder.py.
#
# Should format into a common JSON
# ex. {
#    "time": 12.35,
#    "pitch_confidence": 0.97,
#    "rms_energy": 0.18,
#    "spectral_centroid": 3200.0,
#    "onset_probability": 0.82
#    "estimated_pitch": 440.0,
#    "estimated_tempo": 120.0
# }
# Note-level results (audio/transcription.py) are kept separately in Performance.audio_features
# under the "notes" key by maestro.analyze, because alignment works on notes, not frames.
import numpy as np

from maestro.audio.onset_detection import onset_strength
from maestro.audio.pitch import estimate_f0
from maestro.utils.audio_math import frame_signal


def estimate_tempo(strength, sr, hop_length, bpm_range=(40, 220)):
    """Global tempo (BPM) from the autocorrelation of the onset strength envelope; 0 if undetectable."""
    s = strength - strength.mean()
    if not s.any():
        return 0.0
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    lo, hi = (int(60 * sr / hop_length / b) for b in reversed(bpm_range))
    if hi >= len(ac) or lo >= hi:
        return 0.0
    return float(60 * sr / hop_length / (lo + np.argmax(ac[lo:hi + 1])))


def extract_features(y, sr, hop_length=256, frame_length=1024):
    """Audio -> list of per-frame dicts in the schema above (all frames share one time grid)."""
    t, f0, conf = estimate_f0(y, sr, frame_length=frame_length, hop_length=hop_length)
    strength = onset_strength(y, sr, n_fft=frame_length, hop_length=hop_length)
    frames = frame_signal(y, frame_length, hop_length)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    mag = np.abs(np.fft.rfft(frames * np.hanning(frame_length), axis=1))
    freqs = np.fft.rfftfreq(frame_length, 1 / sr)
    centroid = (mag * freqs).sum(axis=1) / (mag.sum(axis=1) + 1e-12)
    tempo = estimate_tempo(strength, sr, hop_length)
    return [{"time": float(t[i]), "pitch_confidence": float(conf[i]), "rms_energy": float(rms[i]),
             "spectral_centroid": float(centroid[i]), "onset_probability": float(strength[i]),
             "estimated_pitch": float(f0[i]) if conf[i] > 0 else 0.0, "estimated_tempo": tempo}
            for i in range(len(t))]
