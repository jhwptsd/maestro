# Audio feature extraction and analysis.
import numpy as np

from maestro.audio.onset_detection import onset_strength
from maestro.audio.pitch import DEFAULT_PITCH_BACKEND, estimate_pitch
from maestro.utils.audio_math import frame_rms, frame_signal


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
    rms = frame_rms(y, frame_length, hop_length)
    centroid = _spectral_centroid(frame_signal(y, frame_length, hop_length), sr)
    tempo = estimate_tempo(strength, sr, hop_length)

    n = min(len(track.times), len(rms), len(strength))
    voiced = track.confidence[:n] > 0
    columns = zip(track.times[:n].tolist(), track.confidence[:n].tolist(), rms[:n].tolist(), centroid[:n].tolist(),
                  strength[:n].tolist(), np.where(voiced, track.f0[:n], 0.0).tolist())
    return [{"time": t, "pitch_confidence": c, "rms_energy": r, "spectral_centroid": sc, "onset_probability": o,
             "estimated_pitch": f, "estimated_tempo": tempo} for t, c, r, sc, o, f in columns]


def _spectral_centroid(frames, sr, chunk=512):
    """Spectral centroid (Hz) of every frame, computed in chunks so the full spectrogram never sits in memory."""
    window = np.hanning(frames.shape[1])
    freqs = np.fft.rfftfreq(frames.shape[1], 1 / sr)
    out = np.empty(len(frames))
    for start in range(0, len(frames), chunk):
        mag = np.abs(np.fft.rfft(frames[start:start + chunk] * window, axis=1))
        out[start:start + chunk] = (mag * freqs).sum(axis=1) / (mag.sum(axis=1) + 1e-12)
    return out
