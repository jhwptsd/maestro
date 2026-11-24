# Tone descriptors per note
#
# Called per aligned note by assessment/tone.py (and assessment/expression.py for loudness). Works on the audio
# samples of the performance (Performance.audio["samples"]) and the note's time span and pitch from
# audio/transcription.py. The descriptors follow the tone dimensions of Giraldo et al. (2019, "Automatic Assessment
# of Tone Quality in Violin Music Performance", trained on the TELMI data): attack clarity, dynamic stability,
# timbre richness/brightness and noise (a scratchy tone has a high noise ratio). Computed with plain DSP; the optional
# neural judgement lives in assessment/neural.py.
import numpy as np

HOP = 256
FRAME = 1024
HARMONICS = 20


def _frame_rms(x):
    n = 1 + max(0, len(x) - FRAME) // HOP
    return np.array([np.sqrt(np.mean(x[i * HOP:i * HOP + FRAME] ** 2) + 1e-12) for i in range(n)])


def note_tone(y, sr, start, end, hz):
    """Descriptors of the note sounding in [start, end] seconds with fundamental `hz` (None if shorter than 80 ms).

    -> {"loudness_db" (RMS dBFS), "attack_s" (time to reach 90% of the peak level; short = clean attack),
    "dynamic_stability" (0-1: 1 = steady level in the sustained part), "brightness" (spectral centroid in harmonic
    numbers; higher = brighter / more upper partials), "harmonic_ratio" (0-1 share of energy at the harmonics; low =
    noisy / scratchy / whistly)}. Pitch steadiness is not measured here (see audio/vibrato.py and assessment/intonation.py).
    """
    seg = np.asarray(y[int(start * sr):int(end * sr)], float)
    if len(seg) < 0.08 * sr or hz <= 0:
        return None
    rms = _frame_rms(seg)
    peak = rms.max()
    attack = float(np.argmax(rms >= 0.9 * peak) * HOP / sr)
    mid = rms[len(rms) // 5: len(rms) - len(rms) // 5] if len(rms) >= 5 else rms
    stability = float(np.clip(1 - mid.std() / (mid.mean() + 1e-12), 0, 1))

    centre = seg[len(seg) // 2 - min(len(seg) // 2, 2048):len(seg) // 2 + min(len(seg) // 2, 2048)]
    mag = np.abs(np.fft.rfft(centre * np.hanning(len(centre)), 4 * len(centre))) ** 2
    freqs = np.fft.rfftfreq(4 * len(centre), 1 / sr)
    harmonic = np.zeros(len(freqs), bool)
    for k in range(1, HARMONICS + 1):                                # +-3% around each harmonic counts as tone
        harmonic |= np.abs(freqs - k * hz) <= 0.03 * k * hz
    total = mag.sum() + 1e-12
    return {"loudness_db": float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)),
            "attack_s": attack, "dynamic_stability": stability,
            "brightness": float((mag * freqs).sum() / total / hz),
            "harmonic_ratio": float(mag[harmonic].sum() / total)}
