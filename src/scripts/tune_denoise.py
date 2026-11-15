"""Choose audio-denoising parameters from synthetic noisy/clean pairs ("training" without a neural net).

Usage: PYTHONPATH=src:src/tests python src/scripts/tune_denoise.py

For every (reduction_db, alpha) in the grid it denoises recordings with several background-noise levels
and measures (see tests/noise_fixture.py):
  snr_gain       SNR improvement against the clean recording (dB)  <- what we maximise
  vibrato        depth of the pitch modulation after / before denoising (need >= 0.9)
  vibrato_rate   modulation rate change in Hz (need <= 0.3)
  bow_noise      bow-noise (>3 kHz) energy in the note after / before (need 0.85 .. 1.25)
  pitch_cents    median shift of transcribed note pitches vs the clean recording (need <= 3)
Settings that violate a need are rejected; the best of the rest is printed. Copy it into
AUDIO_DEFAULTS in src/maestro/preprocessing/denoise.py. Real recordings: replace the synthetic noise with
clips of your own room noise before trusting the numbers.
"""
import itertools

import numpy as np

from maestro.audio.transcription import transcribe
from maestro.preprocessing.denoise import denoise_audio
from minuet import play_minuet
from noise_fixture import SR, background_noise, band_energy, bowed_note, pitch_modulation, snr_db, vibrato_note

# reduction is capped at 15 dB on purpose: it is the "never erase" limit for quiet bow noise on real recordings
# (20 dB gains only ~0.5 dB more SNR on synthetic data and removes more of the quiet bow noise)
GRID = {"reduction_db": [6, 9, 12, 15], "alpha": [0.8, 0.9, 0.95, 0.98]}
LEVELS = (0.005, 0.02, 0.05)             # background noise rms (clean Minuet rms is ~0.2)


def evaluate(reduction_db, alpha):
    rng = np.random.default_rng(0)
    out = {"snr_gain": [], "vibrato": [], "vibrato_rate": [], "bow_noise": [], "pitch_cents": []}
    clean = play_minuet(noise=0.02, seed=1)
    ref_notes = transcribe(clean, SR)
    for level in LEVELS:
        noisy = clean + background_noise(len(clean), rng, level).astype(np.float32)
        den, _ = denoise_audio(noisy, SR, reduction_db=reduction_db, alpha=alpha)
        out["snr_gain"].append(snr_db(clean, den) - snr_db(clean, noisy))
        notes = transcribe(den, SR)
        if len(notes) == len(ref_notes):
            out["pitch_cents"].append(np.median([abs(1200 * np.log2(a["hz"] / b["hz"])) for a, b in zip(notes, ref_notes)]))
        else:
            out["pitch_cents"].append(99.0)                      # note count changed: reject
        vib, span = vibrato_note()
        vib_noisy = vib + background_noise(len(vib), rng, level).astype(np.float32)
        vib_den, _ = denoise_audio(vib_noisy, SR, reduction_db=reduction_db, alpha=alpha)
        (d0, r0), (d1, r1) = pitch_modulation(vib, SR, span), pitch_modulation(vib_den, SR, span)
        out["vibrato"].append(d1 / d0)
        out["vibrato_rate"].append(abs(r1 - r0))
        bowed, _, span = bowed_note(rng)
        bow_den, _ = denoise_audio(bowed + background_noise(len(bowed), rng, level).astype(np.float32), SR,
                                   reduction_db=reduction_db, alpha=alpha)
        out["bow_noise"].append(band_energy(bow_den, 4000, 9000, *span) / band_energy(bowed, 4000, 9000, *span))
    return {k: float(np.mean(v)) if k != "pitch_cents" else float(np.max(v)) for k, v in out.items()}


def acceptable(m):
    return (m["vibrato"] >= 0.9 and m["vibrato_rate"] <= 0.3 and 0.85 <= m["bow_noise"] <= 1.25
            and m["pitch_cents"] <= 3)


if __name__ == "__main__":
    rows = []
    for r, a in itertools.product(GRID["reduction_db"], GRID["alpha"]):
        m = evaluate(r, a)
        rows.append((r, a, m, acceptable(m)))
        print(f"reduction {r:2d} dB alpha {a:.2f}: snr +{m['snr_gain']:4.1f} dB  vibrato x{m['vibrato']:.2f}  "
              f"rate d{m['vibrato_rate']:.2f} Hz  bow noise x{m['bow_noise']:.2f}  pitch {m['pitch_cents']:.1f} c  "
              f"{'ok' if rows[-1][3] else 'REJECTED'}")
    good = [r for r in rows if r[3]]
    best = max(good, key=lambda r: r[2]["snr_gain"]) if good else None
    print("\nbest acceptable:", None if best is None else f"reduction_db={best[0]}, alpha={best[1]}")
