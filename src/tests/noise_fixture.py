"""Test/tuning fixtures for denoising: background noise, a vibrato note, a bowed note with bow noise,
and the measurements used to check that denoising keeps what matters (vibrato, bow noise, tone)."""
import numpy as np

SR = 22050


def background_noise(n, rng, level=0.02):
    """Stationary room-like noise (white hiss + pink rumble + 50 Hz mains hum with harmonics), rms=level."""
    spec = np.fft.rfft(rng.standard_normal(n))
    pink = np.fft.irfft(spec / np.sqrt(np.maximum(np.arange(len(spec)), 1)), n)
    t = np.arange(n) / SR
    hum = sum(np.sin(2 * np.pi * 50 * k * t) / k for k in (1, 2, 3, 4))
    x = rng.standard_normal(n) + pink / pink.std() + 1.5 * hum
    return level * x / x.std()


def snr_db(clean, estimate):
    """Signal-to-error ratio of `estimate` against `clean` (dB)."""
    n = min(len(clean), len(estimate))
    return float(10 * np.log10((clean[:n] ** 2).sum() / ((estimate[:n] - clean[:n]) ** 2).sum()))


def band_energy(y, lo, hi, t0, t1, sr=SR):
    """Spectral energy of y[t0:t1] between lo and hi Hz."""
    seg = y[int(t0 * sr):int(t1 * sr)]
    power = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    return float(power[(f >= lo) & (f < hi)].sum())


def vibrato_note(freq=440.0, seconds=3.0, depth_cents=40.0, rate_hz=5.5, lead=0.6, sr=SR):
    """One sustained violin-ish note with vibrato after `lead` s of silence -> (audio, (start, end) seconds)."""
    n = int((lead + seconds + 0.4) * sr)
    t = np.arange(n) / sr
    on = (t >= lead) & (t < lead + seconds)
    env = np.minimum(1, (t - lead) / 0.05) * np.minimum(1, (lead + seconds - t) / 0.05) * on
    f = freq * 2 ** (depth_cents / 1200 * np.sin(2 * np.pi * rate_hz * t))
    phase = np.cumsum(f) / sr * 2 * np.pi
    y = sum(a * np.sin(k * phase) for k, a in enumerate([1, .6, .4, .25, .15, .1], start=1)) * env * 0.3
    return y.astype(np.float32), (lead, lead + seconds)


def bow_noise(n, rng, level=0.02, sr=SR):
    """Bow hiss/scratch: broadband noise above 3 kHz (non-tonal, so it does not mask the harmonics)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[np.fft.rfftfreq(n, 1 / sr) < 3000] = 0
    x = np.fft.irfft(spec, n)
    return level * x / x.std()


def bowed_note(rng, bow_level=0.02, sr=SR):
    """Tone + bow noise that exists only while the note sounds -> (audio, clean_tone, (start, end))."""
    tone, span = vibrato_note(depth_cents=20, sr=sr)
    env = (np.abs(tone) > 1e-4).astype(float)
    env = np.convolve(env, np.ones(441) / 441, mode="same")
    return (tone + (bow_noise(len(tone), rng, bow_level, sr) * env).astype(np.float32)), tone, span


def pitch_modulation(y, sr, span):
    """(depth in cents as std of the f0 contour, dominant modulation rate in Hz) inside span=(t0, t1)."""
    from maestro.audio.pitch import estimate_f0
    t, f0, conf = estimate_f0(y, sr)
    sel = (t > span[0] + 0.3) & (t < span[1] - 0.3) & (conf > 0.5)
    cents = 1200 * np.log2(f0[sel] / np.median(f0[sel]))
    spectrum = np.abs(np.fft.rfft(cents - cents.mean(), 4096))
    freqs = np.fft.rfftfreq(4096, t[1] - t[0])
    ok = (freqs > 2) & (freqs < 10)
    return float(cents.std()), float(freqs[ok][np.argmax(spectrum[ok])])
