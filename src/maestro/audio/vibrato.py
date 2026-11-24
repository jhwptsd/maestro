# Detect vibrato from audio - rate and extent
#
# Operates on the frame pitch (Hz per frame, 0 = unvoiced) that audio/features.py stores in
# Performance.audio_features["frames"] ("time", "estimated_pitch"). Called per aligned note by
# assessment/expression.py (is the vibrato even, wide enough, present on long notes?).
# Violin vibrato is a roughly sinusoidal pitch oscillation of about 4.5-8 Hz and +-10-35 cents around the note's
# centre pitch; it is measured, never removed (preprocessing/denoise.py is built to leave it intact).
import numpy as np

RATE_RANGE_HZ = (4.0, 9.0)       # search band for the vibrato rate
MIN_EXTENT_CENTS = 8.0           # smaller oscillations are tracker jitter / natural drift, not vibrato
MIN_SECONDS = 0.3                # shorter spans cannot show two vibrato cycles


def note_vibrato(times, f0, start, end):
    """Vibrato in the span [start, end] seconds of a frame pitch track.

    -> {"present": bool, "rate_hz", "extent_cents" (half peak-to-peak, mean over cycles), "regularity" (0-1: how much
    of the contour's variation is one steady oscillation)}; the three numbers are None when there is too little
    voiced pitch to tell. `present` needs a rate in RATE_RANGE_HZ and an extent of at least MIN_EXTENT_CENTS.
    """
    times, f0 = np.asarray(times, float), np.asarray(f0, float)
    sel = (times >= start) & (times <= end) & (f0 > 0)
    if end - start < MIN_SECONDS or sel.sum() < 8:
        return {"present": False, "rate_hz": None, "extent_cents": None, "regularity": None}
    t = times[sel]
    cents = 1200 * np.log2(f0[sel])
    cents = cents - np.polyval(np.polyfit(t, cents, 2), t)           # remove the note's slow drift / glide
    dt = np.median(np.diff(t))
    grid = np.arange(t[0], t[-1], dt)                                # even grid: voiced gaps are bridged
    cents = np.interp(grid, t, cents) * np.hanning(len(grid))
    spectrum = np.abs(np.fft.rfft(cents, 8 * len(grid))) ** 2
    freqs = np.fft.rfftfreq(8 * len(grid), dt)
    band = (freqs >= RATE_RANGE_HZ[0]) & (freqs <= RATE_RANGE_HZ[1])
    if not band.any() or spectrum[1:].sum() == 0:
        return {"present": False, "rate_hz": None, "extent_cents": None, "regularity": None}
    peak = np.argmax(np.where(band, spectrum, 0))
    rate = float(freqs[peak])
    # amplitude of the dominant oscillation (Hann window halves it, hence * 4 / n)
    extent = float(4 * np.sqrt(spectrum[peak]) / len(grid))
    # a window of +-1 Hz around the peak counts as "the vibrato"; its share of all non-DC power is the regularity
    near = np.abs(freqs - rate) <= 1.0
    regularity = float(spectrum[near].sum() / spectrum[1:].sum())
    return {"present": bool(extent >= MIN_EXTENT_CENTS), "rate_hz": rate, "extent_cents": extent,
            "regularity": regularity}
