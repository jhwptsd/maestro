# Isolate the solo violin of a reference recording (rewritten from the outline).
#
# A professional recording often has an orchestra or piano, an audience and room sound under the soloist. Two steps, both optional
# frameworks or plain DSP:
#   1. `preseparate` (optional, `pip install demucs`): Demucs (htdemucs) removes drums, bass and vocals, keeping the "other" stem
#      (violin + orchestra). Off unless asked for: it is slow and does not take the orchestra away on its own.
#   2. Score-informed harmonic masking: the score and its alignment say which pitch the soloist plays when (`guide_f0`: the pitch tracker's
#      value when it is close to the written one, else the written one), and `isolate_solo` keeps only the spectrogram bins at that pitch and its
#      harmonics (+-70 cents, enough for vibrato and expressive intonation) and drops everything else: accompaniment and noise that do not
#      share the soloist's partials. Whether the soloist is really there is judged per frame from the signal-to-floor ratio of the comb: the mean
#      power at the expected partials against the median power of the whole frame (a robust noise floor that narrow peaks do not move). A soloist
#      stands well above it, however loud the orchestra is elsewhere; where the soloist should be playing but the comb is level with the floor
#      (the orchestra drowns the soloist, or the soloist is not playing what was written) the frame is marked absent, and long enough stretches are
#      reported as gaps instead of being measured. Rests are not gaps.
# Limit: an accompaniment note at the soloist's own pitch (or an octave or fifth above) shares the partials and cannot be separated this way;
# the presence figure is what tells the later stages how far to trust a note.
import numpy as np
from scipy.signal import istft, stft

from maestro.utils.audio_math import cents_between

NPERSEG, HOP = 4096, 1024         # STFT: 186 ms windows every 46 ms at 22.05 kHz (5.4 Hz bins resolve a violin's partials)
HARMONICS = 14
TOLERANCE_CENTS = 70.0            # half-width of the band around each partial
MIN_SNR_DB = 10.0                 # comb-to-floor ratio below which the soloist counts as absent from a frame
MIN_GAP_S = 0.3                   # shorter absences are not reported as gaps


def preseparate(y, sr):
    """The 'other' stem of Demucs (no drums, bass or vocals) as mono at `sr`; None when demucs is not installed."""
    try:
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError:
        return None
    model = get_model("htdemucs").eval()
    wav = torchaudio.functional.resample(torch.tensor(y, dtype=torch.float32), sr, model.samplerate)[None].repeat(model.audio_channels, 1)
    wav = (wav - wav.mean()) / (wav.std() + 1e-8)                        # Demucs expects a normalised mix
    with torch.no_grad():
        stems = apply_model(model, wav[None], device="cpu")[0]           # (sources, channels, samples)
    other = stems[model.sources.index("other")].mean(0)
    return torchaudio.functional.resample(other, model.samplerate, sr).numpy()[:len(y)] * float(y.std() / (other.std() + 1e-8))


def guide_f0(times, f0, notes, alignment, score, max_cents=100.0):
    """Frame pitch (Hz) of the soloist, guided by the score: inside each performed note that was aligned to a written one, the tracker's
    pitch where it is within `max_cents` of the written pitch, else the written pitch; 0 outside the notes (rests, or the soloist absent)."""
    times, f0 = np.asarray(times, float), np.asarray(f0, float)
    guide = np.zeros(len(times))
    for a in alignment:
        if a["perf_idx"] is None:
            continue
        n, written = notes[a["perf_idx"]], score[a["score_idx"]]["hz"]
        span = (times >= n["onset"]) & (times <= n["offset"])
        tracked = span & (f0 > 0)
        close = np.zeros(len(times), bool)
        close[tracked] = np.abs(cents_between(f0[tracked], written)) <= max_cents
        guide[span] = written
        guide[close] = f0[close]
    return guide


def isolate_solo(y, sr, times, guide):
    """Keep only the soloist's partials -> {"samples" (as long as y), "frame_times", "snr_db" (per STFT frame: how far the expected partials
    stand above the noise floor), "present" (bool per frame), "gaps": [{"start", "end", "reason": "accompaniment_dominates"}]}."""
    freqs, frame_times, spec = stft(y, sr, nperseg=NPERSEG, noverlap=NPERSEG - HOP)
    df = freqs[1] - freqs[0]
    g = np.asarray(guide)[np.clip(np.searchsorted(times, frame_times), 0, len(guide) - 1)]
    mask = np.zeros(spec.shape)
    for k in np.flatnonzero(g > 0):
        for h in range(1, HARMONICS + 1):
            centre = h * g[k]
            if centre >= sr / 2:
                break
            half = max(centre * (2 ** (TOLERANCE_CENTS / 1200) - 1), 1.5 * df)
            lo, hi = int(np.ceil((centre - half) / df)), int((centre + half) / df)
            bins = np.arange(max(lo, 0), min(hi, len(freqs) - 1) + 1)
            mask[bins, k] = np.maximum(mask[bins, k], 0.5 * (1 + np.cos(np.pi * (freqs[bins] - centre) / half)))
    power, comb = np.abs(spec) ** 2, mask > 0.5
    snr_db = 10 * np.log10(((power * comb).sum(0) / np.maximum(comb.sum(0), 1) + 1e-12) / (np.median(power, axis=0) + 1e-12))
    _, solo = istft(spec * mask, sr, nperseg=NPERSEG, noverlap=NPERSEG - HOP)
    solo = np.pad(solo, (0, max(0, len(y) - len(solo))))[:len(y)].astype(np.float32)
    present = (g > 0) & (snr_db >= MIN_SNR_DB)
    return {"samples": solo, "frame_times": frame_times, "snr_db": snr_db, "present": present,
            "gaps": _runs(frame_times, (g > 0) & ~present, HOP / sr, "accompaniment_dominates")}


def _runs(times, flags, step, reason):
    """Stretches of True in `flags` (one per time in `times`) lasting at least MIN_GAP_S -> [{"start", "end", "reason"}]."""
    gaps, start = [], None
    for t, flag in zip(list(times) + [None], list(flags) + [False]):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            end = (times[len(times) - 1] if t is None else t) + step
            if end - start >= MIN_GAP_S:
                gaps.append({"start": float(start), "end": float(end), "reason": reason})
            start = None
    return gaps


def presence_in(isolated, start, end):
    """Share (0-1) of the frames between `start` and `end` seconds in which the soloist is present."""
    t = isolated["frame_times"]
    sel = (t >= start) & (t <= end)
    return float(isolated["present"][sel].mean()) if sel.any() else 0.0
