# Denoise audio, video and score before they are used (new file).
#
# Called by preprocessing/audio.py, video.py and score.py (all on by default; pass denoise=False to skip).
# Each function removes *measurement* noise and leaves the performance alone:
#
# AUDIO  `denoise_audio`: spectral Wiener filtering (decision-directed, Ephraim & Malah 1984). The
#   background noise spectrum is measured on the quietest frames of the recording (room noise, hiss,
#   mains hum, fans), and each time-frequency cell is attenuated only as far as it looks like that
#   stationary noise. What the violin produces stays because it is much louder than the floor *and*
#   non-stationary:
#     - tonal content: harmonics are far above the floor, so their gain is ~1; the gain is also spread
#       +-2 bins around every loud cell so vibrato sidebands (the partials sweep +-1 bin) are kept;
#     - vibrato/pitch: only magnitudes are changed, the noisy phase is reused, so frequency modulation
#       is untouched (tests check depth and rate);
#     - bow noise (the scratch/hiss of the bow on the string): it is loud only while a note sounds, so it
#       is not part of the quiet-frame floor and survives, unless it is as quiet as the background itself,
#       in which case the two cannot be told apart by any method;
#     - attenuation is limited to `reduction_db`, so nothing is ever erased completely.
#   Recordings with no quiet stretch at all fall back to a conservative estimate (see `quiet_frac`).
# VIDEO  `video_filters`: a mild ffmpeg `hqdn3d` (spatio-temporal) filter, applied inside the single
#   conversion pass of utils/convert_video_type.py. Its temporal strength is kept low so moving hands and
#   bow are not smeared; sensor noise in static areas drops.
# SCORE  `denoise_score`: removes artefacts of OMR and MIDI import: duplicate notes, impossible pitches,
#   absurdly short notes, ghost notes overlapping a same-pitch note, and millisecond jitter in onsets
#   (which would split a double stop into separate events). Repeated notes, short real notes and
#   articulation are kept.
# src/scripts/tune_denoise.py chooses the audio defaults below from synthetic noisy/clean pairs.
import dataclasses

import numpy as np

from maestro.score.score_representation import Score

# ------------------------------------------------------------------ audio

AUDIO_DEFAULTS = dict(reduction_db=15.0, alpha=0.9, quiet_frac=0.15, n_fft=2048)
HIGHPASS_HZ = 40.0     # violin's lowest fundamental is 196 Hz; below ~40 Hz is rumble
GUARD_BINS = 2         # keep this many bins around every loud cell (vibrato sidebands)


def _stft(y, n_fft, hop):
    win = np.hanning(n_fft + 1)[:-1]                       # periodic Hann: perfect reconstruction at 75% overlap
    padded = np.pad(y, (n_fft, n_fft))
    n = 1 + (len(padded) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return np.fft.rfft(padded[idx] * win, axis=1), win


def _istft(spec, win, hop, length):
    n_fft = len(win)
    frames = np.fft.irfft(spec, n_fft, axis=1) * win
    out = np.zeros(hop * (len(frames) - 1) + n_fft)
    norm = np.zeros_like(out)
    for k, f in enumerate(frames):
        out[k * hop:k * hop + n_fft] += f
        norm[k * hop:k * hop + n_fft] += win ** 2
    return (out / np.maximum(norm, 1e-8))[n_fft:n_fft + length]


def _smooth(x, width):
    """Moving average along the last axis (edge-padded)."""
    k = np.ones(width) / width
    pad = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(width // 2, width // 2)], mode="edge")
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), -1, pad)


def denoise_audio(y, sr, reduction_db=AUDIO_DEFAULTS["reduction_db"], alpha=AUDIO_DEFAULTS["alpha"],
                  quiet_frac=AUDIO_DEFAULTS["quiet_frac"], n_fft=AUDIO_DEFAULTS["n_fft"], noise_profile=None):
    """Reduce stationary background noise in `y` -> (cleaned float32 audio, info dict).

    reduction_db: maximum attenuation of any cell (also the "never erase" guarantee); alpha: smoothing of
    the a-priori SNR over time (higher = smoother, fewer "musical noise" artefacts, slower reaction);
    quiet_frac: share of the quietest frames used to measure the noise (give `noise_profile`, a stretch
    of noise-only samples, to use that instead). info: {"snr_db": estimated signal-to-noise ratio of the
    input, "noise_floor_db": noise level relative to the loudest frame, "applied": bool}.
    """
    y = np.asarray(y, dtype=np.float64)
    off = {"snr_db": None, "noise_floor_db": None, "applied": False}
    if len(y) < 2 * n_fft or not np.any(y):
        return y.astype(np.float32), off
    hop = n_fft // 4
    spec, win = _stft(y, n_fft, hop)
    power = np.abs(spec) ** 2
    energy = power.sum(axis=1)

    # 1. noise spectrum: median of the quietest frames (median of exponential-distributed power = mean * ln 2)
    if noise_profile is not None and len(noise_profile) >= n_fft:
        noise = (np.abs(_stft(np.asarray(noise_profile, dtype=np.float64), n_fft, hop)[0]) ** 2).mean(axis=0)
    else:
        quiet = energy <= np.quantile(energy, quiet_frac)
        noise = np.median(power[quiet], axis=0) / np.log(2)
    noise = np.maximum(_smooth(noise[None], 5)[0], 1e-12)
    loud = energy >= np.median(energy)
    sig = max(energy[loud].mean() - noise.sum(), 1e-12)
    info = {"snr_db": float(10 * np.log10(sig / noise.sum())),
            "noise_floor_db": float(10 * np.log10(noise.sum() / energy.max())), "applied": True}

    # 2. decision-directed Wiener gain per cell, limited to reduction_db
    gmin = 10 ** (-reduction_db / 20)
    gain = np.empty_like(power)
    g_prev, snr_prev = np.ones(power.shape[1]), np.ones(power.shape[1])
    for t in range(len(power)):
        post = power[t] / noise                                           # a-posteriori SNR
        prior = alpha * g_prev ** 2 * snr_prev + (1 - alpha) * np.maximum(post - 1, 0)
        g = np.maximum(prior / (1 + prior), gmin)
        gain[t], g_prev, snr_prev = g, g, post
    # 3. never attenuate the neighbourhood of a loud cell (vibrato sidebands, partial skirts)
    guarded = gain.copy()
    for s in range(1, GUARD_BINS + 1):
        guarded[:, s:] = np.maximum(guarded[:, s:], gain[:, :-s])
        guarded[:, :-s] = np.maximum(guarded[:, :-s], gain[:, s:])
    guarded[:, np.fft.rfftfreq(n_fft, 1 / sr) < HIGHPASS_HZ] = gmin       # rumble / DC

    return _istft(spec * guarded, win, hop, len(y)).astype(np.float32), info


# ------------------------------------------------------------------ video

def video_filters(strength="mild"):
    """ffmpeg filter strings that denoise video (for utils/convert_video_type.to_common_video).

    hqdn3d takes luma_spatial:chroma_spatial:luma_temporal:chroma_temporal. The defaults were picked by
    measuring error against a clean reference on a noisy video with an object moving at 30-150 px/s:
    "mild" (2:2:8:8) cut background error by ~58% and did not smear the moving object (hqdn3d's temporal
    filter backs off where frames differ a lot, i.e. on moving hands and bow); weak temporal terms (e.g.
    3:2:6:4.5) removed less noise.
    """
    return {"mild": ["hqdn3d=2:2:8:8"], "strong": ["hqdn3d=4:3:6:4.5"]}[strength]


# ------------------------------------------------------------------ score

SCORE_DEFAULTS = dict(min_midi=55, max_midi=108, min_duration=0.02, snap=0.015)   # G3 .. C8, seconds


def denoise_score(score, min_midi=SCORE_DEFAULTS["min_midi"], max_midi=SCORE_DEFAULTS["max_midi"],
                  min_duration=SCORE_DEFAULTS["min_duration"], snap=SCORE_DEFAULTS["snap"]):
    """Clean OMR/MIDI artefacts from a Score -> (new Score, report of how many notes were touched).

    min_midi/max_midi: violin range (lower it for scordatura); notes outside are OMR misreads and dropped.
    min_duration: seconds below which a note is a glitch; snap: onsets closer than this (seconds) are the
    same moment (MIDI jitter) and are set equal, so a double stop stays one event.
    """
    report = {"out_of_range": 0, "too_short": 0, "duplicates": 0, "overlapping": 0, "snapped": 0}
    notes = []
    for n in score.notes:
        if not min_midi <= n.midi_pitch <= max_midi:
            report["out_of_range"] += 1
        elif n.offset - n.onset < min_duration:
            report["too_short"] += 1
        else:
            notes.append(dataclasses.replace(n))
    notes.sort(key=lambda n: (n.onset, n.midi_pitch))

    cluster_start = None                                   # snap near-equal onsets to the first of the cluster
    for n in notes:
        if cluster_start is not None and 0 < n.onset - cluster_start.onset <= snap:
            n.offset += cluster_start.onset - n.onset
            n.onset = cluster_start.onset
            report["snapped"] += 1
        elif cluster_start is None or n.onset - cluster_start.onset > snap:
            cluster_start = n

    kept = []
    last_by_pitch = {}
    for n in sorted(notes, key=lambda n: (n.onset, n.midi_pitch)):
        prev = last_by_pitch.get(n.midi_pitch)
        if prev is not None and n.onset < prev.offset - 1e-6:   # same pitch sounding twice at once
            report["duplicates" if abs(n.onset - prev.onset) < 1e-6 else "overlapping"] += 1
            if n.offset > prev.offset:                           # keep the longer extent
                prev.offset, prev.duration = n.offset, max(prev.duration, n.duration)
            continue
        kept.append(n)
        last_by_pitch[n.midi_pitch] = n
    return dataclasses.replace(score, notes=kept), report            # keeps tempo, key and time signature
