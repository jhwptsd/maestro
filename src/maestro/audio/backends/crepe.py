"""PyTorch CREPE backend using torchcrepe's pretrained tiny/full models.

Speed and accuracy notes (measured on a 4-core CPU, 30 s of audio):
 - The network is ~90% of the run time. The full model costs ~17 ms per frame on CPU (0.7x real time); the tiny
   model ~0.9 ms (about 16x faster) and was as accurate as full on clean tones (3.4 cents mean per-note
   error for both). ``model="auto"`` (default) therefore picks "tiny" on CPU and "full" on a GPU, where full
   is cheap. Pass model="full" to force the larger network.
 - torchcrepe's stock decoder adds random +-20 cent noise ("dither") to every frame and snaps pitch to a
   20-cent grid, so results differed from run to run and per-note error was ~7 cents. ``_decode`` keeps its
   Viterbi smoothing but refines each frame to sub-bin precision with a local weighted average and no
   randomness: ~3.4 cents per-note error, <1 cent frame jitter, deterministic.
 - torchcrepe's Viterbi also feeds on softmax(network output). The outputs are already probabilities in
   [0, 1], so the softmax leaves almost no contrast between bins and the smoothing then outweighs the
   evidence: after a wide leap (e.g. 22 semitones) the path stays parked and the whole next note is lost
   (4 of 12 notes on a random-leap test melody, 35 of 40 short notes on a wide-leap one). Normalising the
   outputs across bins instead keeps every note (0 of 12, 1 of 40 lost) at the same 3-4 cent accuracy.
 - torchcrepe resamples with resampy (1.2 s per minute of audio); scipy's polyphase resampler does the same
   job in ~25 ms, so the audio is brought to 16 kHz here first.
"""
import math

import numpy as np

from maestro.audio.backends.base import PitchTrack, frame_times, frame_rms

_CENTS_OFFSET = 1997.3794084376191   # cents of pitch bin 0 (torchcrepe.convert.bins_to_cents)
_REFINE_BINS = 4                     # bins either side of the Viterbi path used for the weighted average
_transition = None


def _transition_matrix():
    """Viterbi transition matrix between the 360 pitch bins (same as torchcrepe's), built once."""
    global _transition
    if _transition is None:
        bins = np.arange(360)
        t = np.maximum(12 - np.abs(bins[:, None] - bins[None, :]), 0).astype(float)
        _transition = t / t.sum(axis=1, keepdims=True)
    return _transition


def _decode(probabilities):
    """torchcrepe decoder: Viterbi path refined to sub-bin precision, without dither.

    probabilities: (batch, 360, frames) network output (bins outside fmin..fmax are -inf).
    Returns (bins (batch, frames) int64, frequency (batch, frames) Hz).
    """
    import librosa
    import torch
    import torchcrepe

    with torch.no_grad():
        valid = torch.isfinite(probabilities)                                           # False outside fmin..fmax
        weights = torch.where(valid, probabilities, torch.zeros_like(probabilities)).clamp(min=0)
        # emission = the network output normalised over bins (+ a floor so silent frames are flat, not log(0))
        emission = torch.where(valid, weights + 1e-6, torch.zeros_like(weights))
        emission = (emission / emission.sum(dim=1, keepdim=True).clamp(min=1e-12)).cpu().numpy()
        path = np.stack([librosa.sequence.viterbi(seq, _transition_matrix()).astype(np.int64) for seq in emission])
        bins = torch.from_numpy(path).to(probabilities.device)                          # (B, T)
        weights = weights.transpose(1, 2)                                               # (B, T, 360)
        offsets = torch.arange(-_REFINE_BINS, _REFINE_BINS + 1, device=bins.device)
        pos = bins.unsqueeze(-1) + offsets                                              # (B, T, 9) neighbouring bins
        valid = (pos >= 0) & (pos < 360)
        w = torch.gather(weights, 2, pos.clamp(0, 359)) * valid
        refined = (w * pos).sum(-1) / w.sum(-1).clamp(min=1e-9)
        refined = torch.where(w.sum(-1) > 0, refined, bins.float())
        cents = torchcrepe.CENTS_PER_BIN * refined + _CENTS_OFFSET
        return bins, torchcrepe.convert.cents_to_frequency(cents)


def _resample(y, sr, target):
    """Polyphase resample to `target` Hz (exact rational ratio)."""
    from scipy.signal import resample_poly      # a hard dependency of torchcrepe

    g = math.gcd(int(target), int(sr))
    return resample_poly(y, int(target) // g, int(sr) // g).astype(np.float32)


def _run_network(y, sr, hop_length, fmin, fmax, model, batch_size, device):
    """Resample to 16 kHz and run torchcrepe -> (f0 Hz, periodicity) per CREPE frame, plus the CREPE hop in samples."""
    import torch
    import torchcrepe

    if device is None or device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if model == "auto":                         # full CREPE is only cheap on an accelerator
        model = "full" if str(device).startswith(("cuda", "mps")) else "tiny"
    rate = torchcrepe.SAMPLE_RATE
    crepe_hop = int(hop_length * rate / sr)     # torchcrepe truncates the hop to whole 16 kHz samples
    audio = torch.from_numpy(np.clip(_resample(y, sr, rate) if sr != rate else y, -1.0, 1.0)).unsqueeze(0)
    with torch.inference_mode():
        pitch, periodicity = torchcrepe.predict(
            audio, rate, hop_length=crepe_hop, fmin=fmin, fmax=fmax, model=model, decoder=_decode,
            batch_size=batch_size, device=device, return_periodicity=True)
    return pitch[0].cpu().numpy().astype(float), periodicity[0].cpu().numpy().astype(float), crepe_hop


def estimate(y, sr, frame_length=1024, hop_length=256, fmin=None, fmax=None,
             model="auto", batch_size=1024, device=None, rms_floor_db=60.0, **_):
    try:
        import torchcrepe
    except ImportError as exc:
        raise ImportError(
            "The default CREPE backend requires PyTorch and torchcrepe. "
            "Install it with: pip install -e '.[crepe]' (or install the matching PyTorch build first)."
        ) from exc

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if model not in ("auto", "tiny", "full"):
        raise ValueError("CREPE model must be 'auto', 'tiny' or 'full'.")
    fmin = 175.0 if fmin is None else float(fmin)
    fmax = min(2000.0, float(torchcrepe.MAX_FMAX)) if fmax is None else float(fmax)
    if not 0 < fmin < fmax <= float(torchcrepe.MAX_FMAX):
        raise ValueError(
            f"CREPE requires 0 < fmin < fmax <= {torchcrepe.MAX_FMAX:g} Hz. "
            "Use pYIN or YIN for violin fundamentals above the CREPE model's upper limit."
        )
    if not y.size:
        empty = np.zeros(0)
        return PitchTrack(empty, empty, empty, "crepe")

    f0, confidence, crepe_hop = _run_network(y, sr, hop_length, fmin, fmax, model, batch_size, device)

    # CREPE's frame spacing (and count) differs slightly from Maestro's `hop_length / sr` timeline, so map
    # each Maestro frame to the nearest CREPE frame in time: every backend then shares one timeline.
    times = frame_times(len(y), sr, hop_length)
    nearest = np.clip(np.rint(times * torchcrepe.SAMPLE_RATE / crepe_hop).astype(int), 0, len(f0) - 1)
    f0, confidence = f0[nearest], confidence[nearest]

    # CREPE reports a pitch even in the noise floor; mark quiet frames unvoiced instead of letting them become notes.
    rms = frame_rms(y, frame_length, hop_length)[:len(times)]
    if rms.max() > 0:
        silent = rms <= rms.max() * 10.0 ** (-float(rms_floor_db) / 20.0)
        confidence[:len(rms)][silent] = 0.0
        f0[:len(rms)][silent] = 0.0
    return PitchTrack(times, f0, confidence, "crepe")
