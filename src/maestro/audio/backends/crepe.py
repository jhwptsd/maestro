"""PyTorch CREPE backend using torchcrepe's pretrained tiny/full models."""
import numpy as np

from maestro.audio.backends.base import PitchTrack, frame_times, frame_rms


def estimate(y, sr, frame_length=1024, hop_length=256, fmin=None, fmax=None,
             model="full", batch_size=1024, device=None, rms_floor_db=60.0, **_):
    try:
        import torch
        import torchcrepe
    except ImportError as exc:
        raise ImportError(
            "The default CREPE backend requires PyTorch and torchcrepe. "
            "Install it with: pip install -e '.[crepe]' (or install the matching PyTorch build first)."
        ) from exc

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    if model not in ("tiny", "full"):
        raise ValueError("CREPE model must be 'tiny' or 'full'.")
    fmin = 175.0 if fmin is None else float(fmin)
    fmax = min(2000.0, float(torchcrepe.MAX_FMAX)) if fmax is None else float(fmax)
    if not 0 < fmin < fmax <= float(torchcrepe.MAX_FMAX):
        raise ValueError(
            f"CREPE requires 0 < fmin < fmax <= {torchcrepe.MAX_FMAX:g} Hz. "
            "Use pYIN or YIN for violin fundamentals above the CREPE model's upper limit."
        )

    if not y.size:
        empty = np.zeros(0, dtype=float)
        return PitchTrack(empty, empty, empty, "crepe")

    if device is None or device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    audio = torch.from_numpy(np.clip(y, -1.0, 1.0)).unsqueeze(0)
    with torch.inference_mode():
        pitch, periodicity = torchcrepe.predict(
            audio,
            sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            model=model,
            batch_size=batch_size,
            device=device,
            return_periodicity=True,
        )

    f0 = pitch[0].detach().cpu().numpy().astype(float)
    confidence = periodicity[0].detach().cpu().numpy().astype(float)
    n = min(len(f0), len(confidence))
    f0, confidence = f0[:n], np.clip(confidence[:n], 0.0, 1.0)

    # torchcrepe resamples to 16 kHz and truncates the hop to whole samples, so its frame
    # spacing (and count) differs slightly from Maestro's `hop_length / sr` timeline. Map each
    # Maestro frame to the nearest CREPE frame in time so every backend shares one timeline.
    times = frame_times(len(y), sr, hop_length)
    crepe_hop = int(hop_length * torchcrepe.SAMPLE_RATE / sr) if sr != torchcrepe.SAMPLE_RATE else hop_length
    idx = np.clip(np.rint(times * torchcrepe.SAMPLE_RATE / crepe_hop).astype(int), 0, n - 1)
    f0, confidence = f0[idx], confidence[idx]
    n = len(times)

    # CREPE can produce a pitch in quiet frames; explicitly mark the recording's
    # noise floor unvoiced instead of interpreting those values as notes.
    rms = frame_rms(y, frame_length, hop_length)
    n_active = min(len(rms), n)
    if n_active and rms.max() > 0:
        floor = rms.max() * (10.0 ** (-float(rms_floor_db) / 20.0))
        silent = rms[:n_active] <= floor
        confidence[:n_active] = np.where(silent, 0.0, confidence[:n_active])
        f0[:n_active] = np.where(silent, 0.0, f0[:n_active])

    return PitchTrack(times, f0, confidence, "crepe")
