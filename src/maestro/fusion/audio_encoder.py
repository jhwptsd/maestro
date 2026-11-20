# Auditory encoder: a residual autoencoder over one-second log-mel patches (see residual_autoencoder.py).
#
# x    log-mel spectrogram of a ~1 s window (WINDOW frames x N_MELS), hop and frame length identical to
#      audio/features.py so every mel frame lines up with a row of Performance.audio_features["frames"]
# side the analysis of the same frames, from those rows: voiced flag, pitch (audio/pitch.py), pitch confidence, RMS
#      level, spectral centroid and onset strength (audio/features.py, audio/onset_detection.py)
# z    16 numbers per window: what the spectrum holds beyond pitch, loudness, brightness and attacks, i.e. the harmonic
#      balance (timbre), bow noise and texture, resonance and room. These are the aspects of sound that the rule-based
#      assessment/tone.py only touches through a few descriptors.
# The decoder maps (z broadcast over time, side features per frame) to the mel frames with a stack of 1-D convolutions.
# Training needs only audio (the side features are computed from it): src/training/train_fusion.py audio ...
# fusion/embed.py turns a Performance into one z per window (+ how well that window was reconstructed).
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from maestro.fusion.residual_autoencoder import ResidualAutoencoder

N_FFT, HOP, N_MELS = 1024, 256, 64        # N_FFT and HOP equal audio/features.py's frame_length and hop_length
WINDOW, STRIDE = 86, 43                   # frames: ~1 s at 22.05 kHz, hop 0.5 s
SIDE_DIM = 6
FMIN, FMAX = 40.0, 8000.0                 # violin fundamentals and the harmonics that shape its timbre


def mel_filterbank(sr, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """(n_mels, n_fft // 2 + 1) triangular HTK-mel filters."""
    hz_to_mel = lambda f: 2595.0 * np.log10(1 + np.asarray(f) / 700.0)
    mel_to_hz = lambda m: 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(min(fmax, sr / 2)), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, mid, hi = edges[i:i + 3]
        fb[i] = np.maximum(0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))
    return fb


def log_mel(y, sr):
    """Mono samples -> (frames, N_MELS) log-mel scaled to about [-1, 1] (dB clipped to [-100, 0], then (dB + 50) / 50).
    Frames are not centred, so frame i covers samples [i * HOP, i * HOP + N_FFT) like audio/features.py."""
    y = torch.as_tensor(np.asarray(y, dtype=np.float32))
    if len(y) < N_FFT:
        y = F.pad(y, (0, N_FFT - len(y)))
    spec = torch.stft(y, N_FFT, HOP, window=torch.hann_window(N_FFT), center=False, return_complex=True).abs() ** 2
    mel = torch.as_tensor(mel_filterbank(sr), dtype=torch.float32) @ spec            # (mels, frames)
    db = 10 * torch.log10(mel + 1e-10).clamp(-100, 0)
    return ((db + 50) / 50).T.numpy()


def side_features(frames):
    """Rows of audio_features["frames"] -> (T, SIDE_DIM) float32, each column roughly in [0, 1]:
    voiced, pitch (semitones above G3 / 50), pitch confidence, RMS level, log centroid, onset strength."""
    col = lambda k: np.array([f[k] for f in frames], dtype=np.float64)
    f0, rms, cen, onset = col("estimated_pitch"), col("rms_energy"), col("spectral_centroid"), col("onset_probability")
    voiced = f0 > 0
    pitch = np.where(voiced, (69 + 12 * np.log2(np.where(voiced, f0, 440.0) / 440.0) - 55) / 50, 0.0)
    level = np.clip((20 * np.log10(rms + 1e-6) + 60) / 60, 0, 1)
    scale = np.percentile(onset, 95) if len(onset) and np.percentile(onset, 95) > 0 else 1.0
    return np.stack([voiced, pitch, col("pitch_confidence"), level, np.log2(cen + 1) / 14, np.clip(onset / scale, 0, 1)],
                    axis=1).astype(np.float32)


def make_windows(mel, side, window=WINDOW, stride=STRIDE):
    """(mel, side) for one recording -> (x (N, W, N_MELS), side (N, W, SIDE_DIM), start frames). Short recordings are
    padded with their last frame to one window."""
    n = min(len(mel), len(side))
    mel, side = mel[:n], side[:n]
    if n < window:
        pad = [(0, window - n), (0, 0)]
        mel, side, n = np.pad(mel, pad, mode="edge"), np.pad(side, pad, mode="edge"), window
    starts = np.arange(0, n - window + 1, stride)
    return (np.stack([mel[s:s + window] for s in starts]).astype(np.float32),
            np.stack([side[s:s + window] for s in starts]).astype(np.float32), starts)


def performance_windows(perf):
    """Windows of a Performance (perf.audio + perf.audio_features) -> (x, side, start times in seconds)."""
    x, side, starts = make_windows(log_mel(perf.audio["samples"], perf.audio["sr"]),
                                   side_features(perf.audio_features["frames"]))
    return x, side, starts * HOP / perf.audio["sr"]


class Encoder(nn.Module):
    """(B, W, N_MELS) -> mean and log-variance of z, by strided 1-D convolutions over time."""

    def __init__(self, z_dim, width, window=WINDOW):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(N_MELS, width, 5, padding=2), nn.GELU(),
                                  *[m for _ in range(3) for m in (nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU())])
        self.head = nn.Linear(width * -(-window // 8), 2 * z_dim)        # three stride-2 convolutions: ceil(window / 8) steps

    def forward(self, x):
        mu, logvar = self.head(self.conv(x.transpose(1, 2)).flatten(1)).chunk(2, dim=1)
        return mu, logvar.clamp(-10, 4)


class Decoder(nn.Module):
    """(z, side (B, W, SIDE_DIM)) -> mel (B, W, N_MELS): z is repeated along time next to the per-frame side features."""

    def __init__(self, z_dim, width):
        super().__init__()
        self.z = nn.Linear(z_dim, 32)
        self.net = nn.Sequential(nn.Conv1d(32 + SIDE_DIM, width, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(width, width, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(width, width, 5, padding=2), nn.GELU(), nn.Conv1d(width, N_MELS, 1))

    def forward(self, z, side):
        w = side.shape[1]
        return self.net(torch.cat([self.z(z).unsqueeze(2).expand(-1, -1, w), side.transpose(1, 2)], dim=1)).transpose(1, 2)


def build(z_dim=16, width=96, window=WINDOW):
    return ResidualAutoencoder(Encoder(z_dim, width, window), Decoder(z_dim, width), z_dim, "audio",
                               {"z_dim": z_dim, "width": width, "window": window})


def reconstruction_loss(x_hat, x):
    return F.mse_loss(x_hat, x)


@torch.no_grad()
def embed(perf, model, device="cpu", batch=64):
    """One z per window -> {"t": start seconds, "window_s", "z": [[...]], "recon_error": per window mean squared error
    of the reconstruction (high = unlike the training audio even given its pitch and loudness)}."""
    x, side, t = performance_windows(perf)
    z, err = [], []
    for i in range(0, len(x), batch):
        xb, sb = torch.as_tensor(x[i:i + batch], device=device), torch.as_tensor(side[i:i + batch], device=device)
        mu = model.encode(xb)
        z.append(mu.cpu())
        err.append(((model.decoder(mu, sb) - xb) ** 2).mean((1, 2)).cpu())
    return {"t": t.tolist(), "window_s": WINDOW * HOP / perf.audio["sr"], "z": torch.cat(z).tolist(),
            "recon_error": torch.cat(err).tolist()}
