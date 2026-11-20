# Residual autoencoder: the shared idea behind the audio, video and score encoders (new file).
#
# The existing analysis (pitch, loudness, bow position, key, ... see assessment/ and the features of audio/, vision/,
# score/) already measures a lot. The fusion encoders are meant to capture what it does NOT: so each encoder is trained
# as a conditional variational autoencoder whose DECODER also receives the analysis features of the same segment:
#
#       x ──encoder──> z ──┐
#                          ├──decoder──> x_hat  ≈ x
#       side (analysis) ───┘
#
# Whatever the side features explain the decoder can read from them for free, so the cheapest use of the small
# bottleneck z is the remaining detail: timbre and noise texture beyond pitch and loudness (audio), posture and wrist/finger
# style beyond the bow and hand measurements (video), phrase shape and note order beyond the summary statistics (score).
# Two things keep z from copying the side information anyway: a KL term (a tight, noisy z) and a cross-covariance penalty
# between z and the side features (`cross_covariance_penalty`). `leakage_r2` measures how much of the side information
# can still be read back from z (0 = none): a training diagnostic, and the test that the idea works.
#
# Each modality file (audio_encoder.py, video_encoder.py, score_encoder.py) supplies its encoder, decoder, feature
# builders and reconstruction loss; this file supplies the model wrapper, the losses and checkpoint I/O. Checkpoints go
# in models/fusion/ (see models/README.md); fusion/embed.py loads them to embed a Performance, and src/training/
# trains them. fusion/fusion.py is meant to combine the three embeddings later.
import torch
from torch import nn


class ResidualAutoencoder(nn.Module):
    """encoder(x) -> (mu, logvar); decoder(z, side) -> reconstruction. `kind` names the modality in checkpoints."""

    def __init__(self, encoder, decoder, z_dim, kind, config):
        super().__init__()
        self.encoder, self.decoder, self.z_dim, self.kind, self.config = encoder, decoder, z_dim, kind, config

    def forward(self, x, side):
        mu, logvar = self.encoder(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decoder(z, side), mu, logvar

    @torch.no_grad()
    def encode(self, x):
        """Deterministic embedding (the posterior mean)."""
        return self.encoder(x)[0]

    def loss(self, x, side, reconstruction, beta=1e-3, leak=1.0):
        """Total loss and its parts for a batch: reconstruction + beta * KL + leak * cross-covariance(z, side).

        `reconstruction(x_hat, x)` is the modality's own loss (MSE for mel/pose, cross-entropy for notes)."""
        x_hat, mu, logvar = self(x, side)
        recon = reconstruction(x_hat, x)
        kl = kl_divergence(mu, logvar)
        xcov = cross_covariance_penalty(mu, summarize(side))
        return recon + beta * kl + leak * xcov, {"recon": float(recon.detach()), "kl": float(kl.detach()), "xcov": float(xcov.detach())}


def summarize(side):
    """Side features of a batch as (B, K): sequences (B, T, K) are averaged over time."""
    return side.mean(1) if side.dim() == 3 else side


def kl_divergence(mu, logvar):
    """KL(q(z|x) || N(0, I)) per example (summed over dimensions), averaged over the batch."""
    return (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()


def cross_covariance_penalty(z, a, eps=1e-6):
    """Mean squared correlation between every z dimension and every side-feature dimension over the batch (0 = linearly
    unrelated). Constant side features carry no information and are ignored."""
    if z.shape[0] < 3:
        return z.new_zeros(())
    zc, ac = z - z.mean(0), a - a.mean(0)
    zc = zc / (zc.std(0) + eps)
    keep = ac.std(0) > 1e-4
    if not keep.any():
        return z.new_zeros(())
    ac = ac[:, keep] / (ac[:, keep].std(0) + eps)
    return ((zc.T @ ac / (z.shape[0] - 1)) ** 2).mean()


@torch.no_grad()
def leakage_r2(z, a, ridge=1e-2, train_fraction=0.7):
    """How much of the side information can be read back from z: R^2 of a ridge regression z -> a fitted on the first
    `train_fraction` of the rows and scored on the rest, averaged over the (non-constant) side dimensions. ~0 means z holds
    nothing the analysis already knows (linearly); near 1 means z mostly repeats it."""
    z, a = z.double(), summarize(a).double()
    n = int(len(z) * train_fraction)
    if n < 4 or len(z) - n < 4:
        return float("nan")
    mz, ma = z[:n].mean(0), a[:n].mean(0)
    zt, at = z[:n] - mz, a[:n] - ma
    w = torch.linalg.solve(zt.T @ zt + ridge * len(zt) * torch.eye(z.shape[1], dtype=z.dtype), zt.T @ at)
    resid = ((z[n:] - mz) @ w + ma - a[n:]) ** 2
    var = a[n:].var(0)
    keep = var > 1e-8
    return float((1 - resid.mean(0)[keep] / var[keep]).clamp(min=0).mean()) if keep.any() else float("nan")


def save_checkpoint(model, path, stats=None):
    """Weights + what is needed to rebuild the model (`kind`, `config`) and `stats` (training diagnostics)."""
    torch.save({"kind": model.kind, "config": model.config, "state": model.state_dict(), "stats": stats or {}}, path)


def load_checkpoint(path, builders, device="cpu"):
    """Rebuild a model from a checkpoint; `builders` maps kind -> constructor taking the saved config."""
    # weights_only: a checkpoint is data (tensors and plain dicts), so nothing in it is ever executed on load
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = builders[ckpt["kind"]](**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval(), ckpt.get("stats", {})
