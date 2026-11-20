# Score encoder: a residual autoencoder over windows of written notes (see residual_autoencoder.py).
#
# x    WINDOW consecutive notes of the score (score/score_representation.py order), each as six categorical tokens: pitch,
#      duration class, position in the bar, articulation, dynamic mark, hairpin (padding for windows at the end)
# side the symbolic analysis of the same window (score/symbolic_features.py and the score's key and time signature),
#      summarised as numbers: pitch-class profile, mean pitch and range, mean duration, note density, share of steps /
#      skips / leaps, share of articulated notes, mean dynamic level, key signature, beats per bar
# z    16 numbers per window: what the statistics cannot say: the ORDER of the notes, the melodic contour and motifs,
#      the rhythmic pattern and the way markings are distributed over it, i.e. the shape of the phrase and where its
#      difficulty lies, which the per-note features of the rule-based assessment do not capture.
# The decoder is a GRU that starts from (z, side) and predicts every token of every note position. Training needs only
# scores (MusicXML / MIDI / PDF through preprocessing/score.py): src/training/train_fusion.py score ... Pretrained
# on a large score collection it gives each stretch of the piece a description that fusion/ can relate to how it was
# played. fusion/embed.py gives one z per window, timed by the performance via the alignment.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from maestro.fusion.residual_autoencoder import ResidualAutoencoder

WINDOW, STRIDE = 16, 8                          # notes
LOW_MIDI = 43                                   # G2: below the violin's range, so out-of-range notes clip to the edges
N_PITCH, N_DUR, N_POS, N_ART, N_DYN, N_HAIR = 58, 13, 24, 5, 9, 3
SIZES = (N_PITCH, N_DUR, N_POS, N_ART, N_DYN, N_HAIR)     # tokens per field; the padding id is the size itself
SIDE_DIM = 23
ARTICULATIONS = {"staccato": 1, "accent": 2, "tenuto": 3}   # anything else written -> 4
DYNAMICS = {"ppp": 1, "pp": 2, "p": 3, "mp": 4, "mf": 5, "f": 6, "ff": 7, "fff": 8}   # other marks (sf, fp ...) -> 0
HAIRPINS = {"crescendo": 1, "diminuendo": 2}


def tokenize(notes, time_signature=None):
    """Score notes (dicts: midi_pitch, duration, beat, articulation, dynamic, hairpin) -> (len, 6) int array."""
    beats = (time_signature or (4, 4))[0]
    out = np.zeros((len(notes), 6), dtype=np.int64)
    for i, n in enumerate(notes):
        dur = np.clip(round(2 * np.log2(max(n["duration"], 1 / 16))), -6, 6) + 6          # eighth-note steps on a log scale
        pos = int(round(N_POS * (n["beat"] - 1) / beats)) % N_POS            # 24ths of the bar: fits halves, thirds, quarters
        art = n.get("articulation")
        out[i] = (np.clip(n["midi_pitch"] - LOW_MIDI, 0, N_PITCH - 1), dur, pos,
                  0 if not art else ARTICULATIONS.get(art, 4), DYNAMICS.get(n.get("dynamic"), 0),
                  HAIRPINS.get(n.get("hairpin"), 0))
    return out


def window_side(notes, key_fifths=None, time_signature=None):
    """Window of note dicts -> (SIDE_DIM,) float32 summary of its symbolic analysis (each value roughly in [0, 1])."""
    midi = np.array([n["midi_pitch"] for n in notes], dtype=float)
    dur = np.array([n["duration"] for n in notes], dtype=float)
    onsets = np.array([n["onset"] for n in notes], dtype=float)
    span = max(max(n.get("offset", n["onset"]) for n in notes) - onsets.min(), 1e-3)             # seconds the window covers
    steps = np.abs(np.diff(midi)) if len(midi) > 1 else np.zeros(1)
    levels = [DYNAMICS.get(n.get("dynamic"), 0) for n in notes]
    return np.concatenate([
        np.bincount((midi % 12).astype(int), minlength=12) / len(midi),
        [(midi.mean() - 55) / 40, np.ptp(midi) / 36, np.log2(dur.mean() + 1e-3) / 4 + 0.5, min(len(notes) / span / 8, 1),
         (steps <= 2).mean(), ((steps > 2) & (steps <= 4)).mean(), (steps > 4).mean(),
         np.mean([bool(n.get("articulation")) for n in notes]), np.mean(levels) / 8,
         (key_fifths or 0) / 7, (time_signature or (4, 4))[0] / 6]]).astype(np.float32)


def make_windows(notes, key_fifths=None, time_signature=None, window=WINDOW, stride=STRIDE):
    """Note dicts -> (tokens (N, W, 6) with the padding id after the notes of a short last window, side (N, SIDE_DIM),
    first note index of every window). A score shorter than one window gives one padded window."""
    tokens = tokenize(notes, time_signature)
    starts = list(range(0, max(len(notes) - window, 0) + 1, stride))
    if starts[-1] + window < len(notes):                      # make sure the end of the score is covered
        starts.append(len(notes) - window)
    x, side = [], []
    for s in starts:
        chunk = tokens[s:s + window]
        x.append(np.concatenate([chunk, np.tile(SIZES, (window - len(chunk), 1))]))
        side.append(window_side(notes[s:s + window], key_fifths, time_signature))
    return np.stack(x), np.stack(side), np.array(starts)


class Encoder(nn.Module):
    """(B, W, 6) tokens -> mean and log-variance of z: field embeddings, a bidirectional GRU, masked mean over the notes."""

    def __init__(self, z_dim, hidden, dims=(32, 8, 8, 4, 4, 2)):
        super().__init__()
        self.emb = nn.ModuleList(nn.Embedding(n + 1, d, padding_idx=n) for n, d in zip(SIZES, dims))
        self.gru = nn.GRU(sum(dims), hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 2 * z_dim)

    def forward(self, x):
        out, _ = self.gru(torch.cat([e(x[..., k]) for k, e in enumerate(self.emb)], dim=-1))
        mask = (x[..., 0] != N_PITCH).unsqueeze(-1).to(out.dtype)
        mu, logvar = self.head((out * mask).sum(1) / mask.sum(1).clamp(min=1)).chunk(2, dim=1)
        return mu, logvar.clamp(-10, 4)


class Decoder(nn.Module):
    """(z, side (B, SIDE_DIM)) -> one logits tensor per field, each (B, W, field size)."""

    def __init__(self, z_dim, hidden, window=WINDOW):
        super().__init__()
        self.start = nn.Linear(z_dim + SIDE_DIM, hidden)
        self.pos = nn.Parameter(torch.randn(window, hidden) * 0.1)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.heads = nn.ModuleList(nn.Linear(hidden, n) for n in SIZES)

    def forward(self, z, side):
        h0 = torch.tanh(self.start(torch.cat([z, side], dim=1))).unsqueeze(0)
        out, _ = self.gru(self.pos.unsqueeze(0).expand(len(z), -1, -1), h0.contiguous())
        return tuple(head(out) for head in self.heads)


def build(z_dim=16, hidden=96, window=WINDOW):
    return ResidualAutoencoder(Encoder(z_dim, hidden), Decoder(z_dim, hidden, window), z_dim, "score",
                               {"z_dim": z_dim, "hidden": hidden, "window": window})


def reconstruction_loss(logits, x, reduction="mean"):
    """Cross-entropy of every field at every real (non-padding) note, summed over fields, averaged over notes."""
    real = x[..., 0] != N_PITCH
    total = sum(F.cross_entropy(l.transpose(1, 2), x[..., k].clamp(max=n - 1), reduction="none") for k, (l, n) in enumerate(zip(logits, SIZES)))
    return (total * real).sum() / real.sum().clamp(min=1)


@torch.no_grad()
def embed(perf, model, device="cpu", batch=64):
    """One z per window of notes, timed in the performance: {"t": performance time of the first matched note of each window,
    "notes": [first, last + 1] score indices, "z", "recon_error"}. Windows with no matched note are left out."""
    notes = perf.score_features
    x, side, starts = make_windows(notes, perf.score.key_fifths, perf.score.time_signature)
    onset = {a["score_idx"]: perf.audio_features["notes"][a["perf_idx"]]["onset"] for a in perf.alignment if a["perf_idx"] is not None}
    times = [next((onset[i] for i in range(s, min(s + WINDOW, len(notes))) if i in onset), None) for s in starts]
    keep = [k for k, t in enumerate(times) if t is not None]
    z, err = [], []
    for i in range(0, len(keep), batch):
        idx = keep[i:i + batch]
        xb, sb = torch.as_tensor(x[idx], device=device), torch.as_tensor(side[idx], device=device)
        mu = model.encode(xb)
        z.append(mu.cpu())
        err.append(torch.stack([reconstruction_loss(model.decoder(mu[j:j + 1], sb[j:j + 1]), xb[j:j + 1]) for j in range(len(idx))]).cpu())
    return {"t": [times[k] for k in keep], "notes": [[int(starts[k]), int(min(starts[k] + WINDOW, len(notes)))] for k in keep],
            "z": torch.cat(z).tolist() if z else [], "recon_error": torch.cat(err).tolist() if err else []}
