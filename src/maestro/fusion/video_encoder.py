# Visual encoder: a residual autoencoder over two-second windows of 3D pose (see residual_autoencoder.py).
#
# x    the pose of a WINDOW-frame stretch at FPS frames per second: every joint of vision/pose_io.VIOPOSE_JOINTS (the 18 VioPose
#      joints plus both hands) in shoulder widths, translated so the window's mean mid-shoulder point is the origin
#      (so body size, camera distance and where the player stands drop out, while sway inside the window stays)
# side the movement analysis of the same frames (vision/motion_features.py): bow position and speed, left-hand distance
#      from the shoulder and its speed, right-hand speed, both elbow angles, shoulder tilt
# z    16 numbers per window: what the full body does beyond those measurements: posture, how the wrist and fingers
#      sit, torso and head movement, the way the bow arm is carried. These are the things assessment/posture.py and
#      assessment/bowing.py only summarise with a handful of statistics.
# The decoder maps (z broadcast over time, side features per frame) back to the joint positions. Training needs only
# poses (pose.json files from earlier Maestro runs, VioPose results, or motion-capture arrays; see
# src/training/fusion_data.py): src/training/train_fusion.py video ...; fusion/embed.py embeds a Performance.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from maestro.fusion.residual_autoencoder import ResidualAutoencoder
from maestro.vision.pose_io import VIOPOSE_JOINTS, body_scale

FPS, WINDOW, STRIDE = 30, 60, 30          # frames: 2 s windows every 1 s
JOINTS = VIOPOSE_JOINTS + ["l_hand", "r_hand"]
SIDE_DIM = 8
N_IN = len(JOINTS) * 3


def _resample(x, fps):
    """Frames of an `fps` recording at FPS frames per second (nearest frame)."""
    if abs(fps - FPS) < 1e-6:
        return x
    idx = np.clip(np.round(np.arange(int(len(x) * FPS / fps)) * fps / FPS).astype(int), 0, len(x) - 1)
    return x[idx]


def pose_tensor(pose):
    """Pose -> (positions (T, N_IN) in shoulder widths, mid-shoulder (T, 3)) at FPS; joints that are missing or untracked
    are 0 (and the mid-shoulder falls back to 0 if the shoulders are not tracked)."""
    scale = body_scale(pose)
    xyz = np.stack([pose.joint(j) / scale if pose.has(j) else np.zeros((len(pose), 3)) for j in JOINTS], axis=1)
    mid = (pose.joint("l_shoulder") + pose.joint("r_shoulder")) / 2 / scale if pose.has("l_shoulder", "r_shoulder") \
        else np.zeros((len(pose), 3))
    return _resample(np.nan_to_num(xyz).reshape(len(pose), -1), pose.fps), _resample(np.nan_to_num(mid), pose.fps)


def side_features(motion, n):
    """vision/motion_features.extract_motion output -> (n, SIDE_DIM) float32 (columns scaled to roughly [-1, 1];
    0 for anything not tracked), resampled to FPS like the pose."""
    cols = np.zeros((n, SIDE_DIM))
    hands, bow = motion["hands"], motion.get("bow")
    if bow:
        cols[:, 0], cols[:, 1] = bow["s"], np.clip(bow["v"] / 5, -1, 1)
    if "left" in hands:
        cols[:, 2] = np.linalg.norm(hands["left"]["rel_shoulder"], axis=1) / 2
        cols[:, 3] = np.clip(hands["left"]["speed"] / 5, 0, 1)
    if "right" in hands:
        cols[:, 4] = np.clip(hands["right"]["speed"] / 5, 0, 1)
    for k, side in ((5, "left"), (6, "right")):
        if side in motion["elbow_angle"]:
            cols[:, k] = motion["elbow_angle"][side] / 180
    if motion["shoulder_tilt_deg"] is not None:
        cols[:, 7] = motion["shoulder_tilt_deg"] / 30
    return _resample(np.nan_to_num(cols), motion["fps"]).astype(np.float32)


def make_windows(pos, mid, side, window=WINDOW, stride=STRIDE):
    """-> (x (N, W, N_IN) centred on each window's mean mid-shoulder, side (N, W, SIDE_DIM), start frames); short
    recordings are padded with their last frame to one window."""
    n = min(len(pos), len(side))
    pos, mid, side = pos[:n], mid[:n], side[:n]
    if n < window:
        pad = [(0, window - n), (0, 0)]
        pos, mid, side, n = np.pad(pos, pad, mode="edge"), np.pad(mid, pad, mode="edge"), np.pad(side, pad, mode="edge"), window
    starts = np.arange(0, n - window + 1, stride)
    x = np.stack([(pos[s:s + window].reshape(window, -1, 3) - mid[s:s + window].mean(0)).reshape(window, -1) for s in starts])
    return x.astype(np.float32), np.stack([side[s:s + window] for s in starts]).astype(np.float32), starts


def pose_windows(pose, motion):
    """(Pose, its motion features) -> (x, side, start times in seconds)."""
    pos, mid = pose_tensor(pose)
    x, side, starts = make_windows(pos, mid, side_features(motion, len(pose)))
    return x, side, starts / FPS


class Encoder(nn.Module):
    """(B, W, N_IN) -> mean and log-variance of z, by strided 1-D convolutions over time."""

    def __init__(self, z_dim, width, window=WINDOW):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(N_IN, width, 5, padding=2), nn.GELU(),
                                  *[m for _ in range(3) for m in (nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU())])
        self.head = nn.Linear(width * -(-window // 8), 2 * z_dim)

    def forward(self, x):
        mu, logvar = self.head(self.conv(x.transpose(1, 2)).flatten(1)).chunk(2, dim=1)
        return mu, logvar.clamp(-10, 4)


class Decoder(nn.Module):
    """(z, side (B, W, SIDE_DIM)) -> pose (B, W, N_IN)."""

    def __init__(self, z_dim, width):
        super().__init__()
        self.z = nn.Linear(z_dim, 32)
        self.net = nn.Sequential(nn.Conv1d(32 + SIDE_DIM, width, 5, padding=2), nn.GELU(),
                                 nn.Conv1d(width, width, 5, padding=2), nn.GELU(), nn.Conv1d(width, N_IN, 1))

    def forward(self, z, side):
        w = side.shape[1]
        return self.net(torch.cat([self.z(z).unsqueeze(2).expand(-1, -1, w), side.transpose(1, 2)], dim=1)).transpose(1, 2)


def build(z_dim=16, width=96, window=WINDOW):
    return ResidualAutoencoder(Encoder(z_dim, width, window), Decoder(z_dim, width), z_dim, "video",
                               {"z_dim": z_dim, "width": width, "window": window})


def reconstruction_loss(x_hat, x):
    return F.mse_loss(x_hat, x)


@torch.no_grad()
def embed(perf, model, device="cpu", batch=64):
    """One z per window -> {"t": start seconds, "window_s", "z", "recon_error"} (see audio_encoder.embed)."""
    from maestro.vision.motion_features import extract_motion
    x, side, t = pose_windows(perf.pose, perf.motion_features or extract_motion(perf.pose))
    z, err = [], []
    for i in range(0, len(x), batch):
        xb, sb = torch.as_tensor(x[i:i + batch], device=device), torch.as_tensor(side[i:i + batch], device=device)
        mu = model.encode(xb)
        z.append(mu.cpu())
        err.append(((model.decoder(mu, sb) - xb) ** 2).mean((1, 2)).cpu())
    return {"t": t.tolist(), "window_s": WINDOW / FPS, "z": torch.cat(z).tolist(), "recon_error": torch.cat(err).tolist()}
