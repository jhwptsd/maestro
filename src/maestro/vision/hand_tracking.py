# Track left and right hands from processed video
#
# Reads Performance.video; outputs left (fingering) and right (bowing) hand landmarks per
# frame, merged into Performance.pose. Feeds vision/visual_pitch.py and assessment/bowing.py.
# Use ViopPose
#
# The hands come from the same VioPose pose as the body (vision/pose_estimation.py -> pose_io.Pose);
# there is no separate hand detector. This module turns the raw joints into the quantities the
# downstream modules need: hand positions in body-size units (shoulder widths, so independent of camera
# distance), positions relative to the same-side shoulder, and speeds.
import numpy as np

from maestro.vision.pose_io import body_scale


def smooth(x, fps, win_s=0.1):
    """Moving average over ~win_s seconds along axis 0 (edge-padded, same length)."""
    w = max(1, int(round(win_s * fps)) | 1)
    if w == 1 or len(x) < w:
        return x
    pad = np.pad(x, [(w // 2, w // 2)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, np.ones(w) / w, mode="valid"), 0, pad)


def track_hands(pose):
    """Pose -> {"fps", "scale", "left"/"right": {"pos", "rel_shoulder", "speed"}} (only sides present).

    pos: (T, 3) hand position / body scale; rel_shoulder: same minus the shoulder (zeros if the shoulder
    is not tracked); speed: (T,) in shoulder widths per second. The hand joint is `l_hand`/`r_hand`,
    else the wrist.
    """
    scale = body_scale(pose)
    out = {"fps": pose.fps, "scale": scale}
    for side in ("l", "r"):
        name = next((n for n in (f"{side}_hand", f"{side}_wrist") if pose.has(n)), None)
        if name is None:
            continue
        pos = smooth(pose.joint(name) / scale, pose.fps)
        shoulder = pose.joint(f"{side}_shoulder") / scale if pose.has(f"{side}_shoulder") else np.zeros_like(pos)
        out["left" if side == "l" else "right"] = {
            "pos": pos, "rel_shoulder": pos - shoulder,
            "speed": np.linalg.norm(np.gradient(pos, 1 / pose.fps, axis=0), axis=1)}
    return out
