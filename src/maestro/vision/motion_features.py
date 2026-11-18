# Extract motion features from processed video such as joint velocities, accelerations, and angular velocities.
#
# Computes velocities/accelerations from pose_estimation.py and hand_tracking.py output.
# Consumed by assessment/bowing.py, assessment/posture.py, and fusion/video_encoder.py.
# Uses VioPose
#
# Output of `extract_motion` is stored in Performance.motion_features. All lengths are in body-size
# units (shoulder widths, see pose_io.body_scale) so metrics are comparable across videos.
#
# Bow movement: the bow's direction of travel is the "bow axis". With bow markers (bow_frog, bow_tip) it
# is the frog-to-tip direction; otherwise it is the main direction of the right hand's motion (PCA), as
# bowing is mostly a back-and-forth along one line. The right hand's coordinate along that axis is the
# bow-travel signal; each direction reversal of it is a bow change, so the signal splits into strokes.
# With markers, sign +1 = down-bow (hand moves away from the string, frog toward the player's right).
import numpy as np

from maestro.vision.hand_tracking import smooth, track_hands

MIN_STROKE_S = 0.08      # shorter "strokes" are tracking jitter, merged into their neighbours
PAUSE_FRACTION = 0.15    # speeds below this fraction of the typical peak speed count as standing still


def joint_angle(a, b, c):
    """Angle at b (degrees) formed by points a-b-c, per frame."""
    u, v = a - b, c - b
    cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def bow_axis(pose, right_hand):
    """Unit vector along the bow: from markers if present, else the first principal axis of the hand path."""
    if pose.has("bow_frog", "bow_tip"):
        d = np.nanmean(pose.joint("bow_frog") - pose.joint("bow_tip"), axis=0)   # frog - tip: + = down-bow
        if np.linalg.norm(d) > 0:
            return d / np.linalg.norm(d), "markers"
    centred = right_hand - right_hand.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    return axis / np.linalg.norm(axis), "pca"


def split_strokes(s, fps):
    """Bow-travel signal s (T,) -> list of strokes {start, end (s), dir (+1/-1), length, mean_speed, peak_speed}.

    Direction is the sign of the smoothed velocity; speeds under PAUSE_FRACTION of the typical peak count
    as a pause and keep the previous direction, boundaries sit mid-pause (the turnaround).
    """
    v = np.gradient(s, 1 / fps)
    peak = np.percentile(np.abs(v), 90)
    if peak <= 0:
        return []
    thr = PAUSE_FRACTION * peak
    d = np.where(np.abs(v) > thr, np.sign(v), 0).astype(int)
    moving = np.flatnonzero(d)
    if moving.size == 0:
        return []
    bounds = [moving[0]]                                   # frames where a stroke starts
    for a, b in zip(moving[:-1], moving[1:]):
        if d[a] != d[b]:
            bounds.append((a + b) // 2)                    # turnaround: middle of the gap
    bounds.append(moving[-1] + 1)
    strokes = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = v[a:b]
        strokes.append({"start": a / fps, "end": b / fps, "dir": int(np.sign(seg.sum())) or 1,
                        "length": float(abs(s[min(b, len(s) - 1)] - s[a])),
                        "mean_speed": float(np.abs(seg).mean()), "peak_speed": float(np.abs(seg).max())})
    # merge strokes that are too short to be real (tracking jitter) into the previous one
    merged = []
    for st in strokes:
        if merged and (st["end"] - st["start"] < MIN_STROKE_S or merged[-1]["dir"] == st["dir"]):
            merged[-1]["end"] = st["end"]
            merged[-1]["length"] += st["length"] if merged[-1]["dir"] == st["dir"] else -st["length"]
            merged[-1]["length"] = abs(merged[-1]["length"])
        else:
            merged.append(dict(st))
    return merged


def extract_motion(pose):
    """Pose -> dict of per-frame motion signals + bow strokes (see the module comment).

    Keys: fps, scale, times, hands (hand_tracking output), bow {axis, axis_source, s, v, strokes},
    elbow_angle {left,right} (degrees), elbow_angular_velocity (deg/s), shoulder_tilt_deg,
    and per-joint speed/acceleration for the hands. Missing joints give missing keys/None.
    """
    fps = pose.fps
    hands = track_hands(pose)
    m = {"fps": fps, "scale": hands["scale"], "times": pose.times, "hands": hands,
         "bow": None, "elbow_angle": {}, "elbow_angular_velocity": {}, "shoulder_tilt_deg": None}
    if "right" in hands:
        pos = hands["right"]["pos"]
        axis, source = bow_axis(pose, pos * hands["scale"])
        s = smooth(((pos - pos.mean(0)) @ axis)[:, None], fps, 0.06)[:, 0]
        m["bow"] = {"axis": axis, "axis_source": source, "s": s, "v": np.gradient(s, 1 / fps),
                    "strokes": split_strokes(s, fps)}
        m["right_hand_acceleration"] = np.gradient(np.gradient(pos, 1 / fps, axis=0), 1 / fps, axis=0)
    if "left" in hands:
        m["left_hand_acceleration"] = np.gradient(np.gradient(hands["left"]["pos"], 1 / fps, axis=0), 1 / fps, axis=0)
    for side, name in (("l", "left"), ("r", "right")):
        if pose.has(f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"):
            ang = joint_angle(pose.joint(f"{side}_shoulder"), pose.joint(f"{side}_elbow"), pose.joint(f"{side}_wrist"))
            m["elbow_angle"][name] = ang
            m["elbow_angular_velocity"][name] = np.gradient(smooth(ang[:, None], fps)[:, 0], 1 / fps)
    if pose.has("l_shoulder", "r_shoulder"):
        d = pose.joint("r_shoulder") - pose.joint("l_shoulder")
        m["shoulder_tilt_deg"] = np.degrees(np.arctan2(d[:, 1], np.hypot(d[:, 0], d[:, 2])))
    return m
