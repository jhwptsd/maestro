# Estimate pitch from video
#
# Uses left-hand tracking (hand_tracking.py) to estimate pitch visually. Can be cross-checked
# against audio/pitch.py in assessment/intonation.py or fused in fusion/.
# Visual pitch estimation from Koepke, Wiles & Zisserman (2019)
#
# Koepke et al. train a CNN to predict pitch from video frames; that needs a large labelled dataset, so this is a
# geometric baseline on the 3D pose instead, and it uses no audio. The left hand slides along the fingerboard, and
# a finger pressed a distance x from the nut shortens a string of length L, raising its pitch by
#       semitones = 12 * log2(L / (L - x))                       (the equal-tempered string-length law)
# The pose gives the hand's distance from the left shoulder; moving up the fingerboard brings the hand toward the
# body. So, per recording:
#   1. d(t)   = distance from the left shoulder to the left hand, in mm (the pose is metric);
#   2. ref    = the 95th percentile of d: the hand's furthest reach, taken to be first position (finger at
#               FIRST_POSITION_MM from the nut);
#   3. x(t)   = FIRST_POSITION_MM + (ref - d(t));  semitones_above_open(t) = 12 log2(L / (L - x(t))).
# Which string is playing cannot be seen, so the result is "semitones above the open string" plus one candidate
# MIDI pitch per open string (G3 D4 A4 E5); the ambiguity is a fifth per string.
#
# Honest limits: the anatomy is assumed (string length, first-position reach, shoulder width), pose error and body
# lean add noise, and the first-position reference makes the absolute scale approximate (a few semitones). It is
# most reliable for *changes* in position (how many semitones the hand moved), less so for absolute pitch.
# `calibrate_visual_pitch` fits a curve from known (distance, pitch) pairs, e.g. pitches measured elsewhere, to get
# absolute MIDI pitch per frame; nothing here depends on how those pairs were obtained.
import numpy as np

OPEN_STRINGS = (55, 62, 69, 76)       # G3 D4 A4 E5 (MIDI)
STRING_LENGTH_MM = 325.0              # vibrating string length of a full-size violin (nut to bridge)
FIRST_POSITION_MM = 55.0              # fingertip distance from the nut in first position (fingers 1-4 span ~35-82 mm)
MIN_FRAMES = 10


def semitones_above_open(x_mm, length=STRING_LENGTH_MM):
    """Pitch rise (semitones) of a string pressed `x_mm` from the nut: 12 * log2(L / (L - x))."""
    return 12.0 * np.log2(length / np.maximum(length - np.asarray(x_mm, dtype=float), 5.0))


def distance_from_nut_mm(semitones, length=STRING_LENGTH_MM):
    """Inverse of `semitones_above_open`."""
    return length * (1.0 - 2.0 ** (-np.asarray(semitones, dtype=float) / 12.0))


def left_hand_distance_mm(pose, hands):
    """Per-frame distance from the left shoulder to the left hand in millimetres, or None if not tracked."""
    if "left" not in hands or not pose.has("l_shoulder"):
        return None
    return np.linalg.norm(hands["left"]["rel_shoulder"], axis=1) * hands["scale"]      # rel_shoulder is in shoulder widths


def calibrate_visual_pitch(distances_mm, midi_pitches, degree=2):
    """Fit hand distance -> pitch from known pairs -> {"coeffs", "n", "r_spearman", "mae_semitones", "within_1_semitone"}.

    The errors are leave-one-out: each pair is predicted by a curve fitted without it. Needs at least 6 pairs.
    """
    x, y = np.asarray(distances_mm, dtype=float), np.asarray(midi_pitches, dtype=float)
    if len(x) < 6:
        raise ValueError(f"need at least 6 (distance, pitch) pairs to calibrate, got {len(x)}")
    degree = min(degree, 2 if len(x) >= 12 else 1)
    loo = np.array([np.polyval(np.polyfit(np.delete(x, i), np.delete(y, i), degree), x[i]) for i in range(len(x))])
    ranks = lambda v: np.argsort(np.argsort(v)).astype(float)
    spearman = float(np.corrcoef(ranks(x), ranks(y))[0, 1]) if x.std() and y.std() else float("nan")
    err = np.abs(loo - y)
    return {"coeffs": np.polyfit(x, y, degree).tolist(), "n": len(x), "r_spearman": spearman,
            "mae_semitones": float(err.mean()), "within_1_semitone": float((err <= 1).mean())}


def estimate_visual_pitch(pose, hands, calibration=None):
    """Left-hand position -> pitch per frame, from video alone (see the module comment).

    Returns {"available": False, "reason"} or {"available": True, "method", "times", "distance_mm" (shoulder to
    hand), "reference_mm", "semitones_above_open" (T,), "candidates" (T, 4) MIDI per open string, "range_semitones"
    (5th-95th percentile of the hand's travel), "mean_semitones"} and, when `calibration` (from
    calibrate_visual_pitch) is given, "midi" (T,) absolute pitch from the fitted curve.
    """
    distance = left_hand_distance_mm(pose, hands)
    if distance is None:
        return {"available": False, "reason": "left hand and left shoulder not tracked"}
    if len(distance) < MIN_FRAMES:
        return {"available": False, "reason": f"only {len(distance)} frames"}
    reference = float(np.percentile(distance, 95))                 # furthest reach = first position
    semitones = semitones_above_open(FIRST_POSITION_MM + (reference - distance))
    out = {"available": True, "method": "geometry", "times": pose.times, "distance_mm": distance,
           "reference_mm": reference, "semitones_above_open": semitones,
           "candidates": semitones[:, None] + np.array(OPEN_STRINGS)[None, :],
           "range_semitones": float(np.percentile(semitones, 95) - np.percentile(semitones, 5)),
           "mean_semitones": float(semitones.mean())}
    if calibration is not None:
        out["method"] = "calibrated"
        out["midi"] = np.polyval(calibration["coeffs"], distance)
        out["calibration"] = calibration
    return out
