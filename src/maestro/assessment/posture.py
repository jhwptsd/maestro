# Evaluate posture based on processed data
#
# Uses pose from vision/pose_estimation.py. Writes to Performance.assessment.
#
# Reads vision/motion_features.py output (angles/tilt) and the Pose itself; result is
# assessment["posture"]. First-pass consistency metrics, all unit-free or in degrees: how level the
# shoulders are, how much the torso/head sways while playing (shoulder widths), and how steady the
# elbows are. Judging "good" posture needs reference data; for now these report stability, and
# low sway/tilt variation is generally better.
import numpy as np


def _sway(x):
    """Spread of a (T, 3) trajectory: root of the summed per-axis variances (same unit as x)."""
    return float(np.sqrt(np.nansum(np.nanvar(x, axis=0))))


def assess_posture(motion, pose):
    """-> {"available", "shoulder_tilt_mean_deg", "shoulder_tilt_std_deg", "shoulder_sway", "head_sway",
    "left_elbow_angle_mean/std_deg", "right_elbow_angle_mean/std_deg", "right_elbow_sway"}; None for
    anything whose joints are not tracked."""
    scale, out = motion["scale"], {"available": True}
    tilt = motion.get("shoulder_tilt_deg")
    out["shoulder_tilt_mean_deg"] = float(np.nanmean(tilt)) if tilt is not None else None
    out["shoulder_tilt_std_deg"] = float(np.nanstd(tilt)) if tilt is not None else None
    out["shoulder_sway"] = (_sway((pose.joint("l_shoulder") + pose.joint("r_shoulder")) / 2 / scale)
                            if pose.has("l_shoulder", "r_shoulder") else None)
    out["head_sway"] = _sway(pose.joint("head") / scale) if pose.has("head") else None
    for side in ("left", "right"):
        ang = motion["elbow_angle"].get(side)
        out[f"{side}_elbow_angle_mean_deg"] = float(np.nanmean(ang)) if ang is not None else None
        out[f"{side}_elbow_angle_std_deg"] = float(np.nanstd(ang)) if ang is not None else None
    out["right_elbow_sway"] = _sway(pose.joint("r_elbow") / scale) if pose.has("r_elbow") else None
    if all(v is None for k, v in out.items() if k != "available"):
        return {"available": False, "reason": "no shoulder, elbow or head joints tracked"}
    return out
