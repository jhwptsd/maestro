# Evaluate posture based on processed data
#
# Uses pose from vision/pose_estimation.py. Writes to Performance.assessment.
#
# Reads vision/motion_features.py output (angles/tilt) and the Pose itself; result is
# assessment["posture"]. First-pass consistency metrics, all unit-free or in degrees: how level the
# shoulders are, how much the torso/head sways while playing (shoulder widths), and how steady the
# elbows are. Judging "good" posture needs reference data; for now these report stability, and
# low sway/tilt variation is generally better.
#
# Events (common format of assessment/events.py), one per WINDOW_S window of the performance that goes wrong:
#   shoulders_uneven   shoulder line tilted more than TILT_DEG from level (violin or shoulder rest at the wrong height,
#                      leaning) -- persistent tilt is reported as a setup problem, transient tilt as leaning
#   head_movement      head travelling faster than HEAD_SPEED shoulder widths per second; if the window holds a leap the
#                      cause is "compensating for a shift", else body-driven phrasing or tension
#   posture_drift      the tilt or the left elbow angle moves steadily over the performance (fatigue / slumping)
# `ctx` (events.Context) places each window at a measure and tells whether it contains leaps.
import numpy as np

from maestro.assessment.events import Context, cause, make_event, module_result, ramp

WINDOW_S = 1.0
TILT_DEG = 10.0          # shoulder tilt from level that is reported
HEAD_SPEED = 0.5         # shoulder widths per second
DRIFT_DEG = 12.0         # change of tilt / elbow angle between the first and last third that counts as drift
LEAP_SEMITONES = 5


def _sway(x):
    """Spread of a (T, 3) trajectory: root of the summed per-axis variances (same unit as x)."""
    return float(np.sqrt(np.nansum(np.nanvar(x, axis=0))))


def _events(motion, pose, ctx):
    t = np.asarray(motion["times"])
    if len(t) < 2:
        return []
    edges = np.arange(t[0], t[-1], WINDOW_S)
    tilt = motion.get("shoulder_tilt_deg")
    head = None
    if pose.has("head"):
        h = pose.joint("head") / motion["scale"]
        head = np.r_[0.0, np.linalg.norm(np.diff(h, axis=0), axis=1)] * motion["fps"]    # shoulder widths per second
    median_tilt = float(np.nanmedian(tilt)) if tilt is not None else 0.0
    windows = [(a, a + WINDOW_S, (t >= a) & (t < a + WINDOW_S)) for a in edges]
    flagged = [abs(np.nanmedian(tilt[m])) > TILT_DEG for _, _, m in windows if m.sum() > 2] if tilt is not None else []
    persistent = bool(flagged) and np.mean(flagged) > 0.6
    events = []
    for a, b, m in windows:
        if m.sum() < 3:
            continue
        if tilt is not None and abs(np.nanmedian(tilt[m])) > TILT_DEG:
            d = float(np.nanmedian(tilt[m]))
            c = (cause("setup_height", 0.6, tilt_deg=d, share=float(np.mean(flagged)))
                 if persistent else
                 cause("leaning",
                       0.5, tilt_deg=d, median_tilt_deg=median_tilt))
            events.append(make_event("posture", "shoulders_uneven", a, b, ramp(abs(d), TILT_DEG, 25) * 0.7 + 0.2, [c], ctx, value=d, expected=0.0, unit="deg"))
        if head is not None and np.nanmean(head[m]) > HEAD_SPEED:
            v = float(np.nanmean(head[m]))
            leap = any(abs(ctx.score[i]["midi_pitch"] - ctx.score[i - 1]["midi_pitch"]) >= LEAP_SEMITONES
                       for i, p in ctx.matches if i > 0 and a <= ctx.notes[p]["onset"] < b)
            c = (cause("shift_compensation",
                       0.55, head_speed=v) if leap else
                 cause("body_driven_playing", 0.35, head_speed=v))
            events.append(make_event("posture", "head_movement", a, b, ramp(v, HEAD_SPEED, 1.5) * 0.7 + 0.2, [c], ctx,
                                     value=v, expected=HEAD_SPEED, unit="shoulder widths/s"))
    series = {"shoulder tilt": tilt, "left elbow angle": motion["elbow_angle"].get("left")}
    for name, x in series.items():
        if x is None or len(x) < 30:
            continue
        third = len(x) // 3
        change = float(np.nanmedian(x[-third:]) - np.nanmedian(x[:third]))
        if abs(change) > DRIFT_DEG:
            events.append(make_event("posture", "posture_drift", float(t[0]), float(t[-1]), ramp(abs(change), DRIFT_DEG, 30) * 0.6 + 0.2,
                                     [cause("fatigue", 0.45, change_deg=change, signal=name)],
                                     ctx, value=change, expected=0.0, unit="deg"))
    return events


def assess_posture(motion, pose, ctx=None):
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
        return {"available": False, "reason": "no shoulder, elbow or head joints tracked", "module": "posture",
                "score": None, "summary": {}, "events": []}
    events = _events(motion, pose, ctx or Context.from_parts([], [], []))
    n_windows = max(1, int((motion["times"][-1] - motion["times"][0]) / WINDOW_S))
    return {**module_result("posture", events, n_windows, {k: v for k, v in out.items() if k != "available"}), **out}
