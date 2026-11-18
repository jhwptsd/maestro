# Compare visual pose estimates with motion-capture annotations (new file).
#
# `compare_to_mocap(estimated, mocap)` is the evaluation behind assessment["mocap"] and the GUI's accuracy
# chart. Both inputs are pose_io.Pose (mocap loaded with pose_io.load_pose from .trc/.csv/.json/.npy).
# It handles the usual mismatches: different frame rates and an unknown time offset (found from the
# right-hand speed), different units/scales/axes (optional similarity alignment), and different marker
# sets (only joints present in both are compared).
#
# Metrics (millimetres when both are metric): MPJPE root-relative (positions after removing the
# shoulder-midpoint, i.e. body posture independent of where the person stands), PA-MPJPE (after a best
# similarity transform, the usual measure when axes/scales differ), velocity error MPJVE (mm/s) and
# bow-motion correlation (does the estimated right hand move along the bow like the real one).
import numpy as np



def _resample(pose, fps, n):
    """Linear resample of xyz to `fps`, first `n` frames."""
    t_new = np.arange(n) / fps
    t_old = pose.times
    return np.stack([np.stack([np.interp(t_new, t_old, pose.joint(j)[:, k]) for k in range(3)], axis=1)
                     for j in pose.joints], axis=1)


def _root(x, names):
    """Subtract the shoulder midpoint (or the mean of all joints) per frame."""
    if "l_shoulder" in names and "r_shoulder" in names:
        return x - (x[:, names.index("l_shoulder")] + x[:, names.index("r_shoulder")])[:, None] / 2
    return x - x.mean(1, keepdims=True)


def _umeyama(src, dst):
    """Similarity transform (scale s, rotation R, translation t) minimising |s R src + t - dst|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(d0.T @ s0 / len(src))
    sign = np.eye(3)
    sign[2, 2] = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    r = u @ sign @ vt
    scale = np.trace(np.diag(sig) @ sign) / (s0 ** 2).sum(1).mean()
    return scale, r, mu_d - scale * r @ mu_s


def estimate_lag(a, b, fps, max_lag):
    """Seconds by which signal `a` lags `b` (positive = a is later), searched within +-max_lag."""
    a, b = a - a.mean(), b - b.mean()
    m = int(max_lag * fps)
    lags = list(range(-m, m + 1))
    score = [np.dot(a[max(0, k):len(a) + min(0, k)], b[max(0, -k):len(b) + min(0, -k)]) /
             max(1, len(a) - abs(k)) for k in lags]
    return lags[int(np.argmax(score))] / fps


def compare_to_mocap(estimated, mocap, max_lag=1.0, align=True):
    """Pose vs mocap -> {"joints", "lag_s", "metric", "n_frames", "pa_mpjpe", "per_joint", "mpjpe_mm",
    "mpjve_mm_s", "bow_motion_corr"}. mpjpe_mm/mpjve_mm_s are None unless both poses are metric (not
    'normalized'); pa_mpjpe/per_joint are in mm if `metric`, else unit-free.

    align: apply the best similarity transform before PA-MPJPE (always computed); the plain MPJPE only
    removes the shoulder midpoint.
    """
    names = [j for j in estimated.joints if j in mocap.joints]
    if len(names) < 3:
        raise ValueError(f"only {len(names)} joints in common between the estimate and the mocap")
    fps = estimated.fps
    metric = estimated.units != "normalized" and mocap.units != "normalized"
    est = estimated.to_mm() if metric else estimated
    ref = mocap.to_mm() if metric else mocap
    n = min(len(est), int(len(ref) / ref.fps * fps))
    e = _resample(est, fps, n)[:, [est.joints.index(j) for j in names]]
    m = _resample(ref, fps, n)[:, [ref.joints.index(j) for j in names]]

    # time offset from the right-hand speed (or the first joint's), then trim both to the overlap
    key = names.index("r_hand") if "r_hand" in names else names.index("r_wrist") if "r_wrist" in names else 0
    speed = lambda x: np.linalg.norm(np.gradient(x[:, key], axis=0), axis=1)
    lag = estimate_lag(speed(e), speed(m), fps, max_lag)
    k = int(round(lag * fps))
    e, m = (e[k:], m[:len(m) - k]) if k >= 0 else (e[:len(e) + k], m[-k:])
    n = min(len(e), len(m))
    e, m = e[:n], m[:n]

    er, mr = _root(e, names), _root(m, names)
    s, rot, t = _umeyama(er.reshape(-1, 3), mr.reshape(-1, 3))
    aligned = (s * er.reshape(-1, 3) @ rot.T + t).reshape(er.shape)
    pa = np.linalg.norm(aligned - mr, axis=2)
    out = {"joints": names, "lag_s": lag, "metric": metric, "n_frames": n,
           "pa_mpjpe": float(pa.mean()),   # mm if `metric`, else in the poses' own unit-free scale
           "per_joint": {j: float(v) for j, v in zip(names, pa.mean(0))}}
    out["mpjpe_mm"] = float(np.linalg.norm(er - mr, axis=2).mean()) if metric else None
    out["mpjve_mm_s"] = float(np.linalg.norm(np.gradient(aligned, 1 / fps, axis=0) - np.gradient(mr, 1 / fps, axis=0),
                                             axis=2).mean()) if metric else None
    # does the estimated right hand move like the real one along its main (bowing) axis?
    axis = np.linalg.svd(mr[:, key] - mr[:, key].mean(0), full_matrices=False)[2][0]
    a, b = (aligned[:, key] - aligned[:, key].mean(0)) @ axis, (mr[:, key] - mr[:, key].mean(0)) @ axis
    out["bow_motion_corr"] = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float("nan")
    return out
