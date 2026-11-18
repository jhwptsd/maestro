# Pose data model and file formats (new file).
#
# `Pose` is what Performance.pose holds: 3D joint positions per video frame. It is produced by
# vision/pose_estimation.py (VioPose or a user-supplied file) and by the motion-capture loaders used in
# vision/mocap_compare.py; vision/hand_tracking.py and vision/motion_features.py consume it.
#
# VioPose's own output (`output/result.npy`, a pickled dict: pred (frames, 18, 3) in mm, input (frames, 18, 2)
# pixels, fps) is read by `viopose_result_to_pose` below. That reader unpickles, so it is for files YOU produced;
# uploads and `load_pose` never unpickle (use src/scripts/convert_viopose.py on your own result.npy first).
#
# Joint names are canonicalised (CANONICAL below) so any source's naming works: VioPose/MediaPipe style
# ("left_shoulder", "nose"), marker sets ("LSHO", "RightWrist") are mapped through aliases. Only the
# joints that exist are required; analyses use what is available (e.g. the bow axis falls back to the
# right-hand trajectory when there are no bow markers).
#
# Supported files: .json (Maestro format, below), .npy (+ joints/fps via arguments or a sidecar .json),
# .csv ("<joint>_x,<joint>_y,<joint>_z" columns), .trc (OpenSim/Vicon marker text). C3D is not supported
# (export to TRC/CSV first).
#   JSON: {"fps": 30, "units": "mm", "joints": [...], "xyz": [[[x, y, z] per joint] per frame]}
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from maestro.utils.errors import InputError

CANONICAL = ["head", "l_shoulder", "l_elbow", "l_wrist", "l_hand", "l_index", "l_pinky", "l_thumb", "l_pelvis",
             "l_knee", "l_ankle", "r_shoulder", "r_elbow", "r_wrist", "r_hand", "r_index", "r_pinky", "r_thumb",
             "r_pelvis", "r_knee", "r_ankle", "scroll", "bow_frog", "bow_tip"]
# VioPose's 18 output joints in order (inference.py `mediapipe_kp_remap`); "hand" is derived from index/pinky/thumb
VIOPOSE_JOINTS = ["l_ankle", "l_elbow", "l_pelvis", "l_knee", "l_shoulder", "l_wrist", "r_ankle", "r_elbow", "r_pelvis",
                  "r_knee", "r_shoulder", "r_wrist", "l_index", "l_pinky", "l_thumb", "r_index", "r_pinky", "r_thumb"]
_SIDE = {"l": "l", "left": "l", "r": "r", "right": "r"}
_PART = {"shoulder": "shoulder", "sho": "shoulder", "shoulders": "shoulder", "elbow": "elbow", "elb": "elbow",
         "wrist": "wrist", "wri": "wrist", "hand": "hand", "finger": "hand", "fin": "hand", "index": "index",
         "indextip": "index", "indexfingertip": "index", "pinky": "pinky", "thumb": "thumb", "pelvis": "pelvis",
         "hip": "pelvis", "knee": "knee", "ankle": "ankle", "toe": "ankle"}
_OTHER = {"head": "head", "nose": "head", "scroll": "scroll", "bowfrog": "bow_frog", "frog": "bow_frog",
          "bowtip": "bow_tip", "tip": "bow_tip"}
UNITS_TO_MM = {"mm": 1.0, "m": 1000.0, "cm": 10.0}


def canonical_name(name):
    """'LeftShoulder' / 'left_shoulder' / 'L_SHO' / 'r wrist' -> 'l_shoulder' / 'r_wrist'; unknown -> as is."""
    key = re.sub(r"[^a-z]", "", str(name).lower())
    if key in _OTHER:
        return _OTHER[key]
    for side in sorted(_SIDE, key=len, reverse=True):
        if key.startswith(side) and key[len(side):] in _PART:
            return f"{_SIDE[side]}_{_PART[key[len(side):]]}"
    return str(name)


@dataclass
class Pose:
    fps: float
    joints: List[str]                  # canonical names
    xyz: np.ndarray                    # (T, J, 3); NaN where a joint was not tracked
    units: str = "mm"                  # "mm" | "m" | "cm" | "normalized" (unit-free, e.g. VioPose-normalised)
    conf: Optional[np.ndarray] = None  # (T, J) tracker confidence in [0, 1], if known

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, dtype=float)
        if self.xyz.ndim != 3 or self.xyz.shape[1] != len(self.joints) or self.xyz.shape[2] not in (2, 3):
            raise InputError(f"Pose must have shape (frames, {len(self.joints)} joints, 2 or 3), got {self.xyz.shape}.")
        if self.xyz.shape[2] == 2:     # 2D keypoints: add a zero depth axis
            self.xyz = np.concatenate([self.xyz, np.zeros(self.xyz.shape[:2] + (1,))], axis=2)
        self.joints = [canonical_name(j) for j in self.joints]

    def __len__(self):
        return self.xyz.shape[0]

    @property
    def times(self):
        return np.arange(len(self)) / self.fps

    def has(self, *names):
        return all(n in self.joints for n in names)

    def joint(self, name):
        """(T, 3) trajectory with gaps linearly interpolated (all-NaN joints stay NaN)."""
        x = self.xyz[:, self.joints.index(name)].copy()
        for k in range(3):
            bad = np.isnan(x[:, k])
            if bad.any() and not bad.all():
                x[bad, k] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad, k])
        return x

    def to_mm(self):
        """Copy in millimetres (unchanged if the units are 'normalized')."""
        f = UNITS_TO_MM.get(self.units)
        return self if f is None else Pose(self.fps, self.joints, self.xyz * f, "mm", self.conf)


def body_scale(pose):
    """Median shoulder width in the pose's own units: the length unit used to make features
    independent of camera distance and of whether the pose is metric. 1.0 if shoulders are missing."""
    if pose.has("l_shoulder", "r_shoulder"):
        w = np.linalg.norm(pose.joint("l_shoulder") - pose.joint("r_shoulder"), axis=1)
        if np.isfinite(w).any() and np.nanmedian(w) > 0:
            return float(np.nanmedian(w))
    return 1.0


# ---------------------------------------------------------------- VioPose

def viopose_result_to_pose(path):
    """Read VioPose's `result.npy` (TRUSTED files only: it is a pickle) -> Pose in millimetres at the video's
    frame rate, with the 18 joints named as in VIOPOSE_JOINTS plus derived `l_hand` / `r_hand` (the mean of
    index, pinky and thumb), which is where the fingering and bowing hands are."""
    try:
        result = np.load(path, allow_pickle=True).item()
        pred = np.asarray(result["pred"], dtype=float)
        fps = float(result["fps"])
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as e:
        raise InputError(f"{Path(path).name} is not a VioPose result.npy: {e}")
    pred = pred.reshape((-1,) + pred.shape[-2:])        # (frames, joints, 3), whatever batch dims were kept
    if pred.shape[1:] != (len(VIOPOSE_JOINTS), 3) or not fps > 0:
        raise InputError(f"Unexpected VioPose output shape {pred.shape} (expected frames x {len(VIOPOSE_JOINTS)} x 3).")
    hands = [np.mean([pred[:, VIOPOSE_JOINTS.index(f"{side}_{part}")] for part in ("index", "pinky", "thumb")], axis=0)
             for side in ("l", "r")]
    return Pose(fps, VIOPOSE_JOINTS + ["l_hand", "r_hand"], np.concatenate([pred, np.stack(hands, axis=1)], axis=1), "mm")


# ---------------------------------------------------------------- reading / writing

def save_pose(pose, path):
    """Write the Maestro JSON format."""
    Path(path).write_text(json.dumps({"fps": pose.fps, "units": pose.units, "joints": pose.joints,
                                      "xyz": np.where(np.isnan(pose.xyz), None, pose.xyz).tolist()}))


def load_pose(path, fps=None, joints=None, units=None, joint_map=None):
    """Read a pose / motion-capture file -> Pose. Raises InputError if it cannot be understood.

    fps/joints/units override or supply what the file lacks; joint_map {file_name: canonical_name}
    renames markers that the built-in aliases do not know.
    """
    path = Path(path)
    if not path.is_file():
        raise InputError(f"File not found: {path}")
    ext = path.suffix.lower()
    try:
        if ext == ".json":
            d = json.loads(path.read_text())
            pose = Pose(fps or d["fps"], joints or d["joints"], np.array(d["xyz"], dtype=float),
                        units or d.get("units", "mm"))
        elif ext == ".npy":
            side = {}
            for cand in (path.with_suffix(".json"), Path(str(path) + ".json")):
                if cand.is_file():
                    side = json.loads(cand.read_text())
            names, rate = joints or side.get("joints"), fps or side.get("fps")
            if not names or not rate:
                raise InputError(f"{path.name} needs joint names and fps (pass them or add a sidecar .json).")
            pose = Pose(rate, names, np.load(path, allow_pickle=False), units or side.get("units", "mm"))
        elif ext == ".csv":
            pose = _read_csv(path, fps, units)
        elif ext == ".trc":
            pose = _read_trc(path, fps, units)
        else:
            raise InputError(f"Unsupported pose file type {ext!r} (use .json, .npy, .csv or .trc).")
    except InputError:
        raise
    except (KeyError, ValueError, TypeError, json.JSONDecodeError, OSError) as e:
        hint = (" (a VioPose result.npy is a pickle: convert it with src/scripts/convert_viopose.py first)"
                if ext == ".npy" and "pickle" in str(e).lower() else "")
        raise InputError(f"Could not read pose file {path.name}: {e}{hint}")
    if joint_map:
        pose.joints = [joint_map.get(j, j) for j in pose.joints]
    return pose


def _read_csv(path, fps, units):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], np.array([[float(v) if v.strip() else np.nan for v in r] for r in rows[1:]])
    cols = {}
    for k, h in enumerate(head):
        m = re.fullmatch(r"(.+?)[_ .]?([xyz])", h.strip(), re.I)
        if m:
            cols.setdefault(m.group(1), {})[m.group(2).lower()] = k
    names = [n for n, c in cols.items() if set(c) >= {"x", "y"}]
    xyz = np.stack([np.stack([body[:, cols[n][a]] if a in cols[n] else np.zeros(len(body)) for a in "xyz"], axis=1)
                    for n in names], axis=1)
    if not fps:
        tcol = next((k for k, h in enumerate(head) if h.strip().lower() in ("time", "t", "timestamp")), None)
        if tcol is None or len(body) < 2:
            raise InputError(f"{path.name}: give fps or include a time column.")
        fps = 1.0 / float(np.median(np.diff(body[:, tcol])))
    return Pose(fps, names, xyz, units or "mm")


def _read_trc(path, fps, units):
    lines = path.read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("Frame#"))
    meta = dict(zip(lines[1].split("\t"), lines[2].split("\t")))
    names = [t for t in lines[i].split("\t")[2:] if t.strip()]
    data = [l.split("\t") for l in lines[i + 2:] if l.strip()]
    xyz = np.array([[float(v) if v.strip() else np.nan for v in r[2:2 + 3 * len(names)]] for r in data])
    return Pose(fps or float(meta["DataRate"]), names, xyz.reshape(len(data), len(names), 3),
                units or meta.get("Units", "mm").strip().lower())
