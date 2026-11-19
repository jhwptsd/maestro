# Pose data model and file format (new file).
#
# `Pose` is what Performance.pose holds: 3D joint positions in millimetres per video frame. It is produced by
# vision/pose_estimation.py (which runs VioPose and reads its output with `viopose_result_to_pose`);
# vision/hand_tracking.py and vision/motion_features.py consume it. A pose can be saved as Maestro JSON
# (`save_pose`, written next to the other outputs as pose.json) and read back with `load_pose`:
#   {"fps": 30, "joints": [...], "xyz": [[[x, y, z] per joint] per frame]}      (null where a joint was not tracked)
#
# VioPose's own output (`output/result.npy`, a pickled dict: pred (frames, 18, 3) mm, input (frames, 18, 2) pixels,
# fps) is read by `viopose_result_to_pose`. That unpickles, so it is only ever used on the file Maestro's own VioPose
# subprocess just wrote (or one you made yourself with src/scripts/convert_viopose.py); `load_pose` reads JSON only.
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from maestro.utils.errors import InputError

# VioPose's 18 output joints in order (inference.py `mediapipe_kp_remap`). Maestro adds the derived hands
# `l_hand` / `r_hand` (mean of index, pinky and thumb): where the fingering and bowing hands are.
VIOPOSE_JOINTS = ["l_ankle", "l_elbow", "l_pelvis", "l_knee", "l_shoulder", "l_wrist", "r_ankle", "r_elbow", "r_pelvis",
                  "r_knee", "r_shoulder", "r_wrist", "l_index", "l_pinky", "l_thumb", "r_index", "r_pinky", "r_thumb"]


@dataclass
class Pose:
    fps: float
    joints: List[str]                  # e.g. "l_shoulder", "r_hand" (see VIOPOSE_JOINTS)
    xyz: np.ndarray                    # (T, J, 3) millimetres; NaN where a joint was not tracked
    audio_used: bool = False           # True if VioPose's audio model (which also listens to the recording) made it

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, dtype=float)
        if self.xyz.ndim != 3 or self.xyz.shape[1] != len(self.joints) or self.xyz.shape[2] != 3:
            raise InputError(f"Pose must have shape (frames, {len(self.joints)} joints, 3), got {self.xyz.shape}.")

    def __len__(self):
        return self.xyz.shape[0]

    @property
    def times(self):
        return np.arange(len(self)) / self.fps

    def has(self, *names):
        return all(n in self.joints for n in names)

    def joint(self, name):
        """(T, 3) trajectory with gaps linearly interpolated (an all-NaN joint stays NaN)."""
        x = self.xyz[:, self.joints.index(name)].copy()
        for k in range(3):
            bad = np.isnan(x[:, k])
            if bad.any() and not bad.all():
                x[bad, k] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad, k])
        return x


def body_scale(pose):
    """Median shoulder width in mm: the length unit that makes features independent of body size and camera
    distance. 1.0 if the shoulders are not tracked."""
    if pose.has("l_shoulder", "r_shoulder"):
        width = np.linalg.norm(pose.joint("l_shoulder") - pose.joint("r_shoulder"), axis=1)
        if np.isfinite(width).any() and np.nanmedian(width) > 0:
            return float(np.nanmedian(width))
    return 1.0


def viopose_result_to_pose(path):
    """Read VioPose's `result.npy` (TRUSTED files only: it is a pickle) -> Pose (mm) with the 18 VIOPOSE_JOINTS
    plus the derived `l_hand` / `r_hand`."""
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
    return Pose(fps, VIOPOSE_JOINTS + ["l_hand", "r_hand"], np.concatenate([pred, np.stack(hands, axis=1)], axis=1))


def save_pose(pose, path):
    """Write the Maestro JSON format."""
    Path(path).write_text(json.dumps({"fps": pose.fps, "joints": pose.joints,
                                      "xyz": np.where(np.isnan(pose.xyz), None, pose.xyz).tolist(),
                                      "audio_used": pose.audio_used}))


def load_pose(path):
    """Read Maestro JSON -> Pose. Raises InputError if the file is missing or not a pose."""
    path = Path(path)
    if not path.is_file():
        raise InputError(f"File not found: {path}")
    if path.suffix.lower() != ".json":
        raise InputError(f"Unsupported pose file {path.name!r}: only Maestro pose JSON is read "
                         "(convert a VioPose result.npy with src/scripts/convert_viopose.py).")
    try:
        d = json.loads(path.read_text())
        return Pose(float(d["fps"]), d["joints"], np.array(d["xyz"], dtype=float), bool(d.get("audio_used", False)))
    except (KeyError, ValueError, TypeError, OSError) as e:        # json.JSONDecodeError is a ValueError
        raise InputError(f"Could not read pose file {path.name}: {e}")
