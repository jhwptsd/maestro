"""Test fixtures for the visual stack: a synthetic violinist and a fake VioPose checkout.

`synth_pose` makes 3D joint trajectories (a Pose) for a performance of the Minuet notes:
 - the right hand moves along a bow axis in strokes whose turnarounds fall on note starts (slurred notes share a
   stroke);
 - the left hand obeys the violin's string-length law: for each note we pick the string that puts the finger at least
   in 1st position, work out its distance from the nut, and place the hand that far down the arm (the hand rests at
   1st position before the first note and after the last, as a player's does);
 - shoulders sway a little and Gaussian noise models tracker error.
`mocap_version` records the same performance as a motion-capture system would (other frame rate, units, axes, time
offset). `write_fake_viopose` creates a stand-in VioPose checkout whose inference.py writes ./output/result.npy in
VioPose's real format, so the runner can be tested without TensorFlow."""
import json
from pathlib import Path

import numpy as np

from maestro.vision.hand_tracking import smooth
from maestro.vision.pose_io import VIOPOSE_JOINTS, Pose
from maestro.vision.visual_pitch import (FIRST_POSITION_MM, OPEN_STRINGS, distance_from_nut_mm, semitones_above_open)
from minuet import BARS

BOW_AXIS = np.array([1.0, 0.15, 0.0]) / np.hypot(1.0, 0.15)    # direction of down-bow hand travel (frog - tip)
JOINTS = ["head", "l_shoulder", "l_elbow", "l_wrist", "l_hand", "r_shoulder", "r_elbow", "r_wrist", "r_hand"]
REACH_MM = 420.0                                               # shoulder-to-left-hand distance in 1st position
LEFT_DIRECTION = np.array([-0.3, 0.2, 0.93]) / np.linalg.norm([-0.3, 0.2, 0.93])


def minuet_notes(bpm=100):
    """[(onset, offset, midi, slurred_to_next)] in seconds for the Minuet fixture (no timing jitter)."""
    out, t = [], 0.4
    for bar in BARS:
        for midi, beats, slur in bar:
            out.append((t, t + beats * 60 / bpm, midi, slur))
            t += beats * 60 / bpm
    return out


def semitones_on_string(midi):
    """(open string pitch, semitones above it) on the highest string that keeps the finger at or beyond 1st position."""
    first = float(semitones_above_open(FIRST_POSITION_MM))
    return next((o, midi - o) for o in sorted(OPEN_STRINGS, reverse=True) if midi - o >= first)


def fingerboard_truth(notes, t):
    """True (semitones above the open string, midi) at times t: held during notes, gliding in between, resting at
    1st position before and after the performance."""
    rest = float(semitones_above_open(FIRST_POSITION_MM))
    nodes_t, nodes_n, nodes_m = [0.0, notes[0][0] - 0.05], [rest, rest], [np.nan, np.nan]   # hold 1st position until just before the first note
    for onset, offset, midi, _ in notes:
        _, n = semitones_on_string(midi)
        nodes_t += [onset, offset - 0.01]
        nodes_n += [n, n]
        nodes_m += [midi, midi]
    nodes_t += [notes[-1][1] + 0.1, notes[-1][1] + 5]
    nodes_n += [rest, rest]
    nodes_m += [np.nan, np.nan]
    return np.interp(t, nodes_t, nodes_n), np.interp(t, nodes_t, np.nan_to_num(nodes_m, nan=0.0))


def steady_mask(notes, t, margin=0.1):
    """True where the finger is held on a note (away from the transitions between notes)."""
    mask = np.zeros(len(t), dtype=bool)
    for onset, offset, _, _ in notes:
        mask |= (t >= onset + margin) & (t <= offset - margin)
    return mask


def synth_pose(notes, fps=30, noise_mm=2.0, with_bow=False, sway_mm=4.0, stroke_jitter=0.0, time_shift=0.0,
               left_hand=True, seed=0):
    """Pose (mm) of a violinist playing `notes` (see minuet_notes)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int((notes[-1][1] + 1.5) * fps)) / fps - time_shift                # a long rest at the end, as in a real clip
    # bow: a new stroke starts at every note that does not continue a slur
    starts = [n[0] for k, n in enumerate(notes) if k == 0 or not notes[k - 1][3]] + [notes[-1][1]]
    nodes_t, nodes_s, s = [0.0, starts[0]], [0.0, 0.0], 0.0
    for k, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        length = min((b - a) * 280.0, 420.0) * (1 + stroke_jitter * rng.uniform(-1, 1))   # ~280 mm/s
        s += length if k % 2 == 0 else -length
        nodes_t.append(b), nodes_s.append(s)
    bow_s = smooth(np.interp(t, nodes_t, nodes_s)[:, None], fps, 0.08)[:, 0]
    sway = sway_mm * np.sin(2 * np.pi * 0.4 * t)

    def col(*v):
        return np.tile(np.array(v, dtype=float), (len(t), 1))
    r_shoulder, l_shoulder = col(150, 0, 0), col(-150, 0, 0)
    r_shoulder[:, 0] += sway
    l_shoulder[:, 0] += sway
    r_hand = col(300, -250, 200) + bow_s[:, None] * BOW_AXIS
    r_elbow = col(250, -200, 100) + 0.35 * bow_s[:, None] * BOW_AXIS
    n_true, _ = fingerboard_truth(notes, t)
    reach = REACH_MM - (distance_from_nut_mm(n_true) - FIRST_POSITION_MM)     # hand comes toward the body up the neck
    l_hand = l_shoulder + reach[:, None] * LEFT_DIRECTION
    joints = {"head": col(0, 250, 50) + sway[:, None] * [0.5, 0, 0], "l_shoulder": l_shoulder, "r_shoulder": r_shoulder,
              "l_elbow": col(-250, -250, 150), "l_wrist": l_hand - [0, 30, 0], "l_hand": l_hand,
              "r_elbow": r_elbow, "r_wrist": r_hand - [0, 30, 0], "r_hand": r_hand}
    names = [n for n in JOINTS if left_hand or n not in ("l_wrist", "l_hand")]
    if with_bow:   # frog at the right hand, tip 650 mm along -axis (so frog - tip = +axis = down-bow)
        joints["bow_frog"], joints["bow_tip"] = r_hand, r_hand - 650 * BOW_AXIS
        names += ["bow_frog", "bow_tip"]
    return Pose(fps, names, np.stack([joints[n] for n in names], axis=1) + rng.normal(0, noise_mm, (len(t), len(names), 3)), "mm")


def mocap_version(notes, time_shift=0.2, fps=60, noise_mm=0.5, seed=1):
    """The same performance as a mocap system records it: 60 fps, metres, rotated 90 degrees about the
    vertical axis, starting `time_shift` s later than the video estimate."""
    p = synth_pose(notes, fps=fps, noise_mm=noise_mm, time_shift=time_shift, seed=seed)
    rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    return Pose(fps, p.joints, (p.xyz @ rot.T) / 1000.0, "m")


def viopose_result(pose):
    """VioPose's result dict for `pose`: pred (frames, 18, 3) mm in VIOPOSE_JOINTS order, input (frames, 18, 2) px, fps.
    The three finger joints sit around the hand so that their mean is exactly the pose's hand."""
    fingers = {"index": [10, 0, 0], "pinky": [-5, 5, 0], "thumb": [-5, -5, 0]}       # offsets average to zero
    columns = []
    for name in VIOPOSE_JOINTS:
        side, part = name[0], name[2:]
        if part in fingers:
            columns.append(pose.joint(f"{side}_hand") + fingers[part])
        elif pose.has(name):
            columns.append(pose.joint(name))
        else:                                                                         # legs: not needed, hold still
            columns.append(np.tile([150.0 if side == "r" else -150.0, -900.0, 0.0], (len(pose), 1)))
    pred = np.stack(columns, axis=1).astype(np.float32)
    return {"pred": pred, "input": pred[:, :, :2] * 0.5 + 300, "fps": pose.fps}


def save_viopose_result(pose, path):
    """Write `viopose_result(pose)` the way VioPose does: np.save of a dict (a pickle)."""
    np.save(path, viopose_result(pose))


FAKE_INFERENCE = """\
import argparse, json, os, pickle, sys
ap = argparse.ArgumentParser()
ap.add_argument("--folder", required=True); ap.add_argument("--video_path", default="best")
ap.add_argument("--audio_path", default=None); ap.add_argument("--vis", default="True")
a = ap.parse_args()
folder = a.folder + ("" if a.audio_path else "_wo_audio")        # as the real inference.py does
json.dump({"args": sys.argv[1:], "cwd": os.getcwd(), "weights": folder}, open("last_call.json", "w"))
if os.environ.get("FAKE_VIOPOSE_FAIL"):
    sys.exit(os.environ["FAKE_VIOPOSE_FAIL"])
if not os.path.isdir(folder):
    sys.exit("FileNotFoundError: " + folder)
os.makedirs("output", exist_ok=True)                              # the real script saves ./output/result.npy
open("output/result.npy", "wb").write(open("canned_result.npy", "rb").read())
"""


def write_fake_viopose(root, pose):
    """A stand-in VioPose checkout in `root` that 'estimates' exactly `pose`."""
    root = Path(root)
    (root / "Logs" / "VioPose").mkdir(parents=True)
    (root / "Logs" / "VioPose_wo_audio").mkdir()
    (root / "inference.py").write_text(FAKE_INFERENCE)
    save_viopose_result(pose, root / "canned_result.npy")
    return root


def last_call(root):
    return json.loads((Path(root) / "last_call.json").read_text())


class Boom:
    """Unpickling this runs `os.system('touch <marker>')`: proof that nothing unpickles an upload."""
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        import os
        return os.system, (f"touch {self.marker}",)


def write_malicious_npy(path, marker):
    """A genuine .npy file (object array) whose unpickling would create `marker`."""
    np.save(path, np.array([Boom(marker)], dtype=object))
