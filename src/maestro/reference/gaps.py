# Camera gaps in a reference video (new file).
#
# A concert video cuts between cameras, shows the audience or the conductor, or the soloist leaves the frame or is hidden. The pose of those
# moments is not the soloist's, so it must not be measured and must not be bridged by interpolation (vision/pose_io.Pose.joint fills gaps linearly,
# which is right for a flicker in the student's own take but would invent a performance across a camera cut). Here:
#   - `scene_cuts`: the times of shot changes (ffmpeg's scene detector);
#   - `find_gaps`: stretches where the pose is lost (too few joints tracked), where a cut happened, or where the framing is clearly different
#     from the rest (the shoulders appear much wider or narrower: a close-up or a wide shot, whose movement is not comparable) -> [{"start", "end", "reason"}];
#   - `segment_motion`: the motion features (vision/motion_features.extract_motion) of each usable stretch between gaps, computed separately, so
#     nothing is measured or interpolated across a gap. Bow strokes of the stretches are returned with absolute times.
import re
import subprocess

import numpy as np

from maestro.utils.ffmpeg import tool
from maestro.vision.motion_features import extract_motion
from maestro.vision.pose_io import Pose, body_scale

CUT_THRESHOLD = 0.4              # ffmpeg scene score above which a frame starts a new shot
MIN_TRACKED = 0.5                # share of joints that must be tracked for a frame to count as the soloist
MIN_GAP_S = 0.25                 # shorter losses are tracker flicker and stay (they are interpolated by the pose layer)
MIN_SEGMENT_S = 1.0              # a stretch shorter than this is too short to measure movement in
REFRAMED = 0.3                   # a stretch whose body scale differs this much (relative) from the typical one is another framing


def scene_cuts(video_path, threshold=CUT_THRESHOLD):
    """Times (s) of shot changes in a video."""
    result = subprocess.run([tool("ffmpeg"), "-hide_banner", "-i", str(video_path), "-vf", f"select='gt(scene,{threshold})',showinfo",
                             "-an", "-f", "null", "-"], capture_output=True, text=True)
    return [float(t) for t in re.findall(r"pts_time:([0-9.]+)", result.stderr)]


def _runs(flags, fps, reason, min_s):
    """Stretches of True in per-frame `flags` of at least `min_s` seconds -> [{"start", "end", "reason"}]."""
    edges = np.diff(np.concatenate([[0], flags.astype(int), [0]]))
    return [{"start": a / fps, "end": b / fps, "reason": reason} for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if (b - a) / fps >= min_s]


def _usable(flags_bad, fps, min_s):
    """[(first frame, last frame + 1)] of the stretches of frames not flagged bad that last at least `min_s`."""
    edges = np.diff(np.concatenate([[0], (~flags_bad).astype(int), [0]]))
    return [(a, b) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if (b - a) / fps >= min_s]


def find_gaps(pose, cuts=()):
    """Gaps of a reference pose -> sorted [{"start", "end", "reason": "pose_lost" | "camera_cut" | "camera_framing_changed"}] (seconds)."""
    fps, n = pose.fps, len(pose)
    tracked = np.isfinite(pose.xyz).all(axis=2).mean(axis=1) >= MIN_TRACKED
    gaps = _runs(~tracked, fps, "pose_lost", MIN_GAP_S)
    cut_frames = np.array([round(c * fps) for c in cuts if 0 < c < n / fps], int)
    bad = ~tracked
    for c in cut_frames:                                        # the stretch after a cut is another shot: start measuring only after it settles
        bad[c:c + 2] = True
        gaps.append({"start": c / fps, "end": (c + 2) / fps, "reason": "camera_cut"})
    stretches = []                                              # usable stretches, split at the cuts too
    for a, b in _usable(bad, fps, 0):
        edges = [a, *[c for c in cut_frames if a < c < b], b]
        stretches += [(x, y) for x, y in zip(edges[:-1], edges[1:]) if (y - x) / fps >= MIN_SEGMENT_S]
    scales = [body_scale(Pose(fps, pose.joints, pose.xyz[a:b])) for a, b in stretches]
    if scales:
        typical = float(np.median(scales))
        for (a, b), scale in zip(stretches, scales):
            if abs(scale - typical) > REFRAMED * typical:
                gaps.append({"start": a / fps, "end": b / fps, "reason": "camera_framing_changed"})
    return sorted(gaps, key=lambda g: g["start"])


def segment_motion(pose, gaps):
    """Motion features of every stretch of the pose outside the gaps -> {"strokes": bow strokes with absolute times (+ "dir", "mean_speed" ...),
    "segments": [{"start", "end"}] measured}. Lengths and speeds are in shoulder widths (extract_motion's unit)."""
    bad = np.zeros(len(pose), bool)
    for g in gaps:
        bad[int(g["start"] * pose.fps):int(np.ceil(g["end"] * pose.fps))] = True
    strokes, segments = [], []
    for a, b in _usable(bad, pose.fps, MIN_SEGMENT_S):
        motion = extract_motion(Pose(pose.fps, pose.joints, pose.xyz[a:b]))
        offset = a / pose.fps
        if motion["bow"]:
            strokes += [{**s, "start": s["start"] + offset, "end": s["end"] + offset} for s in motion["bow"]["strokes"]]
        segments.append({"start": offset, "end": b / pose.fps})
    return {"strokes": strokes, "segments": segments}
