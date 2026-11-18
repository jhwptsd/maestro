# Estimate a pose from a processed video
#
# Reads Performance.video, writes Performance.pose (body/arm keypoints per frame). Feeds
# vision/motion_features.py, assessment/posture.py, and fusion/video_encoder.py.
# Uses VioPose
#
# VioPose (Yoo et al., WACV 2025; github.com/SeongJong-Yoo/VioPose, MIT licence) lifts MediaPipe 2D keypoints to
# 3D (18 joints, millimetres). It needs its own environment (Python 3.9, TensorFlow ~2.13, mediapipe, OpenCV) and
# weights downloaded with its `download.sh`, so Maestro runs it as an external program instead of importing it:
#
#   VIOPOSE_DIR      checkout of the VioPose repository, with `./download.sh` already run (required)
#   VIOPOSE_PYTHON   interpreter of VioPose's environment, e.g. ~/miniconda3/envs/viopose/bin/python (default: python3)
#   VIOPOSE_FOLDER   weights folder inside VIOPOSE_DIR (default: Logs/VioPose)
#   VIOPOSE_AUDIO    "1" to give VioPose the recording's audio track as well (its audio model; default off:
#                    the visual analysis is kept independent of audio for now)
#   VIOPOSE_TIMEOUT  seconds before giving up (default 1800)
#
# VioPose's inference.py reads and writes paths relative to its own directory (./Logs, ./config, ./output) and
# always saves ./output/result.npy, so it is run with that directory as cwd, one run at a time (a lock), and the
# result is read back and deleted. Pose sources, in priority order: a pose file (vision/pose_io.py formats), then
# VioPose; otherwise PoseUnavailableError, which maestro.analyze() treats as "skip the visual analysis".
import os
import subprocess
import threading
from pathlib import Path

from maestro.utils.errors import PoseUnavailableError
from maestro.vision.pose_io import load_pose, viopose_result_to_pose

DEFAULT_FOLDER = "Logs/VioPose"
DEFAULT_TIMEOUT = 1800  # seconds
_one_run_at_a_time = threading.Lock()   # inference.py writes a fixed ./output/result.npy


def run_viopose(video_path, audio_path=None, root=None, python=None, folder=None, timeout=None):
    """Run VioPose's inference.py on `video_path` -> Pose (mm, joints per pose_io.VIOPOSE_JOINTS + hands).

    audio_path: optional wav for VioPose's audio model (without it the *_wo_audio weights are used).
    Arguments default to the VIOPOSE_* environment variables. Raises PoseUnavailableError if it is not set up,
    fails, or times out (the reason is in the message).
    """
    root = Path(root or os.environ.get("VIOPOSE_DIR") or "")
    if not (root / "inference.py").is_file():
        raise PoseUnavailableError("VioPose is not set up: point VIOPOSE_DIR at a VioPose checkout (see vision/pose_estimation.py).")
    folder = folder or os.environ.get("VIOPOSE_FOLDER", DEFAULT_FOLDER)
    weights = root / (folder if audio_path else folder + "_wo_audio")       # inference.py appends _wo_audio itself
    if not weights.is_dir():
        raise PoseUnavailableError(f"VioPose weights not found at {weights}: run ./download.sh in the VioPose folder.")
    cmd = [python or os.environ.get("VIOPOSE_PYTHON", "python3"), "inference.py", "--folder", folder,
           "--video_path", str(Path(video_path).resolve()), "--vis", "False"]
    if audio_path:
        cmd += ["--audio_path", str(Path(audio_path).resolve())]
    result = root / "output" / "result.npy"
    with _one_run_at_a_time:
        result.parent.mkdir(exist_ok=True)
        result.unlink(missing_ok=True)
        try:
            run = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                                 timeout=timeout or float(os.environ.get("VIOPOSE_TIMEOUT", DEFAULT_TIMEOUT)))
        except (OSError, subprocess.TimeoutExpired) as e:
            raise PoseUnavailableError(f"VioPose could not run: {e}")
        if run.returncode != 0 or not result.is_file():
            tail = (run.stderr or run.stdout).strip().splitlines()[-1:] or ["no output"]
            raise PoseUnavailableError(f"VioPose failed: {tail[0]}")
        try:
            return viopose_result_to_pose(result)       # our own subprocess's output: safe to unpickle
        finally:
            result.unlink(missing_ok=True)


def estimate_pose(video_path, audio_path=None, pose_file=None):
    """Pose for the performance: from `pose_file` if given, else by running VioPose (see the module comment)."""
    if pose_file:
        return load_pose(pose_file)
    if not video_path:
        raise PoseUnavailableError("Audio-only upload: there is no video to estimate a pose from.")
    if not os.environ.get("VIOPOSE_DIR"):
        raise PoseUnavailableError("No pose source: upload a pose file, or set VIOPOSE_DIR to run VioPose "
                                   "(see vision/pose_estimation.py).")
    return run_viopose(video_path, audio_path if os.environ.get("VIOPOSE_AUDIO") == "1" else None)
