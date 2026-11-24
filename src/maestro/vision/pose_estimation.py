# Estimate a pose from a processed video
#
# Reads Performance.video, writes Performance.pose (body/arm keypoints per frame). Feeds
# vision/motion_features.py, assessment/posture.py, and fusion/video_encoder.py.
# Uses VioPose
#
# VioPose (Yoo et al., WACV 2025; github.com/SeongJong-Yoo/VioPose, MIT licence) lifts MediaPipe 2D keypoints to
# 3D (18 joints, millimetres). It needs its own environment (Python 3.9, TensorFlow ~2.13, mediapipe, OpenCV) and
# downloaded weights, so Maestro runs it as an external program instead of importing it. Installing it is part of the
# project setup (setup.ps1 / setup.sh -> vision/viopose_setup.py), which records where it is; from then on every
# analysis with a video generates its pose automatically and uses it: nothing has to be uploaded or configured.
#
# Where VioPose is (see viopose_setup.load_config): the environment variables VIOPOSE_DIR / VIOPOSE_PYTHON, else
# third_party/viopose.json written by the setup, else third_party/VioPose. Other optional settings:
#   VIOPOSE_FOLDER   weights folder inside the checkout (default Logs/VioPose)
#   VIOPOSE_AUDIO    "0" to run VioPose without the recording's audio. By default the audio is given to VioPose's
#                    audio model (inference.py --audio_path; trained jointly on audio and video, it is more accurate
#                    for bowing), falling back to the video-only *_wo_audio weights when the audio weights are
#                    missing or that run fails. Pose.audio_used says which one produced the pose.
#   VIOPOSE_TIMEOUT  seconds before giving up (default 1800)
#
# VioPose's inference.py reads and writes paths relative to its own directory (./Logs, ./config, ./output) and
# always saves ./output/result.npy, so it is run with that directory as cwd, one run at a time (a lock), and the
# result is read back and deleted. `pose_file` (a Maestro pose JSON, e.g. a cached pose.json) bypasses VioPose; with no
# VioPose install and no pose file, PoseUnavailableError, which maestro.analyze() treats as "skip the visual analysis".
import os
import subprocess
import threading
import time
from pathlib import Path

from maestro.utils.errors import PoseUnavailableError
from maestro.vision.pose_io import load_pose, viopose_result_to_pose
from maestro.vision.viopose_setup import load_config

DEFAULT_FOLDER = "Logs/VioPose"
DEFAULT_TIMEOUT = 1800  # seconds
_one_run_at_a_time = threading.Lock()   # inference.py writes a fixed ./output/result.npy


def _run(cmd, cwd, timeout, cancel):
    """subprocess.run that can be interrupted: polls `cancel` (a threading.Event) twice a second and kills the process when
    it is set (-> PoseUnavailableError) or when `timeout` seconds have passed (-> subprocess.TimeoutExpired)."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            out, err = proc.communicate(timeout=0.5)
            return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
        except subprocess.TimeoutExpired:
            cancelled = cancel is not None and cancel.is_set()
            if cancelled or time.monotonic() > deadline:
                proc.kill()
                proc.communicate()
                if cancelled:
                    raise PoseUnavailableError("VioPose was cancelled")
                raise subprocess.TimeoutExpired(cmd, timeout)


def run_viopose(video_path, audio_path=None, root=None, python=None, folder=None, timeout=None, cancel=None):
    """Run VioPose's inference.py on `video_path` -> Pose (mm, joints per pose_io.VIOPOSE_JOINTS + hands).

    audio_path: optional wav for VioPose's audio model (without it the *_wo_audio weights are used).
    Arguments default to the installed VioPose (viopose_setup.load_config) and the VIOPOSE_* variables. Raises
    PoseUnavailableError if it is not set up, fails, or times out (the reason is in the message). `cancel`: a
    threading.Event that stops the run (and frees the one-run-at-a-time lock) when set.
    """
    config = load_config()
    root = Path(root or config["dir"] or "")
    if not (root / "inference.py").is_file():
        raise PoseUnavailableError("VioPose is not installed: run setup.ps1 / setup.sh (or "
                                   "python src/scripts/setup_viopose.py).")
    folder = folder or os.environ.get("VIOPOSE_FOLDER", DEFAULT_FOLDER)
    weights = root / (folder if audio_path else folder + "_wo_audio")       # inference.py appends _wo_audio itself
    if not weights.is_dir():
        raise PoseUnavailableError(f"VioPose weights not found at {weights}: run ./download.sh in the VioPose folder.")
    cmd = [str(python or config["python"] or "python3"), "inference.py", "--folder", folder,
           "--video_path", str(Path(video_path).resolve()), "--vis", "False"]
    if audio_path:
        cmd += ["--audio_path", str(Path(audio_path).resolve())]
    result = root / "output" / "result.npy"
    with _one_run_at_a_time:
        result.parent.mkdir(exist_ok=True)
        result.unlink(missing_ok=True)
        try:
            run = _run(cmd, root, timeout or float(os.environ.get("VIOPOSE_TIMEOUT", DEFAULT_TIMEOUT)), cancel)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise PoseUnavailableError(f"VioPose could not run: {e}")
        if run.returncode != 0 or not result.is_file():
            tail = (run.stderr or run.stdout).strip().splitlines()[-1:] or ["no output"]
            raise PoseUnavailableError(f"VioPose failed: {tail[0]}")
        try:
            pose = viopose_result_to_pose(result)       # our own subprocess's output: safe to unpickle
            pose.audio_used = bool(audio_path)
            return pose
        finally:
            result.unlink(missing_ok=True)


def estimate_pose(video_path, audio_path=None, pose_file=None, cancel=None):
    """Pose for the performance: from `pose_file` if given, else generated by running the installed VioPose.

    With `audio_path` (a wav) VioPose's audio model is used unless VIOPOSE_AUDIO=0; if its weights are missing or that
    run fails, the video-only model is run instead (one retry, never a silent loss of the pose)."""
    if pose_file:
        return load_pose(pose_file)
    if not video_path:
        raise PoseUnavailableError("Audio-only upload: there is no video to estimate a pose from.")
    if not audio_path or os.environ.get("VIOPOSE_AUDIO") == "0":
        return run_viopose(video_path, cancel=cancel)
    try:
        return run_viopose(video_path, audio_path, cancel=cancel)
    except PoseUnavailableError as e:
        if any(word in str(e) for word in ("not installed", "could not run", "cancelled")):   # nothing to fall back to, a timeout or
            raise                                                                           # a cancel: do not run twice
        return run_viopose(video_path, cancel=cancel)
