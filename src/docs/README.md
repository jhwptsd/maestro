# Docs
Any relevant documentation, citations, images, papers, et cetera.

## Visual (movement) analysis
The visual stage is independent of the audio tools: it takes a 3D pose and produces movement results, using no audio
results at all (combining the two comes later).

    video -> vision/pose_estimation.py (VioPose, or a pose file) -> Pose (vision/pose_io.py)
          -> vision/hand_tracking.py + vision/motion_features.py   hands, bow strokes, elbow angles, shoulder tilt
          -> assessment/bowing.py, assessment/posture.py           consistency and stability metrics
          -> vision/visual_pitch.py                                pitch implied by the left hand (video only)
          -> vision/mocap_compare.py (optional)                    accuracy against motion-capture annotations

Results are in `Performance.assessment` (`visual_status`, `bowing`, `posture`, `visual_pitch`, `mocap`) and in the GUI
report ("Movement summary" and the movement sections). Without a pose source the stage is skipped and the report says why.

### Getting a pose
1. **VioPose** ([github.com/SeongJong-Yoo/VioPose](https://github.com/SeongJong-Yoo/VioPose), MIT, Yoo et al., WACV 2025).
   It runs as an external program in its own environment (Python 3.9, TensorFlow ~2.13, mediapipe, OpenCV):
   ```
   git clone https://github.com/SeongJong-Yoo/VioPose && cd VioPose
   conda create -n viopose python==3.9 && conda activate viopose && pip install -r requirements.txt && ./download.sh
   ```
   (`download.sh` fetches the weights from Google Drive.) Then tell Maestro where it is:
   `VIOPOSE_DIR=/path/to/VioPose`, `VIOPOSE_PYTHON=/path/to/conda/envs/viopose/bin/python`; optional `VIOPOSE_FOLDER`
   (weights folder, default `Logs/VioPose`), `VIOPOSE_AUDIO=1` (also give VioPose the audio for its audio model; off by
   default), `VIOPOSE_TIMEOUT`. VioPose outputs 18 joints in millimetres (ankle/knee/pelvis/shoulder/elbow/wrist per side
   plus index, pinky and thumb); Maestro derives each hand as the mean of the three finger joints. There are no bow or
   violin keypoints, so the bow axis is the main direction of the right hand's motion.
2. **A pose file**: `.json` (Maestro format, see `vision/pose_io.py`), `.csv`, `.trc`, or `.npy` with a sidecar `.json`.
   `analyze(..., pose_file=...)` or the GUI's optional "Pose file". VioPose's own `result.npy` is a pickle, so Maestro never
   loads one from an upload: convert your own with `python src/scripts/convert_viopose.py result.npy pose.json`.

`analyze(..., mocap_file=...)` (or the GUI's "Motion-capture annotations") compares the pose with a motion-capture
recording of the same performance after matching their clocks, units and axes.

### Visual pitch estimation
`vision/visual_pitch.py` is a geometric baseline (Koepke, Wiles & Zisserman, 2019 train a CNN; that needs a labelled
dataset). A finger pressed `x` mm from the nut raises a string of length `L` by `12 * log2(L / (L - x))` semitones; the
left hand's distance from the shoulder (relative to its furthest reach, taken as first position) gives `x`. Result:
semitones above the open string per frame plus one candidate pitch per open string (the string cannot be seen). It is
approximate (assumed anatomy, a few semitones of absolute offset) and best read as *movement*. `calibrate_visual_pitch`
fits distance -> pitch from any known (distance, pitch) pairs for absolute pitch.

References: VioPose (above); Bowing gestures: Dalmazzo & Ramirez, 2019 (a gesture classifier is future work, once audio
is integrated).
