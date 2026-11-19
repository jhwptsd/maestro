# Docs
Any relevant documentation, citations, images, papers, et cetera.

## Visual (movement) analysis
The visual stage takes a 3D pose and produces movement results. Where a visual model can be configured to use audio it
does by default (`analyze(..., visual_audio=False)` turns that off): VioPose's audio model, and the visual-pitch
calibration. The motion, bowing and posture metrics are pose geometry with no audio input; the assessment events combine them
with the audio results.

    video -> vision/pose_estimation.py (VioPose, or a pose file) -> Pose (vision/pose_io.py)
          -> vision/hand_tracking.py + vision/motion_features.py   hands, bow strokes, elbow angles, shoulder tilt
          -> assessment/bowing.py, assessment/posture.py           consistency and stability metrics
          -> vision/visual_pitch.py                                pitch implied by the left hand (video only)

Results are in `Performance.assessment` (`visual_status`, `bowing`, `posture`, `visual_pitch`) and in the GUI
report ("Movement summary" and the movement sections). Without a pose source the stage is skipped and the report says why.

### Where the pose comes from
VioPose ([github.com/SeongJong-Yoo/VioPose](https://github.com/SeongJong-Yoo/VioPose), MIT, Yoo et al., WACV 2025)
generates it, automatically, for every analysis that has a video. It lifts MediaPipe 2D keypoints to 3D (18 joints in
millimetres: ankle, knee, pelvis, shoulder, elbow and wrist per side, plus index, pinky and thumb; Maestro derives each hand
as the mean of the three finger joints). There are no bow or violin keypoints, so the bow axis is the main direction of the
right hand's motion.

**Install**: the project setup does it (`setup.ps1` / `setup.sh`, or `python src/scripts/setup_viopose.py`): clone into
`third_party/VioPose`, create `third_party/viopose-env` (Python 3.9 via conda, uv or a python3.9 venv; VioPose pins
TensorFlow 2.13 so it cannot share Maestro's environment), install its requirements, download the weights by carrying out
its `download.sh` (Google Drive), and record the locations in `third_party/viopose.json`.

**Run**: `vision/pose_estimation.py` runs VioPose's `inference.py` from its own directory (it uses relative paths and always
writes `./output/result.npy`), one run at a time, reads the result and deletes it. The generated pose is saved as
`pose.json` in the analysis output folder. Overrides: `VIOPOSE_DIR`, `VIOPOSE_PYTHON` (win over the saved config),
`VIOPOSE_FOLDER` (weights folder, default `Logs/VioPose`), `VIOPOSE_AUDIO=0` (run VioPose without the audio; by default its audio model gets the
recording's audio, falling back to the video-only weights if those are missing or the run fails), `VIOPOSE_TIMEOUT`. For offline work, `analyze(..., pose_file="pose.json")` skips VioPose, and
`python src/scripts/convert_viopose.py result.npy pose.json` converts a `result.npy` you produced yourself (it is a pickle,
so Maestro only unpickles the file its own VioPose run just wrote, never an upload).

### Visual pitch estimation
`vision/visual_pitch.py` is a geometric baseline (Koepke, Wiles & Zisserman, 2019 train a CNN; that needs a labelled
dataset). A finger pressed `x` mm from the nut raises a string of length `L` by `12 * log2(L / (L - x))` semitones; the
left hand's distance from the shoulder (relative to its furthest reach, taken as first position) gives `x`. Result:
semitones above the open string per frame plus one candidate pitch per open string (the string cannot be seen). It is
approximate (assumed anatomy, a few semitones of absolute offset) and best read as *movement*. `calibrate_visual_pitch`
fits distance -> pitch from any known (distance, pitch) pairs for absolute pitch. `calibrate_from_audio` builds the pairs from the
heard notes: the audio says which string each note is on, so one curve (distance -> semitones above that string) fits all four
strings. It is used by default (`analyze(..., visual_audio=True)`) when at least 6 notes are aligned; the geometric estimate
(`visual_audio=False`) stays audio-free, which is the one to compare against the audio pitch.

References: VioPose (above); Bowing gestures: Dalmazzo & Ramirez, 2019 (a gesture classifier is future work, once audio
is integrated).
