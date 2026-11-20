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
player page (bow, shoulder-tilt, elbow-angle and left-hand traces, and the skeleton beside the video). Without a pose source the
stage is skipped and those traces are absent.

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


## Fusion encoders and training (`maestro/fusion/`, `src/training/`)
Three residual autoencoders (audio, video, score) learn what the rule-based analysis does not capture. Each is a variational
autoencoder whose decoder receives, besides the latent `z` (16 numbers), the analysis features of the same segment
(`residual_autoencoder.py`):

    x -> encoder -> z --+
                        +-> decoder -> x_hat ~ x        loss = reconstruction + beta * KL + leak * crosscov(z, side)
    side (analysis) ----+

What the side features explain the decoder gets for free, so `z` is left with the rest; the KL term and the cross-covariance
penalty keep it from copying them, and `leakage_r2` (a ridge regression z -> side, scored on held-out windows) measures what is left
(`tests/test_fusion.py` shows an ordinary autoencoder keeping the known factors, this one moving them out while keeping the
hidden ones). Per modality (feature builders, encoder and decoder live in one file each):

| file | input window | side information | embedding use |
|---|---|---|---|
| `audio_encoder.py` | 86 frames (~1 s) of 64-band log-mel, frame grid identical to `audio/features.py` | voiced, pitch, confidence, RMS level, centroid, onset strength | `embed()`: one z per 0.5 s |
| `video_encoder.py` | 60 frames (2 s, 30 fps) of 20 joints, centred on the window's mean mid-shoulder and in shoulder widths | bow position/speed, left-hand distance and speed, right-hand speed, elbow angles, shoulder tilt | one z per second |
| `score_encoder.py` | 16 notes of 6 categorical tokens (pitch, duration, bar position, articulation, dynamic, hairpin) | pitch-class profile, range, density, step/skip/leap shares, articulation share, dynamic level, key, metre | one z per 8 notes, timed through the alignment |

`embed.py` loads `models/fusion/{audio,video,score}.pt`, embeds a Performance (`perf.embeddings`: per modality `t`, `z`,
`recon_error`) and is called by `analyze()` when checkpoints exist. Checkpoints are plain tensors and dicts, loaded with
`weights_only=True`.

**Training data** (see `src/training/datasets.py` for licences): audio from the violin recordings of MusicNet (Zenodo 5120004, CC BY 4.0,
downloaded and extracted by `install_data.py audio`); video from Creative Commons violin videos found with yt-dlp and run through
VioPose (`install_data.py video`); scores from the music21 corpus. The request-only sets (VioDat, SyncViolinist, MOSA) are
added with `install_data.py ingest`. Recordings are partitioned by a hash of their id into `data/<kind>/<dataset>/{train,val,test}/`, so a
recording is in exactly one split, and reinstalling never moves it. Commands: root README, "Train the models".
