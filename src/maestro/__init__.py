# Main entry point for Maestro analysis.
import tempfile
from pathlib import Path

from maestro.alignment.align_audio_score import align_audio_score
from maestro.alignment.correspondence import check_correspondence, played_section
from maestro.assessment.events import Context
from maestro.assessment.expression import assess_expression
from maestro.assessment.intonation import assess_intonation
from maestro.assessment.neural import analyze_windows, load_backend, NeuralUnavailable
from maestro.assessment.rhythm import assess_rhythm
from maestro.assessment.synthesize import synthesize_assessment
from maestro.assessment.tone import assess_tone
from maestro.audio.features import extract_features
from maestro.audio.pitch import DEFAULT_PITCH_BACKEND, estimate_pitch
from maestro.audio.transcription import transcribe
from maestro.preprocessing.audio import preprocess_audio
from maestro.preprocessing.score import preprocess_score
from maestro.preprocessing.video import preprocess_video
from maestro.score.symbolic_features import extract_symbolic_features
from maestro.score.visualize import score_to_svg
from maestro.utils.Performance import Performance
from maestro.utils.errors import PoseUnavailableError
from maestro.assessment.bowing import assess_bowing
from maestro.assessment.posture import assess_posture
from maestro.vision.motion_features import extract_motion
from maestro.vision.pose_estimation import estimate_pose
from maestro.vision.pose_io import save_pose
from maestro.vision.visual_pitch import calibrate_from_audio, estimate_visual_pitch


def analyze_visual(perf, pose_file=None, save_dir=None, use_audio=True):
    """Visual stage: VioPose pose -> motion features -> bowing, posture and visual pitch.

    The pose is generated automatically by the installed VioPose (vision/pose_estimation.py) and used straight away;
    if save_dir is given it is also written there as pose.json. `pose_file` (a saved pose.json) skips VioPose.
    use_audio (default True) lets the visual models use the recording's audio where they can be configured to: VioPose
    runs its audio model on the extracted audio, and the visual pitch is calibrated against the notes heard in the audio
    (when at least 6 are aligned). The motion, bowing and posture metrics themselves are pose geometry and have no audio
    input. Fills perf.pose, perf.motion_features and assessment["visual_status"|"bowing"|"posture"|"visual_pitch"].
    Without VioPose (not installed, or an audio-only upload) the stage is skipped and visual_status says why.
    """
    result = perf.assessment
    wav = _write_audio(perf, save_dir) if use_audio and perf.audio and not pose_file else None
    try:
        perf.pose = estimate_pose(perf.video, wav, pose_file)
    except PoseUnavailableError as e:
        result["visual_status"] = {"available": False, "reason": str(e)}
        return perf
    finally:
        if wav and not save_dir:
            Path(wav).unlink(missing_ok=True)
    if save_dir and not pose_file:
        save_pose(perf.pose, Path(save_dir) / "pose.json")
    perf.motion_features = motion = extract_motion(perf.pose)
    result["visual_status"] = {"available": True, "reason": "", "viopose_audio": perf.pose.audio_used, "pitch_calibrated": False}
    calibration = None
    if use_audio and perf.audio_features:
        calibration = calibrate_from_audio(perf.pose, motion["hands"], perf.audio_features["notes"])
        result["visual_status"]["pitch_calibrated"] = calibration is not None
    result["visual_pitch"] = estimate_visual_pitch(perf.pose, motion["hands"], calibration)
    ctx = Context(perf)                                    # notes + alignment let the events name their measure
    result["bowing"] = assess_bowing(motion, ctx)
    result["posture"] = assess_posture(motion, perf.pose, ctx)
    return perf


def _write_audio(perf, directory):
    """The performance's audio as a wav for VioPose's audio model (in `directory`, else a temp file) -> path."""
    from scipy.io import wavfile
    path = Path(directory) / "viopose_audio.wav" if directory else Path(tempfile.mkstemp(suffix=".wav")[1])
    wavfile.write(path, perf.audio["sr"], perf.audio["samples"].astype("float32"))
    return str(path)


def analyze(video_path, score_path, out_dir=None, denoise=True,
            pitch_backend=DEFAULT_PITCH_BACKEND, pitch_options=None, pose_file=None, neural=None, visual_audio=True):
    """Analyze a performance against a score and return the filled Performance.

    video_path: any video or audio file ffmpeg reads; score_path: MusicXML, .mxl, MIDI, PDF or image.
    Intermediate files (and, if out_dir is given, score.svg) go to out_dir, default a temp dir.
    denoise: clean noise from the audio, video and score first (preprocessing/denoise.py).
    pitch_backend selects ``crepe`` (default), ``yin``, ``pyin`` or ``basic_pitch``.
    pitch_options is a dict of backend-specific settings, e.g. ``{"device": "cpu"}``
    for CREPE. Optional backends are loaded only when selected.
    The movement analysis needs VioPose (installed by the project setup); it runs automatically and its pose is saved as
    work/pose.json. pose_file (a saved pose.json) skips VioPose.
    visual_audio (default True): the visual models use the audio where they can (VioPose's audio model, visual-pitch
    calibration against the heard notes); False keeps the visual analysis audio-free.

    neural: None (default) or "clap" to add the pretrained CLAP tone/expression judgement (assessment/neural.py; needs
    transformers + torch and downloads ~600 MB once; if it cannot load, the analysis continues without it).
    assessment["synthesis"] is the single JSON report built from every module (assessment/synthesize.py).

    Raises utils.errors.InputError for unusable files, and its subclass MismatchError if the recording
    does not appear to be the given score.
    """
    work = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="maestro_"))
    work.mkdir(parents=True, exist_ok=True)
    perf = Performance()
    preprocess_video(perf, video_path, work, denoise)
    preprocess_audio(perf, denoise=denoise)
    preprocess_score(perf, score_path, work, denoise)

    y, sr = perf.audio["samples"], perf.audio["sr"]
    perf.score_features = extract_symbolic_features(perf.score)

    # Estimate once and share the track between transcription and frame features.
    track = estimate_pitch(y, sr, backend=pitch_backend, **(pitch_options or {}))
    notes = transcribe(y, sr, pitch_track=track)
    frames = extract_features(y, sr, pitch_track=track)
    perf.audio_features = {
        "frames": frames,
        "notes": notes,
        "pitch_backend": track.backend,
    }

    perf.alignment = align_audio_score(perf.score_features, notes)
    corr = check_correspondence(perf.score_features, notes, perf.alignment)
    perf.assessment = {
        "correspondence": corr,
        "section": played_section(perf.score_features, perf.alignment),
    }
    analyze_visual(perf, pose_file, work, visual_audio)                  # pose first: its results are evidence for audio-side causes
    try:
        backend = load_backend(neural)
        if backend:
            perf.assessment["neural"] = analyze_windows(y, sr, backend)
    except NeuralUnavailable as e:
        perf.assessment["neural"] = {"error": str(e)}
    ctx = Context(perf)
    a = perf.assessment
    a["intonation"] = assess_intonation(perf.score_features, notes, perf.alignment, ctx=ctx)
    a["rhythm"] = assess_rhythm(perf.score_features, notes, perf.alignment, ctx=ctx)
    a["expression"] = assess_expression(perf.score_features, notes, perf.alignment, ctx=ctx)
    a["tone"] = assess_tone(perf.score_features, notes, perf.alignment, ctx=ctx)
    a["synthesis"] = synthesize_assessment(a)
    if out_dir:
        score_to_svg(perf.score, perf.assessment["intonation"], work / "score.svg")
    return perf


__all__ = ["analyze", "DEFAULT_PITCH_BACKEND"]
