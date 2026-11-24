# Main entry point for Maestro analysis.
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from maestro.alignment.align_audio_score import align_audio_score
from maestro.alignment.correspondence import check_correspondence, played_section
from maestro.assessment.events import Context, unavailable
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
from maestro.reference.compare import compare_to_reference
from maestro.reference.process import process_reference
from maestro.score.symbolic_features import extract_symbolic_features
from maestro.score.visualize import score_to_svg
from maestro.utils.Performance import Performance
from maestro.utils.errors import InputError, PoseUnavailableError, ToolMissingError
from maestro.assessment.bowing import assess_bowing
from maestro.assessment.posture import assess_posture
from maestro.assessment.reference import assess_reference
from maestro.vision.motion_features import extract_motion
from maestro.vision.pose_estimation import estimate_pose
from maestro.vision.pose_io import save_pose
from maestro.vision.visual_pitch import calibrate_from_audio, estimate_visual_pitch


@contextmanager
def _timed(perf, stage):
    """Add the seconds spent in the block to perf.timings[stage]."""
    start = time.perf_counter()
    try:
        yield
    finally:
        perf.timings[stage] = round(perf.timings.get(stage, 0.0) + time.perf_counter() - start, 3)


def _generate_pose(perf, pose_file, save_dir, use_audio, cancel=None):
    """The pose of perf's video (see analyze_visual): from `pose_file`, else VioPose, with the audio as a wav if asked.
    Runs in a background thread during analyze(). Raises PoseUnavailableError."""
    start = time.perf_counter()
    wav = _write_audio(perf, save_dir) if use_audio and perf.audio and not pose_file else None
    try:
        return estimate_pose(perf.video, wav, pose_file, cancel)
    finally:
        if wav and not save_dir:
            Path(wav).unlink(missing_ok=True)
        perf.timings["pose"] = round(time.perf_counter() - start, 3)


def analyze_visual(perf, pose_file=None, save_dir=None, use_audio=True, pose_job=None):
    """Visual stage: VioPose pose -> motion features -> bowing, posture and visual pitch.

    The pose is generated automatically by the installed VioPose (vision/pose_estimation.py) and used straight away;
    if save_dir is given it is also written there as pose.json. `pose_file` (a saved pose.json) skips VioPose.
    use_audio (default True) lets the visual models use the recording's audio where they can be configured to: VioPose
    runs its audio model on the extracted audio, and the visual pitch is calibrated against the notes heard in the audio
    (when at least 6 are aligned). The motion, bowing and posture metrics themselves are pose geometry and have no audio
    input. Fills perf.pose, perf.motion_features and assessment["visual_status"|"bowing"|"posture"|"visual_pitch"].
    Without VioPose (not installed, or an audio-only upload) the stage is skipped and visual_status says why.
    `pose_job`: a Future of `_generate_pose` started earlier (analyze() does, so the pose is estimated while the audio is
    analysed); without it the pose is generated here.
    """
    result = perf.assessment
    try:
        perf.pose = pose_job.result() if pose_job is not None else _generate_pose(perf, pose_file, save_dir, use_audio)
    except PoseUnavailableError as e:
        result["visual_status"] = {"available": False, "reason": str(e)}
        return perf
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
            pitch_backend=DEFAULT_PITCH_BACKEND, pitch_options=None, pose_file=None, neural=None, visual_audio=True, progress=None,
            reference=None, reference_options=None):
    """Analyze a performance against a score and return the filled Performance.

    video_path: any video or audio file ffmpeg reads; score_path: MusicXML, .mxl, MIDI, PDF or image.
    Intermediate files (and, if out_dir is given, score.svg) go to out_dir, default a temp dir.
    denoise: clean noise from the audio, video and score first (preprocessing/denoise.py).
    pitch_backend selects ``crepe`` (default), ``yin``, ``pyin`` or ``basic_pitch``.
    pitch_options is a dict of backend-specific settings, e.g. ``{"device": "cpu"}``
    for CREPE: "cpu", "cuda:0", "mps", or "onnx" for ONNX Runtime's GPU provider (DirectML = AMD/Intel on Windows). The default
    "auto" picks the best available (utils/device.py; MAESTRO_DEVICE overrides) and audio_features["pitch_runtime"] says what
    ran. Optional backends are loaded only when selected.
    The movement analysis needs VioPose (installed by the project setup); it runs automatically and its pose is saved as
    work/pose.json. pose_file (a saved pose.json) skips VioPose. The pose is estimated in the background while the audio
    is analysed (and cancelled if the analysis fails); perf.timings has the seconds per stage.
    visual_audio (default True): the visual models use the audio where they can (VioPose's audio model, visual-pitch
    calibration against the heard notes); False keeps the visual analysis audio-free.

    progress: optional callable, called with the name of each stage as it starts ("video", "audio", "score", "pitch", "notes", "align",
    "pose", "visual", "neural", "assess") and with "done" at the end (utils/progress.Tracker turns these into a progress bar).
    neural: None (default) or "clap" to add the pretrained CLAP tone/expression judgement (assessment/neural.py; needs
    transformers + torch and downloads ~600 MB once; if it cannot load, the analysis continues without it).
    reference: a professional recording (file path or YouTube link) to compare with. It is processed after the student's analysis (reference/):
    clipped to the bars the student played, the soloist isolated, gaps kept where the camera or the sound is lost, and its measurements
    become a baseline with the soloist's interpretive choices marked; the differences are events of the module "reference" (assessment/reference.py,
    in the synthesis like the other modules, with interpretive choices as their cause) and the baseline is perf.reference. A reference
    that cannot be used does not fail the analysis: assessment["reference"] is then an unavailable module that says why. reference_options: keyword arguments of
    reference.process.process_reference (measures, separate, pose_file).
    assessment["synthesis"] is the single JSON report built from every module (assessment/synthesize.py).

    Raises utils.errors.InputError for unusable files, and its subclass MismatchError if the recording
    does not appear to be the given score.
    """
    work = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="maestro_"))
    work.mkdir(parents=True, exist_ok=True)
    perf = Performance()
    step = progress or (lambda stage: None)
    with _timed(perf, "preprocess"):
        step("video")
        preprocess_video(perf, video_path, work, denoise)
        step("audio")
        preprocess_audio(perf, denoise=denoise)
        step("score")
        preprocess_score(perf, score_path, work, denoise)

    # The pose (VioPose, minutes of CPU) does not depend on the audio analysis, so it runs in the background while the audio,
    # score and alignment are analysed, and is cancelled if the analysis fails (e.g. wrong score) so that nothing keeps running.
    cancel = threading.Event()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pose")
    pose_job = pool.submit(_generate_pose, perf, pose_file, work, visual_audio, cancel)
    try:
        _analyze_audio_and_assess(perf, work, pose_file, visual_audio, pitch_backend, pitch_options, neural, pose_job, step, reference, reference_options or {})
    except BaseException:
        cancel.set()
        raise
    finally:
        pool.shutdown(wait=False)
    if out_dir:
        score_to_svg(perf.score, perf.assessment["intonation"], work / "score.svg")
    step("done")
    return perf


def _process_reference(perf, reference, work, pitch_backend, pitch_options, step, options):
    """The reference stage of analyze(): process the professional recording. An unusable reference is reported (as an unavailable
    "reference" module), not raised."""
    try:
        with _timed(perf, "reference"):
            process_reference(perf, reference, work, pitch_backend=pitch_backend, pitch_options=pitch_options, progress=step, **options)
    except (InputError, ToolMissingError) as e:                 # (MismatchError is an InputError)
        perf.assessment["reference"] = unavailable("reference", str(e))


def _analyze_audio_and_assess(perf, work, pose_file, visual_audio, pitch_backend, pitch_options, neural, pose_job, step, reference=None, reference_options=None):
    """The stages of analyze() that follow preprocessing (split out so the pose job can be cancelled around them)."""
    y, sr = perf.audio["samples"], perf.audio["sr"]
    perf.score_features = extract_symbolic_features(perf.score)

    # Estimate once and share the track between transcription and frame features.
    step("pitch")
    with _timed(perf, "pitch"):
        track = estimate_pitch(y, sr, backend=pitch_backend, **(pitch_options or {}))
    step("notes")
    with _timed(perf, "transcribe"):
        notes = transcribe(y, sr, pitch_track=track)
    with _timed(perf, "features"):
        frames = extract_features(y, sr, pitch_track=track)
    perf.audio_features = {
        "frames": frames,
        "notes": notes,
        "pitch_backend": track.backend,
        "pitch_runtime": track.runtime,
    }

    step("align")
    with _timed(perf, "align"):
        perf.alignment = align_audio_score(perf.score_features, notes)
        corr = check_correspondence(perf.score_features, notes, perf.alignment)
    perf.assessment = {
        "correspondence": corr,
        "section": played_section(perf.score_features, perf.alignment),
    }
    step("pose")
    with _timed(perf, "pose_wait"):                      # how long the analysis had to wait for the pose after the audio was done
        pose_job.exception()
    step("visual")
    with _timed(perf, "visual"):
        analyze_visual(perf, pose_file, work, visual_audio, pose_job)    # pose first: its results are evidence for audio-side causes
    if neural:
        step("neural")
    with _timed(perf, "neural"):
        try:
            backend = load_backend(neural)
            if backend:
                perf.assessment["neural"] = analyze_windows(y, sr, backend)
        except NeuralUnavailable as e:
            perf.assessment["neural"] = {"error": str(e)}
    if reference:                                        # after the student's own analysis (it needs the bars played and the movement), before assess
        _process_reference(perf, reference, work, pitch_backend, pitch_options, step, reference_options or {})
    step("assess")
    with _timed(perf, "assess"):
        ctx = Context(perf)
        a = perf.assessment
        a["intonation"] = assess_intonation(perf.score_features, notes, perf.alignment, ctx=ctx)
        a["rhythm"] = assess_rhythm(perf.score_features, notes, perf.alignment, ctx=ctx)
        a["expression"] = assess_expression(perf.score_features, notes, perf.alignment, ctx=ctx)
        a["tone"] = assess_tone(perf.score_features, notes, perf.alignment, ctx=ctx)
        if perf.reference:                               # the comparison as events in the common format, like every other module
            a["reference"] = assess_reference(compare_to_reference(perf), ctx)
        a["synthesis"] = synthesize_assessment(a)


__all__ = ["analyze", "DEFAULT_PITCH_BACKEND"]
