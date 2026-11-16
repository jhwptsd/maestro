# Main entry point for Maestro analysis.
import tempfile
from pathlib import Path

from maestro.alignment.align_audio_score import align_audio_score
from maestro.alignment.correspondence import check_correspondence, played_section
from maestro.assessment.intonation import assess_intonation
from maestro.assessment.rhythm import assess_rhythm
from maestro.audio.features import extract_features
from maestro.audio.pitch import DEFAULT_PITCH_BACKEND, estimate_pitch
from maestro.audio.transcription import transcribe
from maestro.preprocessing.audio import preprocess_audio
from maestro.preprocessing.score import preprocess_score
from maestro.preprocessing.video import preprocess_video
from maestro.score.symbolic_features import extract_symbolic_features
from maestro.score.visualize import score_to_svg
from maestro.utils.Performance import Performance


def analyze(video_path, score_path, out_dir=None, denoise=True,
            pitch_backend=DEFAULT_PITCH_BACKEND, pitch_options=None):
    """Analyze a performance against a score and return the filled Performance.

    video_path: any video or audio file ffmpeg reads; score_path: MusicXML, .mxl, MIDI, PDF or image.
    Intermediate files (and, if out_dir is given, score.svg) go to out_dir, default a temp dir.
    denoise: clean noise from the audio, video and score first (preprocessing/denoise.py).
    pitch_backend selects ``crepe`` (default), ``yin``, ``pyin`` or ``basic_pitch``.
    pitch_options is a dict of backend-specific settings, e.g. ``{"device": "cpu"}``
    for CREPE. Optional backends are loaded only when selected.

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
        "intonation": assess_intonation(perf.score_features, notes, perf.alignment),
        "rhythm": assess_rhythm(perf.score_features, notes, perf.alignment),
    }
    if out_dir:
        score_to_svg(perf.score, perf.assessment["intonation"], work / "score.svg")
    return perf


__all__ = ["analyze", "DEFAULT_PITCH_BACKEND"]
