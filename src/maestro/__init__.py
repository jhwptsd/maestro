# Main function for the app
#
# Entry point: builds a Performance (utils/Performance.py), then runs the pipeline in order:
# preprocessing -> audio/vision/score feature extraction -> alignment -> assessment (+ optional
# fusion model) -> feedback. Each stage reads from and writes to the shared Performance object.
#
# Implemented so far: audio extraction, score conversion/parsing, transcription, score-audio
# alignment, intonation + rhythm assessment, score visualization. Vision, fusion and feedback are TODO.
import tempfile
from pathlib import Path

from maestro.alignment.align_audio_score import align_audio_score
from maestro.alignment.correspondence import check_correspondence, played_section
from maestro.assessment.intonation import assess_intonation
from maestro.assessment.rhythm import assess_rhythm
from maestro.audio.features import extract_features
from maestro.audio.transcription import transcribe
from maestro.preprocessing.audio import preprocess_audio
from maestro.preprocessing.score import preprocess_score
from maestro.preprocessing.video import preprocess_video
from maestro.score.symbolic_features import extract_symbolic_features
from maestro.score.visualize import score_to_svg
from maestro.utils.Performance import Performance


def analyze(video_path, score_path, out_dir=None):
    """Run the implemented pipeline on a video + score; returns the filled Performance.

    `video_path` may be any video or audio file ffmpeg reads; `score_path` MusicXML/.mxl/MIDI/PDF/image.
    Intermediate files and (if `out_dir` is given) the score SVG go to `out_dir` (default: a temp dir).
    Raises utils.errors.InputError for unusable files, and MismatchError (a subclass) if the recording
    does not appear to be the given score.
    """
    work = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="maestro_"))
    perf = Performance()
    preprocess_video(perf, video_path, work)
    preprocess_audio(perf)
    preprocess_score(perf, score_path, work)

    y, sr = perf.audio["samples"], perf.audio["sr"]
    perf.score_features = extract_symbolic_features(perf.score)
    notes = transcribe(y, sr)
    perf.audio_features = {"frames": extract_features(y, sr), "notes": notes}

    perf.alignment = align_audio_score(perf.score_features, notes)
    corr = check_correspondence(perf.score_features, notes, perf.alignment)  # wrong score -> MismatchError
    perf.assessment = {"correspondence": corr, "section": played_section(perf.score_features, perf.alignment),
                       "intonation": assess_intonation(perf.score_features, notes, perf.alignment),
                       "rhythm": assess_rhythm(perf.score_features, notes, perf.alignment)}
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        score_to_svg(perf.score, perf.assessment["intonation"], Path(out_dir) / "score.svg")
    return perf
