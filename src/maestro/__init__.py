# Main function for the app
#
# Entry point: builds a Performance (utils/Performance.py), then runs the pipeline in order:
# preprocessing -> audio/vision/score feature extraction -> alignment -> assessment (+ optional
# fusion model) -> feedback. Each stage reads from and writes to the shared Performance object.
#
# Implemented so far: audio extraction, score conversion/parsing, transcription, score-audio
# alignment, intonation + rhythm assessment, score visualization. Vision, fusion and feedback are TODO.
from pathlib import Path

from maestro.alignment.align_audio_score import align_audio_score
from maestro.assessment.intonation import assess_intonation
from maestro.assessment.rhythm import assess_rhythm
from maestro.audio.features import extract_features
from maestro.audio.transcription import transcribe
from maestro.preprocessing.audio import preprocess_audio
from maestro.preprocessing.score import preprocess_score
from maestro.score.symbolic_features import extract_symbolic_features
from maestro.score.visualize import score_to_svg
from maestro.utils.Performance import Performance


def analyze(video_path, score_path, out_dir=None):
    """Run the implemented pipeline on a video + score; returns the filled Performance.

    If `out_dir` is given, a score SVG annotated with intonation is written to out_dir/score.svg.
    """
    perf = Performance()
    perf.video = str(video_path)
    preprocess_audio(perf)
    preprocess_score(perf, score_path, out_dir)

    y, sr = perf.audio["samples"], perf.audio["sr"]
    perf.score_features = extract_symbolic_features(perf.score)
    notes = transcribe(y, sr)
    perf.audio_features = {"frames": extract_features(y, sr), "notes": notes}

    perf.alignment = align_audio_score(perf.score_features, notes)
    perf.assessment = {"intonation": assess_intonation(perf.score_features, notes, perf.alignment),
                       "rhythm": assess_rhythm(perf.score_features, notes, perf.alignment)}
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        score_to_svg(perf.score, perf.assessment["intonation"], Path(out_dir) / "score.svg")
    return perf
