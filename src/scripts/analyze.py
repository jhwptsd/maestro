"""CLI: python src/scripts/analyze.py VIDEO SCORE [OUT_DIR] [--pitch-backend NAME]."""
import argparse
import json
from pathlib import Path

from maestro import analyze
from maestro.audio.pitch import DEFAULT_PITCH_BACKEND, list_pitch_backends
from maestro.gui.report import build_page, copy_media
from maestro.utils.errors import InputError, ToolMissingError


def main():
    parser = argparse.ArgumentParser(description="Analyze a violin performance against a score.")
    parser.add_argument("video", help="Video or audio recording")
    parser.add_argument("score", help="MusicXML, MIDI, PDF or image score")
    parser.add_argument("out_dir", nargs="?", default="out", help="Output directory (default: out)")
    parser.add_argument(
        "--pitch-backend", type=lambda value: value.lower().replace("-", "_"),
        choices=list_pitch_backends(), default=DEFAULT_PITCH_BACKEND,
        help="Pitch/transcription backend (default: crepe)",
    )
    parser.add_argument("--crepe-device", default=None, help="CREPE device, e.g. cpu or cuda:0")
    parser.add_argument("--crepe-model", choices=("tiny", "full"), default="full")
    parser.add_argument("--neural", choices=("clap",), default=None,
                        help="add the pretrained CLAP tone/expression judgement (needs transformers + torch)")
    args = parser.parse_args()

    pitch_options = {}
    if args.pitch_backend == "crepe":
        if args.crepe_device is not None:
            pitch_options["device"] = args.crepe_device
        pitch_options["model"] = args.crepe_model

    out = Path(args.out_dir)
    try:
        perf = analyze(
            args.video,
            args.score,
            out,
            pitch_backend=args.pitch_backend,
            pitch_options=pitch_options,
            neural=args.neural,
        )
    except (InputError, ToolMissingError, ImportError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")

    (out / "report.html").write_text(build_page(perf, copy_media(perf, out)), encoding="utf-8")
    (out / "assessment.json").write_text(json.dumps(perf.assessment["synthesis"], indent=1), encoding="utf-8")
    print(f"report: {out / 'report.html'}\nassessment events (JSON): {out / 'assessment.json'}")
    print(f"pitch backend: {perf.audio_features['pitch_backend']} ({perf.audio_features.get('pitch_runtime') or 'n/a'})")
    print("seconds per stage: " + ", ".join(f"{k} {v:.1f}" for k, v in perf.timings.items()) + "  (pose runs alongside the audio stages)")
    intonation, rhythm = perf.assessment["intonation"], perf.assessment["rhythm"]
    print(
        f"intonation: mean |dev| {intonation['mean_abs_cents']} cents, "
        f"tuning offset {intonation['tuning_offset_cents']}, "
        f"wrong notes {intonation['n_wrong_notes']}"
    )
    print(f"rhythm: tempo ratio {rhythm['tempo_ratio']}, mean |error| {rhythm['mean_abs_error']} s")


if __name__ == "__main__":
    main()
