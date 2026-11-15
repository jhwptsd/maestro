"""CLI: python src/scripts/analyze.py VIDEO_OR_AUDIO SCORE [OUT_DIR]  (run with PYTHONPATH=src or after `pip install -e .`).

Prints the intonation and timing summaries; writes the annotated score SVG into OUT_DIR (default ./out).
"""
import sys
from pathlib import Path

from maestro import analyze
from maestro.gui.report import build_report, standalone_page
from maestro.utils.errors import InputError, OMRUnavailableError

if __name__ == "__main__":
    video, score = sys.argv[1:3]
    try:
        perf = analyze(video, score, sys.argv[3] if len(sys.argv) > 3 else "out")
    except (InputError, OMRUnavailableError) as e:  # bad file, score/recording mismatch, or no Audiveris for PDFs
        sys.exit(f"error: {e}")
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "out")
    (out / "report.html").write_text(standalone_page(build_report(perf)), encoding="utf-8")
    print(f"report: {out / 'report.html'}")
    i, r = perf.assessment["intonation"], perf.assessment["rhythm"]
    print(f"intonation: mean |dev| {i['mean_abs_cents']} cents, tuning offset {i['tuning_offset_cents']}, "
          f"wrong notes {i['n_wrong_notes']}")
    print(f"rhythm: tempo ratio {r['tempo_ratio']}, mean |error| {r['mean_abs_error']} s")
