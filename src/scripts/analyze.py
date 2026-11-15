"""CLI: python src/scripts/analyze.py VIDEO SCORE [OUT_DIR]  (run with PYTHONPATH=src or after `pip install -e .`).

Prints the intonation and timing summaries; writes the annotated score SVG into OUT_DIR (default ./out).
"""
import sys

from maestro import analyze

if __name__ == "__main__":
    video, score = sys.argv[1:3]
    perf = analyze(video, score, sys.argv[3] if len(sys.argv) > 3 else "out")
    i, r = perf.assessment["intonation"], perf.assessment["rhythm"]
    print(f"intonation: mean |dev| {i['mean_abs_cents']} cents, tuning offset {i['tuning_offset_cents']}, "
          f"wrong notes {i['n_wrong_notes']}")
    print(f"rhythm: tempo ratio {r['tempo_ratio']}, mean |error| {r['mean_abs_error']} s")
