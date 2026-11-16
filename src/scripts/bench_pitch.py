"""Time every pitch backend that is installed: python src/scripts/bench_pitch.py [audio-or-video-file]

Run with PYTHONPATH=src (and src/tests for the synthetic default input). Without a file it tracks ~60 s of a
synthetic violin performance. For each backend it reports the first call (which includes loading models and
imports) and the best of the following runs, as seconds and as a multiple of real time (higher = faster).
Backends whose optional dependency is missing are skipped with the install hint.

Speed levers worth trying per backend (measure with this script before changing defaults):
  crepe        --model tiny (~10x fewer parameters than full), --device cuda:0 or mps, a larger --batch-size on GPU
  yin          already vectorised (hundreds of x real time on one CPU core)
  pyin         the slowest on CPU (a Viterbi decode over many pitch bins); lower `resolution` / `n_thresholds` via
               pitch_options if its accuracy is not needed
  basic_pitch  runs the polyphonic model once per file; only worth it for double stops
"""
import argparse
import sys
import time

import numpy as np

from maestro.audio.pitch import estimate_pitch, list_pitch_backends

HINTS = {"crepe": "pip install -e '.[crepe]'", "pyin": "pip install -e '.[pyin]'",
         "basic_pitch": "pip install -e '.[basic-pitch]'"}


def load_audio(path):
    if path:
        from maestro.preprocessing.audio import extract_audio
        return extract_audio(path)
    sys.path.insert(0, "src/tests")
    from minuet import SR, play_minuet
    return np.tile(play_minuet(), 4), SR                          # ~64 s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file", nargs="?", help="audio/video file (default: synthetic ~60 s violin)")
    ap.add_argument("--repeats", type=int, default=3, help="timed runs after the first call")
    ap.add_argument("--backends", nargs="*", default=list_pitch_backends())
    ap.add_argument("--model", choices=["tiny", "full"], help="CREPE model size")
    ap.add_argument("--device", help="CREPE device, e.g. cpu, cuda:0")
    ap.add_argument("--batch-size", type=int, help="CREPE batch size")
    args = ap.parse_args()
    y, sr = load_audio(args.file)
    print(f"{len(y) / sr:.0f} s of audio at {sr} Hz\n{'backend':12s} {'first call':>11s} {'best run':>10s} {'x real time':>12s}")
    for name in args.backends:
        opts = {}
        if name == "crepe":
            opts = {k: v for k, v in (("model", args.model), ("device", args.device), ("batch_size", args.batch_size)) if v}
        try:
            t = time.perf_counter()
            estimate_pitch(y, sr, backend=name, **opts)
            first = time.perf_counter() - t
        except ImportError:
            print(f"{name:12s} skipped (not installed: {HINTS.get(name, '')})")
            continue
        runs = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            estimate_pitch(y, sr, backend=name, **opts)
            runs.append(time.perf_counter() - t)
        best = min(runs) if runs else first
        print(f"{name:12s} {first:10.2f}s {best:9.2f}s {len(y) / sr / best:11.0f}x")


if __name__ == "__main__":
    main()
