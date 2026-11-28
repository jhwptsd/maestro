# Check what really works on this device (new file): the ground truth the audit and the sandbox can only estimate.
#
# `run()` tries each part of Maestro on a tiny synthetic input and reports pass or fail with the time it took and, when it failed, why: the platform, the numeric
# core, a WAV read without ffmpeg, the YIN pitch tracker, a whole analysis of a synthetic piece, scan cleaning, and then the parts a phone may not have (other
# formats through ffmpeg, the CREPE network, body movement, the music reader, a writing model). A check is `core` when the phone app depends on it. The app shows
# the list under Settings > "Check this device"; `python -m maestro.utils.selftest` prints it (JSON with --json) so a result from a real iPhone or the simulator can
# be pasted into a bug report. Peak memory is reported too, since iOS ends an app that uses too much.
import io
import json
import platform
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

SR = 22050
MIDIS = [69, 71, 72, 74, 76, 74, 72, 71]


def piece(path_wav, path_xml):
    """Write a synthetic eight-note performance (WAV) and its score (MusicXML) -> the notes' MIDI numbers."""
    y = np.zeros(int(4.6 * SR))
    for k, midi in enumerate(MIDIS):
        t = np.arange(int(0.4 * SR)) / SR
        f = 440.0 * 2 ** ((midi - 69) / 12)
        tone = sum(a * np.sin(2 * np.pi * f * h * t) for h, a in ((1, 1.0), (2, 0.5), (3, 0.25)))
        env = np.minimum(1, t / 0.01) * np.minimum(1, (0.4 - t) / 0.02)
        start = int((0.3 + 0.5 * k) * SR)
        y[start:start + len(t)] += 0.3 * tone * env
    with wave.open(str(path_wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    names = {0: ("C", 0), 2: ("D", 0), 4: ("E", 0), 5: ("F", 0), 7: ("G", 0), 9: ("A", 0), 11: ("B", 0)}
    body = "".join(f"<note><pitch><step>{names[m % 12][0]}</step><octave>{m // 12 - 1}</octave></pitch><duration>1</duration></note>" for m in MIDIS)
    attrs = '<attributes><divisions>1</divisions><time><beats>8</beats><beat-type>4</beat-type></time></attributes><direction><sound tempo="120"/></direction>'
    Path(path_xml).write_text('<?xml version="1.0"?><score-partwise version="3.1"><part-list><score-part id="P1"/></part-list>'
                              f'<part id="P1"><measure number="1">{attrs}{body}</measure></part></score-partwise>')
    return y


def _platform():
    return f"{sys.platform}, Python {platform.python_version()}, {platform.machine()}"


def _numpy():
    a = np.random.default_rng(0).standard_normal((300, 300))
    return f"numpy {np.__version__}, 300x300 product {float(np.abs(a @ a).mean()):.2f}"


def _scipy():
    import scipy
    from scipy.signal import resample_poly, stft
    stft(resample_poly(np.random.default_rng(0).standard_normal(8000), 2, 1), 22050)
    return f"scipy {scipy.__version__}: resample and stft work"


def _wav():
    from maestro.preprocessing.audio import native_wav
    with tempfile.TemporaryDirectory() as tmp:
        piece(Path(tmp) / "a.wav", Path(tmp) / "s.xml")
        y, sr = native_wav(Path(tmp) / "a.wav")
    return f"read {len(y) / sr:.1f} s without ffmpeg"


def _yin():
    from maestro.audio.pitch import estimate_pitch
    t = np.arange(SR) / SR
    track = estimate_pitch((0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SR, backend="yin")
    f0 = track.f0[track.f0 > 0]
    cents = 1200 * np.log2(np.median(f0) / 440)
    if abs(cents) > 10:
        raise AssertionError(f"A4 measured {cents:+.1f} cents off")
    return f"A4 found within {abs(cents):.1f} cents"


def _analysis():
    from maestro import analyze
    with tempfile.TemporaryDirectory() as tmp:
        piece(Path(tmp) / "a.wav", Path(tmp) / "s.xml")
        perf = analyze(Path(tmp) / "a.wav", Path(tmp) / "s.xml", Path(tmp) / "out", pitch_backend="yin")
    score = perf.assessment["intonation"]["score"]
    if score is None or score < 80:
        raise AssertionError(f"a clean synthetic piece scored {score}")
    return f"whole analysis ran; intonation {score:.0f}, {len(perf.assessment['synthesis']['events'])} events"


def _scan():
    from PIL import Image
    from maestro.score.scan import clean_page
    buffer = io.BytesIO()
    Image.fromarray((np.tile(np.linspace(110, 240, 600), (400, 1))).astype(np.uint8), "L").save(buffer, "JPEG")
    page = np.asarray(clean_page(buffer.getvalue()))
    return f"page cleaned ({page.shape[1]}x{page.shape[0]})"


def _ffmpeg():
    from maestro.utils import ffmpeg
    if not (ffmpeg.available("ffmpeg") and ffmpeg.available("ffprobe")):
        raise RuntimeError("ffmpeg is not installed: only WAV recordings can be read")
    return "ffmpeg and ffprobe found"


def _import(name, why):
    def check():
        __import__(name)
        return why
    return check


def _crepe():
    from maestro.audio.pitch import estimate_pitch
    t = np.arange(SR) / SR
    track = estimate_pitch((0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SR, backend="crepe", model="tiny")
    f0 = track.f0[track.confidence > 0.5]
    if not len(f0) or abs(1200 * np.log2(np.median(f0) / 440)) > 15:
        raise AssertionError("A4 was not found")
    return f"A4 found by CREPE ({track.runtime})"


def _pose():
    from maestro.vision.viopose_setup import load_config
    if not load_config().get("dir"):
        raise RuntimeError("VioPose is not installed (it needs a computer)")
    return "VioPose found"


def _reader():
    from maestro.score.omr import _find_audiveris
    if not _find_audiveris():
        raise RuntimeError("Audiveris is not installed (it needs Java)")
    return "Audiveris found"


def _writer():
    from maestro.feedback import ollama
    models = ollama.list_models()
    if not models:
        raise RuntimeError("no writing model is available")
    out = ollama.complete(ollama.pick_model(models), "Answer in one short sentence.", [{"role": "user", "content": "Say: ready"}], 40)
    return f"{ollama.pick_model(models)} answered: {out['text'].strip()[:60]}"


# (name, function, core): `core` = the phone app needs it
CHECKS = [("Platform", _platform, True), ("Numbers (numpy)", _numpy, True), ("Signal processing (scipy)", _scipy, True), ("Read a WAV recording", _wav, True),
          ("Pitch tracker (YIN)", _yin, True), ("Whole analysis of a piece", _analysis, True), ("Clean a scanned page (Pillow)", _scan, False),
          ("Other audio and video formats (ffmpeg)", _ffmpeg, False), ("CREPE pitch network", _crepe, False),
          ("Body movement (VioPose)", _pose, False), ("Reading a photographed score (Audiveris)", _reader, False), ("Writing model", _writer, False)]


def peak_memory_mb():
    """The most memory this process has used, in MB (None where the platform cannot say)."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, ValueError):
        return None
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)        # bytes on macOS and iOS, kilobytes elsewhere


def run(checks=None):
    """Run the checks -> [{"name", "ok", "core", "seconds", "detail"}], then a last entry for the peak memory. A failing check is a result, never an exception."""
    results = []
    for name, fn, core in checks or CHECKS:
        start = time.perf_counter()
        try:
            ok, detail = True, fn()
        except Exception as e:                                  # whatever went wrong is the answer
            ok, detail = False, f"{type(e).__name__}: {e}"[:200]
        results.append({"name": name, "ok": ok, "core": core, "seconds": round(time.perf_counter() - start, 2), "detail": detail})
    memory = peak_memory_mb()
    results.append({"name": "Peak memory", "ok": True, "core": False, "seconds": 0.0, "detail": f"{memory} MB" if memory is not None else "unknown"})
    return results


def core_ok(results):
    """True when every check the phone app depends on passed."""
    return all(r["ok"] for r in results if r["core"])


def as_text(results):
    lines = [f"{'PASS' if r['ok'] else 'FAIL'}  {r['name']} ({r['seconds']} s): {r['detail']}" for r in results]
    return "\n".join(lines + ["", "The core works on this device." if core_ok(results) else "Something the phone app needs does not work here."])


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=1) if "--json" in sys.argv else as_text(out))
