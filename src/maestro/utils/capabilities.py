# What this device can do (new file).
#
# Maestro's full analysis uses programs and packages that exist on a computer but not on a phone: ffmpeg (any video or audio format), the CREPE
# network (ONNX Runtime or PyTorch), VioPose (body movement; TensorFlow in its own Python 3.9), Audiveris (reading a scanned score; Java) and
# Ollama (the writing model). The app asks `detect()` and offers what works instead of failing: on a phone only the pure-Python core remains, which is
# a WAV recording (ffmpeg-free path of preprocessing/audio.py), the YIN pitch tracker, and a MusicXML or MIDI score. Nothing is imported here:
# packages are looked for with importlib.util.find_spec, so asking is cheap and safe.
import importlib.util
import shutil
import sys

from maestro.utils import ffmpeg


def _has(module):
    return importlib.util.find_spec(module) is not None


def crepe_possible():
    """True when the CREPE network can run here: ONNX Runtime with an exported model (found, or exportable because torch, torchcrepe and onnx are there to do
    it), or torchcrepe itself."""
    if _has("torchcrepe"):
        return True
    if not _has("onnxruntime"):
        return False
    from maestro.audio.backends import crepe_onnx
    return crepe_onnx.find_model("tiny") is not None or (_has("torch") and _has("onnx"))


def detect():
    """-> {"ffmpeg", "pitch_crepe", "pose", "scan_reader", "scan_camera", "writing_model"?}: True where that part works on this device. (Whether
    Ollama is running is a network question, asked by feedback/ollama.py.)"""
    from maestro.vision.viopose_setup import load_config
    from maestro.score.omr import _find_audiveris
    viopose = False
    try:
        cfg = load_config()
        viopose = bool(cfg) and bool(cfg.get("dir"))
    except Exception:                                           # a broken config is the same as none
        viopose = False
    return {"ffmpeg": ffmpeg.available("ffmpeg") and ffmpeg.available("ffprobe"),
            "pitch_crepe": crepe_possible(),
            "pose": viopose,
            "scan_reader": _find_audiveris() is not None,
            "scan_cleaner": _has("PIL"),
            "mobile": sys.platform in ("ios", "android") or hasattr(sys, "getandroidapilevel")}


def pitch_backend():
    """The pitch tracker to use here: CREPE when its network can run, else the pure-numpy YIN (what a phone uses)."""
    return "crepe" if detect()["pitch_crepe"] else "yin"


def summary():
    """For the settings screen: [(what, works, what it gives you / how to get it)]."""
    d = detect()
    return [("Any video or audio format", d["ffmpeg"], "needs ffmpeg (" + ffmpeg.INSTALL_HINT.get(sys.platform, "install it with your package manager") + "); without it, WAV recordings only"),
            ("Accurate pitch tracking (CREPE)", d["pitch_crepe"], "needs ONNX Runtime and its exported model, or torchcrepe; otherwise the lighter YIN tracker is used"),
            ("Body movement (bowing, posture)", d["pose"], "needs VioPose (computer only, see the README)"),
            ("Reading a photographed score", d["scan_reader"], "needs Audiveris; otherwise use a MusicXML or MIDI score"),
            ("Cleaning scanned pages", d["scan_cleaner"], "needs Pillow")]
