# Load a professional reference recording from a file (rewritten from the outline).
#
# First step of the reference pipeline (load_video -> process -> isolate -> baseline -> compare). `load_reference(source, out_dir)` returns
# {"path": the file, "origin": "file", "title"}. Any audio or video file ffmpeg reads is accepted (checked with utils/convert_video_type.probe, which
# reads by content). The file is converted later, and only the part the student played (reference/process.py), so a long concert recording is not
# re-encoded whole. (Links, e.g. YouTube, are not supported for now: download the recording and give its file.)
from pathlib import Path

from maestro.utils.convert_video_type import probe
from maestro.utils.errors import InputError

MAX_SECONDS = 60 * 60            # longer recordings (a whole opera) are refused: pitch tracking them would take hours


def load_reference(source, out_dir=None):
    """A reference given as a file path -> {"path", "origin", "title"}; InputError if it cannot be used (a link, a missing or unreadable file, no
    audio, too long)."""
    source = str(source).strip()
    if "://" in source:
        raise InputError("Links are not supported: download the recording and give its file.")
    info = probe(source)
    if not info["has_audio"]:
        raise InputError("The reference has no audio track.")
    if info["duration"] > MAX_SECONDS:
        raise InputError(f"The reference is {info['duration'] / 60:.0f} minutes long; the limit is {MAX_SECONDS // 60}.")
    return {"path": source, "origin": "file", "title": Path(source).stem}
