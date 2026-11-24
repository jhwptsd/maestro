# Load a professional reference recording from a file or a YouTube link (rewritten from the outline).
#
# First step of the reference pipeline (load_video -> process -> isolate -> baseline -> compare). `load_reference(source, out_dir)`
# returns {"path": a local media file, "origin": "file" | "youtube", "title", "url"}. Any audio or video file ffmpeg reads is accepted
# (checked with utils/convert_video_type.probe, which reads by content); only YouTube links are accepted as links, downloaded with yt-dlp
# (optional: `pip install yt-dlp`) at most 720 px high, one video, no playlists. The file is converted later, and only the part the student
# played (reference/process.py), so a long concert recording is not re-encoded whole.
# A downloaded recording is kept in the analysis folder for this comparison only; respect the copyright and terms of the source.
import re
from pathlib import Path

from maestro.utils.convert_video_type import probe
from maestro.utils.errors import InputError, ToolMissingError

MAX_SECONDS = 60 * 60            # longer recordings (a whole opera) are refused: pitch tracking them would take hours
YOUTUBE = re.compile(r"^https?://((www|m|music)\.)?(youtube\.com|youtu\.be)/", re.I)
FORMAT = "bv*[height<=720]+ba/b[height<=720]/b"      # best video up to 720 p with the best audio, else the best single file


def is_youtube(source):
    return bool(YOUTUBE.match(str(source).strip()))


def _download(url, out_dir):
    """Download a YouTube video into out_dir -> {"path", "title", "url"}."""
    try:
        import yt_dlp
    except ImportError:
        raise ToolMissingError("Reading a YouTube link needs yt-dlp: run  pip install yt-dlp  (or download the video yourself and "
                               "give its file instead).") from None
    out_dir.mkdir(parents=True, exist_ok=True)
    options = {"quiet": True, "no_warnings": True, "noplaylist": True, "format": FORMAT, "merge_output_format": "mp4",
               "outtmpl": str(out_dir / "reference.%(ext)s")}
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
            if (info.get("duration") or 0) > MAX_SECONDS:
                raise InputError(f"That video is {info['duration'] / 60:.0f} minutes long; the limit is {MAX_SECONDS // 60}.")
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise InputError(f"Could not download that YouTube video: {str(e).splitlines()[-1][:200]}") from None
    path = next(iter(sorted(out_dir.glob("reference.*"))), None)
    if path is None:
        raise InputError("The YouTube download produced no file.")
    return {"path": str(path), "title": info.get("title") or "YouTube video", "url": url}


def load_reference(source, out_dir):
    """A reference given as a file path or a YouTube link -> {"path", "origin", "title", "url"}; InputError if it cannot be used
    (missing or unreadable file, no audio, too long, a link that is not YouTube)."""
    source = str(source).strip()
    if re.match(r"^https?://", source, re.I):
        if not is_youtube(source):
            raise InputError("Only YouTube links are supported: download the recording and give its file instead.")
        found = {**_download(source, Path(out_dir)), "origin": "youtube"}
    else:
        found = {"path": source, "origin": "file", "title": Path(source).stem, "url": None}
    info = probe(found["path"])
    if not info["has_audio"]:
        raise InputError("The reference has no audio track.")
    if info["duration"] > MAX_SECONDS:
        raise InputError(f"The reference is {info['duration'] / 60:.0f} minutes long; the limit is {MAX_SECONDS // 60}.")
    return found
