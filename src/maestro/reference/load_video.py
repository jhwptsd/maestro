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
from maestro.utils.ffmpeg import tool

MAX_SECONDS = 60 * 60            # longer recordings (a whole opera) are refused: pitch tracking them would take hours
YOUTUBE = re.compile(r"^https?://((www|m|music)\.)?(youtube\.com|youtu\.be)/", re.I)
# Two tries. First the best video up to 720 p merged with the best audio, preferring mp4/m4a streams and merged into mkv (which takes any codecs:
# merging VP9 video with Opus audio into mp4, as yt-dlp does by default, is what fails with "Postprocessing: ... (copy)" on some ffmpeg versions).
# If merging still fails, one ready-made file with video and audio, which needs no ffmpeg step at all.
ATTEMPTS = (("bv*[height<=720][ext=mp4]+ba[ext=m4a]/bv*[height<=720]+ba/b[height<=720]/bv*+ba/b", "mkv"), ("b[height<=720]/b", None))


def is_youtube(source):
    return bool(YOUTUBE.match(str(source).strip()))


def _why(error):
    """yt-dlp's own message, cleaned, plus what usually fixes the common failures."""
    lines = re.sub(r"\x1b\[[0-9;]*m", "", str(error)).strip().splitlines() or ["unknown error"]
    text = next((l for l in lines if "ERROR" in l), lines[0]).split("ERROR:", 1)[-1].strip()[:300]
    hint = ("YouTube changes often, so first update the downloader: pip install -U yt-dlp. "
            + ("This video asks for a sign-in (age or bot check) and cannot be fetched: download it yourself and give its file. "
               if re.search(r"sign in|confirm you.re not a bot|age", text, re.I) else "")
            + ("A JavaScript runtime is needed for this video: pip install -U \"yt-dlp[default]\" and install Node.js. "
               if re.search(r"javascript|js runtime|n challenge|signature", text, re.I) else "")
            + "Or download the video yourself and give its file.")
    return f"Could not download that YouTube video ({text}). {hint}"


def _download(url, out_dir):
    """Download a YouTube video into out_dir -> {"path", "title", "url"}."""
    try:
        import yt_dlp
    except ImportError:
        raise ToolMissingError("Reading a YouTube link needs yt-dlp: run  pip install yt-dlp  (or download the video yourself and "
                               "give its file instead).") from None
    out_dir.mkdir(parents=True, exist_ok=True)
    base = {"quiet": True, "no_warnings": True, "noplaylist": True, "outtmpl": str(out_dir / "reference.%(ext)s"),
            "ffmpeg_location": str(Path(tool("ffmpeg")).parent),            # merging video and audio needs ffmpeg, which may not be on PATH (Windows)
            "match_filter": yt_dlp.utils.match_filter_func(f"duration < {MAX_SECONDS}")}
    for k, (format_, container) in enumerate(ATTEMPTS):
        options = {**base, "format": format_, **({"merge_output_format": container} if container else {})}
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)                # (one extraction: a second one is slower and can fail on its own)
            break
        except (yt_dlp.utils.YoutubeDLError, OSError) as e:
            if k + 1 == len(ATTEMPTS) or not re.search(r"postprocessing|ffmpeg|merg", str(e), re.I):
                raise InputError(_why(e)) from None
            for leftover in out_dir.glob("reference.*"):                    # a half-merged file must not be mistaken for the download
                leftover.unlink(missing_ok=True)
    downloads = (info or {}).get("requested_downloads") or []
    path = next((d["filepath"] for d in downloads if d.get("filepath") and Path(d["filepath"]).is_file()), None)
    path = path or next(iter(sorted(out_dir.glob("reference.*"))), None)
    if path is None:
        raise InputError(f"That video was not downloaded: it may be longer than {MAX_SECONDS // 60} minutes, or unavailable to download.")
    return {"path": str(path), "title": (info or {}).get("title") or "YouTube video", "url": url}


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
