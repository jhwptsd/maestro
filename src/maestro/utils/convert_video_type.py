# Convert different video types to a common format for processing.
#
# Used by preprocessing/video.py and reference/load_video.py so that all downstream code
# (vision/, preprocessing/audio.py) sees one video format, frame rate, and resolution.
#
# Common format: MP4 (H.264 + AAC, falling back to MPEG-4 part 2 if ffmpeg lacks libx264), 30 fps,
# at most 720 px high, even dimensions. Requires the ffmpeg/ffprobe binaries on PATH; ffprobe also
# identifies files by *content*, so a wrong extension on a user upload does not matter.
import json
import subprocess
from functools import lru_cache
from pathlib import Path

from maestro.utils.errors import InputError


def probe(path):
    """Inspect a media file -> {"has_video", "has_audio", "duration" (s), "width", "height"}.

    Raises InputError if the file is missing or ffprobe cannot read it (not media / corrupt).
    """
    if not Path(path).is_file():
        raise InputError(f"File not found: {path}")
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
                        str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise InputError(f"Cannot read {Path(path).name} as audio or video.")
    info = json.loads(r.stdout)
    streams = info.get("streams", [])
    # cover art in audio files shows up as a "video" stream; ignore single-frame attached pictures
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    return {"has_video": video is not None,
            "has_audio": any(s.get("codec_type") == "audio" for s in streams),
            "duration": float(info.get("format", {}).get("duration") or 0.0),
            "width": video.get("width") if video else None,
            "height": video.get("height") if video else None}


@lru_cache(maxsize=1)
def _video_encoder():
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "libx264" if " libx264 " in out else "mpeg4"


def to_common_video(path, out_dir=None, fps=30, max_height=720):
    """Re-encode any video ffmpeg can read to the common format -> path of the new .mp4.

    The output is `<stem>_common.mp4` in `out_dir` (default: next to the input).
    """
    path = Path(path)
    out_dir = Path(out_dir) if out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_common.mp4"
    # -2 keeps the aspect ratio with an even width; min() never upscales small videos
    vf = f"fps={fps},scale=-2:'min({max_height},ih)',format=yuv420p"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vf", vf,
           "-c:v", _video_encoder(), "-c:a", "aac", "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise InputError(f"Could not convert {path.name}: {r.stderr.strip().splitlines()[-1] if r.stderr else 'ffmpeg failed'}")
    return out
