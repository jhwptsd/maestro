# Process input video (student or professional)
#
# Validates a user-submitted recording and converts it with utils/convert_video_type.py to the common
# format, stored as a path in Performance.video. Consumed by vision/* and preprocessing/audio.py
# (which reads the *original* upload via Performance.audio_source to avoid a second lossy encode).
# Also reused for reference videos (reference/process.py).
#
# Audio-only uploads (wav/mp3/m4a/...) are accepted: video stays None and vision/ is skipped.
from pathlib import Path

from maestro.preprocessing.denoise import video_filters
from maestro.utils.convert_video_type import probe, to_common_video
from maestro.utils.errors import InputError

MIN_SECONDS = 2.0       # shorter than this cannot contain a meaningful performance
MAX_SECONDS = 30 * 60   # keeps processing time bounded


def preprocess_video(perf, path, out_dir=None, denoise=True):
    """Validate `path` and fill perf.audio_source / perf.video. Raises InputError if unusable.

    denoise: also apply the mild video denoising filter (preprocessing/denoise.py) while converting.
    """
    info = probe(path)  # raises InputError for missing / non-media / corrupt files
    if not info["has_audio"]:
        raise InputError("The submitted file has no audio track, so intonation and rhythm cannot be assessed.")
    if not MIN_SECONDS <= info["duration"] <= MAX_SECONDS:
        raise InputError(f"Recording is {info['duration']:.1f}s long; it must be between "
                         f"{MIN_SECONDS:.0f}s and {MAX_SECONDS // 60} minutes.")
    perf.audio_source = str(path)
    perf.video = (str(to_common_video(path, out_dir, filters=video_filters() if denoise else ()))
                  if info["has_video"] else None)
    perf.preprocessing_report["video"] = bool(denoise and info["has_video"])
    return perf
