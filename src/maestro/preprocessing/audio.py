# Separate audio from input video and process it
#
# Extracts the audio track from the submitted recording (Performance.audio_source, set by
# preprocessing/video.py; falls back to Performance.video) into Performance.audio (mono, fixed
# sample rate). Consumed by audio/* for feature extraction.
#
# Uses the ffmpeg binary (must be on PATH) so any container/codec ffmpeg understands works; the
# result is decoded to a temporary mono 16-bit WAV and read back with the stdlib `wave` module.
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from maestro.utils.errors import InputError

DEFAULT_SR = 22050  # sample rate used throughout the audio pipeline


def extract_audio(video_path, sr=DEFAULT_SR):
    """Decode the audio track of `video_path` -> (samples float32 in [-1, 1], sr)."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        # -vn: drop video; -ac 1: mono; -ar: resample; pcm_s16le: plain 16-bit WAV
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
               "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(wav)]
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            raise InputError(f"Could not extract audio from {Path(video_path).name}.")
        return read_wav(wav)


def read_wav(path):
    """Read a 16-bit mono WAV -> (float32 samples, sample rate)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0, sr


def preprocess_audio(perf, sr=DEFAULT_SR):
    """Fill `perf.audio` = {"samples": ndarray, "sr": int} from the submitted recording."""
    samples, sr = extract_audio(perf.audio_source or perf.video, sr)
    if not np.any(np.abs(samples) > 1e-3):
        raise InputError("The recording is silent.")
    perf.audio = {"samples": samples, "sr": sr}
    return perf
