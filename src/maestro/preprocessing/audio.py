# Separate audio from input video and process it
#
# Extracts the audio track from the submitted recording (Performance.audio_source, set by
# preprocessing/video.py; falls back to Performance.video) into Performance.audio (mono, fixed
# sample rate). Consumed by audio/* for feature extraction.
#
# Uses the ffmpeg binary (must be on PATH) so any container/codec ffmpeg understands works; the
# result is decoded to a temporary mono 16-bit WAV and read back with the stdlib `wave` module.
import math
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from maestro.preprocessing.denoise import denoise_audio
from maestro.utils.errors import InputError
from maestro.utils.ffmpeg import available, tool

DEFAULT_SR = 22050  # sample rate used throughout the audio pipeline


def native_wav(path, sr=DEFAULT_SR):
    """A PCM WAV (8, 16, 24 or 32 bit, any channels and rate) read without ffmpeg -> (mono float32 samples, sr). What a phone records with
    flet's AudioRecorder (WAV) is analysed this way, since a phone has no ffmpeg. InputError if it is not a PCM WAV."""
    try:
        with wave.open(str(path), "rb") as w:
            rate, channels, width, raw = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        raise InputError(f"{Path(path).name} is not a PCM WAV file, and ffmpeg (needed for other formats) is not installed.") from None
    if width == 3:                                               # 24 bit: pad each sample to 32 bit
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        pcm = (np.concatenate([np.zeros((len(b), 1), np.uint8), b], axis=1).view("<i4")[:, 0] / 2 ** 31).astype(np.float32)
    elif width in (1, 2, 4):
        dtype = {1: "u1", 2: "<i2", 4: "<i4"}[width]
        pcm = np.frombuffer(raw, dtype).astype(np.float32)
        pcm = (pcm - 128) / 128 if width == 1 else pcm / 2 ** (8 * width - 1)
    else:
        raise InputError("Unsupported WAV sample size.")
    y = pcm[:len(pcm) // channels * channels].reshape(-1, channels).mean(axis=1)
    if rate != sr:
        g = math.gcd(rate, sr)
        y = resample_poly(y, sr // g, rate // g).astype(np.float32)
    return y, sr


def extract_audio(video_path, sr=DEFAULT_SR):
    """Decode the audio track of `video_path` -> (samples float32 in [-1, 1], sr). Without ffmpeg only PCM WAV files can be read."""
    if not available("ffmpeg"):
        return native_wav(video_path, sr)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        # -vn: drop video; -ac 1: mono; -ar: resample; pcm_s16le: plain 16-bit WAV
        cmd = [tool("ffmpeg"), "-y", "-loglevel", "error", "-i", str(video_path),
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


def preprocess_audio(perf, sr=DEFAULT_SR, denoise=True):
    """Fill `perf.audio` = {"samples": ndarray, "sr": int} from the submitted recording.

    denoise: remove stationary background noise first (preprocessing/denoise.py); vibrato, bow noise and
    tonal content are preserved. The noise estimate is kept in perf.preprocessing_report["audio"].
    """
    samples, sr = extract_audio(perf.audio_source or perf.video, sr)
    if not np.any(np.abs(samples) > 1e-3):
        raise InputError("The recording is silent.")
    if denoise:
        samples, perf.preprocessing_report["audio"] = denoise_audio(samples, sr)
    perf.audio = {"samples": samples, "sr": sr}
    return perf
