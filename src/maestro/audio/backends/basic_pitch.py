"""Spotify Basic Pitch backend for note-event and polyphonic transcription."""
import tempfile
import wave
from pathlib import Path

import numpy as np

from maestro.audio.backends.base import PitchTrack, frame_times
from maestro.utils.audio_math import midi_to_hz


def _write_wav(path, y, sr):
    samples = np.asarray(y, dtype=np.float32).reshape(-1)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(int(sr))
        output.writeframes(pcm.tobytes())


def estimate(y, sr, frame_length=1024, hop_length=256, fmin=None, fmax=None,
             onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=60.0,
             multiple_pitch_bends=True, melodia_trick=True, **_):
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise ImportError(
            "The Basic Pitch backend requires Spotify basic-pitch. "
            "Install it with: pip install -e '.[basic-pitch]'"
        ) from exc

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    fmin = 175.0 if fmin is None else float(fmin)
    fmax = 4200.0 if fmax is None else float(fmax)
    if not 0 < fmin < fmax <= sr / 2:
        raise ValueError("Basic Pitch requires 0 < fmin < fmax <= Nyquist.")

    if not y.size:
        empty = np.zeros(0, dtype=float)
        return PitchTrack(empty, empty, empty, "basic_pitch", notes=[])

    with tempfile.TemporaryDirectory(prefix="maestro_basic_pitch_") as tmp:
        audio_path = Path(tmp) / "input.wav"
        _write_wav(audio_path, y, sr)
        _, _, raw_events = predict(
            audio_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            minimum_frequency=fmin,
            maximum_frequency=fmax,
            multiple_pitch_bends=multiple_pitch_bends,
            melodia_trick=melodia_trick,
        )

    notes = []
    for event in raw_events:
        start, end, midi, amplitude, pitch_bends = event
        midi = float(midi)
        notes.append({
            "onset": float(start),
            "offset": float(end),
            "hz": float(midi_to_hz(midi)),
            "midi": midi,
            "confidence": float(np.clip(amplitude, 0.0, 1.0)),
            "pitch_bends": pitch_bends,
        })
    notes.sort(key=lambda note: (note["onset"], note["midi"]))

    # Project the loudest active note onto frames for plots/features. Full polyphony
    # is retained in PitchTrack.notes and consumed directly by transcribe().
    times = frame_times(len(y), sr, hop_length)
    f0 = np.zeros(len(times), dtype=float)
    confidence = np.zeros(len(times), dtype=float)
    for note in notes:
        active = (times >= note["onset"]) & (times <= note["offset"])
        replace = active & (note["confidence"] >= confidence)
        f0[replace] = note["hz"]
        confidence[replace] = note["confidence"]

    return PitchTrack(times, f0, confidence, "basic_pitch", notes=notes)
