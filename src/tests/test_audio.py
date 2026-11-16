import unittest
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np

from helpers import SR, synth
from minuet import MIDIS, play_minuet
from maestro.alignment.align_audio_score import align_audio_score
from maestro.alignment.correspondence import check_correspondence
from maestro.audio.features import extract_features
from maestro.audio.onset_detection import detect_onsets
from maestro.audio.pitch import (
    DEFAULT_PITCH_BACKEND,
    estimate_f0,
    estimate_pitch,
    list_pitch_backends,
)
from maestro.audio.transcription import transcribe
from maestro.utils.audio_math import midi_to_hz


class Pitch(unittest.TestCase):
    def test_default_backend_is_crepe(self):
        self.assertEqual(DEFAULT_PITCH_BACKEND, "crepe")

    def test_all_backends_are_registered(self):
        self.assertEqual(list_pitch_backends(), ["yin", "pyin", "crepe", "basic_pitch"])

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown pitch backend"):
            estimate_pitch(synth([(69, 0.0, 0.3)]), SR, backend="not-a-backend")

    def test_yin_f0_within_a_few_cents(self):
        for midi in (55, 69, 81, 93):
            times, f0, confidence = estimate_f0(
                synth([(midi, 0, 1.0)], total=1.0), SR, backend="yin"
            )
            middle = slice(len(times) // 4, 3 * len(times) // 4)
            expected = 440 * 2 ** ((midi - 69) / 12)
            self.assertLess(abs(1200 * np.log2(np.median(f0[middle]) / expected)), 5, midi)
            self.assertGreater(np.median(confidence[middle]), 0.9)

    def test_silence_has_zero_yin_confidence(self):
        _, _, confidence = estimate_f0(np.zeros(SR), SR, backend="yin")
        self.assertEqual(confidence.max(), 0)

    def test_pitch_track_keeps_legacy_tuple_unpacking(self):
        track = estimate_pitch(synth([(69, 0.0, 0.5)]), SR, backend="yin")
        times, f0, confidence = track
        self.assertIs(times, track.times)
        self.assertIs(f0, track.f0)
        self.assertIs(confidence, track.confidence)

    def test_basic_pitch_retains_polyphonic_note_events(self):
        basic_pitch = ModuleType("basic_pitch")
        basic_pitch.__path__ = []
        basic_pitch.ICASSP_2022_MODEL_PATH = "mock-model"
        inference = ModuleType("basic_pitch.inference")
        inference.predict = lambda *args, **kwargs: (
            {}, None,
            [(0.1, 0.5, 69, 0.95, None), (0.1, 0.5, 76, 0.85, None)],
        )
        audio = synth([(69, 0.0, 0.7)], total=0.8)
        with patch.dict(sys.modules, {
            "basic_pitch": basic_pitch,
            "basic_pitch.inference": inference,
        }):
            track = estimate_pitch(audio, SR, backend="basic_pitch")
        self.assertEqual(len(track.notes), 2)
        events = transcribe(audio, SR, pitch_track=track)
        self.assertEqual([round(note["midi"]) for note in events], [69, 76])


class Onsets(unittest.TestCase):
    def test_repeated_pitch_onsets(self):
        y = synth([(69, 0.2, 0.3), (69, 0.6, 0.3), (69, 1.0, 0.3)])
        onsets, _ = detect_onsets(y, SR)
        self.assertEqual(len(onsets), 3)
        np.testing.assert_allclose(onsets, [0.2, 0.6, 1.0], atol=0.04)


class RealisticPlaying(unittest.TestCase):
    """Synthetic regression fixture for vibrato, noise, bowing, reverb and pitch glides."""
    SCORE = [{"midi_pitch": m, "hz": float(midi_to_hz(m)), "onset": i * 0.5} for i, m in enumerate(MIDIS)]

    def test_vibrato_does_not_create_onsets(self):
        t = np.arange(int(2.3 * SR)) / SR
        vib = 69 + 0.4 * np.sin(2 * np.pi * 5.5 * t)
        y = (np.sin(np.cumsum(440 * 2 ** ((vib - 69) / 12)) / SR * 2 * np.pi)
             * np.minimum(1, t / 0.03) * 0.5).astype(np.float32)
        self.assertEqual(len(detect_onsets(y, SR)[0]), 1)

    def test_minuet_is_recognised_under_realistic_conditions(self):
        conditions = {
            "vibrato+noise+reverb": {},
            "fast": dict(bpm=130, seed=1),
            "slow, wide vibrato": dict(bpm=70, vibrato_cents=45, seed=2),
            "40 cents flat": dict(tuning_cents=-40, seed=3),
            "noisy": dict(noise=0.04, seed=4),
        }
        for name, kwargs in conditions.items():
            with self.subTest(name=name):
                notes = transcribe(play_minuet(**kwargs), SR, backend="yin")
                self.assertLessEqual(abs(len(notes) - len(MIDIS)), 4)
                matches = check_correspondence(self.SCORE, notes, align_audio_score(self.SCORE, notes))
                self.assertGreater(min(matches.values()), 0.8)

    def test_minuet_pitch_is_accurate(self):
        notes = transcribe(play_minuet(), SR, backend="yin")
        matches = align_audio_score(self.SCORE, notes)
        cents = [
            1200 * np.log2(notes[item["perf_idx"]]["hz"] / score["hz"])
            for item, score in zip(matches, self.SCORE)
            if item["perf_idx"] is not None
        ]
        self.assertGreaterEqual(len(cents), len(MIDIS) - 3)
        self.assertLess(np.mean(np.abs(cents)), 12)


class Transcription(unittest.TestCase):
    def test_note_sequence(self):
        spec = [(69, 0.2, 0.4), (71, 0.7, 0.4), (71, 1.2, 0.4), (76, 1.7, 0.4)]
        notes = transcribe(synth(spec), SR, backend="yin")
        self.assertEqual([round(note["midi"]) for note in notes], [69, 71, 71, 76])
        np.testing.assert_allclose([note["onset"] for note in notes], [item[1] for item in spec], atol=0.04)

    def test_detects_cents_offset(self):
        notes = transcribe(synth([(69.2, 0.2, 0.5)]), SR, backend="yin")
        self.assertEqual(len(notes), 1)
        self.assertAlmostEqual(notes[0]["midi"], 69.2, delta=0.03)


class Features(unittest.TestCase):
    def test_schema_and_values(self):
        features = extract_features(synth([(69, 0.1, 0.8)]), SR, backend="yin")
        self.assertEqual(
            set(features[0]),
            {"time", "pitch_confidence", "rms_energy", "spectral_centroid",
             "onset_probability", "estimated_pitch", "estimated_tempo"},
        )
        middle = features[len(features) // 2]
        self.assertAlmostEqual(middle["estimated_pitch"], 440, delta=3)

    def test_tempo(self):
        beats = [(69, 0.25 + 0.5 * k, 0.3) for k in range(12)]
        self.assertAlmostEqual(
            extract_features(synth(beats), SR, backend="yin")[0]["estimated_tempo"], 120, delta=6
        )


if __name__ == "__main__":
    unittest.main()
