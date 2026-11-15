import unittest

import numpy as np

from helpers import SR, synth
from maestro.audio.features import extract_features
from maestro.audio.onset_detection import detect_onsets
from maestro.audio.pitch import estimate_f0
from maestro.audio.transcription import transcribe


class Pitch(unittest.TestCase):
    def test_f0_within_a_few_cents(self):
        for midi in (55, 69, 81, 93):                      # G3 ... A6
            t, f0, conf = estimate_f0(synth([(midi, 0, 1.0)], total=1.0), SR)
            mid = slice(len(t) // 4, 3 * len(t) // 4)
            expected = 440 * 2 ** ((midi - 69) / 12)
            self.assertLess(abs(1200 * np.log2(np.median(f0[mid]) / expected)), 5, midi)
            self.assertGreater(np.median(conf[mid]), 0.9)

    def test_silence_has_zero_confidence(self):
        _, _, conf = estimate_f0(np.zeros(SR), SR)
        self.assertEqual(conf.max(), 0)


class Onsets(unittest.TestCase):
    def test_repeated_pitch_onsets(self):
        y = synth([(69, 0.2, 0.3), (69, 0.6, 0.3), (69, 1.0, 0.3)])
        onsets, _ = detect_onsets(y, SR)
        self.assertEqual(len(onsets), 3)
        np.testing.assert_allclose(onsets, [0.2, 0.6, 1.0], atol=0.04)


class Transcription(unittest.TestCase):
    def test_note_sequence(self):
        spec = [(69, 0.2, 0.4), (71, 0.7, 0.4), (71, 1.2, 0.4), (76, 1.7, 0.4)]  # includes repeated pitch
        notes = transcribe(synth(spec), SR)
        self.assertEqual([round(n["midi"]) for n in notes], [69, 71, 71, 76])
        np.testing.assert_allclose([n["onset"] for n in notes], [s[1] for s in spec], atol=0.04)

    def test_detects_cents_offset(self):
        notes = transcribe(synth([(69.2, 0.2, 0.5)]), SR)   # 20 cents sharp
        self.assertEqual(len(notes), 1)
        self.assertAlmostEqual(notes[0]["midi"], 69.2, delta=0.03)


class Features(unittest.TestCase):
    def test_schema_and_values(self):
        f = extract_features(synth([(69, 0.1, 0.8)]), SR)
        self.assertEqual(set(f[0]), {"time", "pitch_confidence", "rms_energy", "spectral_centroid",
                                     "onset_probability", "estimated_pitch", "estimated_tempo"})
        mid = f[len(f) // 2]
        self.assertAlmostEqual(mid["estimated_pitch"], 440, delta=3)

    def test_tempo(self):
        beats = [(69, 0.25 + 0.5 * k, 0.3) for k in range(12)]     # 120 BPM
        self.assertAlmostEqual(extract_features(synth(beats), SR)[0]["estimated_tempo"], 120, delta=6)


if __name__ == "__main__":
    unittest.main()
