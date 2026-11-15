import unittest

import numpy as np

from maestro.alignment.align_audio_score import align_audio_score
from maestro.alignment.correspondence import check_correspondence
from maestro.alignment.dtw import dtw
from maestro.assessment.intonation import assess_intonation
from maestro.assessment.rhythm import assess_rhythm
from maestro.utils.audio_math import midi_to_hz
from maestro.utils.errors import InputError, MismatchError


def score_notes(midis, spacing=0.5):
    return [{"midi_pitch": m, "hz": float(midi_to_hz(m)), "onset": i * spacing} for i, m in enumerate(midis)]


def perf_notes(midis, onsets, cents=0.0):
    return [{"midi": m + cents / 100, "hz": float(midi_to_hz(m + cents / 100)), "onset": o} for m, o in zip(midis, onsets)]


class DTW(unittest.TestCase):
    def test_path_is_monotonic_and_follows_diagonal(self):
        path, total = dtw(np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]))
        self.assertEqual(path, [(0, 0), (1, 1), (2, 2), (3, 3)])
        self.assertEqual(total, 0)

    def test_stretched_sequence(self):
        a, b = np.array([1, 2, 3]), np.array([1, 1, 2, 2, 3])
        path, total = dtw(np.abs(a[:, None] - b[None, :]))
        self.assertEqual(total, 0)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (2, 4))


class Alignment(unittest.TestCase):
    def test_identity_and_repeated_pitches(self):
        midis = [69, 69, 71, 69, 69]
        a = align_audio_score(score_notes(midis), perf_notes(midis, np.arange(5)))
        self.assertEqual([x["perf_idx"] for x in a], [0, 1, 2, 3, 4])

    def test_double_stop_matches_heard_note_and_leaves_the_other_unmatched(self):
        # events: A4 | (G3+D4 chord) | B4 ; transcriber only heard the upper chord note
        s = score_notes([69, 55, 71], spacing=0.5)
        s.insert(2, {"midi_pitch": 62, "hz": 0.0, "onset": 0.5})   # D4 shares G3's onset
        a = align_audio_score(s, perf_notes([69, 62, 71], [0, 0.5, 1.0]))
        self.assertEqual([x["perf_idx"] for x in a], [0, None, 1, 2])   # G3 unheard, D4 matched

    def test_double_stop_rolled_into_two_performed_notes(self):
        s = [{"midi_pitch": 55, "hz": 0.0, "onset": 0.0}, {"midi_pitch": 62, "hz": 0.0, "onset": 0.0}]
        a = align_audio_score(s, perf_notes([55, 62], [0.0, 0.1]))
        self.assertEqual([x["perf_idx"] for x in a], [0, 1])

    def test_extra_performed_note_is_skipped(self):
        s = score_notes([69, 71, 72, 74])
        p = perf_notes([69, 71, 60, 72, 74], [0, 1, 2, 3, 4])   # stray note 60
        a = align_audio_score(s, p)
        self.assertEqual([x["perf_idx"] for x in a], [0, 1, 3, 4])


class Intonation(unittest.TestCase):
    def test_cents_and_wrong_notes(self):
        s = score_notes([69, 71, 72, 74])
        p = perf_notes([69, 71, 73, 74], range(4))               # third note a semitone off
        p[0] = perf_notes([69], [0], cents=+20)[0]
        p[1] = perf_notes([71], [1], cents=-10)[0]
        r = assess_intonation(s, p, align_audio_score(s, p))
        self.assertAlmostEqual(r["notes"][0]["cents"], 20, places=3)
        self.assertAlmostEqual(r["notes"][1]["cents"], -10, places=3)
        self.assertEqual(r["n_wrong_notes"], 1)
        self.assertAlmostEqual(r["mean_abs_cents"], 10, places=3)      # (20 + 10 + 0) / 3
        self.assertAlmostEqual(r["tuning_offset_cents"], 0, places=3)


class Rhythm(unittest.TestCase):
    def test_late_note_with_slower_tempo(self):
        midis = [69, 71, 72, 74, 76, 77]
        s = score_notes(midis, 0.5)
        onsets = np.arange(6) * 0.6                    # 20% slower than written
        onsets[3] += 0.1                               # one late note
        p = perf_notes(midis, onsets)
        r = assess_rhythm(s, p, align_audio_score(s, p))
        self.assertAlmostEqual(r["tempo_ratio"], 0.5 / 0.6, delta=0.05)
        errs = {n["score_idx"]: n["error"] for n in r["notes"]}
        self.assertEqual(max(errs, key=lambda k: errs[k]), 3)        # the late one is flagged
        self.assertGreater(errs[3], 0.05)

    def test_too_few_notes(self):
        s = score_notes([69])
        self.assertIsNone(assess_rhythm(s, perf_notes([69], [0]), align_audio_score(s, perf_notes([69], [0])))["tempo_ratio"])


class Correspondence(unittest.TestCase):
    PIECE = [69, 71, 72, 74, 76, 74, 72, 71, 69, 67, 69, 71, 72, 71, 69, 67]

    def check(self, played):
        s, p = score_notes(self.PIECE), perf_notes(played, range(len(played)))
        return check_correspondence(s, p, align_audio_score(s, p))

    def test_same_piece_passes_even_with_mistakes_and_partial_playing(self):
        wrong = [m + (3 if i in (3, 7, 11) else 0) for i, m in enumerate(self.PIECE)]
        self.assertGreater(self.check(wrong)["score_match"], 0.75)
        self.check(self.PIECE[:10])                                   # stopped early

    def test_different_pieces_raise(self):
        other_melody = [67, 67, 74, 74, 76, 76, 74, 72, 72, 71, 71, 69, 69, 67]   # same key
        scale = [60, 62, 64, 65, 67, 69, 71, 72] * 2
        for name, played in {"other melody": other_melody, "scale": scale,
                             "transposed": [m + 5 for m in self.PIECE]}.items():
            with self.subTest(name), self.assertRaises(MismatchError):
                self.check(played)

    def test_no_detected_notes_is_an_input_error(self):
        with self.assertRaises(InputError):
            check_correspondence(score_notes(self.PIECE), [], [])

    def test_chords_count_as_one_event(self):
        s = score_notes([69, 71])
        s.append({"midi_pitch": 62, "hz": 0.0, "onset": 0.5})        # D4 sounds together with B4 (a double stop)
        p = perf_notes([69, 71], [0, 0.5])
        self.assertEqual(check_correspondence(s, p, align_audio_score(s, p))["score_match"], 1.0)


if __name__ == "__main__":
    unittest.main()
