import unittest

import numpy as np

from maestro.alignment.align_audio_score import _pair_leftovers, align_audio_score
from maestro.alignment.correspondence import check_correspondence, played_section
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


class PartialPerformance(unittest.TestCase):
    """The recording may start halfway down the page, stop early, or cover a middle section."""
    PIECE = Correspondence.PIECE                                  # 16 notes, 4 per measure below

    def score(self):
        s = score_notes(self.PIECE)
        for i, n in enumerate(s):
            n["measure"] = i // 4 + 1
        return s

    def align(self, first, last, **kw):
        played = self.PIECE[first:last]
        return align_audio_score(self.score(), perf_notes(played, np.arange(len(played))), **kw)

    def test_alignment_starts_and_ends_where_the_playing_does(self):
        for first, last in ((0, 8), (8, 16), (4, 12), (10, 16)):
            with self.subTest(first=first, last=last):
                matched = {a["score_idx"]: a["perf_idx"] for a in self.align(first, last) if a["perf_idx"] is not None}
                self.assertEqual(matched, {i: i - first for i in range(first, last)})   # only that stretch, one-to-one

    def test_forced_full_alignment_is_still_available(self):
        a = self.align(8, 16, partial=False)                       # start-to-start: second half stretched over the whole piece
        matched = [x["score_idx"] for x in a if x["perf_idx"] is not None]
        self.assertEqual((matched[0], matched[-1]), (0, 15))       # spans the whole score ...
        self.assertEqual(len(matched), 8)                          # ... but 8 played notes can explain only 8 written ones

    def test_missed_note_and_unexplained_note_in_one_gap_become_a_wrong_note(self):
        s = score_notes([74, 72, 74], spacing=0.5)                    # written D5 C5 D5
        p = perf_notes([74, 76, 74], [0.0, 0.5, 1.0])                 # played D5 E5 D5 (E5 for C5)
        out = [{"score_idx": 0, "perf_idx": 0, "cost": 0}, {"score_idx": 1, "perf_idx": None, "cost": None},
               {"score_idx": 2, "perf_idx": 2, "cost": 0}]            # what release-shared leaves behind
        _pair_leftovers(s, p, out)
        self.assertEqual([e["perf_idx"] for e in out], [0, 1, 2])
        self.assertAlmostEqual(out[1]["cost"], 4 / 12)
        # a chord note that simply was not heard (same onset as a matched note) is left alone
        s2 = score_notes([74, 67], spacing=0.0)
        out2 = [{"score_idx": 0, "perf_idx": 0, "cost": 0}, {"score_idx": 1, "perf_idx": None, "cost": None}]
        _pair_leftovers(s2, perf_notes([74, 76], [0.0, 0.01]), out2)
        self.assertIsNone(out2[1]["perf_idx"])

    def test_a_played_note_is_never_shared_between_written_notes(self):
        # written G4 G4 (repeated pitch) but only one G4 was played: exactly one of them is "missed"
        s = score_notes([74, 67, 67, 69], spacing=0.5)
        a = align_audio_score(s, perf_notes([74, 67, 69], [0.0, 0.5, 1.5]))
        self.assertEqual([x["perf_idx"] for x in a], [0, 1, None, 2])    # the G4 that fits the timing keeps it

    def test_played_section_reports_measures_and_coverage(self):
        s = self.score()
        sec = played_section(s, self.align(4, 12))
        self.assertEqual((sec["first_measure"], sec["last_measure"], sec["first_note"], sec["last_note"]), (2, 3, 4, 11))
        self.assertAlmostEqual(sec["coverage"], 0.5)
        self.assertIsNone(played_section(s, align_audio_score(s, [])))

    def test_partial_performance_passes_the_match_check_wrong_piece_does_not(self):
        s = self.score()
        for first, last in ((0, 8), (8, 16), (4, 12)):
            p = perf_notes(self.PIECE[first:last], np.arange(last - first))
            self.assertGreater(min(check_correspondence(s, p, align_audio_score(s, p)).values()), 0.9)
        wrong = perf_notes([67, 67, 74, 74, 76, 76, 74, 72], np.arange(8))
        with self.assertRaises(MismatchError):
            check_correspondence(s, wrong, align_audio_score(s, wrong))

    def test_assessments_cover_only_the_played_notes(self):
        s = self.score()
        p = perf_notes(self.PIECE[8:16], np.arange(8) * 2 + 10, cents=20)      # played 20 cents sharp, starts late
        a = align_audio_score(s, p)
        inton = assess_intonation(s, p, a)
        self.assertEqual([n["score_idx"] for n in inton["notes"]], list(range(8, 16)))
        self.assertAlmostEqual(inton["mean_abs_cents"], 20, delta=1)
        rhythm = assess_rhythm(s, p, a)                                       # start offset must not count as lateness
        self.assertLess(rhythm["mean_abs_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
