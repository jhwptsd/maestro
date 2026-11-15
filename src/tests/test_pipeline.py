"""End to end: synthetic performance (video + MusicXML score) through maestro.analyze."""
import tempfile
import unittest
from pathlib import Path

from helpers import make_video, musicxml, synth, write_wav
from minuet import minuet_musicxml, play_minuet
from maestro import analyze
from maestro.utils.errors import InputError, MismatchError


class Pipeline(unittest.TestCase):
    def test_analyze_detects_sharp_and_late_notes(self):
        midis = [69, 71, 72, 74, 76, 74, 72, 71]
        # student plays at 120 bpm; note 3 is 30 cents sharp, note 5 is 120 ms late
        perf = [(m + (0.3 if i == 3 else 0), 0.3 + 0.5 * i + (0.12 if i == 5 else 0), 0.4) for i, m in enumerate(midis)]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "s.musicxml").write_text(musicxml(midis))
            write_wav(d / "a.wav", synth(perf))
            make_video(d / "v.mp4", d / "a.wav", 5)
            p = analyze(d / "v.mp4", d / "s.musicxml", d / "out")
            self.assertTrue((d / "out" / "score.svg").exists())

        inton = {n["score_idx"]: n["cents"] for n in p.assessment["intonation"]["notes"]}
        self.assertEqual(len(inton), len(midis))
        self.assertAlmostEqual(inton[3], 30, delta=8)
        self.assertLess(max(abs(c) for i, c in inton.items() if i != 3), 10)

        timing = {n["score_idx"]: n["error"] for n in p.assessment["rhythm"]["notes"]}
        self.assertEqual(max(timing, key=lambda k: abs(timing[k])), 5)
        self.assertAlmostEqual(p.assessment["rhythm"]["tempo_ratio"], 1.0, delta=0.1)

    def test_realistic_minuet_through_video_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "minuet.musicxml").write_text(minuet_musicxml())
            write_wav(d / "a.wav", play_minuet())
            make_video(d / "v.mp4", d / "a.wav", 18)          # AAC-compressed, like a real upload
            p = analyze(d / "v.mp4", d / "minuet.musicxml")
        self.assertGreater(min(p.assessment["correspondence"].values()), 0.8)
        self.assertLess(p.assessment["intonation"]["mean_abs_cents"], 15)
        self.assertEqual(p.assessment["intonation"]["n_wrong_notes"], 0)

    def test_recording_that_starts_halfway_down_the_page(self):
        """Student plays only bars 5-8 of the 8-bar Minuet; bars 1-4 stay unmatched and unassessed."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "minuet.musicxml").write_text(minuet_musicxml())
            write_wav(d / "a.wav", play_minuet(bars=slice(4, 8)))
            make_video(d / "v.mp4", d / "a.wav", 9)
            p = analyze(d / "v.mp4", d / "minuet.musicxml")
        sec = p.assessment["section"]
        self.assertEqual((sec["first_measure"], sec["last_measure"]), (5, 8))
        self.assertAlmostEqual(sec["coverage"], 0.5, delta=0.05)
        matched = [a["score_idx"] for a in p.alignment if a["perf_idx"] is not None]
        self.assertGreater(min(matched), 14)                      # bars 1-4 (16 notes) not matched
        self.assertGreater(min(p.assessment["correspondence"].values()), 0.8)
        self.assertLess(p.assessment["intonation"]["mean_abs_cents"], 15)
        self.assertLess(p.assessment["rhythm"]["mean_abs_error"], 0.08)   # start offset is not "late"

    def test_middle_section_and_report_card(self):
        from maestro.gui.report import build_report
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "minuet.musicxml").write_text(minuet_musicxml())
            write_wav(d / "a.wav", play_minuet(bars=slice(2, 6), seed=3))
            make_video(d / "v.mp4", d / "a.wav", 9)
            p = analyze(d / "v.mp4", d / "minuet.musicxml")
        self.assertEqual((p.assessment["section"]["first_measure"], p.assessment["section"]["last_measure"]), (3, 6))
        self.assertIn("bars 3–6", build_report(p))

    def test_wrong_score_for_recording_raises_mismatch(self):
        played = [69, 71, 72, 74, 76, 74, 72, 71]
        score = [60, 62, 64, 65, 55, 57, 59, 60]          # a different piece entirely
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "s.musicxml").write_text(musicxml(score))
            write_wav(d / "a.wav", synth([(m, 0.3 + 0.5 * i, 0.4) for i, m in enumerate(played)]))
            make_video(d / "v.mp4", d / "a.wav", 5)
            with self.assertRaises(MismatchError):
                analyze(d / "v.mp4", d / "s.musicxml")

    def test_midi_score_and_audio_only_upload(self):
        from helpers import write_midi
        midis = [69, 71, 72, 74, 76, 74, 72, 71]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_midi(d / "s.mid", [(i, i + 1, m) for i, m in enumerate(midis)])
            write_wav(d / "a.wav", synth([(m, 0.3 + 0.5 * i, 0.4) for i, m in enumerate(midis)]))
            p = analyze(d / "a.wav", d / "s.mid")
        self.assertEqual(len(p.assessment["intonation"]["notes"]), len(midis))
        self.assertLess(p.assessment["intonation"]["mean_abs_cents"], 10)

    def test_bad_inputs_raise_input_error(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "s.musicxml").write_text(musicxml([69]))
            with self.assertRaises(InputError):
                analyze(d / "nope.mp4", d / "s.musicxml")


if __name__ == "__main__":
    unittest.main()
