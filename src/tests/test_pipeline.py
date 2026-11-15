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
