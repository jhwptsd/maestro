"""End to end: synthetic performance (video + MusicXML score) through maestro.analyze."""
import tempfile
import unittest
from pathlib import Path

from helpers import make_video, musicxml, synth, write_wav
from maestro import analyze


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


if __name__ == "__main__":
    unittest.main()
