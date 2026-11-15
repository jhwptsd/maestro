import tempfile
import unittest
import zipfile
from pathlib import Path

from helpers import SR, make_video, musicxml, synth, write_wav
from maestro.preprocessing.audio import extract_audio
from maestro.utils.convert_file_type import convert_to_musicxml


class AudioExtraction(unittest.TestCase):
    def test_extract_audio_from_video(self):
        with tempfile.TemporaryDirectory() as d:
            y = synth([(69, 0.1, 0.8)])
            write_wav(Path(d) / "a.wav", y)
            make_video(Path(d) / "v.mp4", Path(d) / "a.wav", 1.5)
            out, sr = extract_audio(Path(d) / "v.mp4")
        self.assertEqual(sr, SR)
        self.assertAlmostEqual(len(out) / sr, len(y) / SR, delta=0.15)
        self.assertGreater(abs(out).max(), 0.1)  # real signal, not silence


class ScoreConversion(unittest.TestCase):
    def test_musicxml_passthrough_and_mxl_unzip(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            xml = d / "s.musicxml"
            xml.write_text(musicxml([69]))
            self.assertEqual(convert_to_musicxml(xml), xml)

            mxl = d / "s.mxl"
            with zipfile.ZipFile(mxl, "w") as z:
                z.writestr("META-INF/container.xml",
                           '<container><rootfiles><rootfile full-path="inner.xml"/></rootfiles></container>')
                z.writestr("inner.xml", musicxml([69]))
            out = convert_to_musicxml(mxl, d / "out")
            self.assertEqual(out.read_text(), musicxml([69]))

    def test_unsupported_format(self):
        with self.assertRaises(ValueError):
            convert_to_musicxml("score.docx")


if __name__ == "__main__":
    unittest.main()
