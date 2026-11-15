import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from helpers import SR, make_video, musicxml, synth, write_midi, write_wav
from maestro.preprocessing.audio import extract_audio, preprocess_audio
from maestro.preprocessing.score import preprocess_score
from maestro.preprocessing.video import preprocess_video
from maestro.score.musicxml import parse_musicxml
from maestro.utils.convert_file_type import convert_to_musicxml, detect_score_type
from maestro.utils.convert_video_type import probe
from maestro.utils.errors import InputError
from maestro.utils.Performance import Performance


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
        with tempfile.TemporaryDirectory() as d:
            junk = Path(d) / "score.docx"
            junk.write_bytes(b"\x00\x01 not a score")
            with self.assertRaises(InputError):
                convert_to_musicxml(junk)
            with self.assertRaises(InputError):
                convert_to_musicxml(Path(d) / "missing.xml")

    def test_type_is_detected_by_content_not_extension(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a.pdf").write_text(musicxml([69]))               # MusicXML misnamed .pdf
            self.assertEqual(detect_score_type(d / "a.pdf"), "musicxml")
            (d / "b.xml").write_bytes(b"%PDF-1.4 ...")             # PDF misnamed .xml
            self.assertEqual(detect_score_type(d / "b.xml"), "pdf")
            (d / "c.txt").write_bytes(b"\xef\xbb\xbf\n<?xml version='1.0'?><score-partwise/>")
            self.assertEqual(detect_score_type(d / "c.txt"), "musicxml")   # BOM + whitespace
            (d / "empty.xml").write_bytes(b"")
            with self.assertRaises(InputError):
                detect_score_type(d / "empty.xml")

    def test_midi_converts_to_musicxml(self):
        # 120 bpm: A4, B4 quarters, then a half-note chord (C4+E4) that crosses the barline into bar 2
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_midi(d / "s.mid", [(0, 1, 69), (1, 2, 71), (3, 5, 60), (3, 5, 64)])
            xml = convert_to_musicxml(d / "s.mid", d / "out")
            s = parse_musicxml(xml)
        self.assertEqual([(n.midi_pitch, n.onset) for n in s.notes],
                         [(69, 0.0), (71, 0.5), (60, 1.5), (64, 1.5)])
        self.assertEqual(s.notes[2].duration, 2.0)                  # tie across the barline merged
        self.assertEqual(s.tempo, 120.0)
        self.assertEqual(s.notes[0].measure, 1)

    def test_corrupt_midi(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.mid"
            bad.write_bytes(b"MThd" + b"\x00" * 10)
            with self.assertRaises(InputError):
                convert_to_musicxml(bad)

    def test_preprocess_score_rejects_bad_scores(self):
        cases = {"timewise": "<score-timewise/>", "garbage": "<score-partwise><unclosed>",
                 "no notes": "<score-partwise><part id='P1'><measure number='1'/></part></score-partwise>"}
        with tempfile.TemporaryDirectory() as d:
            for name, text in cases.items():
                p = Path(d) / f"{name}.musicxml".replace(" ", "_")
                p.write_text(text)
                with self.subTest(name), self.assertRaises(InputError):
                    preprocess_score(Performance(), p)
            ok = Path(d) / "ok.musicxml"
            ok.write_text(musicxml([69, 71]))
            self.assertEqual(len(preprocess_score(Performance(), ok).score.notes), 2)


class VideoSubmission(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)
        write_wav(self.d / "a.wav", synth([(69, 0.1, 2.0)]))
        self.addCleanup(self._tmp.cleanup)

    def test_large_video_is_converted_to_common_format(self):
        make_video(self.d / "big.mov", self.d / "a.wav", 2.5, size="1280x960")
        perf = preprocess_video(Performance(), self.d / "big.mov", self.d / "out")
        info = probe(perf.video)
        self.assertTrue(info["has_video"] and info["has_audio"])
        self.assertEqual((info["width"], info["height"]), (960, 720))   # scaled down, aspect kept
        self.assertTrue(perf.video.endswith("_common.mp4"))
        preprocess_audio(perf)
        self.assertGreater(len(perf.audio["samples"]), SR)             # audio comes from the upload

    def test_audio_only_upload_is_accepted(self):
        perf = preprocess_video(Performance(), self.d / "a.wav", self.d / "out")
        self.assertIsNone(perf.video)
        self.assertGreater(len(preprocess_audio(perf).audio["samples"]), SR)

    def test_rejects_unusable_files(self):
        (self.d / "notes.mp4").write_text("this is not a video")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=3",
                        "-c:v", "mpeg4", str(self.d / "silent.mp4")], check=True)   # video, no audio track
        write_wav(self.d / "short.wav", synth([(69, 0.0, 0.5)]))
        for name in ("notes.mp4", "silent.mp4", "short.wav", "missing.mp4"):
            with self.subTest(name), self.assertRaises(InputError):
                preprocess_video(Performance(), self.d / name, self.d / "out")

    def test_silent_audio_is_rejected(self):
        write_wav(self.d / "quiet.wav", synth([(69, 0.0, 3.0)]) * 0)
        perf = preprocess_video(Performance(), self.d / "quiet.wav", self.d / "out")
        with self.assertRaises(InputError):
            preprocess_audio(perf)


if __name__ == "__main__":
    unittest.main()
