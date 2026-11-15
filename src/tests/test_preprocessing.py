import os
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile
from pathlib import Path

from helpers import SR, make_video, musicxml, synth, write_midi, write_wav
from maestro.preprocessing.audio import extract_audio, preprocess_audio
from maestro.preprocessing.score import preprocess_score
from maestro.preprocessing.video import preprocess_video
from maestro.score.musicxml import parse_musicxml
from maestro.utils.convert_file_type import MAX_PDF_PAGES, convert_to_musicxml, detect_score_type
from maestro.utils.convert_video_type import probe
from maestro.utils.errors import InputError, OMRUnavailableError
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


PDF = b"%PDF-1.4\n1 0 obj<</Type/Page>>endobj\ntrailer<<>>\n%%EOF\n"
# stands in for Audiveris: finds the -output dir and input file in argv, writes <out>/<stem>/<stem>.mxl
FAKE_AUDIVERIS = """#!{python}
import os, sys, zipfile
a = sys.argv; out = a[a.index("-output") + 1]; stem = os.path.splitext(os.path.basename(a[-1]))[0]
if os.environ.get("FAKE_OMR_FAIL"):
    sys.exit("Sheet is blank")
os.makedirs(os.path.join(out, stem), exist_ok=True)
with zipfile.ZipFile(os.path.join(out, stem, stem + ".mxl"), "w") as z:
    z.writestr("META-INF/container.xml", '<container><rootfiles><rootfile full-path="s.xml"/></rootfiles></container>')
    z.writestr("s.xml", open(os.environ["FAKE_OMR_XML"]).read())
"""


class PdfScores(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.d / "s.pdf").write_bytes(PDF)
        (self.d / "truth.musicxml").write_text(musicxml([69, 71, 72]))

    def fake_audiveris(self, **env):
        """Put a fake `audiveris` first on PATH for the duration of the test."""
        bin_dir = self.d / "bin"
        bin_dir.mkdir(exist_ok=True)
        exe = bin_dir / "audiveris"
        exe.write_text(FAKE_AUDIVERIS.format(python=sys.executable))
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        patcher = mock.patch.dict(os.environ, {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                                               "FAKE_OMR_XML": str(self.d / "truth.musicxml"), **env})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pdf_goes_through_omr_to_a_parsed_score(self):
        self.fake_audiveris()
        perf = preprocess_score(Performance(), self.d / "s.pdf", self.d / "out")
        self.assertEqual([n.midi_pitch for n in perf.score.notes], [69, 71, 72])

    def test_image_and_misnamed_pdf_use_the_same_route(self):
        self.fake_audiveris()
        (self.d / "scan.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 20)
        (self.d / "really_a_pdf.png").write_bytes(PDF)
        for name in ("scan.png", "really_a_pdf.png"):
            with self.subTest(name):
                self.assertEqual(len(preprocess_score(Performance(), self.d / name, self.d / "out").score.notes), 3)

    def test_missing_audiveris_gives_actionable_error(self):
        with mock.patch.dict(os.environ, {"PATH": str(self.d)}):
            with self.assertRaisesRegex(OMRUnavailableError, "MusicXML"):
                convert_to_musicxml(self.d / "s.pdf", self.d / "out")

    def test_omr_failure_is_reported_as_input_error(self):
        self.fake_audiveris(FAKE_OMR_FAIL="1")
        with self.assertRaisesRegex(InputError, "blank"):
            convert_to_musicxml(self.d / "s.pdf", self.d / "out")

    def test_bad_pdfs_are_rejected_before_omr(self):
        pages = b"%PDF-1.4\n" + b"<</Type/Page>>\n" * (MAX_PDF_PAGES + 1) + b"%%EOF"
        cases = {"truncated": PDF[:-12], "encrypted": PDF.replace(b"trailer<<>>", b"trailer<</Encrypt 5 0 R>>"),
                 "too many pages": pages}
        for name, data in cases.items():
            (self.d / f"{name}.pdf".replace(" ", "_")).write_bytes(data)
            with self.subTest(name), self.assertRaises(InputError):   # no Audiveris needed to get here
                convert_to_musicxml(self.d / f"{name}.pdf".replace(" ", "_"), self.d / "out")
        # "/Type /Pages" (the page tree node) must not be counted as a page
        (self.d / "tree.pdf").write_bytes(b"%PDF-1.4\n<</Type/Pages>>\n" * 100 + b"%%EOF")
        with mock.patch.dict(os.environ, {"PATH": str(self.d)}), self.assertRaises(OMRUnavailableError):
            convert_to_musicxml(self.d / "tree.pdf", self.d / "out")


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
