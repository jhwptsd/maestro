"""What still works on a phone: no ffmpeg (a WAV recording is analysed with plain Python), the YIN tracker, the page-scan cleaner, and the capability check
that tells the app what to offer."""
import io
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from helpers import SR, musicxml, synth, write_wav
from minuet import MIDIS
from maestro import analyze
from maestro.preprocessing.audio import native_wav
from maestro.score import scan
from maestro.utils import capabilities, ffmpeg
from maestro.utils.convert_file_type import check_pdf, detect_score_type
from maestro.utils.convert_video_type import probe
from maestro.utils.errors import InputError, ToolMissingError


def no_ffmpeg():
    """Pretend ffmpeg and ffprobe are not installed (they are not on a phone)."""
    return mock.patch.object(ffmpeg, "_found", return_value=None)


def write_pcm(path, y, rate, width=2, channels=1):
    data = np.clip(y, -1, 1)
    if width == 3:
        ints = (data * (2 ** 23 - 1)).astype("<i4")
        raw = b"".join(int(v).to_bytes(4, "little", signed=True)[:3] for v in np.repeat(ints, channels))
    else:
        raw = (data * (2 ** (8 * width - 1) - 1)).astype({1: "u1", 2: "<i2", 4: "<i4"}[width]).tobytes() if width != 1 else ((data * 127 + 128).astype("u1")).tobytes()
        if channels > 1:
            raw = np.repeat(np.frombuffer(raw, {1: "u1", 2: "<i2", 4: "<i4"}[width]), channels).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels), w.setsampwidth(width), w.setframerate(rate)
        w.writeframes(raw)


class WithoutFfmpeg(unittest.TestCase):
    def test_wav_files_of_any_common_kind_are_read_and_resampled(self):
        tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            for width in (1, 2, 3, 4):
                for channels in (1, 2):
                    path = Path(tmp) / f"t{width}{channels}.wav"
                    write_pcm(path, tone, 44100, width, channels)
                    y, sr = native_wav(path)
                    self.assertEqual(sr, SR)
                    self.assertAlmostEqual(len(y) / sr, 1.0, places=2)
                    self.assertLess(np.abs(y - np.interp(np.arange(len(y)) / SR, np.arange(44100) / 44100, tone)).max(), 0.05, (width, channels))
            (Path(tmp) / "x.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100)
            with self.assertRaisesRegex(InputError, "not a PCM WAV"):
                native_wav(Path(tmp) / "x.mp4")

    def test_probe_reads_a_wav_and_refuses_the_rest_without_ffprobe(self):
        with tempfile.TemporaryDirectory() as tmp, no_ffmpeg():
            write_wav(Path(tmp) / "a.wav", synth([(69, 0.2, 1.0)]))
            info = probe(Path(tmp) / "a.wav")
            self.assertEqual((info["has_video"], info["has_audio"]), (False, True))
            self.assertAlmostEqual(info["duration"], 1.5, delta=0.05)
            (Path(tmp) / "b.mov").write_bytes(b"not a wav")
            with self.assertRaises(ToolMissingError):                                          # anything but a WAV needs ffprobe
                probe(Path(tmp) / "b.mov")

    def test_the_whole_analysis_runs_on_a_wav_with_the_yin_tracker_and_no_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp, no_ffmpeg():
            d = Path(tmp)
            (d / "s.musicxml").write_text(musicxml(MIDIS[:8]))
            write_wav(d / "a.wav", synth([(m, 0.3 + 0.5 * i, 0.4) for i, m in enumerate(MIDIS[:8])]))
            perf = analyze(d / "a.wav", d / "s.musicxml", d / "out", pitch_backend="yin")
        self.assertIsNone(perf.video)
        self.assertTrue(perf.assessment["intonation"]["available"])
        self.assertIn("synthesis", perf.assessment)
        self.assertFalse(perf.assessment["visual_status"]["available"])                   # no video, no body movement: said, not crashed


class Capabilities(unittest.TestCase):
    def test_a_phone_has_only_the_core_and_the_app_is_told_so(self):
        with no_ffmpeg(), mock.patch.object(capabilities, "_has", return_value=False), mock.patch("maestro.score.omr._find_audiveris", return_value=None), \
                mock.patch("maestro.vision.viopose_setup.load_config", return_value={"dir": None, "python": None}):
            found = capabilities.detect()
            self.assertEqual({k: v for k, v in found.items() if k != "mobile"},
                             {"ffmpeg": False, "pitch_crepe": False, "pose": False, "scan_reader": False, "scan_cleaner": False})
            self.assertEqual(capabilities.pitch_backend(), "yin")
            self.assertTrue(all(not works for _, works, _ in capabilities.summary()))
            self.assertTrue(all(how for _, _, how in capabilities.summary()))                # each says how to get it

    def test_a_computer_with_the_crepe_network_uses_it(self):
        with mock.patch.object(capabilities, "_has", side_effect=lambda m: m == "onnxruntime"):
            self.assertEqual(capabilities.pitch_backend(), "crepe")


def photo(width=1200, height=800, shadow=True):
    """A phone-style photo of a page: staff lines in ink, a shadow across the left."""
    from PIL import Image
    x = np.linspace(0.45 if shadow else 0.95, 0.95, width)
    page = np.tile(x * 255, (height, 1))
    for i in range(5):
        page[300 + i * 24:303 + i * 24, :] = 10
    buffer = io.BytesIO()
    Image.fromarray(page.astype(np.uint8), "L").convert("RGB").save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


class Scanning(unittest.TestCase):
    def test_a_shadowed_photo_becomes_black_music_on_white_paper(self):
        page = np.asarray(scan.clean_page(photo()))
        self.assertEqual(page[100, 60], 255)                                  # paper in the shadow
        self.assertEqual(page[100, 1100], 255)                                # paper in the light
        self.assertEqual(page[301, 60], 0)                                    # a staff line in the shadow
        self.assertEqual(page[301, 1100], 0)                                  # and in the light
        self.assertEqual(page[320, 600], 255)                                 # the space between lines
        self.assertEqual(set(np.unique(page)), {0, 255})

    def test_big_photos_are_shrunk(self):
        self.assertEqual(max(scan.clean_page(photo(4800, 3200)).size), scan.LONG_SIDE)

    def test_one_page_is_a_jpeg_and_several_a_pdf_the_pipeline_accepts(self):
        with tempfile.TemporaryDirectory() as tmp:
            one = scan.pages_to_score_file([photo()], tmp)
            self.assertEqual((one.suffix, detect_score_type(one)), (".jpg", "image"))
            many = scan.pages_to_score_file([photo(), photo(shadow=False), photo()], tmp)
            self.assertEqual((many.suffix, detect_score_type(many)), (".pdf", "pdf"))
            check_pdf(many)
            if shutil.which("pdfinfo"):
                self.assertIn("Pages:           3", subprocess.run(["pdfinfo", str(many)], capture_output=True, text=True, check=True).stdout)
            with self.assertRaisesRegex(InputError, "at least one page"):
                scan.pages_to_score_file([], tmp)
            with self.assertRaisesRegex(InputError, "could not be read"):
                scan.clean_page(b"not an image")


if __name__ == "__main__":
    unittest.main()
