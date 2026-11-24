"""Finding ffmpeg/ffprobe: on PATH, in installer locations, and a readable error when they are nowhere."""
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from maestro.utils import ffmpeg
from maestro.utils.convert_video_type import probe
from maestro.utils.errors import OMRUnavailableError, ToolMissingError


class FindTool(unittest.TestCase):
    def setUp(self):
        ffmpeg._cache.clear()
        self.addCleanup(ffmpeg._cache.clear)
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def fake(self, name):
        path = self.dir / name
        path.write_text("#!/bin/sh\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def test_a_program_outside_the_process_path_is_found_in_the_installer_locations(self):
        exe = self.fake("ffprobe")
        with mock.patch.object(ffmpeg.shutil, "which", return_value=None), mock.patch.object(ffmpeg, "_extra_dirs", return_value=[str(self.dir)]):
            self.assertEqual(ffmpeg.tool("ffprobe"), str(exe))
        exe2 = self.fake("ffmpeg.exe")                                               # Windows file name
        with mock.patch.object(ffmpeg.shutil, "which", return_value=None), mock.patch.object(ffmpeg, "_extra_dirs", return_value=[str(self.dir)]):
            self.assertEqual(ffmpeg.tool("ffmpeg"), str(exe2))

    def test_the_path_comes_first(self):
        with mock.patch.object(ffmpeg.shutil, "which", return_value="/usr/bin/ffmpeg"):
            self.assertEqual(ffmpeg.tool("ffmpeg"), "/usr/bin/ffmpeg")

    def test_missing_gives_the_install_command_and_the_new_terminal_hint_and_is_not_cached(self):
        with mock.patch.object(ffmpeg.shutil, "which", return_value=None), mock.patch.object(ffmpeg, "_extra_dirs", return_value=[]):
            with self.assertRaises(ToolMissingError) as cm:
                ffmpeg.tool("ffprobe")
        self.assertIn("ffprobe was not found", str(cm.exception))
        self.assertIn("NEW terminal", str(cm.exception))
        exe = self.fake("ffprobe")                                                   # installed meanwhile: found without a restart
        with mock.patch.object(ffmpeg.shutil, "which", return_value=None), mock.patch.object(ffmpeg, "_extra_dirs", return_value=[str(self.dir)]):
            self.assertEqual(ffmpeg.tool("ffprobe"), str(exe))

    def test_probe_raises_the_readable_error_not_a_file_not_found(self):
        media = self.dir / "a.wav"
        media.write_bytes(b"x")
        with mock.patch.object(ffmpeg.shutil, "which", return_value=None), mock.patch.object(ffmpeg, "_extra_dirs", return_value=[]):
            with self.assertRaises(ToolMissingError):
                probe(media)

    def test_the_audiveris_error_is_the_same_kind(self):
        self.assertTrue(issubclass(OMRUnavailableError, ToolMissingError))


if __name__ == "__main__":
    unittest.main()
