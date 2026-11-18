"""VioPose setup (vision/viopose_setup.py): environment choice, download.sh handling, the step-by-step installer with
every external command replaced by a recorder (no network, no multi-GB installs), idempotence, configuration."""
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from maestro.vision import viopose_setup as vs

DOWNLOAD_SH = """\
cd ./config
gdown CONFIG_ID
cd ..

mkdir ./Logs
cd ./Logs
gdown WEIGHTS_ID
gdown WEIGHTS_WO_AUDIO_ID
unzip ./VioPose.zip
unzip ./VioPose_wo_audio.zip
rm ./VioPose_wo_audio.zip
rm ./VioPose.zip
cd ..

# demo data (not needed)
mkdir ./output
mkdir ./data
mkdir ./data/demo
cd ./data/demo
gdown DEMO_ID
cd ..
"""
ZIPS = {"WEIGHTS_ID": "VioPose.zip", "WEIGHTS_WO_AUDIO_ID": "VioPose_wo_audio.zip"}


class Recorder:
    """Stands in for subprocess.run: records commands and fakes what they would produce on disk."""
    def __init__(self, checkout_files=None):
        self.calls = []
        self.checkout_files = checkout_files if checkout_files is not None else {
            "inference.py": "# fake", "requirements.txt": "tensorflow==2.13.1\n", "download.sh": DOWNLOAD_SH}

    def __call__(self, cmd, cwd=None):
        cmd = [str(c) for c in cmd]
        self.calls.append((cmd, cwd))
        if cmd[:2] == ["git", "clone"]:                          # creates the checkout
            target = Path(cmd[-1])
            target.mkdir(parents=True)
            for name, text in self.checkout_files.items():
                (target / name).write_text(text)
        elif cmd[1:3] == ["-m", "gdown"] and cmd[-1] in ZIPS:     # "downloads" a weights zip into cwd
            with zipfile.ZipFile(Path(cwd) / ZIPS[cmd[-1]], "w") as z:
                z.writestr(ZIPS[cmd[-1]][:-4] + "/weights.txt", "w")

    def commands(self, word):
        return [c for c, _ in self.calls if word in c]


class DownloadScript(unittest.TestCase):
    def test_parse_keeps_only_the_commands_download_sh_uses(self):
        actions = vs.parse_download_script(DOWNLOAD_SH + "echo done\nexport X=1\n")
        self.assertEqual([a for a, _ in actions].count("gdown"), 4)
        self.assertEqual(actions[0], ("cd", ["./config"]))
        self.assertNotIn("echo", [a for a, _ in actions])
        self.assertEqual(vs.parse_download_script("mkdir -p ./a   # comment\n"), [("mkdir", ["./a"])])    # flags dropped

    def test_running_it_downloads_weights_unzips_them_and_skips_demo_data(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "config").mkdir()
            recorder = Recorder()
            vs.run_download_script(DOWNLOAD_SH, root, "py", recorder)
            downloads = [c[-1] for c, _ in recorder.calls]
            self.assertEqual(downloads, ["CONFIG_ID", "WEIGHTS_ID", "WEIGHTS_WO_AUDIO_ID"])      # DEMO_ID skipped
            self.assertEqual(recorder.calls[0][1], (root / "config").resolve())                   # run inside ./config
            self.assertTrue((root / "Logs/VioPose/weights.txt").is_file())
            self.assertTrue((root / "Logs/VioPose_wo_audio/weights.txt").is_file())
            self.assertFalse(list((root / "Logs").glob("*.zip")))                                # zips removed
            self.assertTrue((root / "output").is_dir())
            with tempfile.TemporaryDirectory() as d2:                                             # skip_demo=False keeps it
                (Path(d2) / "config").mkdir()
                again = Recorder()
                vs.run_download_script(DOWNLOAD_SH, d2, "py", again, skip_demo=False)
                self.assertEqual(len(again.calls), 4)


class EnvironmentChoice(unittest.TestCase):
    def which(self, *present):
        return lambda name: f"/bin/{name}" if name in present else None

    def test_prefers_conda_then_uv_then_a_python39_venv(self):
        env = Path("/x/env")
        self.assertEqual(vs.environment_command(env, self.which("conda", "uv"))[:2], ["conda", "create"])
        uv = vs.environment_command(env, self.which("uv"))
        self.assertEqual(uv[:2], ["uv", "venv"])
        self.assertIn("--seed", uv)                                  # uv venvs have no pip unless seeded
        with mock.patch.object(sys, "platform", "linux"):
            run = mock.Mock(return_value=mock.Mock(returncode=0))
            self.assertEqual(vs.environment_command(env, self.which("python3.9"), run), ["python3.9", "-m", "venv", str(env)])
        with mock.patch.object(sys, "platform", "win32"):
            run = mock.Mock(return_value=mock.Mock(returncode=0))
            self.assertEqual(vs.environment_command(env, self.which("py"), run), ["py", "-3.9", "-m", "venv", str(env)])

    def test_nothing_available_is_reported_with_the_options(self):
        with mock.patch.object(sys, "platform", "linux"):
            with self.assertRaisesRegex(vs.SetupError, "Miniconda"):
                vs.environment_command(Path("/x/env"), self.which())

    def test_interpreter_path_follows_the_platform(self):
        with mock.patch.object(sys, "platform", "win32"):
            self.assertEqual(vs.env_python("E"), Path("E/Scripts/python.exe"))
        with mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(vs.env_python("E"), Path("E/bin/python"))


class Installer(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tp = Path(self._tmp.name) / "third_party"
        self.addCleanup(self._tmp.cleanup)
        for name, value in (("CONFIG_PATH", self.tp / "viopose.json"), ("DEFAULT_DIR", self.tp / "VioPose")):
            patcher = mock.patch.object(vs, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.log = []

    def setup(self, recorder, **kw):
        return vs.setup_viopose(self.tp / "VioPose", python=sys.executable, run=recorder, log=self.log.append, **kw)

    def test_full_install_runs_the_steps_in_order_and_records_the_config(self):
        recorder = Recorder()
        config = self.setup(recorder)
        steps = [("clone" if c[0] == "git" else "pip" if "pip" in c else "gdown" if "gdown" in c else "check" if "-c" in c else "?") for c, _ in recorder.calls]
        self.assertEqual(steps, ["clone", "pip", "check", "gdown", "gdown", "gdown"])
        pip = recorder.commands("pip")[0]
        self.assertEqual(pip[:4], [sys.executable, "-m", "pip", "install"])
        self.assertTrue(pip[-1].endswith("requirements.txt"))
        self.assertEqual(Path(config["dir"]), (self.tp / "VioPose").resolve())
        self.assertEqual(config["python"], sys.executable)
        self.assertTrue((self.tp / "VioPose/Logs/VioPose_wo_audio/weights.txt").is_file())

    def test_rerunning_does_nothing_and_a_changed_requirements_file_reinstalls(self):
        recorder = Recorder()
        self.setup(recorder)
        before = len(recorder.calls)
        self.setup(recorder)
        self.assertEqual(len(recorder.calls), before)                # everything already done
        (self.tp / "VioPose/requirements.txt").write_text("tensorflow==2.14.0\n")
        self.setup(recorder)
        self.assertEqual(len(recorder.commands("pip")), 2)           # only the requirements step ran again
        self.assertEqual(len(recorder.calls), before + 2)            # pip + the import check

    def test_unusable_pins_are_left_out_of_the_install(self):
        text, dropped = vs.filter_requirements("tensorflow==2.13.1\njaxlib==0.4.13\nWurlitzer==3.0\nnumpy>=1.2\n")
        self.assertEqual(text, "tensorflow==2.13.1\nnumpy>=1.2\n")
        self.assertEqual(len(dropped), 2)

    def test_failed_bulk_install_falls_back_to_single_pins_and_bad_imports_are_reported(self):
        def run(cmd, **kw):
            if "-r" in cmd or "bad==1" in cmd or "-c" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return mock.Mock(returncode=0)
        calls = []
        failed = vs.install_requirements("py", Path("r.txt"), "# c\ngood==1\nbad==1\n",
                                         lambda c, **kw: (calls.append(c), run(c))[1], self.log.append)
        self.assertEqual(failed, ["bad==1"])
        self.assertEqual(len(calls), 3)                              # bulk, good, bad
        with self.assertRaisesRegex(vs.SetupError, "bad==1"):
            vs.check_imports("py", ["cv2"], failed, run)

    def test_skip_weights_and_dry_run(self):
        recorder = Recorder()
        self.setup(recorder, skip_weights=True)
        self.assertEqual(recorder.commands("gdown"), [])
        self.assertFalse((self.tp / "VioPose/Logs").exists())
        dry = Recorder()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(vs, "CONFIG_PATH", Path(d) / "c.json"):
            vs.setup_viopose(Path(d) / "VioPose", python=sys.executable, dry_run=True, run=dry, log=self.log.append)
            self.assertEqual(dry.calls, [])                          # nothing executed
            self.assertFalse((Path(d) / "c.json").exists())          # nothing written
            self.assertFalse((Path(d) / "VioPose").exists())
        self.assertTrue(any("git clone" in line for line in self.log))      # but the plan was shown

    def test_missing_weights_are_an_error_and_a_missing_python_installs_uv(self):
        broken = Recorder({"inference.py": "#", "requirements.txt": "x\n", "download.sh": "mkdir ./Logs\n"})   # no weights produced
        with self.assertRaisesRegex(vs.SetupError, "weights are missing"):
            self.setup(broken)
        recorder = Recorder()
        with mock.patch.object(vs, "environment_command", side_effect=vs.SetupError("none")), \
                mock.patch.object(vs, "THIRD_PARTY", self.tp):
            vs.setup_viopose(self.tp / "Other", run=recorder, log=self.log.append, skip_weights=True)
        self.assertEqual(recorder.commands("uv")[0][:4], [sys.executable, "-m", "pip", "install"])           # pip install uv
        self.assertIn("venv", recorder.calls[2][0])                                                          # then uv makes the venv


class Config(unittest.TestCase):
    def test_precedence_environment_then_saved_file_then_default_checkout(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict("os.environ", clear=False):
            import os
            for k in ("VIOPOSE_DIR", "VIOPOSE_PYTHON"):
                os.environ.pop(k, None)
            d = Path(d)
            with mock.patch.object(vs, "CONFIG_PATH", d / "viopose.json"), mock.patch.object(vs, "DEFAULT_DIR", d / "VioPose"):
                self.assertEqual(vs.load_config(), {"dir": None, "python": None})                    # nothing installed
                (d / "VioPose").mkdir()
                self.assertEqual(vs.load_config()["dir"], str(d / "VioPose"))                        # default checkout
                vs.write_config("/saved/dir", "/saved/python")
                self.assertEqual(vs.load_config(), {"dir": "/saved/dir", "python": "/saved/python"})  # saved file
                os.environ["VIOPOSE_DIR"] = "/env/dir"
                self.assertEqual(vs.load_config(), {"dir": "/env/dir", "python": "/saved/python"})    # environment wins
                (d / "viopose.json").write_text("{corrupt")
                os.environ.pop("VIOPOSE_DIR")
                self.assertEqual(vs.load_config()["dir"], str(d / "VioPose"))                        # a damaged file is ignored


if __name__ == "__main__":
    unittest.main()
