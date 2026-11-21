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
            "inference.py": "from tensorflow import keras", "requirements.txt": "tensorflow==2.13.1\n", "download.sh": DOWNLOAD_SH}

    def __call__(self, cmd, cwd=None):
        cmd = [str(c) for c in cmd]
        self.calls.append((cmd, cwd))
        if cmd[:2] == ["git", "clone"]:                          # creates the checkout
            target = Path(cmd[-1])
            target.mkdir(parents=True)
            for name, text in self.checkout_files.items():
                (target / name).write_text(text)
        elif cmd[1:3] == ["-c", vs.GDOWN_CODE] and cmd[-1] in ZIPS:     # "downloads" a weights zip into cwd
            with zipfile.ZipFile(Path(cwd) / ZIPS[cmd[-1]], "w") as z:
                z.writestr(ZIPS[cmd[-1]][:-4] + "/weights.txt", "w")

    def commands(self, word):
        return [c for c, _ in self.calls if (vs.GDOWN_CODE if word == "gdown" else word) in c]


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
        patcher = mock.patch.object(vs, "environment_problem", return_value=None)      # the interpreter is "a working 3.9"
        patcher.start()
        self.addCleanup(patcher.stop)
        self.log = []

    def setup(self, recorder, **kw):
        return vs.setup_viopose(self.tp / "VioPose", python=sys.executable, run=recorder, log=self.log.append, **kw)

    def test_full_install_runs_the_steps_in_order_and_records_the_config(self):
        recorder = Recorder()
        config = self.setup(recorder)
        steps = [("clone" if c[0] == "git" else "pip" if "pip" in c else "gdown" if vs.GDOWN_CODE in c else "check" if "-c" in c else "?") for c, _ in recorder.calls]
        self.assertEqual(steps, ["clone", "pip", "check", "pip", "gdown", "gdown", "gdown"])
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
        self.assertEqual(len(recorder.commands("pip")), 3)           # (requirements, gdown) + only the requirements step again
        self.assertEqual(len(recorder.calls), before + 2)            # pip + the import check

    def test_only_the_pins_of_imported_packages_are_installed(self):
        text = "tensorflow==2.13.1\ntriton==2.3.1\nopencv-contrib-python==4.8.1\nKeras==2.13.1\nlibclang==16\n"
        out = vs.core_requirements(text, ["import cv2", "from tensorflow import keras"])
        self.assertEqual(out, "tensorflow==2.13.1\nopencv-contrib-python==4.8.1\nKeras==2.13.1\n")

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
            vs.check_imports("py", ["import cv2"], failed, run)

    def test_damaged_windows_tensorflow_is_reinstalled_from_tensorflow_intel(self):
        recorder = Recorder()
        with mock.patch.object(sys, "platform", "win32"):
            self.assertTrue(vs.repair_tensorflow("py", "tensorflow==2.13.1\n", recorder, self.log.append))
        self.assertIn("tensorflow-intel==2.13.1", recorder.calls[0][0])
        with mock.patch.object(sys, "platform", "linux"):
            self.assertFalse(vs.repair_tensorflow("py", "tensorflow==2.13.1\n", recorder, self.log.append))

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


class EnvironmentChecks(unittest.TestCase):
    """The VioPose environment must really be a working Python 3.9 with pip: checked, and rebuilt when it is not."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tp = Path(self._tmp.name) / "third_party"
        self.addCleanup(self._tmp.cleanup)
        for name, value in (("CONFIG_PATH", self.tp / "viopose.json"), ("DEFAULT_DIR", self.tp / "VioPose")):
            patcher = mock.patch.object(vs, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.log = []

    def test_what_is_wrong_is_named(self):
        for info, expected in (({"version": (3, 9), "pip": True}, None), ({"version": (3, 13), "pip": True}, "Python 3.13"),
                               ({"version": None, "pip": False}, "cannot be run"), ({"version": (3, 9), "pip": False}, "pip")):
            with mock.patch.object(vs, "inspect_interpreter", return_value=info):
                problem = vs.environment_problem("py")
                self.assertTrue(problem is None if expected is None else expected in problem, problem)
        self.assertEqual(vs.inspect_interpreter(self.tp / "no" / "python"), {"version": None, "pip": False})   # never raises
        self.assertEqual(vs.inspect_interpreter(sys.executable)["version"], sys.version_info[:2])

    def fake_env(self):
        env = self.tp / "viopose-env"
        python = vs.env_python(env)
        python.parent.mkdir(parents=True)
        python.write_text("stub")
        return env

    def test_an_existing_environment_of_the_wrong_python_is_rebuilt(self):
        env = self.fake_env()
        (self.tp / ".viopose-requirements").write_text("stale")
        problems = iter(["it is Python 3.13", None])                    # before the rebuild, after it
        recorder = Recorder()
        with mock.patch.object(vs, "environment_problem", side_effect=lambda p: next(problems)), \
                mock.patch.object(vs, "environment_command", return_value=["makeenv", str(env)]):
            vs.setup_viopose(self.tp / "VioPose", skip_weights=True, run=recorder, log=self.log.append)
        self.assertTrue(any("rebuilding" in m and "Python 3.13" in m for m in self.log))
        self.assertEqual(len(recorder.commands("makeenv")), 1)
        self.assertFalse(vs.env_python(env).exists())                   # the old environment was removed first
        self.assertEqual([c for c, _ in recorder.calls if c[:2] == ["git", "clone"]] != [], True)
        self.assertNotEqual((self.tp / ".viopose-requirements").read_text(), "stale")   # reinstalled into the new one

    def test_a_good_environment_is_kept(self):
        env = self.fake_env()
        recorder = Recorder()
        with mock.patch.object(vs, "environment_problem", return_value=None):
            vs.setup_viopose(self.tp / "VioPose", skip_weights=True, run=recorder, log=self.log.append)
        self.assertTrue(any("already created" in m for m in self.log))
        self.assertEqual(recorder.commands("makeenv") + recorder.commands("venv"), [])
        self.assertTrue(vs.env_python(env).exists())

    def test_if_the_first_way_gives_a_wrong_python_uv_fetches_3_9(self):
        env = self.tp / "viopose-env"
        problems = iter(["it is Python 3.13", None])                    # after conda/venv, after uv
        recorder = Recorder()
        with mock.patch.object(vs, "environment_problem", side_effect=lambda p: next(problems)), \
                mock.patch.object(vs, "environment_command", return_value=["makeenv", str(env)]):
            vs.create_environment(env, lambda c, cwd=None: recorder(c, cwd), self.log.append)
        sequence = [c[:3] for c, _ in recorder.calls]
        self.assertEqual(sequence[0][0], "makeenv")
        self.assertIn(["-m", "pip", "install"], [c[1:4] for c, _ in recorder.calls])          # uv was installed ...
        self.assertTrue(any(c[1:3] == ["-m", "uv"] and "3.9" in c for c, _ in recorder.calls))  # ... and asked for 3.9

    def test_nothing_working_is_a_clear_error_not_a_later_import_failure(self):
        class Failing(Recorder):
            def __call__(self, cmd, cwd=None):
                super().__call__(cmd, cwd)
                if "makeenv" in cmd or "uv" in cmd and "venv" in cmd:
                    raise subprocess.CalledProcessError(101, cmd)

        recorder = Failing()
        with mock.patch.object(vs, "environment_command", return_value=["makeenv", "x"]):
            with self.assertRaisesRegex(vs.SetupError, "working Python 3.9 environment"):
                vs.create_environment(self.tp / "viopose-env", lambda c, cwd=None: recorder(c, cwd), self.log.append)

    def test_an_unusable_python_argument_is_refused(self):
        with mock.patch.object(vs, "environment_problem", return_value="it is Python 3.13"):
            with self.assertRaisesRegex(vs.SetupError, "must be a working Python 3.9"):
                vs.setup_viopose(self.tp / "VioPose", python="/opt/py313", skip_weights=True, run=Recorder(), log=self.log.append)

    def test_a_failing_repair_reports_the_import_failure_not_the_pip_error(self):
        env = self.fake_env()
        with mock.patch.object(vs, "environment_problem", return_value=None), \
                mock.patch.object(vs, "check_imports", side_effect=vs.SetupError("importing keras fails")), \
                mock.patch.object(vs, "repair_tensorflow", side_effect=subprocess.CalledProcessError(101, ["pip"])):
            with self.assertRaisesRegex(vs.SetupError, "importing keras fails"):
                vs.setup_viopose(self.tp / "VioPose", skip_weights=True, run=Recorder(), log=self.log.append)
        self.assertTrue(env.exists())


if __name__ == "__main__":
    unittest.main()
