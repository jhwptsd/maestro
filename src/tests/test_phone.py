"""Maestro under an iPhone's restrictions (utils/phone_sandbox.py): no PyTorch, ONNX Runtime, librosa or other packages without an iOS build, no way to start
another program (so no ffmpeg, Audiveris or VioPose). The core must still work, and everything that cannot must say so in words, never crash."""
import asyncio
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import musicxml, synth, write_wav
from minuet import minuet_musicxml, play_minuet
from maestro import analyze
from maestro.app.controller import Controller
from maestro.audio.pitch import estimate_pitch
from maestro.score import omr
from maestro.utils import capabilities
from maestro.utils.convert_video_type import probe
from maestro.utils.errors import InputError, OMRUnavailableError, PoseUnavailableError, ToolMissingError
from maestro.utils.phone_sandbox import NO_IOS_BUILD, phone_sandbox
from maestro.vision.pose_estimation import estimate_pose

PROGRAM_LOOKUPS = {"ffmpeg", "ffprobe", "which audiveris"}        # the app looks for these (and finds nothing); nothing else may be tried


class Sandbox(unittest.TestCase):
    def test_what_has_no_ios_build_cannot_be_imported_and_is_not_installed(self):
        with phone_sandbox() as box:
            for name in ("torch", "onnxruntime", "librosa", "tensorflow"):
                with self.assertRaisesRegex(ImportError, "not available on iOS"):
                    __import__(name)
            import importlib.util
            self.assertIsNone(importlib.util.find_spec("torchcrepe"))                  # "is it installed?" is no, not an error
            self.assertIsNotNone(importlib.util.find_spec("numpy"))
            import numpy, scipy.signal                                                  # noqa: F401  (the core is allowed)
        self.assertEqual(box.tried("import"), ["librosa", "onnxruntime", "tensorflow", "torch"])
        self.assertTrue(all(a["where"].startswith("test_phone.py") for a in box.attempts if a["kind"] == "import"))     # it says who tried

    def test_no_program_can_be_started_or_found(self):
        import shutil
        with phone_sandbox() as box:
            with self.assertRaisesRegex(OSError, "cannot start other programs"):
                subprocess.run(["ffmpeg", "-version"])
            with self.assertRaises(OSError):
                subprocess.Popen(["ls"])
            with self.assertRaises(OSError):
                __import__("os").system("ls")
            self.assertIsNone(shutil.which("ls"))
        self.assertIn("ffmpeg", box.tried("program"))
        self.assertEqual(subprocess.run([sys.executable, "-c", "print(1)"], capture_output=True).returncode, 0)    # everything is restored afterwards
        self.assertIsNotNone(shutil.which("ls"))

    def test_the_blocked_packages_are_the_ones_the_audit_names(self):
        self.assertTrue({"torch", "onnxruntime", "librosa", "tensorflow", "cv2"} <= set(NO_IOS_BUILD))


class TheCoreStillWorks(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = Path(self._tmp.name)
        (self.d / "s.musicxml").write_text(minuet_musicxml(100))
        write_wav(self.d / "take.wav", play_minuet(bars=slice(2, 5), jitter=0.01))

    def test_the_whole_analysis_runs_and_tries_nothing_a_phone_cannot_do_beyond_looking_for_programs(self):
        with phone_sandbox() as box:
            controller = Controller(self.d / "storage")
            controller.set_video(self.d / "take.wav")
            controller.set_score(self.d / "s.musicxml")
            report = controller.analyze(lambda info: None)
        self.assertEqual(box.tried("import"), [])                                         # no hidden dependency on a package without an iOS build
        self.assertLessEqual(set(box.tried("program")), PROGRAM_LOOKUPS)
        self.assertEqual(sorted(report.data["scores"]), ["expression", "intonation", "overall", "rhythm", "tone"])
        self.assertIn("bowing", report.data["unavailable"])                                # no video: said, not hidden

    def test_the_app_chooses_the_yin_tracker_here(self):
        with phone_sandbox():
            self.assertEqual(capabilities.pitch_backend(), "yin")
            self.assertFalse(any(capabilities.detect()[k] for k in ("ffmpeg", "pitch_crepe", "pose", "scan_reader")))

    def test_the_screens_build_and_the_app_runs_an_analysis_inside_the_sandbox(self):
        from test_app import FakePage
        from maestro.app import widgets as w
        from maestro.app.app import App
        page = FakePage()
        with phone_sandbox() as box:
            app = App(page, Controller(self.d / "storage"))
            app.home()
            self.assertIn("Choose a recording", w.texts(page.controls[-1]))
            self.assertTrue(any("Not available here" in t for t in w.texts(page.controls[-1])))
            app.controller.set_video(self.d / "take.wav")
            app.controller.set_score(self.d / "s.musicxml")
            asyncio.run(app.start_analysis())
            self.assertIn("Your report", w.texts(page.controls[-1]))
        self.assertEqual(box.tried("import"), [])


class WhatCannotRunSaysSoInWords(unittest.TestCase):
    """Each missing piece must be a readable error (the app shows it), never a crash or a silent wrong answer."""

    def test_each_unavailable_feature_is_explained(self):
        with tempfile.TemporaryDirectory() as tmp, phone_sandbox():
            d = Path(tmp)
            write_wav(d / "a.wav", synth([(69, 0.2, 1.0)]))
            (d / "b.mov").write_bytes(b"\x00\x00\x00\x18ftypqt  " + b"\x00" * 64)
            with self.assertRaisesRegex(ToolMissingError, "ffmpeg"):                      # any format but WAV
                probe(d / "b.mov")
            with self.assertRaises(PoseUnavailableError):                                 # body movement
                estimate_pose(str(d / "b.mov"))
            with self.assertRaises(OMRUnavailableError):                                  # reading a photographed score
                omr.run_omr(d / "scan.jpg", d)
            with self.assertRaises(ImportError):                                          # the CREPE network
                estimate_pitch(synth([(69, 0.2, 1.0)]), 22050, backend="crepe")
            self.assertEqual(estimate_pitch(synth([(69, 0.2, 1.0)]), 22050, backend="yin").backend, "yin")


class SelfTest(unittest.TestCase):
    """utils/selftest.py: the on-device check."""

    def test_the_core_checks_pass_here_and_a_failure_is_a_result_not_an_exception(self):
        from maestro.utils import selftest
        results = selftest.run()
        core = {r["name"]: r for r in results if r["core"]}
        self.assertEqual(sorted(core), ["Numbers (numpy)", "Pitch tracker (YIN)", "Platform", "Read a WAV recording", "Signal processing (scipy)", "Whole analysis of a piece"])
        self.assertTrue(all(r["ok"] for r in core.values()), {n: r["detail"] for n, r in core.items() if not r["ok"]})
        self.assertTrue(selftest.core_ok(results))
        self.assertEqual(results[-1]["name"], "Peak memory")
        broken = selftest.run([("Broken", lambda: 1 / 0, True), ("Fine", lambda: "ok", False)])
        self.assertEqual([(r["name"], r["ok"]) for r in broken[:2]], [("Broken", False), ("Fine", True)])
        self.assertIn("ZeroDivisionError", broken[0]["detail"])
        self.assertFalse(selftest.core_ok(broken))
        self.assertIn("FAIL  Broken", selftest.as_text(broken))

    def test_inside_the_phone_sandbox_the_core_still_passes_and_the_rest_is_reported_missing(self):
        from maestro.utils import selftest
        with phone_sandbox() as box:
            results = {r["name"]: r for r in selftest.run()}
        self.assertTrue(all(r["ok"] for r in results.values() if r["core"]))
        for missing in ("Other audio and video formats (ffmpeg)", "CREPE pitch network (ONNX Runtime)", "Reading a photographed score (Audiveris)"):
            self.assertFalse(results[missing]["ok"], missing)
        self.assertEqual(box.tried("import"), ["onnxruntime"])                              # only the one check that asks for it


class Audit(unittest.TestCase):
    """src/scripts/ios_audit.py: what the code needs, by feature, and what the package indexes say."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import ios_audit
        cls.audit = ios_audit
        cls.report = ios_audit.audit()

    def test_imports_are_needed_or_optional_and_program_calls_are_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "m.py"
            source.write_text("import os\nimport numpy as np\nfrom scipy import signal\nimport subprocess\n"
                              "try:\n    import torch\nexcept ImportError:\n    torch = None\n"
                              "def f():\n    import librosa\n    subprocess.run(['x'])\n    import shutil\n    return shutil.which('y')\n")
            imports, programs = self.audit.scan_file(source, "m.py")
        self.assertEqual({m: needed for m, _, needed in imports}, {"numpy": True, "scipy": True, "torch": False, "librosa": False})
        self.assertEqual([c for c, _ in programs], ["subprocess.run", "shutil.which"])

    def test_the_longest_matching_prefix_names_the_feature(self):
        self.assertEqual(self.audit.feature_of("audio/backends/crepe.py"), "Pitch: CREPE")
        self.assertEqual(self.audit.feature_of("audio/features.py"), "Notes, alignment and the score (MusicXML, MIDI)")
        self.assertEqual(self.audit.feature_of("assessment/neural.py"), "Tone and expression model (CLAP)")
        self.assertIsNone(self.audit.feature_of("gui/server.py"))

    def test_the_phone_path_needs_nothing_but_numpy_and_scipy_at_the_top_of_a_file(self):
        """The regression guard of the audit: a top-level import of a package with no iOS build in the features a phone uses fails here."""
        for feature in ("Reading a recording (WAV)", "Pitch: YIN", "Notes, alignment and the score (MusicXML, MIDI)", "Intonation, rhythm, expression, tone",
                        "Written feedback (the writing model)"):
            needed = {m for m, info in self.report[feature]["packages"].items() if info["needed"]}
            self.assertLessEqual(needed, {"numpy", "scipy"}, feature)

    def test_the_heavy_things_are_optional_and_the_programs_are_listed(self):
        crepe = self.report["Pitch: CREPE"]["packages"]
        self.assertTrue({"torch", "onnxruntime", "torchcrepe"} <= set(crepe) and not any(crepe[m]["needed"] for m in ("torch", "onnxruntime", "torchcrepe")))
        self.assertTrue(any("subprocess" in p for p in self.report["Reading other audio and video (ffmpeg)"]["programs"]))
        self.assertTrue(any("shutil.which" in p for p in self.report["Scanning a score (reading photos and PDFs)"]["programs"]))

    def test_verdicts_follow_the_package_status(self):
        report = {"A": {"packages": {"torch": {"needed": True, "files": []}}, "programs": []},
                  "B": {"packages": {"torch": {"needed": False, "files": []}, "numpy": {"needed": True, "files": []}}, "programs": []},
                  "C": {"packages": {"numpy": {"needed": True, "files": []}}, "programs": []}}
        self.audit.verdicts(report, {"torch": "none", "numpy": "ios"})
        self.assertEqual([report[k]["verdict"] for k in "ABC"], ["blocked", "degrades", "runs"])
        self.assertEqual(report["B"]["why"], ["torch has no iOS wheel"])
        self.assertEqual(self.audit.verdicts({"C": {"packages": {}, "programs": []}}, None)["C"]["verdict"], "unchecked")

    def test_wheel_status_asks_pip_for_the_iphone_platform(self):
        def pip(returncode, stderr="", name=None):
            def run(cmd, **kw):
                self.assertIn("ios_13_0_arm64_iphoneos", cmd)
                if name:
                    Path(cmd[cmd.index("-d") + 1], name).write_bytes(b"x")
                return subprocess.CompletedProcess(cmd, returncode, "", stderr)
            return run
        for result, expected in ((pip(0, name="x-1-py3-none-any.whl"), "pure"), (pip(0, name="x-1-cp313-cp313-ios_13_0_arm64_iphoneos.whl"), "ios"),
                                 (pip(1, "ERROR: Could not find a version that satisfies the requirement x"), "none"), (pip(1, "connection refused"), "unknown")):
            with mock.patch.object(self.audit.subprocess, "run", side_effect=result):
                self.assertEqual(self.audit.wheel_status("x"), expected)
        with mock.patch.object(self.audit.subprocess, "run", side_effect=pip(1, "no")) as run:
            self.audit.wheel_status("x", flet=True)
        self.assertIn(self.audit.FLET_INDEX, run.call_args.args[0])


class FakeEngine(__import__("maestro.feedback.ondevice", fromlist=["Engine"]).Engine):
    name = "fake"

    def __init__(self, ready=True, answer="Your tempo is steady; work on the shift in bar 4."):
        self.ready, self.answer, self.calls = ready, answer, []

    def available(self):
        return self.ready

    def models(self):
        return [{"id": "tiny-3b", "label": "Tiny 3B"}]

    def generate(self, model, system, messages, max_tokens):
        self.calls.append({"model": model, "system": system, "messages": messages, "max_tokens": max_tokens})
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class OnDeviceModel(unittest.TestCase):
    """The seam for a native writing model: without Ollama, an engine on the device writes the feedback and answers the chat."""

    def setUp(self):
        from maestro.feedback import ondevice
        self.ondevice = ondevice
        self.addCleanup(ondevice.unregister)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.dict("os.environ", {"MAESTRO_CONFIG": self._tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        from maestro.feedback import ollama
        self.ollama = ollama
        patcher = mock.patch.object(ollama, "BASE_URL", "http://127.0.0.1:1")                     # no Ollama, as on a phone
        patcher.start()
        self.addCleanup(patcher.stop)

    def digest(self):
        from test_feedback import sample_perf
        from maestro.feedback import prompts
        return prompts.build_digest(sample_perf())

    def test_without_an_engine_or_ollama_nothing_is_available_and_it_says_so(self):
        with self.assertRaisesRegex(self.ollama.LLMError, "Ollama is not running"):
            self.ollama.list_models()

    def test_a_registered_engine_is_listed_preferred_and_used_without_ollama(self):
        from maestro.feedback import chat, generator
        engine = FakeEngine()
        self.ondevice.register(engine)
        self.assertEqual(self.ollama.list_models(), [{"id": "device:tiny-3b", "label": "Tiny 3B - on this device"}])
        self.assertEqual(self.ollama.pick_model([{"id": "llama3.2:3b"}, {"id": "device:tiny-3b"}]), "device:tiny-3b")      # no server, no download
        feedback = generator.feedback_from_digest(self.digest())
        self.assertEqual((feedback["model"], feedback["text"]), ("device:tiny-3b", engine.answer))
        sent = engine.calls[-1]
        self.assertEqual((sent["model"], sent["max_tokens"]), ("tiny-3b", generator.MAX_TOKENS))                          # the prefix is the app's, not the engine's
        self.assertIn("teacher", sent["system"])
        self.assertIn("Measurements", sent["messages"][-1]["content"])
        reply = chat.reply(self.digest(), feedback, [], "What first?")
        self.assertEqual(reply["reply"], engine.answer)
        self.assertEqual(engine.calls[-1]["messages"][-1], {"role": "user", "content": "What first?"})

    def test_an_engine_that_cannot_run_now_is_ignored_and_its_failures_are_readable(self):
        from maestro.feedback import generator
        engine = FakeEngine(ready=False)
        self.ondevice.register(engine)
        self.assertIsNone(self.ondevice.engine())
        with self.assertRaisesRegex(self.ollama.LLMError, "Ollama is not running"):
            generator.feedback_from_digest(self.digest())
        engine.ready, engine.answer = True, RuntimeError("out of memory")
        with self.assertRaisesRegex(self.ollama.LLMError, "could not answer .out of memory."):
            generator.feedback_from_digest(self.digest())
        engine.answer = "   "
        with self.assertRaisesRegex(self.ollama.LLMError, "no answer"):
            generator.feedback_from_digest(self.digest())

    def test_the_settings_say_the_model_is_on_the_device(self):
        from maestro.app import widgets as w
        from maestro.app.controller import Controller
        from maestro.app.views import settings
        self.ondevice.register(FakeEngine())
        state = Controller.model_state()
        self.assertEqual((state["running"], state["device"], state["model"]), (False, True, "device:tiny-3b"))
        text = " ".join(w.texts(settings.build(mock.Mock(), state)))
        self.assertIn("runs on this device", text)
        self.assertNotIn("Ollama is not running", text)

    def test_an_engine_reached_through_an_async_call_is_driven_from_a_worker_thread(self):
        import asyncio
        import threading
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()

        def stop():
            loop.call_soon_threadsafe(loop.stop)
            thread.join(5)
            loop.close()
        self.addCleanup(stop)
        calls = []

        async def invoke(method, arguments):                                           # what a Flet extension control's method call looks like
            calls.append((method, arguments))
            if method == "status":
                return {"available": True, "models": [{"id": "afm", "label": "Apple on-device model"}]}
            return {"text": "Listen to the soloist's shift in bar 4."}
        engine = self.ondevice.CallbackEngine(invoke, loop)
        self.assertFalse(engine.available())                                               # nothing is known until the engine has been asked
        asyncio.run_coroutine_threadsafe(engine.refresh(), loop).result(5)
        self.ondevice.register(engine)
        self.assertEqual([m["id"] for m in self.ollama.list_models()], ["device:afm"])
        result = self.ollama.complete("device:afm", "SYSTEM", [{"role": "user", "content": "hi"}], 300)
        self.assertEqual(result["text"], "Listen to the soloist's shift in bar 4.")
        self.assertEqual(calls[-1], ("generate", {"model": "afm", "system": "SYSTEM", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 300}))


if __name__ == "__main__":
    unittest.main()
