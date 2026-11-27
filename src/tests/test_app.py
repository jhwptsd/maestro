"""The Flet app without a screen: the controller (analysis, feedback, chat), the views as control trees, and the App's actions against a stand-in page and
stand-in camera, recorder and file picker. (No Flutter client runs here, so what is checked is the logic and the content of every screen, not pixels.)"""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import flet as ft
import flet_permission_handler as fph

from helpers import make_video, musicxml, synth, write_wav
from minuet import MIDIS, minuet_musicxml, play_minuet
from pose_fixture import minuet_notes, synth_pose
from test_feedback import FakeOllama, GOOD_ANSWER
from test_ondevice import photo
from maestro.app import theme, widgets as w
from maestro.app.app import App
from maestro.app.controller import Controller
from maestro.app.views import analyzing, charts, home, report as report_view, settings as settings_view
from maestro.app.views.capture import RecordSheet, ScanSheet, allowed, clock
from maestro.feedback import ollama
from maestro.utils.errors import InputError
from maestro.vision.pose_io import save_pose


class FakePage:
    """What App needs of ft.Page: services, controls, add/update, dialogs, and run_task / run_thread that record what was asked."""

    def __init__(self):
        self.services, self.controls, self.tasks, self.dialogs, self.updates = [], [], [], [], 0

    def add(self, *controls):
        self.controls += controls

    def update(self):
        self.updates += 1

    def run_task(self, fn, *args):
        self.tasks.append((fn, args))

    def run_thread(self, fn, *args):
        fn(*args)

    def show_dialog(self, dialog):
        self.dialogs.append(dialog)

    def pop_dialog(self):
        self.dialogs.pop()

    def task(self, name):
        """The coroutine function recorded under `name` (run it with asyncio.run)."""
        return next((fn, args) for fn, args in self.tasks if fn.__name__ == name)


def files(d, bars=slice(2, 5)):
    (d / "s.musicxml").write_text(minuet_musicxml(100))
    write_wav(d / "take.wav", play_minuet(bars=bars, jitter=0.01))
    return d / "take.wav", d / "s.musicxml"


class WithApp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = Path(self._tmp.name)
        patcher = mock.patch.dict("os.environ", {"MAESTRO_CONFIG": str(self.d / "cfg")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.fake = FakeOllama()
        self.addCleanup(self.fake.close)
        patcher = mock.patch.object(ollama, "BASE_URL", self.fake.url)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.controller = Controller(self.d / "storage")


class ControllerTests(WithApp):
    def test_analysis_gives_a_report_with_progress_that_only_moves_forward(self):
        take, score = files(self.d)
        self.controller.set_video(take)
        self.controller.set_score(score)
        self.assertTrue(self.controller.ready)
        seen = []
        report = self.controller.analyze(seen.append)
        fractions = [i["fraction"] for i in seen]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(seen[-1]["stage"], "done")
        self.assertNotIn("reference", [i["stage"] for i in seen])
        self.assertEqual((report.kind, Path(report.media).name), ("audio", "take.wav"))
        self.assertTrue({"scores", "events", "traces", "unavailable"} <= set(report.data))
        self.assertIn("overall", report.data["scores"])
        self.assertIn("scores_0_to_100", report.digest)
        self.assertIs(self.controller.report, report)

    def test_what_is_needed_is_said(self):
        with self.assertRaisesRegex(InputError, "Choose a recording and a score"):
            self.controller.analyze(lambda info: None)
        self.controller.set_video(self.d / "missing.wav")
        self.controller.set_score(self.d / "missing.musicxml")
        with self.assertRaises(InputError):
            self.controller.analyze(lambda info: None)

    def test_a_take_from_the_camera_and_scanned_pages_become_the_chosen_files(self):
        path = self.controller.save_take(b"RIFFxxxx", ".mp4")
        self.assertEqual((self.controller.video, path.suffix), (str(path), ".mp4"))
        scanned = self.controller.save_scan([photo(), photo()])
        self.assertEqual((self.controller.score, scanned.suffix), (str(scanned), ".pdf"))
        self.controller.set_video(None)
        self.assertFalse(self.controller.ready)

    def test_feedback_and_the_chat_are_kept_on_the_report(self):
        take, score = files(self.d)
        self.controller.set_video(take)
        self.controller.set_score(score)
        report = self.controller.analyze(lambda info: None)
        self.fake.answer = GOOD_ANSWER
        self.assertEqual(self.controller.write_feedback()["text"], GOOD_ANSWER)
        self.assertEqual(report.feedback["text"], GOOD_ANSWER)
        self.fake.answer = "Start with the shift."
        self.assertEqual(self.controller.ask("What first?"), "Start with the shift.")
        self.assertEqual([m["role"] for m in report.chat], ["user", "assistant"])
        self.controller.new_conversation()
        self.assertEqual(report.chat, [])
        report.chat = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}] * 20
        with self.assertRaisesRegex(ollama.LLMError, "got long"):
            self.controller.ask("one more")

    def test_a_phone_without_a_writing_model_is_told_what_is_true(self):
        take, score = files(self.d)
        self.controller.set_video(take)
        self.controller.set_score(score)
        self.controller.analyze(lambda info: None)
        caps = {"ffmpeg": False, "pitch_crepe": False, "pose": False, "scan_reader": False, "scan_cleaner": True, "mobile": True}
        with mock.patch.object(ollama, "BASE_URL", "http://127.0.0.1:1"), mock.patch("maestro.utils.capabilities.detect", return_value=caps):
            with self.assertRaisesRegex(ollama.LLMError, "this phone cannot run one yet"):
                self.controller.write_feedback()
        with mock.patch.object(ollama, "BASE_URL", "http://127.0.0.1:1"):                          # a computer is told to start Ollama
            with self.assertRaisesRegex(ollama.LLMError, "Ollama is not running"):
                self.controller.write_feedback()

    def test_model_state_choice_and_download(self):
        state = self.controller.model_state()
        self.assertEqual((state["running"], state["model"]), (True, "llama3.2:3b"))
        self.fake.models.append("qwen3.5:4b")
        self.controller.choose_model("qwen3.5:4b")
        self.assertEqual(self.controller.model_state()["model"], "qwen3.5:4b")
        with self.assertRaisesRegex(ollama.LLMError, "not downloaded"):
            self.controller.choose_model("nope")
        seen = []
        self.controller.download_model("llama3.2:1b", lambda f, text: seen.append(f))
        self.assertEqual((max(seen), self.controller.model_state()["model"]), (1.0, "llama3.2:1b"))
        with mock.patch.object(ollama, "BASE_URL", "http://127.0.0.1:1"):
            self.assertEqual(self.controller.model_state()["running"], False)


class ViewTests(WithApp):
    def app(self):
        return SimpleNamespace(controller=self.controller, run=lambda *a, **k: None, settings=lambda: None, home=lambda: None)

    def test_home_offers_what_the_device_can_do_and_enables_analyze_only_when_ready(self):
        app = self.app()
        screen = home.build(app)
        text = w.texts(screen)
        for expected in ("Feedback from a teacher who listens.", "Record now", "Choose a recording", "Scan the sheet music", "Choose a score file", "Analyze my playing"):
            self.assertIn(expected, text)
        analyze = next(c for c in w.walk(screen) if isinstance(c, ft.Container) and c.on_click is None and c.gradient and "Analyze my playing" in w.texts(c))
        self.assertEqual(analyze.opacity, 0.4)                                                    # dimmed and inert until both are chosen
        self.controller.set_video(self.d / "take.wav")
        self.controller.set_score(self.d / "s.musicxml")
        ready = next(c for c in w.walk(home.build(app)) if isinstance(c, ft.Container) and c.gradient and "Analyze my playing" in w.texts(c))
        self.assertIsNotNone(ready.on_click)
        self.assertEqual(ready.opacity, 1)
        self.assertNotIn("Record now", w.texts(home.build(app)))                                  # chosen files replace the choices
        self.assertIn("take.wav", w.texts(home.build(app)))

    def test_a_phone_is_told_what_it_cannot_do_and_scanning_is_disabled_without_the_reader(self):
        caps = {"ffmpeg": False, "pitch_crepe": False, "pose": False, "scan_reader": False, "scan_cleaner": True, "mobile": True}
        with mock.patch("maestro.utils.capabilities.detect", return_value=caps):
            text = w.texts(home.build(self.app()))
        self.assertIn("On this device", text)
        self.assertTrue(any("Not available here" in t and "body movement" in t for t in text))
        self.assertIn("A WAV file (other formats need ffmpeg)", text)
        self.assertTrue(any("music reader (Audiveris)" in t for t in text))

    def test_the_progress_screen_never_goes_back_and_eases_inside_a_stage(self):
        screen = analyzing.Analyzing()
        screen.update({"stage": "pitch", "label": "Finding the pitch", "fraction": 0.3, "ceiling": 0.5})
        first = screen.shown
        screen.update({"stage": "pitch", "label": "Finding the pitch", "fraction": 0.2, "ceiling": 0.5})
        self.assertEqual(screen.shown, first)                                                    # a lower report cannot move it back
        screen.since -= 30
        screen.tick()
        self.assertTrue(0.45 < screen.shown <= 0.5)                                              # creeps towards the end of the stage, not past it
        self.assertEqual((screen.percent.value, screen.label.value), (f"{round(screen.shown * 100)}%", "Finding the pitch"))

    def test_settings_show_the_model_the_downloads_and_the_device(self):
        state = self.controller.model_state()
        text = w.texts(settings_view.build(self.app(), state))
        for expected in ("Written feedback", "Using llama3.2:3b.", "In use", "Download", "This device", "Any video or audio format"):
            self.assertIn(expected, text)
        stopped = w.texts(settings_view.build(self.app(), {"running": False, "model": None, "installed": [], "suggested": []}))
        self.assertIn("Ollama is not running", stopped)

    def test_the_device_check_card_lists_what_worked_and_what_did_not(self):
        from maestro.utils import selftest
        results = [{"name": "Pitch tracker (YIN)", "ok": True, "core": True, "seconds": 0.1, "detail": "A4 found within 0.3 cents"},
                   {"name": "Body movement (VioPose)", "ok": False, "core": False, "seconds": 0.0, "detail": "RuntimeError: VioPose is not installed"}]
        text = w.texts(settings_view.build(self.app(), self.controller.model_state(), results))
        for expected in ("Check this device", "Pitch tracker (YIN)", "A4 found within 0.3 cents", "RuntimeError: VioPose is not installed", "The core works here"):
            self.assertIn(expected, text)
        failing = results + [{"name": "Whole analysis of a piece", "ok": False, "core": True, "seconds": 1.0, "detail": "ImportError: scipy"}]
        self.assertIn("Something the phone app needs does not work", w.texts(settings_view.build(self.app(), self.controller.model_state(), failing)))
        self.assertNotIn("The core works here", w.texts(settings_view.build(self.app(), self.controller.model_state())))                # before it is run
        self.assertTrue(selftest.core_ok(results) and not selftest.core_ok(failing))

    def test_report_screen_shows_scores_bars_events_and_a_chat(self):
        take, score = files(self.d)
        self.controller.set_video(take)
        self.controller.set_score(score)
        data = self.controller.analyze(lambda info: None)
        app = SimpleNamespace(controller=self.controller, run=lambda *a, **k: None, home=lambda: None)
        view = report_view.Report(app, data)
        text = w.texts(view.control)
        for expected in ("Your report", "Overall", "What to work on", "Ask your teacher", "Live traces", "Bar by bar", "Most important moments"):
            self.assertIn(expected, text)
        self.assertTrue(any(t.startswith("Bar ") for t in text))                                  # events name their bar
        view.show_feedback("First paragraph.\n\nSecond paragraph.")
        self.assertEqual(w.texts(view.feedback_box), ["First paragraph.", "Second paragraph."])
        view.show_feedback(error="Ollama is not running at http://localhost:11434.")
        self.assertIn("Ollama is not running at http://localhost:11434.", w.texts(view.feedback_box))
        self.assertIn("Writing feedback...", (view.show_feedback(), w.texts(view.feedback_box))[1])
        self.assertEqual(len(view.charts.controls), 3)                                           # pitch, loudness, per-note error (no pose here)
        self.assertIsNone(view.skeleton)

    def test_bars_and_event_lines(self):
        notes = [{"measure": 1, "start": 0.0}, {"measure": 1, "start": 0.5}, {"measure": 2, "start": 1.0}, {"measure": 3, "start": 2.0}]
        events = [{"measure": 2, "severity": 0.8, "start": 1.1}, {"measure": 2, "severity": 0.3, "start": 1.3}, {"measure": 1, "severity": 0.2, "start": 0.4}]
        got = report_view.bars(events, notes)
        self.assertEqual(list(got), [1, 2, 3])
        self.assertEqual(got[2], (0.8, 1.1))                                                      # the worst problem of the bar
        self.assertEqual(got[3], (None, 2.0))                                                     # no problem
        self.assertEqual(report_view.event_line({"measure": 4, "beat": 2.0, "module": "intonation", "category": "sharp_note", "value": 23.456, "unit": "cents", "start": 3.0}),
                         ("Bar 4, beat 2.00", "intonation - sharp note - 23.5 cents"))
        self.assertEqual(report_view.event_line({"measure": None, "module": "tone", "category": "x", "value": None, "unit": None, "start": 3.04})[0], "3.0 s")

    def test_the_skeleton_follows_the_time_and_the_video_gets_a_pose_when_there_is_one(self):
        d = self.d
        (d / "m.musicxml").write_text(minuet_musicxml())
        write_wav(d / "a.wav", play_minuet())
        make_video(d / "v.mp4", d / "a.wav", 18)
        save_pose(synth_pose(minuet_notes(), with_bow=True), d / "pose.json")
        from maestro import analyze
        from maestro.gui.player import player_data
        perf = analyze(d / "v.mp4", d / "m.musicxml", d / "out", pose_file=d / "pose.json", pitch_backend="yin")
        data = player_data(perf, str(perf.video), "video")
        skeleton = charts.Skeleton(data["pose"])
        first = list(skeleton.canvas.shapes)
        self.assertTrue(first)
        skeleton.set_time(8.0)
        self.assertEqual(skeleton.frame, round(8.0 * data["pose"]["fps"]))
        self.assertNotEqual([getattr(s, "x1", None) for s in skeleton.canvas.shapes], [getattr(s, "x1", None) for s in first])
        self.assertTrue(all(0 <= getattr(s, "x1", 0) <= skeleton.SIZE * 2 + 20 for s in skeleton.canvas.shapes))        # inside the canvas
        all_charts = charts.Charts(data["traces"])
        self.assertEqual(len(all_charts.controls), 4)                                            # + the bow
        all_charts.set_time(5.0)
        self.assertEqual(all_charts._heads[0][1].points[0].x, 5.0)                               # the playhead moved


class AppActions(WithApp):
    def make(self):
        page = FakePage()
        app = App(page, self.controller)
        return page, app

    def test_choosing_files_updates_the_home_screen(self):
        page, app = self.make()
        picked = []
        app.picker = SimpleNamespace(pick_files=lambda **kw: _result(picked.append(kw) or [SimpleNamespace(path=str(self.d / "take.wav"))]))
        asyncio.run(app.pick_video())
        self.assertEqual(self.controller.video, str(self.d / "take.wav"))
        self.assertIn("take.wav", w.texts(page.controls[-1]))
        self.assertEqual(picked[0]["allowed_extensions"][:1], ["wav"])
        app.picker = SimpleNamespace(pick_files=lambda **kw: _result([]))                      # cancelled: nothing changes
        asyncio.run(app.pick_score())
        self.assertIsNone(self.controller.score)
        app.picker = SimpleNamespace(pick_files=lambda **kw: _result([SimpleNamespace(path=None)]))
        asyncio.run(app.pick_score())
        self.assertIsNone(self.controller.score)
        self.assertTrue(any(isinstance(d_, ft.SnackBar) for d_ in page.dialogs))                 # told to choose a file on the device

    def test_the_analysis_runs_behind_a_progress_screen_and_ends_on_the_report_with_the_feedback_started(self):
        page, app = self.make()
        take, score = files(self.d)
        self.controller.set_video(take)
        self.controller.set_score(score)
        asyncio.run(app.start_analysis())
        self.assertIn("Your report", w.texts(page.controls[-1]))
        fn, args = page.task("write_feedback")
        self.fake.answer = GOOD_ANSWER
        asyncio.run(fn(*args))
        self.assertIn("Your tempo is lovely and steady, which is a real strength.", w.texts(app.report_view.feedback_box))
        page.tasks.clear()
        app.report_view.question.value = "What first?"
        self.fake.answer = "Start with the shift."
        asyncio.run(app.report_view.send())
        self.assertIn("Start with the shift.", w.texts(app.report_view.log))
        view = app.report_view
        self.assertTrue(view.alive)
        app.home()                                                                                # leaving the report stops its sync loop
        self.assertFalse(view.alive)
        self.assertIsNone(app.report_view)

    def test_unusable_files_send_the_user_back_with_the_reason(self):
        page, app = self.make()
        self.controller.set_video(self.d / "missing.wav")
        self.controller.set_score(self.d / "missing.musicxml")
        asyncio.run(app.start_analysis())
        self.assertIn("Analyze my playing", w.texts(page.controls[-1]))                          # back on the home screen
        self.assertTrue(any(isinstance(d_, ft.SnackBar) for d_ in page.dialogs))

    def test_recording_asks_permission_records_wav_and_hands_over_the_file(self):
        page, app = self.make()
        calls = []

        class Recorder:
            async def start_recording(self, output_path, configuration):
                calls.append(("start", output_path, configuration.encoder.value))

            async def stop_recording(self):
                calls.append(("stop",))
                return str(self.path)
        recorder = Recorder()
        recorder.path = self.d / "take.wav"
        write_wav(recorder.path, synth([(69, 0.2, 1.0)]))
        app.permissions = SimpleNamespace(request=lambda p: _result(fph.PermissionStatus.GRANTED))
        got = []
        sheet = RecordSheet(app, False, got.append, recorder=recorder)
        asyncio.run(sheet.open())
        self.assertEqual(page.dialogs, [sheet.dialog])
        asyncio.run(sheet.toggle())
        self.assertEqual(calls[0][2], "wav")                                                      # WAV: readable without ffmpeg
        self.assertTrue(sheet.recording)
        asyncio.run(sheet.toggle())
        self.assertEqual(got, [str(recorder.path)])
        self.assertEqual([c[0] for c in calls], ["start", "stop"])

    def test_a_refused_permission_is_explained_and_nothing_opens(self):
        page, app = self.make()
        app.permissions = SimpleNamespace(request=lambda p: _result(fph.PermissionStatus.PERMANENTLY_DENIED))
        asyncio.run(RecordSheet(app, False, lambda p: None).open())
        self.assertEqual([type(d_) for d_ in page.dialogs], [ft.SnackBar])
        self.assertIn("microphone", w.texts(page.dialogs[0].content)[0])
        self.assertTrue(asyncio.run(allowed(SimpleNamespace(request=lambda p: _boom()), fph.Permission.CAMERA)))          # no permission system: fine

    def test_video_recording_uses_the_camera_and_saves_the_bytes(self):
        page, app = self.make()
        app.permissions = SimpleNamespace(request=lambda p: _result(fph.PermissionStatus.GRANTED))
        camera = FakeCamera(video=b"\x00\x00\x00\x18ftypmp42 video")
        got = []
        sheet = RecordSheet(app, True, got.append, camera=camera)
        asyncio.run(sheet.open())
        self.assertEqual(camera.initialized[0].lens_direction.value, "front")
        asyncio.run(sheet.toggle())
        asyncio.run(sheet.toggle())
        self.assertEqual(Path(got[0]).suffix, ".mp4")
        self.assertEqual(Path(got[0]).read_bytes(), camera.video)
        self.assertEqual(self.controller.video, got[0])

    def test_scanning_takes_photos_and_hands_over_one_cleaned_file(self):
        page, app = self.make()
        app.permissions = SimpleNamespace(request=lambda p: _result(fph.PermissionStatus.GRANTED))
        camera = FakeCamera(picture=photo())
        got = []
        sheet = ScanSheet(app, got.append, camera=camera)
        asyncio.run(sheet.open())
        self.assertEqual(camera.initialized[0].lens_direction.value, "back")                      # the back camera for paper
        self.assertTrue(sheet.use.disabled)
        asyncio.run(sheet.shoot())
        self.assertEqual((sheet.use.disabled, sheet.use.content), (False, "Use 1 page"))
        asyncio.run(sheet.shoot())
        self.assertEqual(sheet.use.content, "Use 2 pages")
        asyncio.run(sheet.finish())
        self.assertEqual(Path(got[0]).suffix, ".pdf")
        self.assertEqual(self.controller.score, got[0])                                           # the controller made the file and chose it
        self.assertTrue(Path(got[0]).is_file())

    def test_the_device_check_runs_the_real_checks_and_shows_them(self):
        page, app = self.make()
        asyncio.run(app.device_check())
        text = w.texts(page.controls[-1])
        self.assertIn("Whole analysis of a piece", text)
        self.assertIn("The core works here", text)

    def test_clock(self):
        self.assertEqual((clock(5), clock(65), clock(600)), ("0:05", "1:05", "10:00"))


class FakeCamera:
    def __init__(self, video=b"", picture=b""):
        self.video, self.picture, self.initialized = video, picture, []

    async def get_available_cameras(self):
        return [SimpleNamespace(name="b", lens_direction=ft.Colors and __import__("flet_camera").CameraLensDirection.BACK),
                SimpleNamespace(name="f", lens_direction=__import__("flet_camera").CameraLensDirection.FRONT)]

    async def initialize(self, description, preset, enable_audio=True):
        self.initialized.append(description)

    async def start_video_recording(self):
        pass

    async def stop_video_recording(self):
        return self.video

    async def take_picture(self):
        return self.picture


async def _result(value):
    return value


async def _boom():
    raise RuntimeError("no permission system here")


class Packaging(unittest.TestCase):
    def test_the_entry_point_and_the_phone_requirements_are_in_place(self):
        root = Path(__file__).resolve().parents[2]
        self.assertIn("ft.run(main)", (root / "src" / "main.py").read_text())
        phone = (root / "src" / "requirements.txt").read_text()
        for needed in ("numpy", "scipy", "Pillow", "flet-camera", "flet-audio-recorder", "flet-video"):
            self.assertIn(needed, phone)
        for impossible in ("torch", "onnxruntime", "tensorflow"):
            self.assertNotIn(impossible, [line.split("#")[0].strip().split(">")[0] for line in phone.splitlines()])
        import tomllib
        config = tomllib.loads((root / "pyproject.toml").read_text())
        self.assertEqual(config["tool"]["flet"]["app"]["path"], "src")
        self.assertIn("NSMicrophoneUsageDescription", config["tool"]["flet"]["ios"]["info"])
        self.assertIn("app", config["project"]["optional-dependencies"])


class Theme(unittest.TestCase):
    def test_colours_follow_the_report_cut_offs(self):
        self.assertEqual([theme.score_color(v) for v in (90, 85, 70, 59)], [theme.GOOD, theme.GOOD, theme.OK, theme.BAD])
        self.assertEqual([theme.severity_color(s) for s in (0.1, 0.5, 0.9)], ["#d9b13b", theme.OK, theme.BAD])


if __name__ == "__main__":
    unittest.main()
