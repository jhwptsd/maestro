# The Flet app itself (new file): the page, the navigation between screens, and the actions the screens trigger.
#
# `main(page)` is what Flet calls. `App` owns the page, the Controller (the logic) and the device services (file picker, microphone, permissions); the views in
# views/ are built from it and call back into it (`app.run(action)` runs an async action on the event loop or a plain one in a thread, so nothing slow
# ever runs on the interface). One screen is on the page at a time; navigating is replacing it.
import asyncio
import inspect
import os

import flet as ft
import flet_audio_recorder as far
import flet_permission_handler as fph

from maestro.app import theme
from maestro.app.controller import Controller
from maestro.app.views import analyzing, home, settings
from maestro.app.views.capture import RecordSheet, ScanSheet
from maestro.app.views.report import Report
from maestro.utils import capabilities, selftest
from maestro.utils.errors import InputError, ToolMissingError


class App:
    def __init__(self, page, controller=None):
        self.page = page
        self.controller = controller or Controller()
        self.picker, self.recorder, self.permissions = ft.FilePicker(), far.AudioRecorder(), fph.PermissionHandler()
        page.services.extend([self.picker, self.recorder, self.permissions])
        self.report_view = None

    # ---- plumbing -----------------------------------------------------------------------------------------------------
    def run(self, action, *args):
        """Run an action off the interface: a coroutine function on the event loop, a plain function in a thread."""
        return self.page.run_task(action, *args) if inspect.iscoroutinefunction(action) else self.page.run_thread(action, *args)

    def show(self, screen):
        """Put `screen` on the page in place of the current one."""
        if self.report_view:
            self.report_view.alive = False
            self.report_view = None
        self.page.controls.clear()
        self.page.add(ft.SafeArea(screen, expand=True))
        self.page.update()

    def toast(self, message, error=False):
        self.page.show_dialog(ft.SnackBar(ft.Text(message, color="#ffffff"), bgcolor=theme.BAD if error else None, duration=6000 if error else 3000))

    # ---- screens ------------------------------------------------------------------------------------------------------
    def home(self):
        self.show(home.build(self))

    async def settings_screen(self, results=None):
        self.show(settings.build(self, await asyncio.to_thread(self.controller.model_state), results))

    async def device_check(self):
        """Settings > Check this device: try each part for real (off the interface thread) and show what happened."""
        await self.settings_screen(await asyncio.to_thread(selftest.run))

    def settings(self):
        self.run(self.settings_screen)

    def report(self):
        """The report of the analysis just made; the feedback starts being written as it opens."""
        view = Report(self, self.controller.report)
        self.show(view.control)                                  # (this ends the previous report's sync loop, so the new view is set after)
        self.report_view = view
        self.run(view.write_feedback)
        self.run(view.sync)

    # ---- what the screens ask for -------------------------------------------------------------------------------------
    async def pick_video(self):
        caps = capabilities.detect()
        await self._pick(self.controller.set_video, home.MEDIA if caps["ffmpeg"] else ["wav"])

    async def pick_score(self):
        caps = capabilities.detect()
        await self._pick(self.controller.set_score, home.SCORE + (home.SCORE_SCANNED if caps["scan_reader"] else []))

    async def _pick(self, setter, extensions):
        files = await self.picker.pick_files(allow_multiple=False, file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=extensions)
        if not files:
            return
        if not files[0].path:
            return self.toast("Choose a file stored on this device.", error=True)
        setter(files[0].path)
        self.home()

    async def record_take(self):
        """Record with the camera where the video can be read afterwards (ffmpeg), else with the microphone; a camera that will not start falls back to the microphone."""
        def done(path):
            self.controller.set_video(path)
            self.home()
        if capabilities.detect()["ffmpeg"]:
            try:
                return await RecordSheet(self, True, done).open()
            except Exception:
                self.toast("The camera could not be started: recording the sound only.", error=True)
        await RecordSheet(self, False, done).open()

    async def scan_score(self):
        def done(path):
            self.controller.set_score(path)
            self.home()
        await ScanSheet(self, done).open()

    async def use_model(self, model):
        try:
            await asyncio.to_thread(self.controller.choose_model, model)
        except Exception as e:
            return self.toast(str(e), error=True)
        await self.settings_screen()

    async def get_model(self, model):
        """Download a writing model with its progress shown on the settings screen's toast line, then choose it."""
        self.toast(f"Downloading {model}...")
        try:
            await asyncio.to_thread(self.controller.download_model, model, lambda fraction, label: None)
        except Exception as e:
            return self.toast(str(e), error=True)
        self.toast(f"{model} is ready.")
        await self.settings_screen()

    async def start_analysis(self):
        """Run the analysis behind the progress screen, then show the report (or say what was wrong with the files and go back)."""
        view = analyzing.Analyzing()
        self.show(view.control)
        running = True

        async def ticker():
            while running:
                view.tick()
                self.page.update()
                await asyncio.sleep(0.25)
        self.page.run_task(ticker)
        try:
            await asyncio.to_thread(self.controller.analyze, view.update)
        except (InputError, ToolMissingError) as e:
            running = False
            self.home()
            return self.toast(str(e), error=True)
        except Exception as e:                                    # a bug: say so briefly rather than leave the progress screen up
            running = False
            self.home()
            return self.toast(f"Something went wrong: {e}", error=True)
        running = False
        self.report()


def main(page):
    if os.environ.get("FLET_APP_STORAGE_DATA"):                  # on a phone the app keeps its settings in its own folder
        os.environ.setdefault("MAESTRO_CONFIG", os.environ["FLET_APP_STORAGE_DATA"])
    theme.apply(page)
    App(page).home()
