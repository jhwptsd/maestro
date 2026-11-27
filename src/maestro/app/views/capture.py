# Recording the take and scanning the score with the device (new file): sheets with a big button, a timer and the camera.
#
# `RecordSheet` records either audio (WAV, through flet_audio_recorder: what a phone can analyse without ffmpeg) or, where ffmpeg exists to read it, video with
# the camera; `ScanSheet` photographs score pages with the back camera. Both are AlertDialogs the App shows; the camera and recorder objects are passed
# in (`camera=`, `recorder=`) so tests can stand in for the device. Permissions are asked first and a refusal is explained, never silent.
import asyncio
import time

import flet as ft
import flet_audio_recorder as far
import flet_camera as fcam
import flet_permission_handler as fph

from maestro.app import theme, widgets as w


async def allowed(permissions_service, *needed):
    """Ask for each permission in turn -> True when all are granted (a platform without the permission system, e.g. a desktop, counts as granted)."""
    for permission in needed:
        try:
            status = await permissions_service.request(permission)
        except Exception:
            return True
        if status not in (None, fph.PermissionStatus.GRANTED, fph.PermissionStatus.LIMITED):
            return False
    return True


def clock(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


class RecordSheet:
    """Record the playing. `video`: use the camera (needs ffmpeg later to read it); else the microphone as WAV. `done(path)` gets the file."""

    def __init__(self, app, video, done, camera=None, recorder=None):
        self.app, self.video, self.done = app, video, done
        self.camera = camera or (fcam.Camera(expand=True) if video else None)
        self.recorder, self.started, self.recording, self.alive = recorder or app.recorder, 0.0, False, True
        self.timer = ft.Text("0:00", size=34, weight=ft.FontWeight.W_800)
        self.hint = w.muted("Place the device where it hears you well, then tap the red button.", 13, text_align=ft.TextAlign.CENTER)
        self.button = ft.Container(ft.Icon(ft.Icons.FIBER_MANUAL_RECORD, color="#ffffff", size=40), width=84, height=84, border_radius=42, bgcolor=theme.BAD,
                                   alignment=ft.Alignment.CENTER, ink=True, on_click=lambda e: app.run(self.toggle),
                                   shadow=ft.BoxShadow(blur_radius=24, color=ft.Colors.with_opacity(0.4, theme.BAD), offset=ft.Offset(0, 8)))
        stage = ft.Container(self.camera, height=300, border_radius=16, bgcolor="#000000", clip_behavior=ft.ClipBehavior.ANTI_ALIAS) if video else \
            ft.Container(ft.Icon(ft.Icons.GRAPHIC_EQ, size=72, color=theme.ACCENT), height=150, alignment=ft.Alignment.CENTER)
        self.dialog = ft.AlertDialog(title=ft.Text("Record your playing"), content=ft.Container(ft.Column([stage, self.timer, self.hint, self.button], spacing=14,
                                                                                                         horizontal_alignment=ft.CrossAxisAlignment.CENTER), width=420),
                                     actions=[ft.TextButton("Cancel", on_click=lambda e: self.close())], on_dismiss=lambda e: self._stopped())

    async def open(self):
        """Show the sheet after the permissions are given (and start the camera preview)."""
        needed = [fph.Permission.MICROPHONE] + ([fph.Permission.CAMERA] if self.video else [])
        if not await allowed(self.app.permissions, *needed):
            return self.app.toast("Maestro needs the microphone" + (" and camera" if self.video else "") + " to record. Allow it in the device's settings.", error=True)
        self.app.page.show_dialog(self.dialog)
        if self.video:
            await asyncio.sleep(0.4)                              # the preview must be on screen before it can start
            cameras = await self.camera.get_available_cameras()
            front = next((c for c in cameras if c.lens_direction == fcam.CameraLensDirection.FRONT), cameras[0] if cameras else None)
            if front is None:
                self.close()
                return self.app.toast("No camera was found on this device.", error=True)
            await self.camera.initialize(front, fcam.ResolutionPreset.HIGH, enable_audio=True)

    async def toggle(self):
        """The red button: start, or stop and hand over the file."""
        if not self.recording:
            self.started, self.recording = time.monotonic(), True
            self.button.bgcolor, self.hint.value = theme.OK, "Recording. Tap again to stop."
            self.button.content = ft.Icon(ft.Icons.STOP_ROUNDED, color="#ffffff", size=40)
            if self.video:
                await self.camera.start_video_recording()
            else:
                path = str(self.app.controller.storage / "take.wav")
                await self.recorder.start_recording(output_path=path, configuration=far.AudioRecorderConfiguration(encoder=far.AudioEncoder.WAV, channels=1, sample_rate=44100))
            self.app.page.run_task(self._tick)
            self._refresh()
            return
        self.recording = False
        if self.video:
            path = self.app.controller.save_take(await self.camera.stop_video_recording(), ".mp4")
        else:
            path = await self.recorder.stop_recording()
        self.close()
        if path:
            self.done(str(path))

    async def _tick(self):
        while self.recording and self.alive:
            self.timer.value = clock(time.monotonic() - self.started)
            if w.mounted(self.timer):
                self.timer.update()
            await asyncio.sleep(0.5)

    def _stopped(self):
        self.alive = False

    def _refresh(self):
        for c in (self.button, self.hint):
            if w.mounted(c):
                c.update()

    def close(self):
        self.alive = False
        if w.mounted(self.dialog):
            self.app.page.pop_dialog()


class ScanSheet:
    """Photograph the pages of a score with the back camera; `done(path)` gets one cleaned file (a JPEG, or a PDF for several pages)."""

    def __init__(self, app, done, camera=None):
        self.app, self.done, self.photos = app, done, []
        self.camera = camera or fcam.Camera(expand=True)
        self.strip = ft.Row([], spacing=8, scroll=ft.ScrollMode.AUTO)
        self.use = ft.FilledButton(content="Use the pages", icon=ft.Icons.CHECK, disabled=True, on_click=lambda e: app.run(self.finish))
        self.hint = w.muted("Lay the page flat in good light and hold the camera square above it, filling the view. One photo per page.", 12)
        shutter = ft.Container(ft.Icon(ft.Icons.CAMERA_ALT, color="#ffffff", size=32), width=68, height=68, border_radius=34, gradient=theme.gradient(),
                               alignment=ft.Alignment.CENTER, ink=True, on_click=lambda e: app.run(self.shoot))
        self.dialog = ft.AlertDialog(title=ft.Text("Scan the sheet music"), content=ft.Container(ft.Column([
            ft.Container(self.camera, height=340, border_radius=16, bgcolor="#000000", clip_behavior=ft.ClipBehavior.ANTI_ALIAS), self.hint, ft.Row([shutter], alignment=ft.MainAxisAlignment.CENTER),
            self.strip], spacing=12), width=420), actions=[ft.TextButton("Cancel", on_click=lambda e: self.close()), self.use])

    async def open(self):
        if not await allowed(self.app.permissions, fph.Permission.CAMERA):
            return self.app.toast("Maestro needs the camera to scan. Allow it in the device's settings.", error=True)
        self.app.page.show_dialog(self.dialog)
        await asyncio.sleep(0.4)
        cameras = await self.camera.get_available_cameras()
        back = next((c for c in cameras if c.lens_direction == fcam.CameraLensDirection.BACK), cameras[0] if cameras else None)
        if back is None:
            self.close()
            return self.app.toast("No camera was found on this device.", error=True)
        await self.camera.initialize(back, fcam.ResolutionPreset.VERY_HIGH, enable_audio=False)

    async def shoot(self):
        """Take one photo, add its thumbnail to the strip."""
        self.photos.append(await self.camera.take_picture())
        self.use.disabled = False
        self.use.content = f"Use {len(self.photos)} page" + ("s" if len(self.photos) > 1 else "")
        self.strip.controls.append(ft.Image(src=self.photos[-1], height=70, border_radius=8))
        for c in (self.strip, self.use):
            if w.mounted(c):
                c.update()

    async def finish(self):
        """Clean the pages and hand over the file (cleaning is arithmetic on big images, so it runs off the interface thread)."""
        try:
            path = await asyncio.to_thread(self.app.controller.save_scan, self.photos)
        except Exception as e:
            return self.app.toast(str(e), error=True)
        self.close()
        self.done(str(path))

    def close(self):
        if w.mounted(self.dialog):
            self.app.page.pop_dialog()
