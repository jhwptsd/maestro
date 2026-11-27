# The first screen (new file): choose or record the playing, choose or scan the score, analyse.
#
# What is offered follows what this device can do (utils/capabilities.py): on a phone, where there is no ffmpeg, the take is recorded as audio (WAV) and
# the score must be MusicXML or MIDI; on a computer with ffmpeg, Audiveris and VioPose everything is offered.
import flet as ft

from maestro.app import theme, widgets as w
from maestro.utils import capabilities

MEDIA = ["wav", "mp3", "m4a", "aac", "flac", "ogg", "mp4", "mov", "m4v", "webm", "mkv"]
SCORE = ["musicxml", "xml", "mxl", "mid", "midi"]
SCORE_SCANNED = ["pdf", "png", "jpg", "jpeg", "tif", "tiff"]          # these need the music reader (Audiveris)


def hero(app):
    """The gradient banner: name, one line of promise, the settings button."""
    top = ft.Row([ft.Row([ft.Icon(ft.Icons.GRAPHIC_EQ, color="#ffffff", size=26), ft.Text("Maestro", size=22, weight=ft.FontWeight.W_800, color="#ffffff")], spacing=8),
                  ft.IconButton(ft.Icons.TUNE, icon_color="#ffffff", tooltip="Settings", on_click=lambda e: app.settings())],
                 alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
    body = ft.Column([top, ft.Text("Feedback from a teacher who listens.", size=28, weight=ft.FontWeight.W_800, color="#ffffff"),
                      ft.Text("Play a piece, show Maestro the score, and hear what to work on first.", size=14, color=ft.Colors.with_opacity(0.85, "#ffffff"))], spacing=10)
    return ft.Container(body, padding=ft.Padding.only(left=22, right=12, top=18, bottom=30), gradient=theme.gradient(),
                        border_radius=ft.BorderRadius(top_left=0, top_right=0, bottom_left=32, bottom_right=32))


def chosen(icon, name, on_clear):
    """What has been chosen, with a way to undo it."""
    return ft.Container(ft.Row([ft.Icon(icon, color=theme.GOOD), ft.Text(name, expand=True, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS, weight=ft.FontWeight.W_600),
                                ft.IconButton(ft.Icons.CLOSE, icon_size=18, tooltip="Choose another", on_click=on_clear)], spacing=10,
                               vertical_alignment=ft.CrossAxisAlignment.CENTER),
                        padding=ft.Padding.only(left=14, right=4, top=2, bottom=2), border_radius=16, bgcolor=ft.Colors.with_opacity(0.12, theme.GOOD))


def take_card(app, caps):
    c = app.controller
    if c.video:
        inner = chosen(ft.Icons.AUDIOTRACK, c.video.replace("\\", "/").rsplit("/", 1)[-1], lambda e: (c.set_video(None), app.home()))
    else:
        subtitle = "Film yourself with the camera and microphone" if caps["ffmpeg"] else "Record with the microphone (WAV)"
        inner = ft.Column([w.option(ft.Icons.MIC, "Record now", subtitle, lambda e: app.run(app.record_take)),
                           w.option(ft.Icons.FOLDER_OPEN, "Choose a recording", "A WAV, MP3, MP4 or MOV file" if caps["ffmpeg"] else "A WAV file (other formats need ffmpeg)",
                                    lambda e: app.run(app.pick_video))], spacing=10)
    return w.card(ft.Column([w.section("1  Your playing", ft.Icons.HEARING), inner], spacing=14))


def score_card(app, caps):
    c = app.controller
    if c.score:
        inner = chosen(ft.Icons.QUEUE_MUSIC, c.score.replace("\\", "/").rsplit("/", 1)[-1], lambda e: (c.set_score(None), app.home()))
    else:
        can_scan = caps["scan_cleaner"] and caps["scan_reader"]
        scan_hint = ("Photograph each page" if can_scan else
                     "Needs the music reader (Audiveris), which this device does not have: use a MusicXML or MIDI file" if caps["scan_cleaner"] else "Needs Pillow")
        inner = ft.Column([w.option(ft.Icons.DOCUMENT_SCANNER, "Scan the sheet music", scan_hint, lambda e: app.run(app.scan_score), enabled=can_scan),
                           w.option(ft.Icons.FOLDER_OPEN, "Choose a score file", "MusicXML, MIDI" + (", PDF or image" if caps["scan_reader"] else ""),
                                    lambda e: app.run(app.pick_score))], spacing=10)
    return w.card(ft.Column([w.section("2  The score", ft.Icons.LIBRARY_MUSIC), inner], spacing=14))


def device_card(caps):
    """What this device cannot do, in one quiet card (nothing when everything works)."""
    missing = [name for name, works, _ in capabilities.summary() if not works]
    if not missing:
        return None
    return w.card(ft.Column([ft.Row([ft.Icon(ft.Icons.INFO_OUTLINE, size=18, color=theme.OK), ft.Text("On this device", weight=ft.FontWeight.W_600)], spacing=8),
                             w.muted("Not available here: " + ", ".join(m.lower() for m in missing) + ". Maestro uses what the device can do instead; "
                                     "open Settings to see how to get the rest.", 12)], spacing=6), padding=16)


def build(app):
    """The home screen as one control."""
    caps = capabilities.detect()
    body = [take_card(app, caps), score_card(app, caps),
            w.primary_button("Analyze my playing", ft.Icons.AUTO_AWESOME, lambda e: app.run(app.start_analysis), enabled=app.controller.ready)]
    note = device_card(caps)
    if note is not None:
        body.append(note)
    return ft.Column([hero(app), w.centered(ft.Column(body, spacing=16, horizontal_alignment=ft.CrossAxisAlignment.STRETCH))], scroll=ft.ScrollMode.AUTO, expand=True, spacing=0)
