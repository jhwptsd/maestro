# The report screen (new file): scores, the recording with the pose beside it, the written feedback and chat, live traces, problem bars and events.
#
# `Report(app, report)` builds the screen from a controller.Report (gui/player.player_data). The recording plays in a flet_video Video; while the screen is
# on, `sync()` asks the video for its position several times a second and moves the playheads of the charts and the frame of the skeleton (the position is
# polled because it is simple and the same on every platform). Tapping an event or a bar seeks the video there. Everything the model writes is shown as text.
import asyncio

import flet as ft
import flet_video as fv

from maestro.app import theme, widgets as w
from maestro.app.views.charts import Charts, Skeleton

MODULES = ("intonation", "rhythm", "expression", "tone", "bowing", "posture")
EVENTS_SHOWN = 25
SYNC_SECONDS = 0.25


def number(v):
    return "" if v is None else f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


def event_line(e):
    """One event as a title and a detail line: where, then what and how much (numbers and slugs only; the words are the teacher's)."""
    where = f"Bar {e['measure']}" + (f", beat {number(e['beat'])}" if e.get("beat") is not None else "") if e.get("measure") is not None else f"{e['start']:.1f} s"
    detail = f"{e['module']} - {e['category'].replace('_', ' ')}" + (f" - {number(e['value'])} {e['unit']}" if e.get("value") is not None and e.get("unit") else "")
    return where, detail


def bars(events, notes):
    """{bar: (worst severity or None, time to jump to)} for every bar of the piece that was played."""
    out = {n["measure"]: (None, n["start"]) for n in reversed(notes) if n.get("measure") is not None}
    for e in sorted(events, key=lambda e: e["severity"]):
        m = e.get("measure")
        if m is not None:
            out[m] = (e["severity"], e["start"])
    return dict(sorted(out.items()))


class Report:
    def __init__(self, app, report):
        self.app, self.report, self.data = app, report, report.data
        self.alive = True
        self.video = fv.Video(playlist=[fv.VideoMedia(report.media)], autoplay=False, expand=True, fit=ft.BoxFit.CONTAIN)
        T = self.data["traces"]
        self.charts = Charts(T)
        self.skeleton = Skeleton(self.data["pose"]) if self.data.get("pose") else None
        self.feedback_box = ft.Column(spacing=10)
        self.log = ft.Column(spacing=8)
        self.question = ft.TextField(hint_text="Ask about your playing, e.g. why is bar 12 out of tune?", expand=True, border_radius=24, dense=True, filled=True,
                                     on_submit=lambda e: self.app.run(self.send))
        self.control = self._build()

    # ---- layout -------------------------------------------------------------------------------------------------------
    def _build(self):
        d = self.data
        rings = [w.score_ring("overall", d["scores"].get("overall"), size=104, stroke=10)] + [w.score_ring(m, d["scores"][m]) for m in MODULES if m in d["scores"]]
        head = ft.Row([ft.IconButton(ft.Icons.ARROW_BACK, tooltip="Back", on_click=lambda e: self._leave()), w.heading("Your report", 22)], spacing=4)
        unavailable = [w.pill(f"{m}: not assessed", ft.Colors.OUTLINE) for m in d["unavailable"]]
        scores = w.card(ft.Column([ft.Row(rings, wrap=True, spacing=22, run_spacing=16, alignment=ft.MainAxisAlignment.CENTER)] +
                                  ([ft.Row(unavailable, wrap=True, spacing=8)] if unavailable else []), spacing=14))
        media = ft.Container(self.video, height=240, border_radius=16, clip_behavior=ft.ClipBehavior.ANTI_ALIAS, bgcolor="#000000")
        media_card = w.card(ft.Column([media] + ([ft.Container(self.skeleton.canvas, alignment=ft.Alignment.CENTER)] if self.skeleton else []), spacing=12), padding=14)
        feedback = w.card(ft.Column([w.section("What to work on", ft.Icons.AUTO_AWESOME), self.feedback_box], spacing=12))
        chat = w.card(ft.Column([w.section("Ask your teacher", ft.Icons.CHAT_BUBBLE_OUTLINE,
                                           ft.TextButton("New conversation", on_click=lambda e: self.new_conversation())), self.log,
                                 ft.Row([self.question, ft.IconButton(ft.Icons.SEND_ROUNDED, icon_color=theme.ACCENT, on_click=lambda e: self.app.run(self.send))], spacing=6)],
                                spacing=12))
        traces = w.card(ft.Column([w.section("Live traces", ft.Icons.INSIGHTS)] +
                                  [ft.Column([w.muted(title, 12), chart], spacing=4) for title, chart in self.charts.controls], spacing=14))
        return ft.Column([w.centered(ft.Column([head, scores, media_card, feedback, chat, traces, self._problem_bars(), self._events()], spacing=16,
                                       horizontal_alignment=ft.CrossAxisAlignment.STRETCH))],
                         scroll=ft.ScrollMode.AUTO, expand=True)

    def _problem_bars(self):
        chips = []
        for bar, (severity, t) in bars(self.data["events"], self.data["traces"]["notes"]).items():
            color = theme.severity_color(severity) if severity is not None else ft.Colors.OUTLINE_VARIANT
            chips.append(ft.Container(ft.Text(str(bar), size=13, weight=ft.FontWeight.W_700, color="#ffffff" if severity is not None else None), width=40, height=40,
                                      alignment=ft.Alignment.CENTER, border_radius=12, bgcolor=color if severity is not None else ft.Colors.SURFACE_CONTAINER_HIGH, ink=True,
                                      tooltip=f"Bar {bar}", on_click=lambda e, t=t: self.app.run(self.seek, t)))
        return w.card(ft.Column([w.section("Bar by bar", ft.Icons.TIMELINE), w.muted("Coloured bars have a problem; tap one to hear it.", 12),
                                 ft.Row(chips, wrap=True, spacing=8, run_spacing=8)], spacing=12))

    def _events(self):
        events = sorted(self.data["events"], key=lambda e: -e["severity"])
        tiles = []
        for e in events[:EVENTS_SHOWN]:
            title, detail = event_line(e)
            tiles.append(ft.Container(ft.Row([ft.Container(width=10, height=10, border_radius=5, bgcolor=theme.severity_color(e["severity"])),
                                              ft.Column([ft.Text(title, weight=ft.FontWeight.W_600), w.muted(detail, 12)], spacing=1, expand=True),
                                              ft.Icon(ft.Icons.PLAY_ARROW_ROUNDED, color=ft.Colors.ON_SURFACE_VARIANT)], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                                      padding=ft.Padding.symmetric(horizontal=12, vertical=10), border_radius=14, ink=True, bgcolor=ft.Colors.SURFACE_CONTAINER,
                                      on_click=lambda e_, t=max(0.0, e["start"] - 1.0): self.app.run(self.seek, t)))
        more = [w.muted(f"and {len(events) - EVENTS_SHOWN} smaller ones", 12)] if len(events) > EVENTS_SHOWN else []
        return w.card(ft.Column([w.section("Most important moments", ft.Icons.FLAG_OUTLINED)] + (tiles or [w.muted("No problems were found.")]) + more, spacing=8))

    # ---- behaviour ----------------------------------------------------------------------------------------------------
    async def seek(self, seconds):
        await self.video.seek(ft.Duration(milliseconds=int(seconds * 1000)))
        await self.video.play()

    async def sync(self):
        """Move the charts' playheads and the skeleton with the video until the screen is left."""
        last = -1.0
        while self.alive:
            try:
                t = (await self.video.get_current_position()).in_milliseconds / 1000
            except Exception:                                    # the player is not ready yet (or has gone): try again
                t = last
            if abs(t - last) > 0.04:
                last = t
                self.charts.set_time(t)
                if self.skeleton:
                    self.skeleton.set_time(t)
            await asyncio.sleep(SYNC_SECONDS)

    def _leave(self):
        self.alive = False
        self.app.home()

    def show_feedback(self, text=None, error=None):
        """The feedback area: paragraphs of the teacher's words, a spinner while writing, or the problem with a retry."""
        box = self.feedback_box.controls
        box.clear()
        if text is not None:
            box += [ft.Text(p.strip(), size=15, selectable=True) for p in text.split("\n\n") if p.strip()]
        elif error is not None:
            box += [ft.Row([ft.Icon(ft.Icons.ERROR_OUTLINE, color=theme.OK), ft.Text(error, expand=True)], spacing=8),
                    soft_retry(lambda e: self.app.run(self.write_feedback))]
        else:
            box += [ft.Row([ft.ProgressRing(width=20, height=20, stroke_width=3), w.muted("Writing feedback...")], spacing=12)]
        if w.mounted(self.feedback_box):
            self.feedback_box.update()

    async def write_feedback(self):
        """Write the feedback for this report (the model call runs off the interface thread) and show it, or why it could not be written."""
        if self.report.feedback:
            return self.show_feedback(self.report.feedback["text"])
        self.show_feedback()
        try:
            self.show_feedback((await asyncio.to_thread(self.app.controller.write_feedback))["text"])
        except Exception as e:                                    # ollama.LLMError is readable; anything else is shown as it is rather than lost
            self.show_feedback(error=str(e))

    async def send(self):
        text = self.question.value.strip() if self.question.value else ""
        if not text:
            return
        self.question.value = ""
        self.log.controls += [w.bubble(text, True), w.bubble("...", False)]
        self._refresh()
        try:
            reply = await asyncio.to_thread(self.app.controller.ask, text)
        except Exception as e:
            reply = str(e)
        self.log.controls[-1] = w.bubble(reply, False)
        self._refresh()

    def new_conversation(self):
        self.app.controller.new_conversation()
        self.log.controls.clear()
        self._refresh()

    def _refresh(self):
        for c in (self.log, self.question):
            if w.mounted(c):
                c.update()


def soft_retry(on_click):
    return ft.FilledTonalButton(content="Try again", icon=ft.Icons.REFRESH, on_click=on_click)
