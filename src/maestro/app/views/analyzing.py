# The screen shown while the analysis runs (new file): a big ring, the percentage and what is happening.
import math
import time

import flet as ft

from maestro.app import theme, widgets as w

EASE_SECONDS = 12                # how long the ring takes to creep most of the way across a stage while the stage is working (as the web page did)


class Analyzing:
    """`control` is the screen; `update(info)` takes a utils/progress.Tracker report; `tick()` makes the ring creep between reports (a stage's own progress
    is not known, so it eases from where the stage starts towards where it ends and never passes it or goes back)."""

    def __init__(self):
        self.shown, self.info, self.since = 0.0, None, time.monotonic()
        self.ring = ft.ProgressRing(value=0, stroke_width=12, width=170, height=170, color=theme.ACCENT, bgcolor=ft.Colors.with_opacity(0.14, theme.ACCENT),
                                    stroke_cap=ft.StrokeCap.ROUND)
        self.percent = ft.Text("0%", size=40, weight=ft.FontWeight.W_800)
        self.label = ft.Text("Starting", size=15, text_align=ft.TextAlign.CENTER, color=ft.Colors.ON_SURFACE_VARIANT)
        dial = ft.Stack([self.ring, ft.Container(self.percent, width=170, height=170, alignment=ft.Alignment.CENTER)], width=170, height=170)
        self.control = ft.Column([ft.Container(expand=True), dial, ft.Container(self.label, padding=ft.Padding.symmetric(horizontal=28)), ft.Container(w.muted("This can take a few minutes on a long piece. You can leave the app open.", 12, text_align=ft.TextAlign.CENTER),
                                                                    padding=ft.Padding.symmetric(horizontal=28)), ft.Container(expand=True)],
                                 horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=22, expand=True)

    def update(self, info):
        if self.info is None or info["stage"] != self.info["stage"]:
            self.since = time.monotonic()
        self.info = info
        self._show(info["fraction"], info["label"])

    def tick(self):
        if self.info:
            k = 1 - math.exp(-(time.monotonic() - self.since) / EASE_SECONDS)
            self._show(self.info["fraction"] + (self.info["ceiling"] - self.info["fraction"]) * k, self.info["label"])

    def _show(self, fraction, label):
        self.shown = max(self.shown, min(fraction, 1.0))
        self.ring.value = self.shown
        self.percent.value = f"{round(self.shown * 100)}%"
        self.label.value = label
