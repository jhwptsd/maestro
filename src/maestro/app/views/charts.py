# The report's live traces and the pose skeleton (new file), drawn with flet_charts and flet.canvas from the data of gui/player.player_data.
#
# `Charts` builds one line chart per trace (pitch against the written notes, loudness, the error of every note, and with a pose the bow) and a playhead on
# each; `set_time(t)` moves the playheads with the video. `Skeleton` draws the estimated pose, front and side, at the frame of the video's time.
# Long traces are thinned to `POINTS` points so a phone can draw them.
import bisect

import flet as ft
import flet.canvas as cv
import flet_charts as fch

from maestro.app import theme
from maestro.app.widgets import mounted

POINTS = 500


def thin(t, y, limit=POINTS):
    """(t, y) lists -> at most `limit` points, dropping the gaps (None) of an unvoiced pitch trace."""
    pairs = [(a, b) for a, b in zip(t, y) if b is not None]
    step = max(1, len(pairs) // limit)
    return pairs[::step]


def _axis(show=True, size=34):
    return fch.ChartAxis(show_labels=show, label_size=size, show_min=False, show_max=False)


class Charts:
    """The trace charts of a report. `controls`: [(title, chart)]; `set_time(t)` places every playhead."""

    def __init__(self, traces):
        self.controls, self._heads = [], []
        T = traces
        pitch = thin(T["pitch"]["t"], T["pitch"]["midi"])
        if pitch:
            written = [(s, m) for n in T["pitch"]["score"] for s, m in ((n["start"], n["midi"]), (n["end"], n["midi"]))]
            self._add("Pitch against the written notes", [(written, ft.Colors.OUTLINE, 2.0, True), (pitch, theme.ACCENT, 2.0, False)])
        loud = thin(T["loudness"]["t"], T["loudness"]["db"])
        if loud:
            self._add("Loudness (dB)", [(loud, theme.ACCENT_2, 2.0, False)])
        notes = [(n["start"], n["cents"]) for n in T["notes"] if n.get("cents") is not None]
        if notes:
            self._add("How far each note is from pitch (cents)", [(notes, theme.OK, 2.0, False)], dots=True)
        bow = T.get("bow")
        if bow and bow.get("t"):
            self._add("Bow position", [(thin(bow["t"], bow["pos"]), theme.ACCENT, 2.0, False)])

    def _add(self, title, series, dots=False):
        lo = min(p[1] for s in series for p in s[0])
        hi = max(p[1] for s in series for p in s[0])
        pad = (hi - lo) * 0.08 or 1.0
        lo, hi = lo - pad, hi + pad
        head = fch.LineChartData(points=[fch.LineChartDataPoint(0, lo), fch.LineChartDataPoint(0, hi)], color=theme.BAD, stroke_width=2)
        data = [fch.LineChartData(points=[fch.LineChartDataPoint(x, y) for x, y in pts], color=color, stroke_width=width, curved=False,
                                  dash_pattern=[6, 4] if dashed else None, point=dots) for pts, color, width, dashed in series]
        xs = [p[0] for s in series for p in s[0]]
        chart = fch.LineChart(data_series=data + [head], min_x=min(xs), max_x=max(xs), min_y=lo, max_y=hi, height=170, interactive=False,
                              left_axis=_axis(size=40), bottom_axis=_axis(size=26), horizontal_grid_lines=fch.ChartGridLines(color=ft.Colors.with_opacity(0.1, "#808080")),
                              vertical_grid_lines=None, border=ft.Border.all(0, ft.Colors.TRANSPARENT))
        self.controls.append((title, chart))
        self._heads.append((chart, head, lo, hi))

    def set_time(self, t):
        for chart, head, lo, hi in self._heads:
            head.points = [fch.LineChartDataPoint(t, lo), fch.LineChartDataPoint(t, hi)]
            if mounted(chart):
                chart.update()


class Skeleton:
    """The pose as stick figures (front view left, side view right) on a canvas; `set_time(t)` picks the frame."""
    SIZE = 220

    def __init__(self, pose):
        self.pose, self.joints = pose, {j: i for i, j in enumerate(pose["joints"])}
        pts = [p for frame in pose["xyz"] for p in frame if p]
        self.bounds = tuple((min(p[k] for p in pts), max(p[k] for p in pts)) for k in range(3)) if pts else ((0, 1), (0, 1), (0, 1))
        self.canvas = cv.Canvas(shapes=[], width=self.SIZE * 2 + 20, height=self.SIZE)
        self.frame = -1
        self.set_time(0.0)

    def shapes(self, frame):
        """The canvas shapes for one frame: bones as lines and the hands as dots, front view then side view."""
        out, size = [], self.SIZE
        (x0, x1), (y0, y1), (z0, z1) = self.bounds
        span = max(x1 - x0, y1 - y0, z1 - z0) or 1.0
        pts = self.pose["xyz"][frame]
        views = ((0, x0, 0), (2, z0, size + 20))                 # (axis drawn across, its minimum, left offset)
        bone = ft.Paint(color=theme.ACCENT, stroke_width=3, stroke_cap=ft.StrokeCap.ROUND, style=ft.PaintingStyle.STROKE)
        for axis, lo, left in views:
            def at(p):
                return left + (p[axis] - lo) / span * (size - 20) + 10, size - 10 - (p[1] - y0) / span * (size - 20)
            for a, b in self.pose["edges"]:
                pa, pb = pts[self.joints[a]], pts[self.joints[b]]
                if pa and pb:
                    (xa, ya), (xb, yb) = at(pa), at(pb)
                    out.append(cv.Line(xa, ya, xb, yb, paint=bone))
            for name, color in (("r_hand", theme.OK), ("l_hand", theme.ACCENT_2)):
                p = pts[self.joints[name]] if name in self.joints else None
                if p:
                    out.append(cv.Circle(*at(p), 6, paint=ft.Paint(color=color, style=ft.PaintingStyle.FILL)))
        return out

    def set_time(self, t):
        frame = min(max(0, round(t * self.pose["fps"])), len(self.pose["xyz"]) - 1)
        if frame != self.frame:
            self.frame = frame
            self.canvas.shapes = self.shapes(frame)
            if mounted(self.canvas):
                self.canvas.update()
