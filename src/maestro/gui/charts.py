# SVG charts for the analysis results (new file; dependency-free, returns SVG strings).
#
# One function per result produced by maestro.analyze():
#   pitch_trace_chart   <- audio_features["frames"] + transcribed notes + alignment  (what was played)
#   intonation_chart    <- assessment["intonation"]["notes"]  (cents off per note)
#   tempo_chart         <- assessment["rhythm"] (+ score/perf onsets)  (overall tempo, "tempo_ratio")
#   timing_chart        <- assessment["rhythm"]["notes"]  (ms early/late per note)
#   match_chart         <- assessment["correspondence"]  (does the score fit the recording)
# Visual analysis (vision/ + assessment/; none of it uses audio):
#   bow_chart           <- motion_features["bow"]  (right-hand travel along the bow, one colour per direction)
#   stroke_chart        <- assessment["bowing"]["strokes"]  (length of each bow stroke)
#   arm_chart           <- motion_features["elbow_angle"]  (elbow angles over time)
#   visual_pitch_chart  <- assessment["visual_pitch"]  (pitch implied by the left hand's position)
# Assessment events (assessment/synthesize.py, the single JSON report assessment["synthesis"]):
#   module_scores_chart <- synthesis["scores"]  (0-100 per module and overall)
#   event_timeline_chart <- synthesis["events"]  (one lane per module, one block per event, colour = severity)
#   cause_chart         <- synthesis["causes"]  (what most often explains the events)
#   measure_heatmap     <- synthesis["by_measure"]  (worst event per module per measure)
# The annotated score itself comes from score/visualize.py. gui/report.py arranges all of these with
# explanations; gui/server.py serves that report. The pitch/timing thresholds come from the assessment
# modules, so the colours on every chart and the grades on the summary cards always agree.
from html import escape

import numpy as np

from maestro.assessment.intonation import CENTS_GOOD, CENTS_OK
from maestro.assessment.rhythm import MS_GOOD, MS_OK
from maestro.score.score_representation import NOTE_NAMES
from maestro.utils.audio_math import hz_to_midi

GOOD, OK, BAD, GREY, BLUE, ORANGE = "#2a9d3f", "#f08c00", "#d62828", "#999", "#4a6fa5", "#e76f00"


def grade_color(value, good, ok):
    """Colour for an absolute error: green within `good`, orange within `ok`, else red."""
    v = abs(value)
    return GOOD if v <= good else OK if v <= ok else BAD


def note_name(midi):
    midi = int(round(midi))
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def _ticks(lo, hi, n=6):
    """~n round tick values covering [lo, hi]."""
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return [float(v) for v in np.arange(np.ceil(lo / step) * step, hi + step / 2, step)]


class _Plot:
    """Minimal axes: data -> pixel mapping, grid, ticks, labels, legend."""
    L, R, T, B = 64, 16, 32, 46

    def __init__(self, title, xlim, ylim, xlabel, ylabel, w=760, h=300):
        self.w, self.h, self.xlim, self.ylim = w, h, xlim, ylim
        self.p = [f'<rect width="{w}" height="{h}" fill="white"/>',
                  f'<text x="{w / 2}" y="19" text-anchor="middle" font-size="14" font-weight="bold">{escape(title)}</text>',
                  f'<text x="{(self.L + w - self.R) / 2}" y="{h - 8}" text-anchor="middle">{escape(xlabel)}</text>',
                  f'<text transform="translate(14 {(self.T + h - self.B) / 2}) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>']

    def x(self, v):
        return self.L + (v - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (self.w - self.L - self.R)

    def y(self, v):
        return self.h - self.B - (v - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (self.h - self.T - self.B)

    def grid(self, xticks=(), yticks=(), xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}"):
        for v in yticks:
            self.p.append(f'<line x1="{self.L}" x2="{self.w - self.R}" y1="{self.y(v):.1f}" y2="{self.y(v):.1f}" stroke="#e8e8e8"/>'
                          f'<text x="{self.L - 6}" y="{self.y(v) + 3:.1f}" text-anchor="end">{escape(yfmt(v))}</text>')
        for v in xticks:
            self.p.append(f'<line x1="{self.x(v):.1f}" x2="{self.x(v):.1f}" y1="{self.T}" y2="{self.h - self.B}" stroke="#f0f0f0"/>'
                          f'<text x="{self.x(v):.1f}" y="{self.h - self.B + 15}" text-anchor="middle">{escape(xfmt(v))}</text>')
        self.p.append(f'<rect x="{self.L}" y="{self.T}" width="{self.w - self.L - self.R}" '
                      f'height="{self.h - self.T - self.B}" fill="none" stroke="#bbb"/>')

    def band(self, y0, y1, color, opacity=0.12):
        self.p.append(f'<rect x="{self.L}" y="{self.y(y1):.1f}" width="{self.w - self.L - self.R}" '
                      f'height="{self.y(y0) - self.y(y1):.1f}" fill="{color}" opacity="{opacity}"/>')

    def line(self, x0, y0, x1, y1, color, width=1.5, dash=None, tip=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        t = f"<title>{escape(tip)}</title>" if tip else ""
        self.p.append(f'<line x1="{self.x(x0):.1f}" y1="{self.y(y0):.1f}" x2="{self.x(x1):.1f}" y2="{self.y(y1):.1f}" '
                      f'stroke="{color}" stroke-width="{width}"{d}>{t}</line>')

    def rect(self, x0, x1, y0, y1, color, tip=""):
        top, bot = sorted((self.y(y0), self.y(y1)))
        self.p.append(f'<rect x="{self.x(x0):.1f}" y="{top:.1f}" width="{max(self.x(x1) - self.x(x0), 1):.1f}" '
                      f'height="{max(bot - top, 1):.1f}" fill="{color}"><title>{escape(tip)}</title></rect>')

    def dot(self, x, y, color, tip=""):
        self.p.append(f'<circle cx="{self.x(x):.1f}" cy="{self.y(y):.1f}" r="4" fill="{color}" stroke="white">'
                      f'<title>{escape(tip)}</title></circle>')

    def text(self, x, y, s, color="#333", anchor="start"):
        self.p.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" text-anchor="{anchor}">{escape(s)}</text>')

    def legend(self, items, bottom=False):
        """items: [(colour, label)] stacked at the top right (or bottom right) inside the plot."""
        for k, (c, label) in enumerate(items):
            yy = (self.h - self.B - 14 * len(items) + 14 * k + 6) if bottom else self.T + 14 + 14 * k
            self.p.append(f'<rect x="{self.w - self.R - 205}" y="{yy - 8}" width="12" height="8" fill="{c}"/>')
            self.text(self.w - self.R - 188, yy, label)

    def svg(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="100%" '
                f'font-family="sans-serif" font-size="11">' + "".join(self.p) + "</svg>")


SEVERITY_COLORS = ((0.34, GOOD), (0.67, OK), (1.01, BAD))


def severity_color(severity):
    """Green / orange / red for an event severity (0-1), the same scale on every assessment chart."""
    return next(c for limit, c in SEVERITY_COLORS if severity < limit)


def _empty(title, message):
    p = _Plot(title, (0, 1), (0, 1), "", "", h=120)
    p.text(p.w / 2, 70, message, GREY, "middle")
    return p.svg()


def pitch_trace_chart(frames, perf_notes, score_features, alignment):
    """What was played vs what was written: grey = tracked pitch, blue bars = the written pitch of each
    matched note placed where the matching performed note sounded. The vertical gap is the pitch error."""
    pts = [(f["time"], float(hz_to_midi(f["estimated_pitch"]))) for f in frames if f["estimated_pitch"] > 0]
    if not pts:
        return _empty("Pitch over time", "No pitch could be tracked.")
    t_end = max(f["time"] for f in frames)
    written = [(perf_notes[a["perf_idx"]], score_features[a["score_idx"]]["midi_pitch"])
               for a in alignment if a["perf_idx"] is not None]
    allm = [m for _, m in pts] + [m for _, m in written]
    lo, hi = int(np.floor(min(allm))) - 1, int(np.ceil(max(allm))) + 1
    p = _Plot("Pitch over time: played vs written", (0, t_end), (lo, hi), "time in recording (s)", "pitch")
    step = 1 if hi - lo <= 18 else 2 if hi - lo <= 30 else 12
    p.grid(_ticks(0, t_end), range(lo, hi + 1, step), yfmt=note_name)
    for n, midi in written:
        p.line(n["onset"], midi, n["offset"], midi, BLUE, 5, tip=f"written {note_name(midi)}")
    stride, prev = max(1, len(frames) // 1500), None      # grey trace; break the line at unvoiced gaps
    for f in frames[::stride]:
        cur = (f["time"], float(hz_to_midi(f["estimated_pitch"]))) if f["estimated_pitch"] > 0 else None
        if cur and prev and cur[0] - prev[0] < 0.05:
            p.line(prev[0], prev[1], cur[0], cur[1], "#555", 1.5)
        prev = cur
    p.legend([("#555", "pitch you played"), (BLUE, "pitch written in the score")])
    return p.svg()


def intonation_chart(intonation):
    """Bar per note: how many cents sharp (up) or flat (down). Grey = wrong note, not an intonation error."""
    notes = intonation["notes"]
    if not notes:
        return _empty("Intonation per note", "No notes were matched to the score.")
    top = max(50, min(100, 25 * np.ceil(max(abs(n["cents"]) for n in notes) / 25)))
    idx = [n["score_idx"] for n in notes]
    p = _Plot("Intonation per note (cents sharp + / flat −)", (min(idx) - 1, max(idx) + 1), (-top, top),
              "note number in the score", "cents")
    p.grid(_ticks(min(idx), max(idx), 8), _ticks(-top, top), xfmt=lambda v: f"{v:.0f}")
    p.band(-CENTS_GOOD, CENTS_GOOD, GOOD)
    for n in notes:
        c = float(np.clip(n["cents"], -top, top))
        color = GREY if n["wrong_note"] else grade_color(n["cents"], CENTS_GOOD, CENTS_OK)
        label = "wrong note" if n["wrong_note"] else f"{n['cents']:+.0f} cents"
        p.rect(n["score_idx"] - 0.35, n["score_idx"] + 0.35, 0, c, color, f"note {n['score_idx']}: {label}")
    if intonation["tuning_offset_cents"] is not None:
        o = intonation["tuning_offset_cents"]
        p.line(min(idx) - 1, o, max(idx) + 1, o, BLUE, 1.5, "5 4", f"overall tuning {o:+.0f} cents")
    p.legend([(GOOD, f"within ±{CENTS_GOOD} cents"), (OK, f"within ±{CENTS_OK}"), (BAD, "further off"),
              (GREY, "wrong note"), (BLUE, "your average tuning")])
    return p.svg()


def timing_chart(rhythm):
    """Bar per note: how many milliseconds late (up) or early (down) after removing the overall tempo."""
    notes = rhythm["notes"]
    if not notes:
        return _empty("Timing per note", "Not enough matched notes to judge timing.")
    ms = [n["error"] * 1000 for n in notes]
    top = max(150, 50 * np.ceil(max(abs(v) for v in ms) / 50))
    idx = [n["score_idx"] for n in notes]
    p = _Plot("Timing per note (ms late + / early −)", (min(idx) - 1, max(idx) + 1), (-top, top),
              "note number in the score", "milliseconds")
    p.grid(_ticks(min(idx), max(idx), 8), _ticks(-top, top), xfmt=lambda v: f"{v:.0f}")
    p.band(-MS_GOOD, MS_GOOD, GOOD)
    for n, v in zip(notes, ms):
        p.rect(n["score_idx"] - 0.35, n["score_idx"] + 0.35, 0, v, grade_color(v, MS_GOOD, MS_OK),
               f"note {n['score_idx']}: {v:+.0f} ms")
    p.legend([(GOOD, f"within ±{MS_GOOD} ms"), (OK, f"within ±{MS_OK} ms"), (BAD, "further off")])
    return p.svg()


def tempo_chart(score_features, perf_notes, rhythm):
    """Score time vs performed time of every matched note. Slope of the fitted line = your tempo
    relative to the written one; dots off the line are notes played early/late."""
    pairs = [(score_features[n["score_idx"]]["onset"], perf_notes[n["perf_idx"]]["onset"], n["error"])
             for n in rhythm["notes"]]
    if len(pairs) < 2:
        return _empty("Tempo", "Not enough matched notes to estimate tempo.")
    so, po, _ = (np.array(c) for c in zip(*pairs))
    hi = float(max(so.max(), po.max())) * 1.05
    p = _Plot("Tempo: written time vs played time", (0, hi), (0, hi), "time in the score at its written tempo (s)",
              "time when you played the note (s)")
    p.grid(_ticks(0, hi), _ticks(0, hi))
    p.line(0, 0, hi, hi, GREY, 1.5, "5 4")
    a, b = np.polyfit(so, po, 1)
    p.line(0, b, hi, a * hi + b, ORANGE, 2)
    for s, o, e in pairs:
        p.dot(s, o, grade_color(e * 1000, MS_GOOD, MS_OK), f"{e * 1000:+.0f} ms")
    p.legend([(GREY, "playing exactly as written"), (ORANGE, f"your tempo: {rhythm['tempo_ratio']:.0%} of written")], bottom=True)
    return p.svg()


def match_chart(corr, threshold=0.6):
    """Two bars: share of written notes heard in the recording, and of played notes explained by the score."""
    p = _Plot("Does the recording match the score?", (0, 1), (0, 3), "", "", h=150)
    for k, (label, v) in enumerate((("written notes heard", corr["score_match"]),
                                    ("played notes explained", corr["perf_match"]))):
        y0 = 2.55 - k * 1.2
        p.rect(0, 1, y0 - 0.3, y0 + 0.3, "#eee")
        p.rect(0, v, y0 - 0.3, y0 + 0.3, GOOD if v >= threshold else BAD, f"{v:.0%}")
        p.text(p.x(0) + 6, p.y(y0) + 4, f"{label}: {v:.0%}", "white" if v > 0.35 else "#333")
    p.line(threshold, 0.1, threshold, 2.95, "#333", 1.5, "4 3", "minimum to accept")
    p.text(p.x(threshold), p.h - p.B + 14, f"minimum {threshold:.0%}", "#333", "middle")
    return p.svg()


# ----------------------------------------------------------------------------- visual analysis

def _polyline(p, xs, ys, color, width=1.5):
    """Connect consecutive points, downsampled to at most ~1500 segments."""
    step = max(1, len(xs) // 1500)
    xs, ys = xs[::step], ys[::step]
    for k in range(len(xs) - 1):
        p.line(xs[k], ys[k], xs[k + 1], ys[k + 1], color, width)


def bow_chart(motion):
    """Bow travel over time: right-hand position along the bow axis, one colour per stroke direction."""
    bow = motion.get("bow")
    if not bow or not bow["strokes"]:
        return _empty("Bow movement", "No bow strokes were found.")
    t, s = motion["times"], bow["s"]
    lo, hi = float(np.min(s)), float(np.max(s))
    pad = 0.1 * (hi - lo or 1)
    p = _Plot("Bow movement over time", (0, float(t[-1])), (lo - pad, hi + pad), "time in recording (s)",
              "right-hand travel along the bow (shoulder widths)")
    p.grid(_ticks(0, float(t[-1])), _ticks(lo - pad, hi + pad, 5))
    for st in bow["strokes"]:
        a, b = int(st["start"] * motion["fps"]), int(st["end"] * motion["fps"]) + 1
        _polyline(p, t[a:b], s[a:b], BLUE if st["dir"] > 0 else ORANGE, 2)
    markers = bow["axis_source"] == "markers"
    p.legend([(BLUE, "down-bow" if markers else "stroke one way"), (ORANGE, "up-bow" if markers else "stroke the other way")])
    return p.svg()


def stroke_chart(bowing):
    """Bar per bow stroke: how far the hand travelled. Colour = direction."""
    strokes = bowing.get("strokes") or []
    if not strokes:
        return _empty("Bow stroke length", "No bow strokes were found.")
    top = max(s["length"] for s in strokes) * 1.15
    p = _Plot("Length of each bow stroke", (0, len(strokes) + 1), (0, top), "stroke number", "bow travel (shoulder widths)")
    p.grid(_ticks(1, len(strokes), 8), _ticks(0, top, 5), xfmt=lambda v: f"{v:.0f}")
    for k, st in enumerate(strokes, start=1):
        p.rect(k - 0.35, k + 0.35, 0, st["length"], BLUE if st["dir"] > 0 else ORANGE,
               f"stroke {k}: {st['length']:.2f}, {st['mean_speed']:.2f}/s")
    m = bowing["mean_stroke_length"]
    p.line(0, m, len(strokes) + 1, m, "#333", 1.5, "5 4", f"mean {m:.2f}")
    p.legend([(BLUE, "one direction"), (ORANGE, "the other"), ("#333", "average length")])
    return p.svg()


def arm_chart(motion):
    """Elbow angle (shoulder-elbow-wrist) of both arms over time."""
    angles = motion.get("elbow_angle") or {}
    if not angles:
        return _empty("Elbow angles", "Shoulder, elbow and wrist joints were not tracked.")
    t = motion["times"]
    allv = np.concatenate([a[np.isfinite(a)] for a in angles.values()])
    lo, hi = float(allv.min()) - 5, float(allv.max()) + 5
    p = _Plot("Elbow angles over time", (0, float(t[-1])), (lo, hi), "time in recording (s)", "elbow angle (degrees)")
    p.grid(_ticks(0, float(t[-1])), _ticks(lo, hi, 5))
    items = []
    for side, color in (("left", BLUE), ("right", ORANGE)):
        if side in angles:
            _polyline(p, t, angles[side], color, 1.8)
            items.append((color, f"{side} elbow ({'fingering' if side == 'left' else 'bowing'} arm)"))
    p.legend(items)
    return p.svg()


def visual_pitch_chart(vp):
    """Pitch implied by the left hand's position, from video alone: semitones above the open string over time.
    On the A string 0 = A4, +7 = E5, +12 = A5; the string itself cannot be seen."""
    if not vp.get("available"):
        return _empty("Left-hand position as pitch", vp.get("reason", "Not available."))
    t, s = vp["times"], vp["semitones_above_open"]
    lo, hi = float(np.floor(min(s.min(), 0))) - 1, float(np.ceil(s.max())) + 1
    p = _Plot("Pitch implied by left-hand position (video only)", (0, float(t[-1])), (lo, hi), "time in recording (s)",
              "semitones above the open string")
    p.grid(_ticks(0, float(t[-1])), _ticks(lo, hi, 7))
    p.line(0, 0, float(t[-1]), 0, GREY, 1.2, "4 3", "open string")
    _polyline(p, t, s, BLUE, 2)
    p.legend([(BLUE, "estimated from hand position"), (GREY, "open string (0)")])
    return p.svg()


MODULE_ORDER = ("intonation", "rhythm", "expression", "tone", "bowing", "posture")


def module_scores_chart(scores):
    """Horizontal bar per assessed module (0-100, 100 = no events) plus the overall score."""
    items = [(m, v) for m, v in scores.items() if v is not None]
    if not items:
        return _empty("Scores", "Nothing could be assessed.")
    p = _Plot("Score per skill (100 = nothing to correct)", (0, 100), (0, len(items)), "score", "", h=60 + 34 * len(items))
    p.grid(_ticks(0, 100, 5), ())
    for k, (name, v) in enumerate(items):
        p.rect(0, v, len(items) - k - 0.8, len(items) - k - 0.2, severity_color(1 - v / 100), f"{name}: {v:.0f}")
        p.text(p.L - 6, p.y(len(items) - k - 0.45) + 3, name, anchor="end")
        p.text(p.x(v) + 4, p.y(len(items) - k - 0.45) + 3, f"{v:.0f}")
    return p.svg()


def event_timeline_chart(events):
    """One lane per module over time; every event is a block coloured by severity (hover: what and why)."""
    lanes = [m for m in MODULE_ORDER if any(e["module"] == m for e in events)]
    if not lanes:
        return _empty("Events over time", "No problems were found.")
    end = max(e["end"] for e in events) + 0.5
    p = _Plot("When things went wrong", (0, end), (0, len(lanes)), "time in the recording (s)", "", h=70 + 36 * len(lanes))
    p.grid(_ticks(0, end, 8), ())
    for k, lane in enumerate(lanes):
        y0 = len(lanes) - k - 1
        p.text(p.L - 6, p.y(y0 + 0.5) + 3, lane, anchor="end")
        for e in events:
            if e["module"] == lane:
                why = e["causes"][0]["explanation"] if e["causes"] else ""
                bar = max(e["end"], e["start"] + end / 150)
                p.rect(e["start"], bar, y0 + 0.15, y0 + 0.85, severity_color(e["severity"]),
                       f"{e['category']} (severity {e['severity']:.2f}): {e['message']} {why}")
    p.legend([(GOOD, "minor"), (OK, "moderate"), (BAD, "severe")], bottom=True)
    return p.svg()


def cause_chart(causes, top=8):
    """Bars for the causes that explain the most (severity x confidence) of what went wrong."""
    causes = causes[:top]
    if not causes:
        return _empty("Most likely causes", "No problems were found.")
    peak = max(c["weight"] for c in causes) * 1.15
    p = _Plot("Most likely causes", (0, peak), (0, len(causes)), "weight (sum of severity x confidence)", "", h=70 + 28 * len(causes))
    p.L = 190
    p.grid(_ticks(0, peak, 5), ())
    for k, c in enumerate(causes):
        y0 = len(causes) - k - 1
        p.rect(0, c["weight"], y0 + 0.15, y0 + 0.85, BLUE, f"{c['cause']}: {c['count']} events ({', '.join(c['modules'])})")
        p.text(p.L - 6, p.y(y0 + 0.5) + 3, c["cause"].replace("_", " "), anchor="end")
        p.text(p.x(c["weight"]) + 4, p.y(y0 + 0.5) + 3, f"x{c['count']}")
    return p.svg()


def measure_heatmap(events):
    """Grid of module x measure; the colour is the worst event there. Shows which bars to practise, and what at."""
    cells = {}
    for e in events:
        if e["measure"] is not None:
            key = (e["module"], e["measure"])
            cells[key] = max(cells.get(key, 0), e["severity"])
    if not cells:
        return _empty("Problem bars", "No problems were found.")
    measures = sorted({m for _, m in cells})
    lanes = [m for m in MODULE_ORDER if any(k[0] == m for k in cells)]
    p = _Plot("Which bars need work", (min(measures) - 0.5, max(measures) + 0.5), (0, len(lanes)), "measure", "",
              h=70 + 34 * len(lanes))
    p.grid(_ticks(min(measures), max(measures), 10), (), xfmt=lambda v: f"{v:.0f}")
    for k, lane in enumerate(lanes):
        y0 = len(lanes) - k - 1
        p.text(p.L - 6, p.y(y0 + 0.5) + 3, lane, anchor="end")
        for (m, bar), sev in cells.items():
            if m == lane:
                p.rect(bar - 0.45, bar + 0.45, y0 + 0.1, y0 + 0.9, severity_color(sev), f"bar {bar}, {lane}: severity {sev:.2f}")
    return p.svg()
