# Side-by-side score viewer (new file).
#
# Left: the score as the app read it (score/musicxml.py, after denoising), engraved as sheet music.
# Right: what was detected in the recording (audio/transcription.py), engraved on the *same layout*, so
# every measure and note lines up with its counterpart. gui/notation.py does the engraving; this module
# decides what each note on the right shows, from the alignment and the intonation assessment:
#   in tune / a bit off / out of tune   written note, coloured by cents (green / orange / red), with the
#                                       deviation printed above the staff when it is audible
#   wrong note                          the pitch that was actually played, in red (it sits on a different
#                                       line from the written note on the left)
#   not detected                        dashed outline: written inside the played section but not heard
#   not played                          faint: outside the stretch of the score the recording covers
#   extra note                          purple: heard but not written
# Hovering a note highlights its partner (data-pair; the JS lives in gui/report.py and is loaded by the page).
import numpy as np

from maestro.gui import charts
from maestro.gui.notation import Glyph, layout_score, render_panel

PLAYED = "What was detected in your playing"
WRITTEN = "Score as the app reads it"
BAD_NOTE = "#c0392b"
EXTRA = "#8e44ad"


def _status(cents):
    return "ok" if abs(cents) <= charts.CENTS_GOOD else "warn" if abs(cents) <= charts.CENTS_OK else "bad"


def _score_time_of(perf_t, pairs):
    """Map a time in the recording to score time through the matched notes (monotone piecewise linear)."""
    pt, st = zip(*sorted(pairs))
    return float(np.interp(perf_t, pt, np.maximum.accumulate(st)))


def _extra_notes(notes, played, match):
    """Performed notes that no written note explains -> notes for notation.render_panel(extras=...), placed by
    mapping recording time to score time through the matched notes."""
    used = {j for j in match.values() if j is not None}
    pairs = [(played[j]["onset"], notes[i].onset) for i, j in match.items() if j is not None]
    if not pairs:
        return []
    return [(_score_time_of(p["onset"], pairs), int(round(p["midi"])),
             {"color": EXTRA, "pair": f"p{j}",
              "title": f"extra note {charts.note_name(p['midi'])} at {p['onset']:.1f}s (not in the score)"})
            for j, p in enumerate(played) if j not in used]


def _legend_html():
    swatch = lambda color, text, style="": f'<span><i style="background:{color};{style}"></i>{text}</span>'
    return "".join([swatch(charts.GOOD, f"in tune (within {charts.CENTS_GOOD} cents)"),
                    swatch(charts.OK, f"a bit off (within {charts.CENTS_OK})"), swatch(charts.BAD, "out of tune"),
                    swatch(BAD_NOTE, "wrong note (shown where it was played)"),
                    swatch("white", "not detected", f"border:1.5px dashed {BAD_NOTE}"), swatch(EXTRA, "extra note"),
                    swatch("#ccc", "not played")])


def build_viewer(perf):
    """Perf (analyze() result) -> HTML: the two engraved panels, a legend and a one-line summary."""
    score, played = perf.score, perf.audio_features["notes"]
    notes = score.notes
    layout = layout_score(notes, score.key_fifths, score.time_signature)
    match = {a["score_idx"]: a["perf_idx"] for a in perf.alignment}
    inton = {n["score_idx"]: n for n in perf.assessment["intonation"]["notes"]}
    sec = perf.assessment.get("section")
    first, last = (sec["first_note"], sec["last_note"]) if sec else (0, len(notes) - 1)
    counts = {"ok": 0, "warn": 0, "bad": 0, "wrong": 0, "missed": 0, "unplayed": 0}

    def written(i):
        n = notes[i]
        return Glyph(n.midi_pitch, pair=f"s{i}", status="written", title=f"{n.pitch}, bar {n.measure}, beat {n.beat:g}")

    def detected(i):
        n, j = notes[i], match.get(i)
        base = dict(pair=f"s{i}")
        if j is None:
            if first <= i <= last:
                counts["missed"] += 1
                return Glyph(n.midi_pitch, color=BAD_NOTE, dashed=True, status="missed", label="?", label_color=BAD_NOTE,
                             title=f"{n.pitch} (bar {n.measure}) was not detected", **base)
            counts["unplayed"] += 1
            return Glyph(n.midi_pitch, color="#999", opacity=0.25, status="unplayed",
                         title=f"{n.pitch} (bar {n.measure}): outside the part you played", **base)
        d = inton[i]
        if d["wrong_note"]:
            counts["wrong"] += 1
            return Glyph(int(round(played[j]["midi"])), color=BAD_NOTE, status="wrong", label="wrong", label_color=BAD_NOTE,
                         title=f"written {n.pitch}, played {charts.note_name(played[j]['midi'])} (bar {n.measure})", **base)
        st = _status(d["cents"])
        counts[st] += 1
        color = {"ok": charts.GOOD, "warn": charts.OK, "bad": charts.BAD}[st]
        return Glyph(n.midi_pitch, color=color, status=st, label=("" if st == "ok" else f"{d['cents']:+.0f}c"),
                     label_color=color, title=f"{n.pitch}: {d['cents']:+.0f} cents (bar {n.measure})", **base)

    extras = _extra_notes(notes, played, match)
    left = render_panel(layout, written, WRITTEN)
    right = render_panel(layout, detected, PLAYED, extras)          # fills `counts` as a side effect
    summary = (f"{len(notes)} written notes: {counts['ok'] + counts['warn'] + counts['bad']} matched "
               f"({counts['ok']} in tune), {counts['wrong']} wrong, {counts['missed']} not detected, "
               f"{counts['unplayed']} outside the part you played; {len(extras)} extra note{'s' if len(extras) != 1 else ''} detected.")
    return (f'<div class="viewer"><div class="vcol">{left}</div><div class="vcol">{right}</div></div>'
            f'<div class="vlegend">{_legend_html()}</div><p class="vsummary">{summary}</p>')
