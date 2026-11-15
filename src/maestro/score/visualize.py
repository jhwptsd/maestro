# Basic score visualization (new file): piano-roll SVG of the score, optionally annotated with how
# the student's pitch deviated.
#
# Reads the Score from score/musicxml.py and optionally the "intonation" result from
# assessment/intonation.py. Dependency-free (writes SVG text); useful to sanity-check OMR output
# and to show feedback to the user.
from pathlib import Path

from maestro.score.score_representation import NOTE_NAMES


def _color(cents):
    """Green = in tune (<15 cents), orange = <30, red = worse."""
    c = abs(cents)
    return "#2a9d3f" if c < 15 else "#f08c00" if c < 30 else "#d62828"


def score_to_svg(score, intonation=None, path=None, px_per_sec=80, row_h=12):
    """Render notes as bars (x = time, y = pitch) with measure lines; return the SVG string
    (also written to `path` if given).

    If `intonation` is given, a marker is drawn on each aligned note at the pitch actually played
    (relative to the bar: above = sharp, below = flat; 1 semitone = one row), coloured by size.
    """
    notes = score.notes
    if not notes:
        raise ValueError("empty score")
    lo, hi = min(n.midi_pitch for n in notes) - 1, max(n.midi_pitch for n in notes) + 1
    left, top = 40, 20
    width = left + (max(n.offset for n in notes)) * px_per_sec + 20
    height = top + (hi - lo + 1) * row_h + 20
    y = lambda midi: top + (hi - midi) * row_h

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
           f'font-family="sans-serif" font-size="9">', '<rect width="100%" height="100%" fill="white"/>']
    for midi in range(lo, hi + 1):               # pitch grid; label the C's
        out.append(f'<line x1="{left}" x2="{width:.0f}" y1="{y(midi) + row_h / 2}" y2="{y(midi) + row_h / 2}" stroke="#eee"/>')
        if midi % 12 == 0:
            out.append(f'<text x="2" y="{y(midi) + row_h - 2}">{NOTE_NAMES[0]}{midi // 12 - 1}</text>')
    seen = set()
    for n in notes:                              # measure start lines
        if n.measure not in seen:
            seen.add(n.measure)
            x = left + n.onset * px_per_sec
            out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height - 20}" stroke="#bbb"/>'
                       f'<text x="{x + 2:.1f}" y="{top - 6}">{n.measure}</text>')
    for n in notes:                              # the notes themselves
        out.append(f'<rect x="{left + n.onset * px_per_sec:.1f}" y="{y(n.midi_pitch) + 1}" '
                   f'width="{max((n.offset - n.onset) * px_per_sec - 1, 2):.1f}" height="{row_h - 2}" '
                   f'fill="#4a6fa5"><title>{n.pitch}</title></rect>')
    for d in (intonation or {}).get("notes", []):  # played-pitch markers
        n = notes[d["score_idx"]]
        yy = y(n.midi_pitch + d["cents"] / 100) + row_h / 2
        x0, x1 = left + n.onset * px_per_sec, left + n.offset * px_per_sec
        out.append(f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" '
                   f'stroke="{_color(d["cents"])}" stroke-width="3"><title>{d["cents"]:+.0f} cents</title></line>')
    out.append("</svg>")
    svg = "\n".join(out)
    if path:
        Path(path).write_text(svg)
    return svg
