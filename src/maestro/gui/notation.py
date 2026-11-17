# Sheet-music engraver (new file; dependency-free, returns SVG strings).
#
# Draws a Score as standard notation: treble staves in systems, clef, key and time signature, noteheads,
# stems, flags and beams, accidentals, ledger lines, dots, rests, bar lines and measure numbers.
# gui/viewer.py uses it twice, for the score as the app read it (score/musicxml.py -> Score) and for what
# was detected in the recording. `layout_score` fixes where every measure and note sits *from the score
# alone*, so both renderings have identical geometry and line up note for note; `render_panel` draws
# that layout with whatever pitches/colours the caller supplies per note (`Glyph`).
#
# Scope: single treble-clef line (violin); note values by nearest standard duration (tuplets are drawn as
# the nearest plain value); beams only over runs of single notes inside a beat; no slurs, ties or dynamics.
import math
from dataclasses import dataclass
from html import escape
from typing import Callable, Dict, List, Optional, Tuple

SP = 8.0                                   # staff space in px (line to line)
LETTER_PC = [0, 2, 4, 5, 7, 9, 11]         # C D E F G A B
SHARP_ORDER = [3, 0, 4, 1, 5, 2, 6]        # F C G D A E B (letter indices) for sharps
FLAT_ORDER = [6, 2, 5, 1, 4, 0, 3]         # B E A D G C F
TOP_LINE_POS, BOTTOM_LINE_POS, MIDDLE_POS = 38, 30, 34   # diatonic positions of F5 / E4 / B4 (C4 = 28)
VALUES = [(4.0, "whole", 0), (3.0, "half", 1), (2.0, "half", 0), (1.5, "quarter", 1), (1.0, "quarter", 0),
          (0.75, "eighth", 1), (0.5, "eighth", 0), (0.375, "16th", 1), (0.25, "16th", 0), (0.125, "32nd", 0)]
REST_VALUES = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125]
INK = "#222"
CLEF_W = 38                                # width reserved for the clef (px)
HEAD_RX, HEAD_RY = SP * 0.62, SP * 0.46    # notehead radii (the head is drawn tilted by -20 degrees)
STEM_LEN = 3.5 * SP                        # standard stem length from the note at the stem's free end


def _y(top, pos):
    """y of diatonic staff position `pos` on a staff whose top line is at `top`."""
    return top + (TOP_LINE_POS - pos) * SP / 2


# ----------------------------------------------------------------------------- pitch spelling

def infer_key(midis):
    """Key signature (fifths, -7..7) whose major scale needs the fewest accidentals for these pitches."""
    major = (0, 2, 4, 5, 7, 9, 11)
    return min(range(-7, 8), key=lambda f: (sum((m - f * 7) % 12 not in major for m in midis), abs(f)))


def key_alters(fifths):
    """Alteration (-1/0/+1) the key signature applies to each letter C..B."""
    alt = [0] * 7
    for letter in (SHARP_ORDER[:fifths] if fifths > 0 else FLAT_ORDER[:-fifths]):
        alt[letter] = 1 if fifths > 0 else -1
    return alt


def spell(midi, fifths):
    """MIDI -> (staff position, alteration) spelled the way `fifths` would: spellings that match the key
    signature first, then natural, then sharps in sharp keys / flats in flat keys.
    Position = 7 * octave + letter index (C4 = 28)."""
    in_key = key_alters(fifths)
    pc, octave = midi % 12, midi // 12 - 1
    best = None
    for letter, base in enumerate(LETTER_PC):
        for alt in (0, 1, -1):
            if (base + alt) % 12 != pc:
                continue
            cost = (alt != in_key[letter], 0 if alt == 0 else 0.1 if (alt > 0) == (fifths >= 0) else 0.5)
            spelled_octave = octave + (base + alt >= 12) - (base + alt < 0)       # B# is in the next octave, Cb in the previous
            if best is None or cost < best[0]:
                best = (cost, 7 * spelled_octave + letter, alt)
    return best[1], best[2]


# ----------------------------------------------------------------------------- layout

@dataclass
class Slot:
    kind: str                       # "notes" | "rest"
    x: float = 0.0                  # left edge inside the system (set during layout)
    w: float = 0.0
    dur: float = 0.0                # quarters (rests: the rest's value)
    notes: Tuple[int, ...] = ()     # indices into the score's note list, lowest pitch first
    start_q: float = 0.0            # position inside the measure, quarters
    start_sec: float = 0.0          # score time of the slot (for placing extra notes)


@dataclass
class Measure:
    number: int
    slots: List[Slot]
    x: float = 0.0
    w: float = 0.0
    last: bool = False


@dataclass
class System:
    measures: List[Measure]
    y: float = 0.0                  # y of the top staff line
    height: float = 0.0
    header_w: float = 0.0           # clef + key (+ time) signature width


@dataclass
class Layout:
    systems: List[System]
    width: float
    height: float
    fifths: int
    time_sig: Optional[Tuple[int, int]]
    notes: list
    head_x: Dict[int, float]        # note index -> notehead x


def _note_value(dur):
    """Nearest standard value: (quarters, name, dots)."""
    return min(VALUES, key=lambda v: abs(v[0] - dur))


def _split_measures(notes):
    """Group the (onset-sorted) notes into measure occurrences: a new group when the measure number
    changes or the beat position moves backwards (an unrolled repeat)."""
    groups, prev = [], None
    for i, n in enumerate(notes):
        if prev is None or n.measure != prev.measure or n.beat < prev.beat - 1e-9:
            groups.append([])
        groups[-1].append(i)
        prev = n
    return groups


def _rest_slots(gap, at):
    """Rests filling `gap` quarters starting at `at`, greedily from the largest standard value."""
    out = []
    for v in REST_VALUES:
        while gap >= v - 1e-6:
            out.append(Slot("rest", dur=v, start_q=at))
            gap, at = gap - v, at + v
    return out


def _measure_slots(notes, group, beat_type):
    """Slots (note groups at one onset, plus rests for silences between them) of one measure."""
    pos = lambda i: (notes[i].beat - 1.0) * 4.0 / beat_type             # position in the measure, quarters
    by_onset = {}
    for i in group:
        by_onset.setdefault(round(pos(i), 6), []).append(i)
    onsets = sorted(by_onset)
    slots, cursor = [], 0.0
    for on in onsets:
        if on > cursor + 1e-6 and slots:                                # silence before this onset
            slots += _rest_slots(on - cursor, cursor)
        members = sorted(by_onset[on], key=lambda i: notes[i].midi_pitch)
        dur = max(notes[i].duration for i in members)
        slots.append(Slot("notes", dur=dur, notes=tuple(members), start_q=on, start_sec=notes[members[0]].onset))
        cursor = on + dur
    return slots


def layout_score(notes, fifths=None, time_sig=None, width=560.0):
    """Notes (score_representation.Note list, sorted by onset) -> Layout (see the module comment)."""
    beat_type = time_sig[1] if time_sig else 4
    if fifths is None:
        fifths = infer_key([n.midi_pitch for n in notes])
    measures = [Measure(notes[g[0]].measure, _measure_slots(notes, g, beat_type)) for g in _split_measures(notes)]
    if measures:
        measures[-1].last = True

    # natural widths: more room for longer notes (spacing ~ duration^0.6), extra for dots
    for m in measures:
        for s in m.slots:
            dotted = s.kind == "notes" and any(_note_value(notes[i].duration)[2] for i in s.notes)
            s.w = 20 + 22 * (s.dur ** 0.6) + (6 if dotted else 0)
        m.w = 14 + sum(s.w for s in m.slots)

    positions = [spell(n.midi_pitch, fifths)[0] for n in notes] or [MIDDLE_POS]
    above = 12 + max(0, max(positions) - TOP_LINE_POS) / 2 * SP + 3.0 * SP
    below = 8 + max(0, BOTTOM_LINE_POS - min(positions)) / 2 * SP + 3.0 * SP
    key_w = CLEF_W + 7 * abs(fifths) + 12
    first_header = key_w + (18 if time_sig else 0)

    rows, row, row_w = [], [], 0.0                                      # break into systems
    for m in measures:
        avail = width - 8 - (first_header if not rows else key_w)
        if row and row_w + m.w > avail:
            rows.append(row)
            row, row_w = [], 0.0
        row.append(m)
        row_w += m.w
    if row:
        rows.append(row)

    systems, y, head_x = [], 4.0, {}
    for si, row in enumerate(rows):
        header = first_header if si == 0 else key_w
        avail = width - 8 - header
        natural = sum(m.w for m in row)
        # stretch the line to the full width, except a short last line, and never by more than 60%
        scale = min(avail / natural if (si < len(rows) - 1 or natural > 0.7 * avail) else 1.0, 1.6)
        sys_ = System(row, y=y + above, height=above + 4 * SP + below, header_w=header)
        x = header
        for m in row:
            m.x, m.w = x, m.w * scale
            sx = m.x + 10 * scale
            for s in m.slots:
                s.x, s.w = sx, s.w * scale
                for i in s.notes:
                    head_x[i] = sx + 9
                sx += s.w
            x += m.w
        systems.append(sys_)
        y += sys_.height
    return Layout(systems, width, y + 4, fifths, time_sig, notes, head_x)


# ----------------------------------------------------------------------------- glyph drawing

@dataclass
class Glyph:
    """How to draw one score note in one panel."""
    midi: int                       # pitch to display (sets the staff position and accidental)
    color: str = INK
    opacity: float = 1.0
    dashed: bool = False            # outline-only head, e.g. "not detected"
    title: str = ""                 # tooltip
    pair: str = ""                  # hover-link id shared between the two panels
    status: str = ""                # data-status attribute (for tests / styling)
    label: str = ""                 # small text above the staff at this note (e.g. "+31c")
    label_color: str = INK


def _treble_clef(x, y_g):
    """Treble clef; x = left edge, y_g = y of the G line (second line from the bottom)."""
    d = ("M0.9,3.4 C0.9,4.7 -0.5,4.8 -0.6,3.9 C-0.7,3.1 0.3,2.9 0.5,3.7 M0.9,3.4 L0.35,-3.3 "
         "C0.1,-5.2 1.0,-5.6 1.4,-4.9 C1.8,-4.2 1.0,-3.0 -0.4,-1.9 C-2.0,-0.6 -1.6,1.6 0.1,1.9 "
         "C1.7,2.1 2.0,0.0 0.9,-0.6 C0.0,-1.1 -1.0,-0.4 -0.7,0.5")
    return (f'<g transform="translate({x + 2.2 * SP:.1f},{y_g:.1f}) scale({SP:.2f})" fill="none" stroke="{INK}" '
            f'stroke-width="0.3" stroke-linecap="round" stroke-linejoin="round"><path d="{d}"/></g>')


def _accidental(kind, x, y, color=INK):
    """Sharp (+1), flat (-1) or natural (0) centred on (x, y)."""
    s = SP
    if kind > 0:
        return (f'<g stroke="{color}" fill="none" stroke-linecap="round"><path d="M{x - .22 * s:.1f},{y - 1.3 * s:.1f}V{y + 1.5 * s:.1f}'
                f'M{x + .22 * s:.1f},{y - 1.5 * s:.1f}V{y + 1.3 * s:.1f}" stroke-width="0.9"/>'
                f'<path d="M{x - .6 * s:.1f},{y - .35 * s:.1f}L{x + .6 * s:.1f},{y - .75 * s:.1f}M{x - .6 * s:.1f},{y + .75 * s:.1f}'
                f'L{x + .6 * s:.1f},{y + .35 * s:.1f}" stroke-width="2"/></g>')
    if kind < 0:
        return (f'<path d="M{x - .3 * s:.1f},{y - 1.8 * s:.1f}V{y + .8 * s:.1f}C{x + .8 * s:.1f},{y + .2 * s:.1f} {x + .9 * s:.1f},'
                f'{y - 1.1 * s:.1f} {x - .3 * s:.1f},{y + .1 * s:.1f}" stroke="{color}" stroke-width="1.2" fill="none" stroke-linecap="round"/>')
    return (f'<g stroke="{color}" fill="none" stroke-linecap="round"><path d="M{x - .3 * s:.1f},{y - 1.4 * s:.1f}V{y + .7 * s:.1f}'
            f'M{x + .3 * s:.1f},{y - .7 * s:.1f}V{y + 1.4 * s:.1f}" stroke-width="0.9"/>'
            f'<path d="M{x - .3 * s:.1f},{y + .5 * s:.1f}L{x + .3 * s:.1f},{y + .2 * s:.1f}M{x - .3 * s:.1f},{y - .3 * s:.1f}'
            f'L{x + .3 * s:.1f},{y - .6 * s:.1f}" stroke-width="2"/></g>')


def _rest(dur, x, y_top):
    """Rest glyph for a standard rest value, centred at x on a staff whose top line is y_top."""
    mid, s = y_top + 2 * SP, SP
    if dur >= 4:                                                         # whole: hangs from the 4th line
        return f'<rect x="{x - .6 * s:.1f}" y="{y_top + s:.1f}" width="{1.2 * s:.1f}" height="{.5 * s:.1f}" fill="{INK}"/>'
    if dur >= 2:                                                         # half: sits on the middle line
        return f'<rect x="{x - .6 * s:.1f}" y="{mid - .5 * s:.1f}" width="{1.2 * s:.1f}" height="{.5 * s:.1f}" fill="{INK}"/>'
    if dur >= 1:                                                         # quarter: the zigzag
        return (f'<path d="M{x - .3 * s:.1f},{mid - 1.5 * s:.1f}L{x + .5 * s:.1f},{mid - .5 * s:.1f}L{x - .4 * s:.1f},{mid + .4 * s:.1f}'
                f'L{x + .4 * s:.1f},{mid + 1.2 * s:.1f}C{x - .5 * s:.1f},{mid + 1.2 * s:.1f} {x - .6 * s:.1f},{mid + 2 * s:.1f} '
                f'{x + .1 * s:.1f},{mid + 2.2 * s:.1f}" stroke="{INK}" stroke-width="1.8" fill="none" stroke-linejoin="round"/>')
    n = 1 if dur >= .5 else 2 if dur >= .25 else 3                       # eighth/16th/32nd: flagged dots on a slanted stem
    out = f'<line x1="{x + .5 * s:.1f}" y1="{mid - 1.1 * s:.1f}" x2="{x - .2 * s + (n - 1) * .15 * s:.1f}" y2="{mid + (n - .2) * s:.1f}" stroke="{INK}" stroke-width="1.1"/>'
    for k in range(n):
        yy = mid - .6 * s + k * s
        out += (f'<circle cx="{x - .35 * s + k * .15 * s:.1f}" cy="{yy:.1f}" r="{.38 * s:.1f}" fill="{INK}"/>'
                f'<path d="M{x - .3 * s + k * .15 * s:.1f},{yy + .2 * s:.1f}Q{x + .3 * s:.1f},{yy + .3 * s:.1f} {x + .5 * s:.1f},{yy - .4 * s:.1f}" '
                f'stroke="{INK}" stroke-width="1.1" fill="none"/>')
    return out


def _flags(x, y, up, n, color):
    """n flags at the free end (x, y) of a stem: they sweep down-right from an up-stem, up-right from a down-stem."""
    s, d, out = SP, (1 if up else -1), ""
    for k in range(n):
        yy = y + d * k * 0.95 * s
        out += (f'<path d="M{x:.1f},{yy:.1f}C{x + .2 * s:.1f},{yy + d * .9 * s:.1f} {x + 1.3 * s:.1f},{yy + d * 1.2 * s:.1f} '
                f'{x + .9 * s:.1f},{yy + d * 2.6 * s:.1f}" stroke="{color}" stroke-width="1.5" fill="none" stroke-linecap="round"/>')
    return out


# ----------------------------------------------------------------------------- rendering

def _beam_groups(slots, beat_type):
    """Runs of consecutive single-note slots, each <= an eighth, inside one beat -> lists of slot indices."""
    groups, cur, cur_beat = [], [], None
    unit = 1.5 if beat_type == 8 else 1.0
    for k, s in enumerate(slots):
        ok = s.kind == "notes" and len(s.notes) == 1 and s.dur <= 0.5 + 1e-6
        beat = math.floor(s.start_q / unit + 1e-6)
        if ok and cur and beat == cur_beat and slots[cur[-1]].start_q + slots[cur[-1]].dur >= s.start_q - 1e-6:
            cur.append(k)
        else:
            if len(cur) >= 2:
                groups.append(cur)
            cur, cur_beat = ([k], beat) if ok else ([], None)
    if len(cur) >= 2:
        groups.append(cur)
    return groups


def _ledger_lines(lo_pos, hi_pos):
    """Even staff positions that need a ledger line for notes spanning lo_pos..hi_pos."""
    lines = []
    if lo_pos < BOTTOM_LINE_POS:
        lines += range(BOTTOM_LINE_POS - 2, (lo_pos if lo_pos % 2 == 0 else lo_pos + 1) - 1, -2)
    if hi_pos > TOP_LINE_POS:
        lines += range(TOP_LINE_POS + 2, (hi_pos if hi_pos % 2 == 0 else hi_pos - 1) + 1, 2)
    return lines


def _draw_header(out, layout, si, sys_):
    top, x0 = sys_.y, 8.0
    last = si == len(layout.systems) - 1                   # the last system's staff stops at the final barline
    x1 = sys_.measures[-1].x + sys_.measures[-1].w if last else layout.width - 8
    for k in range(5):
        out.append(f'<line x1="{x0}" x2="{x1:.1f}" y1="{top + k * SP:.1f}" y2="{top + k * SP:.1f}" stroke="{INK}" stroke-width="0.8" class="staff"/>')
    out.append(f'<line x1="{x0}" x2="{x0}" y1="{top}" y2="{top + 4 * SP}" stroke="{INK}" stroke-width="1"/>')
    out.append(_treble_clef(x0 + 2, top + 3 * SP))
    f, kx = layout.fifths, x0 + CLEF_W
    # key-signature accidentals sit on fixed lines of the treble staff (F C G D A E B / B E A D G C F)
    for j, pos in enumerate(([38, 35, 39, 36, 33, 37, 34] if f > 0 else [34, 37, 33, 36, 32, 35, 31])[:abs(f)]):
        out.append(_accidental(1 if f > 0 else -1, kx + j * 7, _y(top, pos)))
    if si == 0 and layout.time_sig:
        tx = kx + 7 * abs(f) + 10
        out.append(f'<g font-family="serif" font-weight="bold" font-size="{2.1 * SP:.0f}" text-anchor="middle" fill="{INK}">'
                   f'<text x="{tx:.1f}" y="{top + 2 * SP - 1:.1f}">{layout.time_sig[0]}</text>'
                   f'<text x="{tx:.1f}" y="{top + 4 * SP - 1:.1f}">{layout.time_sig[1]}</text></g>')


def _draw_beam(out, layout, m, group, glyphs, top, up):
    """One beam over `group` (slot indices): stems end on a (gently sloped) beam line; extra beams for 16ths."""
    first_notes = [m.slots[k].notes[0] for k in group]                        # beamed slots hold single notes
    xs = [layout.head_x[i] + (HEAD_RX - 0.6 if up else -HEAD_RX + 0.6) for i in first_notes]
    ys = [_y(top, spell(glyphs[i].midi, layout.fifths)[0]) for i in first_notes]
    ends = [y - STEM_LEN if up else y + STEM_LEN for y in ys]
    slope = max(-0.5 * SP, min(0.5 * SP, ends[-1] - ends[0])) / max(xs[-1] - xs[0], 1)
    line = lambda x: ends[0] + slope * (x - xs[0])
    # no stem shorter than 2.5 spaces: push the whole beam away from the heads if needed
    need = max((line(x) - (y - 2.5 * SP)) if up else ((y + 2.5 * SP) - line(x)) for x, y in zip(xs, ys))
    shift = (-need if up else need) if need > 0 else 0
    by = lambda x: line(x) + shift
    g0 = glyphs[first_notes[0]]
    for x, y in zip(xs, ys):
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y:.1f}" y2="{by(x):.1f}" stroke="{g0.color}" stroke-width="1" opacity="{g0.opacity}"/>')
    thick = 0.5 * SP
    beam = lambda xa, xb, off=0: (f'<line x1="{xa:.1f}" y1="{by(xa) + off:.1f}" x2="{xb:.1f}" y2="{by(xb) + off:.1f}" '
                                  f'stroke="{g0.color}" stroke-width="{thick:.1f}" class="beam" opacity="{g0.opacity}"/>')
    out.append(beam(xs[0], xs[-1] + 0.5))
    sixteenths = [_note_value(m.slots[k].dur)[1] in ("16th", "32nd") for k in group]
    j = 0
    while j < len(group):                                   # secondary beam over runs of 16ths (stub if alone)
        if sixteenths[j]:
            e = j
            while e + 1 < len(group) and sixteenths[e + 1]:
                e += 1
            xa, xb = xs[j], xs[e]
            if j == e:
                nb = j - 1 if j > 0 else j + 1
                xb = xs[j] + (xs[nb] - xs[j]) * 0.4
            out.append(beam(min(xa, xb), max(xa, xb), thick * 1.6 if up else -thick * 1.6))
            j = e + 1
        else:
            j += 1


def _draw_heads(out, layout, here, glyphs, spelled, up, top, name, dots, state):
    """Ledger lines, then noteheads (+ accidental, dot, label) of one slot -> {note index: head y}."""
    key = key_alters(layout.fifths)
    hx = layout.head_x[here[0]]
    ys = {i: _y(top, spelled[i][0]) for i in here}
    lo, hi = min(p[0] for p in spelled.values()), max(p[0] for p in spelled.values())
    for pos in _ledger_lines(lo, hi):
        out.append(f'<line x1="{hx - HEAD_RX - 3:.1f}" x2="{hx + HEAD_RX + 3:.1f}" y1="{_y(top, pos):.1f}" y2="{_y(top, pos):.1f}" '
                   f'stroke="{INK}" stroke-width="0.9"/>')
    flipped, prev = {}, None                                # a second in a chord: the upper/lower head sits beside the stem
    for i in (here if up else here[::-1]):
        flipped[i] = prev is not None and abs(spelled[i][0] - prev[0]) == 1 and not flipped[prev[1]]
        prev = (spelled[i][0], i)
    for i in here:
        g, (pos, acc) = glyphs[i], spelled[i]
        cx = hx + ((2 * HEAD_RX - 1.2) if up else -(2 * HEAD_RX - 1.2)) * flipped[i]
        parts = []
        if acc != state.get(pos, key[pos % 7]):             # accidental only if it changes what is in force
            parts.append(_accidental(acc, cx - HEAD_RX - 7, ys[i], g.color))
            state[pos] = acc
        hollow = name in ("whole", "half") or g.dashed
        dash = ' stroke-dasharray="2 1.5"' if g.dashed else ""
        parts.append(f'<ellipse cx="{cx:.1f}" cy="{ys[i]:.1f}" rx="{HEAD_RX:.1f}" ry="{HEAD_RY:.1f}" '
                     f'transform="rotate(-20 {cx:.1f} {ys[i]:.1f})" fill="{"white" if hollow else g.color}" '
                     f'stroke="{g.color}" stroke-width="{1.6 if hollow else 0.8}"{dash} class="head"/>')
        if dots:                                            # a dot on a line moves up into the space above
            parts.append(f'<circle cx="{cx + HEAD_RX + 4:.1f}" cy="{ys[i] - (SP / 2 if pos % 2 == 0 else 0):.1f}" r="1.3" fill="{g.color}"/>')
        if g.label:
            parts.append(f'<text x="{cx:.1f}" y="{top - 13:.1f}" text-anchor="middle" font-size="8.5" font-weight="bold" '
                         f'fill="{g.label_color}">{escape(g.label)}</text>')
        out.append(f'<g class="note" data-pair="{escape(g.pair)}" data-status="{escape(g.status)}" opacity="{g.opacity}">'
                   f'<title>{escape(g.title)}</title>{"".join(parts)}</g>')
    return ys


def _draw_stem(out, layout, here, glyph, ys, up, top, name):
    """Stem (standard length, reaching at least the middle line) and flags of an unbeamed slot."""
    hx = layout.head_x[here[0]]
    stem_x = hx + HEAD_RX - 0.6 if up else hx - HEAD_RX + 0.6
    near, far = (max(ys.values()), min(ys.values())) if up else (min(ys.values()), max(ys.values()))
    end = far - STEM_LEN if up else far + STEM_LEN
    end = min(end, top + 2 * SP) if up else max(end, top + 2 * SP)
    out.append(f'<line x1="{stem_x:.1f}" x2="{stem_x:.1f}" y1="{near:.1f}" y2="{end:.1f}" stroke="{glyph.color}" '
               f'stroke-width="1" opacity="{glyph.opacity}"/>')
    flags = {"eighth": 1, "16th": 2, "32nd": 3}.get(name, 0)
    if flags:
        out.append(_flags(stem_x, end, up, flags, glyph.color))


def _draw_slot(out, layout, m, k, glyphs, top, state, beamed, drawn):
    """Heads, accidentals, dots, ledger lines and stem/flags/beam of the note slot m.slots[k]."""
    fifths = layout.fifths
    here = sorted(m.slots[k].notes, key=lambda i: glyphs[i].midi)                 # lowest pitch first
    spelled = {i: spell(glyphs[i].midi, fifths) for i in here}
    _, name, dots = _note_value(max(layout.notes[i].duration for i in here))
    group = beamed.get(k)
    # stems point away from the middle line; a beamed group shares one direction (by its average pitch)
    pitched = [spelled[i][0] for i in here] if not group else [
        spell(glyphs[i].midi, fifths)[0] for kk in group for i in m.slots[kk].notes]
    up = sum(pitched) / len(pitched) < MIDDLE_POS
    ys = _draw_heads(out, layout, here, glyphs, spelled, up, top, name, dots, state)
    if name == "whole":
        return
    if group:
        if id(group) not in drawn:                          # draw each beam once, from its first slot
            drawn.add(id(group))
            _draw_beam(out, layout, m, group, glyphs, top, up)
    else:
        _draw_stem(out, layout, here, glyphs[here[0]], ys, up, top, name)


def render_panel(layout, glyph_for: Callable[[int], Glyph], title="", extras=()):
    """Draw the layout -> SVG string.

    glyph_for(i) -> Glyph for score note i. extras: [(score_sec, midi, {color,title,pair})] for notes that
    were detected in the recording but are not in the score: drawn as plain heads at the end of the
    last slot that starts at or before their (score-aligned) time.
    """
    oy = 22 if title else 0
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {layout.width:.0f} {layout.height + oy:.0f}" '
           f'width="100%" font-family="sans-serif" font-size="10">', '<rect width="100%" height="100%" fill="white"/>']
    if title:
        out.append(f'<text x="{layout.width / 2:.0f}" y="15" text-anchor="middle" font-size="13" font-weight="bold">{escape(title)}</text>')
    out.append(f'<g transform="translate(0 {oy})">')
    beat_type = layout.time_sig[1] if layout.time_sig else 4

    for si, sys_ in enumerate(layout.systems):
        top = sys_.y
        _draw_header(out, layout, si, sys_)
        for m in sys_.measures:
            out.append(f'<text x="{m.x + 6:.1f}" y="{top - 6:.1f}" fill="#888" font-size="9">{m.number}</text>')
            end_x = m.x + m.w
            if m.last:
                out.append(f'<line x1="{end_x - 4}" x2="{end_x - 4}" y1="{top}" y2="{top + 4 * SP}" stroke="{INK}" stroke-width="1"/>'
                           f'<line x1="{end_x}" x2="{end_x}" y1="{top}" y2="{top + 4 * SP}" stroke="{INK}" stroke-width="3"/>')
            else:
                out.append(f'<line x1="{end_x}" x2="{end_x}" y1="{top}" y2="{top + 4 * SP}" stroke="{INK}" stroke-width="1"/>')
            glyphs = {i: glyph_for(i) for s in m.slots for i in s.notes}
            beamed = {k: g for g in _beam_groups(m.slots, beat_type) for k in g}
            state, drawn = {}, set()
            for k, s in enumerate(m.slots):
                if s.kind == "rest":
                    out.append(_rest(s.dur, s.x + s.w / 2 - 3, top))
                else:
                    _draw_slot(out, layout, m, k, glyphs, top, state, beamed, drawn)

    slots = [(sys_, s) for sys_ in layout.systems for m in sys_.measures for s in m.slots if s.kind == "notes"]
    for sec, midi, kw in extras:                                # detected but not written
        before = [t for t in slots if t[1].start_sec <= sec + 1e-6] or slots[:1]
        if not before:
            continue
        sys_, s = before[-1]
        pos, acc = spell(midi, layout.fifths)
        x, y, col = s.x + s.w - 4, _y(sys_.y, pos), kw.get("color", "#8e44ad")
        out.append(f'<g class="note extra" data-pair="{escape(kw.get("pair", ""))}" data-status="extra"><title>{escape(kw.get("title", ""))}</title>'
                   f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{SP * .62:.1f}" ry="{SP * .46:.1f}" transform="rotate(-20 {x:.1f} {y:.1f})" '
                   f'fill="{col}" stroke="{col}"/>{_accidental(acc, x - SP, y, col) if acc else ""}</g>')
    out.append("</g></svg>")
    return "".join(out)
