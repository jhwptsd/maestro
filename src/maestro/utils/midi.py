# MIDI -> MusicXML conversion (new file; stdlib only).
#
# Used by utils/convert_file_type.py so users can submit a .mid score. MIDI has no notation, so this
# is an approximation: notes are quantised only to the barline grid (barlines from the first time
# signature), spelled with sharps, merged into one part, and treated as monophonic with chords
# (a note ends when the next onset begins). That is all score/musicxml.py needs for alignment
# and assessment. Format 0/1 files with PPQ timing are supported (not SMPTE).
import struct
from pathlib import Path

from maestro.utils.errors import InputError

_NAMES = [("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0), ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0)]


def _vlq(data, i):
    """Read a variable-length quantity -> (value, next index)."""
    v = 0
    while True:
        b = data[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            return v, i


def read_midi(path):
    """-> (notes [(start_tick, end_tick, pitch)], ppq, tempos [(tick, bpm)], (beats, beat_type))."""
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise InputError("Not a valid MIDI file.")
    try:
        _, ntrks, division = struct.unpack(">HHH", data[8:14])
        if division & 0x8000:
            raise InputError("SMPTE-timed MIDI files are not supported.")
        notes, tempos, timesig, i = [], [], None, 8 + struct.unpack(">I", data[4:8])[0]
        for _ in range(ntrks):
            assert data[i:i + 4] == b"MTrk"
            end = i + 8 + struct.unpack(">I", data[i + 4:i + 8])[0]
            i += 8
            tick, status, active = 0, 0, {}
            while i < end:
                delta, i = _vlq(data, i)
                tick += delta
                if data[i] & 0x80:
                    status = data[i]
                    i += 1
                if status == 0xFF:                                   # meta event
                    mtype = data[i]
                    n, i = _vlq(data, i + 1)
                    body = data[i:i + n]
                    i += n
                    if mtype == 0x51:
                        tempos.append((tick, 60e6 / int.from_bytes(body, "big")))
                    elif mtype == 0x58 and timesig is None:
                        timesig = (body[0], 2 ** body[1])
                elif status in (0xF0, 0xF7):                         # sysex
                    n, i = _vlq(data, i)
                    i += n
                else:
                    kind, chan = status >> 4, status & 0x0F
                    size = 1 if kind in (0xC, 0xD) else 2
                    d1, d2 = data[i], (data[i + 1] if size == 2 else 0)
                    i += size
                    if chan == 9:                                    # drums
                        continue
                    if kind == 0x9 and d2 > 0:
                        active.setdefault((chan, d1), []).append(tick)
                    elif kind in (0x8, 0x9) and active.get((chan, d1)):
                        notes.append((active[(chan, d1)].pop(0), tick, d1))
            i = end
    except (IndexError, AssertionError, struct.error):
        raise InputError("Corrupt MIDI file.")
    if not notes:
        raise InputError("The MIDI file contains no notes.")
    return sorted(notes), division, sorted(tempos), timesig or (4, 4)


def midi_to_musicxml(path, out_path):
    """Write a MusicXML approximation of the MIDI file at `path` to `out_path`."""
    notes, ppq, tempos, (beats, beat_type) = read_midi(path)
    bar = beats * ppq * 4 // beat_type
    n_bars = (max(e for _, e, _ in notes) - 1) // bar + 1
    # split notes at barlines: (bar, start_in_bar, end_in_bar, pitch, continues_from_prev, continues_to_next)
    per_bar = [[] for _ in range(n_bars)]
    for s, e, p in notes:
        for b in range(s // bar, (e - 1) // bar + 1):
            lo, hi = max(s, b * bar), min(e, (b + 1) * bar)
            per_bar[b].append((lo - b * bar, hi - b * bar, p, lo > s, hi < e))

    def note(p, dur, chord=False, tie_stop=False, tie_start=False):
        step, alt = _NAMES[p % 12]
        alter = f"<alter>{alt}</alter>" if alt else ""
        ties = ('<tie type="stop"/>' if tie_stop else "") + ('<tie type="start"/>' if tie_start else "")
        return (f"<note>{'<chord/>' if chord else ''}<pitch><step>{step}</step>{alter}"
                f"<octave>{p // 12 - 1}</octave></pitch><duration>{dur}</duration>{ties}</note>")

    rest = lambda dur: f"<note><rest/><duration>{dur}</duration></note>"
    measures = []
    for b, segs in enumerate(per_bar):
        body = ""
        if b == 0:
            body += (f"<attributes><divisions>{ppq}</divisions><time><beats>{beats}</beats>"
                     f"<beat-type>{beat_type}</beat-type></time></attributes>")
        for t, bpm in tempos:                                        # tempo changes in this bar
            if (b * bar <= t < (b + 1) * bar) or (b == 0 and t == 0):
                body += f'<direction><direction-type/><sound tempo="{bpm:.2f}"/></direction>'
        onsets = sorted({s for s, *_ in segs})
        cursor = 0
        for k, on in enumerate(onsets):
            group = [x for x in segs if x[0] == on]
            nxt = onsets[k + 1] if k + 1 < len(onsets) else bar
            dur = min(min(x[1] for x in group), nxt) - on           # chord notes share one duration
            if on > cursor:
                body += rest(on - cursor)
            for c, (_, hi, p, tie_stop, cont) in enumerate(sorted(group, key=lambda x: x[2])):
                body += note(p, dur, chord=c > 0, tie_stop=tie_stop, tie_start=cont and on + dur == bar)
            cursor = on + dur
        if cursor < bar:
            body += rest(bar - cursor)
        measures.append(f'<measure number="{b + 1}">{body}</measure>')
    Path(out_path).write_text(
        '<?xml version="1.0"?><score-partwise version="3.1"><part-list><score-part id="P1"/></part-list>'
        f'<part id="P1">{"".join(measures)}</part></score-partwise>')
    return Path(out_path)
