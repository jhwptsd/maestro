# Standard format of MusicXML files parsing
#
# Parses MusicXML (from preprocessing/score.py) into the internal format defined in
# score/score_representation.py. Repeats/jumps are unrolled with score/repeats.py so notes appear in
# the order they are *played*, which is what alignment/ compares against the audio.
#
# Handles: multiple stops (chords -> several Notes with the same onset), rests, ties (merged, also
# across repeats), backup/forward (multi-voice), accidentals, first articulation mark, repeats,
# voltas, D.C./D.S./Coda/Fine, tempo changes, pickup measures, changing time signatures/divisions,
# measure-repeat signs (1 or more measures), and picking the violin part of a multi-part score.
# Not handled: grace notes/ornaments (skipped), transposing parts.
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

from maestro.score.repeats import Measure, playback_order, words_to_navigation
from maestro.score.score_representation import Note, Score, midi_to_name

_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
DEFAULT_TEMPO = 120.0


def _midi(pitch_el):
    """<pitch> element -> MIDI number."""
    alter = round(float(pitch_el.findtext("alter", "0")))
    return 12 * (int(pitch_el.findtext("octave")) + 1) + _STEP[pitch_el.findtext("step")] + alter


def _pick_part(root, part_id=None):
    """Part element: the given id, else the first part named like a violin, else the first part."""
    parts = root.findall("part")
    if part_id is None:
        for sp in root.findall("part-list/score-part"):
            if "violin" in (sp.findtext("part-name") or "").lower():
                part_id = sp.get("id")
                break
    return next((p for p in parts if p.get("id") == part_id), parts[0])


def _endings(spec):
    """'1, 2' / '1-3' -> (1, 2) / (1, 2, 3)."""
    out = []
    for tok in re.split(r"[,\s]+", spec.strip()):
        if "-" in tok:
            a, b = tok.split("-")
            out += range(int(a), int(b) + 1)
        elif tok:
            out.append(int(tok))
    return tuple(out)


@dataclass
class _State:
    """Musical context carried from one measure to the next."""
    divisions: int = 1                  # <divisions> per quarter note
    beats: int = 4
    beat_type: int = 4
    ending: tuple = ()                  # volta numbers currently open
    key_fifths: Optional[int] = None    # first key / time signature seen (for notation display)
    time_signature: Optional[tuple] = None
    repeat_size: int = 0                # measure-repeat sign: measures in the group / still to fill
    repeat_left: int = 0


def _read_barlines(mel, m, st):
    """Repeat and ending marks. Left barlines open repeats/endings, right barlines close them."""
    bars = [(bar.find("repeat"), bar.find("ending")) for bar in mel.findall("barline")]
    for rep, ending in bars:
        if rep is not None and rep.get("direction") == "forward":
            m.fwd = True
        if ending is not None and ending.get("type") == "start":
            st.ending = _endings(ending.get("number", "1"))
    m.endings = st.ending
    for rep, ending in bars:
        if rep is not None and rep.get("direction") == "backward":
            m.back = int(rep.get("times", 2))
        if ending is not None and ending.get("type") in ("stop", "discontinue"):
            st.ending = ()


def _read_attributes(el, st):
    """<attributes>: divisions, time and key signature, measure-repeat sign."""
    st.divisions = int(el.findtext("divisions", st.divisions))
    st.beats = int(el.findtext("time/beats", str(st.beats)).split("+")[0])
    st.beat_type = int(el.findtext("time/beat-type", st.beat_type))
    if st.key_fifths is None and el.findtext("key/fifths") is not None:
        st.key_fifths = int(el.findtext("key/fifths"))
    if st.time_signature is None and el.find("time") is not None:
        st.time_signature = (st.beats, st.beat_type)
    sign = el.find("measure-style/measure-repeat")
    if sign is not None and sign.get("type") == "start":
        st.repeat_size = st.repeat_left = int((sign.text or "1").strip() or 1)


def _read_directions(el, m, cursor):
    """Tempo changes and jump marks (D.C., D.S., segno, coda, Fine) from <direction>/<sound>, preferring
    <sound> attributes and falling back to the printed words."""
    for sound in ([el] if el.tag == "sound" else el.iter("sound")):
        a = sound.attrib
        if "tempo" in a:
            m.tempos.append((cursor, float(a["tempo"])))
        m.dacapo |= a.get("dacapo") == "yes"
        for key in ("dalsegno", "tocoda", "segno", "coda"):
            if key in a:
                setattr(m, key, a[key])
        m.fine |= "fine" in a
    for words in el.iter("words"):
        for key, value in words_to_navigation(words.text or "").items():
            if not getattr(m, key):
                setattr(m, key, value)
    if el.find("direction-type/segno") is not None and m.segno is None:
        m.segno = ""
    if el.find("direction-type/coda") is not None and m.coda is None:
        m.coda = ""


def _note_record(el, st, onset, dur):
    """One pitched <note> as a plain dict (onset relative to the measure, in quarters)."""
    ties = {t.get("type") for t in el.findall("tie")} | {t.get("type") for t in el.findall("notations/tied")}
    art = el.find("notations/articulations")
    return {"midi": _midi(el.find("pitch")), "onset": onset, "dur": dur,
            "art": art[0].tag if art is not None and len(art) else None,
            "beat_type": st.beat_type, "tie_start": "start" in ties, "tie_stop": "stop" in ties}


def _read_measures(part):
    """Raw per-measure data (notes with onsets relative to the measure, repeat/jump markings) + the first
    key/time signature."""
    st, measures = _State(), []
    for idx, mel in enumerate(part.findall("measure"), start=1):
        try:
            number = int(mel.get("number", idx))
        except ValueError:                      # e.g. "X1" for implicit measures
            number = idx
        m = Measure(number=number, length=0.0)
        cursor = last_onset = end = 0.0         # quarters from the start of this measure
        _read_barlines(mel, m, st)
        for el in mel:
            if el.tag == "attributes":
                _read_attributes(el, st)
            elif el.tag in ("direction", "sound"):
                _read_directions(el, m, cursor)
            elif el.tag in ("backup", "forward"):
                step = int(el.findtext("duration")) / st.divisions
                cursor += step if el.tag == "forward" else -step
            elif el.tag == "note" and el.find("grace") is None and el.find("cue") is None:
                dur = int(el.findtext("duration")) / st.divisions
                if el.find("chord") is None:    # chord notes share the previous onset
                    last_onset, cursor = cursor, cursor + dur
                end = max(end, cursor)
                if el.find("rest") is None:
                    m.notes.append(_note_record(el, st, last_onset, dur))

        m.length = end if end else st.beats * 4.0 / st.beat_type     # empty measure: use the time signature
        if st.repeat_left > 0 and len(measures) >= st.repeat_size:   # measure-repeat sign: copy the measure
            src = measures[len(measures) - st.repeat_size]           # `repeat_size` back (a 2-measure sign
            m.notes, m.length = [dict(n) for n in src.notes], src.length   # copies two measures)
            st.repeat_left -= 1
        measures.append(m)
    return measures, {"key_fifths": st.key_fifths, "time_signature": st.time_signature}


def parse_musicxml(path, part=None, expand_repeats=True):
    """Parse a partwise MusicXML file -> Score (notes in played order, with absolute seconds).

    part: part id to read (default: the violin part, else the first part).
    expand_repeats: unroll repeats/voltas/D.C./D.S. (False = notes in written order, once).
    """
    measures, meta = _read_measures(_pick_part(ET.parse(path).getroot(), part))
    order = playback_order(measures) if expand_repeats else range(len(measures))

    tempo_map = [(0.0, DEFAULT_TEMPO)]       # (absolute quarter position, bpm)
    recs, open_ties = [], {}                 # open_ties: midi -> record that a tie may extend
    t = 0.0
    for idx in order:
        m = measures[idx]
        for off, bpm in m.tempos:
            if off + t == 0.0:
                tempo_map[0] = (0.0, bpm)
            else:
                tempo_map.append((t + off, bpm))
        for n in m.notes:
            onset = t + n["onset"]
            prev = open_ties.get(n["midi"])
            if n["tie_stop"] and prev:
                prev["dur"] += n["dur"]       # continuation of a tied note (even across a jump)
                if not n["tie_start"]:
                    del open_ties[n["midi"]]
                continue
            rec = {"measure": m.number, "beat": 1.0 + n["onset"] * n["beat_type"] / 4.0, "midi": n["midi"],
                   "dur": n["dur"], "art": n["art"], "onset": onset}
            recs.append(rec)
            if n["tie_start"]:
                open_ties[n["midi"]] = rec
        t += m.length

    tempo_map.sort()

    def seconds(q):
        """Quarter position -> seconds under the piecewise-constant tempo map."""
        sec = 0.0
        for k, (pos, bpm) in enumerate(tempo_map):
            nxt = tempo_map[k + 1][0] if k + 1 < len(tempo_map) else float("inf")
            if q > pos:
                sec += (min(q, nxt) - pos) * 60.0 / bpm
        return sec

    notes = [Note(r["measure"], r["beat"], midi_to_name(r["midi"]), r["midi"], r["dur"], r["art"],
                  seconds(r["onset"]), seconds(r["onset"] + r["dur"])) for r in recs]
    notes.sort(key=lambda n: (n.onset, n.midi_pitch))
    return Score(notes, tempo_map[0][1], meta["key_fifths"], meta["time_signature"])
