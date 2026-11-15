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


def _read_measures(part):
    """Raw per-measure data (notes with onsets relative to the measure, repeat/jump markings)."""
    measures = []
    divisions, beats, beat_type = 1, 4, 4
    meta = {"key_fifths": None, "time_signature": None}   # first key/time signature (for notation display)
    current_ending = ()
    repeat_n = repeat_left = 0     # measure-repeat sign: group size / measures of the group still to fill
    for idx, mel in enumerate(part.findall("measure"), start=1):
        try:
            number = int(mel.get("number", idx))
        except ValueError:                      # e.g. "X1" for implicit measures
            number = idx
        m = Measure(number=number, length=0.0)
        cursor = last_onset = end = 0.0         # quarters from the start of this measure

        # Barlines: left ones open repeats/endings, right ones close them. Handle left first.
        for bar in mel.findall("barline"):
            rep, end_el = bar.find("repeat"), bar.find("ending")
            if rep is not None and rep.get("direction") == "forward":
                m.fwd = True
            if end_el is not None and end_el.get("type") == "start":
                current_ending = _endings(end_el.get("number", "1"))
        m.endings = current_ending
        for bar in mel.findall("barline"):
            rep, end_el = bar.find("repeat"), bar.find("ending")
            if rep is not None and rep.get("direction") == "backward":
                m.back = int(rep.get("times", 2))
            if end_el is not None and end_el.get("type") in ("stop", "discontinue"):
                current_ending = ()

        for el in mel:
            if el.tag == "attributes":
                divisions = int(el.findtext("divisions", divisions))
                beats = int(el.findtext("time/beats", str(beats)).split("+")[0])
                beat_type = int(el.findtext("time/beat-type", beat_type))
                if meta["key_fifths"] is None and el.findtext("key/fifths") is not None:
                    meta["key_fifths"] = int(el.findtext("key/fifths"))
                if meta["time_signature"] is None and el.find("time") is not None:
                    meta["time_signature"] = (beats, beat_type)
                rep = el.find("measure-style/measure-repeat")
                if rep is not None and rep.get("type") == "start":
                    repeat_n = repeat_left = int((rep.text or "1").strip() or 1)
            elif el.tag in ("direction", "sound"):
                for s in ([el] if el.tag == "sound" else el.iter("sound")):
                    a = s.attrib
                    if "tempo" in a:
                        m.tempos.append((cursor, float(a["tempo"])))
                    m.dacapo |= a.get("dacapo") == "yes"
                    for key in ("dalsegno", "tocoda", "segno", "coda"):
                        if key in a:
                            setattr(m, key, a[key])
                    m.fine |= "fine" in a
                for w in el.iter("words"):          # fallback when no <sound> carries the jump
                    for k, v in words_to_navigation(w.text or "").items():
                        if not getattr(m, k):
                            setattr(m, k, v)
                if el.find("direction-type/segno") is not None and m.segno is None:
                    m.segno = ""
                if el.find("direction-type/coda") is not None and m.coda is None:
                    m.coda = ""
            elif el.tag == "backup":
                cursor -= int(el.findtext("duration")) / divisions
            elif el.tag == "forward":
                cursor += int(el.findtext("duration")) / divisions
            elif el.tag == "note":
                if el.find("grace") is not None or el.find("cue") is not None:
                    continue
                dur = int(el.findtext("duration")) / divisions
                if el.find("chord") is None:        # chord notes share the previous onset
                    last_onset, cursor = cursor, cursor + dur
                end = max(end, cursor)
                if el.find("rest") is not None:
                    continue
                ties = {t.get("type") for t in el.findall("tie")} | {t.get("type") for t in el.findall("notations/tied")}
                art = el.find("notations/articulations")
                m.notes.append({"midi": _midi(el.find("pitch")), "onset": last_onset, "dur": dur,
                                "art": art[0].tag if art is not None and len(art) else None,
                                "beat_type": beat_type, "tie_start": "start" in ties, "tie_stop": "stop" in ties})

        m.length = end if end else beats * 4.0 / beat_type   # empty measure: use the time signature
        if repeat_left > 0 and len(measures) >= repeat_n:       # measure-repeat sign: copy the
            src = measures[len(measures) - repeat_n]            # measure `repeat_n` back (so a
            m.notes, m.length = [dict(n) for n in src.notes], src.length   # 2-measure sign copies 2)
            repeat_left -= 1
        measures.append(m)
    return measures, meta


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
