# Standard format of MusicXML files parsing
#
# Parses MusicXML (from preprocessing/score.py) into the internal format defined in
# score/score_representation.py.
#
# Scope: first part only (solo violin), partwise MusicXML, one constant tempo, repeats not
# expanded. Handles chords, rests, ties (merged into one note), backup/forward, accidentals and
# the first articulation mark; grace notes are skipped.
import xml.etree.ElementTree as ET

from maestro.score.score_representation import Note, Score, midi_to_name

_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _midi(pitch_el):
    """<pitch> element -> MIDI number."""
    step = pitch_el.findtext("step")
    alter = round(float(pitch_el.findtext("alter", "0")))
    octave = int(pitch_el.findtext("octave"))
    return 12 * (octave + 1) + _STEP[step] + alter


def parse_musicxml(path):
    """Parse a partwise MusicXML file -> Score."""
    root = ET.parse(path).getroot()
    part = root.find("part")

    tempo = None
    for s in root.iter("sound"):          # first tempo marking wins
        if "tempo" in s.attrib:
            tempo = float(s.attrib["tempo"])
            break
    tempo = tempo or 120.0

    divisions, beat_type = 1, 4
    cursor = 0.0                          # absolute position in quarter notes
    last_onset = 0.0                      # onset of the previous non-chord note (for chords)
    open_ties = {}                        # midi -> Note awaiting a tie "stop"
    raw = []                              # (measure, onset_q, Note-without-seconds fields)

    for m_idx, measure in enumerate(part.findall("measure"), start=1):
        m_start = cursor
        for el in measure:
            if el.tag == "attributes":
                divisions = int(el.findtext("divisions", divisions))
                beat_type = int(el.findtext("time/beat-type", beat_type))
            elif el.tag == "backup":
                cursor -= int(el.findtext("duration")) / divisions
            elif el.tag == "forward":
                cursor += int(el.findtext("duration")) / divisions
            elif el.tag == "note":
                if el.find("grace") is not None:
                    continue
                dur = int(el.findtext("duration")) / divisions
                if el.find("chord") is not None:
                    onset = last_onset
                else:
                    onset, last_onset = cursor, cursor
                    cursor += dur
                if el.find("rest") is not None:
                    continue
                midi = _midi(el.find("pitch"))
                tie_types = {t.get("type") for t in el.findall("tie")}
                if "stop" in tie_types and midi in open_ties:      # extend the tied-from note
                    open_ties[midi]["duration"] += dur
                    if "start" not in tie_types:
                        del open_ties[midi]
                    continue
                art = el.find("notations/articulations")
                rec = {"measure": m_idx,
                       "beat": 1.0 + (onset - m_start) * beat_type / 4.0,
                       "pitch": midi_to_name(midi), "midi_pitch": midi, "duration": dur,
                       "articulation": art[0].tag if art is not None and len(art) else None,
                       "onset_q": onset}
                raw.append(rec)
                if "start" in tie_types:
                    open_ties[midi] = rec

    sec = 60.0 / tempo
    notes = [Note(r["measure"], r["beat"], r["pitch"], r["midi_pitch"], r["duration"], r["articulation"],
                  r["onset_q"] * sec, (r["onset_q"] + r["duration"]) * sec) for r in raw]
    notes.sort(key=lambda n: (n.onset, n.midi_pitch))
    return Score(notes, tempo)
