import tempfile
import unittest
from pathlib import Path

from helpers import musicxml
from maestro.score.musicxml import parse_musicxml
from maestro.score.score_representation import Score
from maestro.score.visualize import score_to_svg

TIE_CHORD_XML = """<score-partwise><part-list><score-part id="P1"/></part-list><part id="P1">
<measure number="1"><attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
<direction><sound tempo="60"/></direction>
<note><pitch><step>A</step><octave>4</octave></pitch><duration>2</duration><tie type="start"/>
  <notations><articulations><staccato/></articulations></notations></note>
<note><rest/><duration>2</duration></note>
<note><pitch><step>C</step><alter>1</alter><octave>5</octave></pitch><duration>4</duration></note>
<note><chord/><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration></note>
</measure>
<measure number="2"><note><pitch><step>A</step><octave>4</octave></pitch><duration>2</duration><tie type="stop"/></note></measure>
</part></score-partwise>"""


class ParseMusicXML(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.musicxml"
            p.write_text(text)
            return parse_musicxml(p)

    def test_pitches_timing_and_tempo(self):
        s = self.parse(musicxml([69, 70, 60, 81], tempo=120))
        self.assertEqual([n.midi_pitch for n in s.notes], [69, 70, 60, 81])
        self.assertEqual([n.pitch for n in s.notes], ["A4", "A#4", "C4", "A5"])
        self.assertAlmostEqual(s.notes[2].onset, 1.0)   # quarter at 120 bpm = 0.5 s
        self.assertEqual(s.notes[3].beat, 4.0)

    def test_ties_rests_chords_articulation(self):
        s = self.parse(TIE_CHORD_XML)
        self.assertEqual(len(s.notes), 3)                      # A4 (tied), C#5, E5
        a4 = next(n for n in s.notes if n.midi_pitch == 69)
        self.assertEqual(a4.duration, 2.0)                     # 1 beat + tied 1 beat
        self.assertEqual(a4.articulation, "staccato")
        self.assertAlmostEqual(a4.offset - a4.onset, 2.0)      # tempo 60 -> 1 s per quarter
        c, e = (n for n in s.notes if n.midi_pitch in (73, 76))
        self.assertEqual(c.onset, e.onset)                     # chord = simultaneous; rest skipped
        self.assertAlmostEqual(c.onset, 2.0)   # after A4 (1 beat) + rest (1 beat) at 60 bpm

    def test_json_roundtrip(self):
        s = self.parse(musicxml([69, 71]))
        self.assertEqual(Score.from_json(s.to_json()), s)


class Visualize(unittest.TestCase):
    def test_svg_has_a_bar_per_note_and_markers(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.musicxml"
            p.write_text(musicxml([69, 71, 72]))
            score = parse_musicxml(p)
            svg = score_to_svg(score, {"notes": [{"score_idx": 0, "cents": 40.0}]}, Path(d) / "o.svg")
            self.assertTrue((Path(d) / "o.svg").exists())
        self.assertEqual(svg.count('fill="#4a6fa5"'), 3)
        self.assertIn("+40 cents", svg)


if __name__ == "__main__":
    unittest.main()
