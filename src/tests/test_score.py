import tempfile
import unittest
from pathlib import Path

from helpers import measure, musicxml, score_xml
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


class Dynamics(unittest.TestCase):
    def test_marks_and_hairpins_follow_the_notes(self):
        step = lambda s: f"<note><pitch><step>{s}</step><octave>4</octave></pitch><duration>1</duration></note>"
        xml = ('<score-partwise><part-list><score-part id="P1"/></part-list><part id="P1"><measure number="1">'
               "<attributes><divisions>1</divisions></attributes>"
               "<direction><direction-type><dynamics><p/></dynamics></direction-type></direction>" + step("C") +
               '<direction><direction-type><wedge type="crescendo"/></direction-type></direction>' + step("D") +
               '<direction><direction-type><wedge type="stop"/></direction-type></direction>'
               "<direction><direction-type><dynamics><f/></dynamics></direction-type></direction>" + step("E") +
               "</measure></part></score-partwise>")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.musicxml"
            path.write_text(xml)
            notes = parse_musicxml(path).notes
        self.assertEqual([(n.dynamic, n.hairpin) for n in notes], [("p", None), ("p", "crescendo"), ("f", None)])


class RepeatsAndEdgeCases(unittest.TestCase):
    """One note per measure with a unique pitch: the pitch sequence is the played measure order."""
    A, B, C, D, E = 60, 62, 64, 65, 67   # measures 1..5

    def played(self, *measures, **kw):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.musicxml"
            p.write_text(score_xml(*measures, **{k: v for k, v in kw.items() if k == "part_list"}))
            return parse_musicxml(p, **{k: v for k, v in kw.items() if k != "part_list"})

    def seq(self, *measures, **kw):
        return [n.midi_pitch for n in self.played(*measures, **kw).notes]

    def test_simple_repeat_and_times(self):
        fwd = '<repeat direction="forward"/>'
        back = lambda t: f'<repeat direction="backward" times="{t}"/>'
        self.assertEqual(self.seq(measure(1, [60]), measure(2, [62], left=fwd, right=back(2)), measure(3, [64])),
                         [60, 62, 62, 64])
        self.assertEqual(self.seq(measure(1, [60], right=back(3))), [60, 60, 60])   # implicit start
        self.assertEqual(self.played(measure(1, [60], right=back(2)), expand_repeats=False).notes[0].midi_pitch, 60)

    def test_first_and_second_endings(self):
        ending = lambda n, t: f'<ending number="{n}" type="{t}"/>'
        back = '<repeat direction="backward"/>'
        ms = [measure(1, [60], left='<repeat direction="forward"/>'),
              measure(2, [62], left=ending(1, "start"), right=ending(1, "stop") + back),
              measure(3, [64], left=ending(2, "start"), right=ending(2, "stop")),
              measure(4, [65])]
        self.assertEqual(self.seq(*ms), [60, 62, 60, 64, 65])

    def test_da_capo_al_fine(self):
        ms = [measure(1, [60]), measure(2, [62], direction='<sound fine="yes"/>'),
              measure(3, [64], direction='<sound dacapo="yes"/>')]
        self.assertEqual(self.seq(*ms), [60, 62, 64, 60, 62])

    def test_dal_segno_al_coda_via_sound_attributes(self):
        ms = [measure(1, [60]), measure(2, [62], direction='<sound segno="s"/>'),
              measure(3, [64], direction='<sound tocoda="c"/>'),
              measure(4, [65], direction='<sound dalsegno="s"/>'),
              measure(5, [67], direction='<sound coda="c"/>')]
        # 1 2 3 4, jump to segno: 2 3, then "To Coda" -> 5
        self.assertEqual(self.seq(*ms), [60, 62, 64, 65, 62, 64, 67])

    def test_navigation_from_text_only(self):
        words = lambda t: f"<direction-type><words>{t}</words></direction-type>"
        ms = [measure(1, [60]), measure(2, [62], direction=words("Fine")),
              measure(3, [64], direction=words("D.C. al Fine"))]
        self.assertEqual(self.seq(*ms), [60, 62, 64, 60, 62])

    def test_repeats_are_skipped_after_da_capo(self):
        ms = [measure(1, [60], right='<repeat direction="backward"/>'),
              measure(2, [62], direction='<sound dacapo="yes"/>')]
        self.assertEqual(self.seq(*ms), [60, 60, 62, 60, 62])

    def test_tie_across_a_repeat_jump_and_timing_of_repeated_notes(self):
        s = self.played(measure(1, [60], right='<repeat direction="backward"/>'), measure(2, [62]))
        self.assertEqual([n.onset for n in s.notes], [0.0, 0.5, 1.0])   # 120 bpm, repeat shifts time
        self.assertEqual([n.measure for n in s.notes], [1, 1, 2])       # written measure numbers kept

    def test_double_stops(self):
        s = self.played(measure(1, [(55, 62, 69), 60]))
        self.assertEqual([n.midi_pitch for n in s.notes], [55, 62, 69, 60])
        self.assertEqual(len({n.onset for n in s.notes[:3]}), 1)         # triple stop = same onset
        self.assertEqual(s.notes[3].beat, 2.0)                           # chord advances time once

    def test_tempo_change(self):
        s = self.played(measure(1, [60, 60], direction='<sound tempo="60"/>'),
                        measure(2, [60], direction='<sound tempo="120"/>'))
        # measure 1 holds 2 quarters at 60 bpm (2 s), so measure 2 starts at 2 s
        self.assertEqual([n.onset for n in s.notes], [0.0, 1.0, 2.0])
        self.assertEqual(s.tempo, 60.0)

    def test_measure_repeat_sign_and_empty_measure(self):
        sign = '<attributes><measure-style><measure-repeat type="start">1</measure-repeat></measure-style></attributes>'
        s = self.played(measure(1, [60, 62]), measure(2, [], attributes=sign), measure(3, [64]))
        self.assertEqual([n.midi_pitch for n in s.notes], [60, 62, 60, 62, 64])
        self.assertEqual(s.notes[-1].beat, 1.0)

    def test_multi_measure_repeat_sign(self):
        sign = lambda t: f'<attributes><measure-style><measure-repeat type="{t}">2</measure-repeat></measure-style></attributes>'
        s = self.played(measure(1, [60]), measure(2, [62]), measure(3, [], attributes=sign("start")),
                        measure(4, [], attributes=sign("stop")), measure(5, [64]))
        self.assertEqual([n.midi_pitch for n in s.notes], [60, 62, 60, 62, 64])
        self.assertEqual([n.measure for n in s.notes], [1, 2, 3, 4, 5])   # copies keep their own numbers
        self.assertEqual(s.notes[-1].onset, 2.0)   # 4 one-quarter measures at 120 bpm = 2 s

    def test_pickup_measure_numbering(self):
        s = self.played(measure(0, [60]).replace("<measure number=\"0\">", '<measure number="0" implicit="yes">'),
                        measure(1, [62]))
        self.assertEqual([n.measure for n in s.notes], [0, 1])

    def test_violin_part_is_chosen_from_a_multi_part_score(self):
        xml = score_xml(measure(1, [60]), part_list='<score-part id="P1"><part-name>Piano</part-name></score-part>'
                        '<score-part id="P2"><part-name>Violin</part-name></score-part>')
        xml = xml.replace("</score-partwise>", '<part id="P2">' + measure(1, [76]) + "</part></score-partwise>")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.musicxml"
            p.write_text(xml)
            self.assertEqual(parse_musicxml(p).notes[0].midi_pitch, 76)
            self.assertEqual(parse_musicxml(p, part="P1").notes[0].midi_pitch, 60)


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
