"""Sheet-music engraving (gui/notation.py) and the side-by-side score viewer (gui/viewer.py)."""
import re
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from helpers import make_video, synth, write_wav
from minuet import minuet_musicxml
from maestro import analyze
from maestro.gui.notation import Glyph, infer_key, key_alters, layout_score, render_panel, spell
from maestro.gui.report import VIEWER_JS, build_report, standalone_page
from maestro.gui.viewer import build_viewer
from maestro.score.musicxml import parse_musicxml
from maestro.score.score_representation import Note, Score

NS = {"s": "http://www.w3.org/2000/svg"}


def make_notes(spec, bpm=120):
    """spec: [(midi, quarters)] in 4/4 -> Notes with measure/beat/onset filled in (midi=None is a rest)."""
    notes, q = [], 0.0
    for midi, dur in spec:
        if midi is not None:
            notes.append(Note(int(q // 4) + 1, 1 + q % 4, "X", midi, dur, None, q * 60 / bpm, (q + dur) * 60 / bpm))
        q += dur
    return notes


def draw(notes, fifths=0, time_sig=(4, 4), glyph=None):
    layout = layout_score(notes, fifths, time_sig)
    svg = render_panel(layout, glyph or (lambda i: Glyph(notes[i].midi_pitch)))
    return layout, svg, ET.fromstring(svg)


def count(root, tag, cls=None):
    return sum(1 for e in root.iter(f"{{{NS['s']}}}{tag}") if cls is None or e.get("class") == cls)


class Spelling(unittest.TestCase):
    def test_key_inference_and_signatures(self):
        g_major = [67, 69, 71, 72, 74, 76, 78, 79]                     # G A B C D E F# G
        self.assertEqual(infer_key(g_major), 1)
        self.assertEqual(infer_key([60, 62, 64, 65, 67, 69, 71]), 0)
        self.assertEqual(infer_key([70, 72, 74, 75, 77, 79, 81]), -2)   # Bb major
        self.assertEqual(key_alters(2), [1, 0, 0, 1, 0, 0, 0])      # D major: C# and F#
        self.assertEqual(key_alters(-2), [0, 0, -1, 0, 0, 0, -1])    # Bb major: Eb and Bb

    def test_spelling_follows_the_key(self):
        pos = lambda m, f: spell(m, f)
        self.assertEqual(pos(66, 1), (31, 1))      # F#4 in G major: F line/space with the key's sharp
        self.assertEqual(pos(66, 0), (31, 1))      # sharps by default
        self.assertEqual(pos(61, -3), (29, -1))    # Db4 in a flat key, not C#
        self.assertEqual(pos(60, 0), (28, 0))      # middle C sits on the first ledger line below the staff
        self.assertEqual(pos(71, 0), (34, 0))      # B4 = middle line


class Engraving(unittest.TestCase):
    def test_structure_of_a_simple_melody(self):
        notes = make_notes([(69, 1), (71, 1), (72, 1), (74, 1), (76, 2), (74, 2)])      # 2 bars of 4/4
        layout, svg, root = draw(notes)
        self.assertEqual(count(root, "line", "staff"), 5 * len(layout.systems))
        self.assertEqual(count(root, "ellipse", "head"), len(notes))
        self.assertIn("serif", svg)                                  # time signature digits
        self.assertEqual(len(layout.head_x), len(notes))
        xs = [layout.head_x[i] for i in range(len(notes))]
        self.assertEqual(xs, sorted(xs))                              # left to right in time (single system here)

    def test_beams_flags_and_rests(self):
        beamed = make_notes([(72, 0.5), (74, 0.5), (76, 1), (77, 2)])           # two eighths share a beam
        _, svg, root = draw(beamed)
        self.assertEqual(count(root, "line", "beam"), 1)
        lone = make_notes([(72, 0.5), (None, 0.5), (76, 1), (77, 2)])           # a lone eighth gets a flag, then a rest
        _, svg, root = draw(lone)
        self.assertEqual(count(root, "line", "beam"), 0)
        self.assertIn('stroke-linecap="round"/>', svg)                         # flag path
        self.assertTrue(re.search(r"<(path|circle|rect)[^>]*stroke=\"#222\"", svg))   # rest glyph drawn

    def test_ledger_lines_accidentals_and_carry_over(self):
        low = make_notes([(55, 1)])                                           # G3: two ledger lines below the staff
        _, svg, root = draw(low)
        ledgers = [e for e in root.iter(f"{{{NS['s']}}}line") if e.get("stroke-width") == "0.9"]
        self.assertEqual(len(ledgers), 2)
        marks = lambda svg: svg.count('stroke-width="2"')                      # one per sharp or natural sign
        notes = make_notes([(66, 1), (66, 1), (65, 1), (66, 1)])               # F# F# F F# in C major, one bar
        svg = render_panel(layout_score(notes, 0, (4, 4)), lambda i: Glyph(notes[i].midi_pitch))
        self.assertEqual(marks(svg), 3)                                        # sharp, (carried: none), natural, sharp
        self.assertEqual(count(ET.fromstring(svg), "ellipse", "head"), 4)

    def test_key_signature_replaces_accidentals(self):
        notes = make_notes([(66, 4)] * 4)                                      # F# in four different bars
        in_c = render_panel(layout_score(notes, 0, (4, 4)), lambda i: Glyph(notes[i].midi_pitch))
        in_g = render_panel(layout_score(notes, 1, (4, 4)), lambda i: Glyph(notes[i].midi_pitch))
        marks = lambda svg: svg.count('stroke-width="2"')
        self.assertEqual(marks(in_c), 4)                                       # a sharp sign in every bar
        self.assertEqual(marks(in_g), 1)                                       # only the key signature's sharp (per system)

    def test_chords_and_dots(self):
        chord = [Note(1, 1.0, "X", m, 1.5, None, 0.0, 0.75) for m in (60, 64, 67, 69)]   # dotted quarter chord
        _, svg, root = draw(chord)
        heads = [e for e in root.iter(f"{{{NS['s']}}}ellipse")]
        self.assertEqual(len(heads), 4)
        self.assertGreater(len({round(float(h.get("cx")), 1) for h in heads}), 1)       # G/A seconds are offset sideways
        self.assertEqual(count(root, "circle"), 4)                                       # one dot per head

    def test_long_pieces_wrap_into_systems_and_repeats_are_separate_measures(self):
        notes = make_notes([(60 + k % 12, 1) for k in range(64)])             # 16 bars
        layout, _, root = draw(notes)
        self.assertGreater(len(layout.systems), 3)
        self.assertTrue(all(s.header_w > 0 for s in layout.systems))
        rep = make_notes([(60, 1), (62, 1), (64, 1), (65, 1)])
        rep += [Note(1, n.beat, "X", n.midi_pitch, 1, None, n.onset + 2, n.offset + 2) for n in rep]   # measure 1 played again
        self.assertEqual(sum(len(m.slots) for s in layout_score(rep, 0, (4, 4)).systems for m in s.measures), 8)
        self.assertEqual(len([m for s in layout_score(rep, 0, (4, 4)).systems for m in s.measures]), 2)


def fake_perf(statuses):
    """A Performance-like object over 8 written notes; statuses[i] in ok/warn/bad/wrong/missed, 'unplayed' outside."""
    notes = make_notes([(60 + 2 * k % 9 + 64 - 60, 1) for k in range(8)])
    played, align, inton = [], [], []
    for i, st in enumerate(statuses):
        if st in ("missed", "unplayed"):
            align.append({"score_idx": i, "perf_idx": None, "cost": None})
            continue
        cents = {"ok": 3.0, "warn": 18.0, "bad": -40.0, "wrong": 300.0}[st]
        midi = notes[i].midi_pitch + cents / 100
        played.append({"onset": notes[i].onset + 0.5, "offset": notes[i].offset + 0.5, "midi": midi, "hz": 440 * 2 ** ((midi - 69) / 12)})
        align.append({"score_idx": i, "perf_idx": len(played) - 1, "cost": 0})
        inton.append({"score_idx": i, "perf_idx": len(played) - 1, "cents": cents, "wrong_note": st == "wrong"})
    played.append({"onset": notes[3].onset + 0.55, "offset": notes[3].onset + 0.8, "midi": 79.0, "hz": 784.0})   # extra
    played.sort(key=lambda p: p["onset"])
    matched_idx = [i for i, s in enumerate(statuses) if s not in ("unplayed",)]
    # re-number perf_idx after sorting
    for a in align:
        if a["perf_idx"] is not None:
            a["perf_idx"] = next(k for k, p in enumerate(played) if p["onset"] == notes[a["score_idx"]].onset + 0.5)
    for d in inton:
        d["perf_idx"] = next(a["perf_idx"] for a in align if a["score_idx"] == d["score_idx"])
    return SimpleNamespace(score=Score(notes, 120, 0, (4, 4)), audio_features={"notes": played}, alignment=align,
                           assessment={"intonation": {"notes": inton},
                                       "section": {"first_note": matched_idx[0], "last_note": matched_idx[-1],
                                                   "first_measure": 1, "last_measure": 2, "coverage": 1.0}})


def statuses_of(html, panel):
    """data-status of every note in the given panel (0 = left, 1 = right)."""
    svgs = re.findall(r"<svg.*?</svg>", html, re.S)
    return re.findall(r'data-status="(\w+)"', svgs[panel])


class SideBySide(unittest.TestCase):
    STATUSES = ["ok", "warn", "bad", "wrong", "missed", "ok", "ok", "unplayed"]

    @classmethod
    def setUpClass(cls):
        cls.html = build_viewer(fake_perf(cls.STATUSES))

    def test_both_panels_are_valid_svg_with_the_same_notes(self):
        panels = re.findall(r"<svg.*?</svg>", self.html, re.S)
        self.assertEqual(len(panels), 2)
        roots = [ET.fromstring(p) for p in panels]
        self.assertEqual([count(r, "line", "staff") for r in roots][0], count(roots[1], "line", "staff"))
        self.assertEqual(statuses_of(self.html, 0), ["written"] * 8)
        self.assertIn("Score as the app reads it", panels[0])
        self.assertIn("What was detected in your playing", panels[1])

    def test_detected_notes_are_classified(self):
        right = statuses_of(self.html, 1)
        for status, n in {"ok": 3, "warn": 1, "bad": 1, "wrong": 1, "missed": 1, "unplayed": 1, "extra": 1}.items():
            self.assertEqual(right.count(status), n, status)
        self.assertIn("+18c", self.html)
        self.assertIn("-40c", self.html)
        self.assertNotIn("+3c", self.html)                         # in-tune notes carry no label
        self.assertIn("wrong", self.html)
        self.assertIn("1 wrong, 1 not detected, 1 outside the part you played; 1 extra note detected", self.html)

    def test_partner_notes_line_up_and_are_linked(self):
        panels = re.findall(r"<svg.*?</svg>", self.html, re.S)
        cx = []
        for p in panels:
            by_pair = {}
            for g in ET.fromstring(p).iter(f"{{{NS['s']}}}g"):
                if g.get("class") == "note" and g.get("data-pair", "").startswith("s"):
                    head = next(e for e in g.iter(f"{{{NS['s']}}}ellipse"))
                    by_pair[g.get("data-pair")] = float(head.get("cx"))
            cx.append(by_pair)
        self.assertEqual(set(cx[0]), set(cx[1]))                    # every written note has a partner on the right
        for pair in cx[0]:
            self.assertAlmostEqual(cx[0][pair], cx[1][pair], places=1)   # same layout: same x

    def test_wrong_note_is_drawn_where_it_was_played(self):
        panels = re.findall(r"<svg.*?</svg>", self.html, re.S)
        cy = lambda svg: {g.get("data-pair"): float(next(g.iter(f"{{{NS['s']}}}ellipse")).get("cy"))
                          for g in ET.fromstring(svg).iter(f"{{{NS['s']}}}g") if g.get("class") == "note" and g.get("data-pair", "").startswith("s")}
        left, right = cy(panels[0]), cy(panels[1])
        self.assertNotAlmostEqual(left["s3"], right["s3"], places=0)      # played a minor third away: different line
        self.assertAlmostEqual(left["s0"], right["s0"], places=1)         # in tune: same line

    def test_legend_and_summary_present(self):
        for text in ("in tune", "wrong note", "not detected", "extra note", "not played", "written notes"):
            self.assertIn(text, self.html)


class RealAnalysis(unittest.TestCase):
    def test_report_has_the_viewer_with_key_and_time_signature(self):
        first_notes = [74, 67, 69, 71, 72, 74, 67, 67]                    # the Minuet's first bar and a half
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            xml = minuet_musicxml().replace("<attributes><divisions>2</divisions>",
                                            "<attributes><divisions>2</divisions><key><fifths>1</fifths></key>")
            (d / "s.musicxml").write_text(xml)
            sc = parse_musicxml(d / "s.musicxml")
            self.assertEqual((sc.key_fifths, sc.time_signature), (1, (3, 4)))
            write_wav(d / "a.wav", synth([(m, 0.4 + 0.5 * i, 0.4) for i, m in enumerate(first_notes)]))
            make_video(d / "v.mp4", d / "a.wav", 5)
            perf = analyze(d / "v.mp4", d / "s.musicxml")
        html = build_report(perf)
        self.assertIn("Score vs what was played", html)
        panels = re.findall(r"<svg.*?</svg>", html.split("Score vs what was played")[1], re.S)[:2]
        for p in panels:
            ET.fromstring(p)
        self.assertEqual(Score.from_json(sc.to_json()), sc)               # key/time signature survive JSON

    def test_page_carries_the_hover_script(self):
        self.assertIn("data-pair", VIEWER_JS)
        self.assertIn("<script>", standalone_page("x"))
        self.assertIn(VIEWER_JS.strip()[:30], standalone_page("x"))


@unittest.skipUnless(shutil.which("node") and subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode == 0,
                     "needs node + playwright")
class HoverInBrowser(unittest.TestCase):
    def test_hovering_a_note_highlights_its_partner(self):
        with tempfile.TemporaryDirectory() as d:
            page = Path(d) / "r.html"
            page.write_text(standalone_page(build_viewer(fake_perf(["ok"] * 8))))
            js = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();"
                  "const p=await b.newPage({viewport:{width:1300,height:900}});await p.goto('file://'+process.argv[1]);"
                  "await p.hover('.vcol:first-child .note[data-pair=\"s2\"] .head');"
                  "const n=await p.evaluate(()=>[...document.querySelectorAll('.note.hl')].map(e=>e.closest('.vcol')===document.querySelector('.vcol:first-child')?'L':'R').join(''));"
                  "await p.mouse.move(0,0);const m=await p.evaluate(()=>document.querySelectorAll('.note.hl').length);"
                  "console.log(n,m);await b.close();})()")
            out = subprocess.run(["node", "-e", js, str(page)], capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual(sorted(out[0]), ["L", "R"])                    # the note and its partner, nothing else
        self.assertEqual(out[1], "0")                                    # highlight clears when the pointer leaves


if __name__ == "__main__":
    unittest.main()
