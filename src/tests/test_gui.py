"""GUI: SVG charts, the explained report, and the upload server (end to end over HTTP)."""
import json
import os
import re
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

from helpers import make_video, musicxml, synth, write_wav
from minuet import minuet_musicxml, play_minuet
from pose_fixture import minuet_notes, synth_pose, write_fake_viopose
from maestro import analyze
from maestro.gui import charts
from maestro.gui.report import build_report, standalone_page
from maestro.gui.server import make_server
from maestro.vision import viopose_setup
from maestro.vision.pose_io import save_pose

MIDIS = [69, 71, 72, 74, 76, 74, 72, 71]


def performance(d, midis=MIDIS, score=MIDIS):
    """Analyze a synthetic recording (note 3 sharp, note 5 late) of `midis` against `score`."""
    perf = [(m + (0.3 if i == 3 else 0), 0.3 + 0.5 * i + (0.12 if i == 5 else 0), 0.4) for i, m in enumerate(midis)]
    (d / "s.musicxml").write_text(musicxml(score))
    write_wav(d / "a.wav", synth(perf))
    make_video(d / "v.mp4", d / "a.wav", 5)
    return analyze(d / "v.mp4", d / "s.musicxml", d / "out")


def svgs(html):
    """Every <svg>...</svg> block of an HTML fragment."""
    out, i = [], 0
    while (i := html.find("<svg", i)) != -1:
        j = html.index("</svg>", i) + 6
        out.append(html[i:j])
        i = j
    return out


class Report(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.perf = performance(Path(cls._tmp.name))
        cls.html = build_report(cls.perf)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_result_has_a_card_and_an_explained_chart(self):
        for text in ("Average pitch error", "Overall tuning", "Wrong notes", "Tempo", "Average timing error",
                     "Timing unevenness", "Score match", "How to read it"):
            self.assertIn(text, self.html)
        # match, 2 engraved panels, piano roll, pitch trace, intonation, tempo, timing + 4 assessment charts
        self.assertEqual(len(svgs(self.html)), 12)
        self.assertEqual(self.html.count('class="how"'), 12)   # 11 explained charts + the "movement analysis not available" notice
        for text in ("Assessment: score per skill", "Assessment: when things went wrong", "Assessment: bars to practise",
                     "Assessment: why it went wrong", "maestro-assessment.json"):
            self.assertIn(text, self.html)
        self.assertIn("Movement analysis", self.html)

    def test_assessment_charts_show_the_events(self):
        report = self.perf.assessment["synthesis"]
        events = [dict(e, severity=0.9, causes=[], measure=2) for e in report["events"][:1]] or [
            {"module": "intonation", "category": "sharp", "start": 1.0, "end": 1.4, "severity": 0.9, "message": "C is sharp.",
             "causes": [], "measure": 2}]
        timeline = charts.event_timeline_chart(events)
        ET.fromstring(timeline)
        self.assertIn(charts.BAD, timeline)                                  # severity 0.9 is drawn red
        self.assertIn("intonation", charts.measure_heatmap(events))
        self.assertIn("Nothing could be assessed", charts.module_scores_chart({"intonation": None}))
        self.assertIn("No problems", charts.event_timeline_chart([]))

    def test_all_charts_are_valid_xml(self):
        for s in svgs(self.html):
            ET.fromstring(s)

    def test_charts_show_the_injected_errors(self):
        i = self.perf.assessment["intonation"]["notes"]
        self.assertRegex(charts.intonation_chart(self.perf.assessment["intonation"]), r"note 3: \+3[0-2] cents")  # injected +30
        late = {int(n): int(v) for n, v in re.findall(r"note (\d): ([+-]\d+) ms",
                                                    charts.timing_chart(self.perf.assessment["rhythm"]))}
        self.assertEqual(max(late, key=late.get), 5)     # the injected 120 ms delay (partly absorbed by the fit)
        self.assertGreater(late[5], 60)
        self.assertEqual(len(i), len(MIDIS))

    def test_standalone_page(self):
        page = standalone_page(self.html)
        self.assertTrue(page.startswith("<!doctype html>") and "<style>" in page)

    def test_charts_degrade_gracefully_without_data(self):
        for s in (charts.intonation_chart({"notes": [], "tuning_offset_cents": None}),
                  charts.timing_chart({"notes": []}),
                  charts.tempo_chart([], [], {"notes": [], "tempo_ratio": None}),
                  charts.pitch_trace_chart([{"time": 0.0, "estimated_pitch": 0.0}], [], [], [])):
            ET.fromstring(s)                    # still valid SVG
            self.assertIn("<text", s)           # with an explanatory message


class VisualReport(unittest.TestCase):
    """The movement section of the report (pose only: the audio side is independent). The pose here comes from a saved
    pose.json; the Server tests below get it from VioPose the way a real run does."""
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        d = cls.d = Path(cls._tmp.name)
        (d / "m.musicxml").write_text(minuet_musicxml())
        write_wav(d / "a.wav", play_minuet())
        make_video(d / "v.mp4", d / "a.wav", 18)
        save_pose(synth_pose(minuet_notes(), with_bow=True), d / "pose.json")
        cls.perf = analyze(d / "v.mp4", d / "m.musicxml", pose_file=d / "pose.json")
        cls.html = build_report(cls.perf)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_movement_cards_and_sections(self):
        for text in ("Movement summary", "Bow strokes", "Bow speed evenness", "Smoothness within strokes", "Down/up balance",
                     "Torso sway", "Shoulder tilt", "Left-hand travel", "Bow movement", "Arm angles",
                     "Left-hand position as pitch (video only)"):
            self.assertIn(text, self.html)
        self.assertNotIn("Not available", self.html)

    def test_visual_charts_are_valid_svg(self):
        svgs_ = svgs(self.html)
        self.assertEqual(len(svgs_), 12 + 4)                           # the 12 audio/score/assessment figures + bow, strokes, arms, visual pitch
        for s in svgs_:
            ET.fromstring(s)

    def test_visual_pitch_chart_shows_the_hand_derived_pitch(self):
        vp = self.perf.assessment["visual_pitch"]
        chart = charts.visual_pitch_chart(vp)
        ET.fromstring(chart)
        self.assertIn("semitones above the open string", chart)
        self.assertIn("estimated from hand position", chart)
        self.assertGreater(vp["range_semitones"], 4)                   # the Minuet's hand travels well over a fourth
        self.assertIn("string-length law", self.html)

    def test_without_a_pose_source_the_section_explains_why(self):
        isolate_from_real_install(self, self.d)
        perf = analyze(self.d / "v.mp4", self.d / "m.musicxml")        # no pose file, VioPose not installed
        html = build_report(perf)
        self.assertFalse(perf.assessment["visual_status"]["available"])
        self.assertIn("setup_viopose.py", html)                         # tells the user how to install it
        self.assertNotIn("Movement summary", html)
        self.assertIn("Average pitch error", html)                      # the audio results are unaffected

    def test_charts_degrade_gracefully(self):
        for chart in (charts.bow_chart({"bow": None}), charts.stroke_chart({"strokes": []}), charts.arm_chart({"elbow_angle": {}}),
                      charts.visual_pitch_chart({"available": False, "reason": "left hand not tracked"})):
            ET.fromstring(chart)


def isolate_from_real_install(test, tmp):
    """Keep a VioPose really installed on this machine out of the test."""
    for name, value in (("CONFIG_PATH", tmp / "no_config.json"), ("DEFAULT_DIR", tmp / "no_default_checkout")):
        patcher = mock.patch.object(viopose_setup, name, value)
        patcher.start()
        test.addCleanup(patcher.stop)
    patcher = mock.patch.dict(os.environ)
    patcher.start()
    test.addCleanup(patcher.stop)
    for k in ("VIOPOSE_DIR", "VIOPOSE_PYTHON"):
        os.environ.pop(k, None)


def multipart(files):
    """files: {field: (filename, bytes)} -> (content_type, body)."""
    b = b"----maestrotest"
    body = b""
    for field, (name, data) in files.items():
        body += (b"--" + b + b'\r\nContent-Disposition: form-data; name="' + field.encode() + b'"; filename="'
                 + name.encode() + b'"\r\nContent-Type: application/octet-stream\r\n\r\n' + data + b"\r\n")
    return "multipart/form-data; boundary=" + b.decode(), body + b"--" + b + b"--\r\n"


class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = make_server(port=0)
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def post(self, files):
        ctype, body = multipart(files)
        req = urllib.request.Request(self.url + "/api/analyze", body, {"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def upload(self, score_midis):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            wav = d / "a.wav"
            write_wav(wav, synth([(m, 0.3 + 0.5 * i, 0.4) for i, m in enumerate(MIDIS)]))
            return self.post({"video": ("take1.wav", wav.read_bytes()),
                              "score": ("piece.musicxml", musicxml(score_midis).encode())})

    def test_viopose_generates_the_pose_for_a_video_upload(self):
        """Upload a video and a score, nothing else: the installed VioPose makes the pose and the report uses it."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            isolate_from_real_install(self, d)
            root = write_fake_viopose(d / "viopose", synth_pose(minuet_notes(), with_bow=True, noise_mm=0))
            with mock.patch.object(viopose_setup, "CONFIG_PATH", d / "viopose.json"):
                viopose_setup.write_config(root, "python3")                         # what setup.ps1 / setup.sh record
                wav = d / "a.wav"
                write_wav(wav, synth([(m, 0.3 + 0.5 * i, 0.4) for i, m in enumerate(MIDIS)]))
                make_video(d / "v.mp4", wav, 5)
                status, data = self.post({"video": ("take.mp4", (d / "v.mp4").read_bytes()),
                                          "score": ("piece.musicxml", musicxml(MIDIS).encode())})
        self.assertEqual(status, 200)
        for text in ("Movement summary", "Bow movement", "Left-hand position as pitch"):
            self.assertIn(text, data["html"])

    def test_extra_form_fields_cannot_choose_paths_or_inject_a_pose(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            isolate_from_real_install(self, d)
            wav = d / "a.wav"
            write_wav(wav, synth([(m, 0.3 + 0.5 * i, 0.4) for i, m in enumerate(MIDIS)]))
            save_pose(synth_pose(minuet_notes(), with_bow=True), d / "pose.json")
            escape = Path(tempfile.gettempdir()) / "escape.txt"
            try:
                status, data = self.post({"video": ("take.wav", wav.read_bytes()),
                                          "score": ("piece.musicxml", musicxml(MIDIS).encode()),
                                          "pose": ("pose.json", (d / "pose.json").read_bytes()),     # no such upload any more
                                          "../escape": ("x.txt", b"must never be written outside the upload folder")})
                self.assertFalse(escape.exists())                      # a crafted field name cannot choose the path
            finally:
                escape.unlink(missing_ok=True)
        self.assertEqual(status, 200)
        self.assertNotIn("Movement summary", data["html"])             # the extra pose was ignored; no VioPose here

    def test_index_has_upload_form(self):
        with urllib.request.urlopen(self.url + "/") as r:
            page = r.read().decode()
        self.assertIn('name="video"', page)
        self.assertIn('name="score"', page)
        self.assertNotIn('name="pose"', page)                       # the pose is generated, never uploaded
        self.assertNotIn("mocap", page.lower())
        self.assertNotIn("/*CSS*/", page)              # stylesheet was inlined

    def test_upload_returns_report(self):
        status, data = self.upload(MIDIS)
        self.assertEqual(status, 200)
        self.assertIn("Intonation", data["html"])

    def test_wrong_score_returns_readable_error(self):
        status, data = self.upload([60, 62, 64, 65, 55, 57, 59, 60])
        self.assertEqual(status, 400)
        self.assertIn("does not appear to match", data["error"])

    def test_missing_file_and_unknown_route(self):
        status, data = self.post({"score": ("s.musicxml", musicxml([69]).encode())})
        self.assertEqual(status, 400)
        self.assertIn("both", data["error"])
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.url + "/nope")
        self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
