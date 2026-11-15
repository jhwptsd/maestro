"""GUI: SVG charts, the explained report, and the upload server (end to end over HTTP)."""
import json
import re
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from helpers import make_video, musicxml, synth, write_wav
from maestro import analyze
from maestro.gui import charts
from maestro.gui.report import build_report, standalone_page
from maestro.gui.server import make_server

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
        self.assertEqual(len(svgs(self.html)), 6)      # match, score, pitch trace, intonation, tempo, timing
        self.assertEqual(self.html.count('class="how"'), 6)

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

    def test_index_has_upload_form(self):
        with urllib.request.urlopen(self.url + "/") as r:
            page = r.read().decode()
        self.assertIn('name="video"', page)
        self.assertIn('name="score"', page)
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
