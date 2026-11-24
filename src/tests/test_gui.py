"""GUI: the player page data, the page itself and the upload server (end to end over HTTP)."""
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np

from helpers import make_video, musicxml, synth, write_wav
from minuet import minuet_musicxml, play_minuet
from pose_fixture import minuet_notes, synth_pose, write_fake_viopose
from maestro import analyze
from maestro.gui.report import build_page, copy_media
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


def page_data(html):
    """The JSON the page embeds (`const D = {...};`)."""
    return json.loads(re.search(r"const D = (\{.*\});\n", html).group(1).replace("<\\/", "</"))


class Page(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.perf = performance(Path(cls._tmp.name))
        cls.html = build_page(cls.perf, "media.mp4")
        cls.data = page_data(cls.html)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_page_plays_the_recording_and_carries_all_the_data(self):
        d = self.data
        self.assertEqual(d["media"], {"url": "media.mp4", "kind": "video"})
        self.assertGreater(d["duration"], 4)
        self.assertEqual(set(d["traces"]), {"pitch", "loudness", "notes"})        # no pose here: no bow/posture traces
        self.assertIsNone(d["pose"])
        self.assertEqual(len(d["traces"]["pitch"]["t"]), len(d["traces"]["pitch"]["midi"]))
        for key in ("scores", "events", "causes", "links", "unavailable"):
            self.assertIn(key, d)

    def test_the_injected_errors_are_in_the_traces(self):
        notes = self.data["traces"]["notes"]
        self.assertAlmostEqual(notes[3]["cents"], 30, delta=8)                    # note 3 was played 30 cents sharp
        late = max(notes, key=lambda n: abs(n["ms"] or 0))
        self.assertEqual(notes.index(late), 5)                                    # note 5 was played late
        self.assertEqual(self.data["traces"]["pitch"]["score"][0]["midi"], MIDIS[0])

    def test_nothing_but_numbers_and_slugs(self):
        for e in self.data["events"]:
            self.assertEqual(set(e), {"id", "module", "category", "start", "end", "measure", "beat", "severity", "value",
                                      "expected", "unit", "causes"})
            self.assertTrue(all(isinstance(c, str) and " " not in c and isinstance(p, float) for c, p in e["causes"]))
        for gone in ("How to read it", "explanation", "hypothesis", '"message"'):             # (the page script itself reads e.message)
            self.assertNotIn(gone, self.html)

    def test_the_score_viewer_is_kept_but_folded_away(self):
        self.assertIn("<details><summary>Score as read vs score as played</summary>", self.html)
        self.assertIn('class="viewer"', self.html)

    def test_data_cannot_close_the_script_tag(self):
        self.perf.assessment["synthesis"]["events"].append(
            {"id": "e9999", "module": "tone", "category": "</script><b>x", "start": 0, "end": 0, "measure": None, "beat": None,
             "severity": 0.1, "value": None, "expected": None, "unit": None, "causes": []})
        try:
            html = build_page(self.perf, "media.mp4")
        finally:
            self.perf.assessment["synthesis"]["events"].pop()
        self.assertEqual(html.count("</script>"), 1)                              # the page's own script only
        self.assertEqual(page_data(html)["events"][-1]["category"], "</script><b>x")


class VisualPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        d = cls.d = Path(cls._tmp.name)
        (d / "m.musicxml").write_text(minuet_musicxml())
        write_wav(d / "a.wav", play_minuet())
        make_video(d / "v.mp4", d / "a.wav", 18)
        cls.pose = synth_pose(minuet_notes(), with_bow=True)
        save_pose(cls.pose, d / "pose.json")
        cls.perf = analyze(d / "v.mp4", d / "m.musicxml", pose_file=d / "pose.json")
        cls.data = page_data(build_page(cls.perf, "media.mp4"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_movement_traces_and_skeleton_share_the_recording_clock(self):
        t, d = self.data["traces"], self.data
        self.assertTrue({"bow", "posture", "hand"} <= set(t))
        self.assertEqual(len(t["bow"]["t"]), len(self.pose.xyz))
        self.assertGreater(len(t["bow"]["strokes"]), 10)
        self.assertEqual(len(d["pose"]["xyz"]), len(self.pose.xyz))                # one skeleton per video frame
        self.assertEqual(d["pose"]["fps"], self.pose.fps)
        self.assertTrue(all(a in d["pose"]["joints"] and b in d["pose"]["joints"] for a, b in d["pose"]["edges"]))

    def test_the_skeleton_is_upright_whichever_way_the_pose_y_axis_points(self):
        from maestro.gui.player import _pose
        from maestro.utils.Performance import Performance
        from maestro.vision.pose_io import Pose
        joints = ["l_shoulder", "r_shoulder", "l_pelvis", "r_pelvis", "l_knee"]
        up = np.array([[[-20, 150, 0], [20, 150, 0], [-10, 100, 0], [10, 100, 0], [-10, 50, 0]]] * 3, dtype=float) * 10   # mm
        for name, xyz in (("y up", up), ("y down", up * np.array([1, -1, 1]))):      # the same person, both conventions
            perf = Performance()
            perf.pose = Pose(30.0, joints, xyz)
            frame = _pose(perf)["xyz"][0]
            self.assertGreater(frame[0][1], frame[2][1], name)                        # shoulders above the pelvis ...
            self.assertGreater(frame[2][1], frame[4][1], name)                        # ... above the knee
            self.assertEqual(frame[0][0], -20.0, name)                               # left stays left (x is not mirrored)

    def test_every_skill_with_a_result_has_a_score(self):
        self.assertTrue({"intonation", "rhythm", "expression", "tone", "bowing", "posture", "overall"} <= set(self.data["scores"]))


HAS_BROWSER = bool(shutil.which("node") and subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode == 0)


@unittest.skipUnless(HAS_BROWSER, "needs node + playwright")
class PageInBrowser(unittest.TestCase):
    """The page really runs: no script errors, the live charts and skeleton are drawn, clicking an event seeks."""

    def test_charts_skeleton_and_seeking_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "m.musicxml").write_text(minuet_musicxml())
            write_wav(d / "a.wav", play_minuet())
            make_video(d / "v.mp4", d / "a.wav", 18)
            save_pose(synth_pose(minuet_notes(), with_bow=True), d / "pose.json")
            perf = analyze(d / "v.mp4", d / "m.musicxml", d / "out", pose_file=d / "pose.json")
            (d / "out" / "r.html").write_text(build_page(perf, copy_media(perf, d / "out")), encoding="utf-8")
            js = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();"
                  "const errors=[];p.on('pageerror',e=>errors.push(String(e)));await p.goto('file://'+process.argv[1]);"
                  "await p.waitForTimeout(500);await p.evaluate(()=>{media.currentTime=8;draw();});await p.waitForTimeout(300);"
                  "const inked=await p.evaluate(()=>[...document.querySelectorAll('canvas')].map(c=>{const g=c.getContext('2d');"
                  "const px=g.getImageData(0,0,c.width,c.height).data;let n=0;for(let i=3;i<px.length;i+=4)if(px[i])n++;return n>0;}));"
                  "const rows=await p.evaluate(()=>document.querySelectorAll('#events tbody tr').length);"
                  "let seeked=null;if(rows){const start=await p.evaluate(()=>D.events.slice().sort((a,b)=>b.severity-a.severity)[0].start);"
                  "await p.click('#events tbody tr');await p.waitForTimeout(300);"
                  "seeked=await p.evaluate(s=>Math.abs(media.currentTime-Math.max(0,s-1))<0.3,start);}"
                  "console.log(JSON.stringify({errors,inked,rows,seeked}));await b.close();})()")
            out = json.loads(subprocess.run(["node", "-e", js, str(d / "out" / "r.html")], capture_output=True, text=True,
                                            check=True).stdout)
        self.assertEqual(out["errors"], [])
        self.assertTrue(all(out["inked"]), out["inked"])                     # every chart, the timeline and the skeleton
        self.assertGreaterEqual(len(out["inked"]), 8)
        if out["rows"]:
            self.assertTrue(out["seeked"])


@unittest.skipUnless(HAS_BROWSER, "needs node + playwright")
class ProgressBarInBrowser(unittest.TestCase):
    """The real start page against the real server: the bar moves while the analysis runs, and a failure is shown."""

    @classmethod
    def setUpClass(cls):
        cls.srv = make_server(port=0)
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}/"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def drive(self, score_midis):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_wav(d / "take.wav", synth([(m, 0.3 + 0.5 * i, 0.4) for i, m in enumerate(MIDIS)]))
            (d / "piece.musicxml").write_text(musicxml(score_midis))
            js = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();"
                  "const errors=[];p.on('pageerror',e=>errors.push(String(e)));await p.goto(process.argv[1]);"
                  "await p.setInputFiles('input[name=video]',process.argv[2]);await p.setInputFiles('input[name=score]',process.argv[3]);"
                  "await p.click('#go');const samples=[];const start=Date.now();"
                  "while(Date.now()-start<120000&&!p.url().includes('/run/')){"
                  "try{samples.push(await p.evaluate(()=>({value:document.getElementById('bar').value,label:document.getElementById('progress-label').textContent,"
                  "shown:!document.getElementById('progress').hidden,error:document.getElementById('status').textContent,"
                  "enabled:!document.getElementById('go').disabled})));}catch(e){await p.waitForTimeout(150);continue;}"      # (the page may navigate to the report mid-sample)
                  "if(samples[samples.length-1].error)break;await p.waitForTimeout(150);}"
                  "console.log(JSON.stringify({samples,url:p.url(),errors}));await b.close();})()")
            out = subprocess.run(["node", "-e", js, self.url, str(d / "take.wav"), str(d / "piece.musicxml")], capture_output=True, text=True,
                                 check=True).stdout
        return json.loads(out)

    def test_the_bar_moves_to_the_end_and_the_report_opens(self):
        result = self.drive(MIDIS)
        self.assertEqual(result["errors"], [])
        self.assertIn("/run/", result["url"])
        samples = [s for s in result["samples"] if s["shown"]]
        self.assertGreaterEqual(len(samples), 1)
        values = [s["value"] for s in samples]
        self.assertTrue(all(isinstance(v, (int, float)) or v.isdigit() for v in values))
        values = [int(v) for v in values]
        self.assertEqual(values, sorted(values))                                        # never goes back
        self.assertTrue(all(s["label"].strip() for s in samples))
        self.assertTrue(any("%" in s["label"] for s in samples))

    def test_a_failed_analysis_shows_the_reason_and_frees_the_button(self):
        result = self.drive([60, 62, 64, 65, 55, 57, 59, 60])
        last = result["samples"][-1]
        self.assertIn("does not appear to match", last["error"])
        self.assertFalse(last["shown"])                                                 # the bar is hidden again
        self.assertTrue(last["enabled"])
        self.assertNotIn("/run/", result["url"])


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

    def start(self, files):
        """POST an upload -> (HTTP status, JSON): 202 {"job": id} when the analysis was started, 400 for an unusable upload."""
        ctype, body = multipart(files)
        req = urllib.request.Request(self.url + "/api/analyze", body, {"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def wait(self, job, timeout=300):
        """Poll a job until it is no longer running -> its last status (and every status seen on the way)."""
        seen, deadline = [], time.time() + timeout
        while time.time() < deadline:
            with urllib.request.urlopen(f"{self.url}/api/job/{job}") as r:
                status = json.loads(r.read())
            seen.append(status)
            if status["state"] != "running":
                return status, seen
            time.sleep(0.1)
        raise AssertionError("the analysis did not finish")

    def post(self, files):
        """Upload and wait for the analysis -> (200, {"url"}) when it worked, else (400, {"error"}): the shape the checks below use."""
        status, data = self.start(files)
        if status != 202:
            return status, data
        final, _ = self.wait(data["job"])
        return (200, {"url": final["url"]}) if final["state"] == "done" else (400, {"error": final["error"]})

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
        page = self.page(data)
        self.assertTrue({"bow", "posture", "hand"} <= set(page["traces"]))      # the pose VioPose generated is in the page
        self.assertIsNotNone(page["pose"])

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
        self.assertIsNone(self.page(data)["pose"])                     # the extra pose was ignored; no VioPose here

    def test_index_has_upload_form(self):
        with urllib.request.urlopen(self.url + "/") as r:
            page = r.read().decode()
        self.assertIn('name="video"', page)
        self.assertIn('name="score"', page)
        self.assertNotIn('name="pose"', page)                       # the pose is generated, never uploaded
        self.assertNotIn("mocap", page.lower())

    def get(self, path, headers=None):
        req = urllib.request.Request(self.url + path, headers=headers or {})
        with urllib.request.urlopen(req) as r:
            return r.status, dict(r.headers), r.read()

    def page(self, answer):
        """The page data of the run an upload answered with."""
        return page_data(self.get(answer["url"])[2].decode())

    def test_upload_returns_the_player_page(self):
        status, data = self.upload(MIDIS)
        self.assertEqual(status, 200)
        page = self.page(data)
        self.assertEqual(page["media"], {"url": "media", "kind": "audio"})     # a wav upload has no picture
        self.assertEqual(len(page["traces"]["notes"]), len(MIDIS))
        self.assertIn("intonation", page["scores"])

    def test_media_is_served_with_range_requests_for_seeking(self):
        _, data = self.upload(MIDIS)
        status, headers, full = self.get(data["url"] + "media")
        self.assertEqual((status, headers["Accept-Ranges"]), (200, "bytes"))
        self.assertEqual(full[:4], b"RIFF")                                     # the uploaded wav, untouched
        status, headers, part = self.get(data["url"] + "media", {"Range": "bytes=100-199"})
        self.assertEqual((status, part), (206, full[100:200]))
        self.assertEqual(headers["Content-Range"], f"bytes 100-199/{len(full)}")
        _, _, tail = self.get(data["url"] + "media", {"Range": "bytes=-50"})
        self.assertEqual(tail, full[-50:])
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get("/run/" + "0" * 32 + "/")
        self.assertEqual(cm.exception.code, 404)

    def test_wrong_score_returns_readable_error(self):
        status, data = self.upload([60, 62, 64, 65, 55, 57, 59, 60])
        self.assertEqual(status, 400)
        self.assertIn("does not appear to match", data["error"])

    def test_a_missing_ffmpeg_is_a_readable_error_not_a_crash(self):
        from maestro.utils.errors import ToolMissingError
        with mock.patch("maestro.gui.server.analyze", side_effect=ToolMissingError("ffprobe was not found. install it")):
            status, data = self.upload(MIDIS)
        self.assertEqual(status, 400)
        self.assertIn("ffprobe was not found", data["error"])

    def test_progress_is_reported_while_the_analysis_runs(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            wav = d / "a.wav"
            write_wav(wav, synth([(m, 0.3 + 0.5 * i, 0.4) for i, m in enumerate(MIDIS)]))
            status, data = self.start({"video": ("take.wav", wav.read_bytes()), "score": ("piece.musicxml", musicxml(MIDIS).encode())})
        self.assertEqual(status, 202)                                                    # answered at once, the analysis runs on
        final, seen = self.wait(data["job"])
        self.assertEqual((final["state"], final["fraction"], final["url"].startswith("/run/")), ("done", 1.0, True))
        fractions = [x["fraction"] for x in seen]
        self.assertEqual(fractions, sorted(fractions))                                   # the bar never goes back
        self.assertTrue(all(0 <= x["fraction"] <= x["ceiling"] <= 1 for x in seen))
        self.assertTrue(all(x["label"] for x in seen))
        self.assertTrue({"stage", "seconds", "seconds_in_stage"} <= set(final))

    def test_a_failed_analysis_ends_in_an_error_state_with_the_reason(self):
        status, data = self.upload([60, 62, 64, 65, 55, 57, 59, 60])                      # (the wrong score)
        self.assertEqual(status, 400)
        self.assertIn("does not appear to match", data["error"])
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(f"{self.url}/api/job/{'0' * 32}")
        self.assertEqual(cm.exception.code, 404)

    def test_missing_file_and_unknown_route(self):
        status, data = self.post({"score": ("s.musicxml", musicxml([69]).encode())})
        self.assertEqual(status, 400)
        self.assertIn("both", data["error"])
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.url + "/nope")
        self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
