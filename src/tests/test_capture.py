"""Recording a take and scanning the score from the start page: against the real server and page in headless Chromium with a fake camera and
microphone. The recorded file must be one the analysis accepts, a scanned page one the music reader gets (a clean image or a valid PDF)."""
import base64
import json
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

from test_gui import HAS_BROWSER
from maestro.gui.server import INDEX, make_server
from maestro.preprocessing.video import preprocess_video
from maestro.utils.convert_file_type import check_pdf, detect_score_type
from maestro.utils.convert_video_type import probe
from maestro.utils.errors import InputError
from maestro.utils.Performance import Performance

FLAGS = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]     # a synthetic camera and microphone, no prompt


def run_page(url, body, args=FLAGS, init=""):
    """Open `url` in Chromium (camera and microphone granted) and run the async JS `body` in the page -> its JSON result (plus script errors)."""
    js = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch({args:JSON.parse(process.argv[2])});"
          "const c=await b.newContext({permissions:['camera','microphone']});const p=await c.newPage();"
          "const errors=[];p.on('pageerror',e=>errors.push(String(e)));" + init +
          "await p.goto(process.argv[1]);"
          "const out=await (async()=>{" + body + "})();console.log(JSON.stringify({out,errors}));await b.close();})()")
    run = subprocess.run(["node", "-e", js, url, json.dumps(args)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-1500:]
    result = json.loads(run.stdout)
    assert result["errors"] == [], result["errors"]
    return result["out"]


# Reads the file the page put into a form input -> {name, type, size, base64}
READ = ("const read = (name) => p.evaluate(async (n) => { const f = document.querySelector('input[name=' + n + ']').files[0]; if (!f) return null;"
        "const bytes = new Uint8Array(await f.arrayBuffer()); let s = ''; for (let i = 0; i < bytes.length; i += 8192) s += String.fromCharCode(...bytes.subarray(i, i + 8192));"
        "return {name: f.name, type: f.type, size: f.size, base64: btoa(s)}; }, name);")


@unittest.skipUnless(HAS_BROWSER, "needs node + playwright")
class Capture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = make_server(port=0)
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}/"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def save(self, found, tmp, name=None):
        path = Path(tmp) / (name or found["name"])
        path.write_bytes(base64.b64decode(found["base64"]))
        return path

    def test_the_start_page_carries_the_capture_code(self):
        self.assertNotIn("/*CAPTURE_JS*/", INDEX)
        self.assertIn("function cleanPage", INDEX)
        self.assertIn("rec-open", INDEX)
        self.assertIn("scan-open", INDEX)

    def test_a_recorded_take_goes_into_the_form_and_is_accepted_by_the_analysis(self):
        found = run_page(self.url, READ + "await p.click('#rec-open'); await p.waitForTimeout(800);"
                         "const panelShown = await p.evaluate(() => !document.getElementById('rec-panel').hidden);"
                         "await p.click('#rec-go'); await p.waitForTimeout(2800); await p.click('#rec-go'); await p.waitForTimeout(1200);"
                         "return {file: await read('video'), panelShown, panelHiddenAfter: await p.evaluate(() => document.getElementById('rec-panel').hidden),"
                         "info: await p.evaluate(() => document.getElementById('rec-info').textContent)};")
        self.assertTrue(found["panelShown"] and found["panelHiddenAfter"])
        self.assertIn("Recorded a take", found["info"])
        file = found["file"]
        self.assertTrue(file["type"].startswith("video/") and file["size"] > 1000)
        with tempfile.TemporaryDirectory() as tmp:
            path = self.save(file, tmp)
            info = probe(path)                                                 # a browser's webm has no duration in its header: found by decoding
            self.assertTrue(info["has_video"] and info["has_audio"])
            self.assertTrue(2.0 < info["duration"] < 5.0, info)
            perf = preprocess_video(Performance(), path, tmp, denoise=False)  # exactly what the analysis does first
            self.assertTrue(Path(perf.video).is_file())

    def test_one_scanned_page_is_a_clean_jpeg_and_several_are_a_valid_pdf(self):
        one = run_page(self.url, READ + "await p.click('#scan-open'); await p.waitForTimeout(800); await p.click('#scan-shot'); await p.waitForTimeout(500);"
                       "const label = await p.evaluate(() => document.getElementById('scan-done').textContent);"
                       "await p.click('#scan-done'); return {file: await read('score'), label};")
        self.assertEqual(one["label"], "Use this page")
        self.assertEqual((one["file"]["name"], one["file"]["type"]), ("score.jpg", "image/jpeg"))
        three = run_page(self.url, READ + "await p.click('#scan-open'); await p.waitForTimeout(800);"
                         "for (let i = 0; i < 3; i++) { await p.click('#scan-shot'); await p.waitForTimeout(400); }"
                         "await p.click('#scan-pages img'); await p.waitForTimeout(200);"        # remove the first page again
                         "const label = await p.evaluate(() => document.getElementById('scan-done').textContent);"
                         "await p.click('#scan-done'); return {file: await read('score'), label, panelHidden: await p.evaluate(() => document.getElementById('scan-panel').hidden)};")
        self.assertEqual(three["label"], "Use 2 pages")
        self.assertTrue(three["panelHidden"])
        with tempfile.TemporaryDirectory() as tmp:
            path = self.save(three["file"], tmp, "score.pdf")
            self.assertEqual(three["file"]["type"], "application/pdf")
            self.assertEqual(detect_score_type(path), "pdf")                    # what the server's file check sees
            check_pdf(path)
            if shutil.which("pdfinfo"):
                text = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True).stdout
                self.assertIn("Pages:           2", text)
            if shutil.which("pdftoppm"):                                        # the pages really decode, as black and white
                subprocess.run(["pdftoppm", "-r", "30", "-gray", str(path), str(Path(tmp) / "page")], check=True)
                self.assertEqual(len(list(Path(tmp).glob("page*.pgm"))), 2)

    def test_a_photo_with_uneven_light_becomes_black_music_on_white_paper(self):
        found = run_page(self.url, "return await p.evaluate(async () => {"
                         "const c = document.createElement('canvas'); c.width = 900; c.height = 600; const g = c.getContext('2d');"
                         "const grad = g.createLinearGradient(0, 0, 900, 0); grad.addColorStop(0, '#6a6a6a'); grad.addColorStop(1, '#f0f0f0');"   # a shadow on the left
                         "g.fillStyle = grad; g.fillRect(0, 0, 900, 600); g.fillStyle = '#000';"
                         "for (let i = 0; i < 5; i++) g.fillRect(0, 200 + i * 20, 900, 3);"                                                      # staff lines
                         "const img = new Image(); img.src = c.toDataURL(); await img.decode();"
                         "const out = cleanPage(img), o = out.getContext('2d');"
                         "const at = (x, y) => o.getImageData(x, y, 1, 1).data[0];"
                         "return {paperShadow: at(60, 100), paperLight: at(800, 100), lineShadow: at(60, 201), lineLight: at(800, 221), between: at(400, 210), w: out.width};});")
        self.assertEqual(found["paperShadow"], 255)                              # the shadow does not turn the paper black
        self.assertEqual(found["paperLight"], 255)
        self.assertEqual(found["lineShadow"], 0)                                 # the staff lines are ink on both sides
        self.assertEqual(found["lineLight"], 0)
        self.assertEqual(found["between"], 255)
        self.assertEqual(found["w"], 900)

    def test_without_camera_streaming_the_buttons_use_the_devices_own_camera_app(self):
        """Browsers without getUserMedia/MediaRecorder: the buttons open <input capture>, and what it returns is used."""
        with tempfile.TemporaryDirectory() as tmp:
            video, photo = Path(tmp) / "clip.mp4", Path(tmp) / "page.png"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=d=2:s=320x240:r=15", "-f", "lavfi", "-i", "sine=d=2", "-shortest",
                            "-c:v", "mpeg4", "-c:a", "aac", str(video)], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=d=1:s=640x480:r=1", "-frames:v", "1", str(photo)], check=True)
            found = run_page(self.url, READ +
                             f"const [chooser] = await Promise.all([p.waitForEvent('filechooser'), p.click('#rec-open')]); await chooser.setFiles({json.dumps(str(video))});"
                             f"const [chooser2] = await Promise.all([p.waitForEvent('filechooser'), p.click('#scan-open')]); await chooser2.setFiles({json.dumps(str(photo))});"
                             "await p.waitForTimeout(800); return {video: await read('video'), score: await read('score'),"
                             "capture: await p.evaluate(() => [document.getElementById('rec-phone').getAttribute('capture'), document.getElementById('scan-phone').getAttribute('capture')])};",
                             args=[], init="await p.addInitScript(() => { delete window.MediaRecorder; Object.defineProperty(navigator, 'mediaDevices', {value: undefined}); });")
        self.assertEqual(found["capture"], ["user", "environment"])             # the front camera for the take, the back one for the page
        self.assertEqual((found["video"]["name"], found["video"]["type"]), ("clip.mp4", "video/mp4"))
        self.assertEqual((found["score"]["name"], found["score"]["type"]), ("score.jpg", "image/jpeg"))    # cleaned like a streamed photo


if __name__ == "__main__":
    unittest.main()
