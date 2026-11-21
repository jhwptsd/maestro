# Local web GUI server (new file; standard library only).
#
# Run `python src/scripts/serve.py` (or `maestro-gui` after `pip install -e .`) and open
# http://127.0.0.1:8000. GET / serves index.html (upload form); POST /api/analyze receives the
# recording + score as multipart form data, runs maestro.analyze() and answers {"url": "/run/<id>/"}, the report page
# built by gui/report.py (or {"error": message} with HTTP 400 for unusable input: InputError, MismatchError, missing
# Audiveris). The page plays the recording, which is served from /run/<id>/media with HTTP range support (browsers need it
# to seek). Runs live in a temporary folder that is removed when the server exits. Binds to localhost only: it is meant
# for personal use.
import atexit
import json
import mimetypes
import re
import shutil
import sys
import tempfile
import traceback
import uuid
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from maestro import analyze
from maestro.gui.report import build_page, media_source
from maestro.utils.errors import InputError, ToolMissingError

MAX_UPLOAD = 1 << 30  # 1 GiB
FIELDS = ("video", "score")                   # the upload form's file inputs; anything else a client sends is ignored
INDEX = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
RUNS_DIR = Path(tempfile.mkdtemp(prefix="maestro_runs_"))     # one folder per analysis, deleted when the server exits
atexit.register(shutil.rmtree, RUNS_DIR, ignore_errors=True)
RUNS = {}                                                     # id -> {"page": html, "media": path}
CHUNK = 1 << 20


def parse_multipart(content_type, body):
    """Multipart body -> {field name: (filename, bytes)} (stdlib `cgi` is gone in Python 3.13)."""
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    return {part.get_param("name", header="content-disposition"): (part.get_filename() or "", part.get_payload(decode=True))
            for part in msg.iter_parts()}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, ctype):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, obj):
        self._send(status, json.dumps(obj), "application/json")

    def _media(self, path):
        """Send a file, honouring `Range: bytes=a-b` (video seeking)."""
        size = Path(path).stat().st_size
        start, end, status = 0, size - 1, 200
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m[1] or m[2]):
            start = int(m[1]) if m[1] else max(size - int(m[2]), 0)
            end = min(int(m[2]), size - 1) if m[1] and m[2] else size - 1
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0 and (block := f.read(min(CHUNK, left))):
                self.wfile.write(block)
                left -= len(block)

    def do_GET(self):
        run = re.fullmatch(r"/run/([0-9a-f]{32})/(media)?", self.path)
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX, "text/html; charset=utf-8")
        elif run and run[1] in RUNS:
            if run[2]:
                try:
                    self._media(RUNS[run[1]]["media"])
                except (BrokenPipeError, ConnectionResetError):      # the browser dropped a range request: normal when seeking
                    pass
            else:
                self._send(200, RUNS[run[1]]["page"], "text/html; charset=utf-8")
        else:
            self._send(404, "Not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/analyze":
            return self._send(404, "Not found", "text/plain")
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= MAX_UPLOAD:
            return self._json(400, {"error": "Upload is empty or larger than 1 GB."})
        try:
            files = parse_multipart(self.headers.get("Content-Type", ""), self.rfile.read(length))
            if not files.get("video", ("", b""))[1] or not files.get("score", ("", b""))[1]:
                raise InputError("Please choose both a recording and a score.")
            run_id = uuid.uuid4().hex
            folder = RUNS_DIR / run_id
            folder.mkdir()
            paths = {}
            for field, (name, data) in files.items():
                if field not in FIELDS or not data:      # ignore unknown fields and empty optional inputs
                    continue
                # only the known field name plus the extension of the user's file name reach the path
                paths[field] = folder / (field + Path(name).suffix[:10])
                paths[field].write_bytes(data)
            perf = analyze(paths["video"], paths["score"], folder / "work")
            RUNS[run_id] = {"page": build_page(perf, "media"), "media": media_source(perf)[0]}
            print(f"analysis {run_id[:8]}: " + ", ".join(f"{k} {v:.1f}s" for k, v in perf.timings.items()) +
                  f" (pitch on {perf.audio_features.get('pitch_runtime') or 'n/a'})", file=sys.stderr)
            self._json(200, {"url": f"/run/{run_id}/"})
        except (InputError, ToolMissingError) as e:
            self._json(400, {"error": str(e)})
        except Exception as e:  # unexpected: log for the developer, short message for the user
            traceback.print_exc()
            self._json(500, {"error": f"Unexpected error: {e}"})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def make_server(host="127.0.0.1", port=8000):
    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    port = int((argv or sys.argv[1:] or [8000])[0])
    srv = make_server(port=port)
    print(f"Maestro GUI running at http://127.0.0.1:{srv.server_port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
