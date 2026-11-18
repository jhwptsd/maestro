# Local web GUI server (new file; standard library only).
#
# Run `python src/scripts/serve.py` (or `maestro-gui` after `pip install -e .`) and open
# http://127.0.0.1:8000. GET / serves index.html (upload form); POST /api/analyze receives the
# recording + score as multipart form data, runs maestro.analyze() and returns {"html": <report>}
# built by gui/report.py (optional "pose" / "mocap" fields enable the movement analysis and its accuracy check), or {"error": message} with HTTP 400 for unusable input (InputError,
# MismatchError, missing Audiveris). Binds to localhost only: it is meant for personal use.
import json
import sys
import tempfile
import traceback
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from maestro import analyze
from maestro.gui.report import CSS, VIEWER_JS, build_report
from maestro.utils.errors import InputError, OMRUnavailableError

MAX_UPLOAD = 1 << 30  # 1 GiB
FIELDS = ("video", "score", "pose", "mocap")  # the upload form's file inputs; pose and mocap are optional
INDEX = (Path(__file__).parent / "index.html").read_text(encoding="utf-8").replace("/*CSS*/", CSS).replace("/*VIEWER_JS*/", VIEWER_JS)


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

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX, "text/html; charset=utf-8")
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
            with tempfile.TemporaryDirectory(prefix="maestro_") as tmp:
                paths = {}
                for field, (name, data) in files.items():
                    if field not in FIELDS or not data:      # ignore unknown fields and empty optional inputs
                        continue
                    # only the known field name plus the extension of the user's file name reach the path
                    paths[field] = Path(tmp) / (field + Path(name).suffix[:10])
                    paths[field].write_bytes(data)
                perf = analyze(paths["video"], paths["score"], Path(tmp) / "work",
                               pose_file=paths.get("pose"), mocap_file=paths.get("mocap"))
                self._json(200, {"html": build_report(perf)})
        except (InputError, OMRUnavailableError) as e:
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
