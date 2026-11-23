# Local web GUI server (new file; standard library only).
#
# Run `python src/scripts/serve.py` (or `maestro-gui` after `pip install -e .`) and open
# http://127.0.0.1:8000. GET / serves index.html (upload form); POST /api/analyze receives the
# recording + score as multipart form data, checks it, starts maestro.analyze() in a background thread and answers {"job": id} at
# once (HTTP 202; {"error": message} with 400 for a missing file); GET /api/job/<id> then reports {state: running | done | error,
# label, fraction, ceiling, ...} for the progress bar of the start page (utils/progress.py), and when it is done {url: "/run/<id>/"}, the
# report page built by gui/report.py, or {error: message} for unusable input (InputError, MismatchError, missing ffmpeg or Audiveris).
# The page plays the recording, which is served from /run/<id>/media with HTTP range support (browsers need it to seek). Runs live in a temporary folder that is removed when the server exits. Binds to localhost only: it is meant
# for personal use.
#
# AI feedback (feedback/): GET /api/llm and POST /api/llm/{connect,models,select,disconnect} manage which provider (a small local model
# through Ollama, OpenRouter or Groq) the user connected, with an API key for the last two; POST /api/llm/pull {model} downloads a local
# model in the background (polled at /api/job/<id>) (feedback/settings.py: stored per user, only ever shown masked), and
# POST /api/feedback {"run": id} writes feedback for an analysis with the chosen model (feedback/generator.py), sending only the
# digest of numbers; POST /api/chat {"run": id, "message": text} continues a conversation with the model about it (feedback/chat.py);
# GET /auth/openrouter/start and /auth/openrouter/callback/<state> are the "Sign in with OpenRouter" flow (feedback/oauth.py). These endpoints spend the user's money and handle their keys, so a web page on another site must not be able to
# call them: every request must name this server in its Host header (against DNS rebinding), a request with an Origin must come
# from this server, and POST bodies must be application/json (which a cross-site page cannot send without a preflight, which
# this server never answers).
import atexit
import json
import mimetypes
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import uuid
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from maestro import analyze
from maestro.feedback import chat, generator, oauth, prompts, providers, settings
from maestro.gui.report import build_page, media_source
from maestro.utils.progress import Tracker
from maestro.utils.errors import InputError, ToolMissingError

MAX_UPLOAD = 1 << 30  # 1 GiB
FIELDS = ("video", "score")                   # the upload form's file inputs; anything else a client sends is ignored
INDEX = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
RUNS_DIR = Path(tempfile.mkdtemp(prefix="maestro_runs_"))     # one folder per analysis, deleted when the server exits
atexit.register(shutil.rmtree, RUNS_DIR, ignore_errors=True)
JOBS = {}                                                     # id -> status of an analysis (see run_analysis): state running | done | error
RUNS = {}                                                     # id -> {"page": html, "media": path, "digest": dict, "feedback": dict once written}
CHUNK = 1 << 20
MAX_JOBS_KEPT = 50                                           # finished analyses whose status stays available
MAX_CHAT_EXCHANGES = 20                                      # questions per analysis (each is a paid call)


def parse_multipart(content_type, body):
    """Multipart body -> {field name: (filename, bytes)} (stdlib `cgi` is gone in Python 3.13)."""
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
    return {part.get_param("name", header="content-disposition"): (part.get_filename() or "", part.get_payload(decode=True))
            for part in msg.iter_parts()}


def run_analysis(job_id, paths, folder):
    """Analyse the upload in a background thread, keeping JOBS[job_id] up to date for the progress bar of the start page, and, when
    it succeeds, register the report in RUNS (the same id is the report's address)."""
    job = JOBS[job_id]

    def update(info):
        if info["stage"] != "done":                  # (analyze's own "done"; the report still has to be built)
            job.update(stage=info["stage"], label=info["label"], fraction=info["fraction"], ceiling=info["ceiling"], stage_started=time.monotonic())

    tracker = Tracker(update)
    try:
        perf = analyze(paths["video"], paths["score"], folder / "work", progress=tracker)
        tracker("report")
        RUNS[job_id] = {"page": build_page(perf, "media", job_id), "media": media_source(perf)[0], "digest": prompts.build_digest(perf)}
        print(f"analysis {job_id[:8]}: " + ", ".join(f"{k} {v:.1f}s" for k, v in perf.timings.items()) +
              f" (pitch on {perf.audio_features.get('pitch_runtime') or 'n/a'})", file=sys.stderr)
        job.update(state="done", stage="done", label="Done", fraction=1.0, ceiling=1.0, url=f"/run/{job_id}/")
    except (InputError, ToolMissingError) as e:
        job.update(state="error", error=str(e))
    except Exception as e:  # unexpected: log for the developer, short message for the user
        traceback.print_exc()
        job.update(state="error", error=f"Unexpected error: {e}")


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

    def _local_request(self):
        """False (and a 403 sent) unless the Host header and any Origin name this server itself."""
        port = self.server.server_address[1]
        hosts = {f"{h}:{port}" for h in ("127.0.0.1", "localhost", "[::1]")}
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in hosts or (origin and origin not in {f"http://{h}" for h in hosts}):
            self._json(403, {"error": "Requests must come from the Maestro page itself."})
            return False
        return True

    def _read_json(self):
        """The JSON object in the request body (application/json only, at most 1 MB), else None after sending an error."""
        length = int(self.headers.get("Content-Length", 0))
        if not self.headers.get("Content-Type", "").startswith("application/json") or not 0 < length <= 1 << 20:
            self._json(400, {"error": "Expected a JSON body."})
            return None
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            body = None
        if not isinstance(body, dict):
            self._json(400, {"error": "Expected a JSON object."})
            return None
        return body

    def _llm_api(self, body):
        """POST /api/llm/* and /api/feedback -> (status, object); providers.LLMError is the caller's to report."""
        route = self.path
        if route == "/api/feedback":
            run = RUNS.get(str(body.get("run", "")))
            if run is None:
                return 404, {"error": "Unknown analysis: open the report again."}
            if "feedback" not in run:                                  # one paid call per analysis, then the saved answer
                run["feedback"] = generator.feedback_from_digest(run["digest"])
            return 200, run["feedback"]
        if route == "/api/chat":
            return self._chat(body)
        if route == "/api/llm/pull":
            return self._pull(str(body.get("model", "")).strip())
        provider = str(body.get("provider", ""))
        settings._known(provider)                                      # unknown name -> LLMError
        if route == "/api/llm/connect":
            if not providers.PROVIDERS[provider].needs_key:            # the local model: just check Ollama answers
                models = providers.get(provider).list_models()
                settings.connect_local(providers.get(provider).pick_model(models) if models else None)
                return 200, {"status": settings.status(), "models": models}
            key = str(body.get("api_key", "")).strip()
            if not key:
                return 400, {"error": "Paste the API key first."}
            p = providers.get(provider)
            models = p.list_models(key)                                # a bad key fails here, before anything is saved
            settings.set_key(provider, key, p.pick_model(models))
            return 200, {"status": settings.status(), "models": models}
        if route == "/api/llm/models":
            key = settings.get_key(provider)
            if not key:
                return 400, {"error": f"{providers.PROVIDERS[provider].label} is not connected."}
            return 200, {"models": providers.get(provider).list_models(key)}
        if route == "/api/llm/select":
            settings.choose(provider, str(body.get("model", "")) or None)
            return 200, {"status": settings.status()}
        if route == "/api/llm/disconnect":
            settings.remove_key(provider)
            return 200, {"status": settings.status()}
        return 404, {"error": "Not found"}

    def _pull(self, model):
        """POST /api/llm/pull {model} -> 202 {"job": id}: download a model into Ollama in the background and, when done, use it. The job
        is polled at /api/job/<id> like an analysis (state, label, fraction)."""
        if not re.fullmatch(r"[\w.\-:/]{1,100}", model):
            return 400, {"error": "Choose a model to download."}
        job_id, now = uuid.uuid4().hex, time.monotonic()
        job = JOBS[job_id] = {"state": "running", "stage": "pull", "label": f"Starting the download of {model}", "fraction": 0.0,
                              "ceiling": 0.0, "started": now, "stage_started": now}

        def work():
            try:
                def progress(fraction, label):
                    job.update(fraction=fraction, ceiling=fraction, label=label)
                providers.get("local").pull(model, progress)
                settings.connect_local(model)
                job.update(state="done", fraction=1.0, ceiling=1.0, label="Ready")
            except providers.LLMError as e:
                job.update(state="error", error=str(e))
            except Exception as e:
                traceback.print_exc()
                job.update(state="error", error=f"Unexpected error: {e}")
        threading.Thread(target=work, daemon=True).start()
        return 202, {"job": job_id}

    def _chat(self, body):
        """POST /api/chat {run, message} -> {reply, ...}: the next turn of the conversation about an analysis (kept server-side);
        {run, reset: true} starts it afresh."""
        run = RUNS.get(str(body.get("run", "")))
        if run is None:
            return 404, {"error": "Unknown analysis: open the report again."}
        if body.get("reset"):
            run["chat"] = []
            return 200, {"turns": 0}
        if body.get("history"):                                        # a reloaded page asks for the conversation so far
            return 200, {"messages": run.get("chat", [])}
        history = run.setdefault("chat", [])
        if len(history) >= 2 * MAX_CHAT_EXCHANGES:
            return 400, {"error": "This conversation has got long: clear it to start a new one."}
        answer = chat.reply(run["digest"], run.get("feedback"), history, str(body.get("message", "")))
        history += [{"role": "user", "content": str(body["message"]).strip()}, {"role": "assistant", "content": answer["reply"]}]
        return 200, {**answer, "turns": len(history) // 2}

    def _redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _openrouter_sign_in(self):
        """GET /auth/openrouter/start and /auth/openrouter/callback/<state>?code=... (see feedback/oauth.py)."""
        url = urllib.parse.urlsplit(self.path)
        port = self.server.server_address[1]
        if url.path == "/auth/openrouter/start":
            if self._local_request():
                self._redirect(oauth.authorization_url(f"http://localhost:{port}/auth/openrouter/callback"))
            return
        state = url.path.rsplit("/", 1)[-1]
        code = urllib.parse.parse_qs(url.query).get("code", [""])[0]
        try:
            key = oauth.exchange(state, code)
            p = providers.PROVIDERS["openrouter"]
            settings.set_key("openrouter", key, p.pick_model(p.list_models(key)))
            self._redirect("/?ai=connected#ai")
        except providers.LLMError as e:
            self._redirect("/?ai_error=" + urllib.parse.quote(str(e)) + "#ai")

    def _job_status(self, job_id):
        """GET /api/job/<id> -> the state of an analysis: {state, stage, label, fraction, ceiling, seconds, seconds_in_stage, url | error}."""
        if not self._local_request():
            return
        job = JOBS.get(job_id)
        if job is None:
            return self._json(404, {"error": "Unknown analysis."})
        now = time.monotonic()
        out = {k: job[k] for k in ("state", "stage", "label", "fraction", "ceiling") if k in job}
        out.update(seconds=round(now - job["started"], 1), seconds_in_stage=round(now - job["stage_started"], 1))
        out.update({k: job[k] for k in ("url", "error") if k in job})
        self._json(200, out)

    def do_GET(self):
        job = re.fullmatch(r"/api/job/([0-9a-f]{32})", self.path)
        if job:
            return self._job_status(job[1])
        if self.path.startswith("/auth/openrouter/"):
            return self._openrouter_sign_in()
        if self.path == "/api/llm":
            if self._local_request():
                self._json(200, settings.status())
            return
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
        if self.path in ("/api/feedback", "/api/chat") or self.path.startswith("/api/llm/"):
            if not self._local_request():
                return
            body = self._read_json()
            if body is None:
                return
            try:
                self._json(*self._llm_api(body))
            except providers.LLMError as e:
                self._json(400, {"error": str(e)})
            except Exception as e:                                     # a bug, not the user's problem: log it, answer briefly
                traceback.print_exc()
                self._json(500, {"error": f"Unexpected error: {e}"})
            return
        if self.path != "/api/analyze":
            return self._send(404, "Not found", "text/plain")
        if not self._local_request():
            return
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
        except InputError as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:  # unexpected: log for the developer, short message for the user
            traceback.print_exc()
            return self._json(500, {"error": f"Unexpected error: {e}"})
        JOBS[run_id] = {"state": "running", "stage": "upload", "label": "Starting", "fraction": 0.0, "ceiling": 0.0, "started": time.monotonic(),
                        "stage_started": time.monotonic()}
        for old in [j for j, v in JOBS.items() if v["state"] != "running"][:-MAX_JOBS_KEPT]:       # keep the list short
            del JOBS[old]
        threading.Thread(target=run_analysis, args=(run_id, paths, folder), daemon=True).start()
        self._json(202, {"job": run_id})

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
