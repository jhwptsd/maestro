"""The written feedback: the local Ollama client (against a local fake of its API), the saved model choice, the digest, the generator, the
chat, the server endpoints and their protection, and the feedback card and chat of the page."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import test_assessment_events as tae
from test_gui import HAS_BROWSER, page_data, performance
from maestro.assessment.synthesize import synthesize_assessment
from maestro.feedback import chat, generator, ollama, prompts, settings
from maestro.gui.report import build_page
from maestro.gui import server as gui_server

GOOD_ANSWER = {"summary": "A steady performance with some sharp notes.",
               "priorities": [{"title": "Intonation after leaps", "bars": [2], "why": "Probably overshooting the shift.", "practice": "Slow shifts."}],
               "strengths": ["Steady tempo"]}


class FakeOllama:
    """A local server that answers like Ollama's native API (/api/tags, /api/chat, /api/pull) and records what it was sent."""

    def __init__(self, models=("llama3.2:3b", "nomic-embed-text"), answer=None):
        outer = self
        self.requests, self.models = [], list(models)
        self.answer = answer if answer is not None else GOOD_ANSWER       # a dict (sent as JSON text) or a string
        self.missing = set()                                              # model names that answer 404, like one that is not downloaded

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, status, body):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.requests.append({"path": self.path, "body": None})
                self.send(200, {"models": [{"name": m, "size": 2e9} for m in outer.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                outer.requests.append({"path": self.path, "body": body})
                if self.path == "/api/pull":                       # newline-delimited progress, layers one after another
                    lines = [{"status": "pulling manifest"}, {"status": "pulling a", "total": 100, "completed": 50},
                             {"status": "pulling a", "total": 100, "completed": 100}, {"status": "pulling b", "total": 10, "completed": 1},
                             {"status": "success"}]
                    if body["name"] == "nope":
                        lines = [{"error": "pull model manifest: file does not exist"}]
                    return self.send(200, "".join(json.dumps(x) + "\n" for x in lines).encode())
                if body["model"] in outer.missing:
                    return self.send(404, {"error": f"model '{body['model']}' not found"})
                text = outer.answer if isinstance(outer.answer, str) else json.dumps(outer.answer)
                self.send(200, {"message": {"role": "assistant", "content": text}, "prompt_eval_count": 5, "eval_count": 7})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def chats(self):
        return [r for r in self.requests if r["path"] == "/api/chat"]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class WithOllama(unittest.TestCase):
    """A fake Ollama, and the settings in a temporary config folder."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.dict(os.environ, {"MAESTRO_CONFIG": self._tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.fake = FakeOllama()
        self.addCleanup(self.fake.close)
        patcher = mock.patch.object(ollama, "BASE_URL", self.fake.url)
        patcher.start()
        self.addCleanup(patcher.stop)


def user(text):
    return [{"role": "user", "content": text}]


class Client(WithOllama):
    def test_lists_installed_models_without_embedding_models(self):
        self.assertEqual([m["id"] for m in ollama.list_models()], ["llama3.2:3b"])

    def test_complete_sets_context_json_mode_and_no_thinking(self):
        out = ollama.complete("llama3.2:3b", "S", user("hi"), 300, json_mode=True)
        body = self.fake.chats()[-1]["body"]
        self.assertEqual((body["format"], body["options"]["num_ctx"], body["options"]["num_predict"], body["stream"]), ("json", 8192, 300, False))
        self.assertIs(body["think"], False)                                      # a thinking model (qwen3.5) would otherwise answer nothing
        self.assertEqual(body["messages"], [{"role": "system", "content": "S"}, *user("hi")])
        self.assertEqual(out["usage"], {"input": 5, "output": 7})

    def test_an_empty_answer_is_explained(self):
        self.fake.answer = ""
        with self.assertRaisesRegex(ollama.LLMError, "no answer"):
            ollama.complete("m", "S", user("hi"))

    def test_not_running_says_how_to_install(self):
        with mock.patch.object(ollama, "BASE_URL", "http://127.0.0.1:1"), self.assertRaisesRegex(ollama.LLMError, "ollama.com/download"):
            ollama.list_models()

    def test_ollama_host_moves_the_server(self):
        with mock.patch.object(ollama, "BASE_URL", None), mock.patch.dict(os.environ, {"OLLAMA_HOST": "127.0.0.1:9999"}):
            self.assertEqual(ollama.base_url(), "http://127.0.0.1:9999")

    def test_model_choice_prefers_the_default_then_known_families(self):
        pick = lambda *ids, **kw: ollama.pick_model([{"id": i} for i in ids], **kw)
        self.assertEqual(pick("qwen3.5:4b", "llama3.2:3b"), "llama3.2:3b")
        self.assertEqual(pick("zzz", "qwen3.5:4b"), "qwen3.5:4b")
        self.assertEqual(pick("zzz"), "zzz")
        self.assertIsNone(pick("a", exclude={"a"}))

    def test_a_model_that_is_not_downloaded_is_replaced_by_an_installed_one(self):
        self.fake.models, self.fake.missing = ["qwen3.5:4b"], {"llama3.2:3b"}
        reply, used, note = ollama.complete_with_fallback("llama3.2:3b", "S", user("hi"))
        self.assertEqual((used, reply["text"] != ""), ("qwen3.5:4b", True))
        self.assertIn("not downloaded", note)
        self.fake.models = []
        with self.assertRaisesRegex(ollama.LLMError, "Download and use"):
            ollama.complete_with_fallback("llama3.2:3b", "S", user("hi"))

    def test_other_errors_are_raised_at_once(self):
        with mock.patch.object(ollama, "BASE_URL", "http://127.0.0.1:1"), self.assertRaisesRegex(ollama.LLMError, "not running"):
            ollama.complete_with_fallback("m", "S", user("hi"))

    def test_pull_progress_never_goes_back_and_reports_errors(self):
        seen = []
        ollama.pull("llama3.2:1b", lambda f, text: seen.append(f))
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(max(seen), 1.0)
        with self.assertRaisesRegex(ollama.LLMError, "does not exist"):
            ollama.pull("nope", lambda f, t: None)


class SavedChoice(WithOllama):
    def test_the_model_choice_is_saved_and_a_bad_file_means_nothing_saved(self):
        self.assertIsNone(settings.get_model())
        settings.set_model("qwen3.5:4b")
        self.assertEqual(settings.get_model(), "qwen3.5:4b")
        (Path(self._tmp.name) / "llm.json").write_text("{not json")
        self.assertIsNone(settings.get_model())
        (Path(self._tmp.name) / "llm.json").write_text(json.dumps({"provider": "groq", "keys": {"groq": "x"}}))     # an old file with API keys
        self.assertIsNone(settings.get_model())


def sample_perf():
    perf = tae.Synthesis("test_unavailable_modules_are_reported_not_dropped").build()
    perf.assessment["synthesis"] = synthesize_assessment(perf.assessment)
    return perf


class Digest(unittest.TestCase):
    def test_digest_is_compact_numeric_and_carries_no_private_data(self):
        perf = sample_perf()
        perf.video, perf.audio_source = "/home/someone/secret/video.mp4", "/home/someone/secret/take.wav"
        digest = prompts.build_digest(perf)
        text = json.dumps(digest)
        self.assertNotIn("someone", text)                                                        # no paths or names
        self.assertLess(len(text), 20000)
        self.assertEqual(json.loads(text), digest)
        severities = [e["severity"] for e in digest["problem_events"]]
        self.assertEqual(severities, sorted(severities, reverse=True))                           # most severe first
        self.assertLessEqual(len(digest["problem_events"]), prompts.MAX_EVENTS)
        self.assertIn("intonation", digest["scores_0_to_100"])
        self.assertTrue(digest["limits"])                                                        # no video analysis in this scene
        self.assertTrue(all(set(e) >= {"skill", "kind", "bar", "severity", "causes"} for e in digest["problem_events"]))
        self.assertIn("seconds", digest["piece"])

    def test_the_prompt_forbids_inventing_and_asks_for_json(self):
        self.assertIn("ONLY the measurements", prompts.SYSTEM)
        self.assertIn("JSON only", prompts.SYSTEM)
        message = prompts.user_message({"scores_0_to_100": {"intonation": 80}})
        self.assertIn('"intonation":80', message)


class Generator(WithOllama):
    def setUp(self):
        super().setUp()
        self.digest = prompts.build_digest(sample_perf())
        self.bar = next(e["bar"] for e in self.digest["problem_events"] if e["bar"] is not None)

    def ask(self, answer, **kw):
        self.fake.answer = answer
        return generator.feedback_from_digest(self.digest, **kw)

    def test_good_answer_is_shaped_and_only_the_compact_digest_is_sent(self):
        result = self.ask({**GOOD_ANSWER, "priorities": [{**GOOD_ANSWER["priorities"][0], "bars": [self.bar]}]})
        self.assertEqual(result["model"], "llama3.2:3b")                                          # the best installed, with nothing set up
        self.assertEqual(result["priorities"][0]["bars"], [self.bar])
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["usage"], {"input": 5, "output": 7})
        sent = self.fake.chats()[-1]["body"]["messages"][-1]["content"]
        self.assertIn(json.dumps(self.digest["scores_0_to_100"], separators=(",", ":")), sent)
        self.assertNotIn("summaries", sent)                                                       # compact: no per-skill summaries
        self.assertLess(len(json.dumps(prompts.compact(self.digest))), len(json.dumps(self.digest)))

    def test_compact_digest_keeps_the_worst_events(self):
        small = prompts.compact(self.digest)
        self.assertLessEqual(len(small["problem_events"]), prompts.COMPACT_EVENTS)
        self.assertEqual(small["problem_events"], [{**e, "causes": e["causes"][:1]} for e in self.digest["problem_events"][:prompts.COMPACT_EVENTS]])

    def test_invented_bars_are_removed_and_reported(self):
        result = self.ask({**GOOD_ANSWER, "priorities": [{**GOOD_ANSWER["priorities"][0], "bars": [self.bar, 999]}] * 5})
        self.assertEqual(len(result["priorities"]), 3)                                           # at most three
        self.assertEqual(result["priorities"][0]["bars"], [self.bar])
        self.assertTrue(any("999" in w for w in result["warnings"]))

    def test_fenced_or_chatty_json_is_still_read_and_plain_text_is_kept(self):
        fenced = "Here you go:\n```json\n" + json.dumps(GOOD_ANSWER) + "\n```"
        self.assertEqual(self.ask(fenced)["summary"], GOOD_ANSWER["summary"])
        result = self.ask("Just practise more slowly.")
        self.assertEqual((result["summary"], result["priorities"]), ("Just practise more slowly.", []))
        self.assertTrue(result["warnings"])

    def test_the_saved_model_is_used_and_the_result_stored_on_the_performance(self):
        self.fake.models = ["llama3.2:3b", "qwen3.5:4b"]
        settings.set_model("qwen3.5:4b")
        perf = sample_perf()
        result = generator.generate_feedback(perf)
        self.assertIs(perf.feedback, result)
        self.assertEqual((result["model"], self.fake.chats()[-1]["body"]["model"]), ("qwen3.5:4b", "qwen3.5:4b"))

    def test_a_missing_saved_model_is_replaced_and_noted(self):
        self.fake.models, self.fake.missing = ["qwen3.5:4b"], {"gone:1b"}
        result = self.ask(GOOD_ANSWER, model="gone:1b")
        self.assertEqual(result["model"], "qwen3.5:4b")
        self.assertTrue(any("not downloaded" in w for w in result["warnings"]))

    def test_nothing_installed_and_ollama_not_running_are_explained(self):
        self.fake.models = []
        with self.assertRaisesRegex(ollama.LLMError, "No local model is downloaded"):
            generator.feedback_from_digest(self.digest)
        with mock.patch.object(ollama, "BASE_URL", "http://127.0.0.1:1"), self.assertRaisesRegex(ollama.LLMError, "Ollama is not running"):
            generator.feedback_from_digest(self.digest)


class Conversation(WithOllama):
    def setUp(self):
        super().setUp()
        self.digest = prompts.build_digest(sample_perf())
        self.fake.answer = "Probably the shift overshoots."

    def ask(self, history, question, feedback=None):
        return chat.reply(self.digest, feedback, history, question)

    def test_the_model_gets_the_measurements_the_feedback_and_the_earlier_turns(self):
        feedback = {"summary": "SUMMARY-GIVEN", "priorities": [], "strengths": ["STRENGTH-GIVEN"]}
        history = [*user("first question"), {"role": "assistant", "content": "first answer"}]
        answer = self.ask(history, "  second question  ", feedback)
        self.assertEqual(answer["reply"], "Probably the shift overshoots.")
        sent = self.fake.chats()[-1]["body"]
        self.assertEqual(sent["messages"][1:], history + user("second question"))
        system = sent["messages"][0]["content"]
        self.assertIn(json.dumps(prompts.compact(self.digest)["scores_0_to_100"], separators=(",", ":")), system)
        self.assertIn("SUMMARY-GIVEN", system)
        self.assertIn("ONLY from those measurements", system)

    def test_empty_long_and_dangling_questions(self):
        with self.assertRaisesRegex(ollama.LLMError, "Type a question"):
            self.ask([], "   ")
        with self.assertRaisesRegex(ollama.LLMError, "under 1000"):
            self.ask([], "x" * 1001)
        old = [{"role": "assistant", "content": "orphan answer"}] + [{"role": "user", "content": f"q{i}"} if i % 2 == 0 else
                                                                    {"role": "assistant", "content": f"a{i}"} for i in range(2 * chat.MAX_TURNS)]
        self.ask(old, "latest")
        messages = self.fake.chats()[-1]["body"]["messages"][1:]
        self.assertEqual(messages[0]["role"], "user")                                                # a window starts with the student
        self.assertLessEqual(len(messages), chat.MAX_TURNS + 1)
        self.assertEqual(messages[-1]["content"], "latest")


class Endpoints(WithOllama):
    """The server's endpoints with Ollama replaced by a fake."""

    @classmethod
    def setUpClass(cls):
        cls.srv = gui_server.make_server(port=0)
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def call(self, path, body=None, headers=None, method=None, raw=None):
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        h = {"Content-Type": "application/json", **(headers or {})}
        request = urllib.request.Request(self.url + path, data=data, headers=h, method=method or ("GET" if data is None else "POST"))
        try:
            with urllib.request.urlopen(request) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_state_select_and_pull(self):
        status, data = self.call("/api/llm")
        self.assertEqual((status, data["running"], data["model"], [m["id"] for m in data["installed"]]), (200, True, "llama3.2:3b", ["llama3.2:3b"]))
        self.assertIn("qwen3.5:4b", [m["id"] for m in data["suggested"]])
        self.assertEqual(self.call("/api/llm/select", {"model": "not-installed"})[0], 400)
        self.fake.models.append("qwen3.5:4b")
        self.assertEqual(self.call("/api/llm/select", {"model": "qwen3.5:4b"})[1]["model"], "qwen3.5:4b")
        self.assertEqual(self.call("/api/llm/pull", {"model": "bad name!"})[0], 400)
        status, data = self.call("/api/llm/pull", {"model": "llama3.2:1b"})
        self.assertEqual(status, 202)
        for _ in range(100):
            job = self.call(f"/api/job/{data['job']}")[1]
            if job["state"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual((job["state"], job["fraction"]), ("done", 1.0))
        self.assertEqual(self.call("/api/llm")[1]["model"], "llama3.2:1b")                        # the downloaded model is now in use
        status, data = self.call("/api/llm/pull", {"model": "nope"})
        for _ in range(100):
            job = self.call(f"/api/job/{data['job']}")[1]
            if job["state"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(job["state"], "error")
        self.assertIn("does not exist", job["error"])

    def test_state_when_ollama_is_not_running(self):
        with mock.patch.object(ollama, "BASE_URL", "http://127.0.0.1:1"):
            data = self.call("/api/llm")[1]
        self.assertEqual((data["running"], data["model"], data["installed"]), (False, None, []))

    def test_feedback_is_written_once_per_analysis(self):
        digest = prompts.build_digest(sample_perf())
        bar = next(e["bar"] for e in digest["problem_events"] if e["bar"] is not None)
        self.fake.answer = {**GOOD_ANSWER, "priorities": [{**GOOD_ANSWER["priorities"][0], "bars": [bar]}]}
        gui_server.RUNS["a" * 32] = {"digest": digest}
        self.addCleanup(gui_server.RUNS.pop, "a" * 32, None)
        status, data = self.call("/api/feedback", {"run": "a" * 32})
        self.assertEqual(status, 200)
        self.assertEqual(data["priorities"][0]["bars"], [bar])
        calls = len(self.fake.chats())
        self.assertEqual(self.call("/api/feedback", {"run": "a" * 32})[1], data)                  # the saved answer: no second call
        self.assertEqual(len(self.fake.chats()), calls)
        self.assertEqual(self.call("/api/feedback", {"run": "b" * 32})[0], 404)

    def test_chat_keeps_the_conversation_per_analysis_and_can_restart_it(self):
        self.fake.answer = "Work on the shift in bar 2."
        digest = prompts.build_digest(sample_perf())
        gui_server.RUNS["c" * 32] = {"digest": digest, "feedback": {"summary": "S", "priorities": [], "strengths": []}}
        self.addCleanup(gui_server.RUNS.pop, "c" * 32, None)
        status, data = self.call("/api/chat", {"run": "c" * 32, "message": "What first?"})
        self.assertEqual((status, data["reply"], data["turns"]), (200, "Work on the shift in bar 2.", 1))
        status, data = self.call("/api/chat", {"run": "c" * 32, "message": "And then?"})
        self.assertEqual(data["turns"], 2)
        self.assertEqual([m["content"] for m in self.fake.chats()[-1]["body"]["messages"][1:]],
                         ["What first?", "Work on the shift in bar 2.", "And then?"])             # the second call carries the first turn
        self.assertEqual(len(self.call("/api/chat", {"run": "c" * 32, "history": True})[1]["messages"]), 4)
        self.assertEqual(self.call("/api/chat", {"run": "c" * 32, "reset": True})[1], {"turns": 0})
        self.assertEqual(self.call("/api/chat", {"run": "c" * 32, "history": True})[1]["messages"], [])
        self.assertEqual(self.call("/api/chat", {"run": "d" * 32, "message": "hi"})[0], 404)
        self.assertEqual(self.call("/api/chat", {"run": "c" * 32, "message": " "})[0], 400)
        with mock.patch.object(gui_server, "MAX_CHAT_EXCHANGES", 1):
            self.call("/api/chat", {"run": "c" * 32, "message": "one"})
            status, data = self.call("/api/chat", {"run": "c" * 32, "message": "two"})
            self.assertEqual(status, 400)
            self.assertIn("clear it", data["error"])

    def test_other_web_pages_cannot_drive_these_endpoints(self):
        self.assertEqual(self.call("/api/llm", headers={"Origin": "https://evil.example"})[0], 403)           # cross-site page
        self.assertEqual(self.call("/api/llm", headers={"Host": "evil.example"})[0], 403)                      # DNS rebinding
        self.assertEqual(self.call("/api/llm/select", {"model": "llama3.2:3b"}, {"Origin": "http://evil.example"})[0], 403)
        self.assertIsNone(settings.get_model())                                                                # nothing was changed
        self.assertEqual(self.call("/api/llm/select", raw=b"model=x", headers={"Content-Type": "text/plain"})[0], 400)
        self.assertEqual(self.call("/api/llm/select", raw=b"[1]")[0], 400)
        self.assertEqual(self.call("/api/analyze", raw=b"x", headers={"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.call("/api/llm", headers={"Origin": self.url})[0], 200)                         # the page itself is fine


class Page(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.perf = performance(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_page_carries_the_digest_and_the_run(self):
        html = build_page(self.perf, "media.mp4", run="r" * 32)
        data = page_data(html)
        self.assertEqual(data["run"], "r" * 32)
        self.assertIn("overall", data["digest"]["scores_0_to_100"])
        self.assertIsNone(data["feedback"])


    def run_page(self, routes, script):
        """Open the page (built with a server run id) in Chromium with /api/* answered by `routes` ({path: (status, json)}), run `script`
        in the page and return its JSON result."""
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "r.html"
            page.write_text(build_page(self.perf, "media.mp4", run="r" * 32), encoding="utf-8")
            js = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();"
                  "const errors=[];p.on('pageerror',e=>errors.push(String(e)));const routes=JSON.parse(process.argv[2]);"
                  # the page's fetch is replaced before it loads: /api/* is answered from `routes` (file:// requests cannot be intercepted)
                  "await p.addInitScript(routes=>{window.fetch=async(url,opts)=>{const path=new URL(url,'http://x').pathname;"
                  "const sent=opts&&opts.body?JSON.parse(opts.body):{};const hit=routes[path+(sent.history?'#history':'')]||routes[path];"
                  "if(!hit)throw new TypeError('no route '+path);return new Response(JSON.stringify(hit[1]),{status:hit[0],headers:{'Content-Type':'application/json'}});};},routes);"
                  "await p.goto('file://'+process.argv[1]);await p.waitForTimeout(600);"
                  "const out=await p.evaluate(async()=>{" + script + "});"
                  "console.log(JSON.stringify({out,errors}));await b.close();})()")
            result = json.loads(subprocess.run(["node", "-e", js, str(page), json.dumps(routes)], capture_output=True, text=True, check=True).stdout)
        self.assertEqual(result["errors"], [])
        return result["out"]

    @unittest.skipUnless(HAS_BROWSER, "needs node + playwright")
    def test_a_failure_shows_the_message_not_undefined_and_offers_a_retry(self):
        out = self.run_page({"/api/chat": (200, {"messages": []}),
                             "/api/feedback": (400, {"error": "Ollama is not running at http://localhost:11434."})},
                            "return {text: document.querySelector('#feedback').textContent, retry: !![...document.querySelectorAll('#feedback button')]"
                            ".find(b => b.textContent === 'Write feedback' && !b.hidden)};")
        self.assertIn("Ollama is not running", out["text"])
        self.assertNotIn("undefined", out["text"])
        self.assertTrue(out["retry"])

    @unittest.skipUnless(HAS_BROWSER, "needs node + playwright")
    def test_feedback_is_written_automatically_and_the_chat_works_as_text(self):
        feedback = {"model": "m", "summary": "Auto summary.", "warnings": [],
                    "priorities": [{"title": "Shifts", "bars": [2], "why": "why text", "practice": "practice text"}], "strengths": ["Steady"]}
        out = self.run_page({"/api/feedback": (200, feedback),
                             "/api/chat#history": (200, {"messages": [{"role": "user", "content": "earlier question"},
                                                                        {"role": "assistant", "content": "earlier <b>answer</b>"}]}),
                             "/api/chat": (200, {"reply": "Reply <img src=x onerror=window.pwned=1>", "turns": 2})},
                            "const text = () => document.querySelector('#feedback').textContent;"
                            "const before = text();"
                            "const input = document.querySelector('#feedback .chat input'); input.value = 'my question';"
                            "document.querySelector('#feedback .chat form').requestSubmit();"
                            "await new Promise(r => setTimeout(r, 500));"
                            "return {before, bubbles: [...document.querySelectorAll('#feedback .bubble')].map(b => b.className + '|' + b.textContent),"
                            "imgs: document.querySelectorAll('#feedback img').length, bold: document.querySelectorAll('#feedback .bubble b').length, pwned: window.pwned === 1};")
        self.assertIn("Auto summary.", out["before"])                                              # no click was needed
        self.assertIn("Shifts", out["before"])
        self.assertEqual(out["bubbles"], ["bubble me|earlier question", "bubble ai|earlier <b>answer</b>", "bubble me|my question",
                                          "bubble ai|Reply <img src=x onerror=window.pwned=1>"])
        self.assertEqual((out["imgs"], out["bold"], out["pwned"]), (0, 0, False))                    # model text is never parsed as HTML

    @unittest.skipUnless(HAS_BROWSER, "needs node + playwright")
    def test_model_text_is_shown_as_text_never_as_html(self):
        evil = "<img src=x onerror=\"window.pwned=1\"><b>bold</b>"
        self.perf.feedback = {"model": "m", "summary": evil, "warnings": ["</script>"],
                              "priorities": [{"title": evil, "bars": [1], "why": evil, "practice": evil}], "strengths": [evil]}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                page = Path(tmp) / "r.html"
                page.write_text(build_page(self.perf, "media.mp4"), encoding="utf-8")
                import subprocess
                js = ("const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();"
                      "const errors=[];p.on('pageerror',e=>errors.push(String(e)));await p.goto('file://'+process.argv[1]);await p.waitForTimeout(400);"
                      "const r=await p.evaluate(()=>({pwned:window.pwned===1,imgs:document.querySelectorAll('#feedback img').length,"
                      "bolds:document.querySelectorAll('#feedback b').length,text:document.querySelector('#feedback').textContent}));"
                      "console.log(JSON.stringify({...r,errors}));await b.close();})()")
                out = json.loads(subprocess.run(["node", "-e", js, str(page)], capture_output=True, text=True, check=True).stdout)
        finally:
            self.perf.feedback = None
        self.assertEqual((out["pwned"], out["imgs"], out["errors"]), (False, 0, []))
        self.assertIn("<img src=x", out["text"])                                                  # visible as text
        self.assertEqual(out["bolds"], 2)                                                         # only our own <b> (title, "Going well")


if __name__ == "__main__":
    unittest.main()
