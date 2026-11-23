"""The LLM feedback layer: the two providers (OpenRouter and Groq, against a local fake of their API), retries and model fallback, the saved settings,
the digest, the generator, the chat, OpenRouter sign-in, the server endpoints and their protection, and the feedback card and chat of the page."""
import hashlib
import http.client
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import test_assessment_events as tae
from test_gui import HAS_BROWSER, page_data, performance
from maestro.assessment.synthesize import synthesize_assessment
from maestro.feedback import chat, generator, oauth, prompts, providers, settings
from maestro.gui.report import build_page
from maestro.gui import server as gui_server

KEY = "sk-test-0123456789abcdefghij"
GOOD_ANSWER = {"summary": "A steady performance with some sharp notes.",
               "priorities": [{"title": "Intonation after leaps", "bars": [2], "why": "Probably overshooting the shift.", "practice": "Slow shifts."}],
               "strengths": ["Steady tempo"]}
FREE = ["google/gemini-2.0-flash:free", "meta-llama/llama-3.3-70b-instruct:free", "mistralai/mistral-7b-instruct:free"]


class FakeProvider:
    """A local server that answers like the OpenAI-style API of OpenRouter and Groq and records what it was sent."""

    def __init__(self, answer=None, status=200, models_status=200):
        outer = self
        self.requests, self.answer, self.status, self.models_status = [], answer if answer is not None else GOOD_ANSWER, status, models_status
        self.missing = set()                                      # model ids that answer 404, like a retired or unavailable model
        self.busy = {}                                            # model id -> how many calls still answer 503 ("overloaded")
        self.models = ["gpt-4o", "text-embedding-3-small", "whisper-1", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

        class Handler(BaseHTTPRequestHandler):
            def reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def record(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else None
                outer.requests.append({"method": self.command, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                       "body": body})
                return body

            def do_GET(self):
                self.record()
                if outer.models_status != 200:
                    return self.reply(outer.models_status, {"error": {"message": "bad key"}})
                return self.reply(200, {"data": [{"id": i} for i in outer.models]})

            def do_POST(self):
                body = self.record()
                model = (body or {}).get("model")
                if model in outer.missing:
                    return self.reply(404, {"error": {"message": f"model {model} not found"}})
                if outer.busy.get(model, 0) > 0:
                    outer.busy[model] -= 1
                    return self.reply(503, {"error": {"message": "This model is currently experiencing high demand."}})
                if outer.status != 200:
                    return self.reply(outer.status, {"error": {"message": "quota exceeded"}})
                text = outer.answer if isinstance(outer.answer, str) else json.dumps(outer.answer)
                return self.reply(200, {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 12, "completion_tokens": 8}})

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def posts(self):
        return [r for r in self.requests if r["method"] == "POST"]


class WithConfig(unittest.TestCase):
    """Settings in a temporary config folder, no provider keys in the environment, no waiting between retries."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        env = {k: v for k, v in os.environ.items() if k not in ("OPENROUTER_API_KEY", "GROQ_API_KEY")}
        env["MAESTRO_CONFIG"] = self._tmp.name
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        sleeper = mock.patch.object(providers.time, "sleep")
        self.sleep = sleeper.start()
        self.addCleanup(sleeper.stop)


def user(text):
    return [{"role": "user", "content": text}]


class ProviderCalls(WithConfig):
    def setUp(self):
        super().setUp()
        self.fake = FakeProvider()
        self.addCleanup(self.fake.close)

    def provider(self, name):
        return providers.get(name, self.fake.url)

    def test_request_and_answer_format(self):
        for name in ("openrouter", "groq"):
            reply = self.provider(name).complete(KEY, "some/model", "SYS", user("USR"), 321, json_mode=True)
            sent = self.fake.requests[-1]
            self.assertEqual(sent["path"], "/v1/chat/completions")
            self.assertEqual(sent["headers"]["authorization"], f"Bearer {KEY}")
            self.assertEqual(sent["body"]["messages"], [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}])
            self.assertEqual(sent["body"]["max_tokens"], 321)                                       # the widely supported spelling ...
            self.assertNotIn("response_format", sent["body"])                                       # ... and no option some models reject
            self.assertNotIn("max_completion_tokens", sent["body"])
            self.assertEqual(json.loads(reply["text"])["summary"], GOOD_ANSWER["summary"])
            self.assertEqual(reply["usage"], {"input": 12, "output": 8})

    def test_a_conversation_is_sent_as_alternating_turns(self):
        turns = [*user("q1"), {"role": "assistant", "content": "a1"}, *user("q2")]
        self.provider("groq").complete(KEY, "m", "S", turns)
        self.assertEqual([m["role"] for m in self.fake.requests[-1]["body"]["messages"]], ["system", "user", "assistant", "user"])

    def test_model_lists_drop_non_chat_models_and_put_free_ones_first(self):
        self.fake.models = ["openai/gpt-4o", "openai/text-embedding-3-small", *FREE, "meta-llama/llama-guard-3-8b"]
        models = self.provider("openrouter").list_models(KEY)
        self.assertEqual([m["id"] for m in models], [*FREE, "openai/gpt-4o"])                      # no embeddings, no guard model
        self.assertEqual(models[1]["label"], "meta-llama/llama-3.3-70b-instruct:free (free)")
        self.fake.models = ["llama-3.3-70b-versatile", "whisper-large-v3", "llama-3.1-8b-instant"]
        self.assertEqual([m["id"] for m in self.provider("groq").list_models(KEY)], ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])

    def test_default_model_choice(self):
        ids = lambda *names: [{"id": n} for n in names]
        openrouter, groq = providers.PROVIDERS["openrouter"], providers.PROVIDERS["groq"]
        self.assertEqual(openrouter.pick_model(ids("openai/gpt-4o", *FREE)), "meta-llama/llama-3.3-70b-instruct:free")
        self.assertEqual(openrouter.pick_model(ids("openai/gpt-4o", "google/gemini-2.0-flash:free", "qwen/qwen-2.5:free")), "google/gemini-2.0-flash:free")
        self.assertEqual(openrouter.pick_model(ids("openai/gpt-4o", "anthropic/claude")), "openai/gpt-4o")      # no free model at all
        self.assertEqual(openrouter.pick_model(ids(*FREE), exclude={"meta-llama/llama-3.3-70b-instruct:free"}), "google/gemini-2.0-flash:free")
        self.assertEqual(groq.pick_model(ids("gemma2-9b-it", "llama-3.3-70b-versatile")), "llama-3.3-70b-versatile")
        self.assertEqual(groq.pick_model(ids("gemma2-9b-it", "llama-3.1-70b-x")), "llama-3.1-70b-x")
        self.assertIsNone(groq.pick_model([]))

    def test_failures_become_readable_messages(self):
        self.fake.status = 429
        with self.assertRaisesRegex(providers.LLMError, "quota"):
            self.provider("groq").complete(KEY, "m", "s", user("u"))
        self.fake.busy = {"m": 1}
        self.fake.status = 200
        with self.assertRaises(providers.LLMError) as cm:
            self.provider("groq").complete(KEY, "m", "s", user("u"))
        self.assertEqual(cm.exception.status, 503)
        self.assertIn("overloaded", str(cm.exception).lower())
        self.assertIn("high demand", str(cm.exception))                                             # the provider's own words are kept
        self.fake.models_status = 401
        with self.assertRaisesRegex(providers.LLMError, "API key is not valid"):
            self.provider("openrouter").list_models("bad")
        with self.assertRaisesRegex(providers.LLMError, "Could not reach"):
            providers.get("groq", "http://127.0.0.1:1").list_models(KEY)                            # nothing listens there
        for gone in ("gemini", "anthropic", "openai", "custom"):
            with self.assertRaisesRegex(providers.LLMError, "Unknown provider"):
                providers.get(gone)
        self.assertEqual(set(providers.PROVIDERS), {"openrouter", "groq"})


class BusyAndMissingModels(WithConfig):
    """Free models get busy and are retired: retry once, then other models, never a paid one, never past two alternatives."""

    def setUp(self):
        super().setUp()
        self.fake = FakeProvider()
        self.fake.models = ["openai/gpt-4o", *FREE]
        self.addCleanup(self.fake.close)
        self.p = providers.get("openrouter", self.fake.url)
        self.first = "meta-llama/llama-3.3-70b-instruct:free"

    def run_fallback(self, model=None):
        return providers.complete_with_fallback(self.p, KEY, model or self.first, "S", user("u"))

    def test_a_busy_model_is_retried_once_after_a_pause(self):
        self.fake.busy = {self.first: 1}
        reply, used, note = self.run_fallback()
        self.assertEqual(used, self.first)
        self.assertIn("busy", note)
        self.sleep.assert_called_once_with(providers.RETRY_DELAY)
        self.assertEqual(len(self.fake.posts()), 2)

    def test_a_model_that_stays_busy_is_replaced_by_another_free_one(self):
        self.fake.busy = {self.first: 9}
        reply, used, note = self.run_fallback()
        self.assertEqual(used, "google/gemini-2.0-flash:free")
        self.assertIn(f"model {self.first} was busy, so google/gemini-2.0-flash:free was used", note)
        self.assertEqual([r["body"]["model"] for r in self.fake.posts()], [self.first, self.first, "google/gemini-2.0-flash:free"])

    def test_it_gives_up_after_two_alternatives_and_never_picks_a_paid_model(self):
        self.fake.busy = {m: 9 for m in FREE}
        with self.assertRaises(providers.LLMError) as cm:
            self.run_fallback()
        self.assertEqual(cm.exception.status, 503)
        tried = {r["body"]["model"] for r in self.fake.posts()}
        self.assertNotIn("openai/gpt-4o", tried)                                                    # a free request stays free
        self.assertEqual(len(tried), 1 + providers.MAX_ALTERNATIVES)

    def test_an_unknown_model_goes_straight_to_another_without_waiting(self):
        self.fake.missing = {self.first}
        reply, used, note = self.run_fallback()
        self.assertEqual(used, "google/gemini-2.0-flash:free")
        self.assertIn("not available to this key", note)
        self.sleep.assert_not_called()

    def test_a_paid_model_may_fall_back_to_any_model(self):
        self.fake.missing = {"openai/gpt-4o"}
        reply, used, note = self.run_fallback("openai/gpt-4o")
        self.assertTrue(used.endswith(":free"))                                                     # the free ones are still preferred

    def test_other_errors_are_raised_at_once(self):
        self.fake.status = 401
        before = len(self.fake.requests)
        with self.assertRaisesRegex(providers.LLMError, "API key is not valid"):
            self.run_fallback()
        self.assertEqual(len(self.fake.requests) - before, 1)
        self.sleep.assert_not_called()

    def test_the_generator_remembers_the_model_that_worked(self):
        digest = prompts.build_digest(sample_perf())
        settings.set_key("openrouter", KEY, self.first)
        self.fake.busy = {self.first: 9}
        with mock.patch.object(providers.PROVIDERS["openrouter"], "base_url", self.fake.url):
            result = generator.feedback_from_digest(digest)
        self.assertEqual(result["model"], "google/gemini-2.0-flash:free")
        self.assertTrue(any("was busy" in w for w in result["warnings"]))
        self.assertEqual(settings.status()["providers"]["openrouter"]["model"], "google/gemini-2.0-flash:free")


class SavedSettings(WithConfig):
    def test_keys_are_saved_privately_masked_and_overridable_by_the_environment(self):
        self.assertEqual(settings.status()["provider"], None)
        settings.set_key("groq", KEY, "llama-3.3-70b-versatile")
        path = Path(self._tmp.name) / "llm.json"
        if sys.platform != "win32":
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)                           # readable by the user only
        self.assertEqual(settings.get_key("groq"), KEY)
        state = settings.status()
        self.assertEqual((state["provider"], state["providers"]["groq"]["model"]), ("groq", "llama-3.3-70b-versatile"))
        self.assertNotIn(KEY, json.dumps(state))                                                 # the full key never reaches the GUI
        self.assertTrue(state["providers"]["groq"]["key"].startswith("sk-") and state["providers"]["groq"]["key"].endswith("ghij"))
        os.environ["OPENROUTER_API_KEY"] = "from-environment-0123456789"
        self.assertEqual(settings.status()["providers"]["openrouter"]["source"], "environment")
        settings.choose("openrouter", "meta-llama/llama-3.3-70b-instruct:free")
        self.assertEqual(settings.status()["provider"], "openrouter")
        settings.remove_key("groq")
        self.assertFalse(settings.status()["providers"]["groq"]["connected"])
        os.environ.pop("OPENROUTER_API_KEY")
        with self.assertRaisesRegex(providers.LLMError, "not connected"):
            settings.choose("groq")

    def test_a_provider_that_no_longer_exists_in_the_saved_file_is_ignored(self):
        (Path(self._tmp.name) / "llm.json").write_text(json.dumps(
            {"provider": "gemini", "keys": {"gemini": "old-key-0123456789", "groq": KEY}, "models": {"gemini": "gemini-2.5-flash"}}))
        state = settings.status()
        self.assertEqual(state["provider"], "groq")                                              # falls back to one that is connected
        self.assertEqual(set(state["providers"]), {"openrouter", "groq"})

    def test_a_corrupt_file_is_treated_as_nothing_saved(self):
        (Path(self._tmp.name) / "llm.json").write_text("{not json")
        self.assertEqual(settings.status()["provider"], None)


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


class Generator(WithConfig):
    def setUp(self):
        super().setUp()
        self.digest = prompts.build_digest(sample_perf())
        self.bar = next(e["bar"] for e in self.digest["problem_events"] if e["bar"] is not None)

    def ask(self, answer, **kw):
        fake = FakeProvider(answer)
        try:
            return generator.feedback_from_digest(self.digest, provider="groq", key=KEY, base_url=fake.url, **kw), fake
        finally:
            fake.close()

    def test_good_answer_is_shaped_and_only_the_digest_is_sent(self):
        answer = {**GOOD_ANSWER, "priorities": [{**GOOD_ANSWER["priorities"][0], "bars": [self.bar]}]}
        result, fake = self.ask(answer)
        self.assertEqual((result["provider"], result["model"]), ("groq", "llama-3.3-70b-versatile"))
        self.assertEqual(result["priorities"][0]["bars"], [self.bar])
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["usage"], {"input": 12, "output": 8})
        sent = fake.requests[-1]
        self.assertIn(json.dumps(self.digest["scores_0_to_100"], separators=(",", ":")), sent["body"]["messages"][1]["content"])
        self.assertNotIn(KEY, json.dumps(sent["body"]))                                         # the key is only a header

    def test_invented_bars_are_removed_and_reported(self):
        answer = {**GOOD_ANSWER, "priorities": [{**GOOD_ANSWER["priorities"][0], "bars": [self.bar, 999]}] * 5}
        result, _ = self.ask(answer)
        self.assertEqual(len(result["priorities"]), 3)                                           # at most three
        self.assertEqual(result["priorities"][0]["bars"], [self.bar])
        self.assertTrue(any("999" in w for w in result["warnings"]))

    def test_fenced_or_chatty_json_is_still_read_and_plain_text_is_kept(self):
        fenced = "Here you go:\n```json\n" + json.dumps(GOOD_ANSWER) + "\n```"
        self.assertEqual(self.ask(fenced)[0]["summary"], GOOD_ANSWER["summary"])
        result, _ = self.ask("Just practise more slowly.")
        self.assertEqual((result["summary"], result["priorities"]), ("Just practise more slowly.", []))
        self.assertTrue(result["warnings"])

    def test_missing_connection_is_explained(self):
        with self.assertRaisesRegex(providers.LLMError, "No AI provider is connected"):
            generator.feedback_from_digest(self.digest)
        with self.assertRaisesRegex(providers.LLMError, "not connected"):
            generator.feedback_from_digest(self.digest, provider="openrouter")

    def test_saved_provider_and_model_are_used_and_the_result_stored_on_the_performance(self):
        fake = FakeProvider()
        self.addCleanup(fake.close)
        settings.set_key("openrouter", KEY, "meta-llama/llama-3.3-70b-instruct:free")
        perf = sample_perf()
        with mock.patch.object(providers.PROVIDERS["openrouter"], "base_url", fake.url):
            result = generator.generate_feedback(perf)
        self.assertEqual((result["provider"], result["model"]), ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"))
        self.assertIs(perf.feedback, result)
        self.assertEqual(fake.requests[-1]["body"]["model"], "meta-llama/llama-3.3-70b-instruct:free")


class Conversation(WithConfig):
    def setUp(self):
        super().setUp()
        self.digest = prompts.build_digest(sample_perf())
        self.fake = FakeProvider("Probably the shift overshoots.")
        self.addCleanup(self.fake.close)

    def ask(self, history, question, feedback=None):
        return chat.reply(self.digest, feedback, history, question, provider="groq", key=KEY, base_url=self.fake.url)

    def test_the_model_gets_the_measurements_the_feedback_and_the_earlier_turns(self):
        feedback = {"summary": "SUMMARY-GIVEN", "priorities": [], "strengths": ["STRENGTH-GIVEN"]}
        history = [*user("first question"), {"role": "assistant", "content": "first answer"}]
        answer = self.ask(history, "  second question  ", feedback)
        self.assertEqual(answer["reply"], "Probably the shift overshoots.")
        sent = self.fake.requests[-1]["body"]
        self.assertEqual(sent["messages"][1:], history + user("second question"))
        system = sent["messages"][0]["content"]
        self.assertIn(json.dumps(self.digest["scores_0_to_100"], separators=(",", ":")), system)
        self.assertIn("SUMMARY-GIVEN", system)
        self.assertIn("ONLY from those measurements", system)
        self.assertNotIn(KEY, json.dumps(sent))

    def test_empty_long_and_dangling_questions(self):
        with self.assertRaisesRegex(providers.LLMError, "Type a question"):
            self.ask([], "   ")
        with self.assertRaisesRegex(providers.LLMError, "under 1000"):
            self.ask([], "x" * 1001)
        old = [{"role": "assistant", "content": "orphan answer"}] + [{"role": "user", "content": f"q{i}"} if i % 2 == 0 else
                                                                    {"role": "assistant", "content": f"a{i}"} for i in range(2 * chat.MAX_TURNS)]
        self.ask(old, "latest")
        messages = self.fake.requests[-1]["body"]["messages"][1:]
        self.assertEqual(messages[0]["role"], "user")                                                # a window starts with the student
        self.assertLessEqual(len(messages), chat.MAX_TURNS + 1)
        self.assertEqual(messages[-1]["content"], "latest")


class OpenRouterSignIn(WithConfig):
    def setUp(self):
        super().setUp()
        oauth._pending.clear()

    def test_the_authorization_url_carries_a_pkce_challenge_for_a_secret_only_this_app_knows(self):
        url = urllib.parse.urlsplit(oauth.authorization_url("http://localhost:8000/auth/openrouter/callback"))
        query = urllib.parse.parse_qs(url.query)
        self.assertEqual(f"{url.scheme}://{url.netloc}{url.path}", oauth.OPENROUTER_AUTH_URL)
        self.assertEqual(query["code_challenge_method"], ["S256"])
        callback = query["callback_url"][0]
        state = callback.rsplit("/", 1)[-1]
        self.assertTrue(callback.startswith("http://localhost:8000/auth/openrouter/callback/") and len(state) >= 16)
        verifier = oauth._pending[state][0]
        self.assertGreaterEqual(len(verifier), 43)                                                      # PKCE: 43 to 128 characters
        expected = base64_url(hashlib.sha256(verifier.encode()).digest())
        self.assertEqual(query["code_challenge"], [expected])
        self.assertNotIn(verifier, url.query)                                                           # the verifier itself never leaves

    def test_exchange_sends_the_verifier_returns_the_key_and_works_once(self):
        seen = []

        class Exchange(BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                data = json.dumps({"key": "sk-or-v1-fromserver"}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Exchange)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        with mock.patch.object(oauth, "OPENROUTER_EXCHANGE_URL", f"http://127.0.0.1:{httpd.server_port}/api/v1/auth/keys"):
            callback = urllib.parse.parse_qs(urllib.parse.urlsplit(oauth.authorization_url("http://localhost:1/cb")).query)["callback_url"][0]
            state = callback.rsplit("/", 1)[-1]
            verifier = oauth._pending[state][0]
            self.assertEqual(oauth.exchange(state, "the-code"), "sk-or-v1-fromserver")
            self.assertEqual(seen, [{"code": "the-code", "code_verifier": verifier, "code_challenge_method": "S256"}])
            with self.assertRaisesRegex(providers.LLMError, "expired or was already used"):
                oauth.exchange(state, "the-code")                                                       # a state works once
            with self.assertRaisesRegex(providers.LLMError, "expired or was already used"):
                oauth.exchange("never-issued", "c")
            state2 = urllib.parse.parse_qs(urllib.parse.urlsplit(oauth.authorization_url("http://localhost:1/cb")).query)["callback_url"][0].rsplit("/", 1)[-1]
            with self.assertRaisesRegex(providers.LLMError, "code"):
                oauth.exchange(state2, "")
            state3 = urllib.parse.parse_qs(urllib.parse.urlsplit(oauth.authorization_url("http://localhost:1/cb")).query)["callback_url"][0].rsplit("/", 1)[-1]
            oauth._pending[state3] = (oauth._pending[state3][0], 0)                                     # created long ago
            with self.assertRaisesRegex(providers.LLMError, "expired"):
                oauth.exchange(state3, "c")


def base64_url(raw):
    import base64
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


class Endpoints(WithConfig):
    """The server's AI endpoints with the providers pointed at a fake."""

    @classmethod
    def setUpClass(cls):
        cls.srv = gui_server.make_server(port=0)
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        super().setUp()
        self.fake = FakeProvider()
        self.addCleanup(self.fake.close)
        for name in providers.PROVIDERS:
            patcher = mock.patch.object(providers.PROVIDERS[name], "base_url", self.fake.url)
            patcher.start()
            self.addCleanup(patcher.stop)

    def call(self, path, body=None, headers=None, method=None, raw=None):
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        h = {"Content-Type": "application/json", **(headers or {})}
        request = urllib.request.Request(self.url + path, data=data, headers=h, method=method or ("GET" if data is None else "POST"))
        try:
            with urllib.request.urlopen(request) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_connect_select_disconnect_and_nothing_secret_comes_back(self):
        self.fake.models = ["llama-3.3-70b-versatile", "gemma2-9b-it", "whisper-large-v3"]
        status, data = self.call("/api/llm/connect", {"provider": "groq", "api_key": KEY})
        self.assertEqual(status, 200)
        self.assertEqual([m["id"] for m in data["models"]], ["gemma2-9b-it", "llama-3.3-70b-versatile"])      # no speech model
        self.assertEqual(data["status"]["provider"], "groq")
        self.assertEqual(data["status"]["providers"]["groq"]["model"], "llama-3.3-70b-versatile")
        self.assertNotIn(KEY, json.dumps(data))
        status, data = self.call("/api/llm")
        self.assertTrue(data["providers"]["groq"]["connected"] and not data["providers"]["openrouter"]["connected"])
        self.assertNotIn(KEY, json.dumps(data))
        self.fake.models = FREE + ["openai/gpt-4o"]
        status, data = self.call("/api/llm/connect", {"provider": "openrouter", "api_key": KEY})
        self.assertEqual(data["status"]["providers"]["openrouter"]["model"], "meta-llama/llama-3.3-70b-instruct:free")     # a free model
        status, data = self.call("/api/llm/select", {"provider": "groq", "model": "gemma2-9b-it"})
        self.assertEqual((data["status"]["provider"], data["status"]["providers"]["groq"]["model"]), ("groq", "gemma2-9b-it"))
        status, data = self.call("/api/llm/models", {"provider": "openrouter"})
        self.assertEqual(len(data["models"]), 4)
        status, data = self.call("/api/llm/disconnect", {"provider": "groq"})
        self.assertFalse(data["status"]["providers"]["groq"]["connected"])
        self.assertEqual(data["status"]["provider"], "openrouter")

    def test_a_bad_key_is_refused_and_not_saved_and_old_providers_are_unknown(self):
        self.fake.models_status = 401
        status, data = self.call("/api/llm/connect", {"provider": "groq", "api_key": "wrong-key-0123456789"})
        self.assertEqual(status, 400)
        self.assertIn("API key is not valid", data["error"])
        self.assertFalse(settings.status()["providers"]["groq"]["connected"])
        self.assertEqual(self.call("/api/llm/connect", {"provider": "groq", "api_key": "  "})[0], 400)
        self.assertEqual(self.call("/api/llm/connect", {"provider": "gemini", "api_key": KEY})[0], 400)

    def test_feedback_is_written_once_per_analysis(self):
        settings.set_key("groq", KEY)
        digest = prompts.build_digest(sample_perf())
        bar = next(e["bar"] for e in digest["problem_events"] if e["bar"] is not None)
        self.fake.answer = {**GOOD_ANSWER, "priorities": [{**GOOD_ANSWER["priorities"][0], "bars": [bar]}]}
        gui_server.RUNS["a" * 32] = {"digest": digest}
        self.addCleanup(gui_server.RUNS.pop, "a" * 32, None)
        status, data = self.call("/api/feedback", {"run": "a" * 32})
        self.assertEqual(status, 200)
        self.assertEqual(data["priorities"][0]["bars"], [bar])
        calls = len(self.fake.posts())
        self.assertEqual(self.call("/api/feedback", {"run": "a" * 32})[1], data)                  # the saved answer: no second call
        self.assertEqual(len(self.fake.posts()), calls)
        self.assertEqual(self.call("/api/feedback", {"run": "b" * 32})[0], 404)

    def test_a_busy_model_is_reported_in_the_feedback_not_as_a_failure(self):
        settings.set_key("openrouter", KEY, "meta-llama/llama-3.3-70b-instruct:free")
        self.fake.models = FREE
        self.fake.busy = {"meta-llama/llama-3.3-70b-instruct:free": 9}
        digest = prompts.build_digest(sample_perf())
        gui_server.RUNS["e" * 32] = {"digest": digest}
        self.addCleanup(gui_server.RUNS.pop, "e" * 32, None)
        status, data = self.call("/api/feedback", {"run": "e" * 32})
        self.assertEqual(status, 200)
        self.assertEqual(data["model"], "google/gemini-2.0-flash:free")
        self.assertTrue(any("busy" in w for w in data["warnings"]))

    def test_chat_keeps_the_conversation_per_analysis_and_can_restart_it(self):
        settings.set_key("groq", KEY)
        self.fake.answer = "Work on the shift in bar 2."
        digest = prompts.build_digest(sample_perf())
        gui_server.RUNS["c" * 32] = {"digest": digest, "feedback": {"summary": "S", "priorities": [], "strengths": []}}
        self.addCleanup(gui_server.RUNS.pop, "c" * 32, None)
        status, data = self.call("/api/chat", {"run": "c" * 32, "message": "What first?"})
        self.assertEqual((status, data["reply"], data["turns"]), (200, "Work on the shift in bar 2.", 1))
        status, data = self.call("/api/chat", {"run": "c" * 32, "message": "And then?"})
        self.assertEqual(data["turns"], 2)
        self.assertEqual([m["content"] for m in self.fake.requests[-1]["body"]["messages"][1:]],
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

    def test_sign_in_with_openrouter_end_to_end(self):
        oauth._pending.clear()
        self.fake.models = ["meta-llama/llama-3.3-70b-instruct:free", "openai/gpt-4o"]
        exchanged = []

        def exchange(state, code):
            exchanged.append((state, code))
            return "sk-or-v1-signedin-0123456789"

        conn = lambda: http.client.HTTPConnection("127.0.0.1", self.srv.server_port)
        c = conn()
        c.request("GET", "/auth/openrouter/start")
        r = c.getresponse()
        r.read()
        self.assertEqual(r.status, 302)
        target = urllib.parse.urlsplit(r.getheader("Location"))
        self.assertEqual(f"{target.scheme}://{target.netloc}{target.path}", oauth.OPENROUTER_AUTH_URL)
        callback = urllib.parse.parse_qs(target.query)["callback_url"][0]
        self.assertTrue(callback.startswith(f"http://localhost:{self.srv.server_port}/auth/openrouter/callback/"))
        state = callback.rsplit("/", 1)[-1]
        with mock.patch.object(oauth, "exchange", exchange):
            c = conn()
            c.request("GET", f"/auth/openrouter/callback/{state}?code=abc123")                 # the browser comes back from OpenRouter
            r = c.getresponse()
            r.read()
        self.assertEqual((r.status, r.getheader("Location")), (302, "/?ai=connected#ai"))
        self.assertEqual(exchanged, [(state, "abc123")])
        status = settings.status()
        self.assertTrue(status["providers"]["openrouter"]["connected"])
        self.assertEqual(status["providers"]["openrouter"]["model"], "meta-llama/llama-3.3-70b-instruct:free")      # a free model
        self.assertEqual(status["provider"], "openrouter")
        settings.remove_key("openrouter")                                                      # a callback not started here saves nothing
        c = conn()
        c.request("GET", "/auth/openrouter/callback/forged-state?code=abc123")
        r = c.getresponse()
        r.read()
        self.assertIn("/?ai_error=", r.getheader("Location"))
        self.assertFalse(settings.status()["providers"]["openrouter"]["connected"])
        c = conn()
        c.request("GET", "/auth/openrouter/start", headers={"Host": "evil.example"})                # DNS rebinding: refused
        r = c.getresponse()
        r.read()
        self.assertEqual(r.status, 403)

    def test_other_web_pages_cannot_drive_these_endpoints(self):
        settings.set_key("groq", KEY)
        self.assertEqual(self.call("/api/llm", headers={"Origin": "https://evil.example"})[0], 403)           # cross-site page
        self.assertEqual(self.call("/api/llm", headers={"Host": "evil.example"})[0], 403)                      # DNS rebinding
        self.assertEqual(self.call("/api/llm/disconnect", {"provider": "groq"}, {"Origin": "http://evil.example"})[0], 403)
        self.assertTrue(settings.status()["providers"]["groq"]["connected"])                                    # nothing was changed
        self.assertEqual(self.call("/api/llm/disconnect", raw=b"provider=groq", headers={"Content-Type": "text/plain"})[0], 400)
        self.assertEqual(self.call("/api/llm/disconnect", raw=b"[1]")[0], 400)
        self.assertEqual(self.call("/api/analyze", raw=b"x", headers={"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.call("/api/llm", headers={"Origin": self.url})[0], 200)                         # the page itself is fine


class Page(WithConfig):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.perf = performance(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_page_carries_the_digest_and_the_run_but_never_a_key(self):
        settings.set_key("groq", KEY)
        html = build_page(self.perf, "media.mp4", run="r" * 32)
        data = page_data(html)
        self.assertEqual(data["run"], "r" * 32)
        self.assertIn("overall", data["digest"]["scores_0_to_100"])
        self.assertIsNone(data["feedback"])
        self.assertNotIn(KEY, html)


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

    CONNECTED = (200, {"provider": "groq", "providers": {"groq": {"label": "Groq (free tier)", "connected": True, "model": "m"}}})

    @unittest.skipUnless(HAS_BROWSER, "needs node + playwright")
    def test_a_failure_shows_the_providers_message_not_undefined_and_offers_a_retry(self):
        out = self.run_page({"/api/llm": self.CONNECTED, "/api/chat": (200, {"messages": []}),
                             "/api/feedback": (400, {"error": "That model was not found for this key (models/x is not found)."})},
                            "return {text: document.querySelector('#feedback').textContent, retry: !![...document.querySelectorAll('#feedback button')]"
                            ".find(b => b.textContent === 'Write feedback' && !b.hidden)};")
        self.assertIn("That model was not found for this key", out["text"])
        self.assertNotIn("undefined", out["text"])
        self.assertTrue(out["retry"])

    @unittest.skipUnless(HAS_BROWSER, "needs node + playwright")
    def test_feedback_is_written_automatically_and_the_chat_works_as_text(self):
        feedback = {"provider": "groq", "model": "m", "summary": "Auto summary.", "warnings": [],
                    "priorities": [{"title": "Shifts", "bars": [2], "why": "why text", "practice": "practice text"}], "strengths": ["Steady"]}
        out = self.run_page({"/api/llm": self.CONNECTED, "/api/feedback": (200, feedback),
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
    def test_without_a_connected_provider_the_page_says_how_to_connect(self):
        out = self.run_page({"/api/llm": (200, {"provider": None, "providers": {}})},
                            "return {text: document.querySelector('#feedback').textContent, chat: !!document.querySelector('#feedback .chat')};")
        self.assertIn("connect a provider", out["text"])
        self.assertFalse(out["chat"])

    @unittest.skipUnless(HAS_BROWSER, "needs node + playwright")
    def test_model_text_is_shown_as_text_never_as_html(self):
        evil = "<img src=x onerror=\"window.pwned=1\"><b>bold</b>"
        self.perf.feedback = {"provider": "groq", "model": "m", "summary": evil, "warnings": ["</script>"],
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
