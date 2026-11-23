"""The LLM feedback layer: the three providers (against a local fake of each API), the saved settings, the digest, the generator,
the server endpoints and their protection, and the feedback card of the page."""
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


class FakeProvider:
    """A local server that answers like the Anthropic, OpenAI and Gemini APIs and records what it was sent."""

    def __init__(self, answer=None, status=200, models_status=200):
        outer = self
        self.requests, self.answer, self.status, self.models_status = [], answer if answer is not None else GOOD_ANSWER, status, models_status
        self.missing = set()                                      # model ids that answer 404, like a retired or unavailable model
        self.openai_models = ["gpt-4o", "text-embedding-3-small", "whisper-1", "o3-mini", "gpt-4o-realtime-preview"]
        self.gemini_models = [("gemini-2.0-flash", "generateContent"), ("gemini-2.5-flash", "generateContent"),
                              ("gemini-2.5-flash-lite", "generateContent"), ("gemini-2.5-pro", "generateContent"),
                              ("gemini-3-flash-preview", "generateContent"), ("embedding-001", "embedContent"),
                              ("gemini-old", "countTokens")]

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
                outer.requests.append({"method": self.command, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                return body

            def do_GET(self):
                self.record()
                if outer.models_status != 200:
                    return self.reply(outer.models_status, {"error": {"message": "bad key"}})
                if self.path.startswith("/v1/models") and "anthropic-version" in {k.lower() for k in self.headers}:
                    return self.reply(200, {"data": [{"id": "claude-sonnet-5-5", "display_name": "Claude Sonnet 5.5"},
                                                      {"id": "claude-haiku-5-5", "display_name": "Claude Haiku 5.5"}]})
                if self.path.startswith("/v1/models"):
                    return self.reply(200, {"data": [{"id": i} for i in outer.openai_models]})
                return self.reply(200, {"models": [{"name": f"models/{i}", "displayName": i, "supportedGenerationMethods": [m]}
                                                   for i, m in outer.gemini_models]})

            def do_POST(self):
                body = self.record()
                model = (body or {}).get("model") or self.path.split("/models/")[-1].split(":")[0]
                if model in outer.missing:
                    return self.reply(404, {"error": {"message": f"models/{model} is not found for API version v1beta"}})
                if outer.status != 200:
                    return self.reply(outer.status, {"error": {"message": "quota exceeded"}})
                text = outer.answer if isinstance(outer.answer, str) else json.dumps(outer.answer)
                if self.path == "/v1/messages":
                    return self.reply(200, {"content": [{"type": "text", "text": text}], "usage": {"input_tokens": 11, "output_tokens": 7}})
                if self.path == "/v1/chat/completions":
                    return self.reply(200, {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 12, "completion_tokens": 8}})
                return self.reply(200, {"candidates": [{"content": {"parts": [{"text": text}]}}],
                                        "usageMetadata": {"promptTokenCount": 13, "candidatesTokenCount": 9}})

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class WithConfig(unittest.TestCase):
    """Settings in a temporary config folder, no provider keys in the environment."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")}
        env["MAESTRO_CONFIG"] = self._tmp.name
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)


class ProviderCalls(WithConfig):
    def setUp(self):
        super().setUp()
        self.fake = FakeProvider()
        self.addCleanup(self.fake.close)

    def provider(self, name):
        return providers.get(name, self.fake.url)

    def test_anthropic_request_and_answer(self):
        p = self.provider("anthropic")
        models = p.list_models(KEY)
        self.assertEqual([m["id"] for m in models], ["claude-sonnet-5-5", "claude-haiku-5-5"])
        reply = p.complete(KEY, "claude-sonnet-5-5", "SYSTEM TEXT", [{"role": "user", "content": "USER TEXT"}], 500)
        sent = self.fake.requests[-1]
        self.assertEqual(sent["path"], "/v1/messages")
        self.assertEqual((sent["headers"]["x-api-key"], sent["headers"]["anthropic-version"]), (KEY, "2023-06-01"))
        self.assertEqual(sent["body"], {"model": "claude-sonnet-5-5", "max_tokens": 500, "system": "SYSTEM TEXT",
                                        "messages": [{"role": "user", "content": "USER TEXT"}]})
        self.assertEqual(json.loads(reply["text"])["summary"], GOOD_ANSWER["summary"])
        self.assertEqual(reply["usage"], {"input": 11, "output": 7})

    def test_openai_request_answer_and_model_filter(self):
        p = self.provider("openai")
        self.assertEqual([m["id"] for m in p.list_models(KEY)], ["gpt-4o", "o3-mini"])           # no embeddings, speech or realtime
        reply = p.complete(KEY, "gpt-4o", "SYS", [{"role": "user", "content": "USR"}], 400, json_mode=True)
        sent = self.fake.requests[-1]
        self.assertEqual(sent["path"], "/v1/chat/completions")
        self.assertEqual(sent["headers"]["authorization"], f"Bearer {KEY}")
        self.assertEqual(sent["body"]["messages"], [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}])
        self.assertEqual((sent["body"]["max_completion_tokens"], sent["body"]["response_format"]), (400, {"type": "json_object"}))
        self.assertEqual(reply["usage"], {"input": 12, "output": 8})

    def test_gemini_request_answer_and_model_filter(self):
        p = self.provider("gemini")
        self.assertEqual([m["id"] for m in p.list_models(KEY)], ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite",
                                                                   "gemini-2.5-pro", "gemini-3-flash-preview"])     # only text generators
        reply = p.complete(KEY, "gemini-2.5-flash", "SYS", [{"role": "user", "content": "USR"}], 300, json_mode=True)
        sent = self.fake.requests[-1]
        self.assertEqual(sent["path"], "/v1beta/models/gemini-2.5-flash:generateContent")
        self.assertEqual(sent["headers"]["x-goog-api-key"], KEY)                                 # in a header, not the URL
        self.assertNotIn(KEY, sent["path"])
        self.assertEqual(sent["body"]["systemInstruction"], {"parts": [{"text": "SYS"}]})
        self.assertEqual(sent["body"]["generationConfig"], {"maxOutputTokens": 300, "responseMimeType": "application/json"})
        self.assertEqual(reply["usage"], {"input": 13, "output": 9})

    def test_failures_become_readable_messages(self):
        self.fake.status = 429
        with self.assertRaisesRegex(providers.LLMError, "quota"):
            self.provider("openai").complete(KEY, "gpt-4o", "s", [{"role": "user", "content": "u"}])
        self.fake.models_status = 401
        with self.assertRaisesRegex(providers.LLMError, "API key is not valid"):
            self.provider("anthropic").list_models("bad")
        with self.assertRaisesRegex(providers.LLMError, "Could not reach"):
            providers.get("gemini", "http://127.0.0.1:1").list_models(KEY)                      # nothing listens there
        with self.assertRaisesRegex(providers.LLMError, "Unknown provider"):
            providers.get("llama")

    def test_a_conversation_is_sent_as_alternating_turns_in_each_providers_format(self):
        turns = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}]
        self.provider("anthropic").complete(KEY, "m", "S", turns)
        self.assertEqual(self.fake.requests[-1]["body"]["messages"], turns)
        self.provider("openai").complete(KEY, "m", "S", turns)
        self.assertEqual([m["role"] for m in self.fake.requests[-1]["body"]["messages"]], ["system", "user", "assistant", "user"])
        self.provider("gemini").complete(KEY, "m", "S", turns)
        self.assertEqual([c["role"] for c in self.fake.requests[-1]["body"]["contents"]], ["user", "model", "user"])      # Gemini says "model"

    def test_openai_compatible_services_get_the_common_subset(self):
        self.fake.openai_models = ["meta-llama/llama-3.3-70b-instruct:free", "openai/gpt-4o", "google/gemini-2.0-flash:free",
                                   "openai/text-embedding-3-small", "mistralai/mistral-7b-instruct:free"]
        for name in ("openrouter", "groq"):
            p = self.provider(name)
            reply = p.complete(KEY, "some/model", "SYS", [{"role": "user", "content": "USR"}], 321, json_mode=True)
            sent = self.fake.requests[-1]["body"]
            self.assertEqual(sent["max_tokens"], 321)                                              # the widely supported spelling ...
            self.assertNotIn("response_format", sent)                                               # ... and no option some models reject
            self.assertNotIn("max_completion_tokens", sent)
            self.assertEqual(json.loads(reply["text"])["summary"], GOOD_ANSWER["summary"])
        models = self.provider("openrouter").list_models(KEY)
        self.assertEqual([m["id"] for m in models], ["google/gemini-2.0-flash:free", "meta-llama/llama-3.3-70b-instruct:free",
                                                      "mistralai/mistral-7b-instruct:free", "openai/gpt-4o"])   # free first, embeddings gone
        self.assertEqual(models[1]["label"], "meta-llama/llama-3.3-70b-instruct:free (free)")
        self.assertEqual(self.provider("openrouter").pick_model(models), "meta-llama/llama-3.3-70b-instruct:free")

    def test_the_custom_service_needs_its_url_and_tolerates_a_trailing_v1(self):
        with self.assertRaisesRegex(providers.LLMError, "base URL"):
            providers.get("custom")
        with self.assertRaisesRegex(providers.LLMError, "http"):
            providers.get("custom", "api.mistral.ai")
        self.assertEqual(providers.get("custom", "https://api.mistral.ai/v1/").base_url, "https://api.mistral.ai")
        self.assertEqual(providers.get("custom", "https://api.mistral.ai").base_url, "https://api.mistral.ai")

    def test_default_model_choice(self):
        gemini = providers.PROVIDERS["gemini"]
        ids = lambda *names: [{"id": n} for n in names]
        self.assertEqual(gemini.pick_model(ids("gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-flash-preview", "gemini-2.5-pro")),
                         "gemini-2.5-flash")                                                         # newest stable Flash, not lite or preview
        self.assertEqual(gemini.pick_model(ids("gemini-1.5-pro", "gemini-2.0-flash")), "gemini-2.0-flash")
        self.assertEqual(gemini.pick_model(ids("gemini-1.5-pro", "gemini-2.5-pro")), "gemini-2.5-pro")
        self.assertIsNone(gemini.pick_model([]))
        self.assertEqual(providers.PROVIDERS["anthropic"].pick_model(ids("claude-haiku-5-5", "claude-sonnet-5-5")), "claude-sonnet-5-5")

    def test_a_model_the_key_cannot_use_is_replaced_once_and_other_errors_are_not_retried(self):
        gemini = self.provider("gemini")
        self.fake.missing = {"gemini-2.5-flash"}
        reply, used, note = providers.complete_with_fallback(gemini, KEY, "gemini-2.5-flash", "S", [{"role": "user", "content": "u"}])
        self.assertEqual(used, "gemini-2.0-flash")                                                  # the newest other stable Flash
        self.assertIn("gemini-2.5-flash was not available", note)
        self.assertEqual(json.loads(reply["text"])["summary"], GOOD_ANSWER["summary"])
        self.fake.status = 429
        before = len(self.fake.requests)
        with self.assertRaisesRegex(providers.LLMError, "quota"):
            providers.complete_with_fallback(self.provider("openai"), KEY, "gpt-4o", "S", [{"role": "user", "content": "u"}])
        self.assertEqual(len(self.fake.requests) - before, 1)                                       # no retry for a quota error
        self.fake.status, self.fake.missing = 200, {"gpt-4o", "o3-mini"}
        with self.assertRaises(providers.LLMError) as cm:
            providers.complete_with_fallback(self.provider("openai"), KEY, "gpt-4o", "S", [{"role": "user", "content": "u"}])
        self.assertEqual(cm.exception.status, 404)                                                  # nothing else to try: the real error


class SavedSettings(WithConfig):
    def test_keys_are_saved_privately_masked_and_overridable_by_the_environment(self):
        self.assertEqual(settings.status()["provider"], None)
        settings.set_key("anthropic", KEY, "claude-haiku-5-5")
        path = Path(self._tmp.name) / "llm.json"
        if sys.platform != "win32":
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)                           # readable by the user only
        self.assertEqual(settings.get_key("anthropic"), KEY)
        state = settings.status()
        self.assertEqual((state["provider"], state["providers"]["anthropic"]["model"]), ("anthropic", "claude-haiku-5-5"))
        self.assertNotIn(KEY, json.dumps(state))                                                 # the full key never reaches the GUI
        self.assertTrue(state["providers"]["anthropic"]["key"].startswith("sk-") and state["providers"]["anthropic"]["key"].endswith("ghij"))
        os.environ["OPENAI_API_KEY"] = "from-environment-0123456789"
        self.assertTrue(settings.status()["providers"]["openai"]["connected"])
        self.assertEqual(settings.status()["providers"]["openai"]["source"], "environment")
        settings.choose("openai", "gpt-4o")
        self.assertEqual(settings.status()["provider"], "openai")
        settings.remove_key("anthropic")
        self.assertFalse(settings.status()["providers"]["anthropic"]["connected"])
        with self.assertRaisesRegex(providers.LLMError, "not connected"):
            settings.choose("gemini")

    def test_gemini_accepts_either_environment_variable(self):
        os.environ["GOOGLE_API_KEY"] = "google-key-0123456789"
        self.assertEqual(settings.get_key("gemini"), "google-key-0123456789")

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
            return generator.feedback_from_digest(self.digest, provider="anthropic", key=KEY, base_url=fake.url, **kw), fake
        finally:
            fake.close()

    def test_good_answer_is_shaped_and_only_the_digest_is_sent(self):
        answer = {**GOOD_ANSWER, "priorities": [{**GOOD_ANSWER["priorities"][0], "bars": [self.bar]}]}
        result, fake = self.ask(answer)
        self.assertEqual((result["provider"], result["model"]), ("anthropic", "claude-sonnet-5-5"))
        self.assertEqual(result["priorities"][0]["bars"], [self.bar])
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["usage"], {"input": 11, "output": 7})
        sent = fake.requests[-1]
        self.assertIn(json.dumps(self.digest["scores_0_to_100"], separators=(",", ":")), sent["body"]["messages"][0]["content"])
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
            generator.feedback_from_digest(self.digest, provider="gemini")

    def test_saved_provider_and_model_are_used_and_the_result_stored_on_the_performance(self):
        fake = FakeProvider()
        self.addCleanup(fake.close)
        settings.set_key("openai", KEY, "o3-mini")
        perf = sample_perf()
        with mock.patch.object(providers.PROVIDERS["openai"], "base_url", fake.url):
            result = generator.generate_feedback(perf)
        self.assertEqual((result["provider"], result["model"]), ("openai", "o3-mini"))
        self.assertIs(perf.feedback, result)
        self.assertEqual(fake.requests[-1]["body"]["model"], "o3-mini")

    def test_an_unavailable_model_is_replaced_remembered_and_reported(self):
        fake = FakeProvider()
        self.addCleanup(fake.close)
        fake.missing = {"gemini-2.5-flash"}                                # the Google 404: the saved model is not offered to this key
        settings.set_key("gemini", KEY, "gemini-2.5-flash")
        with mock.patch.object(providers.PROVIDERS["gemini"], "base_url", fake.url):
            result = generator.feedback_from_digest(self.digest)
        self.assertEqual(result["model"], "gemini-2.0-flash")
        self.assertTrue(any("gemini-2.5-flash was not available" in w for w in result["warnings"]))
        self.assertEqual(settings.status()["providers"]["gemini"]["model"], "gemini-2.0-flash")            # next time it is not even tried
        self.assertEqual(result["summary"], GOOD_ANSWER["summary"])


class Conversation(WithConfig):
    def setUp(self):
        super().setUp()
        self.digest = prompts.build_digest(sample_perf())
        self.fake = FakeProvider("Probably the shift overshoots.")
        self.addCleanup(self.fake.close)

    def ask(self, history, question, feedback=None):
        return chat.reply(self.digest, feedback, history, question, provider="anthropic", key=KEY, base_url=self.fake.url)

    def test_the_model_gets_the_measurements_the_feedback_and_the_earlier_turns(self):
        feedback = {"summary": "SUMMARY-GIVEN", "priorities": [], "strengths": ["STRENGTH-GIVEN"]}
        history = [{"role": "user", "content": "first question"}, {"role": "assistant", "content": "first answer"}]
        answer = self.ask(history, "  second question  ", feedback)
        self.assertEqual(answer["reply"], "Probably the shift overshoots.")
        sent = self.fake.requests[-1]["body"]
        self.assertEqual(sent["messages"], history + [{"role": "user", "content": "second question"}])
        self.assertIn(json.dumps(self.digest["scores_0_to_100"], separators=(",", ":")), sent["system"])
        self.assertIn("SUMMARY-GIVEN", sent["system"])
        self.assertIn("ONLY from those measurements", sent["system"])
        self.assertNotIn(KEY, json.dumps(sent))

    def test_empty_long_and_dangling_questions(self):
        with self.assertRaisesRegex(providers.LLMError, "Type a question"):
            self.ask([], "   ")
        with self.assertRaisesRegex(providers.LLMError, "under 1000"):
            self.ask([], "x" * 1001)
        old = [{"role": "assistant", "content": "orphan answer"}] + [{"role": "user", "content": f"q{i}"} if i % 2 == 0 else
                                                                    {"role": "assistant", "content": f"a{i}"} for i in range(2 * chat.MAX_TURNS)]
        self.ask(old, "latest")
        messages = self.fake.requests[-1]["body"]["messages"]
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
        status, data = self.call("/api/llm/connect", {"provider": "anthropic", "api_key": KEY})
        self.assertEqual(status, 200)
        self.assertEqual([m["id"] for m in data["models"]], ["claude-sonnet-5-5", "claude-haiku-5-5"])
        self.assertEqual(data["status"]["provider"], "anthropic")
        self.assertEqual(data["status"]["providers"]["anthropic"]["model"], "claude-sonnet-5-5")
        self.assertNotIn(KEY, json.dumps(data))
        status, data = self.call("/api/llm")
        self.assertTrue(data["providers"]["anthropic"]["connected"] and not data["providers"]["openai"]["connected"])
        self.assertNotIn(KEY, json.dumps(data))
        self.call("/api/llm/connect", {"provider": "openai", "api_key": KEY})
        status, data = self.call("/api/llm/select", {"provider": "anthropic", "model": "claude-haiku-5-5"})
        self.assertEqual((data["status"]["provider"], data["status"]["providers"]["anthropic"]["model"]), ("anthropic", "claude-haiku-5-5"))
        status, data = self.call("/api/llm/models", {"provider": "anthropic"})
        self.assertEqual(len(data["models"]), 2)
        status, data = self.call("/api/llm/disconnect", {"provider": "anthropic"})
        self.assertFalse(data["status"]["providers"]["anthropic"]["connected"])
        self.assertEqual(data["status"]["provider"], "openai")

    def test_a_bad_key_is_refused_and_not_saved(self):
        self.fake.models_status = 401
        status, data = self.call("/api/llm/connect", {"provider": "gemini", "api_key": "wrong-key-0123456789"})
        self.assertEqual(status, 400)
        self.assertIn("API key is not valid", data["error"])
        self.assertFalse(settings.status()["providers"]["gemini"]["connected"])
        self.assertEqual(self.call("/api/llm/connect", {"provider": "gemini", "api_key": "  "})[0], 400)
        self.assertEqual(self.call("/api/llm/connect", {"provider": "llama", "api_key": KEY})[0], 400)

    def test_feedback_is_written_once_per_analysis(self):
        settings.set_key("anthropic", KEY)
        digest = prompts.build_digest(sample_perf())
        bar = next(e["bar"] for e in digest["problem_events"] if e["bar"] is not None)
        self.fake.answer = {**GOOD_ANSWER, "priorities": [{**GOOD_ANSWER["priorities"][0], "bars": [bar]}]}
        gui_server.RUNS["a" * 32] = {"digest": digest}
        self.addCleanup(gui_server.RUNS.pop, "a" * 32, None)
        status, data = self.call("/api/feedback", {"run": "a" * 32})
        self.assertEqual(status, 200)
        self.assertEqual(data["priorities"][0]["bars"], [bar])
        calls = len([r for r in self.fake.requests if r["method"] == "POST"])
        self.assertEqual(self.call("/api/feedback", {"run": "a" * 32})[1], data)                  # the saved answer: no second paid call
        self.assertEqual(len([r for r in self.fake.requests if r["method"] == "POST"]), calls)
        self.assertEqual(self.call("/api/feedback", {"run": "b" * 32})[0], 404)


    def test_a_custom_service_needs_its_url_and_remembers_it(self):
        self.fake.openai_models = ["mistral-large-latest", "mistral-embed"]
        status, data = self.call("/api/llm/connect", {"provider": "custom", "api_key": KEY})
        self.assertEqual(status, 400)
        self.assertIn("base URL", data["error"])
        status, data = self.call("/api/llm/connect", {"provider": "custom", "api_key": KEY, "base_url": self.fake.url + "/v1"})
        self.assertEqual(status, 200)
        self.assertEqual([m["id"] for m in data["models"]], ["mistral-large-latest"])
        custom = data["status"]["providers"]["custom"]
        self.assertEqual((custom["needs_url"], custom["base_url"], custom["model"]), (True, self.fake.url + "/v1", "mistral-large-latest"))
        status, data = self.call("/api/llm/models", {"provider": "custom"})                       # later calls use the saved URL
        self.assertEqual(status, 200)
        self.assertEqual(self.call("/api/llm/disconnect", {"provider": "custom"})[1]["status"]["providers"]["custom"]["base_url"], "")

    def test_connecting_gemini_picks_a_working_default_model(self):
        status, data = self.call("/api/llm/connect", {"provider": "gemini", "api_key": KEY})
        self.assertEqual(data["status"]["providers"]["gemini"]["model"], "gemini-2.5-flash")      # not a preview, lite or old model

    def test_chat_keeps_the_conversation_per_analysis_and_can_restart_it(self):
        settings.set_key("anthropic", KEY)
        self.fake.answer = "Work on the shift in bar 2."
        digest = prompts.build_digest(sample_perf())
        gui_server.RUNS["c" * 32] = {"digest": digest, "feedback": {"summary": "S", "priorities": [], "strengths": []}}
        self.addCleanup(gui_server.RUNS.pop, "c" * 32, None)
        status, data = self.call("/api/chat", {"run": "c" * 32, "message": "What first?"})
        self.assertEqual((status, data["reply"], data["turns"]), (200, "Work on the shift in bar 2.", 1))
        status, data = self.call("/api/chat", {"run": "c" * 32, "message": "And then?"})
        self.assertEqual(data["turns"], 2)
        self.assertEqual([m["content"] for m in self.fake.requests[-1]["body"]["messages"]],
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
        self.fake.openai_models = ["meta-llama/llama-3.3-70b-instruct:free", "openai/gpt-4o"]
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
        # a callback that does not belong to a sign-in started here saves nothing
        settings.remove_key("openrouter")
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
        settings.set_key("anthropic", KEY)
        self.assertEqual(self.call("/api/llm", headers={"Origin": "https://evil.example"})[0], 403)           # cross-site page
        self.assertEqual(self.call("/api/llm", headers={"Host": "evil.example"})[0], 403)                      # DNS rebinding
        self.assertEqual(self.call("/api/llm/disconnect", {"provider": "anthropic"}, {"Origin": "http://evil.example"})[0], 403)
        self.assertTrue(settings.status()["providers"]["anthropic"]["connected"])                                # nothing was changed
        self.assertEqual(self.call("/api/llm/disconnect", raw=b"provider=anthropic", headers={"Content-Type": "text/plain"})[0], 400)
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
        settings.set_key("anthropic", KEY)
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

    CONNECTED = (200, {"provider": "gemini", "providers": {"gemini": {"label": "Gemini (Google)", "connected": True, "model": "m"}}})

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
        feedback = {"provider": "gemini", "model": "m", "summary": "Auto summary.", "warnings": [],
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
        self.perf.feedback = {"provider": "anthropic", "model": "m", "summary": evil, "warnings": ["</script>"],
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
