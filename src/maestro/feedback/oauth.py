# "Sign in with OpenRouter": the OAuth PKCE flow that hands the app an API key tied to the user's own OpenRouter account (new file).
#
# OpenRouter (an OpenAI-style gateway to many models, some free) documents a sign-in without client registration or a secret:
#   1. the app sends the user's browser to OPENROUTER_AUTH_URL with a callback_url (a localhost address is allowed, on any port) and a PKCE
#      code_challenge (SHA-256 of a random code_verifier that only this app knows);
#   2. the user logs in on OpenRouter and agrees; OpenRouter redirects to the callback with ?code=...;
#   3. the app POSTs {code, code_verifier, code_challenge_method: "S256"} to OPENROUTER_EXCHANGE_URL and receives {"key": "sk-or-..."}:
#      a normal API key whose usage is the user's own, which is saved like a pasted key (feedback/settings.py).
# The Claude.ai, ChatGPT and Gemini apps offer no such sign-in for their subscriptions, so they stay on pasted API keys. gui/server.py serves
# /auth/openrouter/start and /auth/openrouter/callback/<state> with the two functions here. The state in the callback path is a random
# one-time secret that ties the callback to the sign-in this server started, and expires after STATE_LIFETIME seconds.
import base64
import hashlib
import secrets
import threading
import time
from urllib.parse import urlencode

from maestro.feedback.providers import LLMError, http_json

OPENROUTER_AUTH_URL = "https://openrouter.ai/auth"
OPENROUTER_EXCHANGE_URL = "https://openrouter.ai/api/v1/auth/keys"
STATE_LIFETIME = 600
_pending = {}                                    # state -> (code_verifier, created)
_lock = threading.Lock()


def _challenge(verifier):
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


def authorization_url(callback_base):
    """Start a sign-in -> the OpenRouter URL to send the browser to. `callback_base` is this server's callback URL without the
    state, e.g. http://localhost:8000/auth/openrouter/callback"""
    verifier, state = secrets.token_urlsafe(64), secrets.token_urlsafe(16)
    now = time.time()
    with _lock:
        for old in [s for s, (_, t) in _pending.items() if now - t > STATE_LIFETIME]:
            del _pending[old]
        _pending[state] = (verifier, now)
    return OPENROUTER_AUTH_URL + "?" + urlencode({"callback_url": f"{callback_base}/{state}", "code_challenge": _challenge(verifier),
                                                  "code_challenge_method": "S256"})


def exchange(state, code):
    """Finish a sign-in: the callback's state and code -> the API key. LLMError if the state is unknown, used or expired, or the
    exchange fails. A state works once."""
    with _lock:
        verifier, created = _pending.pop(state, (None, 0))
    if verifier is None or time.time() - created > STATE_LIFETIME:
        raise LLMError("This sign-in link has expired or was already used. Start the sign-in again.")
    if not code:
        raise LLMError("OpenRouter did not return a code (the sign-in was cancelled?).")
    data = http_json(OPENROUTER_EXCHANGE_URL, {}, {"code": code, "code_verifier": verifier, "code_challenge_method": "S256"})
    key = data.get("key") if isinstance(data, dict) else None
    if not key:
        raise LLMError("OpenRouter did not return a key.")
    return key
