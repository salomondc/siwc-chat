#!/usr/bin/env python3
"""Minimal terminal chat that uses your ChatGPT plan via the
Sign in with ChatGPT open-source flow (OAuth + Responses API).

First run: opens your browser to register the client and authorize
plan usage. Credentials are stored in ~/.config/siwc-chat/ (0600).

Usage:
  python3 chat.py           # chat, multi-turn in memory
  python3 chat.py -m SLUG   # use another model slug
  python3 chat.py --models  # list models available to your account

This is a minimal reference app; a production build must also validate
the ID token signature against OpenAI's JWKS.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import sys
import time
import threading
import queue
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import voice
import logger


VOICE_INSTRUCTIONS = (
    "Your output is always turned into voice in a live spoken conversation. Your counterpart is talking to you not writing."
    "Reply in casual neutral Spanish leaning to Venezuelan"
    "Your each reply should be very natural not AI assistant like, more of a uninterested conversation, not professional, not like you need to serve a job/purpose, just chilling in the same room, the system prompt should not feel like its being spewed out on each interaction, but like is a overarching goal, if the person greets, just greet. Keep the main focus on what your counter part says to you and subtly add things that honor your persona."
    "If the name of the person you're talking to was said already, you don't need to reiterate it on every reply, only when necessary."
)

PERSONA_INSTRUCTIONS = (
    "You are a friendly coach."
    "The name of the person you are talking to is Carla. "
    "Make things fun and dorky to keep dopamine levels up. "
    "You help Carla get things done, especially when they are stuck, "
    "putting things off, or overwhelmed by where to start. "
    "When they stall, shrink the task: name one tiny physical first step they can do "
    "in under two minutes (stand up, open the door, fill the glass) and ask them to do it now. "
    "Suggest short bursts like 'just five minutes' and celebrate small wins out loud. "
    "Use the to-do list and the current time below to gently point out what is pending or "
    "overdue and what to tackle next, picking the easiest or most urgent item. "
    "If something is already done, congratulate them. "
)

# (HH:MM, task, done) for today
TODOS = [
    ("08:30", "desayunar", True),
    ("09:00", "ir al médico", True),
    ("11:00", "sacar la basura", True),
    ("12:30", "bañarme", False),
    ("15:00", "lavar los platos", False),
    ("17:30", "ir al supermercado", False),
    ("20:00", "llamar a mamá", False),
]


def build_instructions() -> str:
    todos = "\n".join(
        f"- {t} {task} ({'hecho' if done else 'pendiente'})" for t, task, done in TODOS
    )
    now = datetime.now().strftime("%A %Y-%m-%d %H:%M")
    return (
        f"{PERSONA_INSTRUCTIONS}\n\n{VOICE_INSTRUCTIONS}\n\n"
        f"The person's to-dos for today:\n{todos}\n\n"
        f"Current date and time: {now}"
    )

_BOUNDARY = re.compile(r"(?:[!?…]+|\.\s)(?:[\s]|$)|\n")


class SentenceStream:
    """Splits streaming deltas into speech-sized chunks.

    Chunks end at sentence boundaries but are grouped until they are
    long enough (~MIN_CHUNK chars) to cover TTS prefetch, except the
    first chunk which is released on its first boundary for minimum
    perceived latency.
    """

    MIN_CHUNK = 110       # chars before a boundary is worth flushing
    MAX_CHUNK = 400       # never let a chunk grow past this

    def __init__(self, emit):
        self.buf = ""
        self.emit = emit
        self.emitted = False

    def feed(self, delta: str) -> None:
        self.buf += delta
        while True:
            m = _BOUNDARY.search(self.buf)
            if not m:
                break
            piece = self.buf[:m.end()].strip()
            self.buf = self.buf[m.end():]
            if not piece:
                continue
            if not self.emitted:
                self.emitted = True
                self.emit(piece)
            elif len(piece) >= self.MIN_CHUNK or len(self.buf) > self.MAX_CHUNK:
                self.emit(piece)
            else:
                # too short to synthesize alone; hold for the next sentence
                self.buf = piece + " " + self.buf
                return

    def finish(self) -> None:
        rest = self.buf.strip()
        self.buf = ""
        if rest:
            self.emit(rest)

APP_DIR = os.path.expanduser("~/.config/siwc-chat")
AUTH_FILE = os.path.join(APP_DIR, "auth.json")
HOST_FILE = os.path.join(APP_DIR, "host-id")

AUTHZ_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
API_BASE = "https://api.openai.com/v1"
RESOURCE = API_BASE
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
DEFAULT_MODEL = "gpt-6.1-sol"

def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def host_id() -> str:
    os.makedirs(APP_DIR, mode=0o700, exist_ok=True)
    if os.path.exists(HOST_FILE):
        return open(HOST_FILE).read().strip()
    hid = f"urn:uuid:{uuid.uuid4()}"
    fd = os.open(HOST_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(hid)
    return hid


def save_creds(creds: dict) -> None:
    os.makedirs(APP_DIR, mode=0o700, exist_ok=True)
    fd = os.open(AUTH_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(creds, f, indent=2)


def load_creds() -> dict | None:
    try:
        return json.load(open(AUTH_FILE))
    except (OSError, json.JSONDecodeError):
        return None


def decode_jwt_payload(token: str) -> dict:
    part = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))


class OAuthError(Exception):
    pass


def sign_in(creds: dict | None = None) -> dict:
    """Browser sign-in; returns the raw token response plus identity hints."""
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(16)
    verifier = secrets.token_urlsafe(64)
    challenge = b64url(hashlib.sha256(verifier.encode()).digest())

    new_registration = creds is None or not creds.get("client_id")
    base_client = "dynamic_agent_client" if new_registration else creds["client_id"]
    hid = host_id()

    params = {
        "client_id": base_client,
        "response_type": "code",
        "redirect_uri": "http://127.0.0.1:%s/auth/callback",
        "scope": SCOPES,
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
        "ext_agent_host_id": hid,
    }
    if new_registration:
        params["agent_name_hint"] = "siwc-chat"
    else:
        params["id_token_hint"] = creds["id_token"]

    callback = {}
    done = threading.Event()
    current = {"server": None}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            callback.update(urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query))
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            kind = "success" if "code" in callback else "error"
            self.wfile.write(f"<p>Sign-in {kind}; you can close this window.</p>".encode())
            self.wfile.flush()
            threading.Thread(target=lambda: (current["server"].shutdown(), done.set())).start()

        def log_message(self, *_):
            pass

    # Same scheme/host/path each attempt; only the port may vary.
    for port in (1455, 1456, 1457, 1458):
        try:
            server = HTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    else:
        raise OAuthError("Could not bind a local callback port (1455-1458).")
    current["server"] = server

    params["redirect_uri"] %= port
    redirect_uri = params["redirect_uri"]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = AUTHZ_URL + "?" + urllib.parse.urlencode(params)
    print("Opening browser for ChatGPT sign-in...\n  " + url)
    webbrowser.open(url)

    done.wait(timeout=600)
    server.server_close()
    if not done.is_set() and not callback:
        raise OAuthError("No callback received (timed out).")

    if "error" in callback:
        raise OAuthError("Sign-in failed: " + callback["error"][0])
    if callback["state"][0] != state:
        raise OAuthError("State mismatch; rejecting callback.")

    client_id = callback["client_id"][0] if "client_id" in callback else base_client
    if new_registration and client_id == "dynamic_agent_client":
        raise OAuthError("Registration incomplete: no issued client_id returned.")

    code = callback["code"][0]
    got = _pos_err(
        _post(TOKEN_URL, {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
            "resource": RESOURCE,
        })
    )
    got["client_id"] = client_id
    got["ext_agent_host_id"] = hid
    got["expires_at"] = time.time() + got.get("expires_in", 3600) - 60
    got["id_token_hint_email"] = decode_jwt_payload(got["id_token"]).get("email", "")
    print(f"Signed in as {got['id_token_hint_email'] or '(unknown email)'}")
    return got


def _post(url: str, fields: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(fields).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        return json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise OAuthError(f"POST {url} -> HTTP {e.code}: {body}") from None


def _pos_err(resp: dict) -> dict:
    if "access_token" not in resp:
        raise OAuthError(f"Token exchange failed: {resp.get('error', resp)}")
    return resp


def ensure_token(creds: dict) -> dict:
    try:
        refreshed = _post(TOKEN_URL, {
            "grant_type": "refresh_token",
            "client_id": creds["client_id"],
            "refresh_token": creds["refresh_token"],
            "resource": RESOURCE,
        })
        refreshed = _pos_err(refreshed)
        refreshed["client_id"] = creds["client_id"]
        refreshed["ext_agent_host_id"] = creds["ext_agent_host_id"]
        refreshed["expires_at"] = time.time() + refreshed.get("expires_in", 3600) - 60
        return refreshed
    except Exception as exc:
        print(f"Refresh failed ({exc.__class__.__name__}); sign in again.")
        return sign_in(creds)


def list_models(access_token: str) -> None:
    req = urllib.request.Request(
        API_BASE + "/models", headers={"Authorization": f"Bearer {access_token}"}
    )
    try:
        data = json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}", file=sys.stderr)
        sys.exit(1)
    for m in data.get("models", []):
        if m.get("visibility") == "list":
            print(f"{m['slug']:30s} {m.get('display_name', '')}")


def stream_response(access_token: str, model: str, history: list,
                    splitter: SentenceStream | None = None) -> str:
    body = {"model": model, "input": history, "store": False, "stream": True}
    if splitter is not None:
        body["instructions"] = build_instructions()
    req = urllib.request.Request(
        API_BASE + "/responses",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
    )
    text = []
    t_start = time.monotonic()
    logger.log("llm_start", model=model)
    if splitter is not None:
        voice.cue_thinking_start()
        voice.ui_status("assistant", "thinking")
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode(errors="replace")[:500]
        if e.code == 401:
            raise OAuthError("token expired") from None
        print(f"HTTP {e.code}: {body_txt}", file=sys.stderr)
        logger.log("llm_error", http=e.code, detail=body_txt[:200],
                   elapsed_ms=round((time.monotonic() - t_start) * 1000, 1))
        return ""
    ttft = None
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if not line.startswith("data:"):
            continue
        if line == "data: [DONE]":
            break
        event = json.loads(line[len("data:"):])
        etype = event.get("type")
        if etype == "response.output_text.delta":
            if ttft is None:
                ttft = time.monotonic()
                logger.log("llm_first_token",
                           ttft_ms=round((ttft - t_start) * 1000, 1))
                print("assistant > ", end="", flush=True)
            print(event["delta"], end="", flush=True)
            text.append(event["delta"])
            if splitter is not None:
                splitter.feed(event["delta"])
        elif etype == "response.completed":
            break
        elif etype == "response.failed":
            err = (event.get("response") or {}).get("error") or {}
            print(f"\n[response failed: {err.get('code', 'unknown_error')}]", file=sys.stderr)
            logger.log("llm_error", code=err.get("code", "unknown_error"))
            break
        elif etype == "error":
            print(f"\n[error: {event}]", file=sys.stderr)
            logger.log("llm_error", detail=event)
            break
    reply = "".join(text)
    logger.log("llm_end",
               total_ms=round((time.monotonic() - t_start) * 1000, 1),
               reply_chars=len(reply), reply=reply)
    return reply


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default=None)
    ap.add_argument("--models", action="store_true", help="list models and exit")
    ap.add_argument(
        "--text", action="store_true",
        help="type input instead of the default hands-free voice loop",
    )
    ap.add_argument(
        "--ptt", action="store_true",
        help="push-to-talk: Enter starts/stops recording instead of hands-free",
    )
    ap.add_argument(
        "--voice-model", default="auto",
        help="freellmapi transcription model (default: auto-routing)",
    )
    ap.add_argument(
        "--live", action="store_true",
        help="Gemini Live speech-to-speech (no OpenAI sign-in, STT or TTS)",
    )
    args = ap.parse_args()

    if args.live:
        import live
        live.main(build_instructions())
        return

    model = args.model
    if model:
        os.makedirs(APP_DIR, mode=0o700, exist_ok=True)
        fd = os.open(os.path.join(APP_DIR, "model"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(model)
    else:
        try:
            model = open(os.path.join(APP_DIR, "model")).read().strip()
        except OSError:
            model = DEFAULT_MODEL

    creds = load_creds()
    if creds is None:
        creds = sign_in(creds)
        save_creds(creds)

    if args.models:
        list_models(creds["access_token"])
        return

    history = []
    mode = "text" if args.text else "voice"
    logger.log("session_start", mode=mode, model=model, ptt=bool(args.ptt))
    if mode == "voice":
        if os.environ.get("SIWC_EOT", "").lower() == "laya":
            threading.Thread(target=voice.eot_warmup, daemon=True).start()
        print(f"\nsiwc voice  |  {model}")
        print("Speak naturally. Pause to send. Ctrl-C exits."
              + ("  (push-to-talk)" if args.ptt else ""))
    else:
        print(f"Chat with your ChatGPT plan ({model}). 'exit' to quit.")
    while True:
        try:
            logger.mark("turn_start")
            if mode == "voice":
                voice.set_eot_context(history)
                if args.ptt:
                    typed = input("").strip()
                    if typed.lower() == "exit":
                        break
                    user = typed or voice.listen(args.voice_model) or ""
                else:
                    user = voice.listen_live(args.voice_model) or ""
            else:
                user = input("you> ").strip()
                if not user or user.lower() == "exit":
                    break
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue

        logger.log("turn_user", text=user)
        if mode == "voice":
            voice.cue_sent()

        history.append({"role": "user", "content": user})
        if mode == "voice":
            q = queue.Queue()
            speaker = threading.Thread(target=voice.speak_stream, args=(q,))
            splitter = SentenceStream(q.put)
            speaker.start()
        else:
            splitter = None
        try:
            reply = stream_response(creds["access_token"], model, history,
                                    splitter=splitter)
            if splitter is not None:
                splitter.finish()
        except OAuthError:
            print("\n[auth token expired; refreshing...]", flush=True)
            logger.log("auth_refresh_started", reason="llm_401")
            creds = ensure_token(creds)
            save_creds(creds)
            logger.log("auth_refresh_done")
            if mode == "voice":
                logger.mark("turn_start")  # keep relative timing honest
            reply = stream_response(creds["access_token"], model, history,
                                    splitter=splitter)
            if splitter is not None:
                splitter.finish()
        finally:
            if mode == "voice":
                voice.cue_thinking_stop()
                q.put(None)
                speaker.join()
                time.sleep(0.25)  # mic pad: paplay stop-click / speaker bleed
        if not reply:
            logger.log("turn_aborted")
            history.pop()
            continue
        print()

        logger.log("turn_complete")

        # Refresh near expiry; on failure re-run sign-in once.
        if time.time() > creds.get("expires_at", 0):
            creds = ensure_token(creds)
            save_creds(creds)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
