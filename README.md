# siwc-chat

Terminal voice chat with two backends:

- **Gemini Live** (`--live`): speech-to-speech over one websocket. Gemini handles
  turn detection, interruptions and the voice. No STT/TTS pipeline, no OpenAI sign-in.
- **OpenAI via ChatGPT plan** (default): Sign in with ChatGPT (OAuth + Responses API),
  with separate STT and TTS. Kept for future multi-agent orchestration.

## Setup

Requires Python 3.10+ and PulseAudio/PipeWire CLI tools (`parecord`, `paplay`).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env      # then set GEMINI_API_KEY
```

## Run

```bash
.venv/bin/python chat.py --live     # Gemini Live voice
.venv/bin/python chat.py            # OpenAI voice (hands-free)
.venv/bin/python chat.py --text     # OpenAI, typed
.venv/bin/python chat.py --ptt      # OpenAI, push-to-talk
.venv/bin/python chat.py --models   # list models on your ChatGPT account
```

Ctrl-C exits.

## Gemini Live config (`.env`)

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Required. Falls back to `~/.config/siwc-chat/gemini-key`. |
| `GEMINI_LIVE_MODEL` | Live model id (default `models/gemini-3.8-live`). Change it if the connection fails. |
| `GEMINI_LIVE_VOICE` | Prebuilt voice (default `Zephyr`). |
| `SIWC_LIVE_HEADPHONES=1` | Keep the mic open while the model speaks, so you can interrupt. Without it the mic is muted during playback to avoid speaker echo. |

Real environment variables override `.env`.

## OpenAI mode extras

- First run opens your browser to authorize; credentials are stored in
  `~/.config/siwc-chat/` (0600).
- Needs a local freellmapi router (`FREELLMAPI_BASE`, `FREELLMAPI_KEY` or
  `~/.config/siwc-chat/freellmapi-key`) for transcription, and optionally a
  fish.audio key in `~/.config/siwc-chat/fishaudio-key` for TTS.
- Optional Laya end-of-turn detection: `.venv/bin/pip install -r requirements-eot.txt`,
  then run with `SIWC_EOT=laya`.

## Logs

Session events are written as JSONL to `~/.config/siwc-chat/logs/`.
