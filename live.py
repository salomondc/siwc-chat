#!/usr/bin/env python3
"""Gemini Live voice mode for siwc-chat.

One websocket does mic -> model -> speaker: Gemini handles VAD, turn
taking, barge-in, and speech synthesis natively, replacing the
STT -> OpenAI text -> TTS chain in voice.py. The OpenAI/OAuth path in
chat.py is untouched and stays available for orchestration.

Audio I/O uses sounddevice (PortAudio), so it works on Linux, macOS
and Windows.

Config:
  GEMINI_API_KEY in the repo's .env (see .env.example), the environment,
                 or ~/.config/siwc-chat/gemini-key
  GEMINI_LIVE_MODEL   (default below; set to whatever Live model you have)
  GEMINI_LIVE_VOICE   (default Zephyr)
  SIWC_LIVE_IN_DEVICE / SIWC_LIVE_OUT_DEVICE  device index or name substring
                          (list with: python -m sounddevice)
  SIWC_LIVE_DEBUG=1       print mic peak level and when Gemini audio arrives
  SIWC_LIVE_HEADPHONES=1  keep the mic open while the model talks
                          (true barge-in). Without it the mic is muted
                          during playback so speaker bleed can't
                          interrupt the model.
"""
import asyncio
import os
import sys
import threading
import time

import logger

APP_DIR = os.path.expanduser("~/.config/siwc-chat")
KEY_FILE = os.path.join(APP_DIR, "gemini-key")

SEND_RATE = 16000
RECV_RATE = 24000

DEFAULT_MODEL = "models/gemini-3.8-live"  # from the Gemini quickstart; override via env
DEFAULT_VOICE = "Zephyr"


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from the repo's .env (real env vars win)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip("\"'")
        if v:
            os.environ.setdefault(k.strip(), v)


def _api_key() -> str:
    _load_dotenv()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        try:
            key = open(KEY_FILE).read().strip()
        except OSError:
            pass
    if not key:
        raise RuntimeError("No Gemini key; set GEMINI_API_KEY or write it to " + KEY_FILE)
    return key


class Player:
    """Streaming PCM playback via sounddevice; flush() drops queued audio."""

    TAIL_S = 0.3  # keep "playing" briefly after the buffer drains (speaker tail)

    def __init__(self, device=None):
        import sounddevice as sd
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._last_audio = 0.0
        self.stream = sd.RawOutputStream(
            samplerate=RECV_RATE, channels=1, dtype="int16",
            blocksize=0, device=device, callback=self._callback)
        self.stream.start()

    def beep(self) -> None:
        """Short tone so you can confirm the output device works."""
        import math
        import struct
        n = RECV_RATE // 5
        self.write(b"".join(
            struct.pack("<h", int(6000 * math.sin(2 * math.pi * 660 * i / RECV_RATE)))
            for i in range(n)))

    def _callback(self, outdata, frames, _time, status):
        if status:
            print(f"[audio out: {status}]", file=sys.stderr)
        n = frames * 2
        with self._lock:
            chunk = bytes(self._buf[:n])
            del self._buf[:n]
        if chunk:
            self._last_audio = time.monotonic()
        outdata[:len(chunk)] = chunk
        outdata[len(chunk):] = b"\x00" * (n - len(chunk))

    @property
    def playing(self) -> bool:
        with self._lock:
            if self._buf:
                return True
        return time.monotonic() - self._last_audio < self.TAIL_S

    def write(self, pcm: bytes) -> None:
        with self._lock:
            self._buf += pcm

    def flush(self) -> None:
        """Barge-in: drop whatever is still buffered."""
        with self._lock:
            self._buf.clear()
        self._last_audio = 0.0

    def finish(self) -> None:
        """Turn done: audio already queued keeps playing; nothing to do."""

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


async def run(system_instruction: str) -> None:
    from google import genai
    from google.genai import types

    _load_dotenv()
    model = os.environ.get("GEMINI_LIVE_MODEL", DEFAULT_MODEL)
    voice = os.environ.get("GEMINI_LIVE_VOICE", DEFAULT_VOICE)
    headphones = os.environ.get("SIWC_LIVE_HEADPHONES") == "1"

    client = genai.Client(http_options={"api_version": "v1beta"}, api_key=_api_key())
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_instruction,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=104857,
            sliding_window=types.SlidingWindow(target_tokens=52428)),
    )

    import sounddevice as sd
    debug = os.environ.get("SIWC_LIVE_DEBUG") == "1"

    def _dev(name):
        v = os.environ.get(name, "").strip()
        if not v:
            return None
        return int(v) if v.isdigit() else v

    in_dev, out_dev = _dev("SIWC_LIVE_IN_DEVICE"), _dev("SIWC_LIVE_OUT_DEVICE")
    try:
        din = sd.query_devices(in_dev, "input")["name"]
        dout = sd.query_devices(out_dev, "output")["name"]
    except Exception as e:
        raise RuntimeError(f"Audio device problem: {e}\n{sd.query_devices()}")
    print(f"mic: {din}\nspeaker: {dout}")
    player = Player(device=out_dev)
    player.beep()
    loop = asyncio.get_running_loop()
    mic_q: asyncio.Queue = asyncio.Queue(maxsize=100)

    peak = {"v": 0, "t": time.monotonic()}

    def _mic_cb(indata, _frames, _time, status):
        if status:
            print(f"[audio in: {status}]", file=sys.stderr)
        if debug:
            import array
            a = array.array("h", bytes(indata))
            peak["v"] = max(peak["v"], max(map(abs, a), default=0))
            if time.monotonic() - peak["t"] > 1:
                print(f"[mic peak {peak['v']}/32767]", file=sys.stderr)
                peak["v"], peak["t"] = 0, time.monotonic()
        try:
            loop.call_soon_threadsafe(mic_q.put_nowait, bytes(indata))
        except (asyncio.QueueFull, RuntimeError):
            pass

    mic = sd.RawInputStream(
        samplerate=SEND_RATE, channels=1, dtype="int16",
        blocksize=SEND_RATE * 40 // 1000, device=in_dev, callback=_mic_cb)
    mic.start()

    logger.log("session_start", mode="live", model=model, voice=voice,
               headphones=headphones)
    print(f"\nsiwc live  |  {model}  |  {voice}")
    print("Just talk. Ctrl-C exits." + ("" if headphones else
          "  (mic muted while it speaks; SIWC_LIVE_HEADPHONES=1 for barge-in)"))

    try:
        async with client.aio.live.connect(model=model, config=config) as session:

            async def send_mic():
                while True:
                    data = await mic_q.get()
                    if player.playing and not headphones:
                        continue  # half-duplex: don't hear ourselves
                    await session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={SEND_RATE}"))

            async def receive():
                user_buf, model_buf = [], []
                # session.receive() ends at each turn_complete; loop for the session.
                while True:
                    async for msg in session.receive():
                        sc = msg.server_content
                        if msg.data:
                            if debug and not player.playing:
                                print("[gemini audio received]", file=sys.stderr)
                            player.write(msg.data)
                        if sc is None:
                            continue
                        if sc.interrupted:
                            player.flush()
                            logger.log("live_interrupted")
                        if sc.input_transcription and sc.input_transcription.text:
                            user_buf.append(sc.input_transcription.text)
                        if sc.output_transcription and sc.output_transcription.text:
                            model_buf.append(sc.output_transcription.text)
                        if sc.turn_complete:
                            player.finish()
                            u, m = "".join(user_buf).strip(), "".join(model_buf).strip()
                            user_buf.clear()
                            model_buf.clear()
                            if u:
                                print(f"you> {u}")
                                logger.log("turn_user", text=u, via="live")
                            if m:
                                print(f"bot> {m}\n")
                                logger.log("llm_end", reply=m, via="live")
                            logger.log("turn_complete", via="live")

            await asyncio.gather(send_mic(), receive())
    finally:
        mic.stop()
        mic.close()
        player.close()


def main(system_instruction: str) -> None:
    try:
        asyncio.run(run(system_instruction))
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        sys.exit(str(e))
