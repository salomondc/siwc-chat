#!/usr/bin/env python3
"""Gemini Live voice mode for siwc-chat.

One websocket does mic -> model -> speaker: Gemini handles VAD, turn
taking, barge-in, and speech synthesis natively, replacing the
STT -> OpenAI text -> TTS chain in voice.py. The OpenAI/OAuth path in
chat.py is untouched and stays available for orchestration.

Audio I/O reuses the PulseAudio/PipeWire CLI tools the rest of the
project already uses (parecord / paplay), so no PortAudio build needed.

Config:
  GEMINI_API_KEY or ~/.config/siwc-chat/gemini-key
  GEMINI_LIVE_MODEL   (default below; set to whatever Live model you have)
  GEMINI_LIVE_VOICE   (default Zephyr)
  SIWC_LIVE_HEADPHONES=1  keep the mic open while the model talks
                          (true barge-in). Without it the mic is muted
                          during playback so speaker bleed can't
                          interrupt the model.
"""
import asyncio
import os
import shutil
import subprocess
import sys

import logger

APP_DIR = os.path.expanduser("~/.config/siwc-chat")
KEY_FILE = os.path.join(APP_DIR, "gemini-key")

SEND_RATE = 16000
RECV_RATE = 24000
CHUNK_BYTES = SEND_RATE * 2 * 40 // 1000  # 40 ms of s16le mono

DEFAULT_MODEL = "models/gemini-3.8-live"  # from the Gemini quickstart; override via env
DEFAULT_VOICE = "Zephyr"


def _api_key() -> str:
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
    """Streaming raw-PCM playback via paplay; flush() kills queued audio."""

    def __init__(self):
        self.paplay = shutil.which("paplay")
        if not self.paplay:
            raise RuntimeError("paplay not found")
        self.proc = None
        self.playing = False

    def _spawn(self):
        self.proc = subprocess.Popen(
            [self.paplay, "--raw", "--format=s16le", f"--rate={RECV_RATE}",
             "--channels=1", "--latency-msec=60"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write(self, pcm: bytes) -> None:
        if self.proc is None or self.proc.poll() is not None:
            self._spawn()
        try:
            self.proc.stdin.write(pcm)
            self.proc.stdin.flush()
            self.playing = True
        except (BrokenPipeError, OSError):
            self.proc = None

    def flush(self) -> None:
        """Barge-in: drop whatever is still buffered."""
        if self.proc is not None:
            self.proc.kill()
            self.proc = None
        self.playing = False

    def finish(self) -> None:
        """Turn done: let queued audio drain, then close."""
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            proc, self.proc = self.proc, None
            proc.wait()
        self.playing = False

    close = flush


async def run(system_instruction: str) -> None:
    from google import genai
    from google.genai import types

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

    parecord = shutil.which("parecord")
    if not parecord:
        raise RuntimeError("parecord not found")
    player = Player()
    mic = subprocess.Popen(
        [parecord, "--raw", "--format=s16le", f"--rate={SEND_RATE}",
         "--channels=1", "--latency-msec=40"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    logger.log("session_start", mode="live", model=model, voice=voice,
               headphones=headphones)
    print(f"\nsiwc live  |  {model}  |  {voice}")
    print("Just talk. Ctrl-C exits." + ("" if headphones else
          "  (mic muted while it speaks; SIWC_LIVE_HEADPHONES=1 for barge-in)"))

    try:
        async with client.aio.live.connect(model=model, config=config) as session:

            async def send_mic():
                while True:
                    data = await asyncio.to_thread(mic.stdout.read, CHUNK_BYTES)
                    if not data:
                        return
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
                            await asyncio.to_thread(player.write, msg.data)
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
                            await asyncio.to_thread(player.finish)
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
        mic.kill()
        player.close()


def main(system_instruction: str) -> None:
    try:
        asyncio.run(run(system_instruction))
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        sys.exit(str(e))
