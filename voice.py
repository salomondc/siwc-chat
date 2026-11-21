#!/usr/bin/env python3
"""Voice input for siwc-chat: record the mic, transcribe via freellmapi.

Recording uses parecord (pipewire/pulseaudio) with Enter as a
push-to-talk key. Transcription calls the OpenAI-compatible
/v1/audio/transcriptions endpoint on the local freellmapi router,
which routes to a free voice model (Groq Whisper, Mistral Voxtral,
Cloudflare, ...).
"""
import contextlib
import json
import math
import collections
import os
import queue
import re
import array
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
import urllib.error
import urllib.request

import logger

# Hugging Face's multi-line progress bars corrupt an interactive voice prompt.
# We show a single friendly Laya startup status instead.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

APP_DIR = os.path.expanduser("~/.config/siwc-chat")
KEY_FILE = os.path.join(APP_DIR, "freellmapi-key")
FISH_KEY_FILE = os.path.join(APP_DIR, "fishaudio-key")
DEFAULT_BASE = "http://127.0.0.1:3002/v1"

# fish.audio tts (preferred, faster voice); reference_id is the voice model.
FISH_BASE = "https://api.fish.audio"
FISH_REFERENCE_ID = "1505e291ec504760a285fd163a78b5eb"
FISH_MODEL = "speech-1.5"  # cheapest/fastest fish tts model

SAMPLE_RATE = 16000
CHANNELS = 1

# Interaction sound cues (project root)
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
SENT_CUE = os.path.join(PROJ_DIR, "sent.mp3")
THINKING_CUE = os.path.join(PROJ_DIR, "thinking.mp3")

_thinking: dict = {"proc": None, "stop": None}
_ui_lock = threading.Lock()


def ui_status(label: str, detail: str = "") -> None:
    """Print one stable, human-facing voice status line."""
    with _ui_lock:
        suffix = f" {detail}" if detail else ""
        print(f"  {label} >{suffix}", flush=True)


def api_config() -> tuple[str, str]:
    base = os.environ.get("FREELLMAPI_BASE", DEFAULT_BASE).rstrip("/")
    key = os.environ.get("FREELLMAPI_KEY")
    if not key:
        try:
            key = open(KEY_FILE).read().strip()
        except OSError:
            key = ""
    if not key:
        raise RuntimeError(
            "No freellmapi key; set FREELLMAPI_KEY or write it to " + KEY_FILE
        )
    return base, key


def record() -> str | None:
    """Push-to-talk: first Enter starts recording, second Enter stops.

    Returns the path to a recorded wav file, or None on cancel/error.
    """
    parecord = shutil.which("parecord")
    if not parecord:
        raise RuntimeError("parecord not found (install pipewire + pulseaudio-utils)")

    path = os.path.join(tempfile.gettempdir(), f"siwc-voice-{os.getpid()}.wav")
    print("[recording] ... press Enter to stop", flush=True)
    proc = subprocess.Popen(
        [
            parecord,
            f"--rate={SAMPLE_RATE}",
            f"--channels={CHANNELS}",
            "--file-format=wav",
            "--format=s16le",
            "--record",
            path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        input()
    finally:
        proc.terminate()
        proc.wait()

    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    if os.path.exists(path):
        os.unlink(path)
    print("[empty recording]", flush=True)
    return None


def _rms(chunk: bytes) -> float:
    n = len(chunk) // 2
    if not n:
        return 0.0
    samples = array.array("h")
    samples.frombytes(chunk)
    if sys.byteorder != "little":  # parecord gives little-endian s16le
        samples.byteswap()
    return math.sqrt(sum(s * s for s in samples) / n)


# Ambient noise-floor cache, reused across turns. Calibration runs
# synchronously once at session start ("[calibrating mic...]"); after
# that recalibration happens in a background thread kicked off after
# each reply, sampling its own short stream and committing only if no
# speech was detected meanwhile. Cache drift triggers: false triggers,
# failed/empty STT, blips, many turns since last calibration.
_noise_state: dict = {"floor": None, "ts": 0.0, "turns": 0, "sampling": False}
MAX_FLOOR_AGE = 300.0      # seconds
RECAL_AFTER_N_TURNS = 15
CALIB_CHUNKS = 10          # ~1 s of ambient at 100 ms chunks
_BLIP_MS = 400.0           # shorter "speech" like this is a false trigger
_speech_now = False        # whether a stream currently hears speech


def _request_recal(reason: str) -> None:
    if _noise_state["sampling"]:
        return
    logger.log("recal_scheduled", reason=reason)
    _noise_state["sampling"] = True

    def worker():
        try:
            parecord = shutil.which("parecord")
            if not parecord:
                return
            chunk_bytes = SAMPLE_RATE * CHANNELS * 2 * 100 // 1000
            proc = subprocess.Popen(
                [parecord, f"--rate={SAMPLE_RATE}", f"--channels={CHANNELS}",
                 "--format=s16le", "--file-format=raw", "--record"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            t0 = time.monotonic()
            total = 0.0
            for _ in range(CALIB_CHUNKS):
                chunk = proc.stdout.read(chunk_bytes)
                if len(chunk) < chunk_bytes:
                    break
                total += _rms(chunk)
                if _speech_now:
                    logger.log("recal_aborted", reason="speech_during_sample")
                    proc.terminate()
                    return
            proc.terminate()
            if _speech_now:
                logger.log("recal_aborted", reason="speech_during_sample")
                return
            floor = total / CALIB_CHUNKS
            _noise_state["floor"] = floor
            _noise_state["ts"] = time.monotonic()
            _noise_state["reason"] = reason
            logger.log("recal_done", reason=reason, sampled_ms=round(
                (time.monotonic() - t0) * 1000, 1),
                noise_floor=round(floor, 1))
        except Exception as e:
            logger.log("recal_error", error=str(e))
        finally:
            _noise_state["sampling"] = False

    threading.Thread(target=worker, daemon=True).start()


def _noise_state_after_turn(ok: bool, speech_ms: float | None = None) -> None:
    s = _noise_state
    s["turns"] += 1
    if s["floor"] is None:
        return
    if not ok:
        _request_recal("transcription_empty_or_failed")
    elif speech_ms is not None and speech_ms < _BLIP_MS:
        _request_recal("blip_false_trigger")
    elif s["turns"] >= RECAL_AFTER_N_TURNS:
        s["turns"] = 0
        _request_recal("periodic_refresh")


def _write_wav(path: str, data: bytes | bytearray) -> None:
    with contextlib.closing(wave.open(path, "wb")) as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(data))


SPEC_MIN_MS = 2500          # first speculative STT pass after this much speech
SPEC_GAP_MS = 2000          # speak 2s more -> one more speculative pass
SPEC_TAIL_MS = 400          # trust the speculative pass if the utterance
                            # ended within this much extra speech
SPEC_MAX_CALLS = 2
_spec_state: dict = {}

# ---- End-of-turn decision model (Laya, feature-flagged) ----------------
# SIWC_EOT=laya enables background decision scoring. VAD and full STT remain
# authoritative until a fine-tuned Laya checkpoint is available.
_TRAILING_WORDS = {"y", "pero", "porque", "que", "sobre", "como", "con",
                   "para", "de", "o", "en", "al", "la", "el", "los", "las",
                   "a", "se", "ella", "él", "este", "eh", "em", "mm", "and",
                   "but", "because", "so", "or", "of", "to", "on", "in",
                   "the", "a", "an"}
_ANNOTATION = re.compile(r"\[[^\]]*\]")
_eot: dict = {"router": None, "failed": False}  # guarded by _eot_lock
_eot_lock = threading.Lock()
_eot_context: list[dict] = []


def _strip_annotations(text: str) -> str:
    """Drop ASR bracket annotations ([risa], [golpe]) from the spoken words."""
    return re.sub(r"\s+", " ", _ANNOTATION.sub("", text)).strip()


def _unfinished(spoken: str) -> bool:
    """Rule for a trailing-off turn: '...' or a dangling connector/filler.
    Base Laya's `done` answer measured at chance on logged turns, so this
    drives p_done (Laya's raw value is still logged as p_done_laya)."""
    if re.search(r"(\.\.\.|…)\s*$", spoken):
        return True
    if re.search(r"[.?!]\s*$", spoken):
        return False
    words = re.findall(r"[\wáéíóúñü]+", spoken.lower())
    return bool(words) and words[-1] in _TRAILING_WORDS


def set_eot_context(history: list[dict]) -> None:
    """Set the three latest chat exchanges for the next turn decision."""
    global _eot_context
    context = []
    for message in history[-6:]:
        content = str(message.get("content", "")).strip()
        if content:
            # Leave room for the current transcript in Laya's context window.
            context.append({"role": message.get("role", "unknown"),
                            "content": content[-700:]})
    with _eot_lock:
        _eot_context = context


def eot_warmup() -> None:
    """Load Router + one warmup forward so the first real decision is fast."""
    if os.environ.get("SIWC_EOT", "").lower() != "laya":
        return
    _maybe_reexec_venv()
    t0 = time.monotonic()
    try:
        with _eot_lock:
            if _eot["router"] is None:
                from laya import Router
                _eot["router"] = Router()
        router = _eot["router"]
        router.predict("Sí.", {"w": {"type": "noul",
            "instructions": "Has the speaker finished their thought?"}})
    except Exception as e:
        with _eot_lock:
            _eot["failed"] = True
        logger.log("eot_error", stage="warmup", error=str(e)[:200])
        return
    load_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.log("eot_ready", load_ms=load_ms)


def _maybe_reexec_venv() -> None:
    """If laya isn't importable but the project venv has it, re-exec under
    the venv interpreter (once; guarded by SIWC_REEXEC to avoid loops)."""
    if os.environ.get("SIWC_REEXEC") == "1":
        return
    try:
        import laya  # noqa: F401
        return
    except Exception:
        pass
    alt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       ".venv", "bin", "python")
    if os.path.exists(alt):
        env = dict(os.environ, SIWC_REEXEC="1")
        os.execve(alt, [alt] + sys.argv, env)


def eot_decide(text: str, final: bool) -> dict | None:
    """Laya verdict using recent dialogue plus this turn's transcript."""
    if not text.strip():
        return None
    spoken = _strip_annotations(text)
    if not spoken:
        # Annotations only ([click], [golpe]): nothing was actually said.
        return {"p_done": 1.0, "p_gibberish": 1.0, "p_done_laya": None,
                "route": "rule", "ms": 0.0, "context_messages": 0,
                "context": [], "dangling_word": False}
    with _eot_lock:
        router, failed = _eot["router"], _eot["failed"]
    if failed:
        return None
    try:
        if router is None:
            eot_warmup()
            with _eot_lock:
                router = _eot["router"]
            if router is None:
                return None
        with _eot_lock:
            context = list(_eot_context)
        state = {
            "recent_conversation": context,
            "current_user_transcript": spoken,
            "transcript_status": "complete final transcription" if final
            else "provisional transcription while the user may still speak",
        }
        questions = {
            "done": {"type": "noul", "instructions":
                "Decide whether the user has finished their CURRENT spoken "
                "turn and the assistant should answer now. Use the recent "
                "conversation to interpret the current transcript: a short "
                "answer such as 'sí', 'no', or 'toboganes' can be complete "
                "when it answers the assistant's last question. Return false "
                "only when the current transcript is clearly unfinished, such "
                "as a trailing connector, an incomplete clause, or an obvious "
                "self-interruption. Bracketed descriptions such as [risa] and "
                "[ruido de pasos] are context, not proof that speech is over."},
            "gibberish": {"type": "noul", "instructions":
                "Is the CURRENT user transcript mostly unintelligible noise "
                "or ASR corruption? Descriptive bracketed annotations such as "
                "[risa] and [ruido de pasos] are valid conversation context, "
                "not gibberish."},
        }
        t0 = time.monotonic()
        res = router.predict(state, questions)
        ms = round((time.monotonic() - t0) * 1000, 1)
        unfinished = _unfinished(spoken)
        return {"p_done": 0.0 if unfinished else 1.0,
                "p_done_laya": res["answers"]["done"]["noul"],
                "p_gibberish": res["answers"]["gibberish"]["noul"],
                "route": res["routing"]["model"],
                "ms": ms, "context_messages": len(context),
                "context": context, "dangling_word": unfinished}
    except Exception as e:
        logger.log("eot_error", stage="decide", error=str(e)[:200])
        return None


def _spec_pass(audio: bytes, spec_ms: int, box: dict) -> None:
    """Background transcription of speech-so-far (results may be used when
    the utterance ends right after the snapshot; final pass wins otherwise)."""
    path = None
    try:
        path = os.path.join(tempfile.gettempdir(),
                            f"siwc-spec-{os.getpid()}.wav")
        _write_wav(path, audio)
        box["text"] = transcribe(path)
        if box["text"]:
            ui_status("heard", box["text"])
    except Exception as e:
        logger.log("stt_error", provider="speculative", error=str(e))
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def listen_once(max_sec: float = 25.0) -> str | None:
    """Listen continuously; return the path to one utterance as a wav file.

    Runs on a live parecord stream, uses RMS level to detect speech and
    ends the utterance after a sustained pause. Returns None when only
    noise/level below threshold was heard.
    """
    parecord = shutil.which("parecord")
    if not parecord:
        raise RuntimeError("parecord not found (install pipewire + pulseaudio-utils)")

    chunk_ms = 100
    bytes_per_chunk = SAMPLE_RATE * CHANNELS * 2 * chunk_ms // 1000
    bytes_per_chunk = (bytes_per_chunk // 2) * 2
    silence_limit = 9          # ~0.9 s of quiet ends the utterance
                               # (thinking grace; no sent-cue yet = still listening)
    speech_limit = 2           # ~0.2 s above threshold starts it
    max_chunks = int(max_sec * 1000 // chunk_ms)

    proc = subprocess.Popen(
        [
            parecord,
            f"--rate={SAMPLE_RATE}",
            f"--channels={CHANNELS}",
            "--format=s16le",
            "--file-format=raw",
            "--record",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    path = os.path.join(tempfile.gettempdir(), f"siwc-live-{os.getpid()}.wav")

    def finish(utter: bytearray | None) -> str | None:
        proc.terminate()
        try:
            proc.wait()
        except Exception:
            pass
        if not utter:
            return None
        with contextlib.closing(wave.open(path, "wb")) as w:
            w.setnchannels(CHANNELS)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(bytes(utter))
        return path

    try:
        return _listen_on_stream(proc, path, finish, chunk_ms, bytes_per_chunk,
                                 max_chunks, silence_limit, speech_limit)
    except BaseException:
        logger.log("listen_error", error="interrupted")
        try:
            proc.terminate()
            proc.wait()
        except Exception:
            pass
        raise


def _listen_on_stream(proc, path, finish, chunk_ms, bytes_per_chunk, max_chunks,
                      silence_limit, speech_limit):
    global _speech_now, _last_speech_ms
    s = _noise_state
    fresh = s["floor"] is not None and (time.monotonic() - s["ts"]) < MAX_FLOOR_AGE
    if fresh:
        floor = s["floor"]
        threshold = max(floor * 3.0, 150.0)
        logger.log("calibrated_cached", threshold=round(threshold, 1),
                   cache_age_ms=round((time.monotonic() - s["ts"]) * 1000, 1))
    else:
        ui_status("mic", "calibrating room noise (about 1s)")
        total = 0.0
        for _ in range(CALIB_CHUNKS):
            chunk = proc.stdout.read(bytes_per_chunk)
            if len(chunk) < bytes_per_chunk:
                return finish(None)
            total += _rms(chunk)
        floor = total / CALIB_CHUNKS
        threshold = max(floor * 3.0, 150.0)
        s["floor"] = floor
        s["ts"] = time.monotonic()
        logger.log("calibrated", threshold=round(threshold, 1),
                   noise_floor=round(floor, 1))
    logger.log("listen_start")
    ui_status("mic", "listening - speak when ready")

    started = False
    speech_run = 0
    silence_run = 0
    spoken = bytearray()
    prebuf = bytearray()          # rolling pre-trigger audio (first phonemes)
    PRE_CHUNKS = 4                # ~400 ms kept before speech_start
    spec_calls = 0
    spec_ms = 0
    spec_thread = None
    spec_box = None
    last_loud_ms = 0              # audio-time of the last above-threshold chunk
    eot_judged = False
    flag_eot = os.environ.get("SIWC_EOT", "").lower() == "laya"
    waited = 0          # chunks waited before speech started

    def _end(spoken_final, truncated):
        global _last_speech_ms, _spec_state
        if started:
            _last_speech_ms = last_loud_ms  # audio-based, not wall clock
            logger.log("speech_end", speech_ms=_last_speech_ms,
                       truncated=truncated)
            _speech_now = False
            _spec_state = {"thread": spec_thread, "box": spec_box,
                           "spec_ms": spec_ms, "utterance_ms": _last_speech_ms}
        return finish(spoken_final if started else None)

    while waited < max_chunks:
        chunk = proc.stdout.read(bytes_per_chunk)
        if len(chunk) < bytes_per_chunk:
            break
        waited += 1
        level = _rms(chunk)
        # Once an utterance has started, accept quieter syllables so soft
        # trailing words do not look like a pause and get cut off.
        active_threshold = threshold if not started else max(floor * 1.8, 100.0)
        if level > active_threshold:
            speech_run += 1
            silence_run = 0
            if not started and speech_run >= speech_limit:
                started = True
                spoken = bytearray(prebuf)  # keep the onset of the word
                waited = 0  # restart budget as utterance-length budget
                _speech_now = True
                logger.mark("speech_start")
                logger.log("speech_start", level=round(level, 1))
                ui_status("mic", "hearing you")
            if not started:
                prebuf.extend(chunk)
                if len(prebuf) > PRE_CHUNKS * bytes_per_chunk:
                    del prebuf[:bytes_per_chunk]
            if started:
                spoken.extend(chunk)
                speech_ms = (len(spoken) // bytes_per_chunk) * chunk_ms
                last_loud_ms = speech_ms
                next_spec = (SPEC_MIN_MS if spec_calls == 0
                             else spec_ms + SPEC_GAP_MS)
                if spec_calls < SPEC_MAX_CALLS and speech_ms >= next_spec:
                    spec_calls += 1
                    spec_ms = speech_ms
                    spec_box = {"text": None}
                    spec_thread = threading.Thread(
                        target=_spec_pass,
                        args=(bytes(spoken), spec_ms, spec_box),
                        daemon=True)
                    spec_thread.start()
                    logger.log("stt_spec_scheduled", call=spec_calls,
                               spec_ms=spec_ms)
                if len(spoken) >= max_chunks * bytes_per_chunk:
                    return _end(spoken, True)
        else:
            if started:
                silence_run += 1
                spoken.extend(chunk)
                if silence_run == 4 and not eot_judged:
                    eot_judged = True
                    spec_text = None
                    if spec_thread is not None and spec_thread.is_alive():
                        spec_thread.join(timeout=0.35)  # ≤35% of base grace
                    if spec_box is not None:
                        spec_text = spec_box.get("text")
                    if flag_eot and spec_text:
                        verdict = eot_decide(spec_text, final=False)
                        if verdict:
                            dangling = verdict["dangling_word"]
                            logger.log("eot_decision",
                                       p_done=round(verdict["p_done"], 3),
                                       p_gibberish=round(verdict["p_gibberish"], 3),
                                       route=verdict["route"],
                                       eot_ms=verdict["ms"],
                                       context_messages=verdict["context_messages"],
                                       transcript_status="provisional",
                                       p_done_laya=verdict["p_done_laya"],
                                       context=verdict["context"],
                                       text=spec_text[:200],
                                       dangling_word=dangling,
                                       shadow=True)
                if silence_run >= silence_limit:
                    return _end(spoken, False)
            else:
                speech_run = 0
    return _end(spoken, True)


def transcribe(wav_path: str, model: str = "auto") -> str | None:
    """Transcribe a wav. Primary: fish.audio transcribe-1-pro (bilingual,
    Spanish-friendly). Fallback: freellmapi /audio/transcriptions (Groq).
    """
    text = _asr_fish(wav_path)
    if text is not None:
        return text
    return _asr_freellm(wav_path, model)


def _clean_speaker_tag(text: str) -> str:
    # transcribe-1-pro emits diarization tags like <|speaker:0|>
    return re.sub(r"<\|[^|]*\|>\s*", "", text)


def _asr_fish(wav_path: str) -> str | None:
    key = _fish_key()
    if not key:
        return None
    with open(wav_path, "rb") as f:
        audio = f.read()
    boundary = "siwc-asr-boundary-4d8f2e7c3a"
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; '
        f'filename="speech.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
        + audio + b"\r\n",
        f'--{boundary}\r\nContent-Disposition: form-data; name="ignore_timestamps"'
        f"\r\n\r\ntrue\r\n".encode(),
        f"--{boundary}--\r\n".encode(),
    ]
    req = urllib.request.Request(
        FISH_BASE + "/v1/asr",
        data=b"".join(parts),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "model": "transcribe-1-pro",
        },
        method="POST",
    )
    logger.log("stt_start", model="transcribe-1-pro", provider="fish",
               audio_bytes=len(audio))
    t0 = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:200]
        logger.log("stt_error", provider="fish", http=e.code, detail=detail)
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.log("stt_error", provider="fish", error=str(e))
        return None
    text = _clean_speaker_tag((data.get("text") or "")).strip()
    logger.log("stt_end", provider="fish", model="transcribe-1-pro",
               elapsed_ms=round((time.monotonic() - t0) * 1000, 1), text=text)
    return text


def _asr_freellm(wav_path: str, model: str = "auto") -> str | None:
    base, key = api_config()
    boundary = "siwc-voice-boundary-7f3a2b9c1d"
    with open(wav_path, "rb") as f:
        audio = f.read()

    parts = []
    for name, value in (("model", model),):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
            f"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="speech.wav"\r\nContent-Type: audio/wav\r\n\r\n'
        ).encode()
        + audio
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        base + "/audio/transcriptions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    logger.log("stt_start", model=model, provider="freellmapi", audio_bytes=len(audio))
    t0 = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        logger.log("stt_error", http=e.code, detail=detail[:200],
                   elapsed_ms=round((time.monotonic() - t0) * 1000, 1))
        print(f"[transcription failed: HTTP {e.code} {detail}]", flush=True)
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        logger.log("stt_error", error=str(e),
                   elapsed_ms=round((time.monotonic() - t0) * 1000, 1))
        print(f"[transcription failed: {e}]", flush=True)
        return None
    text = (data.get("text") or "").strip()
    logger.log("stt_end", elapsed_ms=round((time.monotonic() - t0) * 1000, 1), text=text)
    return text


def listen(model: str = "auto") -> str | None:
    """Record + transcribe one utterance. None means skip this turn."""
    try:
        wav = record()
    except RuntimeError as e:
        print(f"[voice unavailable: {e}]", flush=True)
        return None
    if wav is None:
        return None
    try:
        text = transcribe(wav, model)
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass
    if not text:
        ui_status("mic", "no speech detected")
        return None
    print(f"you > {text}", flush=True)
    return text


_WHISPER_SFX = re.compile(r"(?:\(|\[)[^)\]]*(?:click|risa|suspense|m[uú]sica|ruido|"
                          r"sound|noise|aplausos|silencio|respira)[^)\]]*(?:\)|\])",
                          re.IGNORECASE)
_LATIN = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")


def clean_transcript(text: str | None) -> str | None:
    """Keep descriptive ASR annotations; reject obvious noise hallucinations."""
    if not text:
        return None
    text = text.strip()
    # Use a masked copy for the guard only. Tags such as [risa] or
    # [ruido de pasos] are useful conversational context and stay intact.
    spoken = _WHISPER_SFX.sub("", text).strip()
    if _CJK.search(spoken) or not _LATIN.search(text):
        # CJK/symbol output on an es/en voice channel is a common ASR
        # hallucination; annotations alone remain valid input.
        return None
    return text


def listen_live(model: str = "auto", **kw) -> str | None:
    """Hands-free: one auto-segmented utterance, transcribed.

    With SIWC_EOT=laya, Laya scores endpointing in the background for logs
    and future fine-tuning. Full STT and VAD remain authoritative.
    """
    flag_eot = os.environ.get("SIWC_EOT", "").lower() == "laya"
    try:
        wav = listen_once(**kw)
    except RuntimeError as e:
        print(f"\n[voice unavailable: {e}]", flush=True)
        return None
    if wav is None:
        _noise_state["turns"] += 1
        if _noise_state["turns"] >= RECAL_AFTER_N_TURNS:
            _noise_state["turns"] = 0
            _request_recal("periodic_refresh")
        return None
    _noise_state_after_turn(ok=True, speech_ms=_last_speech_ms)
    try:
        global _spec_state
        spec = _spec_state
        _spec_state = {}
        if spec.get("thread") is not None:
            logger.log("stt_spec_complete", tail_ms=round(
                spec["utterance_ms"] - spec["spec_ms"], 1),
                used_as_final=False)
        # A partial ASR pass can never prove it contains the final words.
        # Keep it for EOT research, but transcribe the complete wav before
        # a user message is ever sent to the chat model.
        ui_status("stt", "transcribing full turn")
        text = clean_transcript(transcribe(wav, model))
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass
    if not text:
        ui_status("mic", "noise ignored")
        return None
    if flag_eot:
        verdict = eot_decide(text, final=True)
        if verdict:
            done = verdict["p_done"]
            gib = verdict["p_gibberish"]
            logger.log("eot_decision_final", p_done=round(done, 3),
                       p_gibberish=round(gib, 3), route=verdict["route"],
                       eot_ms=verdict["ms"],
                       context_messages=verdict["context_messages"],
                       p_done_laya=verdict["p_done_laya"],
                       context=verdict["context"],
                       text=text[:200])
    print(f"you > {text}", flush=True)
    return text


def _fish_key() -> str:
    key = os.environ.get("FISH_API_KEY")
    if not key:
        try:
            key = open(FISH_KEY_FILE).read().strip()
        except OSError:
            key = ""
    return key


def speech_fish(text: str) -> bytes | None:
    """TTS via api.fish.audio with the configured reference voice (wav pcm)."""
    key = _fish_key()
    if not key:
        return None
    body = json.dumps({
        "text": text.strip(),
        "reference_id": FISH_REFERENCE_ID,
        "format": "wav",
        "normalize": True,
    }).encode()
    req = urllib.request.Request(
        FISH_BASE + "/v1/tts",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "model": FISH_MODEL,
        },
        method="POST",
    )
    logger.log("tts_start", provider="fish")
    t0 = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        audio = resp.read()
        logger.log("tts_ready", provider="fish",
                   synth_ms=round((time.monotonic() - t0) * 1000, 1),
                   audio_bytes=len(audio))
        return audio
    except urllib.error.HTTPError as e:
        logger.log("tts_error", provider="fish", http=e.code)
        if e.code == 402:
            print("[fish.audio: API credit empty; falling back]", flush=True)
        return None
    except (urllib.error.URLError, TimeoutError) as e:
        logger.log("tts_error", provider="fish", error=str(e))
        return None


def speech(
    text: str,
    model: str | None = None,
    voice_name: str | None = None,
) -> bytes | None:
    """Ping /v1/audio/speech on freellmapi for TTS audio (wav).

    Defaults to a single pinned model rather than 'auto': the auto route
    burns seconds failing through models without a key and picks a slow
    premium TTS first. Override with FREELLMAPI_TTS_MODEL.
    """
    model = model or os.environ.get(
        "FREELLMAPI_TTS_MODEL", "gemini-3.1-flash-tts-preview"
    )
    base, key = api_config()
    body = {"model": model, "input": text, "response_format": "wav"}
    if voice_name:
        body["voice"] = voice_name
    req = urllib.request.Request(
        base + "/audio/speech",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    logger.log("tts_start", provider="freellmapi", model=model)
    t0 = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        logger.log("tts_error", provider="freellmapi", http=e.code)
        print(f"[tts failed: HTTP {e.code} {e.read().decode(errors='replace')[:200]}]",
              flush=True)
        return None
    except (urllib.error.URLError, TimeoutError) as e:
        logger.log("tts_error", provider="freellmapi", error=str(e))
        print(f"[tts failed: {e}]", flush=True)
        return None
    audio = resp.read()
    logger.log("tts_ready", provider="freellmapi", model=model,
               synth_ms=round((time.monotonic() - t0) * 1000, 1),
               audio_bytes=len(audio))
    return audio


def play_cue(path: str) -> None:
    """Fire-and-forget one-shot sound (sent blip)."""
    paplay = shutil.which("paplay") or shutil.which("pw-play")
    if paplay and os.path.exists(path):
        subprocess.Popen([paplay, path], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,
                         start_new_session=True)


def cue_sent() -> None:
    logger.log("cue_sent")
    play_cue(SENT_CUE)


def cue_thinking_start() -> None:
    """Ambient 'thinking' loop; stop() kills it when the reply starts."""
    paplay = shutil.which("paplay") or shutil.which("pw-play")
    if not paplay or not os.path.exists(THINKING_CUE):
        return
    stop = threading.Event()
    _thinking["stop"] = stop

    def loop():
        while not stop.is_set():
            proc = subprocess.Popen([paplay, THINKING_CUE],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            _thinking["proc"] = proc
            # wake 0.2s after cue length so long files restart only if needed
            if stop.wait(timeout=17.5):
                proc.terminate()
                return
            proc.terminate()
            proc.wait()

    threading.Thread(target=loop, daemon=True).start()
    logger.log("cue_thinking_start")


def cue_thinking_stop() -> None:
    stop = _thinking.get("stop")
    if stop:
        stop.set()
        proc = _thinking.get("proc")
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        _thinking["stop"] = None
        logger.log("cue_thinking_stop")


def _synth_one(text: str, n: int) -> str | None:
    """Synthesize one chunk to a temp wav; returns path or None."""
    audio = speech_fish(text)
    if not audio:
        audio = speech(text, model=os.environ.get(
            "FREELLMAPI_TTS_MODEL", "gemini-3.1-flash-tts-preview"))
    if not audio:
        logger.log("tts_error", provider="all", sentence=n, text=text[:60])
        return None
    path = os.path.join(tempfile.gettempdir(),
                        f"siwc-tts-{os.getpid()}-{n}.wav")
    with open(path, "wb") as f:
        f.write(audio)
    return path


def speak_stream(q) -> None:
    """Consumer: pipelined TTS playback.

    A synth worker pulls queued chunks and synthesizes them ahead of
    playback (prefetch); the player plays them in order. The next chunk
    is therefore usually ready before the current one finishes playing,
    so there is no dead air between sentences. None marks end of turn.
    """
    synth_q: "queue.Queue" = queue.Queue()
    ready = collections.deque()          # (n, path | None) in order
    lock = threading.Lock()
    has_item = threading.Condition(lock)

    def synth_worker():
        while True:
            item = synth_q.get()
            if item is None:
                with has_item:
                    ready.append((None, None))
                    has_item.notify_all()
                return
            n = item[0]
            text = item[1]
            t0 = time.monotonic()
            path = _synth_one(text, n)
            logger.log("tts_ready", synth_ms=round(
                (time.monotonic() - t0) * 1000, 1), sentence=n,
                prefetched=n > 1)
            with has_item:
                ready.append((n, path))
                has_item.notify_all()

    threading.Thread(target=synth_worker, daemon=True).start()

    def dispatcher():  # q -> synth_q so prefetch starts while playing
        n = 0
        while True:
            item = q.get()
            if item is None:
                synth_q.put(None)
                return
            n += 1
            synth_q.put((n, item))

    threading.Thread(target=dispatcher, daemon=True).start()

    paplay = shutil.which("paplay") or shutil.which("pw-play")
    first = True
    n = 0
    while True:
        with has_item:
            while not ready:
                has_item.wait()
            n, path = ready.popleft()
        if n is None:
            return
        if path is None:  # synth failed for this chunk; skip audio
            continue
        t_play = time.monotonic()
        if first:
            cue_thinking_stop()
            logger.log("playback_start")
            first = False
        try:
            if paplay:
                subprocess.run([paplay, path], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=False)
            else:
                print("(no paplay; would speak sentence", n, ")")
            logger.log("playback_end", play_ms=round(
                (time.monotonic() - t_play) * 1000, 1), sentence=n)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


def say(text: str, suppress_open_mic: float = 0.1) -> None:
    """Synthesize + play a reply. Returns after playback ends (plus a small
    pad so the speaker isn't picked up by the reopened mic)."""
    if not text.strip():
        return
    audio = speech_fish(text)
    if not audio:
        audio = speech(text.strip())
    if not audio:
        logger.log("playback_skipped")
        return
    path = os.path.join(tempfile.gettempdir(), f"siwc-tts-{os.getpid()}.wav")
    with open(path, "wb") as f:
        f.write(audio)
    paplay = shutil.which("paplay") or shutil.which("pw-play")
    t_play = time.monotonic()
    logger.log("playback_start")
    try:
        if paplay:
            subprocess.run([paplay, path], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)
        else:
            print("(no paplay; would speak:", text[:80], ")", flush=True)
        logger.log("playback_end",
                   play_ms=round((time.monotonic() - t_play) * 1000, 1))
    finally:
        time.sleep(suppress_open_mic)
        try:
            os.unlink(path)
        except OSError:
            pass
