#!/usr/bin/env python3
"""Session logging for siwc-chat: one JSONL file per day under
~/.config/siwc-chat/logs/ with wall-clock timestamps, monotonic
offsets relative to turn start / speech start / listen start, and
per-stage durations recorded as events happen.
"""
import datetime
import json
import os
import time

APP_DIR = os.path.expanduser("~/.config/siwc-chat")
LOG_DIR = os.path.join(APP_DIR, "logs")

_markers: dict[str, float] = {}
_anchors = ("turn_start", "speech_start")


def _path() -> str:
    os.makedirs(LOG_DIR, mode=0o700, exist_ok=True)
    today = datetime.date.today().isoformat()
    return os.path.join(LOG_DIR, today + ".jsonl")


def mark(name: str) -> None:
    if name == "turn_start":
        for a in _anchors:
            _markers.pop(a, None)
    _markers[name] = time.monotonic()


def log(event: str, **fields) -> None:
    now = time.monotonic()
    rec = {
        "ts": datetime.datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "event": event,
    }
    for a in _anchors:
        if a in _markers:
            rec["ms_from_" + a] = round((now - _markers[a]) * 1000, 1)
    rec.update(fields)
    try:
        fd = os.open(_path(), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass
