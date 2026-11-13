"""Structured application logging.

Format:  2026-10-05 10:21:03 INFO  monitor   Active window app=chrome.exe
Window titles are never logged at INFO level (they can contain private
information); only executable names and domains are.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s %(levelname)-5s %(name)-12s %(message)s"
_configured = False


def setup_logging(log_dir: Path, debug: bool = False) -> None:
    global _configured
    if _configured:
        return
    _configured = True
    root = logging.getLogger("focusbuddy")
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter(_FMT, "%Y-%m-%d %H:%M:%S")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_dir / "focusbuddy.log", maxBytes=1_000_000,
                                 backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass  # logging must never break the app
    if sys.stderr is not None:  # pythonw has no stderr
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"focusbuddy.{name}")


def kv(**fields) -> str:
    """Render key=value pairs for structured log lines."""
    parts = []
    for k, v in fields.items():
        if v is None:
            continue
        s = str(v)
        if " " in s:
            s = f'"{s}"'
        parts.append(f"{k}={s}")
    return " ".join(parts)
