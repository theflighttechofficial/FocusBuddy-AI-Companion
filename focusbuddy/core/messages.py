"""Dialogue system.

All companion text comes from JSON dialogue files in assets/dialogue so
personalities can be edited / translated without touching code::

    MessageManager.get_message(event="DISTRACTION_WARNING", severity=3,
                               site="youtube.com", minutes=5)
"""

from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from pathlib import Path

from ..config import ASSETS_DIR
from ..utils.logging_setup import get_logger

log = get_logger("messages")


class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class MessageManager:
    def __init__(self, personality: str = "cat", dialogue_dir: Path | None = None, seed: int | None = None):
        self.dir = dialogue_dir or (ASSETS_DIR / "dialogue")
        self._rng = random.Random(seed)
        self._recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=8))
        self.base = self._load("default")
        self.personality = personality
        self.lines = self._load(personality)
        self.defaults: dict = {"name": "friend", "buddy": "Buddy", "site": "this", "minutes": 0, "seconds": 0, "score": 0}

    def _load(self, name: str) -> dict:
        path = self.dir / f"{name}.json"
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            if name != "default":
                log.warning("Dialogue file missing personality=%s", name)
            return {}
        except ValueError as exc:
            log.warning("Dialogue file invalid personality=%s err=%s", name, exc)
            return {}

    def set_personality(self, personality: str) -> None:
        self.personality = personality
        self.lines = self._load(personality)

    def set_defaults(self, **kw) -> None:
        self.defaults.update({k: v for k, v in kw.items() if v not in (None, "")})

    # small talk: the personality's own lines *and* the shared ones, so a chatty companion rarely repeats
    MERGED = frozenset({"IDLE_CHATTER", "CHEER", "COMPLIMENT", "FUN", "CLICKED", "PETTED", "HOVER", "ENCOURAGE",
                        "WELCOME_BACK", "STRETCH", "WATER"})

    def _candidates(self, event: str, severity: int | None) -> list[str]:
        if event in self.MERGED:
            own, shared = self.lines.get(event), self.base.get(event)
            out = (list(own) if isinstance(own, list) else []) + (list(shared) if isinstance(shared, list) else [])
            if out:
                return out
        for source in (self.lines, self.base):
            entry = source.get(event)
            if entry is None:
                continue
            if isinstance(entry, dict):
                if severity is not None:
                    # nearest severity at or below the requested one, else any
                    keys = sorted((int(k) for k in entry if k.isdigit()), reverse=True)
                    for k in keys:
                        if k <= severity:
                            return list(entry[str(k)])
                    if keys:
                        return list(entry[str(keys[-1])])
                merged: list[str] = []
                for v in entry.values():
                    merged.extend(v)
                return merged
            if entry:
                return list(entry)
        return []

    def get_message(self, event: str, severity: int | None = None, **ctx) -> str:
        options = self._candidates(event, severity)
        if not options:
            return ""
        recent = self._recent[event]
        fresh = [o for o in options if o not in recent] or options
        line = self._rng.choice(fresh)
        recent.append(line)
        values = _SafeDict(self.defaults)
        values.update({k: v for k, v in ctx.items() if v is not None})
        try:
            return line.format_map(values)
        except (ValueError, IndexError):
            return line
