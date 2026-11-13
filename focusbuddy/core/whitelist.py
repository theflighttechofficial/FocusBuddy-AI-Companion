"""Temporary allow-list created by "Give me N minutes" (snooze)."""

from __future__ import annotations


class TemporaryWhitelist:
    def __init__(self):
        self._until: dict[str, float] = {}
        self._expired: dict[str, float] = {}   # key -> expiry time (for "detected again" messages)

    def allow(self, key: str, seconds: float, now: float) -> None:
        self._until[key] = now + seconds

    def is_allowed(self, key: str, now: float) -> bool:
        until = self._until.get(key)
        if until is None:
            return False
        if now < until:
            return True
        del self._until[key]
        self._expired[key] = until
        return False

    def remaining(self, key: str, now: float) -> float:
        return max(0.0, self._until.get(key, 0.0) - now)

    def recently_expired(self, key: str, now: float, window: float = 600.0) -> bool:
        t = self._expired.get(key)
        return t is not None and now - t < window

    def consume_expired(self, key: str) -> None:
        self._expired.pop(key, None)

    def revoke(self, key: str) -> None:
        self._until.pop(key, None)

    def active(self, now: float) -> dict[str, float]:
        return {k: v - now for k, v in self._until.items() if v > now}

    def clear(self) -> None:
        self._until.clear()
