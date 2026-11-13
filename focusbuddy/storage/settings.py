"""Typed settings store backed by the ``settings`` table."""

from __future__ import annotations

from typing import Any, Callable

from ..config import DEFAULT_SETTINGS, SENSITIVITY_PRESETS
from .database import Database

Listener = Callable[[str, Any], None]


class Settings:
    def __init__(self, db: Database | None):
        self._db = db
        self._values = dict(DEFAULT_SETTINGS)
        if db is not None:
            stored = db.load_settings()
            for k, v in stored.items():
                if k in DEFAULT_SETTINGS:
                    self._values[k] = self._coerce(k, v)
        self._listeners: list[Listener] = []

    @staticmethod
    def _coerce(key: str, value: Any) -> Any:
        default = DEFAULT_SETTINGS.get(key)
        try:
            if isinstance(default, bool):
                return bool(value)
            if isinstance(default, int):
                return int(value)
            if isinstance(default, float):
                return float(value)
            if isinstance(default, str):
                return str(value)
        except (TypeError, ValueError):
            return default
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def set(self, key: str, value: Any) -> None:
        value = self._coerce(key, value)
        if self._values.get(key) == value:
            return
        self._values[key] = value
        if self._db is not None:
            self._db.save_setting(key, value)
        for cb in list(self._listeners):
            try:
                cb(key, value)
            except Exception:  # a broken listener must not break settings
                import logging
                logging.getLogger("focusbuddy.settings").exception("settings listener failed")

    def update(self, values: dict) -> None:
        for k, v in values.items():
            self.set(k, v)

    def on_change(self, cb: Listener) -> None:
        self._listeners.append(cb)

    def as_dict(self) -> dict:
        return dict(self._values)

    # ---- derived helpers ----
    @property
    def sensitivity(self) -> dict:
        return SENSITIVITY_PRESETS.get(self._values["sensitivity"], SENSITIVITY_PRESETS["balanced"])

    def apply_sensitivity_preset(self, name: str) -> None:
        preset = SENSITIVITY_PRESETS.get(name, SENSITIVITY_PRESETS["balanced"])
        self.set("sensitivity", name)
        self.set("min_distraction_seconds", preset["min_seconds"])
        self.set("escalation_spacing_seconds", preset["spacing"])
