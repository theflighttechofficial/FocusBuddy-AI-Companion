"""Platform factory."""

from __future__ import annotations

import sys

from ..utils.logging_setup import get_logger
from .base import (AutostartManager, NullAutostart, NullWindowManager, WindowInfo,  # noqa: F401
                   WindowManager, WinRect)

log = get_logger("platform")


def create_window_manager() -> WindowManager:
    if sys.platform.startswith("win"):
        try:
            from .windows import WindowsWindowManager
            return WindowsWindowManager()
        except Exception as exc:  # noqa: BLE001
            log.error("Windows backend unavailable err=%s", exc)
    else:
        log.warning("No window backend for platform=%s - monitoring disabled (use --demo)", sys.platform)
    return NullWindowManager()


def create_autostart() -> AutostartManager:
    if sys.platform.startswith("win"):
        try:
            from .windows import WindowsAutostart
            return WindowsAutostart()
        except Exception:  # noqa: BLE001
            pass
    return NullAutostart()
