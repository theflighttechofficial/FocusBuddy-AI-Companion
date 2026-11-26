"""Display-synchronised frame pulses for the companion loop.

A ``QTimer`` can only wait whole milliseconds: 16 ms ticks run at 62.5 Hz against a 60 Hz
display, so about every 0.4 s one picture is shown twice and the next one is skipped - a small,
regular hitch in every walk. On Windows the compositor tells us when it is about to show a new
frame (``DwmFlush`` returns once per display refresh), so a small background thread waits on
that and posts one pulse per refresh to the UI thread. Pulses never pile up: a new one is only
posted after the previous one has been handled.

Anywhere else, or if the compositor does not pace (remote desktop, display off), the caller keeps
using its timer.
"""

from __future__ import annotations

import sys
import threading
import time

from PySide6.QtCore import QObject, Signal

from ..utils.logging_setup import get_logger

log = get_logger("companion")


class VsyncPacer(QObject):
    """Emits ``pulse`` once per display refresh (UI thread, queued). ``available`` is False when the
    platform has no compositor clock; ``healthy`` turns False if the clock stops pacing."""

    pulse = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dwm = None
        self._thread: threading.Thread | None = None
        self._run = False
        self._pending = False
        self._pending_since = 0.0
        self.healthy = True
        self.period = 1 / 60
        if sys.platform.startswith("win"):
            try:
                import ctypes
                self._dwm = ctypes.WinDLL("dwmapi")
                self._dwm.DwmFlush.restype = ctypes.c_long
            except Exception:  # noqa: BLE001
                self._dwm = None

    @property
    def available(self) -> bool:
        return self._dwm is not None and self.healthy

    @property
    def running(self) -> bool:
        return self._run and self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if not self.available:
            return False
        if self.running:
            return True
        self._run = True
        self._pending = False
        self._thread = threading.Thread(target=self._loop, name="FocusBuddyVsync", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._run = False
        t = self._thread
        self._thread = None
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=0.2)

    def consumed(self) -> None:
        """The UI thread handled the last pulse; the next refresh may post another."""
        self._pending = False

    def _loop(self) -> None:
        fast = 0
        last = time.perf_counter()
        avg = 1 / 60
        while self._run:
            try:
                hr = self._dwm.DwmFlush()
            except Exception:  # noqa: BLE001
                hr = -1
            now = time.perf_counter()
            dt = now - last
            last = now
            if hr != 0 or dt < 0.002:
                # no compositor pacing (returns at once): give up, the timer takes over
                fast += 1
                if fast > 30:
                    log.info("Display sync unavailable - using the frame timer")
                    self.healthy = False
                    self._run = False
                    self.pulse.emit()          # wake the UI loop so it re-arms its timer
                    return
                time.sleep(0.004)
                continue
            fast = 0
            if dt < 0.1:
                avg += (dt - avg) * 0.05
                self.period = avg
            # a pulse lost on the way (should not happen) must never stall the loop for good
            if not self._pending or now - self._pending_since > 0.25:
                self._pending = True
                self._pending_since = now
                self.pulse.emit()
