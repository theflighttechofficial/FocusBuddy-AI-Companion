"""In-memory window manager used by tests and ``--demo`` mode.

It lets the full pipeline (monitor -> detector -> intervention -> blocker
-> companion) run on any OS without touching real windows.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
import os
from typing import Callable, Optional

from .base import WindowInfo, WindowManager, WinRect


class SimulatedWindowManager(WindowManager):
    def __init__(self, own_pid: int | None = None):
        self._ids = itertools.count(1001)
        self.windows: dict[int, WindowInfo] = {}
        self.foreground: Optional[int] = None
        self.idle = 0.0
        self.own_pid = own_pid if own_pid is not None else os.getpid()
        self.actions: list[tuple[str, int]] = []
        self.on_change: Optional[Callable[[str, int], None]] = None
        self.tab_titles: dict[int, list[str]] = {}     # handle -> titles of the tabs behind the active one

    def add_window(self, title: str, exe: str, rect: tuple[int, int, int, int], pid: int | None = None,
                   foreground: bool = True) -> int:
        h = next(self._ids)
        self.windows[h] = WindowInfo(handle=h, title=title, exe=exe.lower(), pid=pid or (h * 7),
                                     rect=WinRect(*rect), coords="logical")
        if foreground:
            self.foreground = h
        return h

    def set_title(self, handle: int, title: str) -> None:
        if handle in self.windows:
            self.windows[handle].title = title

    def focus(self, handle: Optional[int]) -> None:
        self.foreground = handle
        if handle in self.windows:
            self.windows[handle].minimized = False

    def remove(self, handle: int) -> None:
        self.windows.pop(handle, None)
        if self.foreground == handle:
            self.foreground = None

    # ------------------------------------------------------------------ API
    def get_active_window(self) -> Optional[WindowInfo]:
        if self.foreground is None:
            return None
        info = self.windows.get(self.foreground)
        if info is None or info.minimized:
            return None
        return replace(info, extra=dict(info.extra or {}))      # a snapshot, like the real backend

    def get_window(self, handle: int) -> Optional[WindowInfo]:
        info = self.windows.get(handle)
        return replace(info, extra=dict(info.extra or {})) if info is not None else None

    def is_window(self, handle: int) -> bool:
        return handle in self.windows

    def minimize_window(self, handle: int) -> bool:
        if handle not in self.windows:
            return False
        self.windows[handle].minimized = True
        if self.foreground == handle:
            self.foreground = None
        self.actions.append(("minimize", handle))
        if self.on_change:
            self.on_change("minimize", handle)
        return True

    def close_window(self, handle: int) -> bool:
        if handle not in self.windows:
            return False
        self.actions.append(("close", handle))
        self.remove(handle)
        if self.on_change:
            self.on_change("close", handle)
        return True

    def close_browser_tab(self, handle: int, expect_title: str) -> bool:
        if handle not in self.windows or self.foreground != handle:
            return False
        if (self.windows[handle].title or "") != (expect_title or ""):
            return False
        self.actions.append(("close_tab_keys", handle))
        nxt = self.tab_titles.get(handle)
        if nxt:
            self.set_title(handle, nxt.pop(0))
        else:
            self.remove(handle)                        # the last tab: the window goes away
        if self.on_change:
            self.on_change("close_tab_keys", handle)
        return True

    def activate_window(self, handle: int) -> bool:
        if handle not in self.windows:
            return False
        self.focus(handle)
        self.actions.append(("activate", handle))
        if self.on_change:
            self.on_change("activate", handle)
        return True

    def get_idle_seconds(self) -> float:
        return self.idle

    def list_windows(self) -> list[WindowInfo]:
        wins = [w for w in self.windows.values() if not w.minimized]
        wins.sort(key=lambda w: 0 if w.handle == self.foreground else 1)
        return wins
