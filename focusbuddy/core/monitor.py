"""Monitoring engine: turns the foreground window into an ``Activity``."""

from __future__ import annotations

import os
from typing import Callable, Optional

from ..browser.bridge import BrowserBridge
from ..browser.title_parser import infer_domain
from ..config import BROWSER_EXES
from ..platform.base import WindowManager
from ..utils.clock import SYSTEM_CLOCK, Clock
from ..utils.logging_setup import get_logger, kv
from .activity import Activity
from .rules import is_game_path, normalize_domain, url_path

log = get_logger("monitor")


class ActivityMonitor:
    def __init__(self, wm: WindowManager, bridge: Optional[BrowserBridge],
                 brands_provider: Callable[[], dict] | None = None,
                 use_title_fallback: Callable[[], bool] | None = None,
                 clock: Clock = SYSTEM_CLOCK, own_pid: int | None = None):
        self.wm = wm
        self.bridge = bridge
        self.brands_provider = brands_provider or (lambda: {})
        self.use_title_fallback = use_title_fallback or (lambda: True)
        self.clock = clock
        self.own_pid = own_pid if own_pid is not None else os.getpid()
        self._last_key: Optional[str] = None
        self.errors = 0

    def poll(self) -> Optional[Activity]:
        now = self.clock.now()
        try:
            info = self.wm.get_active_window()
        except Exception as exc:  # noqa: BLE001 - never let the monitor die
            self.errors += 1
            log.warning("Active window unavailable err=%s", exc)
            return None
        try:
            idle = float(self.wm.get_idle_seconds())
        except Exception:  # noqa: BLE001
            idle = 0.0
        if info is None:
            return None
        if info.pid == self.own_pid:
            return Activity(timestamp=now, window=info, exe=info.exe, title="", is_self=True, idle_seconds=idle)

        exe = (info.exe or "").lower()
        act = Activity(timestamp=now, window=info, exe=exe, title=info.title or "", idle_seconds=idle)
        if is_game_path((info.extra or {}).get("path")):
            act.extra["game"] = True
        family = BROWSER_EXES.get(exe)
        if family:
            act.is_browser = True
            act.browser = family
            tab = self.bridge.get_active_tab(family, info.title) if self.bridge else None
            if tab is not None and tab.url:
                act.url = tab.url
                act.domain = normalize_domain(tab.url)
                act.path = url_path(tab.url)
                act.domain_source = "extension"
                act.tab_id = tab.tab_id
                act.tab_client = tab.client
                act.title = tab.title or act.title
                if tab.url.startswith(("chrome://", "edge://", "about:", "moz-extension://", "chrome-extension://")):
                    act.domain = None
                    act.domain_source = "internal"
            elif self.use_title_fallback():
                d = infer_domain(info.title, self.brands_provider())
                if d:
                    act.domain = d
                    act.domain_source = "title"

        if act.key != self._last_key:
            self._last_key = act.key
            log.info("Active window %s", kv(app=exe or "?", website=act.domain, source=act.domain_source
                                            if act.domain else None))
        return act
