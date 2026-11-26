"""System tray icon, menu and desktop notifications."""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from ..utils.logging_setup import get_logger

log = get_logger("tray")

STATUS_COLORS = {"monitoring": "#22B07D", "focus": "#5B6CFF", "break": "#F5B83D", "paused": "#9AA3B5"}


class Tray(QObject):
    open_dashboard = Signal()
    open_settings = Signal()
    toggle_monitoring = Signal()
    start_focus = Signal(int)      # minutes
    end_focus = Signal()
    toggle_mute = Signal()
    toggle_companion = Signal()
    quit_requested = Signal()
    lock_requested = Signal()
    switch_requested = Signal()
    toggle_nap = Signal()
    toggle_quiet = Signal()

    def __init__(self, icon_factory):
        super().__init__()
        self.icon_factory = icon_factory   # callable(status_color) -> QIcon
        self.available = QSystemTrayIcon.isSystemTrayAvailable()
        self.icon = QSystemTrayIcon()
        self.icon.setToolTip("FocusBuddy")
        self.menu = QMenu()
        self._title = self.menu.addAction("FocusBuddy")
        self._title.setEnabled(False)
        self._status = self.menu.addAction("● Monitoring")
        self._status.setEnabled(False)
        self.menu.addSeparator()
        self.act_pause = self.menu.addAction("Pause Monitoring", self.toggle_monitoring.emit)
        self.focus_menu = self.menu.addMenu("Start Focus Session")
        for m in (25, 45, 60, 90):
            self.focus_menu.addAction(f"{m} minutes", lambda m=m: self.start_focus.emit(m))
        self.act_end = self.menu.addAction("End Focus Session", self.end_focus.emit)
        self.act_end.setVisible(False)
        self.menu.addAction("Open Dashboard", self.open_dashboard.emit)
        self.menu.addAction("Settings", self.open_settings.emit)
        self.menu.addSeparator()
        self.act_mute: QAction = self.menu.addAction("Mute Companion", self.toggle_mute.emit)
        self.act_hide: QAction = self.menu.addAction("Hide Companion", self.toggle_companion.emit)
        self.act_nap: QAction = self.menu.addAction("Take a Nap 💤", self.toggle_nap.emit)
        self.act_quiet: QAction = self.menu.addAction("Quiet Mode", self.toggle_quiet.emit)
        self.act_quiet.setCheckable(True)
        self.menu.addSeparator()
        self.act_lock = self.menu.addAction("Lock", self.lock_requested.emit)
        self.menu.addAction("Switch Profile…", self.switch_requested.emit)
        self.menu.addSeparator()
        self.menu.addAction("Quit FocusBuddy", self.quit_requested.emit)
        self.icon.setContextMenu(self.menu)
        self.icon.activated.connect(self._activated)
        self._last_notify: dict[str, float] = {}
        self.set_status("monitoring", "Monitoring")

    def show(self) -> None:
        if self.available:
            self.icon.show()
        else:
            log.info("System tray not available on this desktop")

    def hide(self) -> None:
        self.icon.hide()

    def _activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.open_dashboard.emit()

    def set_status(self, kind: str, text: str) -> None:
        self._status.setText(f"●  {text}")
        self.icon.setToolTip(f"FocusBuddy - {text}")
        self.icon.setIcon(self.icon_factory(STATUS_COLORS.get(kind, "#22B07D")))

    def set_monitoring(self, on: bool) -> None:
        self.act_pause.setText("Pause Monitoring" if on else "Resume Monitoring")

    def set_focus_running(self, running: bool) -> None:
        self.focus_menu.menuAction().setVisible(not running)
        self.act_end.setVisible(running)

    def set_muted(self, muted: bool) -> None:
        self.act_mute.setText("Unmute Companion" if muted else "Mute Companion")

    def set_companion_visible(self, visible: bool) -> None:
        self.act_hide.setText("Hide Companion" if visible else "Show Companion")

    def set_quiet(self, on: bool) -> None:
        self.act_quiet.setChecked(bool(on))

    def set_napping(self, napping: bool) -> None:
        self.act_nap.setText("Wake Up Companion" if napping else "Take a Nap 💤")

    def notify(self, title: str, message: str, key: str | None = None, min_interval: float = 60.0) -> bool:
        """Desktop notification with per-key rate limiting (avoid spam)."""
        key = key or message
        now = time.perf_counter()
        if now - self._last_notify.get(key, -1e9) < min_interval:
            return False
        self._last_notify[key] = now
        if self.available and QSystemTrayIcon.supportsMessages():
            self.icon.showMessage(title, message, QSystemTrayIcon.Information, 6000)
        log.info("Notification shown key=%s", key)
        return True
