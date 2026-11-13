"""Scripted, safe demo of the full core flow (works on any OS).

Real Qt windows play the parts of "VS Code" and a browser showing a video
site.  A simulated WindowManager reports them as the foreground window and
a simulated browser extension answers tab commands, so the *real* monitor,
detector, intervention engine, blocker and companion run end to end:

    start -> companion walks -> focus session -> distraction opens ->
    score rises -> notice -> walk/run to window -> warning -> countdown ->
    tab closed safely -> back to editor -> celebration -> stats updated
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath
from PySide6.QtWidgets import QWidget

from .app import FocusBuddyApp
from .companion.states import CompanionState as S
from .config import data_dir
from .platform.simulated import SimulatedWindowManager
from .utils.logging_setup import get_logger

log = get_logger("demo")
DEMO_CLIENT = "demo-extension"
CAPTION = 30


class FakeAppWindow(QWidget):
    """A plain window drawn to look like an editor or a browser."""

    def __init__(self, kind: str, title: str):
        super().__init__(None, Qt.Window)
        self.kind = kind
        self.setWindowTitle(title)
        self.url = "youtube.com/watch?v=cat-compilation" if kind == "browser" else ""
        self.on_activate = None
        self.on_geometry = None

    def changeEvent(self, e):
        if e.type() == e.Type.ActivationChange and self.isActiveWindow() and self.on_activate:
            self.on_activate()
        super().changeEvent(e)

    def moveEvent(self, e):
        if self.on_geometry:
            self.on_geometry()

    def resizeEvent(self, e):
        if self.on_geometry:
            self.on_geometry()

    def paintEvent(self, _):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if self.kind == "editor":
            g.fillRect(self.rect(), QColor("#1E1E2E"))
            g.fillRect(0, 0, w, 34, QColor("#181825"))
            g.fillRect(0, 34, 52, h, QColor("#181825"))
            g.setPen(QColor("#CDD6F4"))
            g.setFont(QFont("monospace", 10))
            g.drawText(14, 22, "main.py — focus_project")
            code = ["import numpy as np", "", "def train_model(data):", "    model = build()",
                    "    for epoch in range(10):", "        loss = model.fit(data)", "        log(epoch, loss)",
                    "    return model", "", "# TODO: write tests", "if __name__ == '__main__':",
                    "    train_model(load())"]
            for i, ln in enumerate(code):
                g.setPen(QColor("#6C7086"))
                g.drawText(18, 64 + i * 22, f"{i + 1:>2}")
                g.setPen(QColor("#CBA6F7" if ln.strip().startswith(("def", "import", "for", "return", "if"))
                                else "#A6E3A1" if ln.strip().startswith("#") else "#CDD6F4"))
                g.drawText(66, 64 + i * 22, ln)
        else:
            g.fillRect(self.rect(), QColor("#FFFFFF"))
            g.fillRect(0, 0, w, 74, QColor("#DEE1E6"))
            g.setPen(Qt.NoPen)
            g.setBrush(QColor("#FFFFFF"))
            g.drawRoundedRect(QRectF(10, 6, 230, 30), 8, 8)
            g.setPen(QColor("#333"))
            g.setFont(QFont("sans", 9))
            g.drawText(QRectF(22, 6, 210, 30), Qt.AlignVCenter, "Cat compilation - VideoTube")
            g.setPen(Qt.NoPen)
            g.setBrush(QColor("#F1F3F4"))
            g.drawRoundedRect(QRectF(10, 40, w - 20, 28), 14, 14)
            g.setPen(QColor("#202124"))
            g.drawText(QRectF(28, 40, w - 40, 28), Qt.AlignVCenter, self.url)
            vid = QRectF(24, 92, w - 48, (w - 48) * 0.5)
            grad = QLinearGradient(vid.topLeft(), vid.bottomRight())
            grad.setColorAt(0, QColor("#2B2D42"))
            grad.setColorAt(1, QColor("#8D99AE"))
            g.setBrush(grad)
            g.drawRoundedRect(vid, 12, 12)
            g.setBrush(QColor(255, 255, 255, 220))
            c = vid.center()
            tri = QPainterPath(QPointF(c.x() - 18, c.y() - 24))
            tri.lineTo(c.x() + 26, c.y())
            tri.lineTo(c.x() - 18, c.y() + 24)
            tri.closeSubpath()
            g.drawPath(tri)
            g.setPen(QColor("#111"))
            g.setFont(QFont("sans", 13, QFont.Bold))
            g.drawText(QPointF(26, vid.bottom() + 30), "Funny cat compilation #47 (you won't believe #3)")
            g.setPen(QColor("#666"))
            g.setFont(QFont("sans", 10))
            g.drawText(QPointF(26, vid.bottom() + 52), "12M views · recommended for you")


class DemoDirector:
    def __init__(self, qapp, capture_dir: str | None = None, auto_quit: bool = False, showcase: bool = False):
        self.qapp = qapp
        self.showcase = showcase
        self.auto_quit = auto_quit
        self.capture_dir = Path(capture_dir) if capture_dir else None
        if self.capture_dir:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
        ddir = data_dir() / "demo-session"
        shutil.rmtree(ddir, ignore_errors=True)
        ddir.mkdir(parents=True, exist_ok=True)
        self.wm = SimulatedWindowManager(own_pid=os.getpid())
        self.app = FocusBuddyApp(qapp, demo=True, data_path=ddir, window_manager=self.wm, start_minimized=True)
        s = self.app.settings
        s.update({"first_run_done": True, "blocking_mode": "auto", "website_action": "close_tab",
                  "sensitivity": "strict", "min_distraction_seconds": 2, "escalation_spacing_seconds": 4,
                  "grace_period_seconds": 6, "focus_minutes": 25, "speech_bubbles": True,
                  "notifications_enabled": False, "user_name": "Varun", "character": "cat"})
        self.app.tau_override = 10.0
        self.app._update_message_defaults()

        geo = QGuiApplication.primaryScreen().availableGeometry()
        self.screen = geo
        W, H = geo.width(), geo.height()
        self.editor = FakeAppWindow("editor", "main.py - focus_project - Visual Studio Code")
        self.editor.setGeometry(geo.x() + 30, geo.y() + 60, int(W * 0.48), int(H * 0.6))
        self.browser = FakeAppWindow("browser", "Funny cat compilation #47 - YouTube - Google Chrome")
        self.browser.setGeometry(geo.x() + int(W * 0.44), geo.y() + int(H * 0.14), int(W * 0.52), int(H * 0.6))
        self.h_editor = self._register(self.editor, "code.exe")
        self.h_browser = None
        self.wm.on_change = self._on_wm_change
        self.frames = 0
        self.t0 = 0.0
        self.log: list[str] = []

    # ----------------------------------------------------------- windows
    def _register(self, win: FakeAppWindow, exe: str) -> int:
        # Registered rect includes the (drawn) 30 px caption, like DWM frame bounds on Windows.
        g = win.geometry()
        h = self.wm.add_window(win.windowTitle(), exe, (g.x(), g.y() - CAPTION, g.width(), g.height() + CAPTION),
                               foreground=False)

        def sync():
            gg = win.geometry()
            info = self.wm.windows.get(h)
            if info and info.rect:
                info.rect.x, info.rect.y = gg.x(), gg.y() - CAPTION
                info.rect.w, info.rect.h = gg.width(), gg.height() + CAPTION
        win.on_geometry = sync
        win.on_activate = lambda: self.wm.focus(h)
        return h

    def focus(self, which: str) -> None:
        win, h = (self.editor, self.h_editor) if which == "editor" else (self.browser, self.h_browser)
        if h is None:
            return
        win.show()
        win.raise_()
        self.wm.focus(h)
        self.event(f"user focuses {which}")

    def open_browser(self) -> None:
        self.browser.show()
        self.h_browser = self._register(self.browser, "chrome.exe")
        self.wm.focus(self.h_browser)
        info = self.wm.windows[self.h_browser]
        info.pid = 4242
        # simulated extension: report the active tab like the real extension does
        self.app.bridge._on_report({"client": DEMO_CLIENT, "browser": "chrome", "active": {
            "url": "https://www.youtube.com/watch?v=cat-compilation", "title": "Funny cat compilation #47 - YouTube",
            "tabId": 11, "windowId": 1, "incognito": False, "focused": True}})
        self.event("user opens youtube.com in the browser")

    def _on_wm_change(self, kind: str, handle: int) -> None:
        win = self.browser if handle == self.h_browser else self.editor if handle == self.h_editor else None
        if win is None:
            return
        if kind in ("minimize", "close"):
            win.hide()
            self.event(f"window {kind}d by FocusBuddy")
        elif kind == "activate":
            win.show()
            win.raise_()

    def _extension_tick(self) -> None:
        """Answer bridge commands as the real extension would."""
        b = self.app.bridge
        if b is None:
            return
        for cmd in b._take_commands(DEMO_CLIENT):
            if cmd["type"] == "close_tab" and self.h_browser is not None:
                self._x_pressed_until = time.monotonic() + 0.3   # show the X button "pressed"

                def close_tab(cid=cmd["id"]):
                    if self.h_browser is None:
                        return
                    self.event("extension closes the youtube.com tab")
                    self.browser.hide()
                    self.wm.remove(self.h_browser)
                    self.h_browser = None
                    b._on_ack({"id": cid, "ok": True})
                    b._on_report({"client": DEMO_CLIENT, "browser": "chrome", "active": None})
                    QTimer.singleShot(4000, lambda: self.focus("editor"))   # after relief -> happy -> jump
                    QTimer.singleShot(7000, self.finish)
                QTimer.singleShot(250, close_tab)
        # keep the simulated extension's heartbeat fresh
        if self.h_browser is not None:
            b._on_report({"client": DEMO_CLIENT, "browser": "chrome", "active": {
                "url": "https://www.youtube.com/watch?v=cat-compilation",
                "title": "Funny cat compilation #47 - YouTube", "tabId": 11, "windowId": 1, "focused": True}})

    def event(self, text: str) -> None:
        t = time.monotonic() - self.t0
        line = f"[{t:5.1f}s] {text}  (companion={self.app.companion.state.value})"
        self.log.append(line)
        log.info("DEMO %s", line)
        print(line, flush=True)

    # ------------------------------------------------------------ script
    def start(self) -> None:
        self.t0 = time.monotonic()
        self.editor.show()
        self.wm.focus(self.h_editor)
        self.app.start()
        b = self.app.companion.behavior
        l, t, r, btm = b.env.screen_bounds(b.x, b.y)
        if self.showcase:
            self._start_showcase()
        else:
            b.mover.teleport(l + (r - l) * 0.15, btm)
            b._start_plan([("goto", l + (r - l) * 0.35, btm, False, S.WALKING), ("state", S.IDLE, 1.0)],
                          mode="auto")
        QTimer.singleShot(180_000, self.finish)  # safety net
        self.ext_timer = QTimer()
        self.ext_timer.timeout.connect(self._extension_tick)
        self.ext_timer.start(100)
        self.watch_timer = QTimer()
        self.watch_timer.timeout.connect(self._watch)
        self.watch_timer.start(100)
        self._last_state = None
        if not self.showcase:
            QTimer.singleShot(2500, lambda: (self.app.start_focus(25),
                                             self.event("user starts a 25-min focus session")))
            QTimer.singleShot(6000, self.open_browser)
        if self.capture_dir:
            self.cap_timer = QTimer()
            self.cap_timer.timeout.connect(self.capture)
            self.cap_timer.start(66)

    def _start_showcase(self) -> None:
        """``--showcase``: one continuous performance, then the real distraction flow.

        SLEEP -> WAKE -> YAWN -> STRETCH -> STAND -> LOOK AROUND -> WALK -> JOG -> RUN -> SPRINT -> TURN ->
        JUMP -> LAND -> SHAKE -> SIT -> LOOK AT CURSOR -> CURSOR MOVES -> EYES TRACK -> HEAD FOLLOWS ->
        CURIOUS -> PLAY -> REST, then YouTube opens: notice -> head turn -> eyes track -> alert -> stand ->
        walk / jog / run / sprint -> arrive -> stop -> look at the window -> look at the user -> SHOUT /
        warning -> countdown -> the close button is pressed (safe window API) -> relief -> happy -> jump ->
        spin -> celebrate -> roaming."""
        from .companion.actions import Priority
        from .companion.showcase import acceptance_steps
        c = self.app.companion
        b = c.behavior
        l, t, r, btm = b.env.screen_bounds(b.x, b.y)
        w = r - l
        b.mover.teleport(l + w * 0.08, btm)
        b.mover.facing, b.mover.facing_blend = 1, 1.0
        self.app.start_focus(25)
        self.event("user starts a 25-min focus session")
        steps = acceptance_steps(b.species, l + w * 0.08, w * 0.8, btm, c.px,
                                 mark=lambda label: self.event(f"showcase: {label.lower()}"))
        # the cat is asleep when YouTube opens: ear twitch -> eyes open -> head turns -> stands -> ...
        steps += [("call", lambda: QTimer.singleShot(400, self.open_browser)), ("state", S.SLEEPING, None)]
        b._start_plan(steps, mode="auto", priority=Priority.SPECIAL_ACTION)

    def _watch(self) -> None:
        st = self.app.companion.state
        if st != self._last_state:
            self._last_state = st
            ep = self.app.engine.episode
            extra = f" score={ep.assessment.score} level={ep.level.name}" if ep else ""
            self.event(f"companion -> {st.value}{extra}")

    def finish(self) -> None:
        if getattr(self, "_finished", False):
            return
        self._finished = True
        summ = self.app.stats.today()
        self.event(f"stats: distractions={summ['distractions']} blocked={summ['blocked']} "
                   f"focus={summ['focus_seconds']:.0f}s distraction={summ['distraction_seconds']:.0f}s")
        self.app.show_dashboard("home")
        if self.capture_dir:
            QTimer.singleShot(1200, lambda: self.app.dashboard.grab().save(str(self.capture_dir / "dashboard.png")))
        if self.auto_quit:
            QTimer.singleShot(2000, self.app.quit)

    # ----------------------------------------------------------- capture
    def capture(self) -> None:
        """Compose a 'screenshot' from our own windows (works headless, no compositor needed)."""
        geo = self.screen
        img = QImage(geo.width(), geo.height() + (QGuiApplication.primaryScreen().geometry().height() - geo.height()),
                     QImage.Format_ARGB32)
        g = QPainter(img)
        grad = QLinearGradient(0, 0, img.width(), img.height())
        grad.setColorAt(0, QColor("#A8C0FF"))
        grad.setColorAt(1, QColor("#E0C3FC"))
        g.fillRect(img.rect(), grad)
        order = [self.editor, self.browser]
        if self.wm.foreground == self.h_editor:
            order = [self.browser, self.editor]
        for win in order:
            if win.isVisible():
                gg = win.geometry()
                cap = CAPTION
                g.fillRect(QRectF(gg.x() - 1, gg.y() - cap, gg.width() + 2, cap), QColor("#F3F3F3"))
                g.setPen(QColor("#333"))
                g.setFont(QFont("sans", 9))
                g.drawText(QRectF(gg.x() + 10, gg.y() - cap, gg.width() - 70, cap), Qt.AlignVCenter, win.windowTitle())
                # caption buttons: 46 px wide each, close button at the far right (Windows layout)
                close = QRectF(gg.right() + 1 - 46, gg.y() - cap, 46, cap)
                pressed = getattr(self, "_x_pressed_until", 0) > time.monotonic() and win is self.browser
                if pressed:
                    g.fillRect(close, QColor("#C42B1C"))
                g.setPen(QColor("#FFFFFF" if pressed else "#666"))
                g.drawText(close, Qt.AlignCenter, "✕")
                g.setPen(QColor("#666"))
                g.drawText(QRectF(gg.right() + 1 - 92, gg.y() - cap, 46, cap), Qt.AlignCenter, "□")
                g.drawText(QRectF(gg.right() + 1 - 138, gg.y() - cap, 46, cap), Qt.AlignCenter, "–")
                g.drawPixmap(gg.topLeft(), win.grab())
        # taskbar
        g.fillRect(QRectF(0, geo.bottom() + 1, img.width(), img.height() - geo.bottom()), QColor(30, 30, 40, 230))
        c = self.app.companion
        if c.window.isVisible():
            g.drawPixmap(c.window.pos(), c.window.grab())
        if c.bubble.isVisible():
            g.setOpacity(max(0.0, min(1.0, c.bubble.windowOpacity())))
            g.drawPixmap(c.bubble.pos(), c.bubble.grab())
            g.setOpacity(1.0)
        g.end()
        img.save(str(self.capture_dir / f"frame_{self.frames:04d}.png"))
        self.frames += 1


