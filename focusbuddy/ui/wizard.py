"""First-run setup wizard."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QButtonGroup, QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from ..companion.characters import BUILTIN, selectable_characters, get_character
from ..companion.states import CompanionState as S
from .settings_pages import CharacterCard
from .theme import C, themed
from .widgets import CharacterPreview, label


def choice_card(title: str, desc: str, emoji: str = "") -> QPushButton:
    b = QPushButton()
    b.setObjectName("ChoiceCard")
    b.setCheckable(True)
    b.setCursor(Qt.PointingHandCursor)
    b.setMinimumHeight(110)
    v = QVBoxLayout(b)
    v.setContentsMargins(16, 14, 16, 14)
    t = QLabel(f"{emoji}  {title}" if emoji else title)
    t.setStyleSheet("font-size:15px; font-weight:700; background:transparent;")
    t.setAttribute(Qt.WA_TransparentForMouseEvents)
    d = QLabel(desc)
    d.setWordWrap(True)
    d.setObjectName("CardDesc")
    d.setAttribute(Qt.WA_TransparentForMouseEvents)
    v.addWidget(t)
    v.addWidget(d)
    v.addStretch(1)
    return b


class ExtensionSteps(QWidget):
    """Install the browser extension, step by step, with a live "connected" check (the setup wizard and
    Settings -> Browser integration)."""

    from PySide6.QtCore import Signal as _Signal
    connected_changed = _Signal(bool)

    def __init__(self, app, compact: bool = False):
        super().__init__()
        from ..config import EXTENSION_DIR
        from ..utils import browsers
        self.app = app
        self.skipped = False
        self.connected = False
        bridge = getattr(app, "bridge", None)
        self.available = bridge is not None and not getattr(bridge, "error", "")
        self.folder = str(EXTENSION_DIR)
        found = browsers.installed()
        self.has_browser = bool(found)
        self.browser_id, self.browser_name = (found[0] if found else ("chrome", "Google Chrome"))
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        page = {"chrome": "chrome://extensions", "edge": "edge://extensions",
                "brave": "brave://extensions"}.get(self.browser_id, "chrome://extensions")

        def step(n: int, title: str, desc: str, *buttons) -> None:
            row = QHBoxLayout()
            row.setSpacing(14)
            num = QLabel(str(n))
            num.setFixedSize(32, 32)
            num.setAlignment(Qt.AlignCenter)
            themed(num, lambda: f"background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 {C['primary']}, "
                                f"stop:1 {C['primary_grad']}); color: {C['on_accent']}; border-radius: 16px; "
                                "font-weight: 800; font-size: 14px;")
            row.addWidget(num, 0, Qt.AlignTop)
            col = QVBoxLayout()
            col.setSpacing(2)
            t = QLabel(title)
            t.setStyleSheet("font-size:15px; font-weight:700;")
            col.addWidget(t)
            if desc:
                d = label(desc, "Muted", wrap=True)
                d.setTextInteractionFlags(Qt.TextSelectableByMouse)
                col.addWidget(d)
            row.addLayout(col, 1)
            for b in buttons:
                row.addWidget(b, 0, Qt.AlignVCenter)
            v.addLayout(row)

        self.open_btn = QPushButton(f"Open {self.browser_name.split()[-1]}'s extensions")
        self.open_btn.setObjectName("Primary")
        self.open_btn.setCursor(Qt.PointingHandCursor)
        self.open_btn.clicked.connect(self._open_page)
        step(1, f"Open the extensions page in {self.browser_name}",
             f"Or type {page} into the address bar." if found else
             "Install Google Chrome (or Microsoft Edge / Brave) first, then open its extensions page.",
             self.open_btn)
        step(2, "Turn on “Developer mode”", "The switch in the top-right corner of the extensions page.")
        copy = QPushButton("Copy folder path")
        copy.setCursor(Qt.PointingHandCursor)
        copy.clicked.connect(self._copy)
        openf = QPushButton("Open folder")
        openf.setCursor(Qt.PointingHandCursor)
        openf.clicked.connect(self._open_folder)
        step(3, "Click “Load unpacked” and choose FocusBuddy's extension folder",
             self.folder + "\n(Copy the path and paste it into the folder picker's address bar.)", copy, openf)
        step(4, "That's it!", "Keep the browser open - the status below turns green when the extension says hello.")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(46)
        v.addSpacing(4)
        v.addWidget(self.status)
        self._set_status()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(800)

    def _open_page(self) -> None:
        from ..utils import browsers
        name = browsers.open_extensions_page(self.browser_id) or browsers.open_extensions_page()
        if name is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No browser found",
                                    "Google Chrome, Microsoft Edge or Brave wasn't found. Install one of them, "
                                    "or open its extensions page yourself.")

    def _copy(self) -> None:
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.folder)
        self.status.setText(self.status.text() if self.connected else "📋  Folder path copied - paste it into "
                            "the folder picker after clicking “Load unpacked”.")
        QTimer.singleShot(2500, self._set_status)

    def _open_folder(self) -> None:
        from .settings_pages import open_path
        open_path(self.folder)

    def _poll(self) -> None:
        bridge = getattr(self.app, "bridge", None)
        on = bool(bridge is not None and bridge.connected())
        if on != self.connected:
            self.connected = on
            self._set_status()
            self.connected_changed.emit(on)

    def retheme(self) -> None:
        self._set_status()

    def _set_status(self) -> None:
        if not self.available:
            text, bg, fg = ("⚠  The connection port is busy, so the extension can't connect right now. You can "
                            "continue and set it up later from Settings → Browser integration.",
                            C["danger_soft"], C["danger"])
        elif self.connected:
            names = sorted({c.browser.title() for c in self.app.bridge.clients()}) if self.app.bridge else []
            text = "✅  Connected" + (f" to {', '.join(names)}" if names else "") + " - you're all set!"
            bg, fg = C["focus_soft"], C["focus"]
        else:
            text, bg, fg = ("⏳  Waiting for the extension to connect…", C["primary_soft"], C["primary_text"])
        self.status.setText(text)
        self.status.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:14px; padding:12px 16px; "
                                  "font-size:14px; font-weight:650;")


class ExtensionDialog(QDialog):
    """Once, for people who set FocusBuddy up before the extension step existed: connect the browser."""

    def __init__(self, app, icon=None):
        super().__init__()
        self.setWindowTitle("Connect your browser - FocusBuddy")
        if icon:
            self.setWindowIcon(icon)
        self.setMinimumSize(720, 560)
        v = QVBoxLayout(self)
        v.setContentsMargins(34, 28, 34, 22)
        v.setSpacing(12)
        v.addWidget(label("Connect your browser", "H1"))
        v.addWidget(label("Your companion can see exactly which website you're on - and close a distracting tab "
                          "for you - once its Chrome extension is installed. It takes about a minute.",
                          "Muted", wrap=True))
        self.ext = ExtensionSteps(app)
        v.addWidget(self.ext)
        v.addStretch(1)
        row = QHBoxLayout()
        later = QPushButton("Later")
        later.setObjectName("Ghost")
        later.clicked.connect(self.reject)
        self.done_btn = QPushButton("Done")
        self.done_btn.setObjectName("Primary")
        self.done_btn.setMinimumWidth(120)
        self.done_btn.clicked.connect(self.accept)
        self.done_btn.setEnabled(False)
        self.ext.connected_changed.connect(self.done_btn.setEnabled)
        row.addStretch(1)
        row.addWidget(later)
        row.addWidget(self.done_btn)
        v.addLayout(row)

    def paintEvent(self, _ev):
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QPainter
        from .motion import paint_canvas
        g = QPainter(self)
        paint_canvas(g, QRectF(self.rect()))
        g.end()

    def showEvent(self, ev):
        super().showEvent(ev)
        from .theme import fluent_window
        fluent_window(self)


class SetupWizard(QDialog):
    def paintEvent(self, _ev):
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QPainter
        from .motion import paint_canvas
        g = QPainter(self)
        paint_canvas(g, QRectF(self.rect()))
        g.end()

    def showEvent(self, ev):
        super().showEvent(ev)
        from .theme import fluent_window
        fluent_window(self)

    def __init__(self, app, icon=None):
        super().__init__()
        self.app = app
        self.setWindowTitle("Welcome to FocusBuddy")
        if icon:
            self.setWindowIcon(icon)
        self.setObjectName("Wizard")
        self.setMinimumSize(760, 600)
        s = app.settings
        self.choice = {
            "character": s["character"], "sensitivity": s["sensitivity"], "blocking_mode": s["blocking_mode"],
            "focus_minutes": s["focus_minutes"], "focus_enforcement": s["focus_enforcement"],
        }
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 30, 36, 24)
        root.setSpacing(16)
        self.dots = QHBoxLayout()
        self.dots.setSpacing(6)
        self.dot_widgets = []
        root.addLayout(self.dots)
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        # 1 welcome
        p = self._page("Meet your new focus companion.",
                       "A little friend who lives on your desktop, notices distractions and helps you get back to work - "
                       "kindly, not bossily.")
        row = QHBoxLayout()
        row.addStretch(1)
        for cid, st in (("cat", S.WAVING), ("dog", S.CELEBRATING), ("robot", S.IDLE), ("fox", S.LOOKING_AROUND)):
            row.addWidget(CharacterPreview(BUILTIN[cid], 130, st))
        row.addStretch(1)
        p.layout().addLayout(row)
        self.name = QLineEdit(s["user_name"])
        self.name.setPlaceholderText("What should we call you? (optional)")
        self.name.setMaximumWidth(360)
        p.layout().addWidget(self.name, 0, Qt.AlignHCenter)
        p.layout().addWidget(label("🔒 Everything stays on this computer. No keystrokes, screenshots or uploads.", "Faint"),
                             0, Qt.AlignHCenter)
        p.layout().addStretch(1)

        # 2 character
        p = self._page("Choose your companion", "You can change this any time.")
        grid = QGridLayout()
        grid.setSpacing(10)
        self.char_group = QButtonGroup(self)
        for i, spec in enumerate(selectable_characters()):
            c = CharacterCard(spec)
            c.setMinimumSize(150, 176)
            c.preview.setFixedSize(96, 96)
            self.char_group.addButton(c)
            c.setChecked(spec.id == self.choice["character"])
            c.clicked.connect(lambda _=False, sid=spec.id: self.choice.__setitem__("character", sid))
            grid.addWidget(c, i // 4, i % 4)
        p.layout().addLayout(grid)
        p.layout().addStretch(1)

        # 3 sensitivity
        p = self._page("How sensitive should I be?", "How quickly a distraction gets noticed.")
        self._choices(p, "sensitivity", [
            ("relaxed", "Relaxed", "Only long distractions. Plenty of slack.", "🌿"),
            ("balanced", "Balanced", "Notices after a few minutes. Recommended.", "⚖"),
            ("strict", "Strict", "Speaks up quickly. Great for exam prep.", "🎯")])

        # 4 blocking
        p = self._page("What should I do about distractions?", "You can always override me.")
        p.layout().addWidget(label("During a focus session", "H3"))
        self._choices(p, "focus_enforcement", [
            ("auto_close", "Close them for me", "A short countdown, then I close the tab or app. (Recommended)", "⏱"),
            ("instant", "Block instantly", "Distracting sites never load during the session.", "🛡"),
            ("same", "Same as the rest", "Use the choice for the rest of the day (below).", "🤝")], stretch=False)
        p.layout().addWidget(label("The rest of the day", "H3"))
        self._choices(p, "blocking_mode", [
            ("warn", "Warnings only", "I'll point it out, but never close anything.", "💬"),
            ("ask", "Ask before blocking", "I'll offer to close distractions. (Default)", "🤝"),
            ("auto", "Automatic blocking", "I'll close sites you mark auto-close after a countdown you can cancel.", "⚡")])

        # 5 focus duration
        p = self._page("Default focus session", "Pick a comfortable length.")
        chips = QHBoxLayout()
        self.focus_group = QButtonGroup(self)
        for m in (25, 45, 60):
            b = QPushButton(f"{m} min")
            b.setObjectName("Chip")
            b.setCheckable(True)
            b.setMinimumHeight(44)
            b.setProperty("m", m)
            self.focus_group.addButton(b)
            chips.addWidget(b)
            if m == self.choice["focus_minutes"]:
                b.setChecked(True)
        b = QPushButton("Custom")
        b.setObjectName("Chip")
        b.setCheckable(True)
        b.setMinimumHeight(44)
        b.setProperty("m", 0)
        self.focus_group.addButton(b)
        chips.addWidget(b)
        if not self.focus_group.checkedButton():
            b.setChecked(True)
        self.custom = QSpinBox()
        self.custom.setRange(5, 300)
        self.custom.setSuffix(" min")
        self.custom.setValue(self.choice["focus_minutes"])
        chips.addWidget(self.custom)
        p.layout().addLayout(chips)
        p.layout().addStretch(1)

        # 6 the browser extension (required: website detection and safe tab closing need it)
        p = self._page("Connect your browser",
                       "FocusBuddy needs its small Chrome extension to know which website you're on and to close "
                       "distracting tabs safely. It takes about a minute - and it only talks to this app, on this "
                       "computer.")
        self.ext_page_index = self.stack.count() - 1
        self.ext = ExtensionSteps(app)
        self.ext.connected_changed.connect(lambda _on: self._update_nav())
        p.layout().addWidget(self.ext)
        p.layout().addStretch(1)

        # 7 finish
        p = self._page("Your companion is ready.", "")
        self.final_preview = CharacterPreview(get_character(self.choice["character"]), 180, S.CELEBRATING)
        p.layout().addWidget(self.final_preview, 0, Qt.AlignHCenter)
        self.final_text = label("", "Muted", wrap=True)
        self.final_text.setAlignment(Qt.AlignCenter)
        p.layout().addWidget(self.final_text)
        p.layout().addStretch(1)

        for _ in range(self.stack.count()):
            d = QLabel()
            d.setFixedSize(26, 6)
            self.dots.addWidget(d)
            self.dot_widgets.append(d)
        self.dots.addStretch(1)
        nav = QHBoxLayout()
        self.later = QPushButton("Set up the extension later")
        self.later.setObjectName("Ghost")
        self.later.setCursor(Qt.PointingHandCursor)
        self.later.clicked.connect(self._skip_extension)
        self.back = QPushButton("Back")
        self.back.clicked.connect(lambda: self._go(self.stack.currentIndex() - 1))
        self.next = QPushButton("Next")
        self.next.setObjectName("Primary")
        self.next.setMinimumWidth(130)
        self.next.clicked.connect(self._next)
        nav.addWidget(self.back)
        nav.addStretch(1)
        nav.addWidget(self.later)
        nav.addWidget(self.next)
        root.addLayout(nav)
        self._go(0)

    def _page(self, title: str, subtitle: str) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)
        t = label(title, "H1")
        v.addWidget(t)
        if subtitle:
            v.addWidget(label(subtitle, "Muted", wrap=True))
        v.addSpacing(6)
        self.stack.addWidget(w)
        return w

    def _choices(self, page, key, options, stretch: bool = True):
        row = QHBoxLayout()
        row.setSpacing(12)
        group = QButtonGroup(self)
        for value, title, desc, emoji in options:
            b = choice_card(title, desc, emoji)
            group.addButton(b)
            b.setChecked(self.choice[key] == value)
            b.clicked.connect(lambda _=False, v=value: self.choice.__setitem__(key, v))
            row.addWidget(b)
        page.layout().addLayout(row)
        if stretch:
            page.layout().addStretch(1)

    def _go(self, i: int):
        i = max(0, min(self.stack.count() - 1, i))
        self.stack.setCurrentIndex(i)
        for k, d in enumerate(self.dot_widgets):
            themed(d, lambda on=k <= i: f"background:{C['primary'] if on else C['border']}; border-radius:3px;")
        self.back.setVisible(i > 0)
        last = i == self.stack.count() - 1
        self.next.setText("Start FocusBuddy" if last else "Next")
        self._update_nav()
        if last:
            spec = get_character(self.choice["character"])
            self.final_preview.set_character(spec)
            mode = {"warn": "point out distractions", "ask": "ask before closing distractions",
                    "auto": "auto-close sites you mark, after a countdown"}[self.choice["blocking_mode"]]
            focus = {"auto_close": "In focus sessions I close distractions after a short countdown.",
                     "instant": "In focus sessions distracting sites are blocked instantly.",
                     "same": ""}[self.choice["focus_enforcement"]]
            self.final_text.setText(f"{spec.emoji} {spec.default_name} will watch for distractions "
                                    f"({self.choice['sensitivity']}) and {mode}. {focus} "
                                    f"Default focus session: {self._minutes()} min.")

    def _update_nav(self) -> None:
        """On the extension step, Next waits for the extension to connect."""
        on_ext = self.stack.currentIndex() == self.ext_page_index
        ok = not on_ext or self.ext.connected or self.ext.skipped or not self.ext.available
        self.next.setEnabled(ok)
        self.next.setToolTip("" if ok else "Next unlocks as soon as the extension connects")
        # the setup waits for the extension; a way on appears only when it can't work here (no
        # Chrome / Edge / Brave, the connection port busy) or after a few minutes of trying
        import time as _t
        if on_ext and not hasattr(self, "_ext_since"):
            self._ext_since = _t.monotonic()
            QTimer.singleShot(150_000, self._update_nav)
        stuck = not self.ext.has_browser or not self.ext.available or \
            (hasattr(self, "_ext_since") and _t.monotonic() - self._ext_since > 145)
        self.later.setVisible(on_ext and not self.ext.connected and stuck)

    def _skip_extension(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        r = QMessageBox.question(
            self, "Set up the extension later?",
            "Without the extension FocusBuddy can only guess websites from window titles, and it can't close a "
            "single distracting tab.\n\nYou can install it any time from Settings → Browser integration.\n\n"
            "Continue without it for now?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if r == QMessageBox.Yes:
            self.ext.skipped = True
            self._go(self.stack.currentIndex() + 1)

    def _minutes(self) -> int:
        b = self.focus_group.checkedButton()
        m = b.property("m") if b else 25
        return m or self.custom.value()

    def _next(self):
        if not self.next.isEnabled():
            return
        if self.stack.currentIndex() < self.stack.count() - 1:
            self._go(self.stack.currentIndex() + 1)
            return
        s = self.app.settings
        s.set("user_name", self.name.text().strip())
        s.apply_sensitivity_preset(self.choice["sensitivity"])
        s.set("blocking_mode", self.choice["blocking_mode"])
        s.set("focus_enforcement", self.choice["focus_enforcement"])
        s.set("focus_minutes", self._minutes())
        s.set("first_run_done", True)
        s.set("extension_prompted", True)
        self.app.set_character(self.choice["character"])
        self.accept()
