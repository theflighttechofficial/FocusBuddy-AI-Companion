"""Information pages of the dashboard: Privacy, Help, FAQ and About."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from .. import __version__
from .dashboard import Page
from .motion import GlassCard
from .theme import C, themed
from .widgets import Card, Dot, label


def _para(text: str, obj: str = "Muted") -> QLabel:
    lb = label(text, obj, wrap=True)
    lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lb


def _hero(title: str, subtitle: str, emoji: str) -> GlassCard:
    """A vivid banner at the top of an information page."""
    hero = GlassCard(hero=True, radius=28, hover=False)
    h = QHBoxLayout(hero)
    m = GlassCard.MARGIN
    h.setContentsMargins(m + 30, m + 24, m + 30, m + 24)
    col = QVBoxLayout()
    col.setSpacing(6)
    col.addWidget(label(title, "H1"))
    col.addWidget(label(subtitle, "Muted", wrap=True))
    h.addLayout(col, 1)
    big = QLabel(emoji)
    big.setStyleSheet("font-size: 64px;")
    h.addWidget(big, 0, Qt.AlignVCenter)
    return hero


def _row(text: str, color: str = "primary", sub: str = "") -> QHBoxLayout:
    r = QHBoxLayout()
    r.setSpacing(12)
    r.addWidget(Dot(color, 10), 0, Qt.AlignTop)
    col = QVBoxLayout()
    col.setSpacing(1)
    t = label(text, wrap=True)
    t.setStyleSheet("font-size:14px; font-weight:600;")
    col.addWidget(t)
    if sub:
        col.addWidget(_para(sub, "Faint"))
    r.addLayout(col, 1)
    return r


def _promise(emoji: str, title: str, text: str, color: str) -> GlassCard:
    c = GlassCard(wash=color, hover=True)
    v = QVBoxLayout(c)
    m = GlassCard.MARGIN
    v.setContentsMargins(m + 20, m + 16, m + 20, m + 18)
    v.setSpacing(6)
    e = QLabel(emoji)
    e.setStyleSheet("font-size: 30px;")
    v.addWidget(e)
    v.addWidget(label(title, "H3"))
    v.addWidget(_para(text))
    v.addStretch(1)
    return c


# =========================================================================== Privacy
class PrivacyPage(Page):
    title = "Privacy"

    def __init__(self, app):
        super().__init__(app)
        self.v.addWidget(_hero("Your privacy", "Everything FocusBuddy learns about your day stays on this computer. "
                               "No cloud, no account server, no tracking.", "🔒"))
        row = QHBoxLayout()
        row.setSpacing(0)
        row.addWidget(_promise("💻", "Stays on this PC", "History and settings live in a local database in your data "
                               "folder. Nothing is uploaded - there is no server to upload to.", "focus"))
        row.addWidget(_promise("🙈", "No snooping", "No keystrokes, screenshots, passwords, messages or files - "
                               "ever. Only which app or website is in front.", "primary"))
        row.addWidget(_promise("🎛", "You're in control", "Pause monitoring any time, export or delete your history, "
                               "and choose what may be closed.", "sun"))
        self.v.addLayout(row)

        stored = Card("What FocusBuddy keeps", "Only what it needs to notice distractions and draw your statistics.")
        for text, sub in (("App names and website domains", "e.g. “chrome.exe” and “youtube.com”"),
                          ("How long, a distraction score and what happened", "Noticed, closed, snoozed, kept open ..."),
                          ("Your focus sessions and settings", "Durations, goals, your companion and its memory"),
                          ("Your tasks, hourly totals and chats with your companion",
                           "For the coach, the reports and the chat - delete them any time below")):
            stored.add(_row(text, "focus", sub))
        self.v.addWidget(stored)
        never = Card("What it never collects")
        for text in ("Keystrokes - only the time since your last input, to know whether you're away",
                     "Screenshots or screen recordings", "Passwords, messages or the contents of your files",
                     "Window titles and full web addresses (they are used for a moment, never saved)"):
            never.add(_row(text, "danger"))
        self.v.addWidget(never)

        ext = Card("The browser extension", "Exact website detection and safe tab closing.")
        ext.add(_para("The extension shares only the active tab's address and title, and only with FocusBuddy on "
                      "this computer (127.0.0.1) - never with a website or a server. Private / incognito tabs are "
                      "not reported. It closes a tab only when you approved it or a focus session's rule says so."))
        self.v.addWidget(ext)

        ai = Card("The coach and the chat", "Smart features that never leave this computer.")
        ai.add(_para("The focus coach and the evening summary are worked out on this PC from your own history. "
                     "The chat understands app commands by itself. If you run Ollama (a free local AI program), "
                     "free conversation goes to it at 127.0.0.1 - FocusBuddy refuses any other address - with a "
                     "one-line summary of your day for context. Without Ollama, built-in replies are used. "
                     "Nothing goes to an online AI service. Settings → Smart assistant turns each part off."))
        self.v.addWidget(ai)

        acc = Card("Your account", "Log-in, PIN and lock.")
        acc.add(_para("Your e-mail and password exist only on this device: the password is stored as a salted "
                      "PBKDF2 hash, never in plain text, and nothing is sent anywhere - which is also why there is "
                      "no e-mail reset. A PIN can lock the window (Settings → Account); it doesn't encrypt files."))
        self.v.addWidget(acc)

        data = Card("Your data, your choice")
        btns = QHBoxLayout()
        b1 = QPushButton("Export history (CSV)")
        b1.clicked.connect(self._export)
        b2 = QPushButton("Open data folder")
        b2.clicked.connect(self._open_data)
        b3 = QPushButton("Delete all history")
        b3.setObjectName("Danger")
        b3.clicked.connect(self._clear)
        for b in (b1, b2, b3):
            b.setCursor(Qt.PointingHandCursor)
            btns.addWidget(b)
        btns.addStretch(1)
        data.add(btns)
        data.add(_para("To remove everything, uninstall FocusBuddy and delete its data folder.", "Faint"))
        self.v.addWidget(data)
        self.v.addStretch(1)

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export history", "focusbuddy_history.csv", "CSV (*.csv)")
        if path:
            n = self.app.db.export_history_csv(path)
            QMessageBox.information(self, "Exported", f"Saved {n} events to\n{path}")

    def _open_data(self):
        from .settings_pages import open_path
        open_path(str(self.app.data_dir))

    def _clear(self):
        if QMessageBox.question(self, "Delete all history",
                                "Delete all distraction history, sessions and statistics?\nThis cannot be undone.") \
                == QMessageBox.Yes:
            self.app.db.clear_history()


# =========================================================================== Help
SHORTCUTS = (("Ctrl + 1 … 9", "Go to a page"), ("Ctrl + K", "Search pages and settings"),
             ("Ctrl + H", "Hide / show your companion"), ("Ctrl + L", "Lock (with a PIN)"),
             ("Ctrl + ,", "Settings"), ("F1", "This help"), ("Ctrl + W", "Close the window (FocusBuddy keeps "
                                                                         "running in the tray)"))
COMPANION_TIPS = (("Click", "Say hi - it reacts, and sometimes asks you something"),
                  ("Double-click", "Open this dashboard"),
                  ("Drag", "Pick it up and put it somewhere else"),
                  ("Stroke back and forth", "Pet it - it loves that"),
                  ("Right-click", "Its menu: focus sessions, care (treats, play, pets), size, nap, hide, quit ..."),
                  ("Ctrl + mouse wheel", "Make it bigger or smaller, up to a quarter of the screen"))
TROUBLE = (
    ("My companion disappeared",
     "If you hid it, click “Show …” on Home, press Ctrl+H here, or right-click the tray icon → Show Companion. It "
     "also steps aside while a full-screen app or game runs and comes back afterwards. It may simply be napping "
     "off-screen: tray icon → Wake Up Companion (it also naps while you're away - Settings → Routines)."),
    ("The browser extension won't connect",
     "Keep the browser open, then check Settings → Browser integration: the status turns green within a few "
     "seconds of the extension loading. Make sure “Developer mode” is on and the folder you loaded is FocusBuddy's "
     "browser_extension folder."),
    ("Websites aren't recognised",
     "Without the extension FocusBuddy guesses sites from window titles. Install the extension for exact "
     "addresses, and add your own sites in Sites & Apps."),
    ("The companion feels slow or choppy",
     "Settings → Companion: choose a lower 3D quality or a 30 FPS frame rate, or turn on adaptive quality. Very "
     "large sizes cost more."),
    ("FocusBuddy doesn't start with Windows",
     "Turn on Settings → General → Start with Windows. It then waits quietly in the tray; clicking FocusBuddy "
     "always opens this dashboard."),
)


class HelpPage(Page):
    title = "Help"

    def __init__(self, app):
        super().__init__(app)
        self.v.addWidget(_hero("Help", "Getting started, shortcuts and fixes for common problems.", "💡"))
        start = Card("Getting started")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        steps = (("1", "Connect your browser", "Install the Chrome extension so FocusBuddy sees exact websites."),
                 ("2", "Tell it what distracts you", "Sites & Apps: mark sites as distracting, and which may be "
                                                     "closed automatically."),
                 ("3", "Start a focus session", "Pick a length on Focus. Your companion guards the session and "
                                                "cheers when you finish."),
                 ("4", "Check your progress", "Statistics and History show your focus time and what pulled you away."))
        for i, (n, t, d) in enumerate(steps):
            num = QLabel(n)
            num.setFixedSize(30, 30)
            num.setAlignment(Qt.AlignCenter)
            themed(num, lambda: f"background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 {C['primary']}, "
                                f"stop:1 {C['primary_grad']}); color: {C['on_accent']}; border-radius: 15px; "
                                "font-weight: 800;")
            col = QVBoxLayout()
            col.setSpacing(1)
            tl = label(t)
            tl.setStyleSheet("font-size:14px; font-weight:700;")
            col.addWidget(tl)
            col.addWidget(_para(d, "Faint"))
            cell = QHBoxLayout()
            cell.setSpacing(12)
            cell.addWidget(num, 0, Qt.AlignTop)
            cell.addLayout(col, 1)
            grid.addLayout(cell, i // 2, i % 2)
        start.add(grid)
        btns = QHBoxLayout()
        b = QPushButton("Connect the browser")
        b.setObjectName("Primary")
        b.clicked.connect(lambda: self.window().show_page("settings", anchor="Browser integration")
                          if hasattr(self.window(), "show_page") else None)
        b2 = QPushButton("Run the setup again")
        b2.clicked.connect(lambda: self.app.show_wizard() if hasattr(self.app, "show_wizard") else None)
        for x in (b, b2):
            x.setCursor(Qt.PointingHandCursor)
            btns.addWidget(x)
        btns.addStretch(1)
        start.add(btns)
        self.v.addWidget(start)

        row = QHBoxLayout()
        row.setSpacing(0)
        comp = Card("Your companion", "How to play with it on the desktop.")
        comp.add(self._table(COMPANION_TIPS))
        row.addWidget(comp, 1)
        keys = Card("Keyboard shortcuts", "In this window.")
        keys.add(self._table(SHORTCUTS))
        row.addWidget(keys, 1)
        self.v.addLayout(row)

        fix = Card("Troubleshooting")
        for title, text in TROUBLE:
            t = label(title, "H3")
            t.setProperty("search_title", title)
            fix.add(t)
            fix.add(_para(text))
        logs = QPushButton("Open the log folder")
        logs.setCursor(Qt.PointingHandCursor)
        logs.clicked.connect(self._logs)
        fix.add(logs)
        self.v.addWidget(fix)
        self.v.addStretch(1)

    @staticmethod
    def _table(rows) -> QWidget:
        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(14)
        g.setVerticalSpacing(8)
        for i, (k, d) in enumerate(rows):
            key = QLabel(k)
            themed(key, lambda: f"background:{C['primary_soft']}; color:{C['primary_text']}; border-radius:9px; "
                                "padding:4px 10px; font-weight:700; font-size:12px;")
            g.addWidget(key, i, 0, Qt.AlignLeft | Qt.AlignTop)
            g.addWidget(_para(d), i, 1)
        g.setColumnStretch(1, 1)
        return w

    def _logs(self):
        from .settings_pages import open_path
        open_path(str(self.app.data_dir / "logs"))


# =========================================================================== FAQ
FAQ = (
    ("Does FocusBuddy record my screen or what I type?",
     "No. It never takes screenshots and never records keystrokes. It looks at which app (and, with the "
     "extension, which website) is in front, and how long you've been away from the keyboard and mouse - "
     "nothing else. See the Privacy page."),
    ("Why does it need the Chrome extension?",
     "Windows only tells an app the browser's window title. The extension tells FocusBuddy the exact website of "
     "the active tab, and lets it close one distracting tab instead of the whole browser. It talks only to "
     "FocusBuddy on this computer."),
    ("Can FocusBuddy close something important?",
     "It only closes what you allowed: sites and apps you marked in Sites & Apps, after a countdown you can "
     "cancel, or during a focus session you started. System apps are never touched, and apps get the normal "
     "“save your work?” chance first."),
    ("How do I bring my companion back after hiding it?",
     "Click “Show …” on the Home page, press Ctrl+H in this window, use Settings → Companion → Show the "
     "companion on the desktop, or right-click the tray icon → Show Companion."),
    ("Can my companion be bigger?",
     "Yes: right-click it → Size → Giant makes it about a quarter of your screen's height. You can also hold "
     "Ctrl and scroll over it, or drag the size slider in Settings → Companion."),
    ("It moves a bit choppily on my laptop.",
     "Lower Settings → Companion → 3D quality, choose 30 FPS, or keep adaptive quality on - FocusBuddy then "
     "lowers the quality itself when your PC is busy."),
    ("Is my account online? I forgot my password.",
     "Your account lives only on this device - nothing is sent anywhere, so there's no e-mail reset. If your "
     "profile has a PIN, click “Forgot password?” on the log-in and unlock it with the PIN to choose a new one."),
    ("How does a focus session work?",
     "Pick a length on the Focus page. While it runs, distractions are noticed faster and (if you chose so) "
     "closed after a short countdown. When the timer ends you get a break, and your companion celebrates."),
    ("How does it decide what's a distraction?",
     "From your rules in Sites & Apps, built-in categories (video, social media, games ...) and how long you "
     "stay. It learns a little from your answers too: keep a site open a few times and it relaxes."),
    ("Can I talk to my companion?",
     "Yes - open Chat. Ask it to “start a 45 minute focus on the essay”, “add task: read chapter 3”, “how did I "
     "do today?” or “when do I focus best?” and it does it. For free conversation, install Ollama "
     "(ollama.com) and a model such as llama3.2; FocusBuddy finds it on this PC by itself."),
    ("What does the coach look at?",
     "Only FocusBuddy's own history on this PC: which hours you usually focus or drift, how your sessions end "
     "and how quickly you come back from a distraction. It suggests; it never starts anything on its own."),
    ("Where does my companion go when it naps?",
     "Off the edge of your screen - it yawns, stretches and strolls out to sleep out of sight. Wake it from the "
     "tray icon or its menu (\u201cWake up\u201d), or just come back to your PC if it napped while you were away."),
    ("How do treats and the bond work?",
     "Every 10 focused minutes earns a treat. Feed, play and pet on the Care page (or right-click the companion "
     "\u2192 Care). Focus, finished session goals and a little care each day grow your bond level; streaks earn "
     "badges. Needs only drift while FocusBuddy runs and never drop very low - no guilt, only nudges."),
    ("Does it work offline?", "Completely. FocusBuddy doesn't need the internet at all."),
    ("Which browsers are supported?",
     "Chrome, Edge and Brave with the extension; Firefox with the extension loaded from about:debugging. Other "
     "browsers work with window titles only."),
    ("Can I rename my companion or change its personality?",
     "Yes - Settings → Companion: pick another character, give it a name, and choose how chatty and lively it is."),
    ("How do I uninstall FocusBuddy and remove my data?",
     "Uninstall it from Windows Settings → Apps. Your history stays in the data folder (Privacy → Open data "
     "folder) until you delete that folder too."),
)


class FaqItem(QFrame):
    """A question that opens to show its answer."""

    def __init__(self, q: str, a: str):
        super().__init__()
        self.setObjectName("FaqItem")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.btn = QPushButton(f"＋   {q}")
        self.btn.setObjectName("Faq")
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setCheckable(True)
        self.q = q
        self.ans = _para(a)
        self.ans.setContentsMargins(40, 0, 16, 14)
        self.ans.setVisible(False)
        self.ans.setProperty("search_title", q)
        self.btn.toggled.connect(self._toggle)
        v.addWidget(self.btn)
        v.addWidget(self.ans)

    def _toggle(self, on: bool) -> None:
        self.ans.setVisible(on)
        self.btn.setText(("－   " if on else "＋   ") + self.q)


class FaqPage(Page):
    title = "FAQ"

    def __init__(self, app):
        super().__init__(app)
        self.v.addWidget(_hero("Questions & answers", "The things people ask most. Click a question to open it.",
                               "💬"))
        c = Card()
        self.items = []
        for i, (q, a) in enumerate(FAQ):
            if i:
                sep = QFrame()
                sep.setObjectName("NavSep")
                sep.setFixedHeight(1)
                c.add(sep)
            it = FaqItem(q, a)
            self.items.append(it)
            c.add(it)
        self.items[0].btn.setChecked(True)
        self.v.addWidget(c)
        self.v.addStretch(1)


# =========================================================================== About
WHATS_NEW = (
    "Smoother animals: paws stay planted, no more skating, sinking or stretched-out walks",
    "Natural gaits - calmer walks, real trots and gallops, tails that stream out behind",
    "Naps happen off-screen: your companion strolls away to sleep and comes back yawning",
    "Companion care: treats you earn by focusing, play and pets, bond levels and badges",
    "Morning check-in, evening wind-down and naps while you're away from the PC",
    "Session goals, ambient focus sounds (rain, café, brown noise ...) and Deep focus",
)


class AboutPage(Page):
    title = "About"

    def __init__(self, app):
        super().__init__(app)
        from .about import CREDITS
        hero = _hero("FocusBuddy", f"Version {__version__}  ·  a tiny friend on your desktop that helps you stay "
                     "focused - kindly, not bossily.", "🐾")
        self.v.addWidget(hero)
        row = QHBoxLayout()
        row.setSpacing(0)
        new = Card(f"What's new in {__version__}")
        for t in WHATS_NEW:
            new.add(_row(t, "primary"))
        row.addWidget(new, 3)
        made = Card("Made by")
        made.add(_para("The Flight Tech Labs", "H3"))
        made.add(_para("Built with Python, Qt for Python (PySide6) and NumPy. Runs entirely on your PC."))
        made.lay.addStretch(1)
        row.addWidget(made, 2)
        self.v.addLayout(row)
        cr = Card("Credits & licences")
        cr.add(_para(CREDITS))
        cr.add(_para("The Hunyuan3D-generated models are used under the Tencent Hunyuan 3D Community License "
                     "(its territory and use limits apply); see each character's LICENSE.md in the assets folder.",
                     "Faint"))
        self.v.addWidget(cr)
        self.v.addStretch(1)
