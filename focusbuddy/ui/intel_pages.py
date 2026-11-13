"""2.0 pages: Coach (your patterns and today's plan), Tasks, Chat (talk to your companion) and Reports."""

from __future__ import annotations

import datetime as dt
import threading
import time

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPainterPath
from PySide6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from ..intel.reports import day_report, mins, week_report
from .dashboard import Page
from .motion import GlassCard
from .theme import C, tok
from .widgets import CH_DISTRACT, CH_FOCUS, Card, ColumnChart, Dot, ProgressRing, StatTile, font, label

TONE = {"good": "focus", "warn": "distract", "info": "primary"}


def _clear(layout) -> None:
    while layout.count():
        it = layout.takeAt(0)
        w = it.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
        elif it.layout() is not None:
            _clear(it.layout())


# =========================================================================== Coach
class HourHeat(QWidget):
    """24 cells, one per hour: green where you focus, coral where you drift; the current hour ringed."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(86)
        self.focus = [0.0] * 24
        self.distract = [0.0] * 24
        self.setMouseTracking(True)

    def set_data(self, focus, distract):
        self.focus, self.distract = list(focus), list(distract)
        self.update()

    def paintEvent(self, _ev):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        w = self.width() - 8
        cw = w / 24.0
        top = 6
        ch = self.height() - 30
        mx = max(1.0, max(f + d for f, d in zip(self.focus, self.distract)))
        now_h = dt.datetime.now().hour
        fc, dc = QColor(tok(CH_FOCUS)), QColor(tok(CH_DISTRACT))
        for h in range(24):
            f, d = self.focus[h], self.distract[h]
            tot = f + d
            x = 4 + h * cw
            r = QRectF(x + 2, top, cw - 4, ch)
            base = QColor(C["surface_alt"])
            g.setPen(Qt.NoPen)
            g.setBrush(base)
            g.drawRoundedRect(r, 6, 6)
            if tot > 1:
                share = d / tot
                col = QColor(int(fc.red() * (1 - share) + dc.red() * share), int(fc.green() * (1 - share) + dc.green() * share),
                             int(fc.blue() * (1 - share) + dc.blue() * share))
                col.setAlpha(int(70 + 185 * min(1.0, tot / mx)))
                g.setBrush(col)
                g.drawRoundedRect(r, 6, 6)
            if h == now_h:
                g.setBrush(Qt.NoBrush)
                pen = QColor(tok("primary"))
                g.setPen(pen)
                g.drawRoundedRect(r.adjusted(-1, -1, 1, 1), 7, 7)
            if h % 3 == 0:
                g.setPen(QColor(C["muted"]))
                g.setFont(font(10))
                lbl = f"{(h % 12) or 12}{'a' if h < 12 else 'p'}"
                g.drawText(QRectF(x - 6, top + ch + 4, cw + 12, 16), Qt.AlignCenter, lbl)
        g.end()


class CoachPage(Page):
    title = "Coach"

    def __init__(self, app):
        super().__init__(app)
        self.header("Focus Coach", "Learned from your own history, on this computer: your best hours, the moments you "
                                   "drift, and a plan for today.")
        top = QHBoxLayout()
        top.setSpacing(0)
        self.plan_card = Card("Today's plan")
        self.plan_box = QVBoxLayout()
        self.plan_box.setSpacing(8)
        self.plan_card.add(self.plan_box)
        prow = QHBoxLayout()
        self.btn_focus = QPushButton("Start a focus session")
        self.btn_focus.setObjectName("Primary")
        self.btn_focus.clicked.connect(lambda: self.app.start_focus(int(self.app.settings["focus_minutes"])))
        prow.addWidget(self.btn_focus)
        prow.addStretch(1)
        self.plan_card.add(prow)
        top.addWidget(self.plan_card, 3)
        self.risk_card = Card("Right now")
        self.risk_ring = ProgressRing(130, 11, CH_DISTRACT)
        self.risk_card.add(self.risk_ring)
        self.risk_lbl = label("", "Muted", wrap=True)
        self.risk_card.add(self.risk_lbl)
        top.addWidget(self.risk_card, 2)
        self.v.addLayout(top)
        heat = Card("Your day, hour by hour", "Average over the last four weeks: violet = focused, coral = distracted, "
                                             "brighter = more time. The ringed hour is now.")
        self.heat = HourHeat()
        heat.add(self.heat)
        self.v.addWidget(heat)
        self.ins_title = label("What I've noticed", "H2")
        self.ins_title.setContentsMargins(16, 10, 0, 0)
        self.v.addWidget(self.ins_title)
        self.grid = QGridLayout()
        self.grid.setSpacing(0)
        self.v.addLayout(self.grid)
        self.v.addStretch(1)

    def refresh(self):
        coach = self.app.coach
        try:
            self.app.stats.flush()
        except Exception:  # noqa: BLE001
            pass
        coach._cache = None
        p = coach.profile()
        self.heat.set_data(p.focus, p.distract)
        _clear(self.plan_box)
        for line in coach.plan_today(self.app.db.tasks(include_done=False)):
            row = QHBoxLayout()
            row.addWidget(Dot("primary", 9), 0, Qt.AlignTop)
            row.addWidget(label(line, wrap=True), 1)
            self.plan_box.addLayout(row)
        r = coach.risk_now()
        self.risk_ring.set(r, f"{r * 100:.0f}%", "drift risk")
        hour = dt.datetime.now().strftime("%I %p").lstrip("0")
        if p.hours_tracked < 3:
            self.risk_lbl.setText("Still learning - the risk gauge wakes up after a few hours of use.")
        elif r >= 0.55:
            self.risk_lbl.setText(f"Around {hour} you often drift. A focus session now is your best defence.")
        elif r >= 0.3:
            self.risk_lbl.setText(f"{hour} is a so-so hour for you. Keep one tab, one task.")
        else:
            self.risk_lbl.setText(f"{hour} is usually a good hour for you. Make it count!")
        _clear(self.grid)
        for i, ins in enumerate(coach.insights()):
            c = GlassCard(wash=TONE.get(ins.tone, "primary"))
            lay = QVBoxLayout(c)
            m = GlassCard.MARGIN
            lay.setContentsMargins(m + 18, m + 14, m + 18, m + 16)
            lay.setSpacing(6)
            head = QHBoxLayout()
            head.addWidget(Dot(TONE.get(ins.tone, "primary"), 10))
            head.addWidget(label(ins.title, "H3"), 1)
            lay.addLayout(head)
            lay.addWidget(label(ins.text, "Muted", wrap=True))
            lay.addStretch(1)
            self.grid.addWidget(c, i // 2, i % 2)


# =========================================================================== Tasks
class TaskRow(QFrame):
    def __init__(self, page, t: dict):
        super().__init__()
        self.page, self.t = page, t
        self.setObjectName("TaskRow")
        done = bool(t.get("done_at"))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(10)
        chk = QPushButton("✓" if done else "")
        chk.setFixedSize(28, 28)
        chk.setCursor(Qt.PointingHandCursor)
        chk.setToolTip("Mark as not done" if done else "Mark as done")
        chk.setStyleSheet(f"QPushButton {{ border-radius: 14px; border: 2px solid {tok('focus')}; "
                          f"background: {tok('focus') if done else 'transparent'}; color: white; font-weight: 800; "
                          f"padding: 0; }}")
        chk.clicked.connect(self._toggle)
        lay.addWidget(chk)
        col = QVBoxLayout()
        col.setSpacing(2)
        title = label(t["title"], "H3" if not done else "Faint", wrap=True)
        if done:
            title.setStyleSheet("text-decoration: line-through;")
        col.addWidget(title)
        spent = float(t.get("spent_seconds") or 0)
        est = int(t.get("estimate_minutes") or 25)
        pr = {2: "High priority · ", 0: "Low priority · "}.get(int(t.get("priority", 1)), "")
        sub = f"{pr}{mins(spent)} of ~{est}m" + (f" · {t['sessions']} session{'s' if t['sessions'] != 1 else ''}"
                                                 if t.get("sessions") else "")
        if t.get("project"):
            sub = f"{t['project']} · " + sub
        col.addWidget(label(sub, "Faint"))
        bar = _Bar(min(1.0, spent / max(60.0, est * 60.0)))
        col.addWidget(bar)
        lay.addLayout(col, 1)
        if not done:
            f = QPushButton("▶  Focus")
            f.setObjectName("Primary")
            f.setToolTip("Start a focus session on this task")
            f.clicked.connect(self._focus)
            lay.addWidget(f)
        rm = QPushButton("✕")
        rm.setObjectName("Ghost")
        rm.setFixedWidth(36)
        rm.setToolTip("Delete task")
        rm.clicked.connect(self._delete)
        lay.addWidget(rm)

    def _toggle(self):
        if self.t.get("done_at"):
            self.page.app.db.update_task(self.t["id"], done_at=None)
            self.page.app.tasks_changed()
        else:
            self.page.app.complete_task(self.t["id"])

    def _focus(self):
        est = int(self.t.get("estimate_minutes") or self.page.app.settings["focus_minutes"])
        m = max(5, min(int(self.page.app.settings["focus_minutes"]) if est > 60 else est, 90))
        self.page.app.start_focus(m, task_id=self.t["id"])
        dash = self.window()
        if hasattr(dash, "toast"):
            dash.toast(f"Focusing on “{self.t['title']}” for {m} min", "ok")

    def _delete(self):
        self.page.app.db.delete_task(self.t["id"])
        self.page.app.tasks_changed()


class _Bar(QWidget):
    def __init__(self, v: float):
        super().__init__()
        self.v = v
        self.setFixedHeight(6)

    def paintEvent(self, _):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        g.setPen(Qt.NoPen)
        g.setBrush(QColor(C["surface_alt"]))
        g.drawRoundedRect(r, 3, 3)
        if self.v > 0:
            g.setBrush(QColor(tok("focus")))
            g.drawRoundedRect(QRectF(r.left(), r.top(), r.width() * self.v, r.height()), 3, 3)
        g.end()


class TasksPage(Page):
    title = "Tasks"

    def __init__(self, app):
        super().__init__(app)
        self.header("Tasks & Goals", "Plan the day, then focus on one thing at a time. Time in focus sessions is "
                                     "counted on the task you pick.")
        tiles = QHBoxLayout()
        tiles.setSpacing(0)
        self.t_open = StatTile("Open tasks", "–", "on your list", "primary")
        self.t_done = StatTile("Done today", "–", "ticked off", "focus")
        self.t_time = StatTile("Time on tasks", "–", "in focus sessions today", "sun")
        for t in (self.t_open, self.t_done, self.t_time):
            tiles.addWidget(t)
        self.v.addLayout(tiles)
        add = Card("Add a task")
        row = QHBoxLayout()
        self.inp = QLineEdit()
        self.inp.setPlaceholderText("What do you want to get done?  e.g. Finish the report intro")
        self.inp.returnPressed.connect(self._add)
        row.addWidget(self.inp, 1)
        self.est = QSpinBox()
        self.est.setRange(5, 600)
        self.est.setSingleStep(5)
        self.est.setValue(25)
        self.est.setSuffix(" min")
        row.addWidget(self.est)
        self.prio = QComboBox()
        for v, t in ((1, "Normal"), (2, "High"), (0, "Low")):
            self.prio.addItem(t, v)
        row.addWidget(self.prio)
        self.proj = QLineEdit()
        self.proj.setPlaceholderText("Project (optional)")
        self.proj.setFixedWidth(170)
        row.addWidget(self.proj)
        b = QPushButton("Add")
        b.setObjectName("Primary")
        b.clicked.connect(self._add)
        row.addWidget(b)
        add.add(row)
        self.v.addWidget(add)
        self.list_card = Card("Your list")
        self.box = QVBoxLayout()
        self.box.setSpacing(4)
        self.list_card.add(self.box)
        self.v.addWidget(self.list_card)
        self.v.addStretch(1)
        app.tasks_listeners.append(self._changed)

    def _changed(self):
        if self.isVisible():
            self.refresh()

    def _add(self):
        title = self.inp.text().strip()
        if not title:
            return
        self.app.db.add_task(title, project=self.proj.text().strip(), estimate_minutes=self.est.value(),
                             priority=int(self.prio.currentData()))
        self.inp.clear()
        self.app.tasks_changed()
        self.refresh()

    def refresh(self):
        start = dt.datetime.combine(dt.date.today(), dt.time()).timestamp()
        ts = self.app.db.tasks(include_done=True, done_since=start)
        open_ = [t for t in ts if not t.get("done_at")]
        done = [t for t in ts if t.get("done_at")]
        spent = sum(x["focused_seconds"] for x in self.app.db.sessions_since(start)
                    if x.get("task_id") and x["status"] in ("completed", "cancelled"))
        self.t_open.set(str(len(open_)), number=len(open_), fmt=lambda v: f"{v:.0f}")
        self.t_done.set(str(len(done)), number=len(done), fmt=lambda v: f"{v:.0f}")
        self.t_time.set(mins(spent))
        _clear(self.box)
        if not ts:
            e = label("Nothing here yet. Add your first task above - or tell your companion "
                      "“add task …” in Chat.", "Muted", wrap=True)
            self.box.addWidget(e)
        for t in open_ + done:
            self.box.addWidget(TaskRow(self, t))


# =========================================================================== Chat
class _Bridge(QObject):
    answered = Signal(object)


class Bubble(QFrame):
    def __init__(self, text: str, mine: bool, meta: str = ""):
        super().__init__()
        self.mine = mine
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        lb = QLabel(text)
        lb.setWordWrap(True)
        lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lb.setStyleSheet(f"color: {'white' if mine else C['text']}; font-size: 14px;")
        fm = lb.fontMetrics()
        lb.setMinimumWidth(min(520, max(40, fm.horizontalAdvance(text) + 26)))
        lay.addWidget(lb)
        if meta:
            m = QLabel(meta)
            m.setStyleSheet(f"color: {'rgba(255,255,255,180)' if mine else C['faint']}; font-size: 11px;")
            lay.addWidget(m)
        self.setMaximumWidth(560)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)

    def paintEvent(self, _ev):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p = QPainterPath()
        p.addRoundedRect(r, 16, 16)
        if self.mine:
            from PySide6.QtGui import QLinearGradient
            lg = QLinearGradient(r.topLeft(), r.bottomRight())
            lg.setColorAt(0, QColor(tok("primary")))
            lg.setColorAt(1, QColor(tok("primary_grad")))
            g.fillPath(p, lg)
        else:
            c = QColor(C["surface"])
            c.setAlpha(235)
            g.fillPath(p, c)
            g.setPen(QColor(C["border"]))
            g.drawPath(p)
        g.end()


class ChatPage(Page):
    title = "Chat"
    SUGGEST = ("How did I do today?", "Start a 25 minute focus", "What's on my list?", "When do I focus best?",
               "What distracts me?", "I can't focus", "Tell me a joke")

    def __init__(self, app):
        super().__init__(app)
        self.bridge = _Bridge()
        self.bridge.answered.connect(self._answered)
        hdr = self.header("Chat", "Talk to your companion - it can start focus sessions, keep your tasks, and tell you "
                                  "how you're doing. Runs on this computer.")
        self.status = label("", "Pill")
        hdr.addWidget(self.status)
        clear = QPushButton("Clear")
        clear.setObjectName("Ghost")
        clear.clicked.connect(self._clear)
        hdr.addWidget(clear)
        card = Card()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setMinimumHeight(380)
        self.scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        holder = QWidget()
        holder.setObjectName("Transparent")
        self.msgs = QVBoxLayout(holder)
        self.msgs.setSpacing(10)
        self.msgs.addStretch(1)
        self.scroll.setWidget(holder)
        card.add(self.scroll, 1)
        chips = QHBoxLayout()
        chips.setSpacing(6)
        for s in self.SUGGEST:
            b = QPushButton(s)
            b.setObjectName("Chip")
            b.clicked.connect(lambda _=False, s=s: self.send(s))
            chips.addWidget(b)
        chips.addStretch(1)
        cw = QWidget()
        cw.setLayout(chips)
        cs = QScrollArea()
        cs.setWidgetResizable(True)
        cs.setFrameShape(QFrame.NoFrame)
        cs.setFixedHeight(52)
        cs.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        cs.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        cs.setWidget(cw)
        card.add(cs)
        row = QHBoxLayout()
        self.inp = QLineEdit()
        self.inp.setPlaceholderText("Say something…   e.g. “start a 45 minute focus on the essay”")
        self.inp.returnPressed.connect(lambda: self.send(self.inp.text()))
        row.addWidget(self.inp, 1)
        self.btn = QPushButton("Send")
        self.btn.setObjectName("Primary")
        self.btn.clicked.connect(lambda: self.send(self.inp.text()))
        row.addWidget(self.btn)
        card.add(row)
        self.v.addWidget(card, 1)
        self._loaded = False
        self._busy = False

    def _name(self) -> str:
        try:
            return self.app.api.companion_name()
        except Exception:  # noqa: BLE001
            return "Buddy"

    def _add(self, text: str, mine: bool, meta: str = "") -> None:
        row = QHBoxLayout()
        b = Bubble(text, mine, meta)
        if mine:
            row.addStretch(1)
            row.addWidget(b)
        else:
            row.addWidget(b)
            row.addStretch(1)
        self.msgs.insertLayout(self.msgs.count() - 1, row)
        QTimer.singleShot(30, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))

    def refresh(self):
        def upd():
            ok = self.app.settings["ai_chat_llm"] and self.app.llm.available()
            model = self.app.llm.pick_model() if ok else ""
            self.bridge.answered.emit(("status", model))
        threading.Thread(target=upd, daemon=True).start()
        if not self._loaded:
            self._loaded = True
            hist = self.app.db.chat_history(40)
            for h in hist:
                self._add(h["text"], h["role"] == "user")
            if not hist:
                self._add(f"Hi {self.app.api.user_name()}! I'm {self._name()}. Ask me anything about your focus, or "
                          "tell me what to do - “start a 25 minute focus”, “add task …”, “how did I do today?”.", False)

    def _clear(self):
        self.app.db.clear_chat()
        _clear(self.msgs)
        self.msgs.addStretch(1)
        self.app.assistant.history = []
        self._add("Fresh start! What's up?", False)

    def send(self, text: str) -> None:
        text = (text or "").strip()
        if not text or self._busy:
            return
        self.inp.clear()
        self._add(text, True)
        self.app.db.log_chat("user", text)
        r = self.app.chat(text, allow_llm=False)              # actions run here, on the UI thread
        if r.intent == "unknown" and self.app.settings["ai_chat_llm"] and self.app.llm is not None:
            self._busy = True
            self.btn.setEnabled(False)
            self.status.setText("thinking…")

            def work():
                out = None
                try:
                    if self.app.llm.available():
                        out = self.app.llm.chat(self.app.assistant._llm_messages(text))
                except Exception:  # noqa: BLE001
                    out = None
                self.bridge.answered.emit(("llm", out, r))
            threading.Thread(target=work, daemon=True).start()
            return
        self._show(r.text, r.source, r.emotion)

    def _answered(self, payload):
        kind = payload[0]
        if kind == "status":
            model = payload[1]
            self.status.setText(f"Local AI · {model}" if model else "Built-in replies")
            self.status.setToolTip("Free conversation uses your local Ollama model." if model else
                                   "Install Ollama (ollama.com) and pull a small model for free conversation; the "
                                   "built-in replies always work.")
            return
        _k, out, fallback = payload
        self._busy = False
        self.btn.setEnabled(True)
        self.refresh()
        if out:
            self.app.assistant._remember(self.inp.text() or "", out)
            self._show(out, "ollama", "HAPPY")
        else:
            self._show(fallback.text, fallback.source, fallback.emotion)

    def _show(self, text: str, source: str, emotion: str) -> None:
        self._add(text, False, meta=f"{self._name()}" + (" · local AI" if source == "ollama" else ""))
        self.app.db.log_chat("bot", text)
        # the companion on the desktop says it too, with the feeling
        try:
            from ..companion.emotion import Emotion
            self.app.companion.behavior.set_emotion(Emotion(emotion), 0.8, 5.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.app.settings["speech_bubbles"] and len(text) < 220:
                self.app.companion.say(text, duration=min(9.0, 3.0 + len(text) / 18))
        except Exception:  # noqa: BLE001
            pass


# =========================================================================== Reports
class ReportsPage(Page):
    title = "Reports"

    def __init__(self, app):
        super().__init__(app)
        hdr = self.header("Reports", "Your day and your week, in plain words.")
        copy = QPushButton("Copy summary")
        copy.clicked.connect(self._copy)
        hdr.addWidget(copy)
        row = QHBoxLayout()
        row.setSpacing(0)
        day = Card("Today")
        dl = QHBoxLayout()
        self.ring = ProgressRing(150, 12, CH_FOCUS)
        dl.addWidget(self.ring, 0, Qt.AlignTop)
        self.day_lines = QVBoxLayout()
        self.day_lines.setSpacing(6)
        dl.addLayout(self.day_lines, 1)
        day.add(dl)
        row.addWidget(day, 3)
        self.mem_card = Card("Moments")
        self.mem_box = QVBoxLayout()
        self.mem_box.setSpacing(6)
        self.mem_card.add(self.mem_box)
        row.addWidget(self.mem_card, 2)
        self.v.addLayout(row)
        week = Card("This week", "Focused vs distracted time per day")
        self.chart = ColumnChart(["Focus", "Distraction"], [CH_FOCUS, CH_DISTRACT], fmt=lambda v: f"{v:.1f}h")
        week.add(self.chart)
        self.week_lines = QVBoxLayout()
        self.week_lines.setSpacing(4)
        week.add(self.week_lines)
        self.v.addWidget(week)
        self.v.addStretch(1)
        self._text = ""

    def refresh(self):
        try:
            self.app.stats.flush()
        except Exception:  # noqa: BLE001
            pass
        d = day_report(self.app.db, coach=self.app.coach)
        w = week_report(self.app.db)
        self.ring.set(d["score"] / 100.0, f"{d['score']}%", "focus score")
        _clear(self.day_lines)
        for ln in d["lines"]:
            self.day_lines.addWidget(label(ln, wrap=True))
        if d.get("tip"):
            self.day_lines.addWidget(label("Next: " + d["tip"], "Muted", wrap=True))
        _clear(self.mem_box)
        ms = d["moments"][:8]
        if not ms:
            self.mem_box.addWidget(label("Moments like a long flow, a quick comeback or a finished task show up here.",
                                         "Faint", wrap=True))
        for m in ms:
            r = QHBoxLayout()
            r.addWidget(label(dt.datetime.fromtimestamp(m["ts"]).strftime("%H:%M"), "Faint"))
            r.addWidget(label(m["text"], wrap=True), 1)
            self.mem_box.addLayout(r)
        days = w["days"]
        cats = [dt.datetime.strptime(x["day"], "%Y-%m-%d").strftime("%a") for x in days]
        self.chart.set_data(cats, [[x["focus_seconds"] / 3600 for x in days],
                                   [x["distraction_seconds"] / 3600 for x in days]])
        _clear(self.week_lines)
        for ln in w["lines"] + [f"Streak: {w['streak']} day{'s' if w['streak'] != 1 else ''} 🔥"]:
            self.week_lines.addWidget(label(ln, "Muted", wrap=True))
        self._text = "Today\n" + "\n".join(d["lines"]) + "\n\nThis week\n" + "\n".join(w["lines"])

    def _copy(self):
        QGuiApplication.clipboard().setText(self._text)
        dash = self.window()
        if hasattr(dash, "toast"):
            dash.toast("Summary copied", "ok")
