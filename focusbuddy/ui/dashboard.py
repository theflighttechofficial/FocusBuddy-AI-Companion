"""Main dashboard window: Home, Focus, Statistics, History + settings pages."""

from __future__ import annotations

import datetime as dt

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QComboBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QScrollArea, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .. import __version__
from ..companion.characters import companion_name, get_character
from ..companion.states import CompanionState as S
from ..core.focus_manager import Phase
from .theme import C, fmt_clock, fmt_duration
from .motion import AnimatedStack, AuroraBackground, GlassCard, NavRail, Toast, fade_in
from .widgets import (CH_DISTRACT, CH_FOCUS, Card, CharacterPreview, ColumnChart, HBarChart, LineChart,
                      ProgressRing, StatTile, label)

ACTION_LABELS = {
    "closed": "Closed", "minimized": "Minimized", "redirected": "Redirected", "returned": "Returned to work",
    "left": "Left on own", "snoozed": "Snoozed", "ignored": "Ignored once", "kept_open": "Kept open",
    "failed": "Couldn't close", "switched": "Switched site", "paused": "Monitoring paused", "none": "Noticed",
}


def page_scroll(inner: QWidget) -> QScrollArea:
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setFrameShape(QFrame.NoFrame)
    sc.setWidget(inner)
    return sc


class Page(QWidget):
    title = ""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setObjectName("Page")
        self.body = QWidget()
        self.body.setObjectName("Page")
        self.v = QVBoxLayout(self.body)
        self.v.setContentsMargins(14, 4, 18, 28)
        self.v.setSpacing(2)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(page_scroll(self.body))

    def header(self, title: str, subtitle: str = "") -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(14, 6, 14, 10)
        col = QVBoxLayout()
        col.setSpacing(4)
        col.addWidget(label(title, "H1"))
        if subtitle:
            col.addWidget(label(subtitle, "Muted"))
        row.addLayout(col)
        row.addStretch(1)
        self.v.addLayout(row)
        return row

    def refresh(self):
        pass

    def tick(self):
        pass


# =========================================================================== Home
class HomePage(Page):
    title = "Home"

    def __init__(self, app):
        super().__init__(app)
        # ---- hero: greeting, status, quick actions and the companion
        hero = GlassCard(hero=True, radius=28)
        hl = QHBoxLayout(hero)
        m = GlassCard.MARGIN
        hl.setContentsMargins(m + 30, m + 24, m + 24, m + 22)
        col = QVBoxLayout()
        col.setSpacing(6)
        self.date_lbl = label("", "Eyebrow")
        col.addWidget(self.date_lbl)
        self.greet = label("", "H1")
        col.addWidget(self.greet)
        self.hero_sub = label("", "Muted", wrap=True)
        col.addWidget(self.hero_sub)
        col.addSpacing(10)
        btns = QHBoxLayout()
        btns.setSpacing(10)
        self.btn_start = QPushButton("Start Focus")
        self.btn_start.setObjectName("Primary")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.clicked.connect(lambda: self.app.start_focus(self.app.settings["focus_minutes"]))
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setMinimumHeight(40)
        self.btn_pause.clicked.connect(self._pause_resume)
        self.btn_end = QPushButton("End Session")
        self.btn_end.setObjectName("Danger")
        self.btn_end.setMinimumHeight(40)
        self.btn_end.clicked.connect(self.app.end_focus)
        self.btn_comp = QPushButton("")
        self.btn_comp.setMinimumHeight(40)
        self.btn_comp.setCursor(Qt.PointingHandCursor)
        self.btn_comp.clicked.connect(lambda: self.app.show_companion(not self.app.settings["show_companion"]))
        for b in (self.btn_start, self.btn_pause, self.btn_end, self.btn_comp):
            btns.addWidget(b)
        btns.addStretch(1)
        col.addLayout(btns)
        col.addStretch(1)
        hl.addLayout(col, 1)
        right = QVBoxLayout()
        self.status = label("●  Monitoring", "Pill")
        right.addWidget(self.status, 0, Qt.AlignRight)
        self.preview = CharacterPreview(get_character(self.app.settings["character"]), 176)
        right.addWidget(self.preview, 0, Qt.AlignRight)
        hl.addLayout(right)
        from .motion import HeroFx
        self.hero_fx = HeroFx(hero)
        self.preview.setToolTip("Poke me! (head, body or tail)")
        self.preview.setCursor(Qt.PointingHandCursor)
        self.v.addWidget(hero)

        # ---- 2.2: quick actions (one click) and sections you can drag into your own order
        from .widgets import QuickActions, ReorderColumn
        qa = QuickActions()
        st = self.app.settings
        qa.add("focus", "▶", "Focus", lambda: self.app.start_focus(st["focus_minutes"]), "Start a focus session")
        qa.add("deep", "🧠", "Deep work 50", lambda: self.app.start_focus(50), "A 50-minute deep-work session")
        qa.add("break", "☕", "Break 5", lambda: self.app.start_break(5), "A 5-minute break")
        qa.add("monitor", "⏸", "Pause monitoring", lambda: self.app.set_monitoring(not self.app.monitoring),
               "Pause or resume distraction monitoring")
        qa.add("tasks", "✅", "Add a task", lambda: self.app.show_dashboard("tasks"), "Open your task list")
        qa.add("palette", "⌘", "All commands", self._palette, "Command palette   (Ctrl+K)")
        qa.finish()
        self.quick = qa
        self.v.addWidget(qa)
        self.column = ReorderColumn()

        def wrap(layout):
            w = QWidget()
            w.setObjectName("Transparent")
            layout.setContentsMargins(0, 0, 0, 0)
            w.setLayout(layout)
            return w

        # ---- KPI tiles
        tiles = QHBoxLayout()
        tiles.setSpacing(0)
        self.t_focus = StatTile("Focus time", accent=CH_FOCUS)
        self.t_dist = StatTile("Distraction time", accent=CH_DISTRACT)
        self.t_count = StatTile("Distractions", accent="warning")
        self.t_blocked = StatTile("Blocked", accent="danger")
        self.t_recov = StatTile("Recovered time", accent="focus")
        for t in (self.t_focus, self.t_dist, self.t_count, self.t_blocked, self.t_recov):
            tiles.addWidget(t)
        self.column.add_section("tiles", wrap(tiles))

        # ---- today + session
        top = QHBoxLayout()
        top.setSpacing(0)
        today = Card("Today's Focus")
        row = QHBoxLayout()
        self.ring = ProgressRing(150, 14, CH_FOCUS)
        row.addWidget(self.ring)
        col2 = QVBoxLayout()
        self.focus_big = label("0m", "Big")
        col2.addWidget(self.focus_big)
        self.goal_lbl = label("", "Muted", wrap=True)
        col2.addWidget(self.goal_lbl)
        col2.addStretch(1)
        row.addSpacing(10)
        row.addLayout(col2, 1)
        today.add(row)
        top.addWidget(today, 3)

        sess = Card("Current Session")
        self.sess_time = label("--:--", "Timer")
        sess.add(self.sess_time)
        self.sess_bar = QProgressBar()
        self.sess_bar.setTextVisible(False)
        self.sess_bar.setFixedHeight(10)
        sess.add(self.sess_bar)
        self.sess_sub = label("No session running", "Muted")
        sess.add(self.sess_sub)
        self.comp_line = label("", wrap=True)
        self.comp_line.setStyleSheet("font-size:14px;")
        sess.add(self.comp_line)
        self.comp_state = label("", "Faint")
        sess.add(self.comp_state)
        top.addWidget(sess, 4)
        self.column.add_section("today", wrap(top))

        # ---- 2.0: up next + the coach's word + ask the companion
        smart = QHBoxLayout()
        smart.setSpacing(0)
        nxt = Card("Up next")
        self.next_lbl = label("", wrap=True)
        self.next_lbl.setStyleSheet("font-size:15px; font-weight:600;")
        nxt.add(self.next_lbl)
        self.next_sub = label("", "Faint", wrap=True)
        nxt.add(self.next_sub)
        nrow = QHBoxLayout()
        self.next_btn = QPushButton("▶  Focus on it")
        self.next_btn.setObjectName("Primary")
        self.next_btn.clicked.connect(self._focus_next)
        nrow.addWidget(self.next_btn)
        tasks_btn = QPushButton("All tasks")
        tasks_btn.clicked.connect(lambda: self.app.show_dashboard("tasks"))
        nrow.addWidget(tasks_btn)
        nrow.addStretch(1)
        nxt.add(nrow)
        smart.addWidget(nxt, 3)
        coach = Card("Coach")
        self.coach_title = label("", "H3", wrap=True)
        coach.add(self.coach_title)
        self.coach_text = label("", "Muted", wrap=True)
        coach.add(self.coach_text)
        crow = QHBoxLayout()
        cbtn = QPushButton("Open coach")
        cbtn.clicked.connect(lambda: self.app.show_dashboard("coach"))
        crow.addWidget(cbtn)
        ask = QLineEdit()
        ask.setPlaceholderText("Ask your companion…")
        ask.returnPressed.connect(lambda: self._ask(ask))
        crow.addWidget(ask, 1)
        coach.add(crow)
        smart.addWidget(coach, 4)
        self.column.add_section("smart", wrap(smart))
        self._next_task = None
        app.tasks_listeners.append(lambda: self._refresh_smart() if self.isVisible() else None)

        recent = Card("Recent Distractions", "Time spent today on distracting sites and apps")
        self.recent = HBarChart(CH_DISTRACT, fmt=lambda v: fmt_duration(v, short=True), height=180)
        self.recent.empty_text = "No distractions today. Nice!"
        recent.add(self.recent)
        self.column.add_section("recent", recent)
        self.column.set_order(list(st.get("home_order", []) or []))
        self.column.order_changed.connect(lambda keys: st.set("home_order", list(keys)))
        self.v.addWidget(self.column)
        self.v.addStretch(1)

    def _palette(self):
        d = self.app.dashboard
        if d is not None:
            d.open_palette()

    def _focus_next(self):
        t = self._next_task
        if t is None:
            self.app.show_dashboard("tasks")
            return
        est = int(t.get("estimate_minutes") or self.app.settings["focus_minutes"])
        self.app.start_focus(max(5, min(est if est <= 60 else int(self.app.settings["focus_minutes"]), 90)),
                             task_id=t["id"])

    def _ask(self, edit):
        text = edit.text().strip()
        if not text:
            return
        edit.clear()
        self.app.show_dashboard("chat")
        d = self.app.dashboard
        if d is not None:
            d.pages[d.PAGE_INDEX["chat"]].send(text)

    def _refresh_smart(self):
        ts = self.app.db.tasks(include_done=False)
        if ts:
            t = ts[0]
            self._next_task = t
            self.next_lbl.setText(t["title"])
            self.next_sub.setText(f"{t['project'] + ' · ' if t.get('project') else ''}~{t['estimate_minutes']} min"
                                  f"{' · ' + str(len(ts) - 1) + ' more on your list' if len(ts) > 1 else ''}")
            self.next_btn.setText("▶  Focus on it")
        else:
            self._next_task = None
            self.next_lbl.setText("Nothing planned yet")
            self.next_sub.setText("Add what you want to get done today - your focus time counts toward it.")
            self.next_btn.setText("Add a task")
        ins = self.app.coach.insights()
        pick = next((i for i in ins if i.kind in ("best_hours", "risky_hours", "trend")), ins[0] if ins else None)
        if pick is None:
            self.coach_title.setText("Learning your rhythm")
            self.coach_text.setText("After a few hours of use I'll show your best focus hours here.")
        else:
            self.coach_title.setText(pick.title)
            self.coach_text.setText(pick.text)

    def _pause_resume(self):
        if self.app.focus.phase == Phase.FOCUS:
            self.app.pause_focus()
        elif self.app.focus.phase == Phase.PAUSED:
            self.app.resume_focus()

    def refresh(self):
        s = self.app.settings
        name = s["user_name"].strip()
        hour = dt.datetime.now().hour
        part = "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        self.greet.setText(f"Good {part}, {name}" if name else f"Good {part}")
        self.date_lbl.setText(dt.date.today().strftime("%A, %d %B").upper())
        summ = self.app.stats.today()
        goal = max(1, s["daily_focus_goal_min"]) * 60
        self.ring.set(summ["focus_seconds"] / goal, f"{int(100 * min(1, summ['focus_seconds'] / goal))}%", "of daily goal")
        self.focus_big.setText(fmt_duration(summ["focus_seconds"], short=True) or "0m")
        self.goal_lbl.setText(f"Goal: {fmt_duration(goal, short=True)} · {summ['sessions']} sessions completed")
        dur = lambda v: fmt_duration(v, True) or "0m"          # noqa: E731
        num = lambda v: str(int(round(v)))                       # noqa: E731
        self.t_focus.set("", "productive", number=summ["focus_seconds"], fmt=dur)
        self.t_dist.set("", "on distractions", number=summ["distraction_seconds"], fmt=dur)
        self.t_count.set("", "noticed today", number=summ["distractions"], fmt=num)
        self.t_blocked.set("", f"{summ['prevented']} prevented", number=summ["blocked"], fmt=num)
        self.t_recov.set("", "after nudges", number=summ["recovered_seconds"], fmt=dur)
        pct = int(100 * min(1, summ["focus_seconds"] / goal))
        self.hero_sub.setText(f"You've focused {fmt_duration(summ['focus_seconds'], True) or '0m'} today - "
                              f"{pct}% of your goal. " + ("Keep the streak going!" if pct < 100 else "Goal reached!"))
        today = dt.date.today().strftime("%Y-%m-%d")
        self.recent.set_data(self.app.db.top_distractions(today, 5))
        spec = get_character(s["character"])
        self.preview.set_character(spec)
        try:
            self._refresh_smart()
        except Exception:  # noqa: BLE001 - a coach hiccup must not break Home
            pass
        self.tick()

    def tick(self):
        f = self.app.focus
        spec = get_character(self.app.settings["character"])
        cname = companion_name(spec, self.app.settings["companion_name"])
        kind, text = self.app.status()
        self.status.setText(f"●  {text}")
        self.status.setObjectName("Pill" if kind != "paused" else "PillPaused")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        if f.phase in (Phase.FOCUS, Phase.PAUSED):
            self.sess_time.setText(fmt_clock(f.remaining()))
            self.sess_bar.setValue(int(1000 * f.progress()))
            self.sess_bar.setRange(0, 1000)
            self.sess_sub.setText(("Paused - " if f.phase == Phase.PAUSED else "") +
                                  f"{fmt_duration(f.planned, True)} session · remaining")
            self.btn_start.hide()
            self.btn_pause.show()
            self.btn_end.show()
            self.btn_pause.setText("Resume" if f.phase == Phase.PAUSED else "Pause")
            self.comp_line.setText(f"{spec.emoji} {cname} is watching over your focus.")
            self.preview.set_state(S.FOCUS_MODE)
        elif f.phase == Phase.BREAK:
            self.sess_time.setText(fmt_clock(f.remaining()))
            self.sess_bar.setRange(0, 1000)
            self.sess_bar.setValue(int(1000 * f.progress()))
            self.sess_sub.setText("Break time - stretch and hydrate")
            self.btn_start.show()
            self.btn_start.setText("Skip break & focus")
            self.btn_pause.hide()
            self.btn_end.show()
            self.comp_line.setText(f"{spec.emoji} {cname} is playing during your break.")
            self.preview.set_state(S.PLAYING)
        else:
            self.sess_time.setText(fmt_clock(self.app.settings["focus_minutes"] * 60))
            self.sess_bar.setValue(0)
            self.sess_sub.setText("Ready when you are")
            self.btn_start.show()
            self.btn_start.setText(f"Start {self.app.settings['focus_minutes']}-min Focus")
            self.btn_pause.hide()
            self.btn_end.hide()
            self.comp_line.setText(f"{spec.emoji} {cname} is hanging out on your desktop.")
            self.preview.set_state(S.IDLE)
        shown = bool(self.app.settings["show_companion"])
        mon = bool(self.app.monitoring)
        self.quick.set_text("monitor", "⏸" if mon else "👁", "Pause monitoring" if mon else "Resume monitoring")
        busy = f.phase in (Phase.FOCUS, Phase.PAUSED, Phase.BREAK)
        for k in ("focus", "deep", "break"):
            self.quick.buttons[k].setEnabled(not busy)
        self.btn_comp.setText(f"🙈  Hide {cname}" if shown else f"👋  Show {cname}")
        self.btn_comp.setToolTip(("Hide" if shown else "Bring back") + " the desktop companion   (Ctrl+H)")
        if not shown:
            self.comp_line.setText(f"{spec.emoji} {cname} is hidden. Click “Show {cname}” to bring them back.")
            self.comp_state.setText("Currently: hidden")
            return
        self.comp_state.setText(f"Currently: {self.app.companion.state.value.replace('_', ' ').lower()}")


# =========================================================================== Focus
class FocusPage(Page):
    title = "Focus"

    def __init__(self, app):
        super().__init__(app)
        self.header("Focus Mode", "Pick a duration. Your companion guards your focus while the timer runs.")
        main = QHBoxLayout()
        main.setSpacing(0)
        left = Card()
        center = QVBoxLayout()
        center.setAlignment(Qt.AlignHCenter)
        self.ring = ProgressRing(260, 16, CH_FOCUS)
        center.addWidget(self.ring, 0, Qt.AlignHCenter)
        self.phase_lbl = label("", "H2")
        self.phase_lbl.setAlignment(Qt.AlignCenter)
        center.addWidget(self.phase_lbl)
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.start = QPushButton("Start Focus")
        self.start.setObjectName("Primary")
        self.start.setMinimumWidth(140)
        self.start.clicked.connect(self._start)
        self.pause = QPushButton("Pause")
        self.pause.clicked.connect(self._pause)
        self.end = QPushButton("End")
        self.end.setObjectName("Danger")
        self.end.clicked.connect(self.app.end_focus)
        for b in (self.start, self.pause, self.end):
            btns.addWidget(b)
        btns.addStretch(1)
        center.addLayout(btns)
        left.add(center)
        main.addWidget(left, 3)

        right = Card("Session length")
        chips = QHBoxLayout()
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        for m in (25, 45, 60, 90):
            b = QPushButton(f"{m} min")
            b.setObjectName("Chip")
            b.setCheckable(True)
            b.setProperty("minutes", m)
            self.group.addButton(b)
            chips.addWidget(b)
        b = QPushButton("Custom")
        b.setObjectName("Chip")
        b.setCheckable(True)
        b.setProperty("minutes", 0)
        self.group.addButton(b)
        chips.addWidget(b)
        right.add(chips)
        crow = QHBoxLayout()
        crow.addWidget(label("Custom minutes", "Muted"))
        self.custom = QSpinBox()
        self.custom.setRange(1, 600)
        self.custom.setValue(self.app.settings["focus_minutes"])
        crow.addWidget(self.custom)
        crow.addStretch(1)
        right.add(crow)
        self.group.buttonClicked.connect(self._chip)
        right.add(label("This session", "H3"))
        from PySide6.QtWidgets import QComboBox, QLineEdit, QSlider
        from .widgets import Toggle
        self.goal = QLineEdit()
        self.goal.setPlaceholderText("Goal - e.g. “finish the intro paragraph” (optional)")
        self.goal.setMaxLength(120)
        right.add(self.goal)
        arow = QHBoxLayout()
        arow.addWidget(label("Ambient sound", "Muted"))
        self.ambient = QComboBox()
        from ..utils.ambient import SOUNDS
        self.ambient.addItem("Off", "off")
        for k, v in SOUNDS.items():
            self.ambient.addItem(v, k)
        self.ambient.setCurrentIndex(max(0, self.ambient.findData(self.app.settings["ambient_sound"])))
        self.ambient.currentIndexChanged.connect(lambda _i: self.app.settings.set("ambient_sound", self.ambient.currentData()))
        arow.addWidget(self.ambient)
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setFixedWidth(110)
        self.vol.setValue(int(100 * float(self.app.settings["ambient_volume"])))
        self.vol.valueChanged.connect(lambda v: self.app.settings.set("ambient_volume", round(v / 100.0, 2)))
        self.vol.setToolTip("Volume")
        arow.addWidget(self.vol)
        arow.addStretch(1)
        right.add(arow)
        drow = QHBoxLayout()
        dcol = QVBoxLayout()
        dcol.setSpacing(0)
        dcol.addWidget(label("Deep focus", "Muted"))
        dcol.addWidget(label("Quiet companion, no chatter, distractions closed", "Faint"))
        drow.addLayout(dcol, 1)
        self.deep = Toggle(bool(self.app.settings["deep_focus"]))
        self.deep.toggled.connect(lambda on: self.app.settings.set("deep_focus", bool(on)))
        drow.addWidget(self.deep)
        right.add(drow)
        right.add(label("Break", "H3"))
        brow = QHBoxLayout()
        brow.addWidget(label("Break minutes", "Muted"))
        self.brk = QSpinBox()
        self.brk.setRange(1, 60)
        self.brk.setValue(self.app.settings["break_minutes"])
        self.brk.valueChanged.connect(lambda v: self.app.settings.set("break_minutes", v))
        brow.addWidget(self.brk)
        brow.addStretch(1)
        right.add(brow)
        self.today_lbl = label("", "Muted", wrap=True)
        right.add(self.today_lbl)
        right.lay.addStretch(1)
        main.addWidget(right, 2)
        self.v.addLayout(main)
        tips = Card("During a focus session", soft=True)
        tips.add(label("• Distractions score higher and your companion escalates faster.\n"
                       "• Every distraction is counted for this session.\n"
                       "• When the timer ends you get a break - blocking relaxes while you rest.", wrap=True))
        self.v.addWidget(tips)
        self.v.addStretch(1)
        self._select_default()

    def _select_default(self):
        m = self.app.settings["focus_minutes"]
        for b in self.group.buttons():
            if b.property("minutes") == m:
                b.setChecked(True)
                return
        self.group.buttons()[-1].setChecked(True)
        self.custom.setValue(m)

    def _chip(self, b):
        m = b.property("minutes")
        if m:
            self.app.settings.set("focus_minutes", m)
        self.tick()

    def _minutes(self) -> int:
        b = self.group.checkedButton()
        m = b.property("minutes") if b else 25
        if not m:
            m = self.custom.value()
            self.app.settings.set("focus_minutes", m)
        return m

    def _start(self):
        if self.app.focus.phase == Phase.BREAK:
            self.app.skip_break()
        if hasattr(self.app, "routines"):
            self.app.routines.set_goal(self.goal.text())
        self.app.start_focus(self._minutes())

    def _pause(self):
        if self.app.focus.phase == Phase.FOCUS:
            self.app.pause_focus()
        else:
            self.app.resume_focus()

    def refresh(self):
        summ = self.app.stats.today()
        self.today_lbl.setText(f"Today: {summ['sessions']} sessions · longest {fmt_duration(summ['longest_session_seconds'], True)}"
                               f" · streak {self.app.db.streak_days()} day(s)")
        self.tick()

    def tick(self):
        f = self.app.focus
        if f.phase in (Phase.FOCUS, Phase.PAUSED):
            self.ring.set(f.progress(), fmt_clock(f.remaining()), "paused" if f.phase == Phase.PAUSED else "remaining", CH_FOCUS)
            self.phase_lbl.setText("Focus Mode Active" if f.phase == Phase.FOCUS else "Paused")
            self.start.hide()
            self.pause.show()
            self.end.show()
            self.pause.setText("Resume" if f.phase == Phase.PAUSED else "Pause")
            self.goal.setReadOnly(True)
        elif f.phase == Phase.BREAK:
            self.ring.set(f.progress(), fmt_clock(f.remaining()), "break", "warning")
            self.phase_lbl.setText("Break time ☕")
            self.start.show()
            self.start.setText("Skip break")
            self.pause.hide()
            self.end.hide()
        else:
            m = self._minutes() if self.group.checkedButton() else self.app.settings["focus_minutes"]
            self.ring.set(0, fmt_clock(m * 60), "ready", CH_FOCUS)
            self.phase_lbl.setText("Ready to focus")
            if self.goal.isReadOnly():
                self.goal.setReadOnly(False)
                self.goal.clear()
            self.start.show()
            self.start.setText("Start Focus")
            self.pause.hide()
            self.end.hide()


# =========================================================================== Stats
def fmt_hours(v: float) -> str:
    """Chart axis for times given in hours: minutes while under an hour ("0.0005h" read badly)."""
    if v <= 0:
        return "0"
    if v < 1:
        m = v * 60
        return f"{m:.0f}m" if m >= 1 else f"{m * 60:.0f}s"
    return f"{v:g}h"


class StatsPage(Page):
    title = "Statistics"

    def __init__(self, app):
        super().__init__(app)
        self.header("Statistics", "Everything is computed locally from your own activity.")
        grid = QGridLayout()
        grid.setSpacing(0)
        self.tiles = {
            "focus": StatTile("Today's focus", accent=CH_FOCUS),
            "dist": StatTile("Distraction time", accent=CH_DISTRACT),
            "count": StatTile("Distractions", accent="warning"),
            "blocked": StatTile("Blocked", accent="danger"),
            "recov": StatTile("Recovered", accent="focus"),
            "sessions": StatTile("Focus sessions", accent=CH_FOCUS),
            "longest": StatTile("Longest session", accent=CH_FOCUS),
            "streak": StatTile("Current streak", accent="warning"),
        }
        for i, t in enumerate(self.tiles.values()):
            grid.addWidget(t, i // 4, i % 4)
        self.v.addLayout(grid)

        c1 = Card("Focus vs distraction", "Last 7 days")
        self.fd = ColumnChart(["Focus", "Distraction"], [CH_FOCUS, CH_DISTRACT], fmt=fmt_hours)
        c1.add(self.fd)
        self.v.addWidget(c1)
        row = QHBoxLayout()
        row.setSpacing(0)
        c2 = Card("Distractions per day", "Episodes your companion noticed")
        self.dc = ColumnChart(["Distractions"], [CH_DISTRACT], fmt=lambda v: f"{v:.0f}", height=200)
        c2.add(self.dc)
        row.addWidget(c2, 1)
        c3 = Card("Weekly productivity", "Focus share of focus + distraction time")
        self.prod = LineChart(CH_FOCUS, height=200)
        self.prod.empty_text = "Not enough data yet."
        c3.add(self.prod)
        row.addWidget(c3, 1)
        self.v.addLayout(row)
        c4 = Card("Most common distractions", "Last 7 days")
        self.top = HBarChart(CH_DISTRACT, fmt=lambda v: fmt_duration(v, True))
        c4.add(self.top)
        self.v.addWidget(c4)
        self.v.addStretch(1)

    def refresh(self):
        db = self.app.db
        summ = self.app.stats.today()
        t = self.tiles
        dur = lambda v: fmt_duration(v, True) or "0m"          # noqa: E731
        num = lambda v: str(int(round(v)))                       # noqa: E731
        t["focus"].set("", "productive time today", number=summ["focus_seconds"], fmt=dur)
        t["dist"].set("", "on distracting sites/apps", number=summ["distraction_seconds"], fmt=dur)
        t["count"].set("", "today", number=summ["distractions"], fmt=num)
        t["blocked"].set("", f"{summ['prevented']} prevented today", number=summ["blocked"], fmt=num)
        t["recov"].set("", "productive time after nudges", number=summ["recovered_seconds"], fmt=dur)
        t["sessions"].set("", "completed today", number=summ["sessions"], fmt=num)
        t["longest"].set("", "all time", number=db.longest_session(), fmt=dur)
        t["streak"].set("", "with a completed session", number=db.streak_days(),
                        fmt=lambda v: f"{int(round(v))} days")
        series = db.daily_series(7)
        cats = [dt.datetime.strptime(d["day"], "%Y-%m-%d").strftime("%a") for d in series]
        self.fd.set_data(cats, [[d["focus_seconds"] / 3600 for d in series],
                                [d["distraction_seconds"] / 3600 for d in series]])
        self.dc.set_data(cats, [[float(d["distractions"]) for d in series]])
        prod = []
        for d in series:
            tot = d["focus_seconds"] + d["distraction_seconds"]
            prod.append(100 * d["focus_seconds"] / tot if tot > 60 else None)
        self.prod.set_data(cats, prod)
        since = (dt.date.today() - dt.timedelta(days=6)).strftime("%Y-%m-%d")
        self.top.set_data(db.top_distractions(since, 6))


# =========================================================================== History
class HistoryPage(Page):
    title = "History"

    def __init__(self, app):
        super().__init__(app)
        hdr = self.header("Distraction History", "Stored only on this computer. Window titles are never saved.")
        exp = QPushButton("Export CSV")
        exp.clicked.connect(self._export)
        clr = QPushButton("Clear history")
        clr.setObjectName("Danger")
        clr.clicked.connect(self._clear)
        hdr.addWidget(exp)
        hdr.addWidget(clr)
        filt = QHBoxLayout()
        self.range = QComboBox()
        self.range.addItems(["Today", "Last 7 days", "Last 30 days", "All time"])
        self.range.currentIndexChanged.connect(self.refresh)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search app, website or action…")
        self.search.textChanged.connect(self.refresh)
        filt.addWidget(self.range)
        filt.addWidget(self.search, 1)
        self.v.addLayout(filt)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Time", "Application", "Website", "Duration", "Score", "Action", "Focus"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setMinimumHeight(420)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.v.addWidget(self.table, 1)
        self.count = label("", "Faint")
        self.v.addWidget(self.count)

    def _since(self):
        idx = self.range.currentIndex()
        today = dt.datetime.combine(dt.date.today(), dt.time())
        return {0: today, 1: today - dt.timedelta(days=6), 2: today - dt.timedelta(days=29)}.get(idx)

    def refresh(self):
        since = self._since()
        rows = self.app.db.history(since.timestamp() if since else None, self.search.text().strip(), 1000)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            when = dt.datetime.fromtimestamp(r["timestamp"])
            vals = [when.strftime("%b %d  %H:%M"), r["application"] or "", r["website"] or "—",
                    fmt_duration(r["duration"]), str(r["score"]), ACTION_LABELS.get(r["action"], r["action"]),
                    "Yes" if r["focus_session"] else ""]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if j == 4:
                    sc = r["score"]
                    item.setForeground(QColor(C["danger"] if sc >= 90 else C["distract"] if sc >= 70 else C["text"]))
                self.table.setItem(i, j, item)
        self.count.setText(f"{len(rows)} events")

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export history", "focusbuddy_history.csv", "CSV (*.csv)")
        if path:
            n = self.app.db.export_history_csv(path)
            QMessageBox.information(self, "Exported", f"Saved {n} events to\n{path}")

    def _clear(self):
        if QMessageBox.question(self, "Clear history",
                                "Delete all distraction history, sessions and statistics?\nThis cannot be undone.") \
                == QMessageBox.Yes:
            self.app.db.clear_history()
            self.refresh()


# =========================================================================== Window
class _PageSlot(QWidget):
    """A dashboard page that is built the first time it is needed (shown, refreshed, searched).

    Building all seventeen pages up front made opening the dashboard take seconds; the slot holds
    the page's place in the stack and forwards everything to the real page once it exists."""

    def __init__(self, factory):
        super().__init__()
        self.setObjectName("Page")
        self._factory = factory
        self._page = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

    @property
    def built(self) -> bool:
        return self._page is not None

    def ensure(self):
        if self._page is None:
            self._page = self._factory()
            self._factory = None
            self.layout().addWidget(self._page)
        return self._page

    def refresh(self):
        return self.ensure().refresh()

    def tick(self):
        if self._page is not None and hasattr(self._page, "tick"):
            return self._page.tick()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.ensure(), name)


class Dashboard(QMainWindow):
    NAV = [("home", "Home", "🏠"), ("focus", "Focus", "⏱"), ("tasks", "Tasks", "✅"), ("care", "Care", "💝"),
           ("coach", "Coach", "🧭"),
           ("chat", "Chat", "💬"), ("stats", "Statistics", "📊"), ("reports", "Reports", "📝"),
           ("history", "History", "🕘"), ("rules", "Sites && Apps", "🌐"), ("companion", "Companion", "🐾"),
           ("playground", "Playground", "🎮"), ("settings", "Settings", "⚙"), ("privacy", "Privacy", "🔒"),
           ("help", "Help", "💡"), ("faq", "FAQ", "❓"), ("about", "About", "ℹ")]

    def __init__(self, app, icon):
        super().__init__()
        from PySide6.QtGui import QKeySequence, QShortcut
        from .icons import nav_icon
        from .info_pages import AboutPage, FaqPage, HelpPage, PrivacyPage
        from .settings_pages import CompanionPage, RulesPage, SettingsPage
        from .widgets import Toggle
        self.app = app
        self.setWindowTitle("FocusBuddy")
        self.setWindowIcon(icon)
        self.resize(1240, 820)
        self.setMinimumSize(1020, 660)
        self.bg = AuroraBackground()
        h = QHBoxLayout(self.bg)
        h.setContentsMargins(14, 14, 10, 0)
        h.setSpacing(10)

        # ---------------------------------------------------------- sidebar
        side = QFrame()
        side.setObjectName("Sidebar")
        side.setFixedWidth(244)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(12, 18, 12, 14)
        sv.setSpacing(2)
        brand = QHBoxLayout()
        brand.setContentsMargins(8, 0, 0, 0)
        self.brand_pic = QLabel()
        brand.addWidget(self.brand_pic)
        bcol = QVBoxLayout()
        bcol.setSpacing(0)
        bcol.addWidget(label("FocusBuddy", "H3"))
        bcol.addWidget(label("Focus companion", "Faint"))
        brand.addLayout(bcol)
        brand.addStretch(1)
        sv.addLayout(brand)
        sv.addSpacing(16)
        self.stack = AnimatedStack()
        from .playground import CompanionPlayground
        from .intel_pages import ChatPage, CoachPage, ReportsPage, TasksPage
        from .care_page import CarePage
        factories = [lambda: HomePage(app), lambda: FocusPage(app), lambda: TasksPage(app), lambda: CarePage(app),
                     lambda: CoachPage(app), lambda: ChatPage(app), lambda: StatsPage(app), lambda: ReportsPage(app),
                     lambda: HistoryPage(app), lambda: RulesPage(app), lambda: CompanionPage(app),
                     lambda: CompanionPlayground(app.settings["character"]), lambda: SettingsPage(app),
                     lambda: PrivacyPage(app), lambda: HelpPage(app), lambda: FaqPage(app), lambda: AboutPage(app)]
        # pages are built when first needed; the rest follow a few at a time while the app is idle
        self.pages = [_PageSlot(f) for f in factories]
        self.nav = NavRail()
        for i, (p, (key, text, emoji)) in enumerate(zip(self.pages, self.NAV)):
            self.stack.addWidget(p)
            ic = nav_icon(key, C["muted"], C["on_accent"], 18)
            b = self.nav.add(("   " + text) if ic is not None else f"{emoji}   {text}", ic,
                             separator_before=key in ("rules", "settings", "help"))
            b.setToolTip(f"{text.replace('&&', '&')}" + (f"   (Ctrl+{i + 1})" if i < 9 else ""))
        self.nav.page_requested.connect(self.show_page)
        # the navigation scrolls if the window is too short for all twelve items (no squeezing)
        nav_scroll = QScrollArea()
        nav_scroll.setWidgetResizable(True)
        nav_scroll.setFrameShape(QFrame.NoFrame)
        nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        nav_scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        nav_holder = QWidget()
        nav_holder.setObjectName("Transparent")
        nh = QVBoxLayout(nav_holder)
        nh.setContentsMargins(0, 0, 4, 0)
        nh.addWidget(self.nav)
        nh.addStretch(1)
        nav_scroll.setWidget(nav_holder)
        sv.addWidget(nav_scroll, 1)
        mon = QHBoxLayout()
        mon.setContentsMargins(10, 0, 6, 0)
        self.mon_lbl = label("Monitoring", "Muted")
        mon.addWidget(self.mon_lbl)
        mon.addStretch(1)
        self.mon_toggle = Toggle(self.app.monitoring)
        self.mon_toggle.toggled.connect(lambda on: self.app.set_monitoring(on))
        mon.addWidget(self.mon_toggle)
        sv.addLayout(mon)
        sv.addSpacing(10)
        self.me = QPushButton()
        self.me.setObjectName("ChoiceCard")
        self.me.setCursor(Qt.PointingHandCursor)
        self.me.setMinimumHeight(58)
        ml = QHBoxLayout(self.me)
        ml.setContentsMargins(10, 6, 10, 6)
        self.me_pic = QLabel()
        self.me_pic.setAttribute(Qt.WA_TransparentForMouseEvents)
        ml.addWidget(self.me_pic)
        mcol = QVBoxLayout()
        mcol.setSpacing(0)
        self.me_name = label("", "H3")
        self.me_name.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.me_sub = label("Local profile", "Faint")
        self.me_sub.setAttribute(Qt.WA_TransparentForMouseEvents)
        mcol.addWidget(self.me_name)
        mcol.addWidget(self.me_sub)
        ml.addLayout(mcol, 1)
        dots = label("⋯", "Muted")
        dots.setAttribute(Qt.WA_TransparentForMouseEvents)
        ml.addWidget(dots)
        self.me.clicked.connect(lambda: self._profile_menu(self.me))
        sv.addWidget(self.me)
        sv.addSpacing(4)
        foot = label(f"v{__version__} · your data stays on this PC", "Faint")
        foot.setContentsMargins(8, 0, 0, 0)
        sv.addWidget(foot)
        sidewrap = QVBoxLayout()
        sidewrap.setContentsMargins(0, 0, 0, 14)
        sidewrap.addWidget(side)
        h.addLayout(sidewrap)

        # ---------------------------------------------------------- main column
        main = QVBoxLayout()
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        bar = QFrame()
        bar.setObjectName("TopBar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(24, 2, 18, 6)
        bl.setSpacing(10)
        self.search = QLineEdit()
        self.search.setObjectName("Search")
        self.search.setPlaceholderText("Search or run a command…   Ctrl+K")
        self.search.setClearButtonEnabled(True)
        self.search.setMaximumWidth(460)
        self.search.setMinimumWidth(300)
        bl.addWidget(self.search)
        bl.addStretch(1)
        self.quick = QPushButton("Start focus")
        self.quick.setObjectName("Primary")
        self.quick.setMinimumHeight(36)
        self.quick.clicked.connect(self._quick)
        bl.addWidget(self.quick)
        self.avatar = QPushButton()
        self.avatar.setObjectName("Avatar")
        self.avatar.setFixedSize(40, 40)
        self.avatar.setIconSize(QSize(36, 36))
        self.avatar.setCursor(Qt.PointingHandCursor)
        self.avatar.clicked.connect(lambda: self._profile_menu(self.avatar))
        bl.addWidget(self.avatar)
        main.addWidget(bar)
        main.addWidget(self.stack, 1)
        h.addLayout(main, 1)
        self.setCentralWidget(self.bg)

        self._index: dict[str, tuple[int, object]] = {}
        self._search_ready = False
        self.search.returnPressed.connect(lambda: self._go_search(self.search.text()))
        self.refresh_profile()
        # keyboard shortcuts (like any Windows app)
        for i in range(min(9, len(self.pages))):
            QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self, activated=lambda k=i: self.show_page(k))
        # 2.2: Ctrl+K (or a click in the search box) opens the command palette
        from .palette import CommandPalette
        self.palette = CommandPalette(self.centralWidget(), lambda: list(self.app.settings.get("palette_recent", []) or []),
                                      lambda keys: self.app.settings.set("palette_recent", list(keys)))
        self.search.installEventFilter(self)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self.open_palette)
        QShortcut(QKeySequence("Ctrl+P"), self, activated=self.open_palette)
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self.open_palette)
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self._lock)
        QShortcut(QKeySequence("Ctrl+,"), self, activated=lambda: self.show_page("settings"))
        QShortcut(QKeySequence("F1"), self, activated=lambda: self.show_page("help"))
        QShortcut(QKeySequence("Ctrl+W"), self, activated=self.close)
        QShortcut(QKeySequence("Ctrl+H"), self,
                  activated=lambda: self.app.show_companion(not self.app.settings["show_companion"]))

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000)
        self._n = 0
        self._restore_geometry()
        self.show_page(0)

    PAGE_INDEX = {k: i for i, (k, _t, _e) in enumerate(NAV)}

    # ---------------------------------------------------------- profile / account
    def refresh_profile(self) -> None:
        from .icons import avatar_pixmap
        p = getattr(self.app, "profile", None)
        name = p.name if p is not None else (self.app.settings["user_name"] or "You")
        avatar = p.avatar if p is not None else self.app.settings["character"]
        self.me_pic.setPixmap(avatar_pixmap(avatar, name, 38))
        self.me_name.setText(name)
        if p is not None and getattr(p, "has_account", False):
            text = p.email + (" · PIN" if p.has_pin else "")
            self.me_sub.setToolTip(text)
            self.me_sub.setText(self.me_sub.fontMetrics().elidedText(text, Qt.ElideMiddle, 150))
        else:
            self.me_sub.setText(("PIN protected" if p is not None and p.has_pin else "Local profile"))
        from PySide6.QtGui import QIcon
        self.avatar.setIcon(QIcon(avatar_pixmap(avatar, name, 36, ring=False)))
        self.avatar.setToolTip(f"{name} - account")
        self.brand_pic.setPixmap(avatar_pixmap(self.app.settings["character"], "FB", 40))

    def _profile_menu(self, anchor) -> None:
        from PySide6.QtWidgets import QMenu
        p = getattr(self.app, "profile", None)
        m = QMenu(self)
        head = m.addAction(f"Signed in as {p.name if p else self.app.settings['user_name'] or 'you'}")
        head.setEnabled(False)
        m.addSeparator()
        m.addAction("Account settings", lambda: self.show_page("settings", anchor="Account"))
        lock = m.addAction("Lock                       Ctrl+L", self._lock)
        lock.setEnabled(bool(p is not None and p.has_pin))
        m.addAction("Switch profile…", lambda: self.app.switch_profile())
        m.addSeparator()
        m.addAction("About FocusBuddy        F1", self.show_about)
        m.addAction("Quit FocusBuddy", self.app.quit)
        m.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))

    def _lock(self) -> None:
        if hasattr(self.app, "lock"):
            self.app.lock()

    def show_about(self) -> None:
        from .about import AboutDialog
        AboutDialog(self).exec()

    def toast(self, text: str, kind: str = "info") -> None:
        Toast.show_text(self.centralWidget(), text, kind)

    def _quick(self) -> None:
        f = self.app.focus
        if f.phase in (Phase.FOCUS, Phase.PAUSED, Phase.BREAK):
            self.show_page("focus")
        else:
            self.app.start_focus(self.app.settings["focus_minutes"])
            self.toast(f"Focus session started - {self.app.settings['focus_minutes']} minutes", "ok")

    # ---------------------------------------------------------- command palette
    def open_palette(self, text: str = "") -> None:
        self._ensure_search()
        self.palette.open(self.palette_commands(), text)

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if obj is getattr(self, "search", None) and ev.type() in (QEvent.MouseButtonPress, QEvent.KeyPress):
            text = ev.text() if ev.type() == QEvent.KeyPress and ev.text().isprintable() else ""
            if ev.type() == QEvent.KeyPress and not text:
                return super().eventFilter(obj, ev)
            QTimer.singleShot(0, lambda: self.open_palette(self.search.text() + text))
            QTimer.singleShot(0, self.search.clear)
            return True
        return super().eventFilter(obj, ev)

    def palette_commands(self) -> list:
        """Everything the palette can do right now (titles follow the current state)."""
        from .palette import Command
        from ..companion.characters import selectable_characters
        a, st = self.app, self.app.settings
        f = a.focus
        mins = int(st["focus_minutes"])
        running = f.phase in (Phase.FOCUS, Phase.PAUSED)
        cname = companion_name(get_character(st["character"]), st.get("companion_name", "") or "")

        def toast_after(fn, msg, kind="ok"):
            def run():
                fn()
                self.toast(msg, kind)
            return run

        cmds = [
            Command(f"Start {mins}-minute focus", "Actions",
                    toast_after(lambda: a.start_focus(mins), f"Focus session started - {mins} minutes"),
                    "Focus", "▶", "begin session pomodoro timer", key="act:focus"),
            Command("Start 50-minute deep work", "Actions", toast_after(lambda: a.start_focus(50), "Deep work - 50 minutes"),
                    "", "🧠", "deep long focus", key="act:focus50"),
            Command("Take a 5-minute break", "Actions", toast_after(lambda: a.start_break(5), "Break - 5 minutes"),
                    "", "☕", "rest pause short break", key="act:break5"),
            Command("Take a 15-minute break", "Actions", toast_after(lambda: a.start_break(15), "Break - 15 minutes"),
                    "", "🌿", "rest long break", key="act:break15"),
        ]
        if mins != 25:
            cmds.insert(1, Command("Start 25-minute focus", "Actions",
                                   toast_after(lambda: a.start_focus(25), "25-minute focus started"),
                                   "", "🍅", "pomodoro", key="act:focus25"))
        if running:
            paused = f.phase == Phase.PAUSED
            cmds.append(Command("Resume session" if paused else "Pause session", "Actions",
                                a.resume_focus if paused else a.pause_focus, "", "⏯", "pause resume", key="act:pause"))
            cmds.append(Command("End session", "Actions", a.end_focus, "", "⏹", "stop finish", key="act:end"))
        mon = bool(a.monitoring)
        cmds += [
            Command("Pause monitoring" if mon else "Resume monitoring", "Actions",
                    toast_after(lambda: (a.set_monitoring(not mon), self.mon_toggle.setChecked(not mon)),
                                "Monitoring paused" if mon else "Monitoring on", "info"),
                    "", "⏸" if mon else "👁", "tracking watch off on", key="act:monitor"),
            Command(f"Hide {cname}" if st["show_companion"] else f"Show {cname}", "Actions",
                    lambda: a.show_companion(not st["show_companion"]), "Ctrl+H", "🙈" if st["show_companion"] else "🐾",
                    "companion desktop pet hide show", key="act:companion"),
        ]
        theme = str(st["theme"])
        for mode, title, icon in (("dark", "Switch to dark mode", "🌙"), ("light", "Switch to light mode", "☀"),
                                  ("system", "Follow the system theme", "🖥")):
            if mode != theme:
                cmds.append(Command(title, "Actions", lambda m=mode: st.set("theme", m), "", icon, "theme appearance colour",
                                    key=f"act:theme_{mode}"))
        p = getattr(a, "profile", None)
        if p is not None and p.has_pin:
            cmds.append(Command("Lock FocusBuddy", "Actions", self._lock, "Ctrl+L", "🔒", "pin", key="act:lock"))
        cmds.append(Command("Switch profile…", "Actions", a.switch_profile, "", "👥", "account user", key="act:profile"))
        cmds.append(Command("About FocusBuddy", "Actions", self.show_about, "F1", "ℹ", "version", key="act:about"))
        for spec in selectable_characters():
            if spec.id == st["character"]:
                continue
            nm = companion_name(spec)
            cmds.append(Command(f"Switch companion to {nm}", "Companions", lambda c=spec.id: a.set_character(c),
                                spec.id.replace("_", " "), "🐾", f"character pet {spec.id}", key=f"char:{spec.id}"))
        for i, (key, text, emoji) in enumerate(self.NAV):
            t = text.replace("&&", "&")
            cmds.append(Command(f"Go to {t}", "Pages", lambda k=i: self.show_page(k),
                                f"Ctrl+{i + 1}" if i < 9 else "", emoji, f"page open {key}", key=f"page:{key}"))
        seen = set()
        for k, (idx, target) in self._index.items():
            if target is None:
                continue
            title, _sep, where = k.partition("  ·  ")
            if (title, where) in seen:
                continue
            seen.add((title, where))
            cmds.append(Command(title, "Settings & help", lambda kk=k: self._go_search(kk), where, "⚙", where,
                                key=f"set:{k}"))
        return cmds

    # ---------------------------------------------------------- search
    def _build_pending(self) -> None:
        """Build the next not-yet-built page while the dashboard is open and idle (one per call,
        spaced out so no single frame of an animation waits long); then the search index."""
        if not self.isVisible():
            return
        nxt = next((p for p in self.pages if not p.built), None)
        if nxt is not None:
            if self.stack._anim.state() == self.stack._anim.State.Running or                     getattr(self.stack, "_card_rise", None) is not None:
                QTimer.singleShot(250, self._build_pending)      # a page transition is playing
                return
            try:
                nxt.ensure()
            except Exception:  # noqa: BLE001 - a broken page must not stop the others
                import logging
                logging.getLogger("focusbuddy.ui").exception("page build failed")
            QTimer.singleShot(120, self._build_pending)
            return
        if not self._search_ready:
            self._build_search()

    def _ensure_search(self) -> None:
        if not self._search_ready:
            for p in self.pages:
                p.ensure()
            self._build_search()

    def _build_search(self) -> None:
        from PySide6.QtWidgets import QCompleter
        self._index = {}
        self._search_ready = True
        for i, (p, (key, text, _)) in enumerate(zip(self.pages, self.NAV)):
            self._index[f"{text.replace('&&', '&')}  ·  page"] = (i, None)
            for card in p.findChildren(GlassCard):
                t = getattr(card, "title", "")
                if t:
                    self._index.setdefault(f"{t}  ·  {text.replace('&&', '&')}", (i, card))
            for lb in p.findChildren(QLabel):
                st = lb.property("search_title")
                if st:
                    self._index.setdefault(f"{st}  ·  {text.replace('&&', '&')}", (i, lb))
        comp = QCompleter(sorted(self._index), self)
        comp.setCaseSensitivity(Qt.CaseInsensitive)
        comp.setFilterMode(Qt.MatchContains)
        comp.setMaxVisibleItems(10)
        comp.activated.connect(self._go_search)
        self.search.setCompleter(comp)

    def _go_search(self, text: str) -> None:
        self._ensure_search()
        hit = self._index.get(text)
        if hit is None:
            low = text.lower().strip()
            hit = next((v for k, v in self._index.items() if low and low in k.lower()), None)
        if hit is None:
            return
        idx, target = hit
        self.show_page(idx)
        if target is not None:
            QTimer.singleShot(420, lambda: self._reveal(idx, target))
        QTimer.singleShot(0, self.search.clear)

    def _reveal(self, idx: int, w) -> None:
        page = self.pages[idx]
        area = page.findChild(QScrollArea)
        if area is not None:
            area.ensureWidgetVisible(w, 40, 120)
        card = w
        while card is not None and not isinstance(card, GlassCard):
            card = card.parentWidget()
        if card is not None:
            card.flash()

    # ---------------------------------------------------------- window
    def _restore_geometry(self) -> None:
        import base64
        raw = self.app.settings.get("window_geometry", "") if hasattr(self.app.settings, "get") else ""
        if raw:
            try:
                from PySide6.QtCore import QByteArray
                self.restoreGeometry(QByteArray(base64.b64decode(raw)))
            except Exception:  # noqa: BLE001
                pass

    def _save_geometry(self) -> None:
        import base64
        try:
            self.app.settings.set("window_geometry", base64.b64encode(bytes(self.saveGeometry())).decode())
        except Exception:  # noqa: BLE001
            pass

    def current_page_name(self) -> str:
        idx = self.stack.currentIndex()
        return next((k for k, v in self.PAGE_INDEX.items() if v == idx), "home")

    def showEvent(self, ev):
        super().showEvent(ev)
        from .theme import fluent_window
        fluent_window(self)
        fade_in(self)
        self.nav.select(self.stack.currentIndex(), animate=False)
        if not getattr(self, "_cascaded", False):
            # the first time the window opens, its cards arrive one after another
            self._cascaded = True
            QTimer.singleShot(1500, self._build_pending)
            from .motion import cascade_in, full_motion
            if full_motion():
                QTimer.singleShot(0, lambda: self.stack.currentWidget() is not None and
                                  cascade_in(self.stack.currentWidget()))

    def show_page(self, idx, anchor: str | None = None):
        if isinstance(idx, str):
            idx = self.PAGE_INDEX.get(idx, 0)
        self.nav.select(idx)
        try:
            # fresh numbers first, so the cards arrive (CardRise snapshots) already showing them
            self.pages[idx].refresh()
        except Exception:  # noqa: BLE001 - a page error shouldn't close the window
            import logging
            logging.getLogger("focusbuddy.ui").exception("page refresh failed")
        self.stack.setCurrentIndex(idx)
        self._update_mon()
        if anchor:
            card = next((c for c in self.pages[idx].findChildren(GlassCard) if getattr(c, "title", "") == anchor),
                        None)
            if card is not None:
                QTimer.singleShot(420, lambda: self._reveal(idx, card))

    def _update_mon(self):
        self.mon_toggle.blockSignals(True)
        self.mon_toggle.setChecked(bool(self.app.monitoring))
        self.mon_toggle.blockSignals(False)
        self.mon_lbl.setText("Monitoring" if self.app.monitoring else "Monitoring paused")
        f = self.app.focus
        if f.phase in (Phase.FOCUS, Phase.PAUSED):
            self.quick.setText(f"⏱  {fmt_clock(f.remaining())}" + ("  ·  paused" if f.phase == Phase.PAUSED else ""))
        elif f.phase == Phase.BREAK:
            self.quick.setText(f"☕  Break {fmt_clock(f.remaining())}")
        else:
            self.quick.setText("Start focus")

    def _tick(self):
        if not self.isVisible():
            return
        self._n += 1
        page = self.stack.currentWidget()
        try:
            page.tick()
            if self._n % 15 == 0:
                page.refresh()
        except Exception:  # noqa: BLE001
            pass
        self._update_mon()

    def refresh_character(self):
        self.refresh_profile()

    def retheme(self) -> None:
        """A live theme switch: icons drawn in the old text colour and the profile pictures are redrawn,
        the visible page refreshes (the style sheet and the painted widgets do the rest)."""
        from .icons import nav_icon
        for b, (key, _text, _emoji) in zip(self.nav.buttons, self.NAV):
            ic = nav_icon(key, C["muted"], C["on_accent"], 18)
            if ic is not None:
                b.setIcon(ic)
        self.refresh_profile()
        try:
            self.stack.currentWidget().refresh()
        except Exception:  # noqa: BLE001
            pass

    def closeEvent(self, ev):
        self._save_geometry()
        # closing the dashboard keeps FocusBuddy running in the tray
        if getattr(self.app, "quitting", False):
            ev.accept()
            return
        ev.ignore()
        self.hide()
        # the companion notices: "Wait, we're done already? Should I go curl up for a nap?"
        hook = getattr(self.app, "on_dashboard_closed", None)
        if hook is not None:
            QTimer.singleShot(150, hook)
