"""2.1: the Care page - feed, play and pet your companion; bond level, treats and badges."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..companion.characters import get_character
from ..companion.states import CompanionState as S
from ..intel.care import UNLOCKS
from .dashboard import Page
from .theme import C, tok
from .widgets import CharacterPreview, Card, StatTile, label

NEEDS = (("food", "Food", "🍪", "sun"), ("fun", "Fun", "🎾", "primary"), ("love", "Love", "💛", "focus"))


class NeedBar(QWidget):
    """A rounded bar for one need (0..100), coloured by how full it is."""

    def __init__(self, accent: str):
        super().__init__()
        self.v = 0.0
        self.accent = accent
        self.setFixedHeight(12)
        self.setMinimumWidth(160)

    def set(self, v: float) -> None:
        self.v = max(0.0, min(100.0, float(v)))
        self.update()

    def paintEvent(self, _):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        g.setPen(Qt.NoPen)
        g.setBrush(QColor(C["surface_alt"]))
        g.drawRoundedRect(r, 6, 6)
        col = QColor(tok(self.accent if self.v >= 35 else "distract"))
        g.setBrush(col)
        g.drawRoundedRect(QRectF(r.left(), r.top(), max(r.height(), r.width() * self.v / 100.0), r.height()), 6, 6)
        g.end()


class CarePage(Page):
    title = "Care"

    def __init__(self, app):
        super().__init__(app)
        self.header("Companion Care", "Feed, play and pet your companion. Focus earns treats and grows your bond.")
        top = QHBoxLayout()
        top.setSpacing(0)
        hero = Card()
        row = QHBoxLayout()
        self.preview = CharacterPreview(get_character(app.settings["character"]), 190)
        row.addWidget(self.preview, 0, Qt.AlignVCenter)
        col = QVBoxLayout()
        col.setSpacing(6)
        self.name_lbl = label("", "H2")
        col.addWidget(self.name_lbl)
        self.mood_lbl = label("", "Muted", wrap=True)
        col.addWidget(self.mood_lbl)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self.bars = {}
        self.vals = {}
        for i, (key, name, emoji, accent) in enumerate(NEEDS):
            grid.addWidget(QLabel(f"{emoji}  {name}"), i, 0)
            bar = NeedBar(accent)
            grid.addWidget(bar, i, 1)
            val = label("", "Muted")
            grid.addWidget(val, i, 2)
            self.bars[key], self.vals[key] = bar, val
        grid.setColumnStretch(1, 1)
        col.addLayout(grid)
        btns = QHBoxLayout()
        self.b_feed = QPushButton("Give a treat 🍪")
        self.b_feed.setObjectName("Primary")
        self.b_play = QPushButton("Play 🎾")
        self.b_pet = QPushButton("Pet 💛")
        for b, act in ((self.b_feed, "feed"), (self.b_play, "play"), (self.b_pet, "pet")):
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, a=act: self._do(a))
            btns.addWidget(b)
        btns.addStretch(1)
        col.addLayout(btns)
        self.note = label("", "Faint", wrap=True)
        col.addWidget(self.note)
        row.addLayout(col, 1)
        hero.add(row)
        top.addWidget(hero, 3)
        side = QVBoxLayout()
        side.setSpacing(0)
        self.t_level = StatTile("Bond level", "–", "", "primary")
        self.t_treats = StatTile("Treats", "–", "1 per 10 focused minutes", "sun")
        side.addWidget(self.t_level)
        side.addWidget(self.t_treats)
        top.addLayout(side, 1)
        self.v.addLayout(top)

        lv = Card("Bond", "Grows with every focused minute, finished session goals and a little care each day.")
        self.lv_lbl = label("", wrap=True)
        lv.add(self.lv_lbl)
        self.lv_bar = NeedBar("primary")
        lv.add(self.lv_bar)
        self.unlocks = label("", "Muted", wrap=True)
        lv.add(self.unlocks)
        self.v.addWidget(lv)

        bd = Card("Badges", "Focus streaks and milestones.")
        self.badge_grid = QGridLayout()
        self.badge_grid.setHorizontalSpacing(10)
        self.badge_grid.setVerticalSpacing(10)
        bd.add(self.badge_grid)
        self.v.addWidget(bd)
        self.v.addStretch(1)
        app.care_listeners.append(lambda: self.refresh() if self.isVisible() else None)

    def _do(self, action: str) -> None:
        res = self.app.care_action(action)
        if not res.get("ok"):
            if res.get("reason") == "napping":
                self.note.setText("It's napping right now - wake it up first (right-click it, or the tray icon).")
            elif res.get("reason") == "no_treats":
                self.note.setText("No treats left. Ten focused minutes earns one 🍪")
        else:
            gain = {"feed": "Yum!", "play": "So much fun!", "pet": "Purr…"}[action]
            self.note.setText(gain + (f"  +{res['xp']} bond" if res.get("xp") else ""))
        self.refresh()

    def refresh(self):
        app = self.app
        care = app.routines.care
        care.tick(True)
        spec = get_character(app.settings["character"])
        self.preview.set_character(spec)
        mood = care.mood()
        self.preview.set_state(S.PLAYING if mood == "PLAYFUL" else S.SITTING if mood in ("SAD", "LONELY") else S.IDLE)
        from ..companion.characters import companion_name
        name = companion_name(spec, app.settings["companion_name"])
        gold = care.level >= 10
        self.name_lbl.setText(("✨ " if gold else "") + name)
        line = care.need_line()
        feeling = {"PLAYFUL": "is bursting with energy", "HAPPY": "is happy",
                   "SAD": "is a bit hungry", "BORED": "is bored", "LONELY": "misses you a little"}.get(mood, "is happy")
        napping = app.companion.life.napping
        self.mood_lbl.setText(f"{name} {feeling}." + ("  It's napping off-screen right now 💤" if napping else "")
                              + (f"  {line}" if line and not napping else ""))
        s = care.s
        for key, *_ in NEEDS:
            v = getattr(s, key)
            self.bars[key].set(v)
            self.vals[key].setText(f"{int(round(v))}%")
        self.b_feed.setEnabled(s.treats > 0 and not napping)
        self.b_play.setEnabled(not napping)
        self.b_pet.setEnabled(not napping)
        self.t_level.set(str(care.level), care.title)
        self.t_treats.set(str(s.treats), "1 per 10 focused minutes")
        into, need = care.level_progress()
        if need:
            self.lv_lbl.setText(f"Level {care.level} - {care.title}. {into} / {need} XP to level {care.level + 1}.")
            self.lv_bar.set(100.0 * into / need)
        else:
            self.lv_lbl.setText(f"Level {care.level} - {care.title}. The highest bond there is!")
            self.lv_bar.set(100.0)
        got = [f"✓ L{k}: {v}" for k, v in UNLOCKS.items() if k <= care.level]
        nxt = [f"L{k}: {v}" for k, v in UNLOCKS.items() if k > care.level][:2]
        self.unlocks.setText(("Unlocked - " + "; ".join(got) + ".  " if got else "") +
                             ("Next - " + "; ".join(nxt) + "." if nxt else ""))
        while self.badge_grid.count():
            it = self.badge_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for i, (icon, nm, earned) in enumerate(app.routines.badges()):
            b = QLabel(f"{icon}  {nm}")
            b.setStyleSheet(f"padding:6px 12px; border-radius:12px; font-weight:600; "
                            f"background:{C['primary_soft'] if earned else C['surface_alt']}; "
                            f"color:{C['primary_text'] if earned else C['faint']};")
            self.badge_grid.addWidget(b, i // 4, i % 4)
