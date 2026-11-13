"""Dashboard pages for rules, companion, settings and privacy."""

from __future__ import annotations

import os
import subprocess
import sys

from PySide6.QtCore import QPointF, QRectF, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QLineEdit,
                               QMessageBox, QPushButton, QSlider, QSpinBox, QStyledItemDelegate, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ..companion.characters import selectable_characters, get_character
from ..config import SIZE_LABELS, SIZE_RANGE
from ..core.rules import DEFAULT_RULES, normalize_app, normalize_domain
from .dashboard import Page
from .widgets import Card, CharacterPreview, Toggle, label

CATEGORY_LABELS = [("productive", "Focus / allowed"), ("distracting", "Distracting"), ("blocked", "Blocked")]


# ------------------------------------------------------------------ form helpers
class Form:
    def __init__(self, page: Page, card: Card):
        self.page = page
        self.app = page.app
        self.card = card
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(18)
        self.grid.setVerticalSpacing(12)
        self.grid.setColumnStretch(0, 1)
        card.add(self.grid)
        self.r = 0
        # a setting changed elsewhere (the tray, the companion's menu, Ctrl + wheel ...) shows here too:
        # before, the "show companion" switch stayed on after hiding the companion from its menu
        self._sync: dict = {}
        self.app.settings.on_change(self._changed)

    def _changed(self, key, value) -> None:
        fn = self._sync.get(key)
        if fn is None:
            return
        try:
            fn(value)
        except RuntimeError:                 # the widget is gone
            self._sync.pop(key, None)

    @staticmethod
    def _quiet(widget, fn) -> None:
        widget.blockSignals(True)
        try:
            fn()
        finally:
            widget.blockSignals(False)

    def row(self, title: str, widget: QWidget, help_text: str = ""):
        col = QVBoxLayout()
        col.setSpacing(1)
        t = label(title)
        t.setStyleSheet("font-size:13px; font-weight:600;")
        t.setProperty("search_title", title)
        col.addWidget(t)
        if help_text:
            col.addWidget(label(help_text, "Faint", wrap=True))
        self.grid.addLayout(col, self.r, 0)
        self.grid.addWidget(widget, self.r, 1, Qt.AlignRight | Qt.AlignVCenter)
        self.r += 1
        return widget

    def toggle(self, key: str, title: str, help_text: str = "", on_change=None):
        t = Toggle(bool(self.app.settings[key]))

        def changed(v):
            self.app.settings.set(key, v)
            if on_change:
                on_change(v)
        t.toggled.connect(changed)
        self._sync[key] = lambda v: t.set_quietly(bool(v))
        return self.row(title, t, help_text)

    def spin(self, key: str, title: str, lo: int, hi: int, suffix: str = "", help_text: str = ""):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setSuffix(suffix)
        s.setValue(int(self.app.settings[key]))
        s.setMinimumWidth(110)
        s.valueChanged.connect(lambda v: self.app.settings.set(key, v))
        self._sync[key] = lambda v: self._quiet(s, lambda: s.setValue(int(v)))
        return self.row(title, s, help_text)

    def combo(self, key: str, title: str, options: list[tuple[str, str]], help_text: str = "", on_change=None):
        c = QComboBox()
        c.setMinimumWidth(200)
        for value, text in options:
            c.addItem(text, value)
        idx = max(0, c.findData(self.app.settings[key]))
        c.setCurrentIndex(idx)

        def changed(_i):
            self.app.settings.set(key, c.currentData())
            if on_change:
                on_change(c.currentData())
        c.currentIndexChanged.connect(changed)
        self._sync[key] = lambda v: c.findData(v) >= 0 and self._quiet(c, lambda: c.setCurrentIndex(c.findData(v)))
        return self.row(title, c, help_text)

    def text(self, key: str, title: str, placeholder: str = "", help_text: str = ""):
        e = QLineEdit(str(self.app.settings[key] or ""))
        e.setPlaceholderText(placeholder)
        e.setMinimumWidth(220)
        e.editingFinished.connect(lambda: self.app.settings.set(key, e.text().strip()))
        self._sync[key] = lambda v: self._quiet(e, lambda: e.setText(str(v or "")))
        return self.row(title, e, help_text)

    def slider(self, key: str, title: str, lo: float, hi: float, help_text: str = ""):
        s = QSlider(Qt.Horizontal)
        s.setRange(0, 100)
        s.setFixedWidth(200)
        v = float(self.app.settings[key])
        s.setValue(int(100 * (v - lo) / (hi - lo)))
        s.valueChanged.connect(lambda x: self.app.settings.set(key, round(lo + (hi - lo) * x / 100, 2)))
        self._sync[key] = lambda v: self._quiet(s, lambda: s.setValue(int(100 * (float(v) - lo) / (hi - lo))))
        return self.row(title, s, help_text)


# =========================================================================== Rules
CATEGORY_COLORS = {"productive": "#12B886", "distracting": "#F59F00", "blocked": "#F03E3E"}


class _RuleDelegate(QStyledItemDelegate):
    """Paints the rule rows: the category as a coloured chip, auto-close as a toggle, a Remove link."""

    def __init__(self, table):
        super().__init__(table)
        self.table = table

    def paint(self, g: QPainter, opt, index):
        from .theme import C
        r = opt.rect
        row_hover = index.row() == getattr(self.table, "_hover_row", -1)
        g.save()
        g.setRenderHint(QPainter.Antialiasing)
        if row_hover:
            hc = QColor(C.get("primary", "#7458FF"))
            hc.setAlpha(18)
            g.fillRect(r, hc)
        col = index.column()
        text_col = QColor(C.get("text", "#1E1B2E"))
        muted = QColor(C.get("muted", "#6B6880"))
        if col == 0:
            f = QFont(opt.font)
            f.setPixelSize(14)
            g.setFont(f)
            g.setPen(text_col)
            g.drawText(r.adjusted(14, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, index.data(Qt.DisplayRole) or "")
        elif col == 1:
            cat = index.data(Qt.UserRole) or "distracting"
            name = dict(CATEGORY_LABELS).get(cat, cat)
            c = QColor(CATEGORY_COLORS.get(cat, "#888888"))
            f = QFont(opt.font)
            f.setPixelSize(13)
            f.setWeight(QFont.DemiBold)
            g.setFont(f)
            w = g.fontMetrics().horizontalAdvance(name) + 44
            chip = QRectF(r.left() + 6, r.center().y() - 14, w, 28)
            bg = QColor(c)
            bg.setAlpha(36 if not row_hover else 56)
            g.setPen(Qt.NoPen)
            g.setBrush(bg)
            g.drawRoundedRect(chip, 14, 14)
            g.setBrush(c)
            g.drawEllipse(QPointF(chip.left() + 14, chip.center().y()), 4.5, 4.5)
            g.setPen(c.darker(135))
            g.drawText(chip.adjusted(24, 0, -16, 0), Qt.AlignVCenter | Qt.AlignLeft, name)
            g.drawText(chip.adjusted(0, 0, -8, 0), Qt.AlignVCenter | Qt.AlignRight, "\u25BE")
        elif col == 2:
            on = bool(index.data(Qt.UserRole))
            enabled = index.data(Qt.UserRole + 1) is not False
            track = QRectF(r.center().x() - 20, r.center().y() - 11, 40, 22)
            accent = QColor(C.get("primary", "#7458FF"))
            off = QColor(C.get("toggle_off", "#D9D6E5"))
            tc = accent if on else off
            if not enabled:
                tc = QColor(off)
                tc.setAlpha(110)
            g.setPen(Qt.NoPen)
            g.setBrush(tc)
            g.drawRoundedRect(track, 11, 11)
            knob = QPointF(track.right() - 11 if on else track.left() + 11, track.center().y())
            g.setBrush(QColor(255, 255, 255, 255 if enabled else 180))
            g.drawEllipse(knob, 8, 8)
        elif col == 3:
            f = QFont(opt.font)
            f.setPixelSize(13)
            f.setWeight(QFont.DemiBold)
            g.setFont(f)
            g.setPen(QColor("#E03131") if row_hover else muted)
            g.drawText(r.adjusted(0, 0, -14, 0), Qt.AlignVCenter | Qt.AlignRight, "Remove")
        g.restore()


class RuleTable(QWidget):
    """One kind of rule (websites or applications): add, filter by text / category, change the
    category from a chip menu, allow auto-close, remove. Rows are painted (no widget per row), so
    hundreds of rules stay instant, the wheel scrolls the list (it never changes a category by
    accident) and edits keep the scroll position."""

    def __init__(self, page, kind: str):
        super().__init__()
        self.page = page
        self.app = page.app
        self.kind = kind
        self.rows: list[dict] = []
        self.cat_filter = ""
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 12, 0, 0)
        v.setSpacing(10)
        add = QHBoxLayout()
        self.inp = QLineEdit()
        self.inp.setPlaceholderText("Add a site, e.g. youtube.com  or  netflix.com/browse" if kind == "website"
                                    else "Add an app, e.g. steam.exe  or  game*.exe")
        self.cat = QComboBox()
        for value, text in CATEGORY_LABELS:
            self.cat.addItem(text, value)
        self.cat.setCurrentIndex(1)
        btn = QPushButton("Add")
        btn.setObjectName("Primary")
        btn.clicked.connect(self._add)
        self.inp.returnPressed.connect(self._add)
        add.addWidget(self.inp, 1)
        add.addWidget(self.cat)
        add.addWidget(btn)
        v.addLayout(add)
        self.msg = label("", "Faint")
        self.msg.hide()
        v.addWidget(self.msg)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search rules…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        bar.addWidget(self.search, 1)
        self.chips = []
        for key, text in (("", "All"),) + tuple(CATEGORY_LABELS):
            b = QPushButton(text)
            b.setObjectName("Chip")
            b.setCheckable(True)
            b.setChecked(key == "")
            b.clicked.connect(lambda _=False, k=key: self._set_cat_filter(k))
            bar.addWidget(b)
            self.chips.append((key, b, text))
        v.addLayout(bar)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Pattern", "Category", "Auto-close", ""])
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setMouseTracking(True)
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.verticalHeader().setDefaultSectionSize(46)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for i, w in ((1, 190), (2, 110), (3, 100)):
            hh.setSectionResizeMode(i, QHeaderView.Fixed)
            self.table.setColumnWidth(i, w)
        self.table.setItemDelegate(_RuleDelegate(self.table))
        for c, al in ((0, Qt.AlignLeft), (1, Qt.AlignLeft), (2, Qt.AlignHCenter), (3, Qt.AlignRight)):
            it = self.table.horizontalHeaderItem(c)
            if it is not None:
                it.setTextAlignment(al | Qt.AlignVCenter)
        self.table.cellClicked.connect(self._clicked)
        self.table.cellEntered.connect(self._hover)
        self.table.viewport().installEventFilter(self)
        self.table.setMinimumHeight(380)
        v.addWidget(self.table, 1)
        self.empty = label("", "Muted")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.hide()
        v.addWidget(self.empty)

    # ------------------------------------------------------------ events
    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if obj is self.table.viewport() and ev.type() == QEvent.Leave:
            self._hover(-1, -1)
        return False

    def _hover(self, row, _col):
        if getattr(self.table, "_hover_row", -1) != row:
            self.table._hover_row = row
            self.table.viewport().update()
        self.table.viewport().setCursor(Qt.PointingHandCursor if _col in (1, 2, 3) else Qt.ArrowCursor)

    def _clicked(self, row: int, col: int) -> None:
        it = self.table.item(row, 0)
        if it is None:
            return
        r = it.data(Qt.UserRole)
        if col == 1:
            from PySide6.QtGui import QCursor
            from PySide6.QtWidgets import QMenu
            m = QMenu(self)
            for value, text in CATEGORY_LABELS:
                a = m.addAction(text)
                a.setCheckable(True)
                a.setChecked(value == r["category"])
                a.triggered.connect(lambda _=False, v=value: self._update(row, category=v))
            m.exec(QCursor.pos())
        elif col == 2:
            if r["category"] == "productive":
                self._note("Focus / allowed sites and apps are never closed.")
                return
            self._update(row, auto_close=not r["auto_close"])
        elif col == 3:
            self._remove(row)

    # ------------------------------------------------------------ edits
    def _note(self, text: str, kind: str = "info") -> None:
        dash = self.window()
        if hasattr(dash, "toast"):
            dash.toast(text, kind)
        else:
            self.msg.setText(text)
            self.msg.show()

    def _add(self):
        raw = self.inp.text().strip()
        if not raw:
            return
        if self.kind == "website":
            from urllib.parse import urlsplit
            full = raw if "://" in raw else "http://" + raw
            host = normalize_domain(full)
            try:
                path = urlsplit(full).path.strip("/")
            except ValueError:
                path = ""
            if not host or ("." not in host and host != "localhost") or " " in host:
                self._note("That doesn't look like a website - try something like youtube.com", "warn")
                return
            pattern = host + (("/" + path) if path else "")
        else:
            pattern = normalize_app(raw)
            if not pattern or any(ch in pattern for ch in "\\/:\"<>|"):
                self._note("Type the program's file name, like steam.exe", "warn")
                return
        existed = any(r["pattern"] == pattern for r in self.rows)
        self.app.db.upsert_rule(self.kind, pattern, self.cat.currentData(), False)
        self.inp.clear()
        self.app.reload_rules()
        self.refresh(keep_scroll=False)
        self._note(f"{'Updated' if existed else 'Added'} {pattern} - {self.cat.currentText()}", "ok")
        for i in range(self.table.rowCount()):
            it = self.table.item(i, 0)
            if it is not None and it.data(Qt.UserRole)["pattern"] == pattern:
                self.table.scrollToItem(it, QAbstractItemView.PositionAtCenter)
                self.table._hover_row = i
                break

    def _update(self, row: int, category: str | None = None, auto_close: bool | None = None):
        it = self.table.item(row, 0)
        if it is None:
            return
        r = dict(it.data(Qt.UserRole))
        if category is not None:
            r["category"] = category
            if category == "productive":
                r["auto_close"] = False
        if auto_close is not None:
            r["auto_close"] = bool(auto_close)
        self.app.db.upsert_rule(self.kind, r["pattern"], r["category"], r["auto_close"])
        self.app.reload_rules()
        for x in self.rows:
            if x["pattern"] == r["pattern"]:
                x.update(r)
        self._fill_row(row, r)
        self._update_counts()
        self._apply_filter()

    def _remove(self, row: int):
        it = self.table.item(row, 0)
        if it is None:
            return
        r = it.data(Qt.UserRole)
        self.app.db.delete_rule(self.kind, r["pattern"])
        self.app.reload_rules()
        self.rows = [x for x in self.rows if x["pattern"] != r["pattern"]]
        self.table.removeRow(row)
        self._update_counts()
        self._apply_filter()
        self._note(f"Removed {r['pattern']}")

    # ------------------------------------------------------------ view
    def _fill_row(self, i: int, r: dict) -> None:
        items = [QTableWidgetItem(r["pattern"]) for _ in range(4)]
        items[0].setData(Qt.UserRole, dict(r))
        items[0].setToolTip(r["pattern"])
        items[1].setData(Qt.UserRole, r["category"])
        items[1].setToolTip("Click to change")
        items[2].setData(Qt.UserRole, bool(r["auto_close"]))
        items[2].setData(Qt.UserRole + 1, r["category"] != "productive")
        items[2].setToolTip("Let FocusBuddy close this automatically (only when Automatic blocking is on)")
        items[3].setToolTip("Remove this rule")
        for c, item in enumerate(items):
            item.setText(r["pattern"] if c == 0 else "")
            self.table.setItem(i, c, item)

    def refresh(self, keep_scroll: bool = True):
        sb = self.table.verticalScrollBar()
        pos = sb.value()
        rows = list(self.app.db.rules(self.kind))
        order = {"blocked": 0, "distracting": 1, "productive": 2, "neutral": 3}
        rows.sort(key=lambda r: (order.get(r["category"], 9), r["pattern"]))
        if rows == self.rows and self.table.rowCount() == len(rows):
            return                                    # nothing changed: keep the view as it is
        self.rows = rows
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(len(rows))
            for i, r in enumerate(rows):
                self._fill_row(i, r)
        finally:
            self.table.setUpdatesEnabled(True)
        self._update_counts()
        self._apply_filter()
        if keep_scroll:
            sb.setValue(pos)

    def _set_cat_filter(self, key: str) -> None:
        self.cat_filter = key
        for k, b, _t in self.chips:
            b.setChecked(k == key)
        self._apply_filter()

    def _update_counts(self) -> None:
        n = {k: 0 for k, _ in CATEGORY_LABELS}
        for r in self.rows:
            n[r["category"]] = n.get(r["category"], 0) + 1
        for k, b, text in self.chips:
            b.setText(f"{text}  {len(self.rows) if k == '' else n.get(k, 0)}")

    def _apply_filter(self, *_a) -> None:
        q = self.search.text().strip().lower()
        shown = 0
        for i in range(self.table.rowCount()):
            it = self.table.item(i, 0)
            r = it.data(Qt.UserRole) if it is not None else None
            ok = r is not None and (not q or q in r["pattern"]) and (not self.cat_filter or r["category"] == self.cat_filter)
            self.table.setRowHidden(i, not ok)
            shown += ok
        self.empty.setVisible(shown == 0)
        self.empty.setText("No rules match." if self.rows else "No rules yet - add one above.")


class RulesPage(Page):
    title = "Sites & Apps"

    def __init__(self, app):
        super().__init__(app)
        hdr = self.header("Sites & Apps", "Decide what counts as focus, what is distracting and what may be auto-closed.")
        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(self._reset)
        hdr.addWidget(reset)
        info = Card(soft=True)
        info.add(label("Focus / allowed never trigger your companion.  Distracting sites are scored over time.  "
                       "Blocked sites score high immediately.  Auto-close only works when Automatic blocking is "
                       "enabled in Settings, and system apps (Explorer, Task Manager, Settings, antivirus…) are "
                       "always protected.", wrap=True))
        self.v.addWidget(info)
        tabs = QTabWidget()
        self.sites = RuleTable(self, "website")
        self.apps = RuleTable(self, "app")
        tabs.addTab(self.sites, "Websites")
        tabs.addTab(self.apps, "Applications")
        card = Card()
        card.add(tabs)
        self.v.addWidget(card, 1)

    def refresh(self):
        self.sites.refresh()
        self.apps.refresh()

    def _reset(self):
        if QMessageBox.question(self, "Reset rules", "Replace all website and app rules with the defaults?") == QMessageBox.Yes:
            self.app.db.replace_rules([{"kind": r.kind, "pattern": r.pattern, "category": r.category.value,
                                        "auto_close": r.auto_close} for r in DEFAULT_RULES])
            self.app.reload_rules()
            self.refresh()


# =========================================================================== Companion
class CharacterCard(QPushButton):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.setObjectName("ChoiceCard")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(150, 200)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        self.preview = CharacterPreview(spec, 120)
        self.preview.setAttribute(Qt.WA_TransparentForMouseEvents)
        v.addWidget(self.preview, 0, Qt.AlignHCenter)
        n = label(f"{spec.emoji}  {spec.display_name}")
        n.setStyleSheet("font-size:14px; font-weight:700;")
        n.setAttribute(Qt.WA_TransparentForMouseEvents)
        v.addWidget(n, 0, Qt.AlignHCenter)
        tr = label(", ".join(spec.traits), "Faint", wrap=True)
        tr.setAlignment(Qt.AlignCenter)
        tr.setAttribute(Qt.WA_TransparentForMouseEvents)
        v.addWidget(tr)


class CompanionPage(Page):
    title = "Companion"

    def __init__(self, app):
        super().__init__(app)
        self.header("Companion", "Choose who lives on your desktop and how they behave.")
        pick = Card("Character")
        grid = QGridLayout()
        grid.setSpacing(8)
        self.cards = []
        for i, spec in enumerate(selectable_characters()):
            c = CharacterCard(spec)
            c.clicked.connect(lambda _=False, s=spec: self._choose(s.id))
            grid.addWidget(c, i // 4, i % 4)
            self.cards.append(c)
        pick.add(grid)
        self.desc = label("", "Muted", wrap=True)
        pick.add(self.desc)
        self.v.addWidget(pick)

        size = Card("Size", "Pick a size or drag the slider - up to Giant, about a quarter of your screen's height. "
                    "Tip: hold Ctrl and scroll over the companion to resize it directly. Movement, eye tracking "
                    "and window targeting scale with it.")
        row = QHBoxLayout()
        self.size_group = []
        for key, label_text in SIZE_LABELS + (("custom", "Custom"),):
            b = QPushButton(label_text)
            b.setObjectName("Chip")
            b.setCheckable(True)
            b.setChecked(app.settings["size_preset"] == key)
            b.clicked.connect(lambda _=False, k=key: self._size_preset(k))
            row.addWidget(b)
            self.size_group.append((key, b))
        row.addStretch(1)
        size.add(row)
        fs = Form(self, size)
        self.size_slider = fs.slider("companion_size", "Custom size", *SIZE_RANGE)
        self.size_slider.sliderMoved.connect(lambda _v: self._mark_custom())
        self.v.addWidget(size)

        look = Card("Personality & behaviour")
        f = Form(self, look)
        name = QLineEdit(app.settings["companion_name"])
        name.setPlaceholderText(get_character(app.settings["character"]).default_name)
        name.setFixedWidth(200)
        name.editingFinished.connect(lambda: app.settings.set("companion_name", name.text().strip()))
        self.name_edit = name
        f.row("Name", name, "Leave empty for the character's default name")
        f.combo("personality", "Personality", [("", "Character's own")] + [(p, p.capitalize() if p != "buddy" else "Study buddy")
                                                                           for p in ("cat", "dog", "fox", "rabbit", "robot", "buddy")],
                "Behaviour and dialogue style - e.g. a robot with a dog's energy")
        f.slider("movement_speed", "Movement speed", 0.5, 2.0)
        f.slider("animation_intensity", "Animation intensity", 0.3, 1.5, "Bounce, tail, ears and head follow-through")
        f.combo("roaming", "Roaming", [("floor", "Stay on the taskbar"), ("desktop", "Explore windows & edges"),
                                       ("all", "Explore all monitors")],
                "Where your companion may wander when nothing is happening")
        f.combo("cursor_interaction", "Mouse cursor", [("ignore", "Ignore"), ("look", "Look at it (default)"),
                                                       ("follow", "Follow it"), ("chase", "Chase it (playful)"),
                                                       ("run_away", "Avoid it")],
                "'Look at it' still lets curiosity win now and then")
        f.combo("click_reaction", "Single-click reaction", [("random", "Surprise me"), ("wave", "Wave"),
                                                             ("jump", "Jump"), ("spin", "Spin"), ("look", "Look at me")])
        f.combo("celebration_style", "Celebration", [("", "Character default"), ("jump", "Jump"), ("spin", "Spin"),
                                                      ("wag", "Happy wiggle"), ("dance", "Dance"), ("flip", "Back-flip"),
                                                      ("cheer", "Cheer")])
        f.toggle("petting", "Petting", "Rub the cursor back and forth over your companion")
        f.toggle("edge_peeking", "Peek over screen edges")
        f.toggle("stay_near_during_focus", "Stay close during focus sessions")
        self.v.addWidget(look)

        life = Card("Daily life", "Your companion has its own routine: it wakes up, wanders, gets up to mischief, "
                                  "leaves for a while and comes back - and stays disciplined when you focus.")
        f = Form(self, life)
        f.slider("playfulness", "Playfulness", 0.0, 1.0, "Calm  ↔  playful  ↔  chaotic")
        f.slider("roaming_frequency", "Roaming frequency", 0.0, 1.0, "Rare  ↔  regular  ↔  frequent")
        f.slider("chatter", "Unsolicited dialogue", 0.0, 1.0, "Quiet  ↔  some  ↔  chatty")
        f.combo("focus_behavior", "During focus sessions",
                [("strict", "Strict focus: only essential warnings"),
                 ("balanced", "Balanced: occasional silent reactions"),
                 ("playful", "Playful: more reactions between tasks")])
        f.combo("roam_area", "Where it spends time",
                [("edges", "Screen edges and corners"), ("mostly_edges", "Mostly edges, occasional exploration"),
                 ("desktop", "Roam around the desktop")])
        f.toggle("leave_and_return", "Leave and come back", "Walks off-screen now and then and returns with an entrance")
        f.toggle("quiet_mode", "Quiet mode", "No chatter and no unsolicited interactions - essential warnings only")
        f.combo("close_button", "Closing the dashboard",
                [("ask", "The companion asks: nap, stay or quit"), ("stay", "Just hide the dashboard"),
                 ("nap", "Hide the dashboard; the companion naps")],
                "The dashboard's X never quits FocusBuddy - use Quit FocusBuddy in the tray or right-click menu")
        f.toggle("wake_scene", "Wake-up scene at start", "Curled up at the screen edge, wakes, stretches, enters")
        f.toggle("hide_in_fullscreen", "Step aside for full-screen apps", "Hides during presentations, videos and games")
        self.v.addWidget(life)

        app_card = Card("Appearance")
        f = Form(self, app_card)
        f.combo("renderer", "Renderer", [("production", "3D models (from the character sheets, default)"),
                                         ("illustrated", "Illustrated (flat character-sheet art)"),
                                         ("3d", "3D Procedural (lightweight)"), ("2d", "2D Fallback (flat vector art)")],
                "3D models use your graphics card when available, otherwise a fast CPU renderer; "
                "Illustrated draws the flat character-sheet art")
        f.combo("render_quality", "3D quality", [("low", "Low (simple lighting, battery saver)"),
                                                 ("normal", "Medium (lighting + shadows)"),
                                                 ("high", "High (sharper, bolder outlines, 60 FPS)"),
                                                 ("ultra", "Ultra (best quality, soft shadows)")])
        f.toggle("auto_quality", "Adaptive 3D quality", "Lower the 3D quality automatically if your PC can't keep up")
        f.toggle("shadows", "Shadow", "Soft contact shadow under the character")
        f.slider("companion_opacity", "Opacity", 0.3, 1.0)
        f.combo("animation_fps", "Animation frame rate", [(30, "30 FPS (battery saver)"), (60, "60 FPS"),
                                                           (120, "120 FPS (high refresh displays)")])
        f.toggle("show_companion", "Show the companion on the desktop",
                 "Hidden companions come back from here, the Home page or the tray icon (Ctrl+H)")
        f.toggle("always_on_top", "Stay above other windows")
        f.toggle("click_through", "Click-through mode", "Mouse clicks pass through the character")
        f.toggle("speech_bubbles", "Chatty speech bubbles", "Questions with buttons are always shown")
        self.v.addWidget(app_card)

        adv = Card("Advanced animation", "Low / Normal / High - Normal is the default.")
        fa = Form(self, adv)
        lvls = [("low", "Low"), ("normal", "Normal"), ("high", "High")]
        fa.combo("animation_smoothness", "Animation smoothness", lvls, "How softly poses blend into each other")
        fa.combo("movement_smoothness", "Movement smoothness", lvls, "Gentler acceleration and braking")
        fa.combo("secondary_motion", "Secondary motion", lvls, "Tail, ears, head, hair and antenna follow-through")
        fa.combo("physics_strength", "Physics strength", lvls, "Squash & stretch, landing impact, springiness")
        fa.combo("idle_activity_level", "Idle activity", lvls, "How often small idle actions happen")
        fa.combo("character_energy", "Character energy", lvls, "Baseline energy: sleepy  ↔  bouncy")
        fa.toggle("particles", "Particle effects", "Tiny dust, landing puffs, confetti and robot sparks")
        fa.toggle("time_of_day_behavior", "Time-of-day behaviour", "A little livelier in the morning, sleepier at night")
        fa.toggle("surprise_events", "Surprise moments", "Rare little surprises (a pounce, a spin, a glitch...)")
        fa.combo("emotion_intensity", "Emotion intensity", lvls, "How big the reactions are (trembling, steam, flushing)")
        fa.combo("lifelike_level", "Lifelike detail", [("off", "Off")] + lvls,
                 "Lips that move when it talks, eye micro-movements, eyebrow flashes, small smiles and sighs")
        fa.combo("cursor_reactions", "Cursor reactions", [("off", "Off"), ("subtle", "Subtle (eyes and head follow)"),
                                                          ("playful", "Playful (may pounce at it)")])
        fa.toggle("window_interactions", "Window interactions",
                  "Lean on / peek around windows, react when a window is dragged toward it (never clicks anything)")
        fa.toggle("stumbles", "Clumsy moments", "Very rarely trips over its own feet (never during focus)")
        fa.toggle("angry_reactions", "Grumpy reactions",
                  "Playful stomping, steam and a cartoon \"HEY!\" when you ignore it - off = gentle head shakes")
        self.v.addWidget(adv)

        snd = Card("Sound")
        f2 = Form(self, snd)
        f2.combo("sound_level", "Sound effects", [("off", "Off"), ("low", "Low"), ("medium", "Medium"), ("high", "High")])
        f2.toggle("footstep_sounds", "Footsteps", "Synced to real foot contacts")
        self.v.addWidget(snd)

        dev = Card("Developer", "Tools for character and animation development.")
        f3 = Form(self, dev)
        f3.toggle("debug_overlay", "Animation debug overlay",
                  "FPS, state, action, emotion, velocity, gait phase, foot IK, energy, path and target on screen")
        f3.toggle("debug_skeleton", "Draw skeleton on companion", "Bones, IK targets and foot contacts")
        b = QPushButton("Open Animation Studio")
        b.clicked.connect(self._studio)
        f3.row("Animation Studio", b, "Preview every animation, emotion and speed")
        pg = QPushButton("Open Companion Playground")
        pg.clicked.connect(lambda: self.app.show_dashboard("playground") if hasattr(self.app, "show_dashboard") else None)
        f3.row("Companion Playground", pg, "Trigger any action, build sequences, inspect the skeleton")
        self.v.addWidget(dev)
        self.v.addStretch(1)

    def _size_preset(self, key: str) -> None:
        for k, b in self.size_group:
            b.setChecked(k == key)
        self.app.settings.set("size_preset", key)
        val = self.app.preset_size(key) if hasattr(self.app, "preset_size") else None
        if val is not None:
            lo, hi = SIZE_RANGE
            self.size_slider.blockSignals(True)
            self.size_slider.setValue(int(round(100 * (val - lo) / (hi - lo))))
            self.size_slider.blockSignals(False)

    def _mark_custom(self) -> None:
        for k, b in self.size_group:
            b.setChecked(k == "custom")
        self.app.settings.set("size_preset", "custom")

    def _studio(self) -> None:
        from .studio import AnimationStudio
        self._studio_win = AnimationStudio(self.app.settings["character"])
        self._studio_win.show()

    def _choose(self, cid):
        self.app.set_character(cid)
        self.refresh()

    def refresh(self):
        cur = self.app.settings["character"]
        for c in self.cards:
            c.setChecked(c.spec.id == cur)
        spec = get_character(cur)
        from ..companion.characters import asset_label
        lic = spec.extra.get("license") if spec.extra else ""
        self.desc.setText(f"{spec.default_name} the {spec.display_name.lower()}: {spec.tagline}  "
                          f"Personality: {', '.join(spec.traits)}.  Look: {asset_label(spec)}."
                          + (f"  Licence: {lic}" if lic else ""))
        self.name_edit.setPlaceholderText(spec.default_name)


# =========================================================================== Settings
class SettingsPage(Page):
    title = "Settings"

    def __init__(self, app):
        super().__init__(app)
        self.header("Settings", "Changes apply immediately.")
        s = app.settings

        from .account import AccountCard
        self.account = AccountCard(self, app)
        self.v.addWidget(self.account)

        gen = Card("General")
        f = Form(self, gen)
        nm = QLineEdit(s["user_name"])
        nm.setPlaceholderText("Your name (optional)")
        nm.setFixedWidth(220)
        nm.editingFinished.connect(lambda: s.set("user_name", nm.text().strip()))
        f.row("Your name", nm, "Used for greetings only")
        f.toggle("start_with_windows", "Start with Windows", on_change=app.set_autostart)
        f.toggle("start_minimized", "Start quietly with Windows",
                 "When Windows starts it, FocusBuddy waits in the tray. Opening it yourself always shows "
                 "the dashboard")
        f.toggle("notifications_enabled", "Desktop notifications")
        self.v.addWidget(gen)

        look = Card("Appearance", "Bright and cheerful, switched instantly. \"Use system setting\" follows "
                                  "light / dark mode in "
                                  "Windows Settings > Personalization > Colors.")
        f = Form(self, look)
        f.combo("theme", "App theme", [("system", "Use system setting"), ("light", "Light"), ("dark", "Dark")])
        f.combo("accent", "Accent colour", [("brand", "Sunset Pop - violet & watermelon (FocusBuddy)"),
                                            ("peach", "Mango - orange & sunflower"), ("mint", "Lagoon - jade & sky"),
                                            ("sky", "Ocean - cobalt & cyan"), ("rose", "Berry - raspberry & violet"),
                                            ("system", "Windows accent colour")],
                "Buttons, toggles, charts and highlights")
        f.combo("motion", "Motion", [("full", "Full - fluid transitions and animations"),
                                     ("reduced", "Reduced - gentle fades only"), ("off", "Off - no animation")],
                "Windows' \"Animation effects\" switch also reduces motion")
        self.v.addWidget(look)

        mon = Card("Monitoring")
        f = Form(self, mon)
        f.toggle("monitoring_enabled", "Enable distraction detection", on_change=app.set_monitoring)
        f.combo("sensitivity", "Detection sensitivity",
                [("relaxed", "Relaxed"), ("balanced", "Balanced"), ("strict", "Strict")],
                "How quickly a distraction's score grows", on_change=lambda v: app.settings.apply_sensitivity_preset(v))
        f.spin("min_distraction_seconds", "Minimum distraction duration", 1, 300, " s",
               "Ignore quick glances shorter than this")
        f.spin("daily_distraction_allowance_min", "Daily distraction allowance", 0, 600, " min",
               "After this much distraction time, scores rise faster")
        f.spin("idle_threshold_seconds", "Away after", 30, 3600, " s", "No input for this long = you're away (companion sleeps)")
        f.toggle("context_classifier", "Smart context", "e.g. treat YouTube programming tutorials as less distracting")
        f.toggle("use_title_fallback", "Window title fallback", "Guess the website from the title when the extension isn't connected")
        f.toggle("detect_games", "Detect games automatically",
                 "Anything started from a Steam, Epic, Riot, Xbox, EA, Ubisoft, Battle.net or HoYoverse game folder counts as distracting")
        self.v.addWidget(mon)

        fe = Card("During focus sessions", "Starting a focus session is your go-ahead: distractions get closed for you.")
        f = Form(self, fe)
        f.combo("focus_enforcement", "Distracting sites & apps",
                [("auto_close", "Close automatically after a countdown (recommended)"),
                 ("instant", "Block instantly (3 s, the extension stops sites as they load)"),
                 ("same", "Use the blocking mode below")],
                "“Keep Open” and “Give me a few minutes” still work - within your extra-time limit",
                on_change=lambda _v: app._sync_policy())
        f.spin("focus_speedup", "Notice distractions faster by", 1, 5, "×",
               "During a session a distraction's score builds up this much faster")
        self.v.addWidget(fe)

        th = Card("Score thresholds", "Distraction score is 0–100. Bands decide how your companion responds.")
        f = Form(self, th)
        f.spin("threshold_mild", "Mild (notice)", 1, 99)
        f.spin("threshold_warning", "Warning", 1, 99)
        f.spin("threshold_strong", "Strong (intervention)", 1, 100)
        self.v.addWidget(th)

        blk = Card("Blocking")
        f = Form(self, blk)
        f.combo("blocking_mode", "Blocking mode (outside focus sessions)",
                [("warn", "Warnings only"), ("ask", "Ask before blocking"), ("auto", "Automatic blocking")],
                "Automatic blocking only closes sites/apps you marked auto-close, after a countdown",
                on_change=self._mode_changed)
        f.spin("grace_period_seconds", "Grace period (countdown)", 3, 120, " s")
        f.spin("escalation_spacing_seconds", "Time between escalation steps", 3, 600, " s")
        f.combo("website_action", "For distracting websites",
                [("close_tab", "Close the tab"), ("redirect", "Replace with a focus page"), ("minimize", "Minimize the browser")],
                "With the extension the exact tab is closed; without it, see the shortcut option below")
        f.toggle("keyboard_tab_close", "Close tabs without the extension",
                 "Sends the browser's own Ctrl+W - only to the browser window in front, and only while it still shows "
                 "that exact tab. Otherwise the browser is minimized")
        f.combo("app_action", "For distracting applications",
                [("close", "Close (polite - apps can still ask to save; minimized if they don't close)"),
                 ("minimize", "Minimize window")])
        f.toggle("allow_terminal_close", "Allow acting on terminals", "Off by default - terminals are protected")
        f.combo("override_policy", "Extra time (snooze / ignore / keep open)",
                [("once_per_site", "Once per site per session"), ("once_per_session", "Once per session (any site)"),
                 ("unlimited", "Unlimited"), ("never", "Never")],
                "After it's used up, only “Return to Focus” and “Close” are offered")
        f.spin("snooze_short_min", "“Give me a few minutes” (short)", 1, 60, " min")
        f.spin("snooze_long_min", "“Give me a few minutes” (long)", 1, 120, " min")
        self.v.addWidget(blk)

        foc = Card("Focus")
        f = Form(self, foc)
        f.spin("focus_minutes", "Default focus duration", 1, 600, " min")
        f.spin("break_minutes", "Break duration", 1, 60, " min")
        f.spin("long_break_minutes", "Long break", 1, 120, " min")
        f.spin("long_break_every", "Long break every", 2, 12, " sessions")
        f.spin("daily_focus_goal_min", "Daily focus goal", 15, 960, " min")
        f.toggle("auto_start_break", "Start break automatically")
        f.toggle("auto_start_next", "Auto-start next session after a break")
        f.toggle("relax_blocking_on_break", "Relax blocking during breaks")
        self.v.addWidget(foc)

        smart = Card("Smart assistant", "Everything here runs on this computer. Nothing is sent anywhere.")
        f = Form(self, smart)
        f.toggle("coach_nudges", "Focus coach nudges",
                 "Offers a session when your usual distraction-prone hour starts")
        f.toggle("context_reactions", "Context reactions",
                 "Your companion notices flow, restlessness, long stretches and late nights and reacts")
        f.toggle("daily_report", "Evening summary", "A short recap of your day from your companion after 6 pm")
        f.toggle("ai_chat_llm", "Use a local AI model for chat",
                 "If Ollama is running on this PC, free conversation goes to it; otherwise built-in replies")
        f.text("ai_model", "Local model", "Automatic", "An installed Ollama model name, e.g. llama3.2 - empty picks one")
        f.text("ollama_url", "Ollama address", "http://127.0.0.1:11434",
               "Only this computer's address (localhost) is ever used")
        self.v.addWidget(smart)

        rt = Card("Routines", "Your companion's day around yours: naps, a morning hello and a wind-down.")
        f = Form(self, rt)
        f.toggle("morning_checkin", "Morning check-in", "The first time you're here each day: hello, and your one thing")
        f.toggle("wind_down", "Evening wind-down", "A gentle reminder to wrap up, with your day, then a nap")
        f.text("wind_down_time", "Wind-down time", "21:30", "24-hour time, e.g. 21:30")
        f.toggle("nap_when_away", "Nap while I'm away",
                 "It walks off-screen to nap when you leave the PC, and comes back when you return")
        f.spin("nap_away_minutes", "Nap after you've been away", 2, 120, " min")
        f.text("nap_schedule", "Nap schedule", "e.g. 13:30, 16:00", "Times it goes for a nap by itself (empty = none)")
        f.spin("nap_schedule_minutes", "Scheduled nap length", 5, 120, " min")
        f.combo("ambient_mode", "Ambient sound plays", [("focus", "During focus sessions"), ("always", "Always")],
                "Pick the sound on the Focus page")
        self.v.addWidget(rt)

        br = Card("Browser integration", "Exact website detection and safe tab closing in Chrome, Edge and Brave "
                                         "(Firefox: about:debugging → Load Temporary Add-on → manifest.json).")
        from .wizard import ExtensionSteps
        self.ext_steps = ExtensionSteps(app)
        br.add(self.ext_steps)
        self.v.addWidget(br)

        data = Card("Data")
        row = QHBoxLayout()
        b1 = QPushButton("Open data folder")
        b1.clicked.connect(lambda: open_path(str(app.data_dir)))
        b2 = QPushButton("Run setup wizard again")
        b2.clicked.connect(app.show_wizard)
        row.addWidget(b1)
        row.addWidget(b2)
        row.addStretch(1)
        data.add(row)
        if app.db.degraded:
            warn = label("⚠ The database could not be opened, so data is kept in memory for this session only.", wrap=True)
            warn.setObjectName("Error")
            data.add(warn)
        self.v.addWidget(data)
        self.v.addStretch(1)

    def _mode_changed(self, mode):
        if mode == "auto":
            QMessageBox.information(
                self, "Automatic blocking",
                "Automatic blocking is on.\n\nFocusBuddy will only close websites/apps that you marked "
                "\"Auto-close\" in Sites & Apps, always after a visible countdown you can cancel. "
                "System apps are never touched.")

    def refresh(self):
        self.ext_steps._poll()


def open_path(path: str) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
    except OSError:
        pass
