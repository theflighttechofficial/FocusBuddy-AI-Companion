"""Command palette (Ctrl+K): one search box for every action, page, setting and companion.

Type a few letters - "dar" finds "Switch to dark mode", "st25" finds "Start 25-minute focus" - then
Enter runs the highlighted row. Up / Down / Page Up / Page Down move, Esc closes, the mouse works too.
With an empty box the last few commands come first ("Recent", stored locally in the settings).

The palette is a child overlay of the dashboard window (no extra top-level window): a soft scrim
over the page and a frosted panel that fades and slides in. Matching is a small fuzzy subsequence
score (word starts, runs of consecutive letters and prefixes score higher), pure Python and tested
without Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from PySide6.QtCore import QEasingCurve, QEvent, QPointF, QRectF, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLineEdit, QVBoxLayout, QWidget

from .theme import C, STATE

SECTION_ORDER = ("Recent", "Actions", "Companions", "Pages", "Settings & help")
MAX_RECENT = 5


@dataclass
class Command:
    title: str
    section: str                              # Actions | Companions | Pages | Settings & help
    run: Callable[[], None]
    hint: str = ""                            # right-hand side: a shortcut or where it lives
    icon: str = ""                            # a small emoji glyph
    keywords: str = ""                        # extra words that should find it
    key: str = ""                             # stable id (recent list)
    matched: list = field(default_factory=list)

    def __post_init__(self):
        if not self.key:
            self.key = f"{self.section}:{self.title}"


# ====================================================================== matching
def fuzzy_score(query: str, text: str) -> Optional[tuple[float, list[int]]]:
    """Score ``query`` as a subsequence of ``text`` (case-insensitive); None when it is not one.

    Returns (score, matched character positions). Higher is better: a match at the start of the
    text or of a word, consecutive letters and a short text score more."""
    q = "".join(query.lower().split())
    if not q:
        return 0.0, []
    t = text.lower()
    # greedy forward pass, preferring word starts: try every start of the first letter, keep the best
    best: Optional[tuple[float, list[int]]] = None
    starts = [i for i, ch in enumerate(t) if ch == q[0]]
    for s0 in starts[:12]:
        pos = [s0]
        ok = True
        for ch in q[1:]:
            j = pos[-1] + 1
            # jump to the next word start holding ch if there is one before a plain hit would be far
            k = t.find(ch, j)
            if k < 0:
                ok = False
                break
            w = j
            while True:
                w = t.find(ch, w)
                if w < 0:
                    break
                if w == 0 or not t[w - 1].isalnum():
                    break
                w += 1
            if w >= 0 and k != w and (k - j) > 0:
                k = w
            pos.append(k)
        if not ok:
            continue
        score = 0.0
        for n, p in enumerate(pos):
            word_start = p == 0 or not t[p - 1].isalnum()
            score += 3.0 if word_start else 1.0
            if n and p == pos[n - 1] + 1:
                score += 2.0
        if pos[0] == 0:
            score += 4.0
        gaps = sum(1 for n in range(1, len(pos)) if pos[n] != pos[n - 1] + 1)
        score -= gaps * 1.5 + (pos[-1] - pos[0]) * 0.05 + len(t) * 0.01
        if best is None or score > best[0]:
            best = (score, pos)
    # a plain substring (above all at a word start: "dar" in "dark mode") beats scattered letters
    k = t.find(q)
    while k >= 0:
        ws = k == 0 or not t[k - 1].isalnum()
        score = len(q) * 5.0 + (8.0 if ws else 3.0) + (4.0 if k == 0 else 0.0) - len(t) * 0.01
        if best is None or score > best[0]:
            best = (score, list(range(k, k + len(q))))
        k = t.find(q, k + 1)
    return best


def rank(commands: list[Command], query: str, recent: list[str] | None = None) -> list[Command]:
    """Commands to show for ``query``: grouped by section (Recent first when the query is empty),
    best matches first inside a section."""
    out: list[tuple[int, float, Command]] = []
    query = query.strip()
    if not query:
        rec = [k for k in (recent or [])]
        by_key = {c.key: c for c in commands}
        shown = set()
        for k in rec:
            c = by_key.get(k)
            if c is not None and k not in shown:
                c.matched = []
                out.append((0, -len(shown), Command(c.title, "Recent", c.run, c.hint, c.icon, c.keywords, c.key)))
                shown.add(k)
        for c in commands:
            c.matched = []
            out.append((SECTION_ORDER.index(c.section) if c.section in SECTION_ORDER else 9, 0.0, c))
        out.sort(key=lambda r: (r[0], -r[1]))
        return [c for _s, _v, c in out]
    for c in commands:
        m = fuzzy_score(query, c.title)
        kw = fuzzy_score(query, c.keywords) if c.keywords else None
        if m is None and kw is None:
            continue
        if m is not None:
            score, c.matched = m[0], m[1]
        else:
            score, c.matched = kw[0] * 0.6 - 2.0, []
        sec = SECTION_ORDER.index(c.section) if c.section in SECTION_ORDER else 9
        out.append((sec, score, c))
    if not out:
        return []
    # sections keep their order, but a section whose best hit is much better comes first
    best = {}
    for sec, sc, _c in out:
        best[sec] = max(best.get(sec, -1e9), sc)
    out.sort(key=lambda r: (-best[r[0]], r[0], -r[1]))
    return [c for _s, _v, c in out]


# ====================================================================== widget
class _List(QWidget):
    """The painted result rows (section headers, icon, title with matched letters, hint)."""

    activated = Signal(int)
    hovered = Signal(int)

    ROW = 42
    HEAD = 28

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.items: list[Command] = []
        self.sel = 0
        self.top = 0.0                        # scroll offset (px)
        self._rows: list[tuple[QRectF, int]] = []

    def layout_rows(self) -> list[tuple[str, int, float]]:
        """[(kind, index, y)] with kind "head" or "row" (unscrolled y)."""
        out, y, sec = [], 0.0, None
        for i, c in enumerate(self.items):
            if c.section != sec:
                sec = c.section
                out.append(("head", i, y))
                y += self.HEAD
            out.append(("row", i, y))
            y += self.ROW
        return out

    def content_height(self) -> float:
        rows = self.layout_rows()
        if not rows:
            return 0.0
        kind, _i, y = rows[-1]
        return y + (self.ROW if kind == "row" else self.HEAD)

    def ensure_visible(self) -> None:
        for kind, i, y in self.layout_rows():
            if kind == "row" and i == self.sel:
                h = self.height()
                head = self.HEAD if (i == 0 or self.items[i - 1].section != self.items[i].section) else 0
                if y - head < self.top:
                    self.top = max(0.0, y - head)
                elif y + self.ROW > self.top + h:
                    self.top = y + self.ROW - h
                break
        self.update()

    def wheelEvent(self, ev):
        self.top = max(0.0, min(self.top - ev.angleDelta().y() * 0.5, max(0.0, self.content_height() - self.height())))
        self.update()

    def mouseMoveEvent(self, ev):
        for r, i in self._rows:
            if r.contains(ev.position()):
                if i != self.sel:
                    self.sel = i
                    self.hovered.emit(i)
                    self.update()
                break

    def mousePressEvent(self, ev):
        for r, i in self._rows:
            if r.contains(ev.position()):
                self.activated.emit(i)
                break

    def paintEvent(self, _ev):
        from .widgets import font
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        g.setClipRect(self.rect())
        self._rows = []
        if not self.items:
            g.setPen(QColor(C["faint"]))
            g.setFont(font(14))
            g.drawText(self.rect(), Qt.AlignCenter, "No matches - try fewer letters")
            g.end()
            return
        w = self.width()
        primary = QColor(C["primary"])
        for kind, i, y0 in self.layout_rows():
            y = y0 - self.top
            if y > self.height() or y < -self.ROW:
                continue
            c = self.items[i]
            if kind == "head":
                g.setPen(QColor(C["faint"]))
                g.setFont(font(11, 700))
                g.drawText(QRectF(18, y + 6, w - 36, self.HEAD - 6), Qt.AlignLeft | Qt.AlignVCenter, c.section.upper())
                continue
            r = QRectF(8, y + 2, w - 16, self.ROW - 4)
            self._rows.append((r, i))
            if i == self.sel:
                grad = QLinearGradient(r.topLeft(), r.topRight())
                a = QColor(primary)
                a.setAlpha(46 if not STATE.get("dark") else 70)
                b = QColor(C.get("primary_grad", C["primary"]))
                b.setAlpha(18 if not STATE.get("dark") else 34)
                grad.setColorAt(0.0, a)
                grad.setColorAt(1.0, b)
                path = QPainterPath()
                path.addRoundedRect(r, 12, 12)
                g.fillPath(path, grad)
                bar = QPainterPath()
                bar.addRoundedRect(QRectF(r.left() + 3, r.top() + 10, 3, r.height() - 20), 1.5, 1.5)
                g.fillPath(bar, primary)
            # icon
            g.setFont(font(16))
            g.setPen(QColor(C["text"]))
            g.drawText(QRectF(r.left() + 12, r.top(), 28, r.height()), Qt.AlignCenter, c.icon or "•")
            # title with matched letters in the accent colour
            x = r.left() + 48
            hint_w = 0.0
            if c.hint:
                g.setFont(font(12, 600))
                hint_w = g.fontMetrics().horizontalAdvance(c.hint) + 20
            fx = font(14, 600 if i == self.sel else 500)
            g.setFont(fx)
            fm = g.fontMetrics()
            avail = r.right() - hint_w - 12 - x
            title = fm.elidedText(c.title, Qt.ElideRight, int(avail))
            matched = set(c.matched) if title == c.title else set()
            base_y = r.center().y() + fm.ascent() / 2 - 2
            for ci, ch in enumerate(title):
                g.setPen(primary if ci in matched else QColor(C["text"]))
                g.drawText(QPointF(x, base_y), ch)
                x += fm.horizontalAdvance(ch)
            if c.hint:
                g.setFont(font(12, 600))
                hr = QRectF(r.right() - hint_w - 4, r.center().y() - 11, hint_w - 6, 22)
                pill = QPainterPath()
                pill.addRoundedRect(hr, 11, 11)
                bg = QColor(C["surface_alt"])
                bg.setAlpha(200)
                g.fillPath(pill, bg)
                g.setPen(QColor(C["muted"]))
                g.drawText(hr, Qt.AlignCenter, c.hint)
        g.end()


class CommandPalette(QWidget):
    """Overlay over ``host`` (the dashboard's central widget). ``open(commands)`` shows it."""

    closed = Signal()
    ran = Signal(str)                          # the key of the command that ran

    WIDTH = 620

    def __init__(self, host: QWidget, recent_get=None, recent_set=None):
        super().__init__(host)
        self.host = host
        self.recent_get = recent_get or (lambda: [])
        self.recent_set = recent_set or (lambda keys: None)
        self.commands: list[Command] = []
        self.setAttribute(Qt.WA_StyledBackground, False)
        self.setFocusPolicy(Qt.StrongFocus)
        self.panel = QWidget(self)
        self.panel.setObjectName("Transparent")
        lay = QVBoxLayout(self.panel)
        lay.setContentsMargins(14, 14, 14, 10)
        lay.setSpacing(8)
        self.edit = QLineEdit()
        self.edit.setObjectName("Search")
        self.edit.setPlaceholderText("Type a command, page or setting…")
        self.edit.setMinimumHeight(44)
        self.edit.textChanged.connect(self._filter)
        self.edit.installEventFilter(self)
        lay.addWidget(self.edit)
        self.list = _List()
        self.list.activated.connect(self._run)
        lay.addWidget(self.list, 1)
        self.foot_h = 22
        lay.addSpacing(self.foot_h)
        self._t = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._step)
        self._anim.finished.connect(self._anim_done)
        self._closing = False
        self._fx = QGraphicsOpacityEffect(self.panel)
        self._fx.setOpacity(1.0)
        self.panel.setGraphicsEffect(self._fx)
        host.installEventFilter(self)
        self.hide()

    # ------------------------------------------------------------ api
    def open(self, commands: list[Command], text: str = "") -> None:
        from .motion import animated
        self.commands = commands
        self.setGeometry(self.host.rect())
        self._place()
        self.raise_()
        self.show()
        self.edit.setText(text)
        self._filter(text)
        self.edit.setFocus()
        self.edit.selectAll()
        self._closing = False
        if animated():
            self._anim.stop()
            self._anim.setStartValue(self._t)
            self._anim.setEndValue(1.0)
            self._anim.start()
        else:
            self._step(1.0)

    def close_palette(self) -> None:
        from .motion import animated
        if not self.isVisible():
            return
        self._closing = True
        if animated():
            self._anim.stop()
            self._anim.setStartValue(self._t)
            self._anim.setEndValue(0.0)
            self._anim.start()
        else:
            self._step(0.0)
            self._anim_done()

    def is_open(self) -> bool:
        return self.isVisible() and not self._closing

    # ------------------------------------------------------------ internals
    def _anim_done(self) -> None:
        if self._closing:
            self.hide()
            self._closing = False
            self.closed.emit()

    def _step(self, v) -> None:
        self._t = float(v)
        self._fx.setOpacity(max(0.0, min(1.0, self._t)))
        self._fx.setEnabled(self._t < 0.999)                 # no off-screen pass once fully shown
        self._place()
        self.update()

    def _place(self) -> None:
        w = min(self.WIDTH, max(360, self.width() - 48))
        rows = len(self.list.items)
        h_list = min(self.list.content_height() if rows else 80.0, 9 * _List.ROW + 2 * _List.HEAD)
        h = int(14 + 44 + 8 + max(80.0, h_list) + self.foot_h + 10 + 4)
        h = min(h, max(240, self.height() - 120))
        x = (self.width() - w) // 2
        y = int(max(40, self.height() * 0.12) - (1.0 - self._t) * 18)
        self.panel.setGeometry(x, y, w, h)

    def _filter(self, text: str) -> None:
        self.list.items = rank(self.commands, text, self.recent_get())
        self.list.sel = 0
        self.list.top = 0.0
        self._place()
        self.list.update()
        self.update()

    def _run(self, idx: int) -> None:
        if not (0 <= idx < len(self.list.items)):
            return
        c = self.list.items[idx]
        rec = [c.key] + [k for k in self.recent_get() if k != c.key]
        self.recent_set(rec[:MAX_RECENT])
        self.close_palette()
        self.ran.emit(c.key)
        try:
            c.run()
        except Exception:  # noqa: BLE001 - a failing command must not take the window down
            import logging
            logging.getLogger("focusbuddy.ui").exception("Command %r failed", c.title)

    def _move(self, d: int) -> None:
        n = len(self.list.items)
        if not n:
            return
        self.list.sel = max(0, min(n - 1, self.list.sel + d))
        self.list.ensure_visible()

    def eventFilter(self, obj, ev):
        if obj is self.host and ev.type() == QEvent.Resize and self.isVisible():
            self.setGeometry(self.host.rect())
            self._place()
        elif obj is self.edit and ev.type() == QEvent.KeyPress:
            k = ev.key()
            if k == Qt.Key_Down:
                self._move(1)
                return True
            if k == Qt.Key_Up:
                self._move(-1)
                return True
            if k == Qt.Key_PageDown:
                self._move(6)
                return True
            if k == Qt.Key_PageUp:
                self._move(-6)
                return True
            if k in (Qt.Key_Return, Qt.Key_Enter):
                self._run(self.list.sel)
                return True
            if k == Qt.Key_Escape:
                self.close_palette()
                return True
        return super().eventFilter(obj, ev)

    def mousePressEvent(self, ev):
        if not self.panel.geometry().contains(ev.position().toPoint()):
            self.close_palette()
        ev.accept()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.close_palette()
            return
        super().keyPressEvent(ev)

    def paintEvent(self, _ev):
        from .widgets import font
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        t = self._t
        scrim = QColor(12, 10, 30) if STATE.get("dark") else QColor(40, 30, 70)
        scrim.setAlpha(int((110 if STATE.get("dark") else 70) * t))
        g.fillRect(self.rect(), scrim)
        r = QRectF(self.panel.geometry())
        if t <= 0.01:
            g.end()
            return
        g.setOpacity(t)
        # soft shadow
        for k in range(6, 0, -1):
            sh = QPainterPath()
            sh.addRoundedRect(r.adjusted(-k * 3, -k * 2 + 6, k * 3, k * 4 + 6), 24 + k * 2, 24 + k * 2)
            g.fillPath(sh, QColor(20, 10, 50, 7))
        body = QPainterPath()
        body.addRoundedRect(r, 22, 22)
        g.fillPath(body, QColor(C["bg"]))                     # opaque base: the page never shows through
        surf = QColor(C["surface"])
        g.fillPath(body, surf)
        edge = QLinearGradient(r.topLeft(), r.bottomRight())
        e1 = QColor(C["primary"])
        e1.setAlpha(150)
        e2 = QColor(C.get("primary_grad", C["primary"]))
        e2.setAlpha(110)
        edge.setColorAt(0.0, e1)
        edge.setColorAt(1.0, e2)
        g.setPen(QPen(edge, 1.4))
        g.drawPath(body)
        # footer: keyboard help
        g.setPen(QColor(C["faint"]))
        g.setFont(font(11, 500))
        g.drawText(QRectF(r.left() + 20, r.bottom() - self.foot_h - 8, r.width() - 40, self.foot_h),
                   Qt.AlignLeft | Qt.AlignVCenter, "↑↓  move     ↵  run     esc  close")
        n = len(self.list.items)
        g.drawText(QRectF(r.left() + 20, r.bottom() - self.foot_h - 8, r.width() - 40, self.foot_h),
                   Qt.AlignRight | Qt.AlignVCenter, f"{n} result{'s' if n != 1 else ''}")
        g.end()
