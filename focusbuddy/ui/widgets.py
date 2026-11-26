"""Reusable dashboard widgets: cards, stat tiles, toggles, ring, charts, character preview."""

from __future__ import annotations

import math
import time

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractButton, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QToolTip, QVBoxLayout, QWidget)

from ..companion.animation import Animator
from ..companion.characters import CharacterSpec
from ..companion.renderer import make_renderer
from ..companion.states import CompanionState as S
from .motion import CountUp, GlassCard, animated
from .theme import C, FONT_DISPLAY, FONT_FAMILY, mix, tok

# Chart series colours: palette tokens (resolved when painting, so a theme switch recolours them live;
# the light / dark pair is a CVD-safe violet / orange)
CH_FOCUS = "chart_focus"
CH_DISTRACT = "chart_distract"


def label(text: str = "", obj: str | None = None, wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if obj:
        lb.setObjectName(obj)
    lb.setWordWrap(wrap)
    return lb


def font(px: int, weight: int = 400) -> QFont:
    f = QFont()
    f.setFamilies([n.strip().strip("'\"") for n in FONT_FAMILY.split(",")])
    f.setPixelSize(px)
    f.setWeight(QFont.Weight(weight))
    return f


class Card(GlassCard):
    """Frosted content card (title, optional subtitle) - see motion.GlassCard for the surface."""

    def __init__(self, title: str | None = None, subtitle: str | None = None, soft: bool = False, shadow: bool = True,
                 hover: bool = True):
        super().__init__(hover=hover)
        self.title = title or ""
        self.lay = QVBoxLayout(self)
        m = GlassCard.MARGIN
        self.lay.setContentsMargins(m + 22, m + 18, m + 22, m + 20)
        self.lay.setSpacing(12)
        if title:
            head = QHBoxLayout()
            t = label(title, "H3")
            head.addWidget(t)
            head.addStretch(1)
            self.head = head
            self.lay.addLayout(head)
            if subtitle:
                self.lay.addWidget(label(subtitle, "Faint", wrap=True))

    def add(self, w, stretch: int = 0):
        if isinstance(w, QWidget):
            self.lay.addWidget(w, stretch)
        else:
            self.lay.addLayout(w, stretch)
        return w


class Dot(QWidget):
    """A small gradient dot in a palette colour (token or "#RRGGBB"), recoloured with the theme."""

    def __init__(self, color: str = "primary", size: int = 10):
        super().__init__()
        self.color = color
        self.setFixedSize(size, size)

    def paintEvent(self, _ev):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        c = tok(self.color)
        grad = QLinearGradient(QPointF(0, 0), QPointF(self.width(), self.height()))
        grad.setColorAt(0, QColor(c))
        grad.setColorAt(1, QColor(mix(c, "#FFFFFF", 0.45)))
        g.setPen(Qt.NoPen)
        g.setBrush(grad)
        g.drawEllipse(QRectF(self.rect()))
        g.end()


class StatTile(GlassCard):
    """KPI tile: label, big animated value, caption."""

    def __init__(self, title: str, value: str = "–", sub: str = "", accent: str = "primary"):
        super().__init__(wash=accent)
        self.setMinimumWidth(150)
        self.accent = accent
        lay = QVBoxLayout(self)
        m = GlassCard.MARGIN
        lay.setContentsMargins(m + 18, m + 14, m + 18, m + 16)
        lay.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(Dot(accent))
        top.addWidget(label(title, "Muted"))
        top.addStretch(1)
        lay.addLayout(top)
        self.value = label(value)
        self.value.setStyleSheet(f"font-family:{FONT_DISPLAY}; font-size:28px; font-weight:700;")
        lay.addWidget(self.value)
        self.sub = label(sub, "Faint", wrap=True)
        lay.addWidget(self.sub)
        self._count = None

    def set(self, value: str, sub: str | None = None, number: float | None = None, fmt=None):
        """``number`` + ``fmt`` count the value up smoothly; otherwise the text is shown as is."""
        if number is not None and fmt is not None:
            if self._count is None:
                self._count = CountUp(self.value, fmt)
            self._count.fmt = fmt
            self._count.set(number)
        else:
            self.value.setText(value)
        if sub is not None:
            self.sub.setText(sub)


class Toggle(QAbstractButton):
    """Windows 11 style toggle switch: outlined when off, accent-filled when on, animated knob."""

    W, H = 40, 20

    def __init__(self, checked: bool = False):
        super().__init__()
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(self.W + 2, self.H + 4)
        self._pos = 1.0 if checked else 0.0
        self._hover = False
        from PySide6.QtCore import QVariantAnimation
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(140)
        self._anim.valueChanged.connect(self._on_anim)
        self.toggled.connect(self._animate)

    def sizeHint(self):
        return QSize(self.W + 2, self.H + 4)

    def set_quietly(self, on: bool) -> None:
        """Show a value changed elsewhere: no toggled signal, the knob jumps to its place."""
        self.blockSignals(True)
        self.setChecked(bool(on))
        self.blockSignals(False)
        self._anim.stop()
        self._pos = 1.0 if on else 0.0
        self.update()

    def _on_anim(self, v):
        self._pos = float(v)
        self.update()

    def _animate(self, on: bool):
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def enterEvent(self, ev):
        self._hover = True
        self.update()
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._hover = False
        self.update()
        super().leaveEvent(ev)

    def paintEvent(self, _):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        if not self.isEnabled():
            g.setOpacity(0.45)
        on = self.isChecked()
        r = QRectF(1, 2, self.W, self.H)
        p = self._pos
        if on:
            g.setPen(Qt.NoPen)
            g.setBrush(QColor(C["primary_dark"] if self._hover else C["primary"]))
        else:
            g.setPen(QPen(QColor(C["muted"]), 1.2))
            g.setBrush(QColor(C["surface_hover"] if self._hover else C["surface"]))
        g.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), self.H / 2, self.H / 2)
        k = 12.0 if p > 0.5 else (11.0 if self._hover else 10.0)
        cx = r.left() + self.H / 2 + p * (self.W - self.H)
        g.setPen(Qt.NoPen)
        g.setBrush(QColor(C["on_accent"] if p > 0.5 else C["muted"]))
        g.drawEllipse(QPointF(cx, r.center().y()), k / 2, k / 2)


class ProgressRing(QWidget):
    """Gradient progress ring that eases to its value."""

    def __init__(self, size: int = 150, thickness: int = 12, color: str = "primary"):
        super().__init__()
        self.setFixedSize(size, size)
        self.value = 0.0
        self.shown = 0.0
        self.thickness = thickness
        self.color = color
        self.center_text = ""
        self.sub_text = ""
        from PySide6.QtCore import QVariantAnimation, QEasingCurve
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(1100)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on)

    def _on(self, v):
        self.shown = float(v)
        self.update()

    def set(self, value: float, text: str = "", sub: str = "", color: str | None = None):
        value = max(0.0, min(1.0, value))
        self.center_text, self.sub_text = text, sub
        if color:
            self.color = color
        if abs(value - self.value) > 1e-4:
            self.value = value
            if animated():
                self._anim.stop()
                self._anim.setStartValue(self.shown)
                self._anim.setEndValue(value)
                self._anim.start()
            else:
                self.shown = value
        self.update()

    def paintEvent(self, _):
        from PySide6.QtGui import QConicalGradient
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        t = self.thickness
        r = QRectF(t / 2 + 4, t / 2 + 4, self.width() - t - 8, self.height() - t - 8)
        track = QColor(C["primary_soft"])
        pen = QPen(track, t)
        pen.setCapStyle(Qt.RoundCap)
        g.setPen(pen)
        g.drawArc(r, 0, 360 * 16)
        v = self.shown
        if v > 0.002:
            grad = QConicalGradient(r.center(), 90)
            c1 = QColor(tok(self.color))
            c2 = QColor(mix(tok(self.color), C["primary_grad"], 0.6))
            grad.setColorAt(0.0, c1)
            grad.setColorAt(max(0.01, 1.0 - v), c2)
            grad.setColorAt(1.0, c1)
            glow = QColor(tok(self.color))
            glow.setAlpha(40)
            gp = QPen(glow, t + 8)
            gp.setCapStyle(Qt.RoundCap)
            g.setPen(gp)
            g.drawArc(r, 90 * 16, -int(360 * 16 * v))
            pen = QPen(QBrush(grad), t)
            pen.setCapStyle(Qt.RoundCap)
            g.setPen(pen)
            g.drawArc(r, 90 * 16, -int(360 * 16 * v))
        g.setPen(QColor(C["text"]))
        size = max(14, int(self.width() * 0.17))
        g.setFont(font(size, 750))
        g.drawText(QRectF(0, self.height() * 0.30, self.width(), size * 1.5), Qt.AlignCenter, self.center_text)
        g.setPen(QColor(C["muted"]))
        g.setFont(font(max(10, int(self.width() * 0.075))))
        g.drawText(QRectF(0, self.height() * 0.30 + size * 1.45, self.width(), 20), Qt.AlignCenter, self.sub_text)


class _ChartBase(QWidget):
    def __init__(self, height: int = 220):
        super().__init__()
        self.setMinimumHeight(height)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._hits: list[tuple[QRectF, str]] = []
        self.empty_text = "No data yet - it fills in as you use your computer."
        from PySide6.QtCore import QVariantAnimation, QEasingCurve
        self.grow = 1.0
        self._sig = None
        self._ga = QVariantAnimation(self)
        self._ga.setDuration(900)
        self._ga.setEasingCurve(QEasingCurve.OutCubic)
        self._ga.setStartValue(0.0)
        self._ga.setEndValue(1.0)
        self._ga.valueChanged.connect(lambda v: (setattr(self, "grow", float(v)), self.update()))

    def _animate(self, sig) -> None:
        """Bars / lines grow in when the data changes (not on every refresh)."""
        if sig == self._sig:
            return
        self._sig = sig
        if animated():
            self.grow = 0.0
            self._ga.stop()
            self._ga.start()

    def mouseMoveEvent(self, ev):
        p = ev.position()
        for rect, tip in self._hits:
            if rect.contains(p):
                QToolTip.showText(ev.globalPosition().toPoint(), tip, self)
                return
        QToolTip.hideText()

    @staticmethod
    def nice_max(v: float) -> tuple[float, float]:
        """Clean axis: top and step from the 1-2-2.5-5 series, ~4 intervals."""
        if v <= 0:
            return 1.0, 0.25
        raw = v / 4
        mag = 10 ** math.floor(math.log10(raw))
        step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
        top = math.ceil(v / step - 1e-9) * step
        return top, step

    def _empty(self, g: QPainter) -> None:
        g.setPen(QColor(C["faint"]))
        g.setFont(font(13))
        g.drawText(self.rect(), Qt.AlignCenter, self.empty_text)


def _bar_path(x, y, w, h) -> QPainterPath:
    """Column with a 4px rounded data-end and square baseline."""
    p = QPainterPath()
    r = min(7.0, w / 2, h)
    p.moveTo(x, y + h)
    p.lineTo(x, y + r)
    p.quadTo(x, y, x + r, y)
    p.lineTo(x + w - r, y)
    p.quadTo(x + w, y, x + w, y + r)
    p.lineTo(x + w, y + h)
    p.closeSubpath()
    return p


class ColumnChart(_ChartBase):
    """Grouped columns: one or more series over categories (e.g. days)."""

    def __init__(self, series_names: list[str], colors: list[str], fmt=lambda v: f"{v:g}", height: int = 220):
        super().__init__(height)
        self.series_names = series_names
        self.colors = colors
        self.fmt = fmt
        self.categories: list[str] = []
        self.values: list[list[float]] = []   # per series

    def set_data(self, categories: list[str], values: list[list[float]]):
        self.categories, self.values = categories, values
        self._animate((tuple(categories), tuple(tuple(round(v, 3) for v in s) for s in values)))
        self.update()

    def paintEvent(self, _):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        self._hits = []
        if not self.categories or not any(any(v > 0 for v in s) for s in self.values):
            self._empty(g)
            return
        legend_h = 24 if len(self.series_names) > 1 else 0
        vmax = max(max(s) for s in self.values)
        ymax, step = self.nice_max(vmax)
        g.setFont(font(11))
        # the axis is as wide as its longest label (a fixed 48 px clipped "0.0005h")
        ticks = [self.fmt(k * step) for k in range(int(round(ymax / step)) + 1)]
        left = max(40, max(g.fontMetrics().horizontalAdvance(t) for t in ticks) + 14)
        right, top, bottom = 8, 10 + legend_h, 26
        w, h = self.width() - left - right, self.height() - top - bottom
        # legend (>=2 series)
        if legend_h:
            x = left
            for name, col in zip(self.series_names, self.colors):
                g.setPen(Qt.NoPen)
                g.setBrush(QColor(tok(col)))
                g.drawRoundedRect(QRectF(x, 6, 10, 10), 3, 3)
                g.setPen(QColor(C["text"]))
                g.drawText(QPointF(x + 15, 15), name)
                x += 30 + g.fontMetrics().horizontalAdvance(name)
        # grid + y ticks
        k = 0.0
        while k <= ymax + 1e-9:
            y = top + h - h * (k / ymax)
            g.setPen(QPen(QColor(C["grid"]), 1))
            g.drawLine(QPointF(left, y), QPointF(left + w, y))
            g.setPen(QColor(C["muted"]))
            g.drawText(QRectF(0, y - 8, left - 8, 16), Qt.AlignRight | Qt.AlignVCenter, self.fmt(k))
            k += step
        n = len(self.categories)
        ns = len(self.values)
        band = w / n
        bar_w = min(24.0, (band * 0.7 - 2 * (ns - 1)) / ns)
        group_w = bar_w * ns + 2 * (ns - 1)
        for i, cat in enumerate(self.categories):
            gx = left + band * i + (band - group_w) / 2
            for si, series in enumerate(self.values):
                v = series[i]
                bh = h * (v / ymax) * self.grow if ymax else 0
                x = gx + si * (bar_w + 2)
                if bh > 0.5:
                    g.setPen(Qt.NoPen)
                    col = QColor(tok(self.colors[si]))
                    grad = QLinearGradient(QPointF(x, top + h - bh), QPointF(x, top + h))
                    grad.setColorAt(0, col)
                    grad.setColorAt(1, QColor(mix(tok(self.colors[si]), "#FFFFFF", 0.38)))
                    g.setBrush(grad)
                    g.drawPath(_bar_path(x, top + h - bh, bar_w, bh))
                hit = QRectF(x - 1, top, bar_w + 2, h)
                self._hits.append((hit, f"{cat} · {self.series_names[si]}: {self.fmt(v)}"))
            g.setPen(QColor(C["muted"]))
            g.drawText(QRectF(left + band * i, top + h + 6, band, 16), Qt.AlignCenter, cat)


class HBarChart(_ChartBase):
    """Horizontal bars for ranked categories (most common distractions)."""

    def __init__(self, color: str = CH_DISTRACT, fmt=lambda v: f"{v:g}", height: int = 220):
        super().__init__(height)
        self.color = color
        self.fmt = fmt
        self.items: list[tuple[str, float]] = []

    def set_data(self, items: list[tuple[str, float]]):
        self.items = items
        self._animate(tuple(items))
        self.update()

    def paintEvent(self, _):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        self._hits = []
        if not self.items:
            self._empty(g)
            return
        g.setFont(font(12))
        fm = g.fontMetrics()
        label_w = min(170, max(fm.horizontalAdvance(n) for n, _ in self.items) + 12)
        val_w = 64
        left = label_w
        w = self.width() - left - val_w
        row = min(34.0, self.height() / max(1, len(self.items)))
        vmax = max(v for _, v in self.items) or 1
        for i, (name, v) in enumerate(self.items):
            y = i * row + (row - 16) / 2
            g.setPen(QColor(C["text"]))
            g.drawText(QRectF(0, i * row, label_w - 10, row), Qt.AlignRight | Qt.AlignVCenter,
                       fm.elidedText(name, Qt.ElideMiddle, label_w - 10))
            bw = max(2.0, w * v / vmax * self.grow)
            g.setPen(Qt.NoPen)
            g.setBrush(QColor(C["grid"]))
            g.drawRoundedRect(QRectF(left, y, w, 16), 8, 8)
            grad = QLinearGradient(QPointF(left, y), QPointF(left + bw, y))
            grad.setColorAt(0, QColor(mix(tok(self.color), "#FFFFFF", 0.3)))
            grad.setColorAt(1, QColor(tok(self.color)))
            g.setBrush(grad)
            p = QPainterPath()
            p.addRoundedRect(QRectF(left, y, bw, 16), 8, 8)
            g.drawPath(p)
            g.setPen(QColor(C["muted"]))
            g.drawText(QRectF(left + w + 8, i * row, val_w - 8, row), Qt.AlignLeft | Qt.AlignVCenter, self.fmt(v))
            self._hits.append((QRectF(0, i * row, self.width(), row), f"{name}: {self.fmt(v)}"))


class LineChart(_ChartBase):
    """Single-series line with markers (e.g. weekly productivity %)."""

    def __init__(self, color: str = CH_FOCUS, fmt=lambda v: f"{v:.0f}%", ymax: float | None = 100, height: int = 200):
        super().__init__(height)
        self.color = color
        self.fmt = fmt
        self.fixed_max = ymax
        self.categories: list[str] = []
        self.values: list[float | None] = []

    def set_data(self, categories, values):
        self.categories, self.values = categories, values
        self._animate((tuple(categories), tuple(values)))
        self.update()

    def paintEvent(self, _):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        self._hits = []
        pts_v = [v for v in self.values if v is not None]
        if not pts_v:
            self._empty(g)
            return
        left, right, top, bottom = 48, 16, 12, 26
        w, h = self.width() - left - right, self.height() - top - bottom
        if self.fixed_max:
            ymax, step = self.fixed_max, self.fixed_max / 4
        else:
            ymax, step = self.nice_max(max(pts_v))
        g.setFont(font(11))
        for k in range(int(round(ymax / step)) + 1):
            y = top + h - h * (k * step / ymax)
            g.setPen(QPen(QColor(C["grid"]), 1))
            g.drawLine(QPointF(left, y), QPointF(left + w, y))
            g.setPen(QColor(C["muted"]))
            g.drawText(QRectF(0, y - 8, left - 8, 16), Qt.AlignRight | Qt.AlignVCenter, self.fmt(k * step))
        n = len(self.categories)
        xs = [left + (w * (i + 0.5) / n) for i in range(n)]
        pts = [(xs[i], top + h - h * (v / ymax)) for i, v in enumerate(self.values) if v is not None]
        idx = [i for i, v in enumerate(self.values) if v is not None]
        if self.grow < 1.0:                      # the line draws itself from left to right
            g.setClipRect(QRectF(0, 0, left + w * self.grow + 8, self.height()))
        if len(pts) > 1:
            area = QPainterPath(QPointF(pts[0][0], top + h))
            area.lineTo(*pts[0])
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                mx = (x0 + x1) / 2
                area.cubicTo(QPointF(mx, y0), QPointF(mx, y1), QPointF(x1, y1))
            area.lineTo(pts[-1][0], top + h)
            area.closeSubpath()
            fill = QLinearGradient(QPointF(0, top), QPointF(0, top + h))
            c = QColor(tok(self.color))
            c.setAlpha(70)
            fill.setColorAt(0, c)
            c2 = QColor(tok(self.color))
            c2.setAlpha(0)
            fill.setColorAt(1, c2)
            g.setPen(Qt.NoPen)
            g.setBrush(fill)
            g.drawPath(area)
            line = QPainterPath(QPointF(*pts[0]))
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):        # smooth curve through the points
                mx = (x0 + x1) / 2
                line.cubicTo(QPointF(mx, y0), QPointF(mx, y1), QPointF(x1, y1))
            pen = QPen(QColor(tok(self.color)), 3)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            g.setPen(pen)
            g.setBrush(Qt.NoBrush)
            g.drawPath(line)
        for (x, y), i in zip(pts, idx):
            g.setPen(QPen(QColor(C["surface"]), 2))
            g.setBrush(QColor(tok(self.color)))
            g.drawEllipse(QPointF(x, y), 4.5, 4.5)
            self._hits.append((QRectF(x - 14, top, 28, h), f"{self.categories[i]}: {self.fmt(self.values[i])}"))
        # label the last point only
        lx, ly = pts[-1]
        g.setPen(QColor(C["text"]))
        g.setFont(font(11, 650))
        g.drawText(QRectF(lx - 40, ly - 22, 80, 16), Qt.AlignCenter, self.fmt(self.values[idx[-1]]))
        g.setFont(font(11))
        g.setPen(QColor(C["muted"]))
        for i, cat in enumerate(self.categories):
            g.drawText(QRectF(xs[i] - 30, top + h + 6, 60, 16), Qt.AlignCenter, cat)


class CharacterPreview(QWidget):
    """Live animated character (dashboard, wizard, companion page).

    2.2: it is alive under the mouse - the eyes and head follow the cursor while it hovers, and a
    click pokes the part under it (head, body or tail): the same flinch as on the desktop, a little
    reaction and a few floating hearts / sparkles."""

    clicked = Signal()
    poked = Signal(str)

    def __init__(self, spec: CharacterSpec, size: int = 120, state: S = S.IDLE, animate: bool = True):
        super().__init__()
        self.setFixedSize(size, size)
        self.spec = spec
        self.animator = Animator()
        self.renderer = None                 # built in the background (preview_loader.py); portrait meanwhile
        self._ticket = 0
        self._portrait = None
        self._fade_t0 = 0.0                  # portrait -> live model cross-fade start
        self._wanted = False                 # the model is requested when the preview is first shown
        self._make_renderer()
        self.state = state
        self._t0 = time.perf_counter()
        self._last = self._t0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._frame)
        self._cost = 0.012                   # seconds per frame (measured)
        self.interactive = True
        self.setMouseTracking(True)
        self._hover = None                   # cursor target (-1..1, -1..1) while hovering
        self._look = [0.0, 0.0, 0.0, 0.0]    # eyes x / y, head x / y (smoothed)
        self._front = 0.0
        self._fx: list[dict] = []            # floating hearts / sparkles after a poke
        self._react_until = 0.0
        self._react_from = state
        if animate:
            self.timer.start(self.BASE_MS)

    BASE_MS = 66                             # ~15 FPS is plenty for a portrait
    DUTY = 0.2                               # at most this share of the UI thread goes to ALL previews together
    _live: "weakref.WeakSet" = None          # previews that are animating right now

    def _frame(self):
        # a 3D frame is costly on the UI thread: all previews together keep their share small (the
        # Companion page shows eight), previews scrolled out of view rest, and everything nearly
        # rests while the window is in the background - so the dashboard stays smooth
        live = CharacterPreview._live
        if live is None:
            import weakref
            live = CharacterPreview._live = weakref.WeakSet()
        win = self.window()
        active = win is not None and win.isActiveWindow()
        seen = not self.visibleRegion().isEmpty()
        if seen:
            live.add(self)
        else:
            live.discard(self)
        n = max(1, len(live))
        want = max(self.BASE_MS, int(1000 * self._cost * n / self.DUTY)) if active and seen else 400
        if self.timer.interval() != want:
            self.timer.setInterval(want)
        if not seen:
            return
        self._render()
        self.update()

    def _render(self) -> None:
        """Draw the next frame into a cached image: a repaint for any other reason (a hover, an
        animation passing over it) only copies the image instead of rendering the model again."""
        from PySide6.QtGui import QImage
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        t0 = time.perf_counter()
        dpr = self.devicePixelRatioF()
        w, h = max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr))
        img = getattr(self, "_img", None)
        if img is None or img.width() != w or img.height() != h:
            img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
            img.setDevicePixelRatio(dpr)
        img.fill(Qt.transparent)
        g = QPainter(img)
        if self._react_until and now >= self._react_until:
            self._react_until = 0.0
            self.state = self._react_from
        look = self._update_look(dt)
        if look is not None and self.state not in (S.WALKING, S.RUNNING, S.CHASING):
            pose = self.animator.update(self.state, now, dt, None, None, look)
        else:
            pose = self.animator.pose(self.state, now, dt, 40, 75)
        rect = QRectF(0, 0, self.width(), self.height())
        fade = min(1.0, (now - self._fade_t0) / 0.35) if self._fade_t0 else 1.0
        if self.renderer is None or fade < 1.0:
            self._paint_portrait(g, rect, 1.0 if self.renderer is None else 1.0 - fade)
        if self.renderer is not None:
            g.setOpacity(fade)
            self.renderer.paint(g, rect, pose, 1, self.state.value)
            g.setOpacity(1.0)
            if fade < 1.0:
                QTimer.singleShot(16, self._fade_step)
        g.end()
        self._img = img
        if self.renderer is not None:
            self._cost += (time.perf_counter() - t0 - self._cost) * 0.2

    def _fade_step(self) -> None:
        try:
            self._render()
            self.update()
        except RuntimeError:                 # deleted meanwhile
            pass

    def _paint_portrait(self, g: QPainter, rect: QRectF, opacity: float) -> None:
        """The character's portrait (its sheet art) while the live model loads."""
        if opacity <= 0.0:
            return
        if self._portrait is None:
            self._portrait = False
            try:
                from PySide6.QtGui import QPixmap

                from ..companion.illustrated import pack_dir
                folder = pack_dir(self.spec.id)
                if folder is not None and (folder / "portrait.webp").exists():
                    pm = QPixmap(str(folder / "portrait.webp"))
                    if not pm.isNull():
                        self._portrait = pm
            except Exception:  # noqa: BLE001 - a missing portrait just leaves the space empty
                self._portrait = False
        pm = self._portrait
        if not pm:
            return
        w, h = rect.width() * 0.86, rect.height() * 0.86
        k = min(w / pm.width(), h / pm.height())
        tw, th = pm.width() * k, pm.height() * k
        g.save()
        g.setOpacity(opacity)
        g.setRenderHint(QPainter.SmoothPixmapTransform, True)
        g.drawPixmap(QRectF(rect.center().x() - tw / 2, rect.bottom() - th - rect.height() * 0.05, tw, th), pm,
                     QRectF(pm.rect()))
        g.restore()

    def _make_renderer(self):
        from ..companion.profile import get_profile
        from ..companion.renderer import DEFAULT_MODE
        self.animator.set_profile(get_profile(self.spec.id if self.spec.id in ("cat", "dog", "fox", "rabbit", "robot",
                                                                              "buddy") else self.spec.personality))
        mode, quality = DEFAULT_MODE["mode"], DEFAULT_MODE["quality"]
        from .preview_loader import PreviewLoader
        loader = PreviewLoader.get()
        if self._ticket:
            loader.cancel(self._ticket)
            self._ticket = 0
        heavy = mode in ("production", "illustrated") and self.spec.skeleton != "sprite"
        if heavy and not self._wanted:
            return                           # not on screen yet: load when first shown (showEvent)
        if heavy:
            # heavy: build in the background, show the portrait meanwhile
            spec = self.spec
            self._ticket = loader.request(spec, mode, quality, lambda r, s=spec: self._renderer_ready(s, r))
            return
        self._set_renderer(make_renderer(self.spec, mode, quality), fade=False)

    def _renderer_ready(self, spec: CharacterSpec, renderer) -> None:
        self._ticket = 0
        if spec is not self.spec:
            return                           # the character changed meanwhile
        if renderer is None:                 # couldn't be built off the UI thread: build it here
            from ..companion.renderer import DEFAULT_MODE
            renderer = make_renderer(self.spec, DEFAULT_MODE["mode"], DEFAULT_MODE["quality"])
        self._set_renderer(renderer, fade=self._portrait not in (None, False) and self.isVisible())

    def _set_renderer(self, renderer, fade: bool) -> None:
        self.renderer = renderer
        if hasattr(renderer, "rig"):
            self.animator.set_rig(renderer.rig)
        self._fade_t0 = time.perf_counter() if fade else 0.0
        self._img = None
        self.update()

    def set_character(self, spec: CharacterSpec):
        if spec.id == getattr(self.spec, "id", None) and (self.renderer is not None or self._ticket):
            return                           # (the dashboard refreshes often: no reload of the same model)
        self.spec = spec
        self.renderer = None
        self._portrait = None
        self._make_renderer()
        self._img = None
        self.update()

    def set_state(self, state: S):
        if self._react_until:                # a poke reaction is playing: switch when it ends
            self._react_from = state
            return
        if state == self.state:
            return
        self.state = state
        self.update()

    def showEvent(self, e):
        if not self._wanted:
            self._wanted = True
            if self.renderer is None and not self._ticket:
                self._make_renderer()
        if not self.timer.isActive():
            self.timer.start(self.BASE_MS)
        super().showEvent(e)

    def hideEvent(self, e):
        self.timer.stop()
        super().hideEvent(e)

    # ------------------------------------------------------------ 2.2: alive under the mouse
    def _update_look(self, dt: float):
        """Smoothed gaze toward the cursor while hovering, drifting back to the viewer after."""
        if not self.interactive:
            return None
        tx, ty = self._hover if self._hover is not None else (0.0, 0.0)
        k_eye = 1 - math.exp(-dt / 0.08)
        k_head = 1 - math.exp(-dt / 0.35)
        L = self._look
        L[0] += (tx - L[0]) * k_eye
        L[1] += (ty - L[1]) * k_eye
        L[2] += (tx * 0.6 - L[2]) * k_head
        L[3] += (ty * 0.6 - L[3]) * k_head
        self._front += ((1.0 if self._hover is not None else 0.0) - self._front) * k_head
        if self._hover is None and max(abs(v) for v in L) < 0.01 and self._front < 0.01:
            return None
        from ..companion.attention import AttentionTarget, LookOutput
        return LookOutput(AttentionTarget.USER, None, L[0], L[1], L[2], L[3], self._front, False, 0.0, 0.0,
                          0.25 * self._front, 0.0)

    def mouseMoveEvent(self, e):
        if self.interactive:
            p = e.position()
            w, h = max(1, self.width()), max(1, self.height())
            self._hover = (max(-0.8, min(0.8, (p.x() - w * 0.5) / (w * 0.5))),
                           max(-0.6, min(0.6, (p.y() - h * 0.32) / (h * 0.5))))
        super().mouseMoveEvent(e)

    def leaveEvent(self, e):
        self._hover = None
        super().leaveEvent(e)

    def poke(self, x: float, y: float) -> str:
        """Poke the companion at (x, y) (widget pixels): returns the part that was hit."""
        part = "body"
        img = getattr(self, "_img", None)
        if img is not None:
            try:
                import numpy as np
                from PySide6.QtGui import QImage
                im = img.convertToFormat(QImage.Format_ARGB32)
                bpl = im.bytesPerLine()
                a = np.frombuffer(im.constBits(), np.uint8, count=bpl * im.height()).reshape(im.height(), bpl)
                a = a[:, :im.width() * 4].reshape(im.height(), im.width(), 4)[:, :, 3]
                from ..companion.character import classify_click
                dpr = img.devicePixelRatio() or 1.0
                hit = classify_click(a, x * dpr, y * dpr, getattr(self.animator, "species", self.spec.id), 1)
                part = hit or "body"
            except Exception:  # noqa: BLE001 - a hit test is a nicety
                part = "body"
        self.animator.on_event(f"poke_{part}", 1.0)
        now = time.perf_counter()
        if not self._react_until:
            self._react_from = self.state
        if self.state in (S.IDLE, S.SITTING, S.PETTED, S.WAVING):
            self.state = S.PETTED if part == "head" else S.WAVING if part == "body" else self.state
            self._react_until = now + (1.4 if part != "tail" else 0.0)
        import random
        glyph = {"head": "♥", "body": "✦", "tail": "!"}[part]
        for i in range(5 if part != "tail" else 1):
            self._fx.append({"t0": now + i * 0.06, "x": x + random.uniform(-14, 14), "y": y - 6,
                             "vx": random.uniform(-18, 18), "g": glyph, "life": random.uniform(0.9, 1.3)})
        self.poked.emit(part)
        self.update()
        return part

    def mousePressEvent(self, e):
        if self.interactive and e.button() == Qt.LeftButton:
            self.poke(e.position().x(), e.position().y())
        self.clicked.emit()

    def _paint_fx(self, g: QPainter) -> None:
        now = time.perf_counter()
        self._fx = [f for f in self._fx if now - f["t0"] < f["life"]]
        if not self._fx:
            return
        g.setRenderHint(QPainter.Antialiasing)
        for f in self._fx:
            u = (now - f["t0"]) / f["life"]
            if u < 0:
                continue
            a = int(255 * (1.0 - u) ** 1.5)
            col = QColor("#FF4F8B" if f["g"] == "♥" else C["primary"] if f["g"] == "✦" else "#FFB020")
            col.setAlpha(a)
            g.setPen(col)
            g.setFont(font(int(16 + 8 * (1 - (1 - u) ** 3)), 800))
            g.drawText(QPointF(f["x"] + f["vx"] * u, f["y"] - 46 * u), f["g"])
        QTimer.singleShot(30, self.update)

    def paintEvent(self, _):
        img = getattr(self, "_img", None)
        if img is None or abs(img.width() - self.width() * self.devicePixelRatioF()) > 1:
            self._render()
            img = self._img
        g = QPainter(self)
        g.drawImage(0, 0, img)
        self._paint_fx(g)
        g.end()


# ====================================================================== 2.2: quick actions
class QuickActions(QWidget):
    """A row of one-click pill actions (``add(icon, text, fn, tip)``); hover lifts them."""

    def __init__(self):
        super().__init__()
        self.setObjectName("Transparent")
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(14, 2, 14, 6)
        self.lay.setSpacing(8)
        self.buttons: dict[str, QPushButton] = {}
        self._end = None

    def add(self, key: str, icon: str, text: str, fn, tip: str = "") -> QPushButton:
        b = QPushButton(f"{icon}  {text}")
        b.setObjectName("Quick")
        b.setCursor(Qt.PointingHandCursor)
        b.setMinimumHeight(36)
        if tip:
            b.setToolTip(tip)
        b.clicked.connect(fn)
        if self._end is None:
            self.lay.addWidget(b)
        else:
            self.lay.insertWidget(self.lay.count() - 1, b)
        self.buttons[key] = b
        return b

    def finish(self) -> None:
        self.lay.addStretch(1)
        self._end = True

    def set_text(self, key: str, icon: str, text: str) -> None:
        b = self.buttons.get(key)
        if b is not None and b.text() != f"{icon}  {text}":
            b.setText(f"{icon}  {text}")


# ====================================================================== 2.2: drag-to-reorder sections
class _Section(QWidget):
    """One reorderable block: its content plus a grip (shown on hover) to drag it by."""

    def __init__(self, key: str, content: QWidget, column: "ReorderColumn"):
        super().__init__()
        self.key = key
        self.column = column
        self.setObjectName("Transparent")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(content)
        self.grip = QPushButton("⠿", self)
        self.grip.setObjectName("Grip")
        self.grip.setFixedSize(40, 16)
        self.grip.setCursor(Qt.OpenHandCursor)
        self.grip.setToolTip("Drag to move this section")
        self.grip.hide()
        self.grip.installEventFilter(self)
        self._press = None

    def resizeEvent(self, e):
        self.grip.move((self.width() - self.grip.width()) // 2, 0)     # a small tab on the top edge
        super().resizeEvent(e)

    def enterEvent(self, e):
        self.grip.show()
        self.grip.raise_()
        super().enterEvent(e)

    def leaveEvent(self, e):
        if self.column._drag is not self:
            self.grip.hide()
        super().leaveEvent(e)

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if obj is self.grip:
            t = ev.type()
            if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                self._press = ev.globalPosition()
                return True
            if t == QEvent.MouseMove and self._press is not None:
                if self.column._drag is None and (ev.globalPosition() - self._press).manhattanLength() > 6:
                    self.column.begin_drag(self)
                if self.column._drag is self:
                    self.column.drag_to(self.column.mapFromGlobal(ev.globalPosition().toPoint()))
                return True
            if t == QEvent.MouseButtonRelease:
                self._press = None
                if self.column._drag is self:
                    self.column.end_drag()
                return True
        return super().eventFilter(obj, ev)


class ReorderColumn(QWidget):
    """A vertical stack of sections the user can drag into a new order (by the grip that appears
    on hover). ``order_changed(list of keys)`` fires after a drop; ``set_order`` restores one."""

    order_changed = Signal(list)

    def __init__(self):
        super().__init__()
        self.setObjectName("Transparent")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(2)
        self.sections: list[_Section] = []
        self._drag: _Section | None = None
        self._ghost: QLabel | None = None
        self._target = -1
        self._grab_dy = 0

    def add_section(self, key: str, content: QWidget) -> _Section:
        s = _Section(key, content, self)
        self.sections.append(s)
        self.lay.addWidget(s)
        return s

    def order(self) -> list[str]:
        return [s.key for s in self.sections]

    def set_order(self, keys: list) -> None:
        known = {s.key: s for s in self.sections}
        new = [known[k] for k in keys if k in known]
        new += [s for s in self.sections if s not in new]
        if new == self.sections:
            return
        for s in self.sections:
            self.lay.removeWidget(s)
        self.sections = new
        for s in new:
            self.lay.addWidget(s)

    # ------------------------------------------------------------ dragging
    def begin_drag(self, s: _Section) -> None:
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        self._drag = s
        s.grip.setCursor(Qt.ClosedHandCursor)
        pm = s.grab()
        g = QLabel(self)
        g.setPixmap(pm)
        g.setAttribute(Qt.WA_TransparentForMouseEvents)
        fx = QGraphicsOpacityEffect(g)
        fx.setOpacity(0.85)
        g.setGraphicsEffect(fx)
        g.resize(pm.size() / max(1.0, pm.devicePixelRatio()))
        g.move(s.pos())
        g.show()
        g.raise_()
        self._ghost = g
        self._grab_dy = self.mapFromGlobal(QCursor.pos()).y() - s.y()
        dim = QGraphicsOpacityEffect(s)
        dim.setOpacity(0.25)
        s.setGraphicsEffect(dim)
        self._target = self.sections.index(s)

    def drag_to(self, p) -> None:
        if self._ghost is None:
            return
        self._ghost.move(self._drag.x(), max(-20, min(self.height() - 20, p.y() - self._grab_dy)))
        # the gap closest to the ghost's middle
        mid = self._ghost.y() + self._ghost.height() / 2
        best, bi = None, 0
        for i in range(len(self.sections) + 1):
            y = self.sections[i].y() if i < len(self.sections) else self.height()
            d = abs(mid - y)
            if best is None or d < best:
                best, bi = d, i
        self._target = bi
        self.update()

    def end_drag(self) -> None:
        s = self._drag
        self._drag = None
        if self._ghost is not None:
            self._ghost.deleteLater()
            self._ghost = None
        if s is None:
            return
        s.setGraphicsEffect(None)
        s.grip.setCursor(Qt.OpenHandCursor)
        s.grip.hide()
        old = self.sections.index(s)
        new = self._target if self._target <= old else self._target - 1
        self._target = -1
        if 0 <= new < len(self.sections) and new != old:
            order = self.order()
            order.insert(new, order.pop(old))
            self.set_order(order)
            self.order_changed.emit(order)
            try:
                from .motion import fade_in
                fade_in(s)
            except Exception:  # noqa: BLE001
                pass
        self.update()

    def paintEvent(self, _e):
        if self._drag is None or self._target < 0:
            return
        y = self.sections[self._target].y() if self._target < len(self.sections) else self.height() - 2
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        grad = QLinearGradient(QPointF(20, y), QPointF(self.width() - 20, y))
        grad.setColorAt(0.0, QColor(C["primary"]))
        grad.setColorAt(1.0, QColor(C.get("primary_grad", C["primary"])))
        path = QPainterPath()
        path.addRoundedRect(QRectF(20, y - 2, self.width() - 40, 4), 2, 2)
        g.fillPath(path, QBrush(grad))
        g.end()
