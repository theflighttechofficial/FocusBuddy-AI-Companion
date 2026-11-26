"""Motion & surfaces for the soft premium UI.

* ``AuroraBackground`` - the window canvas: a pastel gradient with large, softly drifting colour
  glows (parallax with the mouse). Animates only while the window is active and the mouse has moved
  recently; static with reduced motion.
* ``GlassCard`` - frosted card with a cached soft shadow (no QGraphicsEffect, so the animated canvas
  stays cheap) and a hover "lift" (shadow + edge glow animate).
* ``AnimatedStack`` - page transitions: the new page rises and fades in while the old one fades out.
* ``NavRail`` - navigation buttons with a springy gradient indicator that glides to the active page.
* ``CountUp`` - animated numbers; ``Ripple`` - press ripples on every button; ``SmoothScroller`` -
  eased wheel scrolling; ``Toast`` - slide-in notifications; ``shake`` - wrong-PIN shake.

All of it reads the motion level from theme.MOTION: "full", "reduced" (fades only) or "off".
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

from PySide6.QtCore import (QEasingCurve, QEvent, QObject, QPoint, QPointF, QPropertyAnimation, QRectF,
                            QSize, Qt, QTimer, QVariantAnimation, Signal)
from PySide6.QtGui import (QBrush, QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                           QRadialGradient, QRegion)
from PySide6.QtWidgets import (QAbstractButton, QAbstractScrollArea, QApplication, QButtonGroup, QFrame, QLabel,
                               QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from .theme import C, STATE, mix as mix_hex, motion_level


def _qc(key_or_hex: str, alpha: int | None = None) -> QColor:
    c = QColor(C.get(key_or_hex, key_or_hex))
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def windows_animations_enabled() -> bool:
    """Windows "Animation effects" (Settings > Accessibility > Visual effects). True elsewhere."""
    import sys
    if not sys.platform.startswith("win"):
        return True
    try:
        import ctypes
        v = ctypes.c_int(1)
        if ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(v), 0):   # SPI_GETCLIENTAREAANIMATION
            return bool(v.value)
    except Exception:  # noqa: BLE001
        pass
    return True


def animated() -> bool:
    return motion_level() != "off"


def full_motion() -> bool:
    return motion_level() == "full"


# ====================================================================== canvas
def paint_canvas(g: QPainter, r: QRectF, blobs=None) -> None:
    """The pastel canvas: diagonal gradient + soft colour glows (static when ``blobs`` is None)."""
    grad = QLinearGradient(r.topLeft(), r.bottomRight())
    grad.setColorAt(0.0, _qc("grad_a"))
    grad.setColorAt(0.55, _qc("grad_b"))
    grad.setColorAt(1.0, _qc("grad_c"))
    g.fillRect(r, grad)
    if blobs is None:
        w, h = r.width(), r.height()
        s = max(w, h)
        blobs = [("blob1", QPointF(w * 0.15, h * 0.18), s * 0.42), ("blob2", QPointF(w * 0.8, h * 0.3), s * 0.46),
                 ("blob3", QPointF(w * 0.55, h * 0.95), s * 0.4), ("blob4", QPointF(w * 0.98, h * 0.88), s * 0.3)]
    g.setRenderHint(QPainter.Antialiasing, True)
    g.setPen(Qt.NoPen)
    a = 150 if not STATE.get("dark") else 120
    for key, c, rad in blobs:
        rg = QRadialGradient(c, rad)
        rg.setColorAt(0.0, _qc(key, a))
        rg.setColorAt(0.45, _qc(key, int(a * 0.45)))
        rg.setColorAt(1.0, _qc(key, 0))
        g.setBrush(QBrush(rg))
        g.drawEllipse(c, rad, rad)


class AuroraBackground(QWidget):
    """The canvas behind a window: a soft gradient with colour glows. Children are laid out on top.

    It is painted once per size / theme (from a small cached image scaled up) and does not animate:
    repainting the whole window 24 times a second for drifting glows made every card, chart and the
    companion preview repaint too, which is what made the dashboard feel stuttery."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Root")
        self.setAttribute(Qt.WA_StyledBackground, False)
        self._cache: Optional[QImage] = None
        self._cache_key = None

    def poke(self) -> None:                 # kept for callers; the canvas is static
        pass

    def _blobs(self):
        w, h = self.width(), self.height()
        s = max(w, h)
        return [
            ("blob1", QPointF(w * 0.12, h * 0.16), s * 0.42),
            ("blob2", QPointF(w * 0.80, h * 0.26), s * 0.46),
            ("blob3", QPointF(w * 0.52, h * 0.96), s * 0.40),
            ("blob4", QPointF(w * 0.98, h * 0.86), s * 0.30),
        ]

    SCALE = 0.25          # the canvas is all soft gradients: render at 1/4 size, scale up smoothly

    def paintEvent(self, _ev):
        w, h = max(1, self.width()), max(1, self.height())
        k = self.SCALE
        key = (w, h, C.get("grad_a"), C.get("blob1"), C.get("blob4"), STATE.get("dark"))
        if self._cache is None or self._cache_key != key:
            img = QImage(max(1, int(w * k)), max(1, int(h * k)), QImage.Format_ARGB32_Premultiplied)
            p = QPainter(img)
            p.scale(k, k)
            paint_canvas(p, QRectF(0, 0, w, h), self._blobs())
            p.end()
            self._cache, self._cache_key = img, key
        g = QPainter(self)
        g.setRenderHint(QPainter.SmoothPixmapTransform, True)
        g.drawImage(QRectF(0, 0, w, h), self._cache)
        g.end()


# ====================================================================== glass card
_SHADOW_CACHE: dict = {}


def _shadow_tile(radius: int, blur: int, color: QColor) -> QImage:
    """A blurred rounded-rect shadow (cached) used as a 9-slice."""
    key = (radius, blur, color.rgba())
    im = _SHADOW_CACHE.get(key)
    if im is not None:
        return im
    size = radius * 2 + blur * 4 + 8
    base = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    base.fill(Qt.transparent)
    p = QPainter(base)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    m = blur * 2
    p.drawRoundedRect(QRectF(m, m, size - 2 * m, size - 2 * m), radius, radius)
    p.end()
    im = _blur(base, blur)
    _SHADOW_CACHE[key] = im
    return im


def _blur(img: QImage, radius: int) -> QImage:
    """Fast separable box blur x3 (≈ gaussian) on the alpha channel (NumPy when available)."""
    try:
        import numpy as np
        w, h = img.width(), img.height()
        ptr = img.constBits()
        arr = np.frombuffer(ptr, dtype=np.uint8, count=img.sizeInBytes()).reshape(h, img.bytesPerLine())[:, :w * 4]
        arr = arr.reshape(h, w, 4).astype(np.float32)
        k = max(1, radius // 2)
        for _ in range(3):
            for axis in (0, 1):
                c = np.cumsum(np.pad(arr, [(k + 1, k) if i == axis else (0, 0) for i in range(3)], mode="edge"),
                              axis=axis)
                if axis == 0:
                    arr = (c[2 * k + 1:] - c[:-2 * k - 1]) / (2 * k + 1)
                else:
                    arr = (c[:, 2 * k + 1:] - c[:, :-2 * k - 1]) / (2 * k + 1)
        out = np.ascontiguousarray(arr.clip(0, 255).astype(np.uint8))
        res = QImage(out.data, w, h, w * 4, QImage.Format_ARGB32_Premultiplied).copy()
        return res
    except Exception:  # noqa: BLE001 - no numpy: an unblurred soft edge is still fine
        return img


class GlassCard(QFrame):
    """Frosted rounded card with a soft cached shadow and a hover lift."""

    MARGIN = 12            # room around the card body for the shadow
    RADIUS = 22

    def __init__(self, parent=None, hover: bool = True, radius: int | None = None, strong: bool = False,
                 tint: bool = False, hero: bool = False, wash: str | None = None):
        super().__init__(parent)
        self.tint = tint
        self.hero = hero                    # a vivid gradient banner (white text, see the QSS #HeroCard)
        self.wash = wash                    # a palette token: a faint colour wash from the top-left
        self.setObjectName("HeroCard" if hero else "GlassCard")
        self.setAttribute(Qt.WA_StyledBackground, False)
        self._lift = 0.0
        self._hover = hover
        self._strong = strong
        self.radius = radius or self.RADIUS
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(260)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._set_lift)
        self._flash = 0.0
        # 2.2: a soft accent spotlight that follows the cursor over the card (full motion only)
        self._spot = None                   # cursor position in card coordinates while hovering
        self._spot_timer = None

    def body_rect(self) -> QRectF:
        m = self.MARGIN
        return QRectF(self.rect()).adjusted(m, m - 3, -m, -m - 2)

    def _set_lift(self, v):
        self._lift = float(v)
        self.update()

    def enterEvent(self, ev):
        if self._hover and animated():
            self._anim.stop()
            self._anim.setStartValue(self._lift)
            self._anim.setEndValue(1.0)
            self._anim.start()
        if self._hover and full_motion() and not self.hero:
            if self._spot_timer is None:
                from PySide6.QtCore import QTimer
                self._spot_timer = QTimer(self)
                self._spot_timer.setInterval(33)
                self._spot_timer.timeout.connect(self._track_spot)
            self._spot_timer.start()
            self._track_spot()
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        if self._hover and animated():
            self._anim.stop()
            self._anim.setStartValue(self._lift)
            self._anim.setEndValue(0.0)
            self._anim.start()
        if self._spot_timer is not None:
            self._spot_timer.stop()
        if self._spot is not None:
            self._spot = None
            self.update()
        super().leaveEvent(ev)

    def _track_spot(self) -> None:
        from PySide6.QtGui import QCursor
        p = self.mapFromGlobal(QCursor.pos())
        if not self.rect().contains(p):
            self._spot_timer.stop()
            self._spot = None
            self.update()
            return
        q = QPointF(p)
        if self._spot is None or (q - self._spot).manhattanLength() > 1.5:
            self._spot = q
            self.update()

    def flash(self) -> None:
        """Highlight pulse (e.g. a search result scrolled into view)."""
        a = QVariantAnimation(self)
        a.setDuration(1400)
        a.setStartValue(1.0)
        a.setEndValue(0.0)
        a.setEasingCurve(QEasingCurve.InOutSine)
        a.valueChanged.connect(lambda v: (setattr(self, "_flash", float(v)), self.update()))
        a.start(QVariantAnimation.DeleteWhenStopped)

    def paintEvent(self, _ev):
        key = (self.width(), self.height(), round(self._lift, 2), round(self._flash, 2), STATE.get("dark"),
               C.get("surface"), C.get("primary"), C.get("primary_grad"), self.tint, self._strong, self.radius,
               self.hero, self.wash, C.get(self.wash or "primary"))
        cache = getattr(self, "_pm", None)
        if cache is None or self._pm_key != key:
            dpr = self.devicePixelRatioF()
            pm = QPixmap(int(self.width() * dpr), int(self.height() * dpr))
            pm.setDevicePixelRatio(dpr)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            self._paint_card(p)
            p.end()
            self._pm, self._pm_key = pm, key
        g = QPainter(self)
        g.drawPixmap(0, 0, self._pm)
        if self._spot is not None:
            body = self.body_rect()
            path = QPainterPath()
            path.addRoundedRect(body, self.radius, self.radius)
            g.setRenderHint(QPainter.Antialiasing)
            g.setClipPath(path)
            rad = max(120.0, min(body.width(), 420.0) * 0.55)
            rg = QRadialGradient(self._spot, rad)
            dark = STATE.get("dark")
            rg.setColorAt(0.0, _qc("primary", 30 if not dark else 40))
            rg.setColorAt(0.5, _qc("primary", 10 if not dark else 14))
            rg.setColorAt(1.0, _qc("primary", 0))
            g.fillRect(body, QBrush(rg))
        g.end()

    def _paint_card(self, g: QPainter) -> None:
        g.setRenderHint(QPainter.Antialiasing)
        body = self.body_rect()
        dark = bool(STATE.get("dark"))
        # shadow (9-slice of a cached blurred tile)
        lift = self._lift
        sc = _qc("shadow", int((26 if not dark else 90) + 22 * lift))
        blur = 14
        tile = _shadow_tile(int(self.radius), blur, sc)
        off = 4 + 5 * lift
        sr = body.adjusted(-blur * 1.2, -blur * 1.2 + off, blur * 1.2, blur * 1.2 + off)
        self._nine(g, tile, sr, int(self.radius + blur * 2 + 4))
        # body
        path = QPainterPath()
        path.addRoundedRect(body, self.radius, self.radius)
        fill = QColor(C["surface"])
        fill.setAlpha((214 if not self._strong else 236) if not dark else (205 if not self._strong else 232))
        g.fillPath(path, fill)
        if self.hero:                       # the banner: accent -> its partner -> sunshine, with bubbles
            hg = QLinearGradient(body.topLeft(), body.bottomRight())
            hg.setColorAt(0.0, _qc("primary"))
            hg.setColorAt(0.62, _qc("primary_grad"))
            hg.setColorAt(1.0, QColor(mix_hex(C["primary_grad"], C["sun"], 0.55)))
            g.fillPath(path, hg)
            g.save()
            g.setClipPath(path)
            g.setPen(Qt.NoPen)
            for fx, fy, fr, a in ((0.86, 0.08, 0.34, 34), (0.97, 0.78, 0.22, 26), (0.58, 1.05, 0.2, 22),
                                  (0.08, -0.15, 0.18, 20)):
                g.setBrush(QColor(255, 255, 255, a))
                rr = body.height() * fr * 1.6
                g.drawEllipse(QPointF(body.left() + body.width() * fx, body.top() + body.height() * fy), rr, rr)
            g.restore()
        if self.wash:                       # KPI tiles: a faint wash of their colour
            wg = QLinearGradient(body.topLeft(), body.bottomRight())
            wg.setColorAt(0.0, _qc(self.wash, 46 if not dark else 60))
            wg.setColorAt(0.6, _qc(self.wash, 8))
            wg.setColorAt(1.0, _qc(self.wash, 0))
            g.fillPath(path, wg)
        if self.tint:                       # soft accent wash
            wash = QLinearGradient(body.topLeft(), body.bottomRight())
            wash.setColorAt(0.0, _qc("primary", 70 if not dark else 90))
            wash.setColorAt(0.55, _qc("blob2", 60 if not dark else 70))
            wash.setColorAt(1.0, _qc("blob1", 90 if not dark else 80))
            g.fillPath(path, wash)
        # top sheen
        sheen = QLinearGradient(body.topLeft(), body.bottomLeft())
        sheen.setColorAt(0.0, QColor(255, 255, 255, 70 if not dark else 14))
        sheen.setColorAt(0.35, QColor(255, 255, 255, 0))
        g.fillPath(path, sheen)
        # edge: bright hairline, accent glow on hover / flash
        edge = QColor(255, 255, 255, 235 if not dark else 30)
        g.setPen(QPen(edge, 1.0))
        g.drawPath(path)
        glow = max(lift * 0.55, self._flash)
        if glow > 0.01:
            col = _qc("primary", int(150 * glow))
            g.setPen(QPen(col, 1.6 + 1.2 * self._flash))
            g.drawRoundedRect(body.adjusted(0.5, 0.5, -0.5, -0.5), self.radius, self.radius)

    @staticmethod
    def _nine(g: QPainter, tile: QImage, r: QRectF, corner: int) -> None:
        tw, th = tile.width(), tile.height()
        c = min(corner, tw // 2 - 1, int(r.width() // 2), int(r.height() // 2))
        if c <= 0:
            return
        src = [(0, c), (c, tw - 2 * c), (tw - c, c)]
        srcy = [(0, c), (c, th - 2 * c), (th - c, c)]
        dx = [(r.left(), c), (r.left() + c, r.width() - 2 * c), (r.right() - c, c)]
        dy = [(r.top(), c), (r.top() + c, r.height() - 2 * c), (r.bottom() - c, c)]
        for i in range(3):
            for j in range(3):
                sx, sw = src[i]
                sy, sh = srcy[j]
                x, w = dx[i]
                y, h = dy[j]
                if w > 0 and h > 0 and sw > 0 and sh > 0:
                    g.drawImage(QRectF(x, y, w, h), tile, QRectF(sx, sy, sw, sh))


class HeroFx(QWidget):
    """Life in a hero banner: soft bubbles drifting up through it and a few twinkling sparkles.
    It sits behind the banner's text and buttons, follows the banner's size, and only runs while
    the banner is on screen in the active window (full motion)."""

    def __init__(self, card: "GlassCard", count: int = 14, seed: int = 5):
        super().__init__(card)
        import random
        self.card = card
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.lower()
        r = random.Random(seed)
        # (x 0..1, phase 0..1, radius 0..1, speed, sway, alpha)
        self.bubbles = [(r.random(), r.random(), 0.25 + 0.75 * r.random() ** 2, 0.025 + 0.04 * r.random(),
                         r.uniform(0.01, 0.03), r.randint(22, 52)) for _ in range(count)]
        self.sparkles = [(r.random(), r.random(), r.random() * 6.28, r.uniform(1.2, 2.4)) for _ in range(9)]
        self._t0 = time.perf_counter()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        card.installEventFilter(self)
        self.setGeometry(card.rect())

    def eventFilter(self, obj, ev):
        if obj is self.card:
            if ev.type() == QEvent.Resize:
                self.setGeometry(self.card.rect())
            elif ev.type() == QEvent.Show and full_motion():
                self._timer.start(40)
            elif ev.type() == QEvent.Hide:
                self._timer.stop()
        return False

    def _tick(self):
        win = self.window()
        if not full_motion() or win is None or not win.isActiveWindow() or not self.isVisible():
            if self._timer.interval() != 250:
                self._timer.setInterval(250)      # rest in the background
            return
        if self._timer.interval() != 40:
            self._timer.setInterval(40)
        self.update()

    def paintEvent(self, _ev):
        body = self.card.body_rect()
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        clip = QPainterPath()
        clip.addRoundedRect(body, self.card.radius, self.card.radius)
        g.setClipPath(clip)
        g.setPen(Qt.NoPen)
        t = time.perf_counter() - self._t0
        H = body.height()
        for x, ph, rad, sp, sway, a in self.bubbles:
            u = (ph + t * sp) % 1.0                         # 0 bottom .. 1 top
            rr = (6 + 26 * rad) * (0.7 + 0.3 * math.sin(u * math.pi))
            cx = body.left() + body.width() * (x + sway * math.sin(t * 0.7 + ph * 9))
            cy = body.bottom() + rr - u * (H + 2 * rr)
            fade = math.sin(u * math.pi)
            g.setBrush(QColor(255, 255, 255, int(a * fade)))
            g.drawEllipse(QPointF(cx, cy), rr, rr)
        for x, y, ph, sp in self.sparkles:
            k = max(0.0, math.sin(t * sp + ph)) ** 3
            if k < 0.05:
                continue
            cx = body.left() + body.width() * (0.35 + 0.6 * x)
            cy = body.top() + H * (0.12 + 0.76 * y)
            s = 2.5 + 4.5 * k
            g.setBrush(QColor(255, 255, 255, int(200 * k)))
            path = QPainterPath()
            path.moveTo(cx, cy - s)
            path.quadTo(cx, cy, cx + s, cy)
            path.quadTo(cx, cy, cx, cy + s)
            path.quadTo(cx, cy, cx - s, cy)
            path.quadTo(cx, cy, cx, cy - s)
            g.drawPath(path)
        g.end()


# ====================================================================== page transitions
class _Overlay(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.old: Optional[QPixmap] = None
        self.new: Optional[QPixmap] = None
        self.t = 0.0
        self.dy = 26.0
        self.slide = True

    def paintEvent(self, _ev):
        g = QPainter(self)
        g.setRenderHint(QPainter.SmoothPixmapTransform)
        t = self.t
        if self.old is not None:
            g.setOpacity(max(0.0, 1.0 - t * 1.8))
            g.drawPixmap(QPointF(0, -10 * t if self.slide else 0), self.old)
        if self.new is not None:
            g.setOpacity(min(1.0, t * 1.25))
            g.drawPixmap(QPointF(0, self.dy * (1.0 - t) if self.slide else 0), self.new)
        g.end()


class AnimatedStack(QStackedWidget):
    """QStackedWidget whose page changes rise + fade (full motion) or fade (reduced motion)."""

    DURATION = 360

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ov = _Overlay(self)
        self._ov.hide()
        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.valueChanged.connect(self._tick)
        self._anim.finished.connect(self._done)
        self._target = -1

    def setCurrentIndex(self, idx: int) -> None:          # noqa: N802 - Qt API
        rise = getattr(self, "_card_rise", None)
        if rise is not None:
            # the previous page's cards were still arriving: land them now (no ghosts over the new page)
            self._card_rise = None
            try:
                rise._a.stop()
                rise._finish()
            except RuntimeError:
                pass
        cur = self.currentIndex()
        if idx == cur or not animated() or not self.isVisible() or cur < 0:
            super().setCurrentIndex(idx)
            return
        old = snapshot(self.currentWidget()) if self.currentWidget() is not None else None
        w = self.widget(idx)
        w.resize(self.size())
        super().setCurrentIndex(idx)
        self._ov.slide = full_motion()
        self._ov.t = 0.0
        self._ov.setGeometry(self.rect())
        self._target = idx
        if full_motion() and w.findChildren(GlassCard):
            # the old page lifts away while the new page's cards cascade in one after another
            self._ov.old, self._ov.new = old, None
            self._ov.show()
            self._ov.raise_()
            cascade_in(w)
            self._anim.stop()
            self._anim.setDuration(240)
            self._anim.setEasingCurve(QEasingCurve.OutCubic)
            self._anim.start()
            return
        new = snapshot(w)
        self._ov.old, self._ov.new = old, new
        self._ov.show()
        self._ov.raise_()
        w.setVisible(False)
        self._anim.stop()
        self._anim.setDuration(self.DURATION if full_motion() else 200)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.start()

    def _tick(self, v):
        self._ov.t = float(v)
        self._ov.update()

    def _done(self):
        self._ov.hide()
        self._ov.old = self._ov.new = None
        w = self.widget(self._target) if self._target >= 0 else None
        if w is not None:
            w.setVisible(True)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._ov.setGeometry(self.rect())


# ====================================================================== navigation rail
def _paint_indicator(g: QPainter, r: QRectF) -> None:
    """The selected navigation item: a vivid gradient pill (accent -> its partner colour) with a soft
    glow under it; the item's text and icon turn white on it."""
    g.setRenderHint(QPainter.Antialiasing)
    r = r.adjusted(0.5, 0.5, -0.5, -0.5)
    g.setPen(Qt.NoPen)
    for k, a in ((5, 16), (3, 26), (1.5, 38)):          # glow
        g.setBrush(_qc("primary", a if not STATE.get("dark") else a + 10))
        g.drawRoundedRect(r.adjusted(-k * 0.3, k * 0.6, k * 0.3, k * 1.1), 14 + k, 14 + k)
    bg = QLinearGradient(r.topLeft(), r.topRight())
    bg.setColorAt(0, _qc("primary"))
    bg.setColorAt(1, _qc("primary_grad"))
    g.setBrush(bg)
    g.drawRoundedRect(r, 14, 14)
    sheen = QLinearGradient(r.topLeft(), r.bottomLeft())
    sheen.setColorAt(0, QColor(255, 255, 255, 46))
    sheen.setColorAt(0.5, QColor(255, 255, 255, 0))
    g.setBrush(sheen)
    g.drawRoundedRect(r, 14, 14)


class NavRail(QWidget):
    """Vertical navigation with a gliding gradient indicator.

    The indicator is painted by the rail itself, always exactly one button tall: it glides between
    the old and the new button's place (a separate child widget with an animated geometry could be
    left stretched over the whole rail when an animation was cut short by a re-layout)."""

    page_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Transparent")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(3)
        self.group = QButtonGroup(self)
        self.group.idClicked.connect(self.page_requested.emit)
        self.buttons: list[QPushButton] = []
        self._cur = -1
        self._from = -1
        self._t = 1.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(320)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.valueChanged.connect(self._step)

    def _step(self, v) -> None:
        self._t = float(v)
        self.update()

    def add(self, text: str, icon=None, separator_before: bool = False) -> QPushButton:
        if separator_before:
            sep = QFrame()
            sep.setObjectName("NavSep")
            sep.setFixedHeight(1)
            self.lay.addSpacing(4)
            self.lay.addWidget(sep)
            self.lay.addSpacing(4)
        b = QPushButton(text)
        b.setObjectName("Nav")
        b.setCheckable(True)
        b.setCursor(Qt.PointingHandCursor)
        if icon is not None:
            b.setIcon(icon)
            b.setIconSize(QSize(18, 18))
        self.group.addButton(b, len(self.buttons))
        self.lay.addWidget(b)
        self.buttons.append(b)
        return b

    def select(self, idx: int, animate: bool = True) -> None:
        if not (0 <= idx < len(self.buttons)):
            return
        self.buttons[idx].setChecked(True)
        if idx == self._cur:
            self.update()
            return
        prev = self._cur
        self._cur = idx
        if animate and animated() and prev >= 0 and self.isVisible():
            self._from = prev
            self._anim.stop()
            self._anim.start()
        else:
            self._anim.stop()
            self._from, self._t = idx, 1.0
        self.update()

    def indicator_rect(self) -> QRectF:
        """Where the indicator is now (empty when nothing is selected)."""
        if not (0 <= self._cur < len(self.buttons)):
            return QRectF()
        b = QRectF(self.buttons[self._cur].geometry())
        if not (0 <= self._from < len(self.buttons)) or self._t >= 1.0:
            return b
        a = QRectF(self.buttons[self._from].geometry())
        t = self._t
        y = a.top() + (b.top() - a.top()) * t
        return QRectF(b.left(), y, b.width(), b.height())

    def paintEvent(self, _ev):
        r = self.indicator_rect()
        if r.isEmpty():
            return
        g = QPainter(self)
        _paint_indicator(g, r)
        g.end()


# ====================================================================== micro-interactions
class CountUp(QObject):
    """Animate a number shown in a label: ``CountUp(label, fmt).set(42)``."""

    def __init__(self, label: QLabel, fmt: Callable[[float], str], duration: int = 900):
        super().__init__(label)
        self.label = label
        self.fmt = fmt
        self.value: Optional[float] = None
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(duration)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(lambda v: self.label.setText(self.fmt(float(v))))

    def set(self, value: float) -> None:
        value = float(value)
        if self.value is not None and abs(value - self.value) < 1e-9:
            return
        start = self.value if self.value is not None else 0.0
        self.value = value
        if not animated() or abs(value - start) < 1e-9:
            self.label.setText(self.fmt(value))
            return
        self._anim.stop()
        self._anim.setStartValue(start)
        self._anim.setEndValue(value)
        self._anim.start()


class Ripple(QWidget):
    """An expanding soft circle from the press point, clipped to the button's rounded shape."""

    def __init__(self, button: QWidget, pos: QPoint):
        super().__init__(button)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(button.rect())
        self.center = QPointF(pos)
        self.t = 0.0
        on_accent = button.objectName() in ("Primary",) or (isinstance(button, QAbstractButton)
                                                            and button.isCheckable() and button.isChecked()
                                                            and button.objectName() == "Chip")
        self.color = QColor(255, 255, 255) if on_accent else QColor(C["primary"])
        r = button.rect()
        self.maxr = math.hypot(max(pos.x(), r.width() - pos.x()), max(pos.y(), r.height() - pos.y()))
        self.radius = min(r.height() / 2, 20)
        a = QVariantAnimation(self)
        a.setDuration(520)
        a.setStartValue(0.0)
        a.setEndValue(1.0)
        a.setEasingCurve(QEasingCurve.OutCubic)
        a.valueChanged.connect(self._tick)
        a.finished.connect(self.deleteLater)
        a.start()
        self.show()

    def _tick(self, v):
        self.t = float(v)
        self.update()

    def paintEvent(self, _ev):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(self.rect()), self.radius, self.radius)
        g.setClipPath(clip)
        c = QColor(self.color)
        c.setAlpha(int((70 if self.color == QColor(255, 255, 255) else 45) * (1.0 - self.t)))
        g.setPen(Qt.NoPen)
        g.setBrush(c)
        r = self.maxr * (0.15 + 0.85 * self.t)
        g.drawEllipse(self.center, r, r)
        g.end()


class _ButtonFx(QObject):
    """App-wide event filter: ripples on button presses, eased wheel scrolling."""

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QEvent.MouseButtonPress and isinstance(obj, QPushButton) and full_motion():
            if obj.isEnabled() and obj.objectName() not in ("Avatar",):
                Ripple(obj, ev.position().toPoint())
        elif t == QEvent.Wheel and full_motion():
            area = obj.parent() if obj.objectName() == "qt_scrollarea_viewport" else None
            if isinstance(area, QAbstractScrollArea) and not (ev.modifiers() & Qt.ControlModifier):
                sb = area.verticalScrollBar()
                if sb is not None and sb.maximum() > 0 and ev.angleDelta().y() != 0:
                    SmoothScroller.of(sb).scroll(-ev.angleDelta().y())
                    return True
        return False


class SmoothScroller(QObject):
    _instances: dict = {}

    @classmethod
    def of(cls, sb) -> "SmoothScroller":
        s = cls._instances.get(id(sb))
        if s is None or s.sb is not sb:
            s = cls(sb)
            cls._instances[id(sb)] = s
            sb.destroyed.connect(lambda *_a, k=id(sb): cls._instances.pop(k, None))
        return s

    def __init__(self, sb):
        super().__init__(sb)
        self.sb = sb
        self.target = sb.value()
        self.anim = QPropertyAnimation(sb, b"value", self)
        self.anim.setDuration(300)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)

    def scroll(self, delta: int) -> None:
        if self.anim.state() != QPropertyAnimation.Running:
            self.target = self.sb.value()
        step = delta * 0.9
        self.target = max(self.sb.minimum(), min(self.sb.maximum(), int(self.target + step)))
        self.anim.stop()
        self.anim.setStartValue(self.sb.value())
        self.anim.setEndValue(self.target)
        self.anim.start()


_FX: list = []


def install(app: QApplication) -> None:
    """Install the app-wide micro-interactions once."""
    if _FX:
        return
    fx = _ButtonFx(app)
    app.installEventFilter(fx)
    _FX.append(fx)


def shake(widget: QWidget, amplitude: int = 12) -> None:
    """Horizontal shake (wrong PIN)."""
    if not animated():
        return
    a = QPropertyAnimation(widget, b"pos", widget)
    p = widget.pos()
    a.setDuration(420)
    for k, f in enumerate((0, -1, 1, -0.8, 0.8, -0.5, 0.5, -0.2, 0)):
        a.setKeyValueAt(k / 8, QPoint(p.x() + int(f * amplitude), p.y()))
    a.start(QPropertyAnimation.DeleteWhenStopped)


def snapshot(widget: QWidget) -> QPixmap:
    """The widget and its children on a transparent pixmap (``QWidget.grab`` paints the window's
    background colour under a see-through widget, which flashes as a pale box in an animation)."""
    dpr = widget.devicePixelRatioF()
    pm = QPixmap(max(1, int(widget.width() * dpr)), max(1, int(widget.height() * dpr)))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    widget.render(pm, QPoint(0, 0), QRegion(), QWidget.RenderFlag.DrawChildren)
    return pm


class CardRise(QWidget):
    """Cards arriving one after another: each rises a little, scales up from 96 % and fades in,
    a beat after the one before. Drawn from snapshots on one overlay (so it stays smooth however
    heavy the cards are); the real cards are revealed when their snapshot has landed."""

    def __init__(self, host: QWidget, items: list, step_ms: int, duration: int):
        super().__init__(host)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(host.rect())
        self.items = items                    # [(card, pixmap, QPointF top-left, delay ms)]
        self.duration = duration
        self.t = 0.0
        total = (len(items) - 1) * step_ms + duration if items else duration
        self._done = set()
        a = QVariantAnimation(self)
        a.setDuration(total)
        a.setStartValue(0.0)
        a.setEndValue(float(total))
        a.valueChanged.connect(self._tick)
        a.finished.connect(self._finish)
        self._a = a
        self.show()
        self.raise_()
        a.start()

    def _tick(self, v):
        self.t = float(v)
        for i, (card, _pm, _pos, delay) in enumerate(self.items):
            if i not in self._done and self.t >= delay + self.duration:
                self._done.add(i)
                self._reveal(card)
        self.update()

    @staticmethod
    def _reveal(card) -> None:
        try:
            eff = card.graphicsEffect()
            if eff is not None and getattr(card, "_rise_effect", None) is eff:
                card.setGraphicsEffect(None)
        except RuntimeError:                 # the card was deleted meanwhile
            pass

    def _finish(self):
        for card, *_ in self.items:
            self._reveal(card)
        self.hide()
        self.deleteLater()

    def paintEvent(self, _ev):
        g = QPainter(self)
        g.setRenderHint(QPainter.SmoothPixmapTransform)
        for i, (_card, pm, pos, delay) in enumerate(self.items):
            if i in self._done:
                continue
            u = max(0.0, min(1.0, (self.t - delay) / self.duration))
            if u <= 0.0:
                continue
            e = 1.0 - (1.0 - u) ** 3                       # ease out
            w = pm.width() / pm.devicePixelRatio()
            h = pm.height() / pm.devicePixelRatio()
            sc = 0.96 + 0.04 * e
            dy = 28.0 * (1.0 - e)
            g.setOpacity(e)
            g.save()
            g.translate(pos.x() + w / 2, pos.y() + h / 2 + dy)
            g.scale(sc, sc)
            g.drawPixmap(QRectF(-w / 2, -h / 2, w, h), pm, QRectF(pm.rect()))
            g.restore()
        g.end()


def cascade_in(page: QWidget, start: float = 0.0, step_ms: int = 55, duration: int = 420,
               max_cards: int = 12) -> int:
    """The page's cards arrive one after another (top to bottom, left to right): each rises,
    grows and fades in a little after the one before (see ``CardRise``). Only the cards that are
    on screen; returns how many animate."""
    if not animated():
        return 0
    from PySide6.QtWidgets import QGraphicsOpacityEffect
    host = page.parentWidget() or page
    vis = page.visibleRegion().boundingRect()
    cards = []
    for c in page.findChildren(GlassCard):
        if not c.isVisible() or c.graphicsEffect() is not None:
            continue
        # top-level cards only (a card inside a card arrives with its parent)
        par, nested = c.parentWidget(), False
        while par is not None and par is not page:
            if isinstance(par, GlassCard):
                nested = True
                break
            par = par.parentWidget()
        if nested:
            continue
        pos = c.mapTo(page, QPoint(0, 0))
        if vis.isValid() and not vis.intersects(c.geometry().translated(pos - c.pos())):
            continue
        cards.append((pos.y() // 40, pos.x(), c))
    cards.sort(key=lambda t: (t[0], t[1]))
    cards = [c for _y, _x, c in cards[:max_cards]]
    if not cards:
        return 0
    if not full_motion():
        # reduced motion: a plain, quick fade
        for i, c in enumerate(cards):
            eff = QGraphicsOpacityEffect(c)
            eff.setOpacity(start)
            c.setGraphicsEffect(eff)
            a = QPropertyAnimation(eff, b"opacity", c)
            a.setDuration(220)
            a.setStartValue(start)
            a.setEndValue(1.0)

            def _end(card=c, effect=eff):
                if card.graphicsEffect() is effect:
                    card.setGraphicsEffect(None)
            a.finished.connect(_end)
            a.start()
            c._cascade_anim = a
        return len(cards)
    items = []
    for i, c in enumerate(cards):
        pm = snapshot(c)
        eff = QGraphicsOpacityEffect(c)
        eff.setOpacity(0.0)
        c.setGraphicsEffect(eff)
        c._rise_effect = eff
        items.append((c, pm, QPointF(c.mapTo(host, QPoint(0, 0))), i * step_ms))
    host._card_rise = CardRise(host, items, step_ms, duration)
    return len(cards)


def fade_in(widget: QWidget, duration: int = 260) -> None:
    """Window fade-in on show."""
    if not animated() or not widget.isWindow():
        return
    widget.setWindowOpacity(0.0)
    a = QPropertyAnimation(widget, b"windowOpacity", widget)
    a.setDuration(duration)
    a.setStartValue(0.0)
    a.setEndValue(1.0)
    a.setEasingCurve(QEasingCurve.OutCubic)
    a.start(QPropertyAnimation.DeleteWhenStopped)


class Toast(QWidget):
    """Slide-in notification at the bottom-right of a window."""

    _current: Optional["Toast"] = None

    def __init__(self, parent: QWidget, text: str, kind: str = "info", ms: int = 2600):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.text = text
        self.kind = kind
        fm = self.fontMetrics()
        w = min(420, fm.horizontalAdvance(text) + 64)
        self.resize(w, 46)
        self._y_show = parent.height() - self.height() - 22
        self._y_hide = parent.height() + 6
        self.move(parent.width() - w - 24, self._y_hide)
        self.show()
        self.raise_()
        a = QPropertyAnimation(self, b"pos", self)
        a.setDuration(380 if animated() else 1)
        a.setEasingCurve(QEasingCurve.OutBack)
        a.setStartValue(self.pos())
        a.setEndValue(QPoint(self.x(), self._y_show))
        a.start()
        self._a = a
        QTimer.singleShot(ms, self._hide)

    @classmethod
    def show_text(cls, parent: QWidget, text: str, kind: str = "info") -> "Toast":
        if cls._current is not None:
            try:
                cls._current.deleteLater()
            except RuntimeError:
                pass
        cls._current = Toast(parent, text, kind)
        return cls._current

    def _hide(self):
        try:
            a = QPropertyAnimation(self, b"pos", self)
        except RuntimeError:
            return
        a.setDuration(260 if animated() else 1)
        a.setEasingCurve(QEasingCurve.InCubic)
        a.setStartValue(self.pos())
        a.setEndValue(QPoint(self.x(), self._y_hide))
        a.finished.connect(self.deleteLater)
        a.start()
        self._a = a

    def paintEvent(self, _ev):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        g.setPen(Qt.NoPen)
        g.setBrush(_qc("tooltip_bg", 240))
        g.drawRoundedRect(r, 16, 16)
        dot = {"ok": "focus", "warn": "warning", "error": "danger"}.get(self.kind, "primary")
        g.setBrush(_qc(dot))
        g.drawEllipse(QPointF(22, r.center().y()), 5, 5)
        g.setPen(_qc("tooltip_fg"))
        g.drawText(r.adjusted(38, 0, -14, 0), Qt.AlignVCenter | Qt.AlignLeft, self.text)
        g.end()
