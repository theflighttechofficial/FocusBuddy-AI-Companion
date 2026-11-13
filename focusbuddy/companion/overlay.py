"""Transparent, frameless desktop overlay windows: the companion + its speech bubble."""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPoint, QPointF, QRectF, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from ..ui.theme import C, FONT_FAMILY

OVERLAY_FLAGS = (Qt.FramelessWindowHint | Qt.Tool | Qt.WindowDoesNotAcceptFocus
                 | Qt.NoDropShadowWindowHint)


class CompanionWindow(QWidget):
    """Tiny translucent window the character is painted into.

    Fully transparent pixels are click-through on Windows (layered window),
    so only the character itself catches the mouse.
    """

    clicked = Signal()
    double_clicked = Signal()
    context_menu = Signal(QPoint)
    drag_started = Signal()
    dragged = Signal(QPoint)       # global position of the cursor
    drag_finished = Signal()
    hovered = Signal(bool)
    resize_steps = Signal(int)     # Ctrl + mouse wheel over the character: +1 bigger / -1 smaller

    def __init__(self, size, always_on_top: bool = True):
        super().__init__(None, OVERLAY_FLAGS | (Qt.WindowStaysOnTopHint if always_on_top else Qt.Widget))
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setWindowTitle("FocusBuddy Companion")
        self.set_size(size)
        self.setMouseTracking(True)
        self.painter_cb = None          # callable(QPainter, QRectF)
        # a frame rendered ahead of time (``present``): the window then moves and shows it in one go,
        # so the picture and its position on screen always belong to the same moment
        self.frame = None
        self._press_pos: QPoint | None = None
        self._grab_offset = QPoint()
        self._dragging = False
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.timeout.connect(self.clicked.emit)

    def set_size(self, w, h: int | None = None) -> None:
        """Square ``set_size(n)`` (legacy) or ``set_size(w, h)`` / ``set_size((w, h))``."""
        if isinstance(w, (tuple, list)):
            w, h = w
        self.setFixedSize(int(w), int(w if h is None else h))

    def set_click_through(self, on: bool) -> None:
        visible = self.isVisible()
        self.setWindowFlag(Qt.WindowTransparentForInput, on)
        if visible:
            self.show()

    def set_always_on_top(self, on: bool) -> None:
        visible = self.isVisible()
        self.setWindowFlag(Qt.WindowStaysOnTopHint, on)
        if visible:
            self.show()

    # ------------------------------------------------------------- painting
    def paintEvent(self, _ev) -> None:
        g = QPainter(self)
        g.setCompositionMode(QPainter.CompositionMode_Source)
        fr = self.frame
        if fr is not None and not fr.isNull() and fr.deviceIndependentSize().toSize() == self.size():
            g.drawImage(0, 0, fr)                      # the pre-rendered frame: a fast blit
            g.end()
            return
        g.fillRect(self.rect(), Qt.transparent)
        g.setCompositionMode(QPainter.CompositionMode_SourceOver)
        if self.painter_cb:
            self.painter_cb(g, QRectF(self.rect()))
        g.end()

    def present(self, frame, x: int, y: int) -> None:
        """Show a pre-rendered frame at (x, y): move, then repaint synchronously (a quick blit), so the
        window never shows an old picture at a new position. Moving the window every timer tick while
        the picture was re-rendered less often made the feet skate over the ground between frames."""
        self.frame = frame
        if self.x() != x or self.y() != y:
            self.move(x, y)
        if self.isVisible():
            self.repaint()

    # ---------------------------------------------------------------- mouse
    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self._press_pos = ev.globalPosition().toPoint()
            self._grab_offset = ev.position().toPoint()
            self._dragging = False
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:
        if self._press_pos is not None and ev.buttons() & Qt.LeftButton:
            gp = ev.globalPosition().toPoint()
            if not self._dragging and (gp - self._press_pos).manhattanLength() > 5:
                self._dragging = True
                self._click_timer.stop()
                self.drag_started.emit()
            if self._dragging:
                self.dragged.emit(gp)
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            if self._dragging:
                self._dragging = False
                self.drag_finished.emit()
            elif self._press_pos is not None:
                self._click_timer.start(QGuiApplication.styleHints().mouseDoubleClickInterval())
            self._press_pos = None
        ev.accept()

    def mouseDoubleClickEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self._click_timer.stop()
            self.double_clicked.emit()
        ev.accept()

    def contextMenuEvent(self, ev) -> None:
        self.context_menu.emit(ev.globalPos())
        ev.accept()

    def wheelEvent(self, ev) -> None:
        if ev.modifiers() & Qt.ControlModifier:
            d = ev.angleDelta().y()
            if d:
                self.resize_steps.emit(1 if d > 0 else -1)
            ev.accept()
            return
        ev.ignore()

    def enterEvent(self, _ev) -> None:
        self.hovered.emit(True)

    def leaveEvent(self, _ev) -> None:
        self.hovered.emit(False)


class BubbleWindow(QWidget):
    """Speech bubble with optional buttons and countdown."""

    action = Signal(str)
    dismissed = Signal()

    TAIL = 12
    RADIUS = 16
    PAD = 14

    def __init__(self):
        super().__init__(None, OVERLAY_FLAGS | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setWindowTitle("FocusBuddy Bubble")
        self.tail_down = True
        self.tail_x = 40.0
        self.accent = C["primary"]

        root = QVBoxLayout(self)
        root.setContentsMargins(self.PAD + 4, self.PAD + 2, self.PAD + 4, self.PAD + self.TAIL + 4)
        self.content = QWidget()
        self.content.setAttribute(Qt.WA_TranslucentBackground, True)
        sp = self.content.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.content.setSizePolicy(sp)
        root.addWidget(self.content)
        outer = QVBoxLayout(self.content)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)
        self.title = QLabel()
        self.title.hide()
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setMaximumWidth(270)
        self.text.setMinimumWidth(140)
        self.restyle()
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()
        self.buttons = QHBoxLayout()
        self.buttons.setSpacing(6)
        outer.addWidget(self.title)
        outer.addWidget(self.text)
        outer.addWidget(self.progress)
        outer.addLayout(self.buttons)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.fade_out)
        # appear/disappear: fade + springy scale around the tail tip
        self._anim = QVariantAnimation(self)
        self._anim.valueChanged.connect(self._on_anim)
        self._anim.finished.connect(self._anim_done)
        self._progress = 1.0
        self._snap = None
        self._closing = False
        self.has_buttons = False

    # ----------------------------------------------------------------- api
    def restyle(self) -> None:
        """Re-read the theme colours (light / dark can change while the app runs)."""
        self.title.setStyleSheet(f"font-family:{FONT_FAMILY}; font-size:11px; font-weight:700; color:{C['muted']};"
                                 "letter-spacing:0.5px; background:transparent;")
        self.text.setStyleSheet(f"font-family:{FONT_FAMILY}; font-size:14px; color:{C['text']}; background:transparent;")

    def show_message(self, text: str, buttons: list[tuple[str, str, bool]] | None = None,
                     duration: float | None = None, title: str = "", accent: str | None = None) -> None:
        if getattr(self, "_theme_key", None) != (C["text"], C["surface"]):
            self._theme_key = (C["text"], C["surface"])
            self.restyle()
        self.accent = accent or C["primary"]
        self.title.setText(title.upper())
        self.title.setVisible(bool(title))
        self.text.setText(text)
        self._set_buttons(buttons or [])
        self.progress.hide()
        self.adjustSize()
        self._hide_timer.stop()
        if not buttons:
            if duration is None:
                duration = min(9.0, 2.8 + len(text) / 18)
            self._hide_timer.start(int(duration * 1000))
        self._closing = False
        if not self.isVisible():
            self._start_anim(appear=True)
            self.show()
        else:
            self._anim.stop()
            self._finish_anim_visuals()
        self.update()

    # --------------------------------------------------------- animation
    def _start_anim(self, appear: bool) -> None:
        self._anim.stop()
        self.layout().activate()
        self._snap = self.content.grab()
        self.content.hide()
        if appear:
            self._anim.setDuration(240)
            self._anim.setEasingCurve(QEasingCurve.Linear)
            self._anim.setStartValue(0.0)
            self._anim.setEndValue(1.0)
            self._progress = 0.0
            self.setWindowOpacity(0.0)
        else:
            self._anim.setDuration(170)
            self._anim.setEasingCurve(QEasingCurve.Linear)
            self._anim.setStartValue(self._progress)
            self._anim.setEndValue(0.0)
        self._anim.start()

    def _on_anim(self, v) -> None:
        self._progress = float(v)
        self.setWindowOpacity(min(1.0, self._progress * 1.6))
        self.update()

    def _finish_anim_visuals(self) -> None:
        self._progress = 1.0
        self._snap = None
        self.content.show()
        self.setWindowOpacity(1.0)

    def _scale(self) -> float:
        p = self._progress
        if self._closing:
            return 0.9 + 0.1 * p
        x = p - 1   # ease-out-back
        k = 1.7
        return 0.78 + 0.22 * (1 + (k + 1) * x * x * x + k * x * x)

    def set_countdown(self, remaining: float, total: float) -> None:
        self.progress.show()
        self.progress.setRange(0, 1000)
        self.progress.setValue(int(1000 * max(0.0, remaining) / max(0.1, total)))
        self.progress.setStyleSheet(f"QProgressBar{{background:{C['border']};border-radius:3px;}}"
                                    f"QProgressBar::chunk{{background:{C['danger']};border-radius:3px;}}")

    def set_text(self, text: str) -> None:
        self.text.setText(text)

    def fade_out(self) -> None:
        if not self.isVisible() or self._closing:
            return
        self._closing = True
        self._start_anim(appear=False)

    def hide_now(self) -> None:
        self._hide_timer.stop()
        self._anim.stop()
        self._closing = False
        self._finish_anim_visuals()
        self.hide()

    def _anim_done(self) -> None:
        if self._closing:
            self._closing = False
            self.hide()
            self._finish_anim_visuals()
            self.setWindowOpacity(0.0)
            self.dismissed.emit()
        else:
            self._finish_anim_visuals()

    def _set_buttons(self, buttons: list[tuple[str, str, bool]]) -> None:
        while self.buttons.count():
            item = self.buttons.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.has_buttons = bool(buttons)
        for action_id, label, primary in buttons:
            b = QPushButton(label)
            b.setCursor(Qt.PointingHandCursor)
            b.setFocusPolicy(Qt.NoFocus)
            if primary:
                from ..ui.theme import mix
                hover = mix(self.accent, "#000000", 0.14)
                css = (f"QPushButton{{background:{self.accent};color:white;border:none;border-radius:8px;"
                       f"padding:6px 12px;font-family:{FONT_FAMILY};font-size:12px;font-weight:600;}}"
                       f"QPushButton:hover{{background:{hover};}}")
            else:
                css = (f"QPushButton{{background:{C['surface_alt']};color:{C['text']};border:1px solid {C['border']};"
                       f"border-radius:8px;padding:6px 12px;font-family:{FONT_FAMILY};font-size:12px;font-weight:600;}}"
                       f"QPushButton:hover{{background:{C['surface_hover']};}}")
            b.setStyleSheet(css)
            b.clicked.connect(lambda _=False, a=action_id: self.action.emit(a))
            self.buttons.addWidget(b)

    # --------------------------------------------------------------- layout
    def place(self, anchor: QPointF, screen_rect: QRectF, below_offset: float = 90.0) -> None:
        """Place the bubble so its tail points at ``anchor`` (global coords)."""
        self.adjustSize()
        w, h = self.width(), self.height()
        x = anchor.x() - w * 0.5
        x = max(screen_rect.left() + 6, min(x, screen_rect.right() - w - 6))
        y = anchor.y() - h
        self.tail_down = True
        if y < screen_rect.top() + 4:
            self.tail_down = False
            y = anchor.y() + below_offset
        self.tail_x = max(self.RADIUS + 8, min(anchor.x() - x, w - self.RADIUS - 8))
        layout = self.layout()
        if self.tail_down:
            layout.setContentsMargins(self.PAD + 4, self.PAD + 2, self.PAD + 4, self.PAD + self.TAIL + 4)
        else:
            layout.setContentsMargins(self.PAD + 4, self.PAD + self.TAIL + 4, self.PAD + 4, self.PAD + 2)
        self.move(int(x), int(y))

    def paintEvent(self, _ev) -> None:
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        if self.tail_down:
            body = r.adjusted(0, 0, 0, -self.TAIL)
        else:
            body = r.adjusted(0, self.TAIL, 0, 0)
        path = QPainterPath()
        path.addRoundedRect(body, self.RADIUS, self.RADIUS)
        tail = QPainterPath()
        tx = self.tail_x
        if self.tail_down:
            tail.moveTo(tx - 9, body.bottom() - 1)
            tail.lineTo(tx + 1, body.bottom() + self.TAIL)
            tail.lineTo(tx + 9, body.bottom() - 1)
        else:
            tail.moveTo(tx - 9, body.top() + 1)
            tail.lineTo(tx + 1, body.top() - self.TAIL)
            tail.lineTo(tx + 9, body.top() + 1)
        tail.closeSubpath()
        path = path.united(tail)
        if self._snap is not None or self._progress < 1.0:
            sc = self._scale()
            ox = self.tail_x
            oy = r.bottom() if self.tail_down else r.top()
            g.translate(ox, oy)
            g.scale(sc, sc)
            g.translate(-ox, -oy)
        # soft shadow
        g.setPen(Qt.NoPen)
        for i, a in ((3, 18), (2, 26)):
            g.setBrush(QColor(20, 30, 60, a))
            g.drawPath(path.translated(0, i))
        g.setBrush(QColor(C["surface"]))
        g.setPen(QPen(QColor(C["border"]), 1.2))
        g.drawPath(path)
        # accent strip
        g.setPen(Qt.NoPen)
        g.setBrush(QColor(self.accent))
        g.drawRoundedRect(QRectF(body.left() + 12, body.top() + 6, 26, 3.5), 2, 2)
        if self._snap is not None:
            g.drawPixmap(self.content.pos(), self._snap)
        g.end()

    def mousePressEvent(self, ev) -> None:
        if not self.has_buttons:
            self.fade_out()
        ev.accept()


def make_font(px: int, bold: bool = False) -> QFont:
    f = QFont()
    f.setFamilies([n.strip().strip("'\"") for n in FONT_FAMILY.split(",")])
    f.setPixelSize(px)
    f.setBold(bold)
    return f
