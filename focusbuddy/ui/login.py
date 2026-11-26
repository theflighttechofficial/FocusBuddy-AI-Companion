"""Log in / sign up: an app-style account screen (e-mail + password), shown once per device.

It looks like a website's or an app's log-in rather than a lock screen: a brand panel with the
companions on the left, a card with *Log in* / *Sign up* tabs on the right, labelled fields, a
show / hide toggle on the password, *Keep me signed in*, a full-width primary button with a busy
state, inline errors and a shake on a wrong password.

The account lives on this device only (``core.profiles``: salted PBKDF2 hash, nothing sent
anywhere). With *Keep me signed in* the session is remembered, so FocusBuddy opens straight to the
companion next time - the log-in appears again only after signing out.

Modes: ``"login"``, ``"signup"`` (no account yet) and ``"upgrade"`` (an existing profile from
before accounts gets an e-mail and password - its data stays where it is).
"""

from __future__ import annotations

import math
import time
from typing import Optional

from PySide6.QtCore import (QEasingCurve, QPointF, QPropertyAnimation, QRectF, Qt, QTimer, Signal,
                            Property)
from PySide6.QtGui import QColor, QFont, QIcon, QLinearGradient, QPainter, QPainterPath, QRadialGradient
from PySide6.QtWidgets import (QCheckBox, QDialog, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from ..core.profiles import AVATARS, Profile, ProfileStore
from .icons import avatar_pixmap
from .motion import GlassCard, fade_in, full_motion, shake
from .theme import C, fluent_window


def _label(text: str = "", obj: str | None = None, wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if obj:
        lb.setObjectName(obj)
    lb.setWordWrap(wrap)
    return lb


# ------------------------------------------------------------------ brand panel
class BrandPanel(QWidget):
    """The left half: a soft gradient with drifting glows, the name, a tagline, what FocusBuddy
    does, and the companions floating in a gentle orbit - sized to the window (it opens full screen).

    Smooth and cheap: the gradient and glows are painted into a small cached image a few times a
    second and scaled up (they are soft anyway); only the orbiting portraits move every frame."""

    FEATURES = ("A friendly nudge when you drift to a distracting site",
                "Focus sessions, breaks and your progress at a glance",
                "Pets, people or a little robot - pick your buddy")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._t0 = time.perf_counter()
        self._pix_size = 0
        self._pix: list = []
        self._bg = None
        self._bg_t = -1.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        if full_motion():
            self._timer.start(33)

    def _avatars(self, size: int) -> list:
        size = int(max(60, min(170, size)))
        if abs(size - self._pix_size) > 6 or not self._pix:
            self._pix = [avatar_pixmap(a, a, size, ring=False) for a in AVATARS]
            self._pix_size = size
        return self._pix

    def _background(self, r: QRectF, t: float):
        """The gradient + drifting glows at 1/4 size, re-rendered at most ~8 times a second."""
        from PySide6.QtGui import QImage
        w, h = max(8, int(r.width() / 4)), max(8, int(r.height() / 4))
        if self._bg is not None and self._bg.width() == w and self._bg.height() == h and t - self._bg_t < 0.12:
            return self._bg
        img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img.fill(Qt.transparent)
        g = QPainter(img)
        g.setRenderHint(QPainter.Antialiasing)
        q = QRectF(0, 0, w, h)
        lg = QLinearGradient(q.topLeft(), q.bottomRight())
        lg.setColorAt(0.0, QColor(C.get("primary", "#7C6CF6")))
        lg.setColorAt(0.6, QColor(C.get("primary_grad", "#FF4F8B")))
        lg.setColorAt(1.0, QColor(C.get("sun", "#FFB020")))
        g.fillRect(q, lg)
        for k, (col, ph) in enumerate(((QColor(255, 214, 196, 120), 0.0), (QColor(180, 230, 255, 110), 2.1),
                                       (QColor(255, 255, 255, 70), 4.2))):
            cx = q.center().x() + math.cos(t * 0.25 + ph) * w * 0.32
            cy = q.center().y() + math.sin(t * 0.21 + ph * 1.3) * h * 0.3
            rg = QRadialGradient(QPointF(cx, cy), w * (0.45 + 0.08 * k))
            rg.setColorAt(0.0, col)
            rg.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
            g.fillRect(q, rg)
        g.end()
        self._bg, self._bg_t = img, t
        return img

    def paintEvent(self, _ev):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        g.setRenderHint(QPainter.SmoothPixmapTransform)
        r = QRectF(self.rect())
        t = time.perf_counter() - self._t0
        card = r.adjusted(10, 10, -4, -10)
        path = QPainterPath()
        path.addRoundedRect(card, 26, 26)
        g.setClipPath(path)
        g.drawImage(card, self._background(card, t))
        big = min(card.width(), card.height())
        pad = max(40.0, card.width() * 0.07)
        # name + tagline, scaled to the panel
        g.setPen(QColor(255, 255, 255))
        f = QFont(self.font())
        hs = int(max(30, min(56, big * 0.06)))
        f.setPixelSize(hs)
        f.setWeight(QFont.Bold)
        g.setFont(f)
        y = card.top() + pad
        g.drawText(QRectF(card.left() + pad, y, card.width() - 2 * pad, hs * 1.4), Qt.AlignLeft, "FocusBuddy")
        y += hs * 1.45
        ts = int(max(15, min(22, big * 0.022)))
        f.setPixelSize(ts)
        f.setWeight(QFont.Normal)
        g.setFont(f)
        g.setPen(QColor(255, 255, 255, 230))
        g.drawText(QRectF(card.left() + pad, y, card.width() - 2 * pad, ts * 3.2), Qt.AlignLeft | Qt.TextWordWrap,
                   "A tiny friend on your desktop that helps you stay focused.")
        # what it does (only when there's room for it)
        if card.height() > 640 and card.width() > 520:
            fy = card.bottom() - pad - 26 - 3 * ts * 2.0
            f.setPixelSize(int(ts * 0.9))
            g.setFont(f)
            for i, text in enumerate(self.FEATURES):
                # each line slides in once, a moment after the other
                k = max(0.0, min(1.0, (t - 0.35 - 0.18 * i) / 0.5)) if full_motion() else 1.0
                k = 1 - (1 - k) ** 3
                g.setOpacity(k)
                yy = fy + i * ts * 2.0 + (1 - k) * 14
                g.setPen(Qt.NoPen)
                g.setBrush(QColor(255, 255, 255, 235))
                g.drawEllipse(QPointF(card.left() + pad + 6, yy + ts * 0.8), 4.5, 4.5)
                g.setPen(QColor(255, 255, 255, 240))
                g.drawText(QRectF(card.left() + pad + 22, yy, card.width() - 2 * pad - 22, ts * 1.6),
                           Qt.AlignLeft | Qt.AlignVCenter, text)
            g.setOpacity(1.0)
        # the companions on a slowly turning ring
        cx, cy = card.center().x(), card.center().y() + big * 0.03
        R = big * 0.3
        pix = self._avatars(big * 0.14)
        n = len(pix)
        items = []
        for i, pm in enumerate(pix):
            a = t * 0.18 + i * 2 * math.pi / n
            depth = (math.sin(a) + 1) * 0.5                    # 0 back .. 1 front
            x = cx + math.cos(a) * R
            yv = cy + math.sin(a) * R * 0.42 + math.sin(t * 1.6 + i) * big * 0.006
            items.append((depth, x, yv, pm))
        for depth, x, yv, pm in sorted(items, key=lambda it: it[0]):
            sc = 0.62 + 0.38 * depth
            w = pm.width() * sc / pm.devicePixelRatio()
            g.setOpacity(0.55 + 0.45 * depth)
            g.drawPixmap(QRectF(x - w / 2, yv - w / 2, w, w), pm, QRectF(pm.rect()))
        g.setOpacity(1.0)
        f.setPixelSize(12)
        g.setFont(f)
        g.setPen(QColor(255, 255, 255, 200))
        g.drawText(QRectF(card.left() + pad, card.bottom() - 34, card.width() - 2 * pad, 24), Qt.AlignLeft,
                   "Your account and activity stay on this device.")
        g.end()


# ------------------------------------------------------------------ tabs
class Tabs(QWidget):
    """Two-tab switcher with a sliding pill under the active tab."""

    changed = Signal(int)

    def __init__(self, labels: list[str], parent=None):
        super().__init__(parent)
        self.setFixedHeight(42)
        self.labels = labels
        self._idx = 0
        self._pos = 0.0
        self.anim = QPropertyAnimation(self, b"pos", self)
        self.anim.setDuration(260)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.setCursor(Qt.PointingHandCursor)

    def _get_pos(self) -> float:
        return self._pos

    def _set_pos(self, v: float) -> None:
        self._pos = float(v)
        self.update()

    pos = Property(float, _get_pos, _set_pos)

    def current(self) -> int:
        return self._idx

    def set_current(self, i: int, animate: bool = True) -> None:
        if i == self._idx and abs(self._pos - i) < 1e-3:
            return
        self._idx = i
        if animate and full_motion():
            self.anim.stop()
            self.anim.setStartValue(self._pos)
            self.anim.setEndValue(float(i))
            self.anim.start()
        else:
            self._set_pos(float(i))
        self.changed.emit(i)

    def mousePressEvent(self, ev):
        w = self.width() / len(self.labels)
        self.set_current(min(len(self.labels) - 1, int(ev.position().x() // w)))

    def paintEvent(self, _ev):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        bg = QPainterPath()
        bg.addRoundedRect(r, r.height() / 2, r.height() / 2)
        g.fillPath(bg, QColor(C.get("surface_alt", "#F8F5FD")))
        w = r.width() / len(self.labels)
        pill = QRectF(r.left() + 3 + self._pos * w, r.top() + 3, w - 6, r.height() - 6)
        pp = QPainterPath()
        pp.addRoundedRect(pill, pill.height() / 2, pill.height() / 2)
        g.fillPath(pp, QColor(C.get("surface", "#FFFFFF")))
        g.setPen(QColor(C.get("border", "#ECE5F6")))
        g.drawPath(pp)
        f = QFont(self.font())
        f.setPixelSize(14)
        f.setWeight(QFont.DemiBold)
        g.setFont(f)
        for i, t in enumerate(self.labels):
            near = max(0.0, 1.0 - abs(self._pos - i))
            c1, c2 = QColor(C.get("muted", "#6E6888")), QColor(C.get("text", "#2A2340"))
            col = QColor(int(c1.red() + (c2.red() - c1.red()) * near), int(c1.green() + (c2.green() - c1.green()) * near),
                         int(c1.blue() + (c2.blue() - c1.blue()) * near))
            g.setPen(col)
            g.drawText(QRectF(r.left() + i * w, r.top(), w, r.height()), Qt.AlignCenter, t)
        g.end()


# ------------------------------------------------------------------ field
class Field(QWidget):
    """A labelled input; ``password=True`` adds a show / hide toggle. Its height is fixed (label +
    box), so a tight window can never make the box slide under the next label."""

    EDIT_H = 44

    def __init__(self, title: str, placeholder: str = "", password: bool = False, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(5)
        self.title = QLabel(title)
        self.title.setObjectName("FieldTitle")
        v.addWidget(self.title)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        # the theme's QSS min-height (22 px + padding) would otherwise win over a fixed height and the
        # layout could squeeze the box under its label: the field's own height is set here, in QSS too
        self.edit.setStyleSheet("QLineEdit { min-height: 28px; max-height: 28px; }"
                                "QLineEdit:focus { min-height: 28px; max-height: 28px; }")
        self.edit.setFixedHeight(self.EDIT_H)
        v.addWidget(self.edit)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.eye: Optional[QPushButton] = None
        if password:
            self.edit.setEchoMode(QLineEdit.Password)
            self.eye = QPushButton("Show", self.edit)
            self.eye.setObjectName("Ghost")
            self.eye.setCursor(Qt.PointingHandCursor)
            self.eye.setFocusPolicy(Qt.NoFocus)
            self.eye.setFixedSize(64, 30)
            self.eye.setStyleSheet("font-size:12px; font-weight:600; padding:0px;")
            self.eye.clicked.connect(self.toggle_echo)
            self.edit.setTextMargins(0, 0, 66, 0)

    def toggle_echo(self):
        hidden = self.edit.echoMode() == QLineEdit.Password
        self.edit.setEchoMode(QLineEdit.Normal if hidden else QLineEdit.Password)
        if self.eye is not None:
            self.eye.setText("Hide" if hidden else "Show")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.eye is not None:
            self.eye.move(self.edit.width() - self.eye.width() - 6, (self.edit.height() - self.eye.height()) // 2)

    def text(self) -> str:
        return self.edit.text()

    def setText(self, t: str) -> None:
        self.edit.setText(t)


# ------------------------------------------------------------------ the window
class LoginWindow(QDialog):
    """``mode``: "login", "signup" or "upgrade" (``profile_id`` = the profile to give an account)."""

    signed_in = Signal(str)

    def __init__(self, store: ProfileStore, mode: str = "login", profile_id: str = "", icon: QIcon | None = None):
        super().__init__()
        self.store = store
        self.mode = mode
        self.upgrade_id = profile_id
        self.result_id = ""
        self.want_profiles = False            # "Use a profile without an account" (the old chooser)
        self.setWindowTitle("FocusBuddy")
        if icon:
            self.setWindowIcon(icon)
        self.resize(980, 720)
        self.setMinimumSize(820, 560)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)    # FocusBuddy opens full screen
        self.setObjectName("LoginWindow")
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 18, 0)
        outer.setSpacing(0)
        self.brand = BrandPanel()
        outer.addWidget(self.brand, 5)
        # the form side scrolls when the window is shorter than the form (a laptop at 150 % scaling)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        holder = QWidget()
        holder.setAttribute(Qt.WA_TranslucentBackground)
        right = QVBoxLayout(holder)
        right.setContentsMargins(26, 30, 8, 26)
        right.addStretch(1)
        self.card = GlassCard(hover=False, strong=True, radius=24)
        self.card.setMinimumWidth(410)
        self.card.setMaximumWidth(460)
        self.card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        cl = QVBoxLayout(self.card)
        m = GlassCard.MARGIN
        cl.setContentsMargins(m + 28, m + 26, m + 28, m + 24)
        cl.setSpacing(12)
        self.title = _label("", "H1")
        self.title.setStyleSheet("font-size:26px;")
        self.sub = _label("", "Muted", wrap=True)
        cl.addWidget(self.title)
        cl.addWidget(self.sub)
        self.tabs = Tabs(["Log in", "Sign up"])
        self.tabs.changed.connect(self._tab)
        cl.addWidget(self.tabs)
        cl.addSpacing(2)
        self.f_name = Field("Name", "What should your buddy call you?")
        self.f_email = Field("E-mail", "you@example.com")
        self.f_pw = Field("Password", "Your password", password=True)
        self.f_pw2 = Field("Confirm password", "Type it again", password=True)
        for f in (self.f_name, self.f_email, self.f_pw, self.f_pw2):
            cl.addWidget(f)
            f.edit.returnPressed.connect(self._submit)
            f.edit.textEdited.connect(lambda _t: self._error(""))
        self.hint = _label("", "Faint", wrap=True)
        cl.addWidget(self.hint)
        row = QHBoxLayout()
        self.keep = QCheckBox("Keep me signed in")
        self.keep.setChecked(True)
        row.addWidget(self.keep)
        row.addStretch(1)
        self.forgot = QPushButton("Forgot password?")
        self.forgot.setObjectName("Ghost")
        self.forgot.setCursor(Qt.PointingHandCursor)
        self.forgot.clicked.connect(self._forgot)
        row.addWidget(self.forgot)
        cl.addLayout(row)
        self.err = _label("", "Error", wrap=True)
        self.err.setVisible(False)
        cl.addWidget(self.err)
        self.go = QPushButton("Log in")
        self.go.setObjectName("Primary")
        self.go.setMinimumHeight(46)
        self.go.setCursor(Qt.PointingHandCursor)
        self.go.clicked.connect(self._submit)
        cl.addWidget(self.go)
        self.alt = QPushButton("")
        self.alt.setObjectName("Ghost")
        self.alt.setCursor(Qt.PointingHandCursor)
        self.alt.clicked.connect(self._alt)
        cl.addWidget(self.alt, 0, Qt.AlignHCenter)
        right.addWidget(self.card, 0, Qt.AlignHCenter)
        right.addStretch(1)
        self.profiles_link = QPushButton("Use a profile without an account")
        self.profiles_link.setObjectName("Ghost")
        self.profiles_link.setCursor(Qt.PointingHandCursor)
        self.profiles_link.clicked.connect(self._profiles)
        self.profiles_link.setVisible(any(not p.has_account for p in store.profiles.values()) and mode != "upgrade")
        right.addWidget(self.profiles_link, 0, Qt.AlignHCenter)
        self.scroll.setWidget(holder)
        outer.addWidget(self.scroll, 6)
        self._busy_timer = QTimer(self)
        self._busy_timer.timeout.connect(self._busy_tick)
        self._busy_n = 0
        self._apply_mode(animate=False)

    # ------------------------------------------------------------ chrome
    def showEvent(self, ev):
        super().showEvent(ev)
        fluent_window(self)
        fade_in(self)
        if full_motion():
            # the card rises into place
            eff = QGraphicsOpacityEffect(self.card)
            self.card.setGraphicsEffect(eff)
            a = QPropertyAnimation(eff, b"opacity", self)
            a.setDuration(380)
            a.setStartValue(0.0)
            a.setEndValue(1.0)
            a.setEasingCurve(QEasingCurve.OutCubic)
            a.finished.connect(lambda: self.card.setGraphicsEffect(None))
            a.start()
            self._card_anim = a
        QTimer.singleShot(60, self._focus_first)

    def paintEvent(self, _ev):
        g = QPainter(self)
        g.fillRect(self.rect(), QColor(C.get("bg", "#F6F1FB")))
        g.end()

    def _fit_card(self):
        """The card's height from its visible contents at its real width. (Qt's height-for-width of a
        card with wrapped, style-sheet-sized labels comes out short, and the layout then squeezed the
        inputs under their labels.)"""
        lay = self.card.layout()
        m = lay.contentsMargins()
        inner = max(1, self.card.width() - m.left() - m.right())
        need = m.top() + m.bottom()
        n = 0
        for i in range(lay.count()):
            it = lay.itemAt(i)
            w = it.widget()
            if w is not None:
                if not w.isVisible() and not (w.isVisibleTo(self.card)):
                    continue
                h = w.heightForWidth(inner) if w.hasHeightForWidth() else w.sizeHint().height()
                need += max(h, w.minimumSizeHint().height(), w.minimumHeight())
                n += 1
            elif it.layout() is not None:
                need += it.layout().sizeHint().height()
                n += 1
            else:
                need += it.sizeHint().height()
        need += lay.spacing() * max(0, n - 1)
        self.card.setMinimumHeight(need)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        QTimer.singleShot(0, self._fit_card)

    def _focus_first(self):
        for f in (self.f_name, self.f_email, self.f_pw):
            if f.isVisible() and not f.text():
                f.edit.setFocus()
                return
        self.f_pw.edit.setFocus()

    # ------------------------------------------------------------ modes
    def _apply_mode(self, animate: bool = True):
        m = self.mode
        signup = m in ("signup", "upgrade")
        self.tabs.setVisible(m != "upgrade")
        self.tabs.set_current(1 if signup else 0, animate)
        if m == "login":
            self.title.setText("Welcome back")
            self.sub.setText("Log in to pick up where you left off.")
            self.go.setText("Log in")
            self.alt.setText("New here? Create an account")
        elif m == "signup":
            self.title.setText("Create your account")
            self.sub.setText("It takes a few seconds. Your buddy is waiting.")
            self.go.setText("Create account")
            self.alt.setText("Already have an account? Log in")
        else:
            p = self.store.get(self.upgrade_id)
            self.title.setText(f"Hi {p.name}!" if p else "Finish setting up")
            self.sub.setText("FocusBuddy now has accounts. Add an e-mail and a password to your profile - "
                             "your history, rules and companion stay just as they are.")
            self.go.setText("Save and continue")
            self.alt.setText("")
            if p is not None and not self.f_name.text():
                self.f_name.setText(p.name)
        self.alt.setVisible(m != "upgrade")
        self.f_name.setVisible(signup)
        self.f_pw2.setVisible(signup)
        self.forgot.setVisible(m == "login")
        self.f_pw.edit.setPlaceholderText("Your password" if m == "login" else "At least 8 characters")
        self.hint.setText("" if m == "login" else "Use 8 or more characters with a mix of letters and numbers.")
        self.hint.setVisible(m != "login")
        self._error("")
        QTimer.singleShot(0, self._focus_first)
        QTimer.singleShot(0, self._fit_card)

    def _tab(self, i: int):
        want = "signup" if i == 1 else "login"
        if self.mode != "upgrade" and want != self.mode:
            self.mode = want
            self._apply_mode()

    def _alt(self):
        self.mode = "login" if self.mode == "signup" else "signup"
        self._apply_mode()

    # ------------------------------------------------------------ actions
    def _error(self, text: str):
        self.err.setText(text)
        self.err.setVisible(bool(text))
        QTimer.singleShot(0, self._fit_card)

    def _fail(self, text: str):
        self._error(text)
        shake(self.card)

    def _busy(self, on: bool):
        self.go.setEnabled(not on)
        if on:
            self._busy_label = self.go.text()
            self._busy_n = 0
            self._busy_timer.start(160)
            self._busy_tick()
        else:
            self._busy_timer.stop()
            self.go.setText(getattr(self, "_busy_label", self.go.text()))

    def _busy_tick(self):
        self._busy_n = (self._busy_n + 1) % 4
        self.go.setText("One moment" + "." * self._busy_n)

    def _submit(self):
        if not self.go.isEnabled():
            return
        email, pw = self.f_email.text().strip(), self.f_pw.text()
        if not ProfileStore.valid_email(email):
            self._fail("Please enter a valid e-mail address.")
            self.f_email.edit.setFocus()
            return
        if self.mode == "login":
            p = self.store.by_email(email)
            if p is not None and self.store.wait_seconds("pw:" + p.id) > 0:
                self._fail(f"Too many tries. Wait {int(self.store.wait_seconds('pw:' + p.id)) + 1} s.")
                return
            self._busy(True)
            QTimer.singleShot(10, lambda: self._do_login(email, pw))
            return
        problem = ProfileStore.password_problem(pw)
        if problem:
            self._fail(problem)
            self.f_pw.edit.setFocus()
            return
        if pw != self.f_pw2.text():
            self._fail("The passwords don't match.")
            self.f_pw2.edit.setFocus()
            return
        self._busy(True)
        QTimer.singleShot(10, lambda: self._do_signup(email, pw))

    def _do_login(self, email: str, pw: str):
        p = self.store.log_in(email, pw)
        self._busy(False)
        if p is None:
            self.f_pw.setText("")
            self._fail("That e-mail and password don't match an account on this device.")
            return
        self._finish(p)

    def _do_signup(self, email: str, pw: str):
        name = self.f_name.text().strip()
        try:
            if self.mode == "upgrade":
                if name:
                    self.store.update(self.upgrade_id, name=name[:40])
                p = self.store.set_account(self.upgrade_id, email, pw)
            else:
                p = self.store.sign_up(name or email.split("@")[0].title(), email, pw)
        except ValueError as e:
            self._busy(False)
            msg = str(e)
            self._fail(msg[:1].upper() + msg[1:] + ("" if msg.endswith(".") else "."))
            return
        self._busy(False)
        self._finish(p)

    def _forgot(self):
        """No e-mail reset (nothing leaves this device). A profile with a PIN can set a new password
        after its PIN is entered; otherwise the screen explains the way out."""
        self._error("")
        p = self.store.by_email(self.f_email.text())
        if p is not None and p.has_pin:
            from .signin import SignInWindow
            w = SignInWindow(self.store, "unlock", p.id, allow_switch=False)
            w.exec()
            if w.result_id == p.id:
                self.upgrade_id = p.id
                self.mode = "upgrade"
                self._apply_mode()
                self.title.setText("Choose a new password")
                self.sub.setText("Your PIN checked out. Pick a new password for this account.")
                self.f_email.setText(p.email)
            return
        self.hint.setVisible(True)
        self.hint.setText("Your account is stored only on this device, so there is no e-mail reset. "
                          "If the profile has a PIN, type its e-mail above and click “Forgot password?” "
                          "again to unlock it with the PIN and choose a new password.")

    def _profiles(self):
        self.want_profiles = True
        self.accept()

    def _finish(self, p: Profile):
        self.result_id = p.id
        self.store.touch(p.id)
        if self.keep.isChecked():
            self.store.remember(p.id)
        else:
            self.store.forget_session()
        self.signed_in.emit(p.id)
        self.accept()
