"""Sign-in, lock and profile creation (local profiles, optional PIN)."""

from __future__ import annotations

import datetime as dt
from typing import Optional

from PySide6.QtCore import QRegularExpression, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QVBoxLayout, QWidget)

from ..core.profiles import AVATARS, Profile, ProfileStore
from .icons import avatar_pixmap
from .motion import AnimatedStack, AuroraBackground, GlassCard, fade_in, shake
from .theme import C, fluent_window, themed


def _label(text: str = "", obj: str | None = None, center: bool = True, wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if obj:
        lb.setObjectName(obj)
    if center:
        lb.setAlignment(Qt.AlignHCenter)
    lb.setWordWrap(wrap)
    return lb


class ProfileTile(QPushButton):
    def __init__(self, profile: Optional[Profile], size: int = 84):
        super().__init__()
        self.profile = profile
        self.setObjectName("ChoiceCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(132, 150)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 12, 8, 8)
        v.setSpacing(6)
        pic = QLabel()
        pic.setAlignment(Qt.AlignCenter)
        pic.setAttribute(Qt.WA_TransparentForMouseEvents)
        if profile is not None:
            pic.setPixmap(avatar_pixmap(profile.avatar, profile.name, size))
            name = profile.name + ("  🔒" if profile.has_pin else "")
        else:
            pic.setText("+")
            themed(pic, lambda: f"font-size:40px; font-weight:300; color:{C['primary']};"
                               f"border:2px dashed {C['border_strong']}; border-radius:{size // 2}px;")
            pic.setFixedSize(size, size)
            name = "Add profile"
        v.addWidget(pic, 0, Qt.AlignHCenter)
        n = QLabel(name)
        n.setAlignment(Qt.AlignHCenter)
        n.setAttribute(Qt.WA_TransparentForMouseEvents)
        n.setStyleSheet("font-weight:650; font-size:13px;")
        v.addWidget(n)


class SignInWindow(QDialog):
    """``mode``: "choose" (pick / create a profile), "unlock" (PIN for one profile) or "create"."""

    signed_in = Signal(str)

    def __init__(self, store: ProfileStore, mode: str = "choose", profile_id: str = "", icon: QIcon | None = None,
                 allow_switch: bool = True):
        super().__init__()
        self.store = store
        self.mode = mode
        self.allow_switch = allow_switch
        self.result_id: str = ""
        self.switch_requested = False
        self.setWindowTitle("FocusBuddy - Sign in" if mode != "unlock" else "FocusBuddy - Locked")
        if icon:
            self.setWindowIcon(icon)
        self.resize(820, 640)
        self.setMinimumSize(640, 560)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.bg = AuroraBackground()
        outer.addWidget(self.bg)
        lay = QVBoxLayout(self.bg)
        lay.setContentsMargins(30, 24, 30, 30)
        # clock (lock-screen feel)
        top = QVBoxLayout()
        top.setSpacing(0)
        self.clock = _label("", "Big")
        self.date = _label("", "Muted")
        top.addWidget(self.clock)
        top.addWidget(self.date)
        lay.addLayout(top)
        self._tick()
        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(1000)
        lay.addStretch(1)
        self.card = GlassCard(hover=False, strong=True, radius=28)
        self.card.setFixedWidth(560)
        cl = QVBoxLayout(self.card)
        m = GlassCard.MARGIN
        cl.setContentsMargins(m + 26, m + 22, m + 26, m + 24)
        self.stack = AnimatedStack()
        cl.addWidget(self.stack)
        lay.addWidget(self.card, 0, Qt.AlignHCenter)
        lay.addStretch(2)
        foot = _label("Profiles and your activity data stay on this PC.", "Faint")
        lay.addWidget(foot)
        self.page_choose = self._build_choose()
        self.page_pin = self._build_pin()
        self.page_create = self._build_create()
        for p in (self.page_choose, self.page_pin, self.page_create):
            self.stack.addWidget(p)
        if mode == "unlock" and store.get(profile_id):
            self._open_pin(store.get(profile_id))
        elif mode == "create" or not store.profiles:
            self._open_create(first=not store.profiles)
        else:
            self.stack.setCurrentIndex(0)

    # ------------------------------------------------------------ chrome
    def showEvent(self, ev):
        super().showEvent(ev)
        fluent_window(self)
        fade_in(self)

    def _tick(self):
        now = dt.datetime.now()
        self.clock.setText(now.strftime("%H:%M"))
        self.date.setText(now.strftime("%A, %d %B"))

    # ------------------------------------------------------------ choose
    def _build_choose(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Transparent")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        v.addWidget(_label("FOCUSBUDDY", "Eyebrow"))
        v.addWidget(_label("Who's focusing today?", "H1"))
        v.addWidget(_label("Pick your profile to continue.", "Muted"))
        v.addSpacing(14)
        self.tiles = QGridLayout()
        self.tiles.setSpacing(12)
        v.addLayout(self.tiles)
        v.addStretch(1)
        self._fill_tiles()
        return w

    def _fill_tiles(self):
        while self.tiles.count():
            it = self.tiles.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        items = list(self.store.list()) + [None]
        for i, p in enumerate(items):
            t = ProfileTile(p)
            t.clicked.connect(lambda _=False, pr=p: self._pick(pr))
            self.tiles.addWidget(t, i // 3, i % 3, Qt.AlignHCenter)

    def _pick(self, profile: Optional[Profile]):
        if profile is None:
            self._open_create(first=False)
        elif profile.has_pin:
            self._open_pin(profile)
        else:
            self._finish(profile.id)

    # ------------------------------------------------------------ PIN
    def _build_pin(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Transparent")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        self.pin_pic = QLabel()
        self.pin_pic.setAlignment(Qt.AlignCenter)
        v.addWidget(self.pin_pic)
        self.pin_name = _label("", "H2")
        v.addWidget(self.pin_name)
        v.addWidget(_label("Enter your PIN", "Muted"))
        v.addSpacing(6)
        self.pin = QLineEdit()
        self.pin.setObjectName("Pin")
        self.pin.setEchoMode(QLineEdit.Password)
        self.pin.setAlignment(Qt.AlignCenter)
        self.pin.setMaxLength(8)
        self.pin.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d{0,8}")))
        self.pin.setFixedWidth(260)
        self.pin.returnPressed.connect(self._try_pin)
        v.addWidget(self.pin, 0, Qt.AlignHCenter)
        self.pin_msg = _label("", "Faint")
        v.addWidget(self.pin_msg)
        row = QHBoxLayout()
        self.pin_back = QPushButton("Other profile")
        self.pin_back.setObjectName("Ghost")
        self.pin_back.clicked.connect(lambda: self._show(0))
        ok = QPushButton("Sign in")
        ok.setObjectName("Primary")
        ok.setMinimumWidth(140)
        ok.clicked.connect(self._try_pin)
        row.addWidget(self.pin_back)
        row.addStretch(1)
        row.addWidget(ok)
        v.addSpacing(8)
        v.addLayout(row)
        return w

    def _open_pin(self, profile: Profile):
        self._pin_profile = profile
        self.pin_pic.setPixmap(avatar_pixmap(profile.avatar, profile.name, 96))
        self.pin_name.setText(profile.name)
        self.pin.clear()
        self.pin_msg.setText("")
        self.pin_back.setVisible(self.allow_switch and (self.mode != "unlock" or len(self.store.profiles) > 1))
        self.pin_back.setText("Other profile" if self.mode != "unlock" else "Switch profile")
        self._show(1)
        QTimer.singleShot(50, self.pin.setFocus)

    def _try_pin(self):
        p = self._pin_profile
        wait = self.store.wait_seconds(p.id)
        if wait > 0:
            self.pin_msg.setText(f"Too many tries. Wait {int(wait) + 1} s.")
            shake(self.card)
            return
        if self.store.verify_pin(p.id, self.pin.text()):
            self._finish(p.id)
        else:
            wait = self.store.wait_seconds(p.id)
            self.pin_msg.setText(f"That PIN isn't right. Wait {int(wait) + 1} s." if wait else
                                 "That PIN isn't right. Try again.")
            self.pin.clear()
            shake(self.card)

    # ------------------------------------------------------------ create
    def _build_create(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Transparent")
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        v.addWidget(_label("WELCOME", "Eyebrow"))
        self.create_title = _label("Create your profile", "H1")
        v.addWidget(self.create_title)
        v.addWidget(_label("Your profile keeps your focus history, rules and companion separate.", "Muted",
                           wrap=True))
        v.addSpacing(8)
        self.c_name = QLineEdit()
        self.c_name.setPlaceholderText("Your name")
        self.c_name.setMaxLength(40)
        v.addWidget(self.c_name)
        v.addWidget(_label("Pick an avatar", "Faint", center=False))
        grid = QHBoxLayout()
        grid.setSpacing(6)
        self.c_group = QButtonGroup(self)
        for i, a in enumerate(list(AVATARS) + ["initials"]):
            b = QPushButton()
            b.setObjectName("Avatar")
            b.setCheckable(True)
            b.setFixedSize(46, 46)
            b.setIconSize(QSize(42, 42))
            b.setIcon(QIcon(avatar_pixmap(a, "Me", 42, ring=False)))
            b.setProperty("avatar", a)
            b.setCursor(Qt.PointingHandCursor)
            self.c_group.addButton(b, i)
            grid.addWidget(b)
            if i == 0:
                b.setChecked(True)
        grid.addStretch(1)
        v.addLayout(grid)
        self.c_group.buttonToggled.connect(self._avatar_ring)
        self._avatar_ring()
        self.c_lock = QCheckBox("Protect this profile with a PIN")
        v.addWidget(self.c_lock)
        pins = QHBoxLayout()
        self.c_pin1, self.c_pin2 = QLineEdit(), QLineEdit()
        for e, ph in ((self.c_pin1, "PIN (4-8 digits)"), (self.c_pin2, "Repeat PIN")):
            e.setEchoMode(QLineEdit.Password)
            e.setMaxLength(8)
            e.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d{0,8}")))
            e.setPlaceholderText(ph)
            e.setEnabled(False)
            pins.addWidget(e)
        self.c_lock.toggled.connect(lambda on: (self.c_pin1.setEnabled(on), self.c_pin2.setEnabled(on)))
        v.addLayout(pins)
        self.c_msg = _label("", "Faint", center=False)
        v.addWidget(self.c_msg)
        row = QHBoxLayout()
        self.c_back = QPushButton("Back")
        self.c_back.setObjectName("Ghost")
        self.c_back.clicked.connect(lambda: self._show(0))
        ok = QPushButton("Create profile")
        ok.setObjectName("Primary")
        ok.setMinimumWidth(160)
        ok.clicked.connect(self._create)
        self.c_name.returnPressed.connect(self._create)
        row.addWidget(self.c_back)
        row.addStretch(1)
        row.addWidget(ok)
        v.addLayout(row)
        return w

    def _avatar_ring(self, *_a):
        for b in self.c_group.buttons():
            col = C["primary"] if b.isChecked() else "transparent"
            b.setStyleSheet(f"QPushButton#Avatar {{ border: 3px solid {col}; border-radius: 23px; }}")

    def _open_create(self, first: bool):
        self.create_title.setText("Create your profile" if first else "Add a profile")
        self.c_back.setVisible(not first)
        self._show(2)
        QTimer.singleShot(50, self.c_name.setFocus)

    def _create(self):
        name = self.c_name.text().strip()
        if not name:
            self.c_msg.setText("Please enter a name.")
            shake(self.card)
            return
        pin = ""
        if self.c_lock.isChecked():
            pin = self.c_pin1.text()
            if not ProfileStore.valid_pin(pin):
                self.c_msg.setText("The PIN needs 4 to 8 digits.")
                shake(self.card)
                return
            if pin != self.c_pin2.text():
                self.c_msg.setText("The two PINs don't match.")
                shake(self.card)
                return
        b = self.c_group.checkedButton()
        avatar = b.property("avatar") if b is not None else "cat"
        p = self.store.create(name, avatar, pin)
        self._finish(p.id)

    # ------------------------------------------------------------ flow
    def _show(self, idx: int):
        if idx == 0:
            if self.mode == "unlock" and self.allow_switch:
                self.switch_requested = True       # the app restarts into the profile chooser
                self.accept()
                return
            self._fill_tiles()
        self.stack.setCurrentIndex(idx)

    def _finish(self, pid: str):
        self.result_id = pid
        self.store.touch(pid)
        self.signed_in.emit(pid)
        self.accept()
