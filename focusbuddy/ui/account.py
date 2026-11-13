"""Settings -> Account: the local profile, its avatar, PIN, lock options, switching and deleting."""

from __future__ import annotations

from PySide6.QtCore import QRegularExpression
from PySide6.QtGui import QIcon, QRegularExpressionValidator
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu, QMessageBox,
                               QPushButton, QVBoxLayout)

from ..core.profiles import AVATARS, ProfileStore
from .icons import avatar_pixmap
from .motion import shake
from .widgets import Card, Toggle, label


class PinDialog(QDialog):
    """Set / change a PIN (asks for the current one first when there is one)."""

    def __init__(self, store: ProfileStore, pid: str, parent=None):
        super().__init__(parent)
        self.store, self.pid = store, pid
        p = store.get(pid)
        self.setWindowTitle("Change PIN" if p.has_pin else "Set up a PIN")
        self.setMinimumWidth(380)
        v = QVBoxLayout(self)
        v.setContentsMargins(24, 22, 24, 20)
        v.setSpacing(10)
        t = label("Change your PIN" if p.has_pin else "Protect your profile with a PIN", "H2")
        t.setWordWrap(True)
        v.addWidget(t)
        v.addWidget(label("4 to 8 digits. It locks the FocusBuddy window on this PC.", "Muted", wrap=True))
        self.cur = self._field("Current PIN") if p.has_pin else None
        self.new1 = self._field("New PIN")
        self.new2 = self._field("Repeat new PIN")
        for e in (self.cur, self.new1, self.new2):
            if e is not None:
                v.addWidget(e)
        self.msg = label("", "Faint")
        v.addWidget(self.msg)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("Save PIN")
        ok.setObjectName("Primary")
        ok.clicked.connect(self._save)
        row.addWidget(cancel)
        row.addWidget(ok)
        v.addLayout(row)

    @staticmethod
    def _field(ph: str) -> QLineEdit:
        e = QLineEdit()
        e.setEchoMode(QLineEdit.Password)
        e.setMaxLength(8)
        e.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d{0,8}")))
        e.setPlaceholderText(ph)
        return e

    def _save(self):
        if self.cur is not None and not self.store.verify_pin(self.pid, self.cur.text()):
            self.msg.setText("The current PIN isn't right.")
            shake(self)
            return
        if not ProfileStore.valid_pin(self.new1.text()):
            self.msg.setText("The PIN needs 4 to 8 digits.")
            shake(self)
            return
        if self.new1.text() != self.new2.text():
            self.msg.setText("The two PINs don't match.")
            shake(self)
            return
        self.store.set_pin(self.pid, self.new1.text())
        self.accept()


def ask_pin(store: ProfileStore, pid: str, parent, why: str) -> bool:
    """Confirm with the current PIN (True when the profile has none)."""
    p = store.get(pid)
    if p is None or not p.has_pin:
        return True
    text, ok = QInputDialog.getText(parent, "Confirm with your PIN", why, QLineEdit.Password)
    if not ok:
        return False
    if store.verify_pin(pid, text):
        return True
    QMessageBox.warning(parent, "FocusBuddy", "That PIN isn't right.")
    return False


class AccountCard(Card):
    LOCK_OPTIONS = [(0, "Never"), (5, "After 5 minutes away"), (15, "After 15 minutes away"),
                    (30, "After 30 minutes away"), (60, "After 1 hour away")]

    def __init__(self, page, app):
        super().__init__("Account", "Your local profile on this PC. Each profile has its own history, rules "
                                    "and companion.")
        self.page, self.app = page, app
        self.store: ProfileStore | None = getattr(app, "profiles", None)
        if self.store is None or getattr(app, "profile", None) is None:
            self.add(label("Profiles are not available in this mode (demo / tests).", "Muted"))
            return
        top = QHBoxLayout()
        top.setSpacing(14)
        self.pic = QLabel()
        top.addWidget(self.pic)
        col = QVBoxLayout()
        col.setSpacing(2)
        self.name = label("", "H2")
        self.name.setProperty("search_title", "Profile name")
        col.addWidget(self.name)
        self.sub = label("", "Muted")
        col.addWidget(self.sub)
        top.addLayout(col, 1)
        ren = QPushButton("Rename")
        ren.clicked.connect(self._rename)
        av = QPushButton("Change avatar")
        av.clicked.connect(lambda: self._avatar_menu(av))
        top.addWidget(ren)
        top.addWidget(av)
        self.add(top)

        rows = QVBoxLayout()
        rows.setSpacing(12)
        # PIN
        pin_row = QHBoxLayout()
        pcol = QVBoxLayout()
        pcol.setSpacing(1)
        t = label("PIN")
        t.setStyleSheet("font-size:13px; font-weight:600;")
        t.setProperty("search_title", "PIN")
        pcol.addWidget(t)
        self.pin_state = label("", "Faint")
        pcol.addWidget(self.pin_state)
        pin_row.addLayout(pcol, 1)
        self.pin_btn = QPushButton()
        self.pin_btn.setObjectName("Primary")
        self.pin_btn.clicked.connect(self._pin)
        self.pin_rm = QPushButton("Remove PIN")
        self.pin_rm.clicked.connect(self._remove_pin)
        pin_row.addWidget(self.pin_rm)
        pin_row.addWidget(self.pin_btn)
        rows.addLayout(pin_row)
        # ask on start
        r2 = QHBoxLayout()
        c2 = QVBoxLayout()
        c2.setSpacing(1)
        t2 = label("Ask for my PIN when FocusBuddy starts")
        t2.setStyleSheet("font-size:13px; font-weight:600;")
        t2.setProperty("search_title", "Ask for PIN at start")
        c2.addWidget(t2)
        c2.addWidget(label("Your companion still starts; the window stays locked until you sign in.", "Faint"))
        r2.addLayout(c2, 1)
        self.ask = Toggle(self.app.profile.require_pin_on_start)
        self.ask.toggled.connect(lambda on: self.store.update(self.app.profile.id, require_pin_on_start=bool(on)))
        r2.addWidget(self.ask)
        rows.addLayout(r2)
        # auto lock
        r3 = QHBoxLayout()
        t3 = label("Lock automatically")
        t3.setStyleSheet("font-size:13px; font-weight:600;")
        t3.setProperty("search_title", "Lock automatically")
        r3.addWidget(t3, 1)
        self.lock_combo = QComboBox()
        self.lock_combo.setMinimumWidth(220)
        for v, txt in self.LOCK_OPTIONS:
            self.lock_combo.addItem(txt, v)
        self.lock_combo.setCurrentIndex(max(0, self.lock_combo.findData(self.app.profile.lock_after_min)))
        self.lock_combo.currentIndexChanged.connect(
            lambda _i: self.store.update(self.app.profile.id, lock_after_min=int(self.lock_combo.currentData())))
        r3.addWidget(self.lock_combo)
        rows.addLayout(r3)
        self.add(rows)
        # actions
        act = QHBoxLayout()
        out = QPushButton("Sign out")
        out.setProperty("search_title", "Sign out")
        out.clicked.connect(lambda: self.app.switch_profile())
        act.addWidget(out)
        sw = QPushButton("Switch profile…")
        sw.clicked.connect(lambda: self.app.switch_profile())
        act.addWidget(sw)
        lock = QPushButton("Lock now")
        lock.clicked.connect(lambda: self.app.lock())
        self.lock_now = lock
        act.addWidget(lock)
        act.addStretch(1)
        self.del_btn = QPushButton("Delete this profile…")
        self.del_btn.setObjectName("Danger")
        self.del_btn.clicked.connect(self._delete)
        act.addWidget(self.del_btn)
        self.add(act)
        self.refresh()

    def refresh(self):
        if self.store is None or getattr(self.app, "profile", None) is None:
            return
        p = self.store.get(self.app.profile.id) or self.app.profile
        self.app.profile = p
        self.pic.setPixmap(avatar_pixmap(p.avatar, p.name, 64))
        self.name.setText(p.name)
        import datetime as dt
        who = p.email if getattr(p, "has_account", False) else "Local profile"
        self.sub.setText(f"{who} · since {dt.date.fromtimestamp(p.created).strftime('%d %b %Y')}"
                         f" · {len(self.store.profiles)} profile{'s' if len(self.store.profiles) != 1 else ''} on this PC")
        self.pin_state.setText("Your profile is protected with a PIN." if p.has_pin else
                               "No PIN - anyone using this PC can open your FocusBuddy.")
        self.pin_btn.setText("Change PIN" if p.has_pin else "Set up PIN")
        self.pin_rm.setVisible(p.has_pin)
        self.ask.setEnabled(p.has_pin)
        self.lock_combo.setEnabled(p.has_pin)
        self.lock_now.setEnabled(p.has_pin)
        self.del_btn.setEnabled(len(self.store.profiles) > 1)
        dash = getattr(self.app, "dashboard", None)
        if dash is not None and hasattr(dash, "refresh_profile"):
            dash.refresh_profile()

    def _toast(self, text: str):
        dash = getattr(self.app, "dashboard", None)
        if dash is not None and hasattr(dash, "toast"):
            dash.toast(text, "ok")

    def _rename(self):
        p = self.app.profile
        text, ok = QInputDialog.getText(self, "Rename profile", "Name:", QLineEdit.Normal, p.name)
        if ok and text.strip():
            self.store.update(p.id, name=text.strip()[:40])
            self.app.settings.set("user_name", text.strip()[:40])
            self.refresh()
            self._toast("Profile renamed")

    def _avatar_menu(self, anchor):
        m = QMenu(self)
        for a in list(AVATARS) + ["initials"]:
            act = m.addAction(QIcon(avatar_pixmap(a, self.app.profile.name, 28, ring=False)),
                              "Initials" if a == "initials" else a.replace("_", " ").title())
            act.triggered.connect(lambda _=False, av=a: (self.store.update(self.app.profile.id, avatar=av),
                                                         self.refresh(), self._toast("Avatar updated")))
        m.setStyleSheet("QMenu::icon { padding-left: 6px; }")
        m.exec(anchor.mapToGlobal(anchor.rect().bottomLeft()))

    def _pin(self):
        if PinDialog(self.store, self.app.profile.id, self).exec():
            self.refresh()
            self._toast("PIN saved")

    def _remove_pin(self):
        if ask_pin(self.store, self.app.profile.id, self, "Enter your current PIN to remove it:"):
            self.store.clear_pin(self.app.profile.id)
            self.refresh()
            self._toast("PIN removed")

    def _delete(self):
        p = self.app.profile
        if len(self.store.profiles) <= 1:
            return
        r = QMessageBox.warning(self, "Delete profile",
                                f"Delete the profile \"{p.name}\" and all of its focus history, rules and settings "
                                "on this PC?\n\nThis can't be undone.",
                                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if r != QMessageBox.Yes or not ask_pin(self.store, p.id, self, "Enter your PIN to delete this profile:"):
            return
        self.app.delete_current_profile()

