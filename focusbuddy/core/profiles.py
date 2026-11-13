"""Local user profiles with an optional PIN.

Profiles live in ``<data dir>/profiles.json``; each profile keeps its own database, logs and
characters in its own folder (``<data dir>/profiles/<id>/``). An install from before profiles
existed becomes the ``default`` profile, which keeps using the data folder itself, so nothing moves.

A PIN (4-8 digits) is stored only as a salted PBKDF2-SHA256 hash. It locks the FocusBuddy window
on a shared PC; it is not encryption - the database files themselves are not encrypted (the
README says so). Wrong PINs back off: after 5 failures a profile waits 30 s, doubling each time.

Accounts: a profile can carry an e-mail and a password, like a website or app account, so
FocusBuddy opens with a familiar log-in screen. Everything stays on this device: the password is
stored only as a salted PBKDF2-SHA256 hash, nothing is sent anywhere. "Keep me signed in" stores
a remembered session (the profile id and a random token whose hash is kept in ``profiles.json``)
so the log-in appears only once; signing out forgets it. Wrong passwords back off like PINs.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

PBKDF2_ITERATIONS = 200_000
PIN_RE = re.compile(r"^\d{4,8}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")
MIN_PASSWORD = 8
AVATARS = ("cat", "dog", "fox", "rabbit", "robot", "buddy", "study_girl", "saree_girl")


@dataclass
class Profile:
    id: str
    name: str
    avatar: str = "cat"                  # a companion's portrait, or "initials"
    pin_hash: str = ""
    pin_salt: str = ""
    require_pin_on_start: bool = False    # a PIN locks the window only when the user asks for it
    pin_prompt_chosen: bool = False      # the user set "ask for my PIN" themselves (kept as they set it)
    lock_after_min: int = 0              # lock the window after this long away (0 = never)
    created: float = field(default_factory=time.time)
    last_used: float = 0.0
    email: str = ""                      # the account's log-in (empty: a profile without an account)
    pw_hash: str = ""
    pw_salt: str = ""

    @property
    def has_account(self) -> bool:
        return bool(self.email and self.pw_hash)

    @property
    def has_pin(self) -> bool:
        return bool(self.pin_hash)

    @property
    def initials(self) -> str:
        parts = [p for p in re.split(r"\s+", self.name.strip()) if p]
        return ("".join(p[0] for p in parts[:2]) or "?").upper()


def _hash(pin: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, PBKDF2_ITERATIONS).hex()


class ProfileStore:
    def __init__(self, base: Path):
        self.base = Path(base)
        self.path = self.base / "profiles.json"
        self.profiles: dict[str, Profile] = {}
        self.last: str = ""
        self.session: dict = {}                            # {"profile": id, "token_hash": hex, "since": t}
        self._session_token: str = ""                      # the plain token of a session made in this run
        self._fails: dict[str, tuple[int, float]] = {}    # id -> (failures, locked until)
        self.load()

    # ------------------------------------------------------------ persistence
    def load(self) -> None:
        self.profiles = {}
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                for d in raw.get("profiles", []):
                    known = {k: v for k, v in d.items() if k in Profile.__dataclass_fields__}
                    p = Profile(**known)
                    if not p.pin_prompt_chosen:
                        # profiles from versions that asked for the PIN by default: off unless chosen
                        p.require_pin_on_start = False
                    self.profiles[p.id] = p
                self.last = str(raw.get("last", ""))
                sess = raw.get("session") or {}
                self.session = sess if isinstance(sess, dict) else {}
                self._load_token()
            except (OSError, ValueError, TypeError):
                self.profiles = {}
        elif (self.base / "focusbuddy.db").is_file():
            # an install from before profiles: it becomes the "default" profile, data stays in place
            name = self._legacy_name() or os.environ.get("USERNAME") or os.environ.get("USER") or "Me"
            self.profiles["default"] = Profile("default", name.strip().title()[:40] or "Me", avatar="cat",
                                               last_used=time.time())
            self.last = "default"
            self.save()

    def _legacy_name(self) -> str:
        try:
            import sqlite3
            con = sqlite3.connect(str(self.base / "focusbuddy.db"))
            row = con.execute("SELECT value FROM settings WHERE key='user_name'").fetchone()
            con.close()
            return json.loads(row[0]) if row and row[0] else ""
        except Exception:  # noqa: BLE001
            return ""

    def save(self) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        data = {"version": 2, "last": self.last, "session": self.session,
                "profiles": [asdict(p) for p in self.profiles.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    # ------------------------------------------------------------ queries
    def list(self) -> list[Profile]:
        return sorted(self.profiles.values(), key=lambda p: (-p.last_used, p.created))

    def get(self, pid: str) -> Optional[Profile]:
        return self.profiles.get(pid)

    def last_profile(self) -> Optional[Profile]:
        return self.profiles.get(self.last) or (self.list()[0] if self.profiles else None)

    def data_dir(self, profile: Profile) -> Path:
        d = self.base if profile.id == "default" else self.base / "profiles" / profile.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------ changes
    def create(self, name: str, avatar: str = "cat", pin: str = "") -> Profile:
        name = (name or "").strip()[:40] or "Me"
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "user"
        pid, n = base, 2
        while pid in self.profiles or pid == "default" and self.profiles:
            pid = f"{base}-{n}"
            n += 1
        if not self.profiles and not (self.base / "focusbuddy.db").exists():
            pid = "default"                        # the first profile uses the data folder itself
        p = Profile(pid, name, avatar=avatar if avatar in AVATARS or avatar == "initials" else "cat",
                    last_used=time.time())
        self.profiles[pid] = p
        if pin:
            self.set_pin(pid, pin, save=False)
        self.last = pid
        self.save()
        self.data_dir(p)
        return p

    def update(self, pid: str, **changes) -> Profile:
        p = self.profiles[pid]
        for k, v in changes.items():
            if k in ("name", "avatar", "require_pin_on_start", "lock_after_min"):
                setattr(p, k, v)
                if k == "require_pin_on_start":
                    p.pin_prompt_chosen = True
        self.save()
        return p

    def touch(self, pid: str) -> None:
        p = self.profiles.get(pid)
        if p:
            p.last_used = time.time()
            self.last = pid
            self.save()

    def delete(self, pid: str) -> None:
        """Remove a profile and its data folder. The last remaining profile can't be deleted."""
        if pid not in self.profiles or len(self.profiles) <= 1:
            raise ValueError("cannot delete the only profile")
        p = self.profiles.pop(pid)
        if self.session.get("profile") == pid:
            self.session = {}
        if p.id != "default":
            shutil.rmtree(self.base / "profiles" / p.id, ignore_errors=True)
        if self.last == pid:
            self.last = self.list()[0].id
        self.save()

    # ------------------------------------------------------------ accounts
    @staticmethod
    def valid_email(email: str) -> bool:
        return bool(EMAIL_RE.match((email or "").strip()))

    @staticmethod
    def password_problem(password: str) -> str:
        """Why a new password is not good enough ("" = fine)."""
        pw = password or ""
        if len(pw) < MIN_PASSWORD:
            return f"Use at least {MIN_PASSWORD} characters."
        if pw.isdigit() or pw.isalpha():
            return "Mix letters with numbers or symbols."
        return ""

    def by_email(self, email: str) -> Optional[Profile]:
        e = (email or "").strip().lower()
        for p in self.profiles.values():
            if p.email and p.email.lower() == e:
                return p
        return None

    def set_account(self, pid: str, email: str, password: str, save: bool = True) -> Profile:
        """Attach (or change) the e-mail and password of a profile."""
        email = (email or "").strip()
        if not self.valid_email(email):
            raise ValueError("enter a valid e-mail address")
        other = self.by_email(email)
        if other is not None and other.id != pid:
            raise ValueError("that e-mail already has an account on this device")
        problem = self.password_problem(password)
        if problem:
            raise ValueError(problem)
        p = self.profiles[pid]
        salt = secrets.token_bytes(16)
        p.email, p.pw_salt, p.pw_hash = email, salt.hex(), _hash(password, salt)
        self._fails.pop("pw:" + pid, None)
        if save:
            self.save()
        return p

    def sign_up(self, name: str, email: str, password: str, avatar: str = "cat") -> Profile:
        """A new profile with an account (checks first: nothing is created on a bad e-mail / password)."""
        if not self.valid_email(email):
            raise ValueError("enter a valid e-mail address")
        if self.by_email(email) is not None:
            raise ValueError("that e-mail already has an account on this device")
        problem = self.password_problem(password)
        if problem:
            raise ValueError(problem)
        p = self.create(name or email.split("@")[0], avatar)
        return self.set_account(p.id, email, password)

    def log_in(self, email: str, password: str) -> Optional[Profile]:
        """The profile whose account matches, or None (wrong e-mail / password, or waiting after
        too many wrong tries - see ``wait_seconds("pw:" + id)``)."""
        p = self.by_email(email)
        if p is None:
            hashlib.pbkdf2_hmac("sha256", b"x", b"y" * 16, PBKDF2_ITERATIONS)   # same time either way
            return None
        key = "pw:" + p.id
        if self.wait_seconds(key) > 0:
            return None
        ok = hmac.compare_digest(_hash(password or "", bytes.fromhex(p.pw_salt)), p.pw_hash)
        if ok:
            self._fails.pop(key, None)
            return p
        n, _ = self._fails.get(key, (0, 0.0))
        n += 1
        self._fails[key] = (n, time.time() + 30 * 2 ** (n - 5) if n >= 5 else 0.0)
        return None

    # ------------------------------------------------------------ remembered session
    def _token_file(self) -> Path:
        return self.base / "session.key"

    def _load_token(self) -> None:
        try:
            self._session_token = base64.b64decode(self._token_file().read_bytes()).decode("ascii")
        except (OSError, ValueError):
            self._session_token = ""

    def remember(self, pid: str) -> None:
        """Keep ``pid`` signed in on this device (until ``forget_session``)."""
        token = secrets.token_hex(24)
        self.session = {"profile": pid, "token_hash": hashlib.sha256(token.encode()).hexdigest(),
                        "since": time.time()}
        self._session_token = token
        self.base.mkdir(parents=True, exist_ok=True)
        tf = self._token_file()
        tf.write_bytes(base64.b64encode(token.encode("ascii")))
        try:
            os.chmod(tf, 0o600)
        except OSError:
            pass
        self.save()

    def remembered(self) -> Optional[Profile]:
        """The signed-in profile of this device, if its session is valid."""
        pid = str(self.session.get("profile", ""))
        p = self.profiles.get(pid)
        if p is None or not self._session_token:
            return None
        want = str(self.session.get("token_hash", ""))
        got = hashlib.sha256(self._session_token.encode()).hexdigest()
        return p if want and hmac.compare_digest(want, got) else None

    def forget_session(self) -> None:
        self.session = {}
        self._session_token = ""
        try:
            self._token_file().unlink()
        except OSError:
            pass
        self.save()

    # ------------------------------------------------------------ PIN
    @staticmethod
    def valid_pin(pin: str) -> bool:
        return bool(PIN_RE.match(pin or ""))

    def set_pin(self, pid: str, pin: str, save: bool = True) -> None:
        if not self.valid_pin(pin):
            raise ValueError("PIN must be 4-8 digits")
        salt = secrets.token_bytes(16)
        p = self.profiles[pid]
        p.pin_salt, p.pin_hash = salt.hex(), _hash(pin, salt)
        self._fails.pop(pid, None)
        if save:
            self.save()

    def clear_pin(self, pid: str) -> None:
        p = self.profiles[pid]
        p.pin_salt = p.pin_hash = ""
        self.save()

    def wait_seconds(self, pid: str) -> float:
        n, until = self._fails.get(pid, (0, 0.0))
        return max(0.0, until - time.time())

    def verify_pin(self, pid: str, pin: str) -> bool:
        p = self.profiles.get(pid)
        if p is None:
            return False
        if not p.has_pin:
            return True
        if self.wait_seconds(pid) > 0:
            return False
        ok = hmac.compare_digest(_hash(pin or "", bytes.fromhex(p.pin_salt)), p.pin_hash)
        if ok:
            self._fails.pop(pid, None)
        else:
            n, _ = self._fails.get(pid, (0, 0.0))
            n += 1
            until = time.time() + 30 * 2 ** (n - 5) if n >= 5 else 0.0
            self._fails[pid] = (n, until)
        return ok
