"""Blocking facade: picks the safest supported action, executes it and *verifies* it worked.

Website (browser) chain
    1. the FocusBuddy extension closes / redirects exactly that tab (acknowledged by the extension)
    2. no extension, or no acknowledgement within a few seconds: the browser's own Ctrl+W shortcut,
       sent only to the verified foreground browser window whose title is still that tab
    3. still showing: the browser window is minimized

Application chain
    1. a polite WM_CLOSE (the app may still ask to save) - or minimize, if that is the chosen action
    2. still on screen a few seconds later (many chat / game launchers ignore or delay WM_CLOSE):
       the window is minimized

Every step re-checks that the window still exists and belongs to the same process.
FocusBuddy never force-kills a process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..core.activity import Activity
from ..core.rules import Rule
from ..utils.logging_setup import get_logger, kv
from .application import ApplicationBlocker
from .browser import BrowserBlocker
from .safety import SafetyGuard

log = get_logger("blocker")

ACK_TIMEOUT = 4.0        # seconds the extension gets to confirm a tab close
APP_CLOSE_GRACE = 3.0    # seconds an app gets to close its window after WM_CLOSE
KEYS_GRACE = 1.5         # seconds for the browser to react to Ctrl+W


@dataclass
class BlockResult:
    ok: bool
    action: str            # closed | redirected | minimized | none
    message_event: str     # dialogue event to show
    reason: str = ""
    pending: bool = False  # the result is being verified (extension ack / window check)


@dataclass
class _Pending:
    kind: str              # ext | keys | app
    activity: Activity
    deadline: float
    cid: Optional[int] = None
    title: str = ""
    extra: dict = field(default_factory=dict)


class Blocker:
    def __init__(self, apps: ApplicationBlocker, browser: BrowserBlocker, safety: SafetyGuard, settings,
                 clock: Callable[[], float] | None = None):
        self.apps = apps
        self.browser = browser
        self.safety = safety
        self.settings = settings
        self.clock = clock or time.time
        self.pending: list[_Pending] = []

    # ------------------------------------------------------------------ choice
    def preferred_action(self, activity: Activity) -> str:
        if activity.is_browser:
            pref = self.settings.get("website_action", "close_tab")
            if pref in ("close_tab", "redirect") and self.browser.can_control_tab(activity):
                return "close_tab" if pref == "close_tab" else "redirect_tab"
            if pref in ("close_tab", "redirect") and self.settings.get("keyboard_tab_close", True):
                return "close_tab_keys"
            return "minimize"
        return "close_window" if self.settings.get("app_action", "close") == "close" else "minimize"

    # ------------------------------------------------------------------- block
    def block(self, activity: Activity, rule: Optional[Rule], automatic: bool, enforced: bool = False) -> BlockResult:
        action = self.preferred_action(activity)
        decision = self.safety.check(activity, action, rule, automatic, enforced)
        log.info("Block requested %s", kv(app=activity.exe, website=activity.domain, action=action,
                                           automatic=automatic, enforced=enforced or None,
                                           allowed=decision.allowed, reason=decision.reason or None))
        if not decision.allowed:
            event = "PROTECTED_APP" if decision.reason == "protected application" else "CLOSE_FAILED"
            return BlockResult(False, "none", event, decision.reason)
        action = decision.action
        now = self.clock()

        if action in ("close_tab", "redirect_tab"):
            # Make sure the browser window is still there before asking the extension.
            if not self.apps.verify(activity):
                return BlockResult(False, "none", "CLOSE_FAILED", "window changed")
            cid = self.browser.send(activity, action)
            if cid is not None:
                self.pending.append(_Pending("ext", activity, now + ACK_TIMEOUT, cid=cid, title=activity.title,
                                             extra={"action": action}))
                return BlockResult(True, "closed" if action == "close_tab" else "redirected",
                                   "CLOSED" if action == "close_tab" else "REDIRECTED", pending=True)
            action = "close_tab_keys" if self.settings.get("keyboard_tab_close", True) else "minimize"

        if action == "close_tab_keys":
            r = self._keys(activity, now)
            if r is not None:
                return r
            action = "minimize"

        if action == "minimize":
            ok = self.apps.minimize(activity)
            no_ext = activity.is_browser and self.settings.get("website_action") != "minimize" \
                and not self.browser.can_control_tab(activity)
            return BlockResult(ok, "minimized" if ok else "none",
                               ("NO_EXTENSION" if no_ext else "MINIMIZED") if ok else "CLOSE_FAILED")
        if action == "close_window":
            ok = self.apps.close(activity)
            if ok:
                self.pending.append(_Pending("app", activity, now + APP_CLOSE_GRACE))
            return BlockResult(ok, "closed" if ok else "none", "CLOSED" if ok else "CLOSE_FAILED", pending=ok)
        return BlockResult(False, "none", "CLOSE_FAILED", "unsupported")

    def _keys(self, activity: Activity, now: float) -> Optional[BlockResult]:
        """Ctrl+W to the verified browser window. None = could not send (caller falls back)."""
        if activity.window is None or not self.apps.verify(activity):
            return None
        if not self.apps.wm.close_browser_tab(activity.window.handle, activity.window.title or ""):
            return None
        log.info("Tab closed with the browser shortcut %s", kv(website=activity.domain, app=activity.exe))
        self.pending.append(_Pending("keys", activity, now + KEYS_GRACE, title=activity.window.title or ""))
        return BlockResult(True, "closed", "CLOSED", pending=True)

    # ------------------------------------------------------------ verification
    def check_pending(self, acks: list[dict] | None = None, now: float | None = None) -> list[BlockResult]:
        """Called from the monitoring loop: confirm closes, fall back where one didn't happen.
        Returns the results of any fallback actions taken."""
        now = self.clock() if now is None else now
        acked = {a.get("id"): a for a in (acks or [])}
        out: list[BlockResult] = []
        keep: list[_Pending] = []
        for p in self.pending:
            if p.kind == "ext":
                ack = acked.get(p.cid)
                if ack is not None:
                    if ack.get("ok") or ack.get("error") == "tab changed":
                        continue                                    # done (or the user already moved on)
                    log.warning("Extension could not close the tab %s", kv(err=ack.get("error"), id=p.cid))
                    out.append(self._fallback_browser(p, now))
                elif now >= p.deadline:
                    log.warning("No answer from the extension %s", kv(id=p.cid, website=p.activity.domain))
                    out.append(self._fallback_browser(p, now))
                else:
                    keep.append(p)
            elif p.kind == "keys":
                if now < p.deadline:
                    keep.append(p)
                    continue
                fresh = self.apps.wm.get_window(p.activity.window.handle) if p.activity.window else None
                if fresh is not None and fresh.title == p.title and not fresh.minimized:
                    out.append(self._minimize(p.activity, "tab still open"))
            elif p.kind == "app":
                if now < p.deadline:
                    keep.append(p)
                    continue
                fresh = self.apps.wm.get_window(p.activity.window.handle) if p.activity.window else None
                if fresh is not None and fresh.pid == p.activity.window.pid and not fresh.minimized:
                    out.append(self._minimize(p.activity, "app ignored the close request"))
        self.pending = keep
        return out

    def _fallback_browser(self, p: _Pending, now: float) -> BlockResult:
        a = p.activity
        fresh = self.apps.wm.get_window(a.window.handle) if a.window else None
        if fresh is None or fresh.title != (a.window.title if a.window else None):
            # the tab / window is already gone or shows something else: nothing left to do
            return BlockResult(True, "closed", "CLOSED", "gone")
        if self.settings.get("keyboard_tab_close", True):
            r = self._keys(a, now)
            if r is not None:
                return r
        return self._minimize(a, "extension did not close the tab")

    def _minimize(self, activity: Activity, why: str) -> BlockResult:
        ok = self.apps.minimize(activity)
        log.info("Fallback minimize %s", kv(app=activity.exe, website=activity.domain, ok=ok, why=why))
        return BlockResult(ok, "minimized" if ok else "none", "MINIMIZED" if ok else "CLOSE_FAILED", why)
