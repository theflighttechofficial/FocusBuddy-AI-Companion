"""Local bridge between the desktop app and the FocusBuddy browser extension.

Why an extension?  A desktop app cannot reliably read the URL of the
active browser tab (titles are inconsistent, UI-automation is fragile and
slow) and cannot close a single tab safely.  The extension knows the
active tab exactly and can close it with the browser's own API.

Protocol (HTTP/JSON on 127.0.0.1 only):
    GET  /ping                          -> {"app": "FocusBuddy", "version": ...}
    POST /report  {client, browser, active}          -> {"commands": [...]}
    POST /poll    {client, browser, active, wait}    -> long-poll for commands
    POST /ack     {client, id, ok, error}
    POST /event   {client, type, host}               -> e.g. "blocked" (focus policy enforced in the browser)

Every /report, /poll and /event answer carries the focus ``policy``: during a focus session with
"instant" enforcement the extension redirects distracting sites itself the moment they load (it
works even between two monitoring polls). The policy expires on its own (``until``) if the app stops.

Security: binds to loopback only; requires the ``X-FocusBuddy: 1`` header
(web pages cannot send it cross-origin without a CORS preflight, which we
never approve) and rejects requests whose Origin is a normal web origin.
"""

from __future__ import annotations

import itertools
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .. import __version__
from ..config import CHROMIUM_FAMILY
from ..utils.logging_setup import get_logger

log = get_logger("bridge")

MAX_BODY = 64 * 1024
ALLOWED_ORIGINS = ("chrome-extension://", "moz-extension://", "safari-web-extension://", "extension://")
FRESH_SECONDS = 75.0


@dataclass
class TabReport:
    client: str
    browser: str
    url: Optional[str]
    title: str
    tab_id: Optional[int]
    window_id: Optional[int]
    incognito: bool
    focused: bool
    received: float


@dataclass
class ClientInfo:
    client: str
    browser: str
    last_seen: float
    commands: list = field(default_factory=list)


class BrowserBridge:
    def __init__(self, port: int, host: str = "127.0.0.1"):
        self.port = port
        self.host = host
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._clients: dict[str, ClientInfo] = {}
        self._reports: dict[str, TabReport] = {}
        self._acks: list[dict] = []
        self._ids = itertools.count(1)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None
        self._events: list[dict] = []
        # focus-session policy for the extension ({"active": bool, "block": [...], ...}); set by the app
        self.policy_provider = lambda: {"active": False}

    # ------------------------------------------------------------ lifecycle
    def start(self) -> bool:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "FocusBuddyBridge/1"

            def log_message(self, fmt, *args):  # silence default stderr logging
                pass

            def _deny(self, code=403):
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _ok(self, payload: dict):
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def _allowed(self) -> bool:
                origin = self.headers.get("Origin")
                if origin and not origin.startswith(ALLOWED_ORIGINS):
                    return False
                return self.headers.get("X-FocusBuddy") == "1"

            def do_OPTIONS(self):  # never approve CORS preflights from web pages
                self._deny(403)

            def do_GET(self):
                if self.path.split("?")[0] == "/ping" and self._allowed():
                    self._ok({"app": "FocusBuddy", "version": __version__})
                else:
                    self._deny(404 if self._allowed() else 403)

            def do_POST(self):
                if not self._allowed():
                    return self._deny(403)
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                    if n > MAX_BODY:
                        return self._deny(413)
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError
                except (ValueError, OSError):
                    return self._deny(400)
                route = self.path.split("?")[0]
                try:
                    if route == "/report":
                        bridge._on_report(body)
                        self._ok({"commands": bridge._take_commands(body.get("client", "")),
                                  "policy": bridge._policy()})
                    elif route == "/poll":
                        bridge._on_report(body)
                        wait = max(0.0, min(float(body.get("wait", 20)), 25.0))
                        pol = bridge._policy()
                        if "policySig" in body and body.get("policySig") != pol.get("sig"):
                            wait = 0.0                      # the extension's policy is out of date: answer now
                        self._ok({"commands": bridge._wait_commands(body.get("client", ""), wait),
                                  "policy": pol})
                    elif route == "/ack":
                        bridge._on_ack(body)
                        self._ok({"ok": True})
                    elif route == "/event":
                        bridge._on_event(body)
                        self._ok({"ok": True, "policy": bridge._policy()})
                    else:
                        self._deny(404)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        class Server(ThreadingHTTPServer):
            # On Windows SO_REUSEADDR lets two processes share a port - never allow that.
            allow_reuse_address = not sys.platform.startswith("win")

        try:
            self._server = Server((self.host, self.port), Handler)
            self._server.daemon_threads = True
        except OSError as exc:
            self.error = f"Port {self.port} unavailable ({exc.strerror or exc})"
            log.warning("Browser bridge could not start port=%s err=%s", self.port, exc)
            return False
        self._thread = threading.Thread(target=self._server.serve_forever, name="browser-bridge", daemon=True)
        self._thread.start()
        log.info("Browser bridge listening addr=%s:%s", self.host, self.port)
        return True

    def stop(self) -> None:
        if self._server:
            with self._cond:
                self._cond.notify_all()
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    # ------------------------------------------------------- server side
    def _on_report(self, body: dict) -> None:
        client = str(body.get("client") or "")[:64]
        if not client:
            return
        browser = str(body.get("browser") or "chrome")[:20].lower()
        now = time.time()
        active = body.get("active")
        with self._lock:
            info = self._clients.get(client)
            if info is None:
                info = self._clients[client] = ClientInfo(client, browser, now)
                log.info("Browser extension connected browser=%s", browser)
            info.last_seen = now
            info.browser = browser
            if isinstance(active, dict):
                tab_id = active.get("tabId")
                self._reports[client] = TabReport(
                    client=client, browser=browser,
                    url=active.get("url") if isinstance(active.get("url"), str) else None,
                    title=str(active.get("title") or "")[:300],
                    tab_id=int(tab_id) if isinstance(tab_id, int) else None,
                    window_id=active.get("windowId") if isinstance(active.get("windowId"), int) else None,
                    incognito=bool(active.get("incognito")), focused=bool(active.get("focused", True)),
                    received=now)

    def _take_commands(self, client: str) -> list:
        with self._lock:
            info = self._clients.get(client)
            if not info or not info.commands:
                return []
            cmds, info.commands = info.commands, []
            return cmds

    def _wait_commands(self, client: str, wait: float) -> list:
        deadline = time.time() + wait
        with self._cond:
            while True:
                info = self._clients.get(client)
                if info and info.commands:
                    cmds, info.commands = info.commands, []
                    return cmds
                remaining = deadline - time.time()
                if remaining <= 0 or self._server is None:
                    return []
                self._cond.wait(min(remaining, 1.0))

    def _policy(self) -> dict:
        try:
            p = self.policy_provider()
            p = dict(p) if isinstance(p, dict) else {"active": False}
        except Exception:  # noqa: BLE001 - a broken provider must not break the bridge
            log.exception("policy provider failed")
            p = {"active": False}
        core = {k: v for k, v in p.items() if k not in ("until", "sig")}
        p["sig"] = str(abs(hash(json.dumps(core, sort_keys=True))) % 10 ** 12)
        return p

    def _on_event(self, body: dict) -> None:
        kind = str(body.get("type") or "")[:32]
        host = str(body.get("host") or "")[:200]
        if kind:
            with self._lock:
                self._events.append({"type": kind, "host": host, "client": str(body.get("client") or "")[:64]})
                del self._events[:-100]

    def drain_events(self) -> list[dict]:
        with self._lock:
            ev, self._events = self._events, []
        return ev

    def notify_policy_changed(self) -> None:
        """Wake long-polling extensions so they pick up a new policy at once."""
        with self._cond:
            for c in self._clients.values():
                c.commands.append({"id": next(self._ids), "type": "policy"})
            self._cond.notify_all()

    def _on_ack(self, body: dict) -> None:
        with self._lock:
            self._acks.append({"id": body.get("id"), "ok": bool(body.get("ok")),
                               "error": str(body.get("error") or "")[:200]})

    # ------------------------------------------------------------ app side
    def connected(self, within: float = FRESH_SECONDS) -> bool:
        now = time.time()
        with self._lock:
            return any(now - c.last_seen < within for c in self._clients.values())

    def clients(self) -> list[ClientInfo]:
        with self._lock:
            return list(self._clients.values())

    def get_active_tab(self, browser: str, window_title: str = "") -> Optional[TabReport]:
        """Best matching fresh report for a foreground browser window."""
        now = time.time()
        title = (window_title or "").lower()
        with self._lock:
            reports = [r for r in self._reports.values() if now - r.received < FRESH_SECONDS]
        if not reports:
            return None

        def family_ok(r: TabReport) -> bool:
            if r.browser == browser:
                return True
            return r.browser in CHROMIUM_FAMILY and browser in CHROMIUM_FAMILY

        def title_ok(r: TabReport) -> bool:
            return bool(r.title) and r.title.lower()[:60] in title

        candidates = [r for r in reports if family_ok(r)]
        if not candidates:
            return None
        titled = [r for r in candidates if title_ok(r)]
        if titled:
            return max(titled, key=lambda r: r.received)
        exact = [r for r in candidates if r.browser == browser and r.focused]
        pool = exact or [r for r in candidates if r.focused] or candidates
        best = max(pool, key=lambda r: r.received)
        # Without a title match, only trust the report if the window title is unknown
        # or the report is very recent (the user just switched tabs).
        if title and now - best.received > 15:
            return None
        return best

    def send_command(self, client: str, command: dict) -> Optional[int]:
        with self._cond:
            info = self._clients.get(client)
            if info is None:
                return None
            cid = next(self._ids)
            info.commands.append({"id": cid, **command})
            self._cond.notify_all()
        log.info("Browser command queued type=%s id=%s", command.get("type"), cid)
        return cid

    def drain_acks(self) -> list[dict]:
        with self._lock:
            acks, self._acks = self._acks, []
        return acks
