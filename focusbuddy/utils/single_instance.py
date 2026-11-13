"""Single-instance guard: a second launch just opens the running app's dashboard."""

from __future__ import annotations

import getpass
import sys

from PySide6.QtNetwork import QLocalServer, QLocalSocket


class SingleInstance:
    def __init__(self, name: str):
        try:
            user = getpass.getuser()
        except Exception:  # noqa: BLE001
            user = "user"
        self.key = f"{name}-{user}"
        self.server: QLocalServer | None = None
        self._callback = None

    def acquire(self) -> bool:
        sock = QLocalSocket()
        sock.connectToServer(self.key)
        if sock.waitForConnected(300):
            sock.disconnectFromServer()
            return False
        QLocalServer.removeServer(self.key)  # stale socket after a crash
        self.server = QLocalServer()
        if not self.server.listen(self.key):
            return True  # can't guard - run anyway
        self.server.newConnection.connect(self._on_connection)
        return True

    def notify_running(self) -> None:
        if sys.platform.startswith("win"):
            # this short-lived process was started by the user's click: let the running FocusBuddy take
            # the foreground (Windows otherwise only flashes its taskbar button)
            try:
                import ctypes
                ctypes.windll.user32.AllowSetForegroundWindow(-1)        # ASFW_ANY
            except Exception:  # noqa: BLE001
                pass
        sock = QLocalSocket()
        sock.connectToServer(self.key)
        if sock.waitForConnected(500):
            sock.write(b"show")
            sock.flush()
            sock.waitForBytesWritten(500)
            sock.disconnectFromServer()

    def on_activate(self, cb) -> None:
        self._callback = cb

    def _on_connection(self) -> None:
        conn = self.server.nextPendingConnection()
        if conn is None:
            return
        conn.waitForReadyRead(300)
        conn.readAll()
        conn.disconnectFromServer()
        if self._callback:
            self._callback()

    def release(self) -> None:
        """Stop guarding (before a relaunch, so the new process can take over)."""
        if self.server is not None:
            self.server.close()
            QLocalServer.removeServer(self.key)
            self.server = None
