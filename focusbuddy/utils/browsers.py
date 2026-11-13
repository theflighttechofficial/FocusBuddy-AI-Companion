"""Find the installed Chromium browsers and open their extensions page (setup wizard, Settings).

Chrome only lets the user add an unpacked extension themselves (Developer mode -> Load unpacked), so
FocusBuddy can't install the connector silently - it opens the right page and the folder to pick.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# (id, display name, Windows executable, extensions page, other-OS commands)
BROWSERS = (
    ("chrome", "Google Chrome", "chrome.exe", "chrome://extensions/", ("google-chrome", "google-chrome-stable",
                                                                         "chromium", "chromium-browser")),
    ("edge", "Microsoft Edge", "msedge.exe", "edge://extensions/", ("microsoft-edge", "microsoft-edge-stable")),
    ("brave", "Brave", "brave.exe", "brave://extensions/", ("brave-browser", "brave")),
)


def _app_path(exe: str) -> str | None:
    """A Windows program's path from the registry's "App Paths" (how Run / Start find it)."""
    try:
        import winreg
    except ImportError:
        return None
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{exe}") as k:
                path = winreg.QueryValueEx(k, "")[0]
                if path and Path(path).is_file():
                    return str(path)
        except OSError:
            continue
    return None


def find(browser_id: str) -> str | None:
    """The executable of a browser, or None when it isn't installed."""
    for bid, _name, exe, _page, cmds in BROWSERS:
        if bid != browser_id:
            continue
        if sys.platform.startswith("win"):
            return _app_path(exe)
        for c in cmds:
            p = shutil.which(c)
            if p:
                return p
    return None


def installed() -> list[tuple[str, str]]:
    """[(id, display name)] of the Chromium browsers on this computer, Chrome first."""
    return [(bid, name) for bid, name, *_ in BROWSERS if find(bid)]


def open_extensions_page(browser_id: str | None = None) -> str | None:
    """Open the extensions page of ``browser_id`` (or the first browser found). Returns the browser's
    display name, or None if no supported browser was found."""
    for bid, name, _exe, page, _cmds in BROWSERS:
        if browser_id and bid != browser_id:
            continue
        exe = find(bid)
        if not exe:
            continue
        try:
            kw = {"close_fds": True}
            if sys.platform.startswith("win"):
                kw["creationflags"] = 0x00000008          # DETACHED_PROCESS
            subprocess.Popen([exe, page], **kw)  # noqa: S603 - the user's own browser
            return name
        except OSError:
            continue
    return None
