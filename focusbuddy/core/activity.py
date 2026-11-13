"""The Activity snapshot produced by the monitor on every poll."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..platform.base import WindowInfo

PRETTY_APPS = {
    "code.exe": "VS Code", "chrome.exe": "Chrome", "msedge.exe": "Edge", "firefox.exe": "Firefox",
    "brave.exe": "Brave", "opera.exe": "Opera", "winword.exe": "Word", "excel.exe": "Excel",
    "powerpnt.exe": "PowerPoint", "pycharm64.exe": "PyCharm", "idea64.exe": "IntelliJ IDEA",
    "windowsterminal.exe": "Terminal", "explorer.exe": "File Explorer", "steam.exe": "Steam",
    "discord.exe": "Discord", "spotify.exe": "Spotify", "notepad.exe": "Notepad",
    "taskmgr.exe": "Task Manager", "whatsapp.exe": "WhatsApp", "telegram.exe": "Telegram",
}


@dataclass
class Activity:
    timestamp: float
    window: Optional[WindowInfo]
    exe: str = ""
    title: str = ""                       # kept in memory only, never stored/logged
    is_browser: bool = False
    browser: Optional[str] = None         # chrome | edge | firefox | ...
    url: Optional[str] = None
    domain: Optional[str] = None
    path: str = ""
    domain_source: str = "none"           # extension | title | none
    tab_id: Optional[int] = None
    tab_client: Optional[str] = None
    idle_seconds: float = 0.0
    is_self: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Identity of the thing the user is looking at (domain beats app)."""
        if self.domain:
            return self.domain
        return self.exe or "unknown"

    @property
    def app_name(self) -> str:
        if self.exe in PRETTY_APPS:
            return PRETTY_APPS[self.exe]
        name = self.exe[:-4] if self.exe.endswith(".exe") else self.exe
        return name.capitalize() if name else "Unknown app"

    @property
    def display(self) -> str:
        return self.domain or self.app_name

    @property
    def handle(self) -> Optional[int]:
        return self.window.handle if self.window else None
