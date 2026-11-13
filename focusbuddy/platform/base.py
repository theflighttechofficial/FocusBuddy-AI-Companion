"""Platform abstraction layer.

Everything OS-specific lives behind these interfaces so macOS / Linux
backends can be added later without touching the core:

    WindowManager       - active window, geometry, minimize/close/activate
    AutostartManager    - "start with OS"
    (NotificationManager lives in ui/tray.py - Qt handles it cross-platform)
    (BrowserManager      lives in browser/bridge.py - extension based)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WinRect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass
class WindowInfo:
    handle: int
    title: str
    exe: str                       # lower-case executable name, e.g. "chrome.exe"
    pid: int
    rect: Optional[WinRect] = None  # physical pixels (native) unless coords == "logical"
    monitor_name: Optional[str] = None
    monitor_rect: Optional[WinRect] = None
    minimized: bool = False
    coords: str = "physical"        # physical | logical
    extra: dict = field(default_factory=dict)

    @property
    def app_name(self) -> str:
        name = self.exe[:-4] if self.exe.endswith(".exe") else self.exe
        return name or "unknown"


class WindowManager(ABC):
    """Read and (carefully) act on top-level windows."""

    @abstractmethod
    def get_active_window(self) -> Optional[WindowInfo]:
        ...

    @abstractmethod
    def get_window(self, handle: int) -> Optional[WindowInfo]:
        """Fresh info for a handle, or None if the window no longer exists."""

    @abstractmethod
    def is_window(self, handle: int) -> bool:
        ...

    @abstractmethod
    def minimize_window(self, handle: int) -> bool:
        ...

    @abstractmethod
    def close_window(self, handle: int) -> bool:
        """Politely ask a window to close (WM_CLOSE). Apps may prompt to save."""

    @abstractmethod
    def activate_window(self, handle: int) -> bool:
        ...

    @abstractmethod
    def get_idle_seconds(self) -> float:
        """Seconds since last keyboard/mouse input (no key content is read)."""

    def close_browser_tab(self, handle: int, expect_title: str) -> bool:
        """Close the *active tab* of a browser window with the browser's own shortcut (Ctrl+W).

        Used only when the FocusBuddy extension can't close the tab. Acts only if ``handle`` is the
        foreground window and its title is still exactly ``expect_title`` (the tab the decision was
        about), so the keystroke can never reach another window or another tab. Backends that
        can't do this return False."""
        return False

    def list_windows(self) -> list[WindowInfo]:
        """Visible top-level windows, front-most first (for the companion's world model)."""
        return []

    def get_window_position(self, handle: int) -> Optional[WinRect]:
        info = self.get_window(handle)
        return info.rect if info else None

    @property
    def available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return type(self).__name__


class AutostartManager(ABC):
    @abstractmethod
    def is_enabled(self) -> bool:
        ...

    @abstractmethod
    def set_enabled(self, on: bool) -> bool:
        ...


class NullWindowManager(WindowManager):
    """Used on unsupported platforms: monitoring simply reports nothing."""

    def get_active_window(self):
        return None

    def get_window(self, handle):
        return None

    def is_window(self, handle):
        return False

    def minimize_window(self, handle):
        return False

    def close_window(self, handle):
        return False

    def activate_window(self, handle):
        return False

    def get_idle_seconds(self):
        return 0.0

    @property
    def available(self) -> bool:
        return False


class NullAutostart(AutostartManager):
    def is_enabled(self) -> bool:
        return False

    def set_enabled(self, on: bool) -> bool:
        return False
