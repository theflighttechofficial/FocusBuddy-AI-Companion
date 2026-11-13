"""Application window actions (minimize / polite close) via the WindowManager."""

from __future__ import annotations

from ..core.activity import Activity
from ..platform.base import WindowManager


class ApplicationBlocker:
    def __init__(self, wm: WindowManager):
        self.wm = wm          # also used by Blocker for the Ctrl+W tab close and verification

    def verify(self, activity: Activity) -> bool:
        """Re-check that the window still exists and belongs to the same process."""
        if activity.window is None:
            return False
        fresh = self.wm.get_window(activity.window.handle)
        if fresh is None:
            return False
        return fresh.pid == activity.window.pid and fresh.exe == activity.window.exe

    def minimize(self, activity: Activity) -> bool:
        return self.verify(activity) and self.wm.minimize_window(activity.window.handle)

    def close(self, activity: Activity) -> bool:
        return self.verify(activity) and self.wm.close_window(activity.window.handle)

    def activate(self, handle: int) -> bool:
        return self.wm.activate_window(handle)
