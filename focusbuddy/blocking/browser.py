"""Browser tab actions via the FocusBuddy extension."""

from __future__ import annotations

from typing import Optional

from ..browser.bridge import BrowserBridge
from ..core.activity import Activity


class BrowserBlocker:
    def __init__(self, bridge: Optional[BrowserBridge]):
        self.bridge = bridge

    def can_control_tab(self, activity: Activity) -> bool:
        return (self.bridge is not None and activity.domain_source == "extension"
                and activity.tab_id is not None and activity.tab_client is not None)

    def close_tab(self, activity: Activity) -> bool:
        return self.send(activity, "close_tab") is not None

    def redirect_tab(self, activity: Activity) -> bool:
        return self.send(activity, "redirect_tab") is not None

    def send(self, activity: Activity, kind: str) -> Optional[int]:
        """Queue a tab command for the extension; returns its id (acknowledged later) or None."""
        if not self.can_control_tab(activity):
            return None
        return self.bridge.send_command(activity.tab_client, {
            "type": kind, "tabId": activity.tab_id, "expectHost": activity.domain or ""})
