"""Map native (physical-pixel) window rectangles to Qt logical coordinates.

Qt 6 is per-monitor DPI aware.  Win32 reports physical pixels, Qt
positions windows in device-independent pixels, and each monitor can have
its own scale factor.  We match the Win32 monitor to the QScreen by device
name and convert relative to that monitor's origin.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtGui import QGuiApplication

from ..companion.behavior import Rect
from .base import WindowInfo


def to_logical(info: Optional[WindowInfo]) -> Optional[Rect]:
    if info is None or info.rect is None:
        return None
    r = info.rect
    if info.coords == "logical":
        return Rect(r.x, r.y, r.w, r.h, info.handle)
    screens = QGuiApplication.screens()
    screen = None
    if info.monitor_name:
        for s in screens:
            if s.name() == info.monitor_name:
                screen = s
                break
    screen = screen or QGuiApplication.primaryScreen()
    if screen is None:
        return Rect(r.x, r.y, r.w, r.h, info.handle)
    dpr = screen.devicePixelRatio() or 1.0
    geo = screen.geometry()
    if info.monitor_rect is not None:
        mx, my = info.monitor_rect.x, info.monitor_rect.y
        return Rect(geo.x() + (r.x - mx) / dpr, geo.y() + (r.y - my) / dpr, r.w / dpr, r.h / dpr, info.handle)
    return Rect(r.x / dpr, r.y / dpr, r.w / dpr, r.h / dpr, info.handle)
