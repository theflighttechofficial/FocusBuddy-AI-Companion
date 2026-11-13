"""RoamingController: where should the companion wander next?

Destination kinds
    RANDOM_SAFE_POINT   somewhere on the current floor / surface
    SCREEN_EDGE         the end of the floor (peek over the edge)
    WINDOW_EDGE         the top edge of a visible window (leap up onto it)
    NEAR_USER_CURSOR    the floor under the mouse pointer
    NEAR_ACTIVE_WINDOW  next to the window the user works in
    OTHER_MONITOR       occasionally visit another screen

Recent destinations and per-kind cooldowns keep the walks varied.
"""

from __future__ import annotations

import enum
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional


class DestinationKind(str, enum.Enum):
    RANDOM_SAFE_POINT = "RANDOM_SAFE_POINT"
    SCREEN_EDGE = "SCREEN_EDGE"
    WINDOW_EDGE = "WINDOW_EDGE"
    NEAR_USER_CURSOR = "NEAR_USER_CURSOR"
    NEAR_ACTIVE_WINDOW = "NEAR_ACTIVE_WINDOW"
    OTHER_MONITOR = "OTHER_MONITOR"
    DISTRACTION_WINDOW = "DISTRACTION_WINDOW"


@dataclass
class Destination:
    kind: DestinationKind
    x: float
    y: float
    window: Optional[int] = None      # handle when standing on a window
    edge: int = 0                     # -1 left edge / +1 right edge for SCREEN_EDGE


COOLDOWN = {
    DestinationKind.OTHER_MONITOR: 300.0,
    DestinationKind.WINDOW_EDGE: 25.0,
    DestinationKind.SCREEN_EDGE: 30.0,
    DestinationKind.NEAR_USER_CURSOR: 20.0,
    DestinationKind.NEAR_ACTIVE_WINDOW: 40.0,
}


class RoamingController:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        self.recent: deque[tuple[float, float]] = deque(maxlen=6)
        self._last_used: dict[DestinationKind, float] = {}

    def _cool(self, kind: DestinationKind, now: float) -> bool:
        return now - self._last_used.get(kind, -1e9) >= COOLDOWN.get(kind, 0.0)

    def _fresh(self, x: float, y: float, min_dist: float = 110.0) -> bool:
        return all(math.dist((x, y), r) >= min_dist for r in self.recent)

    def remember(self, d: Destination, now: float) -> None:
        self.recent.append((d.x, d.y))
        self._last_used[d.kind] = now

    def choose(self, *, now: float, pos: tuple[float, float], span: tuple[float, float], surface_y: float,
               screen: tuple[float, float, float, float], screens: list, windows: list, cursor=None,
               active_window=None, level: str = "desktop", half: float = 45.0, char_h: float = 120.0,
               allow_other_monitor: bool = True) -> Destination:
        """span = walkable x-range on the current surface; screen = (l, t, r, b) available area."""
        r = self.rng
        lo, hi = span
        weights: list[tuple[DestinationKind, float]] = [(DestinationKind.RANDOM_SAFE_POINT, 6.0),
                                                        (DestinationKind.SCREEN_EDGE, 1.0)]
        if level in ("desktop", "all") and windows:
            weights.append((DestinationKind.WINDOW_EDGE, 1.6))
        if cursor is not None:
            weights.append((DestinationKind.NEAR_USER_CURSOR, 1.0))
        if active_window is not None:
            weights.append((DestinationKind.NEAR_ACTIVE_WINDOW, 1.0))
        if level == "all" and allow_other_monitor and len(screens) > 1:
            weights.append((DestinationKind.OTHER_MONITOR, 0.6))
        weights = [(k, w) for k, w in weights if self._cool(k, now)]
        for _ in range(6):
            total = sum(w for _, w in weights)
            pick = r.uniform(0, total)
            kind = weights[-1][0]
            for k, w in weights:
                pick -= w
                if pick <= 0:
                    kind = k
                    break
            d = self._make(kind, pos, lo, hi, surface_y, screen, screens, windows, cursor, active_window,
                           half, char_h)
            if d is not None and (self._fresh(d.x, d.y) or kind != DestinationKind.RANDOM_SAFE_POINT):
                self.remember(d, now)
                return d
        # fallback: any point on our surface away from here
        x = r.uniform(lo, hi)
        d = Destination(DestinationKind.RANDOM_SAFE_POINT, x, surface_y)
        self.remember(d, now)
        return d

    # ------------------------------------------------------------------
    def _make(self, kind, pos, lo, hi, surface_y, screen, screens, windows, cursor, active_window,
              half, char_h) -> Optional[Destination]:
        r = self.rng
        x0 = pos[0]
        if kind == DestinationKind.RANDOM_SAFE_POINT:
            if hi - lo < 40:
                return None
            dist = r.uniform(90, max(120.0, min(520.0, (hi - lo) * 0.6))) * r.choice((-1, 1))
            x = max(lo, min(hi, x0 + dist))
            if abs(x - x0) < 50:
                x = max(lo, min(hi, x0 - dist))
            return Destination(kind, x, surface_y)
        if kind == DestinationKind.SCREEN_EDGE:
            edge = -1 if abs(x0 - lo) > abs(x0 - hi) else 1
            if r.random() < 0.35:
                edge = -edge
            return Destination(kind, lo if edge < 0 else hi, surface_y, edge=edge)
        if kind == DestinationKind.WINDOW_EDGE:
            tops = [w for w in windows if w.get("perchable")]
            if not tops:
                return None
            w = r.choice(tops)
            x = r.uniform(w["x"] + half, max(w["x"] + half, w["x"] + w["w"] - half))
            return Destination(kind, x, w["y"], window=w.get("handle"))
        if kind == DestinationKind.NEAR_USER_CURSOR and cursor is not None:
            x = max(lo, min(hi, cursor[0] + r.uniform(-90, 90)))
            return Destination(kind, x, surface_y)
        if kind == DestinationKind.NEAR_ACTIVE_WINDOW and active_window is not None:
            ax = active_window[0] + (active_window[2] if r.random() < 0.5 else 0) + r.uniform(-60, 60)
            return Destination(kind, max(lo, min(hi, ax)), surface_y)
        if kind == DestinationKind.OTHER_MONITOR and len(screens) > 1:
            others = [s for s in screens if not (s[0] <= x0 <= s[2])]
            if not others:
                return None
            s = r.choice(others)
            x = r.uniform(s[0] + half * 2, s[2] - half * 2)
            return Destination(kind, x, s[3])
        return None
