"""Lightweight 2-D path planning for the companion (no game engine).

The desktop is a set of *surfaces* the companion can stand on: the floor of
each monitor (top of the taskbar) and the top edges of visible windows.
Travelling between heights uses ballistic leaps; travelling in the open
avoids blocking windows with corner waypoints:

    direct path -> obstacle intersection test -> corner waypoint(s) -> target
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def inflated(self, m: float) -> "Box":
        return Box(self.x - m, self.y - m, self.w + 2 * m, self.h + 2 * m)

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.right and self.y <= py <= self.bottom


@dataclass
class Waypoint:
    x: float
    y: float
    mode: str = "move"      # move | leap


def segment_intersects_box(x1: float, y1: float, x2: float, y2: float, b: Box) -> bool:
    """Liang-Barsky clip test."""
    dx, dy = x2 - x1, y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - b.x, b.right - x1, y1 - b.y, b.bottom - y1)
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
            continue
        t = qi / pi
        if pi < 0:
            u0 = max(u0, t)
        else:
            u1 = min(u1, t)
        if u0 > u1:
            return False
    return True


def path_length(points: list[tuple[float, float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


class PathPlanner:
    MARGIN = 14.0

    def blocked(self, a: tuple[float, float], b: tuple[float, float], obstacles: Iterable[Box]) -> bool:
        for ob in obstacles:
            if segment_intersects_box(a[0], a[1], b[0], b[1], ob):
                return True
        return False

    def plan_free(self, start: tuple[float, float], goal: tuple[float, float], obstacles: list[Box],
                  inside: Callable[[float, float], bool] | None = None) -> list[Waypoint]:
        """Open-space route around windows: direct, else 1 or 2 corner detours.

        Obstacles that contain the start or the goal (e.g. the window we stand on / walk to)
        are ignored; corner waypoints sit just outside the inflated obstacles."""
        inside = inside or (lambda x, y: True)
        obs = [o.inflated(self.MARGIN) for o in obstacles]
        obs = [o for o in obs if not o.contains(*start) and not o.contains(*goal)]
        if not self.blocked(start, goal, obs):
            return [Waypoint(*goal)]
        corners: list[tuple[float, float]] = []
        e = 1.0
        for o in obs:
            corners += [(o.x - e, o.y - e), (o.right + e, o.y - e), (o.x - e, o.bottom + e), (o.right + e, o.bottom + e)]
        corners = [c for c in corners if inside(*c) and not any(o.contains(*c) for o in obs)]
        best: Optional[list[tuple[float, float]]] = None
        for c in corners:
            if not self.blocked(start, c, obs) and not self.blocked(c, goal, obs):
                cand = [start, c, goal]
                if best is None or path_length(cand) < path_length(best):
                    best = cand
        if best is None:
            for c1 in corners:
                if self.blocked(start, c1, obs):
                    continue
                for c2 in corners:
                    if c2 == c1 or self.blocked(c1, c2, obs) or self.blocked(c2, goal, obs):
                        continue
                    cand = [start, c1, c2, goal]
                    if best is None or path_length(cand) < path_length(best):
                        best = cand
        if best is None:
            return [Waypoint(*goal)]          # give up gracefully: go direct
        return [Waypoint(x, y) for x, y in best[1:]]

    def plan_surface(self, start: tuple[float, float], start_span: tuple[float, float] | None,
                     goal: tuple[float, float], obstacles: list[Box], scale: float = 1.0,
                     inside: Callable[[float, float], bool] | None = None) -> list[Waypoint]:
        """Route between standing surfaces at different heights.

        start_span: (left, right) of the surface we stand on (None = floor).
        Going up: walk to a launch point beside the goal, leap onto it, walk.
        Going down: walk to the end of our surface, hop off, walk.
        """
        sx, sy = start
        gx, gy = goal
        dy = gy - sy
        if abs(dy) < 8 * scale:
            return self.plan_free(start, goal, obstacles, inside)
        direction = 1 if gx >= sx else -1
        out: list[Waypoint] = []
        if dy < 0:
            # going up: choose a launch spot on our level that has a clear arc to the goal
            reach = min(150 * scale, max(40 * scale, abs(gx - sx)))
            options = [gx - direction * reach, gx + direction * reach]
            lo, hi = (start_span if start_span else (-1e9, 1e9))
            best = None
            for lx in options:
                lx = max(lo, min(hi, lx))
                land = (gx - math.copysign(min(30 * scale, abs(gx - lx)), gx - lx), gy)
                hits = sum(1 for o in obstacles if not o.contains(*land) and
                           segment_intersects_box(lx, sy - 40 * scale, land[0], land[1] - 30 * scale, o))
                cost = hits * 10000 + abs(lx - sx)
                if best is None or cost < best[0]:
                    best = (cost, lx, land)
            _, lx, land = best
            if abs(lx - sx) > 3:
                out.append(Waypoint(lx, sy))
            out.append(Waypoint(land[0], land[1], "leap"))
            if abs(land[0] - gx) > 2:
                out.append(Waypoint(gx, gy))
            return out
        # going down
        if start_span is not None:
            lo, hi = start_span
            edge_x = hi if direction > 0 else lo
            if not (lo <= gx <= hi):
                out.append(Waypoint(edge_x, sy))
                land_x = edge_x + direction * 28 * scale
            else:
                land_x = gx
            out.append(Waypoint(land_x, gy, "leap"))
        else:
            out.append(Waypoint(gx, gy, "leap"))
        if abs(out[-1].x - gx) > 2:
            out.append(Waypoint(gx, gy))
        return out
