"""AttentionController: what the companion is looking at.

Several things compete for attention; the highest-priority live target wins:

    USER (scripted "look at you" beat) > DISTRACTION_WINDOW > USER_INTERACTION
    > NOTIFICATION > MOUSE > MOVEMENT_TARGET > ACTIVE_WINDOW > NONE

The chosen point drives the eyes (fast), the head (slower) and, when the
target is behind the companion, a request to turn the body.
"""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass
from typing import Optional


class AttentionTarget(str, enum.Enum):
    NONE = "NONE"
    MOUSE = "MOUSE"
    ACTIVE_WINDOW = "ACTIVE_WINDOW"
    DISTRACTION_WINDOW = "DISTRACTION_WINDOW"
    USER_INTERACTION = "USER_INTERACTION"
    MOVEMENT_TARGET = "MOVEMENT_TARGET"
    NOTIFICATION = "NOTIFICATION"
    USER = "USER"            # look out of the screen, at the person


PRIORITY = {
    AttentionTarget.USER: 110,
    AttentionTarget.DISTRACTION_WINDOW: 100,
    AttentionTarget.USER_INTERACTION: 80,
    AttentionTarget.NOTIFICATION: 60,
    AttentionTarget.MOUSE: 50,
    AttentionTarget.MOVEMENT_TARGET: 40,
    AttentionTarget.ACTIVE_WINDOW: 20,
    AttentionTarget.NONE: 0,
}


@dataclass
class _Entry:
    kind: AttentionTarget
    point: Optional[tuple[float, float]]
    until: Optional[float]


@dataclass
class LookOutput:
    kind: AttentionTarget
    point: Optional[tuple[float, float]]
    look_x: float          # eyes, relative to facing (+ = forward)
    look_y: float
    head_x: float          # head (slower)
    head_y: float
    front: float           # 0..1 face the viewer
    behind: bool           # target is behind the body -> wants to turn
    neck_x: float = 0.0    # neck joins in when the head runs out of range (slower)
    torso_x: float = 0.0   # torso twists last (slowest)
    converge: float = 0.0  # 0..0.8 both eyes turn inward toward a near target
    distance: float = 0.0  # px to the target (0 = none)


def _approach(cur: float, target: float, dt: float, tau: float) -> float:
    return cur + (target - cur) * (1 - math.exp(-dt / max(1e-4, tau)))


class AttentionController:
    EYE_TAU = 0.06
    HEAD_TAU = 0.2
    NECK_TAU = 0.35
    TORSO_TAU = 0.6
    RANGE = 170.0          # px for full eye deflection
    NEAR = 240.0           # px: closer than this the eyes start to converge
    EYE_LIMIT = 1.0        # eye deflection clamp (normalised)
    HEAD_START = 0.25      # head joins when the eyes are past this
    NECK_START = 0.55
    TORSO_START = 0.8

    def __init__(self):
        self._entries: dict[AttentionTarget, _Entry] = {}
        self._now = 0.0
        self.look_x = self.look_y = 0.0
        self.head_x = self.head_y = 0.0
        self.front = 0.0
        self.neck_x = 0.0
        self.torso_x = 0.0
        self.converge = 0.0
        self.idle_glance = (0.0, 0.0)
        self._next_glance = 0.0
        # attention decay: don't stare at the cursor forever - look, look away, look back
        self.gaze_hold = 1.0             # personality scale (dog stares longer, cat looks away sooner)
        self._gaze_kind = None
        self._gaze_t0 = 0.0
        self._away_until = -1.0
        self._away_dir = (0.0, 0.0)
        self._hold = 3.5
        self._rng = random.Random(23)

    def set(self, kind: AttentionTarget | str, point: Optional[tuple[float, float]] = None,
            duration: Optional[float] = None, now: Optional[float] = None) -> None:
        kind = AttentionTarget(kind)
        now = self._now if now is None else now
        until = None if duration is None else now + duration
        old = self._entries.get(kind)
        if old is not None and until is not None and (old.until is None or old.until > until):
            until = old.until          # a short glance never cuts a longer interest short
        self._entries[kind] = _Entry(kind, point, until)

    def clear(self, kind: AttentionTarget | str) -> None:
        self._entries.pop(AttentionTarget(kind), None)

    def current(self, now: Optional[float] = None) -> tuple[AttentionTarget, Optional[tuple[float, float]]]:
        now = self._now if now is None else now
        for k in [k for k, e in self._entries.items() if e.until is not None and now >= e.until]:
            del self._entries[k]
        if not self._entries:
            return AttentionTarget.NONE, None
        best = max(self._entries.values(), key=lambda e: PRIORITY[e.kind])
        return best.kind, best.point

    def update(self, dt: float, now: float, eye_origin: tuple[float, float], facing: float,
               rng=None) -> LookOutput:
        self._now = now
        kind, point = self.current(now)
        front_t = 0.0
        behind = False
        conv_t = 0.0
        dist = 0.0
        decays = kind in (AttentionTarget.MOUSE, AttentionTarget.ACTIVE_WINDOW, AttentionTarget.MOVEMENT_TARGET)
        if kind != self._gaze_kind:
            self._gaze_kind, self._gaze_t0 = kind, now
            self._hold = self._rng.uniform(2.5, 5.0) * self.gaze_hold
        looking_away = False
        if decays and point is not None:
            if now < self._away_until:
                looking_away = True
            elif now - self._gaze_t0 > self._hold:
                self._away_until = now + self._rng.uniform(0.7, 1.8)
                self._away_dir = (self._rng.choice((-0.55, -0.4, 0.5)), self._rng.uniform(-0.3, 0.25))
                self._gaze_t0 = self._away_until
                self._hold = self._rng.uniform(2.5, 5.0) * self.gaze_hold
                looking_away = True
        if looking_away:
            tx, ty = self._away_dir
        elif kind == AttentionTarget.USER:
            tx, ty, front_t = 0.0, 0.0, 1.0
            conv_t = 0.25                      # the user is "close": a gentle, cute inward focus
        elif point is not None:
            dx = point[0] - eye_origin[0]
            dy = point[1] - eye_origin[1]
            dist = math.hypot(dx, dy)
            if dist < self.NEAR:
                conv_t = 0.8 * (1.0 - dist / self.NEAR) ** 1.5
            f = 1.0 if facing >= 0 else -1.0
            rel = dx * f
            tx = max(-1.0, min(1.0, rel / self.RANGE))
            ty = max(-1.0, min(1.0, dy / self.RANGE))
            if rel < -60 and abs(dx) > abs(dy) * 0.6:
                behind = True
                tx = max(tx, -0.55)   # eyes can't roll fully backwards
        else:
            # idle micro-saccades: small random glances every couple of seconds
            if rng is not None and now >= self._next_glance:
                self.idle_glance = (rng.uniform(-0.5, 0.6), rng.uniform(-0.35, 0.25))
                self._next_glance = now + rng.uniform(1.2, 3.5)
            tx, ty = self.idle_glance
        lim = self.EYE_LIMIT
        tx, ty = max(-lim, min(lim, tx)), max(-lim, min(lim, ty))
        self.look_x = _approach(self.look_x, tx, dt, self.EYE_TAU)
        self.look_y = _approach(self.look_y, ty, dt, self.EYE_TAU)
        self.head_x = _approach(self.head_x, tx * 0.6, dt, self.HEAD_TAU)
        self.head_y = _approach(self.head_y, ty * 0.6, dt, self.HEAD_TAU)
        # chain: neck and torso only take part for big look angles
        ax = abs(tx)
        sgn = 1.0 if tx >= 0 else -1.0
        neck_t = sgn * max(0.0, ax - self.NECK_START) / (1 - self.NECK_START)
        torso_t = sgn * max(0.0, ax - self.TORSO_START) / (1 - self.TORSO_START)
        self.neck_x = _approach(self.neck_x, neck_t, dt, self.NECK_TAU)
        self.torso_x = _approach(self.torso_x, torso_t, dt, self.TORSO_TAU)
        self.converge = _approach(self.converge, conv_t, dt, 0.1)
        self.front = _approach(self.front, front_t, dt, 0.12)
        return LookOutput(kind, point, self.look_x, self.look_y, self.head_x, self.head_y, self.front, behind,
                          self.neck_x, self.torso_x, self.converge, dist)
