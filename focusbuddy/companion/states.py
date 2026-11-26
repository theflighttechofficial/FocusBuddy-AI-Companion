"""Companion state machine.

Every visual behaviour of the companion is a state.  Transitions are
validated against an explicit table so impossible jumps (e.g. a sleeping
pet instantly swiping a window closed) cannot happen.
"""

from __future__ import annotations

import enum
from typing import Callable, Optional


class CompanionState(str, enum.Enum):
    IDLE = "IDLE"
    WALKING = "WALKING"
    RUNNING = "RUNNING"
    SITTING = "SITTING"
    SLEEPING = "SLEEPING"
    LOOKING_AROUND = "LOOKING_AROUND"
    STRETCHING = "STRETCHING"
    PLAYING = "PLAYING"
    WAVING = "WAVING"
    WATCHING = "WATCHING"
    ALERT = "ALERT"
    WARNING = "WARNING"
    CHASING = "CHASING"
    CLOSING_DISTRACTION = "CLOSING_DISTRACTION"
    CELEBRATING = "CELEBRATING"
    CONFUSED = "CONFUSED"
    FOCUS_MODE = "FOCUS_MODE"
    HELD = "HELD"          # being dragged by the user
    FALLING = "FALLING"    # dropped after a drag / ground vanished
    # --- added by the animation upgrade ---
    TURNING = "TURNING"    # turning around in place
    JUMPING = "JUMPING"    # airborne (leap onto / off a window, hop)
    LANDING = "LANDING"    # absorbing a landing
    YAWNING = "YAWNING"
    GROOMING = "GROOMING"
    SCRATCHING = "SCRATCHING"
    PEEKING = "PEEKING"    # peeking over a screen / window edge
    WAKING = "WAKING"
    PRESSING = "PRESSING"  # physically pressing a window's close button
    READING = "READING"
    PETTED = "PETTED"


S = CompanionState

# States the companion uses while handling a distraction.
INTERVENTION_STATES = {S.WATCHING, S.ALERT, S.WARNING, S.CHASING, S.CLOSING_DISTRACTION, S.PRESSING}
MOVING_STATES = {S.WALKING, S.RUNNING, S.CHASING, S.TURNING, S.JUMPING}
CALM_STATES = {S.IDLE, S.WALKING, S.SITTING, S.SLEEPING, S.LOOKING_AROUND, S.STRETCHING,
               S.PLAYING, S.WAVING, S.FOCUS_MODE, S.YAWNING, S.GROOMING, S.SCRATCHING, S.PEEKING,
               S.READING, S.PETTED, S.TURNING, S.LANDING, S.WAKING}

_ALL = set(S)

# Allowed transitions: source -> set(targets).
_AWAKE_ONLY = {S.FALLING, S.LANDING}           # only physics may enter these
_NEEDS_TARGET = {S.CLOSING_DISTRACTION}        # must already be dealing with a distraction
TRANSITIONS: dict[S, set[S]] = {}
for _s in S:
    TRANSITIONS[_s] = _ALL - _NEEDS_TARGET - {S.FALLING} - {S.LANDING}
# Those that are already handling a distraction may aim/close.
for _s in (S.WATCHING, S.ALERT, S.WARNING, S.CHASING, S.RUNNING, S.PRESSING, S.TURNING, S.JUMPING,
           S.LANDING, S.WALKING):
    TRANSITIONS[_s] = TRANSITIONS[_s] | {S.CLOSING_DISTRACTION}
# A sleeping companion has to wake up first.
TRANSITIONS[S.SLEEPING] = {S.IDLE, S.ALERT, S.STRETCHING, S.WATCHING, S.CONFUSED, S.HELD, S.SITTING, S.WAVING,
                           S.WAKING, S.YAWNING, S.PETTED, S.FALLING}
TRANSITIONS[S.WALKING] -= {S.SLEEPING}
TRANSITIONS[S.RUNNING] -= {S.SLEEPING, S.STRETCHING}
TRANSITIONS[S.CHASING] -= {S.SLEEPING, S.STRETCHING}
for _s in (S.WATCHING, S.ALERT, S.WARNING, S.JUMPING):
    TRANSITIONS[_s] -= {S.SLEEPING}
TRANSITIONS[S.CLOSING_DISTRACTION] = {S.CELEBRATING, S.CONFUSED, S.IDLE, S.WALKING, S.RUNNING, S.WARNING,
                                      S.WATCHING, S.ALERT, S.HELD, S.FOCUS_MODE, S.PRESSING, S.TURNING,
                                      S.JUMPING, S.CHASING, S.FALLING}
# Being dragged: the only way out is being dropped (or a panic reaction).
TRANSITIONS[S.HELD] = {S.FALLING, S.IDLE, S.CONFUSED, S.HELD}
TRANSITIONS[S.FALLING] = {S.IDLE, S.CONFUSED, S.SITTING, S.HELD, S.WALKING, S.ALERT, S.LANDING}
# Airborne: land, fall or get grabbed.
TRANSITIONS[S.JUMPING] = {S.LANDING, S.FALLING, S.HELD, S.IDLE, S.WALKING, S.RUNNING, S.CHASING, S.WARNING,
                          S.CLOSING_DISTRACTION, S.ALERT, S.WATCHING}
TRANSITIONS[S.LANDING] = TRANSITIONS[S.LANDING] | {S.FALLING}
TRANSITIONS[S.WAKING] -= {S.SLEEPING}
# physics may always drop the companion (e.g. the window it stood on closed)
for _s in S:
    if _s not in (S.HELD,):
        TRANSITIONS[_s] = TRANSITIONS[_s] | {S.FALLING}
TRANSITIONS[S.HELD] = {S.FALLING, S.IDLE, S.CONFUSED, S.HELD}
del _s


class InvalidTransition(ValueError):
    pass


class StateMachine:
    def __init__(self, initial: S = S.IDLE, now: Callable[[], float] | None = None):
        import time
        self._now = now or time.perf_counter
        self.state = initial
        self.previous: Optional[S] = None
        self.entered_at = self._now()
        self._listeners: list[Callable[[S, S], None]] = []

    def can(self, target: S) -> bool:
        return target == self.state or target in TRANSITIONS[self.state]

    def transition(self, target: S, strict: bool = False) -> bool:
        """Move to ``target``.  Returns False (or raises in strict mode) if not allowed."""
        if target == self.state:
            return True
        if target not in TRANSITIONS[self.state]:
            if strict:
                raise InvalidTransition(f"{self.state.value} -> {target.value}")
            return False
        old = self.state
        self.previous, self.state = old, target
        self.entered_at = self._now()
        for cb in list(self._listeners):
            cb(old, target)
        return True

    def path_to(self, target: S) -> list[S]:
        """Shortest allowed path (BFS) - used e.g. to wake up before chasing."""
        if self.can(target):
            return [target]
        from collections import deque
        q = deque([(self.state, [])])
        seen = {self.state}
        while q:
            cur, path = q.popleft()
            for nxt in TRANSITIONS[cur]:
                if nxt in seen:
                    continue
                if nxt == target:
                    return path + [nxt]
                seen.add(nxt)
                q.append((nxt, path + [nxt]))
        return []

    def force_path(self, target: S) -> bool:
        for st in self.path_to(target):
            self.transition(st)
        return self.state == target

    @property
    def time_in_state(self) -> float:
        return self._now() - self.entered_at

    def on_change(self, cb: Callable[[S, S], None]) -> None:
        self._listeners.append(cb)
