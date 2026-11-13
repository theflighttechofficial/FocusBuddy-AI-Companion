"""Focus / break timer (Pomodoro style).  Pure logic with an injectable clock.

Phases: IDLE -> FOCUS <-> PAUSED -> (complete) -> BREAK -> IDLE / next FOCUS
``tick()`` returns the list of events that happened since the last tick.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

from ..utils.clock import SYSTEM_CLOCK, Clock


class Phase(str, enum.Enum):
    IDLE = "idle"
    FOCUS = "focus"
    PAUSED = "paused"
    BREAK = "break"


@dataclass
class FocusEvent:
    kind: str                 # focus_started, focus_paused, focus_resumed, focus_halfway,
                              # focus_completed, focus_cancelled, break_started, break_completed
    minutes: float = 0.0
    data: dict = field(default_factory=dict)


class FocusTimer:
    def __init__(self, clock: Clock = SYSTEM_CLOCK):
        self.clock = clock
        self.phase = Phase.IDLE
        self.planned = 0.0            # seconds
        self.break_planned = 0.0
        self._started_at = 0.0
        self._accum = 0.0             # focused seconds before the current run segment
        self._run_since: Optional[float] = None
        self._break_since = 0.0
        self._halfway_sent = False
        self.completed_count = 0      # completed sessions since app start (for long breaks)
        self._pending: list[FocusEvent] = []

    # ------------------------------------------------------------- queries
    def elapsed(self) -> float:
        if self.phase in (Phase.FOCUS, Phase.PAUSED):
            run = (self.clock.now() - self._run_since) if self._run_since is not None else 0.0
            return min(self.planned, self._accum + run)
        return 0.0

    def remaining(self) -> float:
        if self.phase in (Phase.FOCUS, Phase.PAUSED):
            return max(0.0, self.planned - self.elapsed())
        if self.phase == Phase.BREAK:
            return max(0.0, self.break_planned - (self.clock.now() - self._break_since))
        return 0.0

    def progress(self) -> float:
        if self.phase in (Phase.FOCUS, Phase.PAUSED) and self.planned:
            return self.elapsed() / self.planned
        if self.phase == Phase.BREAK and self.break_planned:
            return 1 - self.remaining() / self.break_planned
        return 0.0

    @property
    def started_at(self) -> float:
        return self._started_at

    @property
    def active(self) -> bool:
        return self.phase in (Phase.FOCUS, Phase.PAUSED)

    # ------------------------------------------------------------ commands
    def start(self, minutes: float) -> bool:
        if self.phase in (Phase.FOCUS, Phase.PAUSED) or minutes <= 0:
            return False
        now = self.clock.now()
        self.phase = Phase.FOCUS
        self.planned = minutes * 60.0
        self._started_at = now
        self._accum = 0.0
        self._run_since = now
        self._halfway_sent = False
        self._pending.append(FocusEvent("focus_started", minutes))
        return True

    def pause(self) -> bool:
        if self.phase != Phase.FOCUS:
            return False
        now = self.clock.now()
        self._accum += now - (self._run_since or now)
        self._run_since = None
        self.phase = Phase.PAUSED
        self._pending.append(FocusEvent("focus_paused", self.remaining() / 60))
        return True

    def resume(self) -> bool:
        if self.phase != Phase.PAUSED:
            return False
        self._run_since = self.clock.now()
        self.phase = Phase.FOCUS
        self._pending.append(FocusEvent("focus_resumed", self.remaining() / 60))
        return True

    def cancel(self) -> bool:
        if self.phase in (Phase.FOCUS, Phase.PAUSED):
            focused = self.elapsed()
            self.phase = Phase.IDLE
            self._run_since = None
            self._pending.append(FocusEvent("focus_cancelled", focused / 60, {"focused_seconds": focused}))
            return True
        if self.phase == Phase.BREAK:
            self.phase = Phase.IDLE
            self._pending.append(FocusEvent("break_completed", 0, {"skipped": True}))
            return True
        return False

    def start_break(self, minutes: float) -> bool:
        if self.phase in (Phase.FOCUS, Phase.PAUSED) or minutes <= 0:
            return False
        self.phase = Phase.BREAK
        self.break_planned = minutes * 60.0
        self._break_since = self.clock.now()
        self._pending.append(FocusEvent("break_started", minutes))
        return True

    def tick(self) -> list[FocusEvent]:
        now = self.clock.now()
        if self.phase == Phase.FOCUS:
            if not self._halfway_sent and self.planned >= 20 * 60 and self.elapsed() >= self.planned / 2:
                self._halfway_sent = True
                self._pending.append(FocusEvent("focus_halfway", self.remaining() / 60))
            if self.elapsed() >= self.planned:
                self.phase = Phase.IDLE
                self._run_since = None
                self.completed_count += 1
                self._pending.append(FocusEvent("focus_completed", self.planned / 60,
                                                {"focused_seconds": self.planned, "count": self.completed_count}))
        elif self.phase == Phase.BREAK:
            if now - self._break_since >= self.break_planned:
                self.phase = Phase.IDLE
                self._pending.append(FocusEvent("break_completed", self.break_planned / 60))
        events, self._pending = self._pending, []
        return events
