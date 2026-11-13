"""The context engine: what's going on with you right now, so the companion can react like a friend
who's sitting next to you - not on a timer.

It watches only what FocusBuddy already sees (which app / site is in front and how it's
classified, how long you've been idle, the focus timer, the clock) and turns it into moments:

* ``FLOW``         - 25+ minutes of steady productive work (outside a session): quiet pride
* ``RESTLESS``     - hopping between windows a lot for a few minutes: a gentle "pick one thing?"
* ``MARATHON``     - over 90 minutes at the computer without a real pause: time to stretch
* ``LATE_NIGHT``   - still working after 11 pm: the companion gets sleepy, says so (once a night)
* ``QUICK_RETURN`` - you left a distraction within two minutes: a happy cheer
* ``MILESTONE``    - 1, 2, 3, 4 ... hours of focus today
* ``STREAK_RISK``  - evening, a streak going, no session yet today: an offer to keep it alive
* ``SLUMP``        - a lot of distraction in the last half hour: concern, not scolding

Each moment also sets a *mood* the companion carries for a while (proud, tired, worried, happy),
and notable moments go into the companion's memory (``companion_memory``) for the chat and the
daily report. Typing is never watched: FocusBuddy doesn't record keystrokes.
"""

from __future__ import annotations

import datetime as dt
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

COOLDOWN = {"FLOW": 3600, "RESTLESS": 1500, "MARATHON": 2700, "LATE_NIGHT": 6 * 3600, "QUICK_RETURN": 600,
            "MILESTONE": 0, "STREAK_RISK": 3 * 3600, "SLUMP": 1800}
MOOD = {"FLOW": "PROUD", "RESTLESS": "WORRIED", "MARATHON": "TIRED", "LATE_NIGHT": "SLEEPY",
        "QUICK_RETURN": "HAPPY", "MILESTONE": "EXCITED", "STREAK_RISK": "DETERMINED", "SLUMP": "WORRIED"}


@dataclass
class Moment:
    kind: str
    mood: str
    data: dict = field(default_factory=dict)


class ContextEngine:
    def __init__(self, now: Callable[[], float] | None = None, wall: Callable[[], dt.datetime] | None = None):
        import time
        self._now = now or time.time
        self._wall = wall or dt.datetime.now
        self.switches: deque = deque(maxlen=200)          # timestamps of window changes
        self.recent: deque = deque(maxlen=600)            # (t, category) samples, ~10 min at 1 Hz
        self._last_key = ""
        self.productive_since: Optional[float] = None
        self.active_since: Optional[float] = None
        self.last_fired: dict = {}
        self.distraction_started: Optional[float] = None
        self.focus_today = 0.0
        self._milestones_done: set = set()
        self._day = ""
        self.mood = "HAPPY"
        self.mood_until = 0.0
        self.pending: list[Moment] = []

    # ------------------------------------------------------------ input
    def observe(self, key: str, category: str, idle: float, focus_phase: str = "idle", dt_s: float = 1.0,
                focus_today: float | None = None, streak: int = 0, sessions_today: int = 0) -> list[Moment]:
        """One poll. ``category``: productive | neutral | distracting | blocked. Returns new moments."""
        now = self._now()
        wall = self._wall()
        day = wall.strftime("%Y-%m-%d")
        if day != self._day:
            self._day = day
            self._milestones_done = set()
        out: list[Moment] = []
        away = idle >= 300
        # presence: a real pause (5+ min idle) resets the marathon clock
        if away:
            self.active_since = None
            self.productive_since = None
        elif self.active_since is None:
            self.active_since = now
        if key and key != self._last_key:
            self.switches.append(now)
            self._last_key = key
        self.recent.append((now, category))
        distracting = category in ("distracting", "blocked")
        # distraction episodes: leaving one quickly is worth a cheer
        if distracting and self.distraction_started is None:
            self.distraction_started = now
        elif not distracting and self.distraction_started is not None:
            took = now - self.distraction_started
            self.distraction_started = None
            if category == "productive" and 8 <= took <= 120:
                out += self._fire("QUICK_RETURN", now, seconds=took)
        # flow: steady productive time outside a session
        if category == "productive" and not away:
            if self.productive_since is None:
                self.productive_since = now
            elif now - self.productive_since >= 25 * 60 and focus_phase != "focus":
                out += self._fire("FLOW", now, minutes=(now - self.productive_since) / 60)
        elif category != "neutral":
            self.productive_since = None
        # restless: many switches in the last 3 minutes
        recent_sw = sum(1 for t in self.switches if now - t <= 180)
        if recent_sw >= 24 and not away:
            out += self._fire("RESTLESS", now, switches=recent_sw)
        # marathon
        if self.active_since is not None and now - self.active_since >= 90 * 60:
            out += self._fire("MARATHON", now, minutes=(now - self.active_since) / 60)
        # late night
        if not away and (wall.hour >= 23 or wall.hour < 4):
            out += self._fire("LATE_NIGHT", now, hour=wall.hour)
        # slump: a lot of distraction in the last 10 minutes of samples
        if len(self.recent) >= 300:
            share = sum(1 for _t, c in self.recent if c in ("distracting", "blocked")) / len(self.recent)
            if share >= 0.6 and focus_phase != "focus":
                out += self._fire("SLUMP", now, share=share)
        # milestones of focus today
        if focus_today is not None:
            self.focus_today = focus_today
            for h in range(1, 13):
                if focus_today >= h * 3600 and h not in self._milestones_done:
                    self._milestones_done.add(h)
                    if focus_today < (h + 1) * 3600:          # only the newest one when starting late in the day
                        out += self._fire("MILESTONE", now, hours=h)
        # streak at risk in the evening
        if streak >= 2 and sessions_today == 0 and 19 <= wall.hour < 23 and not away:
            out += self._fire("STREAK_RISK", now, streak=streak)
        for m in out:
            self.mood = m.mood
            self.mood_until = now + 600
        if now > self.mood_until:
            self.mood = "HAPPY"
        self.pending += out
        return out

    def _fire(self, kind: str, now: float, **data) -> list[Moment]:
        cd = COOLDOWN.get(kind, 600)
        last = self.last_fired.get(kind)
        if last is not None and now - last < cd:
            return []
        self.last_fired[kind] = now
        return [Moment(kind, MOOD.get(kind, "HAPPY"), data)]

    # ------------------------------------------------------------ queries
    def switch_rate(self, window: float = 180.0) -> float:
        now = self._now()
        return sum(1 for t in self.switches if now - t <= window) / (window / 60.0)

    def summary(self) -> dict:
        now = self._now()
        return {"mood": self.mood,
                "active_minutes": (now - self.active_since) / 60 if self.active_since else 0.0,
                "flow_minutes": (now - self.productive_since) / 60 if self.productive_since else 0.0,
                "switches_per_min": round(self.switch_rate(), 1),
                "late": self._wall().hour >= 23 or self._wall().hour < 4}


def memory_text(m: Moment) -> Optional[str]:
    """How a moment is remembered (None = not worth remembering)."""
    d = m.data
    if m.kind == "FLOW":
        return f"You were in the zone for {d.get('minutes', 25):.0f} minutes."
    if m.kind == "MILESTONE":
        return f"You reached {d.get('hours')} hour{'s' if d.get('hours', 1) > 1 else ''} of focus."
    if m.kind == "QUICK_RETURN":
        return "You bounced straight back from a distraction."
    if m.kind == "MARATHON":
        return f"A {d.get('minutes', 90):.0f}-minute stretch without a break."
    if m.kind == "LATE_NIGHT":
        return "Working late tonight."
    if m.kind == "SLUMP":
        return "A wobbly half hour with lots of distractions."
    return None
