"""Productivity statistics: activity time, distraction events, recovered time."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from ..storage.database import PREVENTED_ACTIONS, Database, day_key
from ..utils.clock import SYSTEM_CLOCK, Clock
from .activity import Activity
from .detector import Assessment
from .rules import Category

FLUSH_EVERY = 30.0
RECOVERY_WINDOW = 600.0  # productive time within 10 min after a prevented distraction = "recovered"


class StatsTracker:
    def __init__(self, db: Database, clock: Clock = SYSTEM_CLOCK, idle_threshold: float = 300.0):
        self.db = db
        self.clock = clock
        self.idle_threshold = idle_threshold
        self._buf: dict[tuple[str, str, str], float] = defaultdict(float)
        self._hourly: dict[tuple[str, int, str], float] = defaultdict(float)
        self._recovered: dict[str, float] = defaultdict(float)
        self._last_flush = clock.now()
        self._recovery_until = 0.0
        self._distraction_today_base: Optional[float] = None
        self._base_day = ""
        self.session_id: int = -1
        self.session_distractions = 0
        self.session_distraction_seconds = 0.0

    # ---------------------------------------------------------------- ticks
    def add_tick(self, activity: Optional[Activity], assessment: Optional[Assessment], dt: float) -> None:
        if activity is None or assessment is None or activity.is_self or dt <= 0:
            self._maybe_flush()
            return
        dt = min(dt, 5.0)  # sleep / suspend gaps don't count
        now = self.clock.now()
        cat = assessment.category
        if assessment.whitelisted and cat != Category.PRODUCTIVE:
            cat = Category.NEUTRAL
        distracting = cat in (Category.DISTRACTING, Category.BLOCKED)
        # Idle in front of a productive/neutral app isn't focus; idle in front of a video still counts.
        if activity.idle_seconds >= self.idle_threshold and not distracting:
            self._maybe_flush()
            return
        day = day_key(now)
        self._buf[(day, cat.value, activity.key)] += dt
        import datetime as _dt
        self._hourly[(day, _dt.datetime.fromtimestamp(now).hour, cat.value)] += dt
        if distracting and self.session_id >= 0:
            self.session_distraction_seconds += dt
        if cat == Category.PRODUCTIVE and now < self._recovery_until:
            self._recovered[day] += dt
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        if self.clock.now() - self._last_flush >= FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        self._last_flush = self.clock.now()
        if self._hourly:
            hourly, self._hourly = dict(self._hourly), defaultdict(float)
            self.db.add_activity_hourly(hourly)
        if self._buf:
            items, self._buf = dict(self._buf), defaultdict(float)
            self.db.add_activity_time(items)
            # keep the cached "distraction today" in sync
            if self._distraction_today_base is not None:
                for (day, cat, _), secs in items.items():
                    if day == self._base_day and cat in ("distracting", "blocked"):
                        self._distraction_today_base += secs
        for day, secs in list(self._recovered.items()):
            if secs:
                self.db.bump_stats(day, recovered_seconds=secs)
        self._recovered.clear()

    def distraction_seconds_today(self) -> float:
        today = day_key(self.clock.now())
        if self._distraction_today_base is None or self._base_day != today:
            self._base_day = today
            self._distraction_today_base = self.db.day_summary(today)["distraction_seconds"]
        pending = sum(s for (d, c, _), s in self._buf.items() if d == today and c in ("distracting", "blocked"))
        return self._distraction_today_base + pending

    # -------------------------------------------------------------- episodes
    def record_episode(self, ep, now: float) -> None:
        self.db.add_distraction_event(
            timestamp=ep.started_at, application=ep.activity.app_name,
            website=ep.activity.domain, duration=ep.duration(now), score=ep.max_score,
            level=int(ep.level), action=ep.action, focus_session=ep.focus_session)
        day = day_key(now)
        prevented = ep.action in PREVENTED_ACTIONS and (ep.action != "left" or int(ep.level) >= 2)
        if prevented:
            self.db.bump_stats(day, prevented=1)
            self._recovery_until = now + RECOVERY_WINDOW
        if self.session_id >= 0 and int(ep.level) >= 1:
            self.session_distractions += 1

    # --------------------------------------------------------------- session
    def session_started(self, started_at: float, planned_seconds: int) -> None:
        self.session_id = self.db.start_session(started_at, planned_seconds)
        self.session_distractions = 0
        self.session_distraction_seconds = 0.0

    def session_finished(self, focused_seconds: float, status: str) -> None:
        if self.session_id < 0:
            return
        real_focus = max(0.0, focused_seconds - self.session_distraction_seconds)
        self.db.finish_session(self.session_id, self.clock.now(), int(real_focus), self.session_distractions, status)
        self.session_id = -1

    # ----------------------------------------------------------------- reads
    def today(self) -> dict:
        self.flush()
        return self.db.day_summary(day_key(self.clock.now()))
