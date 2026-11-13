"""Companion care: feed, play, pet - and a bond that grows while you focus.

A small, kind tamagotchi layer. Three needs (0..100) drift down slowly while FocusBuddy runs:

* **food**  - fed with treats (you earn a treat for every 10 focused minutes, more for streaks)
* **fun**   - a play session (fetch, chasing a toy, a little dance ...)
* **love**  - petting it (on the Care page, or rub the cursor over it on the desktop)

They never fall below a floor and never drop while FocusBuddy is closed: a companion that's been
left alone is a bit hungry, not hurt. There is no punishment, only nudges.

**Bond** experience comes from focus (1 per focused minute), care (5 per action, once every few
minutes) and finished session goals; levels unlock titles and little extras (new tricks shown on
the Care page). **Badges** mark focus streaks (3, 7, 14, 30, 60, 100 days) and milestones.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Optional

FLOOR = 20.0                    # needs never fall below this
DECAY_PER_HOUR = {"food": 5.0, "fun": 7.0, "love": 4.0}
CARE_GAIN = {"feed": ("food", 28.0), "play": ("fun", 30.0), "pet": ("love", 18.0)}
CARE_XP = 5
CARE_XP_GAP = 180.0              # seconds between care actions that earn XP (no click-farming)
TREAT_EVERY_MIN = 10
STREAK_BADGES = (3, 7, 14, 30, 60, 100)
LEVELS = (0, 60, 180, 400, 750, 1250, 2000, 3000, 4500, 6500)   # XP needed for level 1..10
TITLES = ("New friend", "Buddy", "Study pal", "Focus partner", "Trusted sidekick", "Best friend",
          "Partner in focus", "Kindred spirit", "Legendary duo", "Soulmates of focus")
UNLOCKS = {2: "Plays fetch / chase on request", 3: "Sits beside you while you read", 4: "A happy dance when you finish",
           5: "Brings you little gifts more often", 6: "Purrs / wags when you return", 7: "Hums while you work",
           8: "Remembers your best focus hour", 9: "A victory lap on big days", 10: "A golden name on the Care page"}
BADGES = {"streak_3": ("🔥", "3-day streak"), "streak_7": ("🌟", "A week of focus"),
          "streak_14": ("💫", "Two-week streak"), "streak_30": ("🏆", "A month of focus"),
          "streak_60": ("👑", "60-day streak"), "streak_100": ("💎", "100 days!"),
          "first_session": ("🌱", "First focus session"), "hours_10": ("⏱", "10 hours of focus"),
          "hours_50": ("🚀", "50 hours of focus"), "goals_10": ("🎯", "10 session goals done"),
          "early_bird": ("🌅", "Focused before 8 am"), "night_owl": ("🦉", "Focused after 10 pm")}


@dataclass
class CareState:
    food: float = 80.0
    fun: float = 70.0
    love: float = 75.0
    xp: int = 0
    treats: int = 3
    badges: list = field(default_factory=list)
    last: float = 0.0                       # last decay update (time.time)
    last_xp_care: float = 0.0
    focus_min_carry: float = 0.0            # focused minutes not yet turned into a treat
    goals_done: int = 0
    streak_rewarded: int = 0                # highest streak badge already given


class CompanionCare:
    def __init__(self, data: Optional[dict] = None, now=time.time):
        self._now = now
        self.s = CareState()
        if isinstance(data, dict):
            for k, v in data.items():
                if hasattr(self.s, k):
                    setattr(self.s, k, v)
        if not self.s.last:
            self.s.last = self._now()
        self.listeners: list = []           # called with (event, data) on level-ups / badges

    # ------------------------------------------------------------ state
    def snapshot(self) -> dict:
        return asdict(self.s)

    def tick(self, running: bool = True) -> None:
        """Decay the needs for the time since the last update (only while FocusBuddy runs)."""
        now = self._now()
        hours = max(0.0, (now - self.s.last) / 3600.0)
        self.s.last = now
        if not running or hours <= 0:
            return
        hours = min(hours, 0.5)              # a long gap (sleep, closed app) doesn't count
        for need, rate in DECAY_PER_HOUR.items():
            v = getattr(self.s, need)
            setattr(self.s, need, max(FLOOR, v - rate * hours))

    @property
    def level(self) -> int:
        lv = 1
        for i, need in enumerate(LEVELS):
            if self.s.xp >= need:
                lv = i + 1
        return lv

    @property
    def title(self) -> str:
        return TITLES[min(len(TITLES), self.level) - 1]

    def level_progress(self) -> tuple[int, int]:
        """(xp into this level, xp needed for the next) - (x, 0) at the top level."""
        lv = self.level
        if lv >= len(LEVELS):
            return self.s.xp - LEVELS[-1], 0
        return self.s.xp - LEVELS[lv - 1], LEVELS[lv] - LEVELS[lv - 1]

    @property
    def happiness(self) -> float:
        return (self.s.food + self.s.fun + self.s.love) / 3.0

    def mood(self) -> str:
        """The emotion the needs suggest (the companion's baseline when nothing else is going on)."""
        s = self.s
        low = min(s.food, s.fun, s.love)
        if low < 32:
            return {s.food: "SAD", s.fun: "BORED", s.love: "LONELY"}[low]
        if self.happiness > 82:
            return "PLAYFUL"
        return "HAPPY"

    def need_line(self) -> Optional[str]:
        """A gentle hint when a need is low, or None."""
        s = self.s
        if s.food < 35:
            return "My tummy's rumbling… a treat would be lovely 🍪" if s.treats else \
                "I'm a little hungry. Ten focused minutes earns a treat!"
        if s.fun < 35:
            return "Play with me for a minute? 🎾"
        if s.love < 35:
            return "A little pat would make my day 💛"
        return None

    # ------------------------------------------------------------ actions
    def can(self, action: str) -> bool:
        return action != "feed" or self.s.treats > 0

    def do(self, action: str) -> dict:
        """feed / play / pet. Returns {"ok", "need", "value", "xp", "level_up"}."""
        if action not in CARE_GAIN:
            raise ValueError(action)
        if not self.can(action):
            return {"ok": False, "reason": "no_treats"}
        need, gain = CARE_GAIN[action]
        if action == "feed":
            self.s.treats -= 1
        v = min(100.0, getattr(self.s, need) + gain)
        setattr(self.s, need, v)
        if action != "pet":                  # every care action is also a bit of love
            self.s.love = min(100.0, self.s.love + 5.0)
        xp = 0
        now = self._now()
        if now - self.s.last_xp_care >= CARE_XP_GAP:
            self.s.last_xp_care = now
            xp = CARE_XP
        lv0 = self.level
        self.s.xp += xp
        up = self.level > lv0
        if up:
            self._emit("level_up", {"level": self.level, "title": self.title, "unlock": UNLOCKS.get(self.level)})
        return {"ok": True, "need": need, "value": v, "xp": xp, "level_up": up}

    # ------------------------------------------------------------ focus rewards
    def on_focus(self, minutes: float, when: Optional[float] = None) -> dict:
        """Focused minutes (a finished or partly-finished session): XP and treats."""
        minutes = max(0.0, float(minutes))
        lv0 = self.level
        self.s.xp += int(round(minutes))
        self.s.focus_min_carry += minutes
        treats = int(self.s.focus_min_carry // TREAT_EVERY_MIN)
        self.s.focus_min_carry -= treats * TREAT_EVERY_MIN
        self.s.treats = min(99, self.s.treats + treats)
        self.s.fun = min(100.0, self.s.fun + 0.3 * minutes)        # it enjoyed working with you
        if "first_session" not in self.s.badges and minutes >= 5:
            self._badge("first_session")
        import datetime as _dt
        hour = _dt.datetime.fromtimestamp(when or self._now()).hour
        if minutes >= 15 and hour < 8:
            self._badge("early_bird")
        if minutes >= 15 and hour >= 22:
            self._badge("night_owl")
        up = self.level > lv0
        if up:
            self._emit("level_up", {"level": self.level, "title": self.title, "unlock": UNLOCKS.get(self.level)})
        return {"treats": treats, "xp": int(round(minutes)), "level_up": up}

    def on_goal_done(self) -> None:
        self.s.goals_done += 1
        self.s.xp += 10
        self.s.treats = min(99, self.s.treats + 1)
        if self.s.goals_done >= 10:
            self._badge("goals_10")

    def on_totals(self, focus_hours: float) -> None:
        if focus_hours >= 10:
            self._badge("hours_10")
        if focus_hours >= 50:
            self._badge("hours_50")

    def on_streak(self, days: int) -> Optional[int]:
        """Streak milestones: a badge and bonus treats once each. Returns the milestone hit, if any."""
        hit = None
        for n in STREAK_BADGES:
            if days >= n and self.s.streak_rewarded < n:
                self.s.streak_rewarded = n
                self._badge(f"streak_{n}")
                self.s.treats = min(99, self.s.treats + max(2, n // 3))
                self.s.xp += 5 * n
                hit = n
        return hit

    def _badge(self, key: str) -> None:
        if key in self.s.badges or key not in BADGES:
            return
        self.s.badges.append(key)
        self._emit("badge", {"key": key, "icon": BADGES[key][0], "name": BADGES[key][1]})

    def _emit(self, event: str, data: dict) -> None:
        for fn in list(self.listeners):
            try:
                fn(event, data)
            except Exception:  # noqa: BLE001 - a listener must never break the care state
                pass
