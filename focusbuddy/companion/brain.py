"""Companion brain: energy, time of day, context and controlled randomness.

    Brain -> (Personality, Context, Emotion, Energy, Time of day) -> ActionSelector -> plan

* ``EnergyModel``   0..100.  Running, jumping and playing cost energy; sitting,
                    loafing and sleeping restore it; it drifts back toward a
                    baseline on its own.  Low energy -> more sitting, yawning
                    and naps, slower walking; high energy -> more play.
* ``time_of_day``   morning is livelier, late evening / night sleepier (subtle,
                    can be switched off).
* ``ActionSelector`` weighted random choice: personality weights x energy x
                    context (focus = quiet study partner, break = playful) x
                    time of day, with cooldowns (jump 30 s, dance 2 min, sleep
                    5 min ...) and a short memory that avoids repeating itself.
* Surprise events   rare species-specific moments (a cat hunting an invisible
                    bug, a dog chasing its tail, a robot glitching ...).

Pure Python, deterministic with a seeded RNG -> unit tested.
"""

from __future__ import annotations

import math
import random
import os
import time as _time
from typing import Optional

from .actions import ACTIONS, resolve

# idle "kinds" handled by Behavior._decide, plus "act:<ACTION>" for library actions
REST = {"sit", "nap", "yawn", "read", "act:LOAF", "act:LIE_DOWN", "act:DOWN", "act:CHARGE", "act:READ",
        "act:WRITE", "act:THINK", "act:SIT"}
ACTIVE = {"walk", "run", "play", "explore", "other_monitor", "peek", "act:JUMP", "act:HOP", "act:BOUNCE",
          "act:SPIN", "act:DANCE", "act:ROLL", "act:POUNCE", "act:DOUBLE_HOP", "act:PLAY_BOW", "act:PLAY",
          "act:SHAKE", "act:PAW_AT_OBJECT"}

COOLDOWNS = {"nap": 300.0, "run": 20.0, "play": 25.0, "explore": 12.0, "peek": 40.0, "other_monitor": 300.0,
             "stretch": 40.0, "yawn": 45.0, "scratch": 40.0, "groom": 40.0, "surprise": 600.0, "curious_cursor": 90.0}


def time_of_day(hour: Optional[float] = None) -> str:
    if hour is None:
        fake = os.environ.get("FOCUSBUDDY_FAKE_HOUR")          # tests / demos: deterministic time of day
        if fake:
            hour = float(fake)
        else:
            lt = _time.localtime()
            hour = lt.tm_hour + lt.tm_min / 60.0
    if 6 <= hour < 11:
        return "morning"
    if 11 <= hour < 17:
        return "day"
    if 17 <= hour < 22:
        return "evening"
    return "night"


TOD_ENERGY = {"morning": 10.0, "day": 0.0, "evening": -8.0, "night": -20.0}


class EnergyModel:
    def __init__(self, baseline: float = 60.0):
        self.base = max(10.0, min(95.0, baseline))
        self.level = self.base
        self.offset = 0.0           # settings (Low / Normal / High) + time of day

    @property
    def target(self) -> float:
        return max(5.0, min(100.0, self.base + self.offset))

    def spend(self, amount: float) -> None:
        self.level = max(0.0, min(100.0, self.level - amount))

    def update(self, dt: float, state_name: str, gait_level: float, moving: bool) -> float:
        rate = 0.0
        if moving:
            rate -= 0.12 + 0.55 * max(0.0, gait_level) ** 1.5
        elif state_name in ("SLEEPING",):
            rate += 2.5
        elif state_name in ("SITTING", "FOCUS_MODE", "READING", "GROOMING"):
            rate += 0.35
        elif state_name in ("PLAYING", "CELEBRATING"):
            rate -= 0.6
        else:
            rate += 0.12
        rate += 0.02 * (self.target - self.level)     # drift back toward the baseline
        self.level = max(0.0, min(100.0, self.level + rate * dt))
        return self.level

    @property
    def mood(self) -> str:
        e = self.level
        return "tired" if e < 25 else "low" if e < 45 else "high" if e > 75 else "normal"


SURPRISES = {
    "cat": "bug_hunt", "dog": "tail_chase", "fox": "stealth", "rabbit": "binky",
    "robot": "glitch", "buddy": "eureka",
}


class ActionSelector:
    def __init__(self, rng: random.Random, species: str = "cat"):
        self.rng = rng
        self.species = species
        self.cooldown_until: dict[str, float] = {}
        self.recent: list[str] = []
        self.surprises = True
        self.surprise_chance = 0.02
        self.use_time_of_day = True

    def note(self, name: str, now: float) -> None:
        cd = COOLDOWNS.get(name)
        if cd:
            self.cooldown_until[name] = now + cd
        self.recent.append(name)
        del self.recent[:-4]

    def cooling(self, name: str, now: float) -> bool:
        return now < self.cooldown_until.get(name, -1e9)

    def library_weights(self, context: str) -> dict[str, float]:
        """Weights for action-library actions this species can do in the given context."""
        out: dict[str, float] = {}
        tag = {"focus": "focus", "break": "break"}.get(context, "idle")
        for a in ACTIONS.values():
            if a.weight <= 0:
                continue
            tags = a.tags
            if context == "focus":
                ok = "focus" in tags
            elif context == "break":
                ok = "break" in tags or "play" in tags
            else:
                ok = "idle" in tags or "play" in tags or "explore" in tags
            if not ok:
                continue
            if a.species is not None and self.species not in a.species:
                continue
            w = a.weight * (0.6 if tag == "idle" else 1.0)
            out[f"act:{a.name}"] = w
        return out

    def weights(self, base: dict[str, float], *, context: str, energy: float, now: float,
                tod: str | None = None, playfulness: float = 0.5, action_ready=None,
                mood: float = 0.0) -> list[tuple[str, float]]:
        """``mood`` -1..1 (MoodModel.norm): good mood -> a little more play, low mood -> more sitting."""
        e = max(0.0, min(1.0, energy / 100.0))
        md = max(-1.0, min(1.0, mood))
        w = dict(base)
        for k, v in self.library_weights(context).items():
            w[k] = w.get(k, 0.0) + v
        tod = tod if (tod and self.use_time_of_day) else ("day" if not self.use_time_of_day else (tod or "day"))
        for k in list(w):
            v = w[k]
            if k in REST:
                v *= 1.6 - e
                if k == "nap":
                    v *= (1.25 - e) ** 2 * 1.6
            elif k in ACTIVE:
                v *= 0.3 + 1.4 * e
                if k.startswith("act:") and k[4:] in ("JUMP", "HOP", "BOUNCE", "SPIN", "DANCE", "ROLL", "POUNCE",
                                                      "DOUBLE_HOP", "PLAY_BOW", "PLAY"):
                    v *= 0.6 + playfulness
            if md:
                if k in ACTIVE:
                    v *= 1.0 + 0.35 * md
                elif k in REST:
                    v *= 1.0 - 0.3 * md
            if tod == "morning" and k in ACTIVE:
                v *= 1.2
            elif tod == "night":
                v *= 1.5 if k in REST else 0.7 if k in ACTIVE else 1.0
                if k == "nap":
                    v *= 1.6
            elif tod == "evening" and k in REST:
                v *= 1.15
            if self.cooling(k, now):
                v = 0.0
            if k.startswith("act:") and action_ready is not None and not action_ready(k[4:], now):
                v = 0.0
            if self.recent and k == self.recent[-1] and k not in ("walk",):
                v *= 0.35            # avoid doing the same thing twice in a row
            w[k] = v
        return [(k, v) for k, v in w.items() if v > 0]

    def choose(self, options: list[tuple[str, float]]) -> str:
        total = sum(v for _, v in options)
        if total <= 0:
            return "idle"
        pick = self.rng.uniform(0, total)
        for name, v in options:
            pick -= v
            if pick <= 0:
                return name
        return options[-1][0]

    def maybe_surprise(self, now: float, context: str, curiosity: float = 0.5) -> Optional[str]:
        if not self.surprises or context == "focus" or self.cooling("surprise", now):
            return None
        if self.rng.random() < self.surprise_chance * (0.5 + curiosity):
            self.note("surprise", now)
            return SURPRISES.get(self.species, "binky")
        return None


def can_do(name: str, species: str) -> bool:
    return resolve(name, species) is not None


# ---------------------------------------------------------------------- mood
MOOD_EVENTS = {
    "petted": 10.0, "focus_complete": 20.0, "user_returned": 10.0, "played": 4.0, "celebrated": 6.0,
    "ignored": -5.0, "repeated_clicks": -10.0, "long_inactivity": -5.0, "woken": -6.0, "override_denied": -4.0,
    "dropped_hard": -4.0, "greeted": 6.0,
}


class MoodModel:
    """Overall mood -100 (gloomy) .. +100 (delighted). Events nudge it; it drifts back slowly.

    Mood only *biases* behaviour (more playing when happy, more sitting / sulking when low) -
    the effect is deliberately subtle."""

    def __init__(self, baseline: float = 25.0):         # a cheerful companion: mood rests on the happy side
        self.baseline = baseline
        self.value = baseline
        self.log: list[tuple[float, str, float]] = []

    def event(self, name: str, now: float = 0.0, amount: float | None = None) -> float:
        d = MOOD_EVENTS.get(name, 0.0) if amount is None else amount
        self.value = max(-100.0, min(100.0, self.value + d))
        self.log.append((now, name, d))
        del self.log[:-20]
        return self.value

    def update(self, dt: float) -> None:
        # half-life of a few minutes back toward the baseline
        self.value += (self.baseline - self.value) * (1 - math.exp(-dt / 240.0))

    @property
    def norm(self) -> float:
        return self.value / 100.0

    @property
    def label(self) -> str:
        v = self.value
        return "gloomy" if v < -50 else "low" if v < -15 else "great" if v > 50 else "good" if v > 15 else "okay"

    def state(self, energy: float = 50.0, annoyed: bool = False, idle_seconds: float = 0.0,
              curious: bool = False) -> str:
        """Short-term mood category: HAPPY CALM CURIOUS BORED ANNOYED EXCITED SLEEPY."""
        v = self.value
        if annoyed or v < -40:
            return "ANNOYED"
        if energy < 22:
            return "SLEEPY"
        if idle_seconds > 120 and v < 30:
            return "BORED"
        if v > 45 and energy > 65:
            return "EXCITED"
        if curious:
            return "CURIOUS"
        if v > 20:
            return "HAPPY"
        return "CALM"


# how a mood category nudges idle choices (multipliers; subtle)
MOOD_BIAS = {
    "HAPPY": {"play": 1.25, "walk": 1.1, "act:DANCE": 1.2, "nap": 0.85},
    "CALM": {"sit": 1.15, "look": 1.1},
    "CURIOUS": {"explore": 1.4, "look": 1.3, "peek": 1.3, "act:SNIFF": 1.4, "act:LISTEN": 1.4},
    "BORED": {"act:BORED": 1.5, "sit": 1.2, "yawn": 1.5, "play": 0.8},
    "ANNOYED": {"sit": 1.2, "play": 0.6, "watch_user": 0.6, "act:GRUMPY": 2.0},
    "EXCITED": {"run": 1.5, "play": 1.6, "act:JUMP": 1.4, "act:BOUNCE": 1.4, "nap": 0.5},
    "SLEEPY": {"nap": 1.8, "sit": 1.3, "yawn": 1.8, "run": 0.4, "play": 0.5},
}


# --------------------------------------------------------------- short-term memory
class ShortTermMemory:
    """Tiny, local, in-memory recollection of the last few minutes (never written to disk).

    It remembers *interactions with the companion* (clicks, petting, warnings it gave),
    not what the user types or reads."""

    WINDOW = 600.0

    def __init__(self):
        self.last_interaction = -1e9
        self.last_click = -1e9
        self.clicks: list[float] = []
        self.recent_actions: list[tuple[float, str]] = []
        self.last_emotion = "NEUTRAL"
        self.recent_distraction: tuple[float, str] | None = None
        self.ignored_warnings = 0
        self.last_warning = -1e9
        self.annoyed_until = -1e9

    def click(self, now: float) -> int:
        self.clicks = [t for t in self.clicks if now - t < 3.0] + [now]
        self.last_click = self.last_interaction = now
        return len(self.clicks)

    def interact(self, now: float) -> None:
        self.last_interaction = now

    def action(self, now: float, name: str) -> None:
        self.recent_actions.append((now, name))
        self.recent_actions = [(t, n) for t, n in self.recent_actions if now - t < self.WINDOW][-12:]

    def did_recently(self, name: str, now: float, within: float) -> bool:
        return any(n == name and now - t < within for t, n in self.recent_actions)

    def warned(self, now: float, target: str) -> int:
        """A warning was shown; returns how many in a row were ignored (same target, close together)."""
        if self.recent_distraction and self.recent_distraction[1] == target and now - self.last_warning < 900:
            self.ignored_warnings += 1
        else:
            self.ignored_warnings = 0
        self.recent_distraction = (now, target)
        self.last_warning = now
        return self.ignored_warnings

    def resolved(self) -> None:
        self.ignored_warnings = 0

    def summary(self, now: float) -> dict:
        return {"since_interaction": round(now - self.last_interaction, 1), "clicks": len(self.clicks),
                "ignored_warnings": self.ignored_warnings, "last_emotion": self.last_emotion,
                "recent": [n for _, n in self.recent_actions[-4:]]}
