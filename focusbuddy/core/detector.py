"""Distraction detection and scoring (0-100).

Score = base(category)
      + dwell     (grows with continuous time on the same target)
      + focus     (a focus session is running)
      + revisits  (keeps coming back to the same site)
      + switching (rapid app hopping)
      + budget    (daily distraction allowance exceeded)
      + context   (classifier adjustment, e.g. educational video)

Temporarily whitelisted targets ("Give me 2 minutes") always score 0.
Productive targets always score 0.

Bands (configurable):  0-39 normal | 40-69 mild | 70-89 warning | 90-100 strong
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from .activity import Activity
from .classifier import ContextClassifier
from .rules import Category, Rule, RuleSet
from .whitelist import TemporaryWhitelist

BASE = {Category.PRODUCTIVE: 0.0, Category.NEUTRAL: 8.0, Category.DISTRACTING: 35.0, Category.BLOCKED: 55.0}
DWELL_MAX = 47.0
NEUTRAL_DWELL_MAX = 12.0
FOCUS_BONUS = 10.0
REVISIT_STEP, REVISIT_MAX, REVISIT_WINDOW = 4.0, 12.0, 1800.0
SWITCH_WINDOW, SWITCH_FREE, SWITCH_STEP, SWITCH_MAX = 120.0, 6, 2.0, 15.0
BUDGET_BONUS = 10.0
RETURN_GRACE = 30.0   # back on the same target within this long: its time keeps counting (no tab-flip reset)


@dataclass
class Thresholds:
    mild: int = 40
    warning: int = 70
    strong: int = 90

    def band(self, score: float) -> str:
        if score >= self.strong:
            return "strong"
        if score >= self.warning:
            return "warning"
        if score >= self.mild:
            return "mild"
        return "normal"


@dataclass
class Assessment:
    score: int
    category: Category
    band: str
    target_key: str
    rule: Optional[Rule]
    dwell_seconds: float
    whitelisted: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def is_distraction(self) -> bool:
        return self.band != "normal" and self.category != Category.PRODUCTIVE and not self.whitelisted


@dataclass
class DetectorContext:
    focus_active: bool = False
    tau: float = 100.0
    thresholds: Thresholds = field(default_factory=Thresholds)
    distraction_seconds_today: float = 0.0
    allowance_seconds: float = 1800.0
    detect_games: bool = True
    focus_speedup: float = 2.0        # during a focus session distractions build up this much faster


class DistractionDetector(ABC):
    """Interface so an ML detector can replace the rule-based one later."""

    @abstractmethod
    def analyze_activity(self, activity: Activity, ctx: DetectorContext) -> Assessment:
        ...


class ActivityTracker:
    """Dwell time, revisits and switch frequency."""

    def __init__(self):
        self.current_key: Optional[str] = None
        self.since: float = 0.0
        self.visits: deque[tuple[str, float]] = deque(maxlen=400)
        self.switches: deque[float] = deque(maxlen=400)
        self._left: dict[str, tuple[float, float]] = {}      # key -> (time spent, when it was left)

    def observe(self, key: str, now: float) -> None:
        if key != self.current_key:
            if self.current_key is not None:
                self.switches.append(now)
                self._left[self.current_key] = (max(0.0, now - self.since), now)
                if len(self._left) > 64:
                    self._left = dict(sorted(self._left.items(), key=lambda kv: kv[1][1])[-32:])
            self.current_key = key
            # flicking away for a moment (another tab, a glance at the IDE) does not reset the
            # time on a target: back within RETURN_GRACE, the earlier time keeps counting
            spent, left_at = self._left.pop(key, (0.0, -1e18))
            self.since = now - spent if now - left_at <= RETURN_GRACE else now
            self.visits.append((key, now))

    def dwell(self, now: float) -> float:
        return max(0.0, now - self.since) if self.current_key else 0.0

    def revisits(self, key: str, now: float) -> int:
        n = sum(1 for k, t in self.visits if k == key and now - t <= REVISIT_WINDOW)
        return max(0, n - 1)

    def recent_switches(self, now: float) -> int:
        return sum(1 for t in self.switches if now - t <= SWITCH_WINDOW)


class RuleBasedDetector(DistractionDetector):
    def __init__(self, rules: RuleSet, whitelist: TemporaryWhitelist,
                 classifier: Optional[ContextClassifier] = None,
                 classifier_enabled: Callable[[], bool] | None = None):
        self.rules = rules
        self.whitelist = whitelist
        self.classifier = classifier
        self.classifier_enabled = classifier_enabled or (lambda: True)
        self.tracker = ActivityTracker()

    def set_rules(self, rules: RuleSet) -> None:
        self.rules = rules

    def analyze_activity(self, activity: Activity, ctx: DetectorContext) -> Assessment:
        now = activity.timestamp
        key = activity.key
        self.tracker.observe(key, now)
        category, rule = self.rules.classify(activity.exe, activity.domain, activity.path, activity.is_browser)
        dwell = self.tracker.dwell(now)
        reasons: list[str] = []
        if category == Category.NEUTRAL and rule is None and ctx.detect_games and \
                (activity.extra or {}).get("game") and not activity.is_browser:
            # an exe started from a game / launcher install folder
            category, rule = Category.DISTRACTING, Rule("app", activity.exe, Category.DISTRACTING)
            reasons.append("game folder")

        if self.whitelist.is_allowed(key, now):
            return Assessment(0, category, "normal", key, rule, dwell, True, ["temporarily allowed"])
        if category == Category.PRODUCTIVE:
            return Assessment(0, category, "normal", key, rule, dwell, False, ["productive"])

        score = BASE[category]
        reasons.append(f"{category.value} +{BASE[category]:.0f}")
        tau = max(5.0, ctx.tau)
        if ctx.focus_active:
            tau = max(5.0, tau / max(1.0, ctx.focus_speedup))
        if category in (Category.DISTRACTING, Category.BLOCKED):
            d = DWELL_MAX * (1 - math.exp(-dwell / tau))
        else:
            d = NEUTRAL_DWELL_MAX * (1 - math.exp(-dwell / (4 * tau)))
        score += d
        if d >= 1:
            reasons.append(f"time {int(dwell)}s +{d:.0f}")

        if ctx.focus_active:
            fb = FOCUS_BONUS if category != Category.NEUTRAL else 3.0
            score += fb
            reasons.append(f"focus session +{fb:.0f}")

        if category in (Category.DISTRACTING, Category.BLOCKED):
            rv = min(REVISIT_MAX, REVISIT_STEP * self.tracker.revisits(key, now))
            if rv:
                score += rv
                reasons.append(f"revisits +{rv:.0f}")
            if ctx.distraction_seconds_today > ctx.allowance_seconds > 0:
                score += BUDGET_BONUS
                reasons.append(f"daily allowance used +{BUDGET_BONUS:.0f}")

        sw = self.tracker.recent_switches(now)
        if sw > SWITCH_FREE:
            s = min(SWITCH_MAX, SWITCH_STEP * (sw - SWITCH_FREE))
            score += s
            reasons.append(f"app switching +{s:.0f}")

        if self.classifier is not None and self.classifier_enabled():
            adj = self.classifier.adjust(activity, category)
            if adj.delta:
                score += adj.delta
                reasons.append(f"{adj.reason} {adj.delta:+.0f}")

        if category == Category.BLOCKED and dwell >= 1.0:
            # explicitly blocked: no slow ramp - a warning at once, the strongest response in a session
            floor = ctx.thresholds.strong if ctx.focus_active else ctx.thresholds.warning
            if score < floor:
                score = float(floor)
                reasons.append("blocked")
        score_i = int(round(max(0.0, min(100.0, score))))
        return Assessment(score_i, category, ctx.thresholds.band(score_i), key, rule, dwell, False, reasons)
