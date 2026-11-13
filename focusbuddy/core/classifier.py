"""Context classification hook (Phase 8 foundation).

The detector asks a ``ContextClassifier`` to adjust its verdict using
context the rules cannot see - e.g. YouTube + "Python tutorial" is
probably productive, YouTube + "funny compilation" probably isn't.

``KeywordContextClassifier`` is a deliberately light heuristic (no model,
no network, no inference cost).  ``FeedbackLearner`` remembers what the
user repeatedly allows and softens future scores for that context.
A real ML model can implement ``ContextClassifier`` later and be swapped
in without touching the detector.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

from .activity import Activity
from .rules import Category

VIDEO_SITES = {"youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "twitch.tv", "bilibili.com"}
SHORT_FORM_PATHS = ("/shorts", "/reel", "/reels")          # endless short videos: never "educational"

EDU_WORDS = [
    "tutorial", "lecture", "course", "lesson", "how to", "explained", "documentation", "programming",
    "python", "java", "javascript", "typescript", "react", "algorithm", "data structure", "machine learning",
    "deep learning", "neural network", "math", "calculus", "linear algebra", "statistics", "physics",
    "chemistry", "biology", "exam", "study", "learn", "crash course", "walkthrough", "coding", "interview",
    "leetcode", "sql", "database", "system design", "docker", "kubernetes", "c++", "rust", "golang",
    "nptel", "mit opencourseware", "cs50", "freecodecamp", "khan academy", "gate", "jee", "neet", "lab",
    "assignment", "project", "research", "paper", "conference", "talk", "webinar", "workshop",
]
ENT_WORDS = [
    "funny", "meme", "prank", "trailer", "reaction", "vlog", "#shorts", "shorts", "music video",
    "official video", "gaming", "gameplay", "highlights", "compilation", "asmr", "episode", "season",
    "minecraft", "fortnite", "gta", "roast", "challenge", "unboxing", "tiktok", "satisfying", "lofi",
]


def _has_any(text: str, words: list[str]) -> list[str]:
    hits = []
    for w in words:
        if re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", text):
            hits.append(w)
    return hits


@dataclass
class ContextAdjustment:
    delta: float = 0.0
    reason: str = ""


class ContextClassifier(ABC):
    @abstractmethod
    def adjust(self, activity: Activity, category: Category) -> ContextAdjustment:
        ...


class KeywordContextClassifier(ContextClassifier):
    def adjust(self, activity: Activity, category: Category) -> ContextAdjustment:
        if category != Category.DISTRACTING or not activity.domain:
            return ContextAdjustment()
        base = activity.domain
        if not any(base == v or base.endswith("." + v) for v in VIDEO_SITES):
            return ContextAdjustment()
        path = (activity.path or "").lower()
        if any(path.startswith(p) for p in SHORT_FORM_PATHS):
            return ContextAdjustment(+5.0, "short-form videos")
        title = (activity.title or "").lower()
        if not title:
            return ContextAdjustment()
        edu = _has_any(title, EDU_WORDS)
        ent = _has_any(title, ENT_WORDS)
        if edu and not ent:
            return ContextAdjustment(-25.0, "looks educational")
        if ent and not edu:
            return ContextAdjustment(+5.0, "entertainment content")
        return ContextAdjustment()


class FeedbackLearner(ContextClassifier):
    """Learns from overrides: if the user keeps choosing 'Keep it open' /
    'Ignore' for a site + topic, future scores for that context are softened.
    Only keyword tokens are kept, in memory, never raw titles."""

    def __init__(self, max_bonus: float = 20.0):
        self.allowed = Counter()
        self.max_bonus = max_bonus

    @staticmethod
    def _tokens(activity: Activity) -> set[str]:
        title = (activity.title or "").lower()
        hits = set(_has_any(title, EDU_WORDS))
        return {f"{activity.key}|{h}" for h in hits} | {f"{activity.key}|*"}

    def record_override(self, activity: Activity) -> None:
        for t in self._tokens(activity):
            self.allowed[t] += 1

    def adjust(self, activity: Activity, category: Category) -> ContextAdjustment:
        if category not in (Category.DISTRACTING,):
            return ContextAdjustment()
        best = max((self.allowed.get(t, 0) for t in self._tokens(activity) if not t.endswith("|*")), default=0)
        if best >= 2:
            return ContextAdjustment(-min(self.max_bonus, 6.0 * best), "you usually allow this")
        # a site the user keeps allowing whatever the topic: a smaller, capped softening
        site = self.allowed.get(f"{activity.key}|*", 0)
        if site >= 3:
            return ContextAdjustment(-min(self.max_bonus / 2, 3.0 * site), "you often allow this site")
        return ContextAdjustment()


class CompositeClassifier(ContextClassifier):
    def __init__(self, *parts: ContextClassifier):
        self.parts = list(parts)

    def adjust(self, activity: Activity, category: Category) -> ContextAdjustment:
        total, reasons = 0.0, []
        for p in self.parts:
            a = p.adjust(activity, category)
            if a.delta:
                total += a.delta
                reasons.append(a.reason)
        return ContextAdjustment(total, ", ".join(reasons))
