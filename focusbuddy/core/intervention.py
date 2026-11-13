"""Gradual response system.

    Level 1  NOTICE        companion stops and looks at the distraction
    Level 2  WARNING       walks to the window: [Return to Focus] [Give Me 2 Minutes] [Ignore Once]
    Level 3  INTERVENTION  runs to it:          [Close It] [Keep It Open] [Give Me 2 Minutes]
    Level 4  COUNTDOWN     with automatic blocking + an auto-close rule, or during a focus session
                           with focus enforcement on (any distracting / blocked target):
                           "Closing in 10..."   [Keep Open] [Close Now] [Give Me 5 Minutes]

Escalation is driven by the distraction score band and limited to one
level per ``escalation_spacing_seconds``.  The user can always override.
The engine is UI-agnostic; it talks to an ``InterventionView``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from ..utils.clock import SYSTEM_CLOCK, Clock
from ..utils.logging_setup import get_logger, kv
from .activity import Activity
from .detector import Assessment
from .rules import Category
from .whitelist import TemporaryWhitelist

log = get_logger("intervention")

AWAY_GRACE = 3.0  # seconds away before a distraction episode counts as finished
BLOCK_COOLDOWN = 30.0  # after blocking a target, don't start a new episode for it right away


class Level(enum.IntEnum):
    NONE = 0
    NOTICE = 1
    WARNING = 2
    INTERVENTION = 3
    COUNTDOWN = 4


@dataclass
class Episode:
    key: str
    activity: Activity
    assessment: Assessment
    started_at: float
    focus_session: bool = False
    level: Level = Level.NONE
    level_at: float = 0.0
    max_score: int = 0
    action: str = "none"
    ignored: bool = False
    again: bool = False
    away_since: Optional[float] = None
    countdown_until: Optional[float] = None
    countdown_total: float = 0.0
    block_pending: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def display(self) -> str:
        return self.activity.display

    def duration(self, now: float) -> float:
        return max(0.0, now - self.started_at)


class InterventionView(Protocol):
    def show_notice(self, ep: Episode) -> None: ...
    def show_warning(self, ep: Episode) -> None: ...
    def show_intervention(self, ep: Episode) -> None: ...
    def show_countdown(self, ep: Episode, remaining: float, total: float, first: bool) -> None: ...
    def show_blocked(self, ep: Episode, result) -> None: ...
    def show_block_failed(self, ep: Episode, result) -> None: ...
    def show_returned(self, ep: Episode, celebrate: bool) -> None: ...
    def show_snoozed(self, ep: Episode, minutes: int) -> None: ...
    def show_ignored(self, ep: Episode, kind: str) -> None: ...
    def show_return_to_focus(self, ep: Episode, ok: bool) -> None: ...
    def clear(self) -> None: ...
    # optional:
    #   request_block(ep, do_block)      animate (e.g. press the close button), then call do_block()
    #   show_override_denied(ep)         the user has no extra time left for this site


class InterventionEngine:
    def __init__(self, settings, whitelist: TemporaryWhitelist, blocker, view: InterventionView,
                 clock: Clock = SYSTEM_CLOCK, recorder: Callable[[Episode, float], None] | None = None,
                 return_to_focus: Callable[[Episode], bool] | None = None, learner=None):
        self.settings = settings
        self.whitelist = whitelist
        self.blocker = blocker
        self.view = view
        self.clock = clock
        self.recorder = recorder or (lambda ep, now: None)
        self.return_to_focus_cb = return_to_focus or (lambda ep: False)
        self.learner = learner
        self.episode: Optional[Episode] = None
        self._cooldown: dict[str, float] = {}
        self._overrides: dict[str, int] = {}
        self._scope: Optional[str] = None

    # ------------------------------------------------------- extra-time policy
    OVERRIDE_ACTIONS = ("snooze_short", "snooze_long", "ignore", "keep_open")

    def set_scope(self, scope: Optional[str]) -> None:
        """A focus session (or a new day) starts a fresh extra-time allowance."""
        if scope != self._scope:
            self._scope = scope
            self._overrides.clear()

    def _check_scope(self) -> None:
        import datetime as _dt
        if self._scope is None or self._scope.startswith("day:"):
            day = "day:" + _dt.date.fromtimestamp(self.clock.now()).isoformat()
            self.set_scope(day)

    def overrides_left(self, key: str) -> Optional[int]:
        """Remaining 'extra time' (snooze / ignore / keep open) for ``key``. None = unlimited."""
        self._check_scope()
        policy = self.settings.get("override_policy", "once_per_site")
        if policy == "unlimited":
            return None
        if policy == "never":
            return 0
        if policy == "once_per_session":
            return max(0, 1 - sum(self._overrides.values()))
        return max(0, 1 - self._overrides.get(key, 0))      # once_per_site

    def can_override(self, key: str) -> bool:
        left = self.overrides_left(key)
        return left is None or left > 0

    def _use_override(self, key: str) -> None:
        self._overrides[key] = self._overrides.get(key, 0) + 1

    # ------------------------------------------------------------------ policy
    def enforced(self, ep: Episode) -> bool:
        """A focus session in which distractions are closed automatically (the user consented by
        starting the session with focus enforcement on)."""
        if not ep.focus_session or self.settings.get("focus_enforcement", "auto_close") not in ("auto_close", "instant"):
            return False
        rule = ep.assessment.rule
        return rule is not None and rule.category in (Category.DISTRACTING, Category.BLOCKED)

    def _max_level(self, ep: Episode, on_break: bool) -> Level:
        if on_break and self.settings.get("relax_blocking_on_break", True):
            return Level.NOTICE
        if self.enforced(ep):
            return Level.COUNTDOWN
        mode = self.settings.get("blocking_mode", "ask")
        if mode == "warn":
            return Level.WARNING
        if mode == "auto":
            rule = ep.assessment.rule
            if rule is not None and rule.auto_close and rule.category in (Category.DISTRACTING, Category.BLOCKED):
                return Level.COUNTDOWN
        return Level.INTERVENTION

    def _band_level(self, a: Assessment) -> Level:
        return {"mild": Level.NOTICE, "warning": Level.WARNING, "strong": Level.INTERVENTION}.get(a.band, Level.NONE)

    # ------------------------------------------------------------------ update
    def update(self, activity: Optional[Activity], assessment: Optional[Assessment],
               focus_active: bool = False, on_break: bool = False) -> None:
        now = self.clock.now()
        if activity is None or assessment is None or activity.is_self:
            return  # FocusBuddy itself (or nothing) is in front: freeze, don't decide
        ep = self.episode
        distracting = assessment.is_distraction

        if ep is not None and ep.key != assessment.target_key:
            if distracting:
                # hopped from one distraction straight to another: keep some escalation
                inherited = min(ep.level, Level.WARNING)
                self._finish(ep, now, ep.action if ep.action != "none" else "switched", show=False)
                ep = self._start(activity, assessment, now, focus_active)
                ep.level = inherited
                ep.level_at = now if inherited else 0.0
            else:
                if ep.away_since is None:
                    ep.away_since = now
                if now - ep.away_since >= AWAY_GRACE:
                    productive = assessment.category == Category.PRODUCTIVE
                    celebrate = productive and ep.level >= Level.NOTICE
                    if ep.action == "none":
                        action = ("returned" if celebrate else "left") if ep.level >= Level.NOTICE else "none"
                    else:
                        action = ep.action
                    self._finish(ep, now, action, celebrate=celebrate, show=True)
                return

        if ep is None:
            if not distracting or now < self._cooldown.get(assessment.target_key, 0.0):
                return
            ep = self._start(activity, assessment, now, focus_active)

        ep.away_since = None
        ep.activity = activity
        ep.assessment = assessment
        ep.max_score = max(ep.max_score, assessment.score)
        if focus_active:
            ep.focus_session = True
        if not distracting or ep.ignored or ep.block_pending:
            return

        if ep.level == Level.COUNTDOWN:
            remaining = (ep.countdown_until or now) - now
            if remaining <= 0:
                self._execute(ep, automatic=True)
            else:
                self.view.show_countdown(ep, remaining, ep.countdown_total, False)
            return

        cap = self._max_level(ep, on_break)
        band = self._band_level(assessment)
        target = min(band, cap if cap != Level.COUNTDOWN else Level.INTERVENTION)
        # focus enforcement: the countdown follows the warning (no extra "intervention" step)
        trigger = Level.WARNING if (cap == Level.COUNTDOWN and self.enforced(ep)) else Level.INTERVENTION
        if cap == Level.COUNTDOWN and band >= trigger:
            target = Level.COUNTDOWN
        if target <= ep.level:
            return
        spacing = float(self.settings.get("escalation_spacing_seconds", 20))
        if ep.level == Level.NONE:
            if assessment.dwell_seconds < float(self.settings.get("min_distraction_seconds", 5)) and not ep.again:
                return
        elif now - ep.level_at < spacing:
            return
        nxt = Level(ep.level + 1)
        if nxt == Level.INTERVENTION and target == Level.COUNTDOWN:
            nxt = Level.COUNTDOWN
        if target == Level.COUNTDOWN and trigger == Level.WARNING and nxt == Level.WARNING and \
                assessment.category == Category.BLOCKED:
            nxt = Level.COUNTDOWN              # explicitly blocked during a focus session: straight to it
        self._enter(ep, nxt, now)

    # ----------------------------------------------------------------- helpers
    def _start(self, activity: Activity, assessment: Assessment, now: float, focus_active: bool) -> Episode:
        ep = Episode(assessment.target_key, activity, assessment, now, focus_session=focus_active)
        if self.whitelist.recently_expired(ep.key, now):
            ep.again = True
            self.whitelist.consume_expired(ep.key)
        self.episode = ep
        return ep

    def _enter(self, ep: Episode, level: Level, now: float) -> None:
        ep.level = level
        ep.level_at = now
        log.info("Escalation %s", kv(level=level.name, target=ep.key, score=ep.assessment.score,
                                      focus=ep.focus_session or None))
        if level == Level.NOTICE:
            self.view.show_notice(ep)
        elif level == Level.WARNING:
            self.view.show_warning(ep)
        elif level == Level.INTERVENTION:
            self.view.show_intervention(ep)
        elif level == Level.COUNTDOWN:
            total = max(3.0, float(self.settings.get("grace_period_seconds", 10)))
            if self.enforced(ep) and self.settings.get("focus_enforcement") == "instant":
                total = 3.0
            ep.countdown_total = total
            ep.countdown_until = now + total
            self.view.show_countdown(ep, total, total, True)

    def _execute(self, ep: Episode, automatic: bool) -> None:
        if ep.block_pending:
            return
        request = getattr(self.view, "request_block", None)
        if self.enforced(ep) and self.settings.get("focus_enforcement") == "instant":
            request = None                    # no walk over: close right away
        if callable(request):
            # Let the companion physically walk over and press the close button first.
            ep.block_pending = True
            ep.countdown_until = None
            request(ep, lambda: self._do_block(ep, automatic))
        else:
            self._do_block(ep, automatic)

    def _do_block(self, ep: Episode, automatic: bool) -> None:
        now = self.clock.now()
        ep.block_pending = False
        if automatic and self.episode is not ep:
            log.info("Block skipped - user already left %s", kv(target=ep.key))
            return
        result = self.blocker.block(ep.activity, ep.assessment.rule, automatic, enforced=self.enforced(ep))
        if result.ok:
            self._cooldown[ep.key] = now + BLOCK_COOLDOWN
            self._finish(ep, now, result.action, show=False)
            self.view.show_blocked(ep, result)
        else:
            ep.ignored = True
            ep.action = "failed"
            ep.countdown_until = None
            self.view.show_block_failed(ep, result)

    def _finish(self, ep: Episode, now: float, action: str, celebrate: bool = False, show: bool = True) -> None:
        ep.action = action
        if ep.level >= Level.NOTICE or action not in ("none", "left"):
            try:
                self.recorder(ep, now)
            except Exception:  # noqa: BLE001 - stats must not break interventions
                log.exception("recording episode failed")
        log.info("Episode finished %s", kv(target=ep.key, action=action, level=ep.level.name,
                                            duration=int(ep.duration(now))))
        if self.episode is ep:
            self.episode = None
        if show:
            if celebrate:
                self.view.show_returned(ep, True)
            elif ep.level >= Level.NOTICE:
                self.view.clear()

    # ----------------------------------------------------------------- actions
    def handle_action(self, action: str) -> None:
        ep = self.episode
        now = self.clock.now()
        log.info("User selected %s", kv(action=action, target=ep.key if ep else None))
        if ep is None:
            self.view.clear()
            return
        if action in self.OVERRIDE_ACTIONS:
            if not self.can_override(ep.key):
                log.info("Extra time refused %s", kv(target=ep.key, policy=self.settings.get("override_policy")))
                denied = getattr(self.view, "show_override_denied", None)
                if callable(denied):
                    denied(ep)
                return
            self._use_override(ep.key)
        if action == "return":
            ok = self.return_to_focus_cb(ep)
            ep.action = "returned"
            ep.ignored = True
            ep.countdown_until = None
            self.view.show_return_to_focus(ep, ok)
        elif action in ("snooze_short", "snooze_long"):
            minutes = int(self.settings.get("snooze_short_min" if action == "snooze_short" else "snooze_long_min", 2))
            self.whitelist.allow(ep.key, minutes * 60, now)
            log.info("Temporary whitelist created %s", kv(target=ep.key, minutes=minutes))
            self._finish(ep, now, "snoozed", show=False)
            self.view.show_snoozed(ep, minutes)
        elif action in ("ignore", "keep_open"):
            ep.ignored = True
            ep.countdown_until = None
            ep.action = "ignored" if action == "ignore" else "kept_open"
            if self.learner is not None:
                self.learner.record_override(ep.activity)
            self.view.show_ignored(ep, action)
        elif action in ("close", "close_now"):
            self._execute(ep, automatic=False)
        else:
            self.view.clear()

    def reset(self) -> None:
        """Monitoring paused / app closing: end the current episode quietly."""
        ep = self.episode
        if ep is not None:
            self._finish(ep, self.clock.now(), ep.action if ep.action != "none" else "paused", show=False)
            self.view.clear()
