"""The companion's life: a daily routine, entrances and exits, cheeky gags, the close-button
scene and a little memory of you.

``Behavior`` (behavior.py) moves the body: locomotion, actions, emotions, attention, the
distraction choreography. ``Life`` sits on top of it and decides *what the companion is doing with
its day*::

    observe        focus state, distraction episodes, cursor, user idle time, app lifecycle,
                   clicks / petting / dragging, full-screen apps
    LifeMachine    explicit life states with a validated transition table:
                   SLEEPING -> WAKING -> ENTERING -> ROAMING -> INTERACTING / PLAYING -> ROAMING
                   -> EXITING -> AWAY -> ENTERING ...  plus FOCUSING, DISTRACTION_WARNING,
                   RESTING, DRAGGED_BY_USER and GOODBYE
    GagEngine      a catalogue of behaviours (troublemaker, attention seeker, commentator, study
                   partner, species signatures, routine) - each with an eligibility rule, a
                   priority, a step sequence, a cooldown, interruptibility, emotions and lines -
                   chosen by mood, energy, personality settings, the user's focus and a slowly
                   learned affinity; every decision is logged with its reasons (Playground)
    scenes         wake-up, species entrances / exits, nap, the close-button question, goodbye
    LifeMemory     what it remembers between launches (stored locally in the SQLite settings
                   table, never sent anywhere): last state, mood, energy, entrances / exits,
                   recently used gags and lines, focus sessions and milestones, interactions and
                   ignored waves -> a subtle, reversible affinity

Rules that always hold: distraction handling outranks every antic; during a focus session the
companion only does what the chosen focus behaviour allows; "quiet" stops chatter and
unsolicited interactions; it never steals focus, never makes surprise sounds of its own and hides
for full-screen apps. Pure Python on top of Behavior - unit tested without Qt.
"""

from __future__ import annotations

import enum
import math
import random
import time as _time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from .actions import Priority
from .attention import AttentionTarget as A
from .emotion import Emotion
from .lines import LinePicker, flavor_of
from .states import CompanionState as S


# ===================================================================== life states
class LifeState(str, enum.Enum):
    SLEEPING = "SLEEPING"                  # curled up (launch, nap mode)
    WAKING = "WAKING"
    ENTERING = "ENTERING"                  # coming onto the desktop
    ROAMING = "ROAMING"                    # free life
    INTERACTING = "INTERACTING"            # with the user: clicks, questions, attention, remarks
    PLAYING = "PLAYING"                    # antics
    RESTING = "RESTING"                    # dozing while the user is away / a quiet sit
    FOCUSING = "FOCUSING"                  # study partner during a focus session
    DISTRACTION_WARNING = "DISTRACTION_WARNING"
    DRAGGED_BY_USER = "DRAGGED_BY_USER"
    EXITING = "EXITING"                    # walking off-screen
    AWAY = "AWAY"                          # off-screen for a while (or hidden for a full-screen app)
    GOODBYE = "GOODBYE"                    # the quit / nap farewell


L = LifeState
_AWAKE = {L.ROAMING, L.INTERACTING, L.PLAYING, L.RESTING, L.FOCUSING}
LIFE_TRANSITIONS: dict[LifeState, set[LifeState]] = {
    L.SLEEPING: {L.WAKING, L.DRAGGED_BY_USER, L.GOODBYE},
    L.WAKING: {L.ENTERING, L.ROAMING, L.FOCUSING, L.DISTRACTION_WARNING, L.DRAGGED_BY_USER, L.GOODBYE,
               L.INTERACTING},
    L.ENTERING: {L.ROAMING, L.FOCUSING, L.DISTRACTION_WARNING, L.DRAGGED_BY_USER, L.INTERACTING, L.GOODBYE},
    L.ROAMING: {L.INTERACTING, L.PLAYING, L.RESTING, L.FOCUSING, L.DISTRACTION_WARNING, L.DRAGGED_BY_USER,
                L.EXITING, L.GOODBYE},
    L.INTERACTING: {L.ROAMING, L.PLAYING, L.RESTING, L.FOCUSING, L.DISTRACTION_WARNING, L.DRAGGED_BY_USER,
                    L.EXITING, L.GOODBYE},
    L.PLAYING: {L.ROAMING, L.INTERACTING, L.RESTING, L.FOCUSING, L.DISTRACTION_WARNING, L.DRAGGED_BY_USER,
                L.GOODBYE},
    L.RESTING: {L.ROAMING, L.SLEEPING, L.INTERACTING, L.FOCUSING, L.DISTRACTION_WARNING, L.DRAGGED_BY_USER,
                L.GOODBYE, L.WAKING},
    L.FOCUSING: {L.ROAMING, L.INTERACTING, L.PLAYING, L.RESTING, L.DISTRACTION_WARNING, L.DRAGGED_BY_USER,
                 L.GOODBYE},
    L.DISTRACTION_WARNING: {L.ROAMING, L.FOCUSING, L.INTERACTING, L.DRAGGED_BY_USER, L.GOODBYE},
    L.DRAGGED_BY_USER: {L.ROAMING, L.FOCUSING, L.DISTRACTION_WARNING, L.INTERACTING, L.RESTING},
    L.EXITING: {L.AWAY, L.DISTRACTION_WARNING, L.DRAGGED_BY_USER, L.GOODBYE},
    L.AWAY: {L.ENTERING, L.GOODBYE},
    L.GOODBYE: {L.SLEEPING, L.ROAMING, L.AWAY, L.EXITING},
}


class LifeMachine:
    """Explicit life state with a validated transition table and a short history (Playground)."""

    def __init__(self, state: LifeState = L.SLEEPING, now: Callable[[], float] = _time.monotonic):
        self.state = state
        self._now = now
        self.since = now()
        self.history: deque = deque(maxlen=60)
        self.refused = 0

    def can(self, target: LifeState) -> bool:
        return target == self.state or target in LIFE_TRANSITIONS[self.state]

    def go(self, target: LifeState, reason: str = "", force: bool = False) -> bool:
        if target == self.state:
            return True
        if not self.can(target) and not force:
            self.refused += 1
            self.history.append((self._now(), self.state.value, target.value, f"refused: {reason}"))
            return False
        self.history.append((self._now(), self.state.value, target.value,
                             reason + (" (forced)" if not self.can(target) else "")))
        self.state = target
        self.since = self._now()
        return True

    def elapsed(self) -> float:
        return self._now() - self.since


# ===================================================================== settings & memory
@dataclass
class LifeSettings:
    playfulness: float = 0.8               # calm 0 .. playful .. chaotic 1
    roaming_frequency: float = 0.6         # rare 0 .. frequent 1
    chatter: float = 0.35                  # unsolicited dialogue: quiet 0 .. chatty 1
    focus_behavior: str = "balanced"       # strict | balanced | playful
    roam_area: str = "mostly_edges"        # edges | mostly_edges | desktop
    quiet: bool = False                    # "be quiet": no chatter, no unsolicited interactions
    leave_and_return: bool = True          # wander off-screen now and then
    user_name: str = "friend"


@dataclass
class LifeMemory:
    """Persisted between launches (local SQLite settings table, as a JSON dict)."""
    last_state: str = "SLEEPING"
    mood: float = 0.0
    energy: float = 60.0
    last_entrance: str = ""
    last_exit: str = ""
    gag_counts: dict = field(default_factory=dict)       # favourite antics (how often each ran)
    loved: dict = field(default_factory=dict)            # antics the user reacted to (click / pet right after)
    recent_gags: list = field(default_factory=list)
    recent_lines: list = field(default_factory=list)
    sessions_total: int = 0
    sessions_day: str = ""
    sessions_today: int = 0
    milestones: list = field(default_factory=list)
    interactions: int = 0
    ignored: int = 0
    affinity: float = 0.0                   # -1 (often ignored) .. +1 (often played with)
    last_seen: float = 0.0
    naps: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict], now_wall: Optional[float] = None) -> "LifeMemory":
        m = cls()
        if isinstance(d, dict):
            for k, v in d.items():
                if hasattr(m, k):
                    try:
                        setattr(m, k, type(getattr(m, k))(v) if not isinstance(getattr(m, k), (dict, list))
                                else v)
                    except (TypeError, ValueError):
                        pass
        # affinity drifts back toward neutral while apart (reversible, never a grudge)
        now_wall = _time.time() if now_wall is None else now_wall
        if m.last_seen > 0 and now_wall > m.last_seen:
            days = (now_wall - m.last_seen) / 86400.0
            m.affinity *= 0.97 ** days
        m.affinity = max(-1.0, min(1.0, m.affinity))
        return m


# ===================================================================== gags
@dataclass
class Gag:
    name: str
    category: str                          # trouble | attention | comment | study | species | routine
    build: Callable[["Life"], Optional[list]]
    weight: float = 1.0
    cooldown: float = 300.0
    priority: Priority = Priority.IDLE
    flavors: tuple = ()                    # () = everyone
    contexts: tuple = ("roam",)            # roam | focus | break
    needs: tuple = ()                      # cursor | windows | chat | attention | user_active
    line: str = ""                         # line said when it starts (if chatter allows)
    life_state: LifeState = L.PLAYING
    interruptible: bool = True
    emotions: tuple = ()
    description: str = ""


@dataclass
class Decision:
    t: float
    context: str
    chosen: str
    candidates: list                       # [(name, weight)]
    rejected: list                         # [(name, reason)]
    note: str = ""


class GagEngine:
    """Chooses the next antic: eligibility -> weights -> weighted random, with per-gag cooldowns,
    a short no-repeat memory and a log of why (for the Playground)."""

    def __init__(self, gags: list[Gag], rng: random.Random):
        self.gags = {g.name: g for g in gags}
        self.rng = rng
        self.last_run: dict[str, float] = {}
        self.recent: deque = deque(maxlen=6)
        self.log: deque = deque(maxlen=40)

    def cooling(self, g: Gag, now: float) -> bool:
        return now - self.last_run.get(g.name, -1e12) < g.cooldown

    def note(self, name: str, now: float) -> None:
        self.last_run[name] = now
        self.recent.append(name)

    def choose(self, life: "Life", context: str, now: float, categories: Optional[set] = None,
               force: Optional[str] = None) -> Optional[Gag]:
        cands, rejected = [], []
        for g in self.gags.values():
            if force is not None and g.name != force:
                continue
            why = life.why_not(g, context, now, categories, ignore_cooldown=force is not None)
            if why:
                rejected.append((g.name, why))
                continue
            w = life.gag_weight(g, context)
            if g.name in self.recent:
                w *= 0.25 if g.name != self.recent[-1] else 0.05
            if w <= 0:
                rejected.append((g.name, "weight 0"))
                continue
            cands.append((g, w))
        chosen = None
        if cands:
            total = sum(w for _, w in cands)
            pick = self.rng.uniform(0, total)
            for g, w in cands:
                pick -= w
                if pick <= 0:
                    chosen = g
                    break
            chosen = chosen or cands[-1][0]
        self.log.append(Decision(now, context, chosen.name if chosen else "-",
                                 sorted(((g.name, round(w, 2)) for g, w in cands), key=lambda x: -x[1]),
                                 rejected))
        return chosen


# ===================================================================== the life controller
SPECIES_ENTRANCE = {"cat": "peek_climb", "dog": "trot_in", "rabbit": "hop_in", "fox": "sneak_in",
                    "robot": "descend", "buddy": "walk_wave", "girl": "walk_wave", "saree": "walk_twirl"}
WAKE_FLOURISH = {
    "cat": [("STRETCH", None), ("SHAKE_OFF", None)],                 # stretches and shakes its paws
    "dog": [("STRETCH", None), ("TAIL_WAG", 1.6)],                   # wakes, wags its tail
    "rabbit": [("EAR_TWITCH", None), ("NOSE_TWITCH", None)],         # ears twitch before a hop
    "fox": [("LOOK_AROUND", 1.4), ("PEEK_BEHIND", None)],            # cautious
    "robot": [("BOOT", None), ("SYSTEM_CHECK", None)],               # boots up with blinking lights
    "buddy": [("STRETCH", None), ("SHAKE_HEAD", None), ("WAVE", None)],   # gets up, fixes hair, waves
    "girl": [("STRETCH", None), ("SHAKE_HEAD", None), ("WAVE", None)],
    "saree": [("STRETCH", None), ("SHAKE_HEAD", None), ("WAVE", None)],
}
EXIT_FLOURISH = {"cat": "TAIL_SWISH", "dog": "TAIL_WAG", "rabbit": "EAR_TWITCH", "fox": "SMUG",
                 "robot": "SYSTEM_CHECK", "buddy": "WAVE", "girl": "WAVE", "saree": "WAVE"}


class Life:
    def __init__(self, behavior, *, char_id: str = "cat", seed: Optional[int] = None,
                 now: Optional[Callable[[], float]] = None, memory: Optional[dict] = None):
        self.b = behavior
        self._now = now or behavior._now
        self.rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
        self.char_id = char_id
        self.flavor = flavor_of(char_id, behavior.species)
        self.settings = LifeSettings()
        self.memory = LifeMemory.from_dict(memory)
        self.sm = LifeMachine(L.SLEEPING, self._now)
        self.lines = LinePicker(self.rng, self.memory.recent_lines)
        self.engine = GagEngine(build_catalogue(), self.rng)
        # callbacks (the Qt companion wires these)
        self.say: Callable[[str, Optional[float]], None] = lambda text, secs=None: None
        self.ask: Callable[[str, list], None] = lambda text, buttons: None
        self.hide_bubble: Callable[[], None] = lambda: None
        self.set_shown: Callable[[bool], None] = lambda on: None
        self.on_quit_ready: Optional[Callable[[], None]] = None
        # scene / gag bookkeeping
        self.scene = ""                     # wake | enter | exit | nap | wake_nap | close_ask | goodbye | gag:<name>
        self._cb = None                     # the plan callback of the running scene / gag
        self._scene_gag: Optional[Gag] = None
        self._scene_t0 = 0.0
        self.base = L.ROAMING               # what we go back to after a scene
        self.focus_active = False
        self.focus_started = 0.0
        self.break_active = False
        self.distraction = False
        self.fullscreen = False
        self.hidden_for_fullscreen = False
        self.away_until = 0.0
        self.away_reason = ""
        now_ = self._now()
        self.next_gag_at = now_ + 25.0
        self.roaming_since = now_
        self.awaiting_user_until = 0.0
        self.awaiting_gag = ""
        self.ask_until = 0.0
        self.asking = ""                    # close | nap | stuck | sleepy_focus
        self.active_streak = 0.0            # seconds of continuous user activity (outside focus)
        self.next_break_offer = now_ + 3600.0
        self.next_nap_ask = now_ + 900.0
        self.focus_flags: set = set()
        self.distraction_keys: dict = {}
        self.pending_smug = False
        self.last_gag_end = now_
        self._pending: list = []            # (at, callable)
        self.started = False
        if memory:
            self.load(memory)

    def set_character(self, char_id: str) -> None:
        self.char_id = char_id
        self.flavor = flavor_of(char_id, self.b.species)

    def load(self, memory: Optional[dict]) -> None:
        """Restore the persisted memory (and with it mood / energy)."""
        self.memory = LifeMemory.from_dict(memory)
        self.lines.recent = list(self.memory.recent_lines)
        m = self.memory
        b = self.b
        try:
            b.mood.value = max(-100.0, min(100.0, 0.5 * m.mood + 0.5 * b.mood.baseline))
            gap_h = (_time.time() - m.last_seen) / 3600.0 if m.last_seen else 99.0
            b.energy.level = max(10.0, min(100.0, m.energy if gap_h < 1.0 else
                                           max(m.energy, b.energy.base + (15.0 if gap_h > 6 else 5.0))))
        except (AttributeError, TypeError):
            pass

    def skip_intro(self) -> None:
        """No wake-up scene: straight to free roaming where the companion stands."""
        self.started = True
        self.sm = LifeMachine(L.ROAMING, self._now)
        self.set_shown(True)

    # ----------------------------------------------------------- settings
    def apply_settings(self, **kw) -> None:
        for k, v in kw.items():
            if hasattr(self.settings, k):
                setattr(self.settings, k, v)
        s = self.settings
        b = self.b
        b.idle_activity = max(0.0, min(1.0, s.playfulness))
        b.walking_frequency = max(0.05, min(1.0, s.roaming_frequency))
        b.edge_bias = {"edges": 0.85, "mostly_edges": 0.55, "desktop": 0.1}.get(s.roam_area, 0.55)

    # ----------------------------------------------------------- helpers
    def chat_ok(self, chance: float = 1.0) -> bool:
        """Unsolicited dialogue gate: chatter setting, quiet mode, focus behaviour."""
        s = self.settings
        if s.quiet or s.chatter <= 0.01:
            return False
        if self.focus_active and s.focus_behavior == "strict":
            return False
        if self.focus_active and s.focus_behavior == "balanced":
            chance *= 0.3
        return self.rng.random() < chance * (0.25 + 0.75 * s.chatter)

    def line(self, key: str, **fields) -> str:
        return self.lines.pick(key, self.flavor, name=self.settings.user_name, **fields)

    def speak(self, key: str, chance: float = 1.0, secs: Optional[float] = None, essential: bool = False,
              **fields) -> bool:
        if not essential and not self.chat_ok(chance):
            return False
        text = self.line(key, **fields)
        if text:
            self.say(text, secs)
        self.memory.recent_lines = list(self.lines.recent)
        return bool(text)

    def screen(self) -> tuple[float, float, float, float]:
        return self.b.env.screen_bounds(self.b.x, self.b.y)

    @property
    def size(self) -> float:
        return float(self.b.char_size)

    def floor(self, x: Optional[float] = None) -> float:
        return self.b.floor(x)

    def side_x(self, side: int, where: str) -> float:
        """x near a screen side (-1 left, +1 right): 'out' fully off-screen, 'peek' half hidden,
        'in' just inside, 'rest' a cosy spot in the corner."""
        l, t, r, btm = self.screen()
        edge = l if side < 0 else r
        outward = {"out": 0.9, "peek": 0.15, "bed": -0.4, "in": -1.1, "rest": -0.75}[where] * self.size
        return edge + side * outward

    def nearest_side(self) -> int:
        l, _, r, _ = self.screen()
        return -1 if abs(self.b.x - l) < abs(self.b.x - r) else 1

    def roam_x(self, side: int) -> float:
        l, _, r, _ = self.screen()
        inward = -side
        return (l if side < 0 else r) + inward * self.size * self.rng.uniform(1.8, 3.4)

    def _base(self) -> LifeState:
        return L.FOCUSING if self.focus_active else L.ROAMING

    def busy(self) -> bool:
        return bool(self.scene)

    # ----------------------------------------------------------- scenes
    def _run(self, name: str, steps: list, state: Optional[LifeState], priority: Priority = Priority.INTERACTION,
             mode: str = "directed", done: Optional[Callable[[], None]] = None, gag: Optional[Gag] = None) -> None:
        def cb():
            if self._cb is not cb:
                return
            self._cb = None
            self.scene = ""
            self._scene_gag = None
            self.last_gag_end = self._now()
            if done is not None:
                done()
            else:
                self._back_to_base()
        self.scene = name
        self._scene_gag = gag
        self._scene_t0 = self._now()
        self._cb = cb
        if state is not None:
            self.sm.go(state, name)
        self.b._start_plan(steps, mode=mode, after=cb, priority=priority)

    def _back_to_base(self, release: bool = True) -> None:
        self.sm.go(self._base(), "scene done", force=self.sm.state in (L.GOODBYE, L.AWAY, L.SLEEPING))
        if release and self.b.mode == "directed":
            self.b.release()

    def _cancel_scene(self, reason: str) -> None:
        if self.scene:
            self.sm.history.append((self._now(), self.sm.state.value, self.sm.state.value,
                                    f"scene {self.scene} cancelled: {reason}"))
        self.scene = ""
        self._cb = None
        self._scene_gag = None

    # entrances ------------------------------------------------------------
    def entrance_steps(self, side: int, style: Optional[str] = None, from_peek: bool = False) -> list:
        b = self.b
        y = self.floor(self.side_x(side, "in"))
        style = style or SPECIES_ENTRANCE.get(self.flavor, "walk_wave")
        peek_x, in_x = self.side_x(side, "peek"), self.side_x(side, "in")
        inward = -side
        P = Priority.INTERACTION
        steps: list = []
        if style == "descend":
            # a tiny mechanical landing from above
            l, t, r, btm = self.screen()
            def place():
                b.mover.teleport(in_x, t + self.size * 1.2)
                b.mover.facing = inward
                b.mover.facing_blend = float(inward)
                b.surface = None
            steps += [("call", place), ("call", lambda: self.set_shown(True)), ("leap", in_x, y),
                      ("state", S.LANDING, 0.45), ("action", "BOOT", None, P), ("action", "SCAN", None, P)]
        else:
            if not from_peek:
                steps += [("move", peek_x, y, "walk", False)]
            if style == "peek_climb":
                steps += [("state", S.PEEKING, 1.3), ("state", S.LOOKING_AROUND, 1.3),
                          ("look", A.USER, None, 0.8, S.IDLE), ("move", in_x, y, "walk", False),
                          ("action", "SLOW_BLINK", None, P)]
            elif style == "trot_in":
                steps += [("move", in_x + inward * self.size * 0.6, y, "jog", False), ("action", "TAIL_WAG", 1.4, P),
                          ("action", "GREET", None, P)]
            elif style == "hop_in":
                steps += [("action", "EAR_TWITCH", None, P), ("action", "HOP", None, P),
                          ("move", in_x, y, "walk", False), ("action", "NOSE_TWITCH", None, P)]
            elif style == "sneak_in":
                steps += [("action", "PEEK_BEHIND", None, P), ("state", S.PEEKING, 1.0),
                          ("move", in_x, y, "slow", False), ("action", "SMUG", None, P)]
            elif style == "walk_twirl":
                steps += [("move", in_x, y, "walk", False), ("action", "SPIN", None, P), ("action", "WAVE", None, P)]
            else:   # walk_wave
                steps += [("move", in_x, y, "walk", False), ("action", "WAVE", None, P)]
        steps += [("goto", self.roam_x(side), y, False, S.WALKING), ("state", S.IDLE, 0.4)]
        self.memory.last_entrance = style
        return steps

    def place_offscreen(self, side: int) -> None:
        b = self.b
        x = self.side_x(side, "out")
        b._plan = []
        b._deferred = None
        b.mover.stop()
        b.mover.teleport(x, self.floor(x))
        b.surface = None
        b.mover.facing = -side
        b.mover.facing_blend = float(-side)

    def wake_up(self, side: Optional[int] = None) -> None:
        """Launch: curled up at the screen edge -> wake -> stretch / yawn -> notice you -> enter."""
        b = self.b
        self.started = True
        side = side if side is not None else self.rng.choice((-1, 1))
        bed_x = self.side_x(side, "bed")              # curled up in the corner, just inside the edge
        b.mover.stop()
        b.mover.teleport(bed_x, self.floor(bed_x))
        b.surface = None
        b.mover.facing = -side
        b.mover.facing_blend = float(-side)
        b.set_state(S.SLEEPING)
        self.sm = LifeMachine(L.SLEEPING, self._now)
        self.set_shown(True)
        P = Priority.INTERACTION
        steps: list = [("state", S.SLEEPING, 1.2), ("call", lambda: b.events.append(("wake", 0.0))),
                       ("call", lambda: self.sm.go(L.WAKING, "launch")), ("state", S.WAKING, 1.0)]
        for name, secs in WAKE_FLOURISH.get(self.flavor, WAKE_FLOURISH["cat"]):
            steps.append(("action", name, secs, P))
        steps += [("state", S.YAWNING, 1.5), ("look", A.USER, None, 1.0, S.IDLE),
                  ("emotion", Emotion.HAPPY, 0.8, 3.0),
                  ("call", lambda: self.speak("WAKE", 0.9)),
                  ("call", lambda: self.sm.go(L.ENTERING, "launch"))]
        steps += self.entrance_steps(side, from_peek=True)
        self._run("wake", steps, None)

    def enter(self, side: Optional[int] = None, style: Optional[str] = None, urgent: bool = False) -> None:
        side = side if side is not None else self.rng.choice((-1, 1))
        if not urgent:
            styles = [SPECIES_ENTRANCE.get(self.flavor, "walk_wave"), "walk_wave", "peek_climb"]
            if style is None and self.memory.last_entrance in styles and self.rng.random() < 0.5:
                styles = [s for s in styles if s != self.memory.last_entrance] or styles
                style = self.rng.choice(styles)
        self.place_offscreen(side)
        self.set_shown(True)
        self.sm.go(L.ENTERING, "return" if not urgent else "urgent", force=True)
        self.away_reason = ""
        self.b.autonomous = True
        if urgent:
            return                          # the distraction choreography walks it in
        self.speak("ENTER", 0.5)
        self._run("enter", self.entrance_steps(side, style), None)

    def leave(self, minutes: Optional[float] = None) -> bool:
        """Walk off-screen and stay away for a while (a different entrance on the way back)."""
        if self.focus_active or self.distraction:
            return False
        if self.scene.startswith("gag:"):             # an antic in progress: wrap it up first
            self._cancel_scene("leaving")
            self.sm.go(self._base(), "leaving", force=True)
        if not self.sm.can(L.EXITING):
            return False
        side = self.nearest_side()
        b = self.b
        y = self.floor()
        out_x = self.side_x(side, "out")
        in_x = self.side_x(side, "in")
        flourish = EXIT_FLOURISH.get(self.flavor, "WAVE")
        gait = "slow" if self.flavor == "fox" else "walk"
        steps = [("action", flourish, None, Priority.INTERACTION), ("goto", in_x, y, False, S.WALKING),
                 ("move", out_x, y, gait, False)]
        if minutes is None:
            s = self.settings
            minutes = self.rng.uniform(3.0, 12.0) * (1.4 - 0.8 * s.roaming_frequency)
        self.memory.last_exit = flourish
        self.speak("LEAVE", 0.5)

        def gone():
            self.go_away(minutes * 60.0, "patrol")
        self._run("exit", steps, L.EXITING, done=gone)
        return True

    def go_away(self, seconds: float, reason: str) -> None:
        b = self.b
        b._plan = []
        b._deferred = None
        b.mover.stop()
        b.autonomous = False
        b.mode = "directed"
        self.sm.go(L.AWAY, reason, force=True)
        self.away_until = self._now() + seconds
        self.away_reason = reason
        self.set_shown(False)

    # nap / sleep ------------------------------------------------------------
    def nap(self, announce: bool = True, rest_seconds: Optional[float] = None) -> None:
        """Nap time: a big yawn and a stretch, a sleepy look back at you, then it pads off the nearest
        screen edge to sleep out of sight - until woken (tray / menu / dashboard) or, with
        ``rest_seconds``, until it has slept that long. It comes back yawning from the same side."""
        if self.scene.startswith("gag:"):
            self._cancel_scene("nap")
        b = self.b
        side = self.nearest_side()
        y = self.floor()
        in_x = self.side_x(side, "in")
        out_x = self.side_x(side, "out")
        P = Priority.INTERACTION
        steps = [("state", S.YAWNING, 1.5), ("action", "STRETCH", None, P),
                 ("look", A.USER, None, 0.7, S.IDLE)]
        if self.flavor not in ("robot",):
            steps.append(("action", "WAVE" if self.flavor in ("buddy", "girl", "saree") else "SLOW_BLINK", None, P))
        far = abs(b.x - in_x) > self.size * 5.0           # a long way to the edge: a sleepy trot
        steps += [("goto_gait", in_x, y, "jog") if far else ("goto", in_x, y, False, S.WALKING),
                  ("move", out_x, y, "slow", False)]
        if announce:
            self.speak("CLOSE_NAP" if self.asking == "close" else "NAP_YES", 1.0, essential=True)
        self.memory.naps += 1
        self.nap_side = side
        self.sm.go(L.RESTING, "nap", force=True)

        def gone():
            self.go_away(rest_seconds if rest_seconds is not None else 10.0 ** 9, "nap")
        self._run("nap", steps, None, done=gone)

    def wake_from_nap(self) -> None:
        """Back from a nap: in from the side it left by, still a little sleepy (yawn, stretch, look)."""
        P = Priority.INTERACTION
        if self.sm.state == L.AWAY or self.scene == "nap":
            if self.scene == "nap":                     # still on the way out: just turn round
                self._cancel_scene("woken on the way")
                self.sm.go(self._base(), "woken", force=True)
                self._run("wake_nap", [("state", S.YAWNING, 1.2), ("look", A.USER, None, 0.8, S.IDLE)], None)
                return
            side = getattr(self, "nap_side", 0) or self.nearest_side()
            self.place_offscreen(side)
            self.set_shown(True)
            self.away_reason = ""
            self.away_until = 0.0
            self.b.autonomous = True
            self.sm.go(L.ENTERING, "nap over", force=True)
            y = self.floor(self.side_x(side, "in"))
            steps: list = [("call", lambda: self.b.events.append(("wake", 0.0))),
                           ("move", self.side_x(side, "in") - side * self.size * 0.4, y, "slow", False),
                           ("state", S.YAWNING, 1.4)]
            for name, secs in WAKE_FLOURISH.get(self.flavor, WAKE_FLOURISH["cat"])[:2]:
                steps.append(("action", name, secs, P))
            steps += [("look", A.USER, None, 0.8, S.IDLE), ("call", lambda: self.speak("WAKE", 0.8))]
            self.b.energy.level = min(100.0, self.b.energy.level + 40.0)
            self._run("wake_nap", steps, None)
            return
        steps = [("call", lambda: self.b.events.append(("wake", 0.0))), ("state", S.WAKING, 1.0)]
        for name, secs in WAKE_FLOURISH.get(self.flavor, WAKE_FLOURISH["cat"])[:2]:
            steps.append(("action", name, secs, P))
        steps += [("state", S.YAWNING, 1.3), ("look", A.USER, None, 0.8, S.IDLE)]
        self.sm.go(L.WAKING, "woken", force=True)
        self.b.energy.level = min(100.0, self.b.energy.level + 35.0)
        self._run("wake_nap", steps, None)

    @property
    def napping(self) -> bool:
        """Asleep - off-screen after a nap, on its way there, or (the launch scene) curled up on screen."""
        return (self.sm.state == L.SLEEPING and self.started) or self.scene == "nap" or \
            (self.sm.state == L.AWAY and self.away_reason == "nap")

    # the close button -------------------------------------------------------
    def close_scene(self) -> None:
        """The dashboard is closing: stop, turn to the user and ask what happens next."""
        if self.sm.state in (L.AWAY, L.GOODBYE) or self.distraction:
            return
        if self.sm.state == L.SLEEPING:
            return                          # already napping: let it sleep
        self.asking = "close"
        self.ask_until = self._now() + 25.0
        P = Priority.INTERACTION
        steps = [("look", A.USER, None, 0.6, S.IDLE), ("action", "SURPRISED", None, P),
                 ("look", A.USER, None, 25.0, S.IDLE)]
        self._run("close_ask", steps, L.INTERACTING)
        self.ask(self.line("CLOSE_ASK"), [("life:nap", "Go nap 💤", False), ("life:stay", "Stay here", True),
                                          ("life:quit", "Quit FocusBuddy", False)])

    def answer(self, action: str) -> bool:
        """A bubble button from a life question ("life:<choice>"). Returns True if handled."""
        if not action.startswith("life:"):
            return False
        choice = action[5:]
        asking, self.asking = self.asking, ""
        self.hide_bubble()
        if choice == "nap":
            self.asking = "close" if asking == "close" else ""
            self.nap(announce=True)
            self.asking = ""
        elif choice == "stay":
            P = Priority.INTERACTION
            self.speak("CLOSE_STAY", 1.0, essential=True)
            self._run("stay", [("emotion", Emotion.EXCITED, 1.0, 3.0), ("jump", None, 0.0, 0.0, 1),
                               ("action", "TAIL_WAG" if self.flavor in ("dog", "cat", "fox") else "CHEER", 1.2, P),
                               ("state", S.IDLE, 0.4)], L.INTERACTING)
        elif choice == "quit":
            self.goodbye(self.on_quit_ready)
        elif choice == "nap_no":
            self.speak("NAP_NO", 1.0, essential=True)
            self.b.energy.level = min(100.0, self.b.energy.level + 20.0)
            self._back_to_base()
        elif choice == "hint":
            self.speak("HINT", 1.0, secs=8.0, essential=True)
            self._back_to_base()
        elif choice in ("fine", "dismiss"):
            self._back_to_base()
        elif choice == "focus_nap":
            self.nap(announce=False, rest_seconds=self.rng.uniform(120, 300))
        return True

    def goodbye(self, then: Optional[Callable[[], None]] = None) -> None:
        """Quit: a farewell, then walk off-screen; ``then`` runs when it's gone."""
        b = self.b
        side = self.nearest_side()
        y = self.floor()
        out_x = self.side_x(side, "out")
        P = Priority.SPECIAL_ACTION
        self.distraction = False
        steps = [("look", A.USER, None, 0.6, S.IDLE), ("action", "WAVE", None, P),
                 ("call", lambda: self.speak("GOODBYE", 1.0, secs=2.5, essential=True)),
                 ("action", "FAREWELL", None, P), ("move", out_x, y, "walk", False)]
        self.sm.go(L.GOODBYE, "quit", force=True)

        def gone():
            self.set_shown(False)
            if then is not None:
                then()
        self._run("goodbye", steps, None, priority=P, done=gone)

    # ----------------------------------------------------------- events from the app
    def on_focus(self, kind: str, **data) -> None:
        now = self._now()
        if kind == "focus_started":
            self.focus_active = True
            self.break_active = False
            self.focus_started = now
            self.focus_flags = set()
            if self.sm.state in (L.SLEEPING, L.AWAY):
                self.ensure_present()
            if not self.distraction and not self.scene.startswith("close"):
                self._cancel_scene("focus started")
                self.sm.go(L.FOCUSING, "focus started", force=True)
            self.next_gag_at = now + self._gag_gap()
        elif kind in ("focus_completed", "focus_cancelled"):
            self.focus_active = False
            if self.sm.state == L.FOCUSING:
                self.sm.go(L.ROAMING, kind)
            if kind == "focus_completed":
                self._count_session()
            self.roaming_since = now
        elif kind == "break_started":
            self.break_active = True
        elif kind == "break_completed":
            self.break_active = False

    def _count_session(self) -> None:
        m = self.memory
        today = _time.strftime("%Y-%m-%d")
        if m.sessions_day != today:
            m.sessions_day, m.sessions_today = today, 0
        m.sessions_today += 1
        m.sessions_total += 1
        n = m.sessions_today
        key = None
        if m.sessions_total in (1, 5, 10, 25, 50, 100, 250, 500) and f"total:{m.sessions_total}" not in m.milestones:
            m.milestones.append(f"total:{m.sessions_total}")
            key = ("MILESTONE", m.sessions_total)
        elif n in (2, 3, 5, 8, 12) and f"{today}:{n}" not in m.milestones:
            m.milestones.append(f"{today}:{n}")
            key = ("STREAK", n)
        m.milestones = m.milestones[-60:]
        if key is not None and self.settings.focus_behavior != "strict":
            # after the session celebration has played: a short extra cheer, unique to the character
            self.later(8.5, lambda k=key: self._streak_cheer(*k))

    def _streak_cheer(self, key: str, n: int) -> None:
        if self.distraction or self.scene or self.sm.state not in _AWAKE:
            return
        self.speak(key, 1.0, essential=not self.settings.quiet, n=n)
        g = self.engine.gags.get("celebrate_streak")
        if g is not None:
            self.run_gag(g, note=False)

    def distraction_started(self, key: str = "") -> None:
        """The intervention engine took over: antics stop; the companion must be present."""
        if self.sm.state == L.GOODBYE:
            return
        self.ensure_present()
        self._cancel_scene("distraction")
        self.asking = ""
        self.distraction = True
        self.sm.go(L.DISTRACTION_WARNING, key or "distraction", force=True)
        if key:
            c = self.distraction_keys.get(key, 0) + 1
            self.distraction_keys[key] = c
            if c >= 2:
                self.pending_smug = True       # "called it": the same distraction again

    def distraction_ended(self, returned: bool = False) -> None:
        if not self.distraction:
            return
        self.distraction = False
        self.sm.go(self._base(), "distraction over", force=True)
        if returned and self.pending_smug:
            self.pending_smug = False
            self.later(6.0, lambda: self._maybe_force("smug_predict"))
        self.next_gag_at = max(self.next_gag_at, self._now() + 20.0)

    def _maybe_force(self, name: str) -> None:
        g = self.engine.gags.get(name)
        if g is None or self.scene or self.distraction:
            return
        if self.why_not(g, self.context(), self._now(), None, ignore_cooldown=False):
            return
        self.run_gag(g)

    def ensure_present(self) -> None:
        """Back on the desktop now (a warning needs the companion): from away / hidden / napping."""
        if self.sm.state == L.AWAY:
            self.enter(self.nearest_side(), urgent=True)
        elif self.sm.state == L.SLEEPING:
            self.sm.go(L.WAKING, "needed", force=True)
            self.b.events.append(("wake", 0.0))
        elif self.sm.state == L.EXITING:
            self._cancel_scene("needed")
        self.b.autonomous = True
        self.set_shown(True)

    def on_click(self) -> None:
        self._interacted()
        if self.sm.state == L.SLEEPING and self.started and self.scene in ("", "nap"):
            self._cancel_scene("woken by a click")
            self.wake_from_nap()
        elif self.napping:
            self.wake_from_nap()

    def on_pet(self) -> None:
        self._interacted()
        if self.chat_ok(0.4):
            self.speak("PET_REPLY", 1.0, essential=True)

    def _interacted(self) -> None:
        m = self.memory
        now = self._now()
        m.interactions += 1
        m.affinity = min(1.0, m.affinity + 0.03)
        if self.awaiting_user_until > now:
            # the wave / gift worked: the user reacted
            m.affinity = min(1.0, m.affinity + 0.03)
            if self.awaiting_gag:
                m.loved[self.awaiting_gag] = m.loved.get(self.awaiting_gag, 0) + 1
            self.awaiting_user_until = 0.0
            self.awaiting_gag = ""

    def on_drag(self, held: bool) -> None:
        if held:
            if self.sm.state != L.DRAGGED_BY_USER:
                self._resume = self._base() if not self.distraction else L.DISTRACTION_WARNING
                self._cancel_scene("dragged")
                self.sm.go(L.DRAGGED_BY_USER, "picked up", force=True)
            self._interacted()

    def set_fullscreen(self, on: bool) -> None:
        """A full-screen app or presentation is in front: hide quietly, come back afterwards."""
        if on == self.fullscreen:
            return
        self.fullscreen = on
        if on:
            if self.sm.state in (L.GOODBYE,) or self.distraction:
                return
            self._cancel_scene("full-screen app")
            self.hidden_for_fullscreen = True
            self.go_away(1e9, "fullscreen")
        elif self.hidden_for_fullscreen:
            self.hidden_for_fullscreen = False
            self.away_until = self._now() + 1.5

    # ----------------------------------------------------------- gag rules
    def context(self) -> str:
        return "focus" if self.focus_active else "break" if self.break_active else "roam"

    def why_not(self, g: Gag, context: str, now: float, categories: Optional[set],
                ignore_cooldown: bool = False) -> str:
        s = self.settings
        if g.flavors and self.flavor not in g.flavors:
            return "species"
        ok = context in g.contexts or (context == "break" and "roam" in g.contexts)
        if not ok:
            return f"not in {context}"
        if categories is not None and g.category not in categories:
            return "category off"
        if self.distraction:
            return "warning active"
        if not ignore_cooldown and self.engine.cooling(g, now):
            return "cooldown"
        if s.quiet and (g.category in ("attention", "comment") or "chat" in g.needs):
            return "quiet"
        if context == "focus":
            if s.focus_behavior == "strict":
                return "strict focus"
            if s.focus_behavior == "balanced" and (g.category not in ("study",) or "chat" in g.needs):
                return "balanced focus: silent study only"
        if "chat" in g.needs and s.chatter <= 0.05:
            return "chatter off"
        b = self.b
        if "cursor" in g.needs and (b.cursor is None or b.cursor_interaction == "off"):
            return "no cursor"
        if "windows" in g.needs and not [w for w in b.windows() if w.get("perchable")]:
            return "no windows"
        if "user_active" in g.needs and b.user_idle_seconds > 90:
            return "user away"
        if "away_ok" in g.needs and (not s.leave_and_return or s.roaming_frequency < 0.15):
            return "leaving off"
        if b.surface is not None and g.category in ("routine",):
            return "on a window"
        return ""

    def gag_weight(self, g: Gag, context: str) -> float:
        s = self.settings
        m = self.memory
        w = g.weight
        play = s.playfulness
        if g.category == "trouble":
            w *= 0.2 + 1.6 * play
        elif g.category == "attention":
            w *= (0.4 + play) * (1.0 + 0.6 * m.affinity)
        elif g.category == "comment":
            w *= 0.2 + 1.6 * s.chatter
        elif g.category == "species":
            w *= 0.7 + 0.6 * play
        if "away_ok" in g.needs:
            w *= (0.3 + 1.2 * s.roaming_frequency) * (1.0 + 0.8 * max(0.0, -m.affinity))
        if "cursor" in g.needs:
            w *= 1.0 + 0.6 * m.affinity
        energy = self.b.energy.level
        if g.category in ("trouble", "species") and energy < 25:
            w *= 0.4
        w *= 1.0 + 0.15 * min(3, m.loved.get(g.name, 0))      # favourite antics come back a bit more
        return max(0.0, w)

    def _gag_gap(self) -> float:
        s = self.settings
        if self.focus_active:
            return {"strict": 1e9, "balanced": 420.0, "playful": 160.0}.get(s.focus_behavior, 420.0)
        base = 150.0 - 110.0 * s.playfulness
        return max(30.0, base * self.rng.uniform(0.7, 1.3))

    def run_gag(self, g: Gag, note: bool = True) -> bool:
        steps = g.build(self)
        if not steps:
            return False
        now = self._now()
        if note:
            self.engine.note(g.name, now)
            self.memory.gag_counts[g.name] = self.memory.gag_counts.get(g.name, 0) + 1
            self.memory.recent_gags = (self.memory.recent_gags + [g.name])[-12:]
        if g.line:
            self.speak(g.line, 0.7)
        prio = g.priority if not g.interruptible else Priority.IDLE
        self._run(f"gag:{g.name}", steps, g.life_state if g.life_state != L.FOCUSING or self.focus_active
                  else self._base(), priority=prio, mode="auto", gag=g)
        return True

    def trigger(self, name: str) -> bool:
        """Playground / tests: run one gag now (cooldowns ignored, other rules kept)."""
        g = self.engine.choose(self, self.context(), self._now(), force=name)
        return self.run_gag(g) if g is not None else False

    def later(self, seconds: float, fn: Callable[[], None]) -> None:
        self._pending.append((self._now() + seconds, fn))

    # ----------------------------------------------------------- per frame
    def update(self, dt: float) -> None:
        now = self._now()
        b = self.b
        for item in [p for p in self._pending if p[0] <= now]:
            self._pending.remove(item)
            try:
                item[1]()
            except Exception:  # noqa: BLE001
                import logging
                logging.getLogger("focusbuddy.companion").exception("life callback failed")
        self.memory.mood = float(getattr(b.mood, "value", 0.0))
        self.memory.energy = float(b.energy.level)
        self.memory.last_state = self.sm.state.value
        st = self.sm.state
        # dragging is the user's: the life state follows the body
        if b.state == S.HELD and st != L.DRAGGED_BY_USER:
            self.on_drag(True)
            return
        if st == L.DRAGGED_BY_USER:
            if b.state not in (S.HELD, S.FALLING, S.LANDING):
                self.sm.go(getattr(self, "_resume", self._base()), "dropped", force=True)
            return
        # a scene / gag whose plan was replaced (cursor startle, a click reaction ...) has ended
        if self.scene and self._cb is not None and b._after_plan is not self._cb and \
                not (b._deferred is not None and b._deferred[2] is self._cb):
            gag = self._scene_gag
            self._cancel_scene("interrupted")
            if gag is not None or self.sm.state in (L.PLAYING, L.INTERACTING, L.ENTERING, L.WAKING):
                self.sm.go(self._base(), "interrupted", force=True)
        if st == L.AWAY:
            if now >= self.away_until and not self.fullscreen:
                if self.away_reason == "nap":
                    self.wake_from_nap()
                else:
                    self.enter(self.rng.choice((-1, 1)))
            return
        if st == L.GOODBYE or st == L.SLEEPING or st == L.EXITING or self.distraction:
            return
        # questions time out quietly
        if self.asking and now > self.ask_until:
            asking, self.asking = self.asking, ""
            self.hide_bubble()
            self._back_to_base()
        # idle-sleep of the body (user away) = resting
        if st in (L.ROAMING, L.FOCUSING) and b.state == S.SLEEPING and b.mode == "auto" and not self.scene:
            self.sm.go(L.RESTING, "user away")
        elif st == L.RESTING and b.state != S.SLEEPING and not self.scene:
            self.sm.go(self._base(), "awake again")
        # activity streak (outside focus sessions) -> an offer to take a break
        if b.user_idle_seconds < 90:
            self.active_streak += dt
        elif b.user_idle_seconds > 300:
            self.active_streak = 0.0
        if self.awaiting_user_until and now > self.awaiting_user_until:
            self.awaiting_user_until = 0.0
            self._ignored()
        if self.scene or b.mode != "auto" or b.state in (S.HELD, S.FALLING) or self.sm.state not in _AWAKE:
            return
        if b._plan and b.plan_priority > Priority.IDLE:
            return
        self._routine(now)

    def _ignored(self) -> None:
        m = self.memory
        m.ignored += 1
        m.affinity = max(-1.0, m.affinity - 0.05)
        gag = self.awaiting_gag
        self.awaiting_gag = ""
        if gag and not self.settings.quiet and not self.focus_active and self.rng.random() < 0.6:
            self.later(1.0, lambda: self._maybe_force("offended"))

    def _routine(self, now: float) -> None:
        s = self.settings
        b = self.b
        if self.sm.state == L.RESTING:
            return
        ctx = self.context()
        # long stretch of work outside a session: offer a break (never during a strict focus)
        if not self.focus_active and self.active_streak > 90 * 60 and now >= self.next_break_offer and \
                not s.quiet and s.focus_behavior != "strict":
            self.next_break_offer = now + 3600.0
            self.active_streak = 0.0
            self.speak("BREAK_OFFER", 1.0, secs=7.0, essential=True, m=90)
            return
        # sleepy after a long time roaming on low energy: ask (or quietly nap when quiet)
        if not self.focus_active and b.energy.level < 28 and now - self.roaming_since > 15 * 60 and \
                now >= self.next_nap_ask:
            self.next_nap_ask = now + 1800.0
            if s.quiet:
                self.nap(announce=False, rest_seconds=self.rng.uniform(240, 600))
            else:
                self.asking = "nap"
                self.ask_until = now + 20.0
                self.ask(self.line("NAP_ASK"), [("life:focus_nap", "Sure, nap 💤", True),
                                                ("life:nap_no", "Stay awake", False)])
                self._run("ask_nap", [("state", S.YAWNING, 1.5), ("look", A.USER, None, 20.0, S.SITTING)],
                          L.INTERACTING)
            return
        # focus: stuck (no input for a while) / sleepy during a long session
        if self.focus_active and s.focus_behavior != "strict" and not s.quiet:
            el = now - self.focus_started
            if "stuck" not in self.focus_flags and 240 < b.user_idle_seconds < 900 and el > 300:
                self.focus_flags.add("stuck")
                self.asking = "stuck"
                self.ask_until = now + 25.0
                self.ask(self.line("STUCK"), [("life:hint", "Give me a hint", True), ("life:fine", "I'm fine", False)])
                self._run("ask_stuck", [("action", "CONFUSED", None, Priority.INTERACTION),
                                        ("look", A.USER, None, 25.0, S.SITTING)], L.INTERACTING)
                return
            if "sleepy" not in self.focus_flags and el > 50 * 60 and b.user_idle_seconds < 120:
                self.focus_flags.add("sleepy")
                self.asking = "sleepy_focus"
                self.ask_until = now + 25.0
                self.ask(self.line("SLEEPY_FOCUS"), [("life:focus_nap", "Nap beside me 💤", True),
                                                     ("life:nap_no", "Stay with me", False)])
                self._run("ask_sleepy", [("state", S.YAWNING, 1.5), ("look", A.USER, None, 25.0, S.FOCUS_MODE)],
                          L.INTERACTING)
                return
        if now < self.next_gag_at:
            return
        self.next_gag_at = now + self._gag_gap()
        g = self.engine.choose(self, ctx, now)
        if g is not None:
            self.run_gag(g)

    # ----------------------------------------------------------- inspection (Playground)
    def explain(self) -> list[str]:
        out = [f"life: {self.sm.state.value}  ({self.sm.elapsed():.0f}s)  scene: {self.scene or '-'}",
               f"context: {self.context()}  mood {self.b.mood.label}  energy {self.b.energy.level:.0f}  "
               f"affinity {self.memory.affinity:+.2f}",
               f"next antic in {max(0.0, self.next_gag_at - self._now()):.0f}s"]
        for d in list(self.engine.log)[-3:][::-1]:
            cands = ", ".join(f"{n} {w}" for n, w in d.candidates[:5]) or "none"
            out.append(f"chose {d.chosen} [{d.context}] from: {cands}")
            if d.rejected:
                out.append("  skipped: " + ", ".join(f"{n} ({r})" for n, r in d.rejected[:6]))
        for t, a, b_, why in list(self.sm.history)[-4:][::-1]:
            out.append(f"{a} -> {b_}: {why}")
        return out

    def snapshot(self) -> dict:
        self.memory.last_seen = _time.time()
        self.memory.recent_lines = list(self.lines.recent)
        return self.memory.to_dict()


# ===================================================================== the catalogue
def _y(L_: Life) -> float:
    b = L_.b
    return b.surface.y if b.surface is not None else b.floor()


def _span_x(L_: Life, dist: float, toward: Optional[int] = None) -> float:
    b = L_.b
    lo, hi = b.span()
    d = toward if toward is not None else b.mover.facing
    x = b.x + d * dist
    if not lo <= x <= hi:
        x = b.x - d * dist
    return max(lo, min(hi, x))


P_I = Priority.IDLE
P_X = Priority.INTERACTION


def g_peek_window(L_: Life):
    b = L_.b
    steps = b.window_hangout_steps() if hasattr(b, "window_hangout_steps") else []
    if steps:
        return steps + [("action", "PEEK_BEHIND", None, P_I), ("state", S.IDLE, 0.5)]
    wins = [w for w in b.windows() if w.get("perchable")]
    if not wins:
        return None
    w = L_.rng.choice(wins)
    edge = w["x"] - L_.size * 0.3 if L_.rng.random() < 0.5 else w["x"] + w["w"] + L_.size * 0.3
    x, y = b.clamp_to_screen(edge, b.floor())
    return [("goto", x, y, False, S.WALKING), ("face", w["x"] + w["w"] / 2),
            ("action", "PEEK_BEHIND", None, P_I), ("state", S.PEEKING, 1.6), ("state", S.IDLE, 0.4)]


def g_fake_leave(L_: Life):
    b = L_.b
    x1 = _span_x(L_, L_.size * 2.4)
    return [("goto", x1, _y(L_), False, S.WALKING), ("action", "TURN_HOP", None, P_I),
            ("call", lambda: L_.speak("FAKE_LEAVE", 0.6)), ("action", "LAUGH", 1.2, P_I),
            ("look", A.USER, None, 0.8, S.IDLE)]


def g_cursor_caught(L_: Life):
    b = L_.b
    c = b.cursor
    if c is None:
        return None
    away = b.x - (1 if c[0] > b.x else -1) * 200
    return [("look", A.MOUSE, c, 2.4, S.SITTING), ("action", "SURPRISED", None, P_I), ("face", away),
            ("call", lambda: L_.speak("CAUGHT_STARING", 0.5)), ("action", "SLOW_BLINK" if L_.flavor == "cat"
                                                                  else "LOOK_AROUND", None, P_I),
            ("state", S.IDLE, 0.6)]


def g_forgot_why(L_: Life):
    x1 = _span_x(L_, L_.size * L_.rng.uniform(2.0, 3.5))
    return [("goto", x1, _y(L_), True, S.RUNNING), ("state", S.IDLE, 0.5), ("action", "CONFUSED", None, P_I),
            ("call", lambda: L_.speak("FORGOT", 0.6)), ("action", "THINK", 1.4, P_I), ("state", S.SITTING, 3.0)]


def g_poke_bubble(L_: Life):
    def pop():
        L_.hide_bubble()
    return [("call", lambda: L_.say("…", 3.0)), ("look", A.NOTIFICATION, None, 0.8, S.IDLE),
            ("action", "PAW_AT_OBJECT", 1.4, P_I), ("call", pop), ("action", "SURPRISED", None, P_I),
            ("action", "PROUD", 1.0, P_I)]


def g_proud_innocent(L_: Life):
    silly = L_.rng.choice(["SPIN", "TUMBLE", "JUMP_SPIN", "ROLL"])
    return [("action", silly, None, P_I), ("call", lambda: L_.speak("PROUD", 0.5)), ("action", "PROUD", 1.2, P_I),
            ("action", "SLOW_BLINK" if L_.flavor == "cat" else "LOOK_AROUND", None, P_I), ("state", S.IDLE, 0.5)]


def g_follow_cursor(L_: Life):
    c = L_.b.cursor
    if c is None:
        return None
    return [("look", A.MOUSE, c, 3.0, S.SITTING), ("action", "LEAN_TO_CURSOR", 1.2, P_I), ("state", S.IDLE, 0.5)]


def g_wave_hello(L_: Life):
    def wait():
        L_.awaiting_user_until = L_._now() + 25.0
        L_.awaiting_gag = "wave_hello"
    return [("look", A.USER, None, 0.8, S.IDLE), ("action", "WAVE", None, P_I), ("call", wait),
            ("look", A.USER, None, 2.0, S.SITTING)]


def g_bring_gift(L_: Life):
    b = L_.b
    if L_.flavor == "dog":
        steps = b.fetch_steps()
    else:
        x0 = b.x
        x1 = _span_x(L_, L_.size * 3.0)
        steps = [("goto", x1, _y(L_), True, S.RUNNING), ("action", "SNIFF", 1.0, P_I),
                 ("action", "PAW_AT_OBJECT", 1.0, P_I), ("goto", x0, _y(L_), False, S.WALKING),
                 ("action", "PROUD", 1.0, P_I)]

    def give():
        L_.speak("GIFT", 0.9, secs=4.0)
        L_.awaiting_user_until = L_._now() + 25.0
        L_.awaiting_gag = "bring_gift"
    return steps + [("look", A.USER, None, 0.8, S.IDLE), ("call", give), ("action", "HAPPY", 1.2, P_I)]


def g_remark(L_: Life):
    return [("look", A.USER, None, 0.6, S.IDLE), ("call", lambda: L_.speak("REMARK", 1.0, secs=4.0)),
            ("state", S.IDLE, 1.0)]


def g_challenge(L_: Life):
    if L_.memory.sessions_today > 0:
        return None
    return [("look", A.USER, None, 0.6, S.IDLE), ("action", "SMUG", None, P_I),
            ("call", lambda: L_.speak("CHALLENGE", 1.0, secs=6.0)), ("action", "DETERMINED", None, P_I)]


def g_offended(L_: Life):
    steps = L_.b.sulk_steps()
    steps.insert(1, ("call", lambda: L_.speak("OFFENDED", 0.8)))
    steps.insert(-2, ("call", lambda: L_.speak("OVER_IT", 0.6)))
    return steps


def g_smug_predict(L_: Life):
    return [("look", A.USER, None, 0.6, S.IDLE), ("action", "SMUG", None, P_I),
            ("call", lambda: L_.speak("SMUG", 0.9)), ("state", S.IDLE, 0.6)]


def g_celebrate_streak(L_: Life):
    v = L_.b.variant("CELEBRATE") if hasattr(L_.b, "variant") else "CHEER"
    return [("jump", None, 0.0, 0.0, 1), ("action", v if v != "CELEBRATE" else "CHEER", 1.8, P_X),
            ("action", "PROUD", 1.0, P_X)]


def g_study_read(L_: Life):
    prop = {"robot": "PROCESSING", "buddy": "READ", "girl": "READ", "saree": "WRITE"}.get(L_.flavor)
    if prop is None:
        prop = "LOAF" if L_.flavor in ("cat", "rabbit") else "DOWN" if L_.flavor == "dog" else "LIE_DOWN"
    return [("action", prop, L_.rng.uniform(15, 30), P_I), ("state", S.FOCUS_MODE, 4.0)]


def g_study_nod(L_: Life):
    return [("look", A.USER, None, 0.8, None), ("action", "NOD", None, P_I),
            ("call", lambda: L_.speak("STUDY_CHEER", 0.5) if L_.settings.focus_behavior == "playful" else None),
            ("state", S.FOCUS_MODE, 3.0)]


def g_leave_patrol(L_: Life):
    # handled by Life.leave (it walks off-screen and comes back later with another entrance)
    L_.later(0.0, lambda: L_.leave())
    return [("state", S.IDLE, 0.2)]


# species signatures -----------------------------------------------------------------
def _seq(*items, end=("state", S.IDLE, 0.5)):
    def build(L_: Life):
        out = []
        for it in items:
            if callable(it):
                r = it(L_)
                if r is None:
                    return None
                out += r
            else:
                out.append(("action", it[0], it[1], P_I) if isinstance(it, tuple) else ("action", it, None, P_I))
        return out + [end]
    return build


def _sprint(L_: Life):
    return [("goto_gait", _span_x(L_, L_.size * L_.rng.uniform(3.0, 5.0)), _y(L_), "sprint")]


def _hops(L_: Life):
    return [("action", "HOP", None, P_I), ("action", "DOUBLE_HOP", None, P_I)]


def _feint(L_: Life):
    x1 = _span_x(L_, L_.size * 1.4)
    return [("action", "SNEAK", 1.4, P_I), ("goto", x1, _y(L_), False, S.WALKING), ("action", "FREEZE", 0.8, P_I),
            ("turn_around",), ("action", "SMUG", None, P_I)]


def _pretend_not_hear(L_: Life):
    away = L_.b.x - (L_.b.mover.facing * 200)
    return [("face", away), ("action", "TAIL_SWISH", None, P_I), ("state", S.SITTING, 3.0),
            ("look", A.USER, None, 0.8, None)]


def build_catalogue() -> list[Gag]:
    T, AT, C, ST, SP, RT = "trouble", "attention", "comment", "study", "species", "routine"
    return [
        # the little troublemaker
        Gag("peek_window", T, g_peek_window, 1.2, 300, needs=("windows",), description="sneak behind a window, peek around it"),
        Gag("fake_leave", T, g_fake_leave, 0.8, 600, line="", description="pretend to walk away, then turn back"),
        Gag("cursor_caught", T, g_cursor_caught, 1.0, 420, needs=("cursor",), description="stare at the cursor, look away innocently"),
        Gag("forgot_why", T, g_forgot_why, 0.8, 480, description="start somewhere, forget why, sit down"),
        Gag("poke_bubble", T, g_poke_bubble, 0.5, 900, needs=("chat",), description="poke a bubble until it pops"),
        Gag("proud_innocent", T, g_proud_innocent, 0.8, 480, description="something silly, proud, then innocent"),
        # the attention seeker
        Gag("follow_cursor", AT, g_follow_cursor, 0.9, 240, needs=("cursor", "user_active"), life_state=L.INTERACTING,
            description="follows the cursor with its eyes"),
        Gag("wave_hello", AT, g_wave_hello, 0.7, 900, needs=("user_active",), life_state=L.INTERACTING,
            description="waves and waits for you (ignored -> mock offence)"),
        Gag("bring_gift", AT, g_bring_gift, 0.6, 1200, needs=("user_active",), life_state=L.INTERACTING,
            description="brings an imaginary gift / toy / book"),
        # the cheeky commentator
        Gag("remark", C, g_remark, 1.0, 600, needs=("chat", "user_active"), life_state=L.INTERACTING,
            description="a short remark"),
        Gag("challenge", C, g_challenge, 0.5, 3600, needs=("chat", "user_active"), life_state=L.INTERACTING,
            description="challenges you to finish a focus session"),
        Gag("offended", C, g_offended, 0.0, 900, life_state=L.INTERACTING, priority=P_X, interruptible=False,
            description="pretends to be offended when ignored, then gets over it (event-driven)"),
        Gag("smug_predict", C, g_smug_predict, 0.0, 900, life_state=L.INTERACTING,
            description="smug look after predicting a distraction (event-driven)"),
        Gag("celebrate_streak", C, g_celebrate_streak, 0.0, 60, contexts=("roam", "focus", "break"),
            life_state=L.INTERACTING, priority=P_X, interruptible=False,
            description="celebrates a streak / milestone (event-driven)"),
        # the study partner (during focus sessions)
        Gag("study_read", ST, g_study_read, 1.5, 300, contexts=("focus",), life_state=L.FOCUSING,
            description="studies alongside you with a book / laptop"),
        Gag("study_nod", ST, g_study_nod, 1.0, 600, contexts=("focus",), life_state=L.FOCUSING,
            description="a quiet encouraging nod"),
        # species signatures
        Gag("cat_slow_blink", SP, _seq("SLOW_BLINK", "TAIL_SWISH"), 1.0, 300, flavors=("cat",)),
        Gag("cat_groom", SP, _seq(("GROOM", 3.0), "LICK_PAW"), 1.0, 400, flavors=("cat",)),
        Gag("cat_sudden_sprint", SP, _seq(_sprint, "SKID_STOP", "SLOW_BLINK"), 0.7, 600, flavors=("cat",)),
        Gag("cat_ignore_you", SP, _seq(_pretend_not_hear), 0.5, 900, flavors=("cat",),
            description="pretends not to hear you"),
        Gag("cat_curl_ball", SP, _seq(("CURL_UP", 1.0), ("LOAF", 8.0)), 0.6, 900, flavors=("cat",)),
        Gag("dog_happy_hops", SP, _seq(("TAIL_WAG", 1.2), _hops, "PANT"), 1.0, 400, flavors=("dog",)),
        Gag("dog_fetch", SP, lambda L_: L_.b.fetch_steps(), 0.8, 600, flavors=("dog",)),
        Gag("dog_trot", SP, _seq(lambda L_: [("goto_gait", _span_x(L_, L_.size * 3.0), _y(L_), "jog")],
                                 ("TAIL_WAG", 1.4)), 1.0, 300, flavors=("dog",)),
        Gag("rabbit_freeze", SP, _seq(("FREEZE", 1.4), "EAR_TWITCH", _hops), 1.0, 400, flavors=("rabbit",)),
        Gag("rabbit_sniff", SP, _seq("SNIFF", "NOSE_TWITCH", ("GROOM", 2.5)), 1.0, 300, flavors=("rabbit",)),
        Gag("fox_feint", SP, _seq(_feint), 1.0, 500, flavors=("fox",), description="playful feint"),
        Gag("fox_sideways", SP, _seq("TURN_LOOK", lambda L_: [("state", S.IDLE, 1.2)], "SMUG"), 1.0, 400,
            flavors=("fox",), description="sideways glance, dramatic pause"),
        Gag("robot_scan", SP, _seq("SCAN", "PROCESSING", "SYSTEM_CHECK"), 1.0, 400, flavors=("robot",)),
        Gag("robot_glitch_dance", SP, _seq("ERROR", ("DANCE", 2.4), "RECALIBRATE"), 0.7, 900, flavors=("robot",)),
        Gag("robot_charge", SP, _seq(("CHARGE", 10.0), "BOOT"), 0.5, 1200, flavors=("robot",),
            description="a little charging nap"),
        Gag("human_watch", SP, _seq("LOOK_AT_WATCH", "STRETCH"), 1.0, 500, flavors=("buddy", "girl", "saree")),
        Gag("human_read", SP, _seq(("READ", 10.0)), 0.8, 600, flavors=("buddy", "girl", "saree")),
        Gag("human_laptop", SP, _seq(("WRITE", 10.0), "THINK"), 0.8, 600, flavors=("buddy", "girl")),
        Gag("saree_twirl", SP, _seq("SPIN", "CLAP", "WAVE"), 1.0, 500, flavors=("saree",),
            description="a warm, playful twirl"),
        # 2.0 little human moments (everyone; the robot gets its own versions through substitutes)
        Gag("sneeze", T, _seq("SNEEZE", lambda L_: [("call", lambda: L_.speak("SNEEZE", 0.7, secs=3.0))]), 0.35, 1800,
            description="an unexpected sneeze, then an embarrassed 'excuse me'"),
        Gag("hiccups", T, _seq(lambda L_: [("call", lambda: L_.speak("HICCUP", 0.6, secs=3.0))], "HICCUP"),
            0.25, 2400, description="a short fit of hiccups"),
        Gag("shiver", T, _seq("SHIVER", lambda L_: [("call", lambda: L_.speak("SHIVER", 0.5, secs=3.0))]), 0.25, 2400,
            description="a little shiver, as if a draught came in"),
        Gag("daydream", ST, _seq(("DAYDREAM", 6.0), lambda L_: [("call", lambda: L_.speak("DAYDREAM", 0.4, secs=3.0))]),
            0.6, 1200, contexts=("roam", "break"), description="drifts off into a daydream, then snaps back"),
        Gag("hum_tune", ST, _seq(lambda L_: [("call", lambda: L_.speak("HUM", 0.4, secs=2.5))], ("HUM", 4.0)), 0.6, 900,
            contexts=("roam", "break"), description="hums a little tune, bobbing along"),
        # routine: wander off-screen for a while and come back with another entrance
        Gag("leave_patrol", RT, g_leave_patrol, 0.45, 1500, needs=("away_ok",), life_state=L.ROAMING,
            description="walks off-screen and returns later"),
    ]
