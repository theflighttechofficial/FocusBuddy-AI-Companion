"""Companion brain: personality-driven idle life + directed, window-aware actions.

Qt-free so it can be unit-tested.  The Companion controller calls
``update(dt, now)`` once per frame; the app issues directives such as
``investigate(rect)`` or ``press_close(rect, on_impact)``.

The brain runs a *plan*: a queue of steps.

    ("goto", x, y, run, moving_state)      high-level travel; expanded by the path planner into
    ("move", x, y, gait, chase)            ground travel (walk / run / walk_then_run)
    ("leap", x, y)                          ballistic jump between surfaces
    ("state", CompanionState, secs|None)    hold a pose (None = until the next directive)
    ("face", x)                             turn toward x (animated), waits for the turn
    ("look", AttentionTarget, point, secs, state|None)
    ("emotion", Emotion, intensity, hold)
    ("press", on_impact)                    physically press a button; calls on_impact at contact
    ("call", fn)
    ("action", NAME, secs|None, priority)   play a library action (actions.py); full-body actions hold
                                            the plan, upper-body / additive ones run alongside it
    ("jump", height, forward, delay, count) physical jump(s) with gravity (crouch -> air -> land)

Sub-systems owned here: LocomotionController (movement.py), EmotionController,
AttentionController, PathPlanner, RoamingController, StateMachine, ActionPlayer
(actions.py) and the brain (brain.py: ActionSelector, EnergyModel, time of day).

Interruptions: every plan carries a priority.  A new plan never cuts a
full-body action at an awkward moment - it waits for the action's next exit
point (max 0.8 s) and never interrupts a jump mid-air - unless it is a
CRITICAL reaction (being grabbed).  Interrupted actions blend out from where
they are; nothing resets.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from .actions import VARIANTS, ActionPlayer, Priority, pick_variant, resolve, species_of
from .attention import AttentionController, AttentionTarget, LookOutput
from .brain import MOOD_BIAS, TOD_ENERGY, ActionSelector, EnergyModel, MoodModel, ShortTermMemory, time_of_day
from .emotion import Emotion, EmotionController
from .movement import Mover
from .pathing import Box, PathPlanner
from .profile import CharacterProfile, get_profile
from .roaming import DestinationKind, RoamingController
from .states import CompanionState as S
from .states import StateMachine

A = AttentionTarget


class Environment(Protocol):
    def screen_bounds(self, x: float, y: float) -> tuple[float, float, float, float]:
        """(left, top, right, bottom) of the *available* area of the screen at x,y."""

    def all_bounds(self) -> list[tuple[float, float, float, float]]:
        ...


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    handle: Optional[int] = None

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass
class Surface:
    kind: str                 # window | ledge
    left: float
    right: float
    y: float
    handle: Optional[int] = None


STATE_EMOTION = {
    S.ALERT: (Emotion.ALERT, 2.5), S.WATCHING: (Emotion.SUSPICIOUS, 5.0), S.CHASING: (Emotion.ALERT, 3.0),
    S.WARNING: (Emotion.SUSPICIOUS, 8.0), S.CELEBRATING: (Emotion.CELEBRATING, 3.0),
    S.SLEEPING: (Emotion.SLEEPY, None), S.WAKING: (Emotion.SLEEPY, 2.5), S.YAWNING: (Emotion.SLEEPY, 2.0),
    S.CONFUSED: (Emotion.WORRIED, 3.0), S.PLAYING: (Emotion.EXCITED, 3.0), S.WAVING: (Emotion.HAPPY, 4.0),
    S.PETTED: (Emotion.HAPPY, 5.0), S.PRESSING: (Emotion.ANNOYED, 1.2),
    S.CLOSING_DISTRACTION: (Emotion.SUSPICIOUS, None), S.PEEKING: (Emotion.CURIOUS, 3.0),
    S.LOOKING_AROUND: (Emotion.CURIOUS, 3.0),
}

# gentler stand-ins when "angry reactions" are switched off
TAME = {"SHOUT": "POINT", "ANGRY": "SHAKE_HEAD", "TANTRUM": "SHAKE_HEAD", "GRUMPY": "SHAKE_HEAD", "HUFF": "NOD",
        "SHOUT_SOFT": "POINT", "SHOUT_BIG": "POINT", "STOMP": "SHAKE_HEAD"}

IDLE_INTERRUPTIBLE = {S.IDLE, S.SITTING, S.LOOKING_AROUND, S.FOCUS_MODE, S.READING, S.WALKING, S.TURNING}


class Behavior:
    def __init__(self, env: Environment, *, char_size: float = 120.0, seed: int | None = None,
                 now: Callable[[], float] | None = None, profile: CharacterProfile | None = None):
        import time
        self.env = env
        self._now = now or time.perf_counter
        self.rng = random.Random(seed)
        self.profile = profile or get_profile("cat")
        self.sm = StateMachine(S.IDLE, now=self._now)
        self.sm.on_change(self._on_state)
        b = env.all_bounds()[0]
        self.char_size = char_size
        self.mover = Mover(b[0] + (b[2] - b[0]) * 0.8, b[3], profile=self.profile, scale=char_size / 124.0)
        self.emotion = EmotionController(self.profile.baseline)
        self.attention = AttentionController()
        self.paths = PathPlanner()
        self.roaming = RoamingController(self.rng)
        self.look: Optional[LookOutput] = None
        self.mode = "auto"           # auto | directed | held
        self.focus_mode = False
        self.break_mode = False
        self.walking_frequency = 0.5
        self.idle_activity = 0.5
        self.user_speed = 1.0
        self.user_idle_seconds = 0.0
        self.idle_sleep_after = 300.0
        self.roam_level = "desktop"     # floor | desktop | all
        self.cursor_mode = "look"       # ignore | look | follow | chase | run_away
        self.edge_peeking = True
        self.stay_near_focus = True
        self.cursor: Optional[tuple[float, float]] = None
        self.active_window: Optional[tuple[float, float, float, float]] = None
        self.surface: Optional[Surface] = None
        self.focus_home: Optional[tuple[float, float]] = None
        self.press_tip = (40.0, -24.0)    # renderer units (paw tip while pressing)
        self.path_points: list[tuple[float, float]] = []
        self.events: list[tuple[str, float]] = []
        self._plan: list[tuple] = []
        self._step_until: Optional[float] = None
        self._step_action = None          # full-body action started by the current plan step
        self._step_started = 0.0
        self._after_plan: Optional[Callable[[], None]] = None
        self._press_fired = False
        self.on_land: Optional[Callable[[], None]] = None
        self.on_arrive: Optional[Callable[[], None]] = None
        self.target_rect: Optional[Rect] = None
        self._was_sleeping_for_idle = False
        self._cursor_next = 0.0
        self._turn_cooldown = 0.0
        self._last_cursor: Optional[tuple[float, float]] = None
        self._explicit_emotion_at = -10.0
        self._falling_to: Optional[Surface] = None
        self._jump: dict = {}
        # --- gen 2: actions, brain, energy ---------------------------------
        self.species = species_of(self.profile)
        self.actions = ActionPlayer(self.species, self.rng)
        self.brain = ActionSelector(self.rng, self.species)
        self.energy = EnergyModel(self.profile.energy * 100.0)
        self.energy_setting = 0.0          # Low -15 / Normal 0 / High +15
        self.time_of_day_enabled = True
        self.autonomous = True             # False in the playground (only scripted actions)
        self.plan_priority = Priority.IDLE
        self._deferred: Optional[tuple] = None
        self._cursor_still: Optional[tuple[float, float, float]] = None
        self.clock_hour: Optional[float] = None    # override for tests / demo
        # emotion pack
        self.angry_reactions = True        # playful grumpiness (stomp, shout "HEY!"); off = gentler substitutes
        self._clicks: list[float] = []
        self._drag_shake = 0.0
        self._last_drag: Optional[tuple[float, float, float]] = None
        self._dizzy_after_land = False
        self._scare_cd = 0.0
        self._cursor_prev: Optional[tuple[float, float, float]] = None
        self.sequence_loop: Optional[list] = None
        # living-character upgrade: mood, short-term memory, environment awareness
        self.mood = MoodModel()
        self.attention.gaze_hold = getattr(self.profile, "gaze_hold", 1.0)
        self.memory = ShortTermMemory()     # local, in-memory only - never written to disk
        self.life_rng = random.Random((seed or 0) * 7919 + 17)   # stumbles / variants (keeps self.rng stable)
        self.stumbles = True                # rare trip / fall-over moments
        self.stumble_rate = 1.0 / 1500.0    # per second of walking (very low)
        self.cursor_interaction = "subtle"  # off | subtle | playful
        self.window_awareness = True
        self._win_prev: dict = {}
        self._win_react_cd = 0.0
        self._ride_react_cd = 0.0
        self._idle_mood_noted = False
        self._cursor_hist: list[tuple[float, float, float]] = []
        self._cursor_world_cd = 0.0
        self._circle_acc = 0.0
        self._circle_prev: Optional[float] = None
        self._circle_t0 = 0.0
        self._open_notice_cd = 0.0
        self._idle_var_next = 4.0
        self._idle_var_recent: list = []
        self.edge_bias = 0.0                # 0 = roam anywhere .. 1 = keep to the screen edges / corners (Life)

    # ------------------------------------------------------------- settings
    def set_profile(self, profile: CharacterProfile) -> None:
        self.profile = profile
        self.attention.gaze_hold = getattr(profile, "gaze_hold", 1.0)
        self.mover.profile = profile
        self.emotion.baseline = Emotion(profile.baseline)
        self.energy.base = max(10.0, min(95.0, profile.energy * 100.0))

    def set_species(self, species: str) -> None:
        self.species = species
        self.actions.species = species
        self.brain.species = species

    @property
    def context(self) -> str:
        if self.target_rect is not None and self.mode == "directed":
            return "distraction"
        return "focus" if self.focus_mode else "break" if self.break_mode else "idle"

    def set_size(self, char_size: float) -> None:
        self.char_size = char_size
        self.mover.scale = char_size / 124.0

    # ------------------------------------------------------------- geometry
    @property
    def x(self) -> float:
        return self.mover.pos.x

    @property
    def y(self) -> float:
        return self.mover.pos.y

    @property
    def state(self) -> S:
        return self.sm.state

    @property
    def half(self) -> float:
        return self.char_size * 0.35

    @property
    def unit(self) -> float:
        """Logical pixels per renderer unit."""
        return self.char_size / 140.0

    def floor(self, x: float | None = None, y: float | None = None) -> float:
        b = self.env.screen_bounds(self.x if x is None else x, self.y if y is None else y)
        return b[3]

    def on_floor(self) -> bool:
        return self.surface is None and abs(self.y - self.floor()) < 2

    def windows(self) -> list[dict]:
        fn = getattr(self.env, "windows", None)
        try:
            w = fn() if fn else None
            return list(w) if w else []
        except Exception:  # noqa: BLE001
            return []

    def world_known(self) -> bool:
        fn = getattr(self.env, "windows", None)
        try:
            return fn is not None and fn() is not None
        except Exception:  # noqa: BLE001
            return False

    def clamp_to_screen(self, x: float, y: float) -> tuple[float, float]:
        l, t, r, btm = self.env.screen_bounds(x, y)
        half = self.half
        x = min(max(x, l + half), r - half)
        y = min(max(y, t + self.char_size * 0.95), btm)
        return x, y

    # idle variations: small layered motions so standing / sitting never looks like one frozen loop
    IDLE_VARIATIONS = ("WEIGHT_SHIFT", "EAR_TWITCH", "TAIL_SWISH", "TAIL_WAG", "SLOW_BLINK", "SMALL_STRETCH",
                       "LOOK_UP", "TURN_LOOK", "NOD", "NOSE_TWITCH", "SCAN", "HEAD_ROTATION")

    def idle_variation_weights(self) -> dict:
        """Weights for the next idle variation from species, personality, mood and energy."""
        sp = self.species
        prof = self.profile
        mood = self.mood_state
        e = self.energy.level
        quad = sp in ("cat", "dog", "fox", "rabbit")
        w = {"WEIGHT_SHIFT": 1.2 if not quad else 0.8, "SLOW_BLINK": 0.7 + (0.8 if sp == "cat" else 0.0),
             "SMALL_STRETCH": 0.3 + (0.6 if e < 40 else 0.0), "LOOK_UP": 0.4 + 0.4 * float(prof.curiosity),
             "TURN_LOOK": 0.5 + 0.8 * float(getattr(prof, "nervousness", 0.3)), "NOD": 0.4 if not quad else 0.0}
        if quad:
            w["EAR_TWITCH"] = 0.8 + 1.2 * float(getattr(prof, "nervousness", 0.3))
            w["TAIL_SWISH"] = 0.9 if sp in ("cat", "fox") else 0.4
            if sp in ("cat", "dog", "fox"):
                w["TAIL_WAG"] = (1.6 if sp == "dog" else 0.2) * (1.5 if mood in ("HAPPY", "EXCITED") else 1.0)
            if sp == "rabbit":
                w["NOSE_TWITCH"] = 1.6
        if sp == "robot":
            w.update({"SCAN": 1.0, "HEAD_ROTATION": 0.8, "SLOW_BLINK": 0.2})
        if mood == "CURIOUS":
            w["LOOK_UP"] = w.get("LOOK_UP", 0) * 1.8
            w["TURN_LOOK"] = w.get("TURN_LOOK", 0) * 1.6
        elif mood == "SLEEPY":
            w["SLOW_BLINK"] *= 2.0
            w["SMALL_STRETCH"] *= 1.6
        elif mood == "BORED":
            w["WEIGHT_SHIFT"] *= 1.5
        for n in self._idle_var_recent[-2:]:          # don't repeat the same thing twice in a row
            if n in w:
                w[n] *= 0.25
        return {k: v for k, v in w.items() if v > 0}

    def _idle_variation(self, now: float) -> bool:
        if now < self._idle_var_next or self._plan and self.plan_priority > Priority.IDLE:
            return False
        if self.state not in (S.IDLE, S.SITTING, S.WATCHING, S.FOCUS_MODE) or self.mover.speed > 5:
            return False
        if self.actions.current("full") is not None and self.actions.current("full").priority > Priority.IDLE:
            return False
        calm = 0.6 if self.focus_mode else 1.0                  # focus mode: fewer, quieter variations
        self._idle_var_next = now + self.life_rng.uniform(3.0, 7.0) / (calm * max(0.4, self.profile.idle_frequency))
        w = self.idle_variation_weights()
        if self.focus_mode:
            w = {k: v for k, v in w.items() if k in ("WEIGHT_SHIFT", "SLOW_BLINK", "EAR_TWITCH", "TAIL_SWISH")}
        names = [n for n in w if self.actions.ready(n, now)]
        if not names:
            return False
        total = sum(w[n] for n in names)
        r = self.life_rng.random() * total
        for n in names:
            r -= w[n]
            if r <= 0:
                break
        if self.actions.play(n, now, Priority.MICRO_MOVEMENT) is not None:
            self._idle_var_recent = (self._idle_var_recent + [n])[-6:]
            return True
        return False

    def apply_root_motion(self, dx: float) -> None:
        """Root motion from an animation clip (ROOT_MOTION policy): move along the floor in the facing
        direction, never off the current surface / screen and never while the mover is travelling."""
        if self.mover.airborne or self.mover.target is not None:
            return
        lo, hi = self.span()
        self.mover.pos.x = max(lo, min(hi, self.mover.pos.x + dx * self.mover.facing))

    def span(self) -> tuple[float, float]:
        l, t, r, b = self.env.screen_bounds(self.x, self.y)
        lo, hi = l + self.half, r - self.half
        if self.surface is not None:
            lo = max(lo, self.surface.left + self.half * 0.6)
            hi = min(hi, self.surface.right - self.half * 0.6)
            if hi < lo:
                lo = hi = (self.surface.left + self.surface.right) / 2
        return lo, hi

    def _inside_screens(self, x: float, y: float) -> bool:
        return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in self.env.all_bounds())

    def perch_for(self, rect: Rect) -> tuple[float, float]:
        """Where to stand beside a window's close button (on its title bar)."""
        l, t, r, b = self.env.screen_bounds(rect.cx, rect.cy)
        x = rect.right - self.char_size * 0.75
        x = max(rect.x + self.char_size * 0.4, x)
        top = max(rect.y, t)
        if top - self.char_size * 0.95 >= t:
            y = top + 2                     # stand on the title bar
        else:
            y = top + self.char_size * 0.98  # window touches screen top: stand inside it
        return self.clamp_to_screen(x, y)

    def close_button(self, rect: Rect) -> tuple[float, float]:
        """Centre of the window's close (X) button in logical px (Windows caption layout)."""
        return rect.right - 23.0, rect.y + 15.0

    def press_spot(self, rect: Rect) -> tuple[float, float]:
        """Where the feet must be so the paw tip lands on the close button."""
        bx, by = self.close_button(rect)
        tx, ty = self.press_tip
        u = self.unit
        x = bx - tx * u
        y = by - ty * u
        l, t, r, b = self.env.screen_bounds(rect.cx, rect.cy)
        x = max(l + self.half * 0.5, min(r - self.half * 0.5, x))
        y = max(t + self.char_size * 0.6, min(b, y))
        return x, y

    def _surface_for_rect(self, rect: Rect, y: float) -> Surface:
        return Surface("window", rect.x, rect.right, y, rect.handle)

    def _surface_at(self, x: float, y: float) -> Optional[Surface]:
        if abs(y - self.floor(x, y)) < 3:
            return None
        for w in self.windows():
            if abs(w["y"] - y) < 4 and w["x"] - 4 <= x <= w["x"] + w["w"] + 4:
                return Surface("window", w["x"], w["x"] + w["w"], w["y"], w.get("handle"))
        if self.target_rect is not None and self.target_rect.x - 4 <= x <= self.target_rect.right + 4:
            return Surface("window", self.target_rect.x, self.target_rect.right, y, self.target_rect.handle)
        return Surface("ledge", x - self.half * 2, x + self.half * 2, y)

    # ------------------------------------------------------------- plumbing
    def set_state(self, st: S) -> bool:
        if self.sm.can(st):
            return self.sm.transition(st)
        return self.sm.force_path(st)

    def set_emotion(self, e: Emotion | str, intensity: float = 1.0, hold: float | None = 3.0) -> None:
        self._explicit_emotion_at = self._now()
        self.emotion.set(e, intensity, hold, now=self._now())

    def _on_state(self, old: S, new: S) -> None:
        if self._now() - self._explicit_emotion_at < 0.25:
            return
        if new in STATE_EMOTION:
            e, hold = STATE_EMOTION[new]
            self.emotion.set(e, 1.0, hold, now=self._now())
        elif old in (S.SLEEPING, S.WATCHING, S.WARNING, S.CLOSING_DISTRACTION) and new in (S.IDLE, S.WALKING):
            self.emotion.reset()

    def _start_plan(self, steps: list[tuple], mode: str = "directed", after: Callable | None = None,
                    priority: Priority | None = None) -> None:
        if priority is None:
            priority = Priority.IDLE if mode == "auto" else Priority.DISTRACTION_RESPONSE
        now = self._now()
        if priority < Priority.CRITICAL_REACTION:
            cur = self.actions.current("full")
            busy_air = self.mover.jstate is not None
            if busy_air or (cur is not None and not self.actions.at_exit(cur, now)):
                # wait for a natural exit point (or landing) instead of cutting the action
                self._deferred = (steps, mode, after, priority, now)
                return
        self._deferred = None
        self.actions.stop_all(now, below=Priority.CRITICAL_REACTION)
        self.mode = mode
        self.plan_priority = priority
        self._plan = list(steps)
        self._after_plan = after
        self._step_until = None
        self._step_action = None
        self._begin_step()

    def _route(self, x: float, y: float, gait: str = "walk", chase: bool = False,
               dest_rect: Optional[Rect] = None) -> list[tuple]:
        start = (self.x, self.y)
        span = self.span() if self.surface is not None else None
        obstacles = []
        for w in self.windows():
            if self.surface is not None and w.get("handle") == self.surface.handle:
                continue
            if dest_rect is not None and w.get("handle") == dest_rect.handle:
                continue
            if w["w"] * w["h"] > 120_000:
                obstacles.append(Box(w["x"], w["y"], w["w"], w["h"]))
        wps = self.paths.plan_surface(start, span, (x, y), obstacles, self.char_size / 124.0, self._inside_screens)
        self.path_points = [start] + [(w.x, w.y) for w in wps]
        steps: list[tuple] = []
        for wp in wps:
            if wp.mode == "leap":
                steps.append(("leap", wp.x, wp.y))
            else:
                steps.append(("move", wp.x, wp.y, gait, chase))
        return steps

    # ------------------------------------------------------------- directives
    def notice(self, rect: Optional[Rect]) -> None:
        """Level 1: stop, turn the head and look at the distraction."""
        if self.mode == "held":
            return
        self.target_rect = rect
        self.mover.stop()
        steps: list[tuple] = []
        if self.state == S.SLEEPING:
            steps.append(("state", S.WAKING, 0.5))
        if rect:
            self.attention.set(A.DISTRACTION_WINDOW, (rect.cx, rect.cy), 20.0, now=self._now())
            steps.append(("face", rect.cx))
        steps += [("emotion", Emotion.CURIOUS, 1.0, 1.0), ("state", S.ALERT, 0.6),
                  ("emotion", Emotion.SUSPICIOUS, 1.0, None), ("state", S.WATCHING, None)]
        self._start_plan(steps)

    def investigate(self, rect: Optional[Rect], run: bool, arrive_state: S = S.WARNING,
                    on_ready: Callable[[], None] | None = None, gait: str | None = None, mood: str = "neutral") -> None:
        """Walk -> accelerate -> run the last distance -> stop beside the window -> look at it ->
        look at the user -> ``on_ready`` (show the bubble) -> hold ``arrive_state``."""
        if self.mode == "held":
            return
        self.target_rect = rect
        steps: list[tuple] = []
        if self.state == S.SLEEPING:
            # asleep: one ear twitches first, then the eyes open
            steps += [("action", "EAR_TWITCH", None, Priority.DISTRACTION_RESPONSE), ("state", S.SLEEPING, 0.5),
                      ("state", S.WAKING, 0.6)]
        fresh = self.state not in (S.WATCHING, S.ALERT, S.WARNING, S.CHASING, S.CLOSING_DISTRACTION)
        if fresh and rect is not None:
            # NOTICE: stop, head turns and the eyes track the window before the body reacts
            steps += [("look", A.DISTRACTION_WINDOW, (rect.cx, rect.cy), 0.35, S.IDLE)]
        if fresh:
            steps += [("emotion", Emotion.ALERT, 1.0, 2.0), ("state", S.ALERT, 0.45),
                      ("action", "STAND", 0.35, Priority.DISTRACTION_RESPONSE),
                      ("emotion", Emotion.SUSPICIOUS, 0.9, 6.0)]
        if rect is not None:
            self.attention.set(A.DISTRACTION_WINDOW, (rect.cx, rect.cy), 60.0, now=self._now())
            px, py = self.perch_for(rect)
            # WALK -> JOG -> RUN -> short SPRINT -> natural deceleration
            steps.append(("goto_rect", px, py, gait or ("run" if run else "escalate"), True, rect))
            steps.append(("face", rect.right))
            if mood == "angry":
                # arrives annoyed: stiffens, stomps, glares at the window
                steps += [("emotion", Emotion.ANNOYED, 1.0, 6.0), ("action", "STOMP", None, Priority.DISTRACTION_RESPONSE)]
            steps.append(("look", A.DISTRACTION_WINDOW, (rect.cx, rect.cy), 0.6, S.WATCHING))
            steps.append(("look", A.DISTRACTION_WINDOW, self.close_button(rect), 0.45, None))   # eyes the X
            steps.append(("look", A.USER, None, 0.7, None))
            arrive = {"grumpy": "GRUMPY", "angry": "SHOUT"}.get(mood, "POINT")
            if arrive == "SHOUT":
                steps += self.shout_steps(self.shout_level())
            elif arrive != "POINT":
                steps.append(("action", arrive, None, Priority.DISTRACTION_RESPONSE))
            steps.append(("action", "POINT", 1.2, Priority.DISTRACTION_RESPONSE))
        if on_ready:
            steps.append(("call", on_ready))
        steps.append(("state", arrive_state, None))
        self._start_plan(steps, priority=Priority.DISTRACTION_RESPONSE)

    def prepare_press(self, rect: Optional[Rect], shout: bool = False) -> None:
        """Go stand next to the close button with the paw ready (countdown)."""
        if self.mode == "held" or rect is None:
            return
        self.target_rect = rect
        self.attention.set(A.DISTRACTION_WINDOW, self.close_button(rect), 60.0, now=self._now())
        sx, sy = self.press_spot(rect)
        steps: list[tuple] = []
        if self.state == S.SLEEPING:
            steps.append(("state", S.WAKING, 0.4))
        steps += [("goto_rect", sx, sy, "run", True, rect), ("face", self.close_button(rect)[0])]
        if shout:
            steps += self.shout_steps(self.shout_level())
        steps.append(("state", S.CLOSING_DISTRACTION, None))
        self._start_plan(steps)

    def press_close(self, rect: Optional[Rect], on_impact: Callable[[], None]) -> None:
        """Walk to the close button and physically press it; ``on_impact`` fires at contact."""
        self._press_fired = False
        if rect is None or self.mode == "held":
            on_impact()
            return
        self.target_rect = rect
        bx, by = self.close_button(rect)
        self.attention.set(A.DISTRACTION_WINDOW, (bx, by), 30.0, now=self._now())
        sx, sy = self.press_spot(rect)
        steps: list[tuple] = []
        if math.hypot(self.x - sx, self.y - sy) > 6:
            steps.append(("goto_rect", sx, sy, "run", True, rect))
        steps += [("face", bx), ("press", on_impact), ("state", S.IDLE, 0.25)]
        self._start_plan(steps, after=lambda: None)

    def hold(self, st: S) -> None:
        """Hold a state where we are."""
        if self.mode == "held":
            return
        if self.target_rect is not None:
            self.mover.face_towards(self.target_rect.right)
        self._start_plan([("state", st, None)])

    def perform(self, st: S, seconds: float, then_home: bool = True) -> None:
        """Play a one-shot state (celebrate, confused, wave...) then resume life."""
        if self.mode == "held":
            return
        steps: list[tuple] = [("state", st, seconds)]
        self._start_plan(steps, after=self.release if then_home else None)

    def celebrate_return(self) -> None:
        """User went back to work: alert fades -> happy -> small celebration -> roam."""
        if self.mode == "held":
            return
        self.attention.clear(A.DISTRACTION_WINDOW)
        self.target_rect = None
        self.attention.set(A.USER, None, 1.5, now=self._now())
        now = self._now()
        self.mood.event("user_returned", now)
        self.memory.resolved()
        jump = self.variant("JUMP")
        celebrate = self.variant("CELEBRATE")
        # relief -> happy -> jump -> spin -> celebrate -> back to roaming
        steps = [("action", "RELIEVED", 1.0, Priority.CELEBRATION), ("emotion", Emotion.HAPPY, 1.0, 1.2),
                 ("state", S.CELEBRATING, 0.6)]
        steps += [("jump", None, 0.0, 0.0, 1)] if jump == "JUMP" else [("action", jump, None, Priority.CELEBRATION)]
        steps += [("action", "SPIN", None, Priority.CELEBRATION),
                  ("emotion", Emotion.CELEBRATING, 1.0, 3.0)]
        steps += [("state", S.CELEBRATING, 1.4)] if celebrate == "CELEBRATE" else \
            [("action", celebrate, 2.2, Priority.CELEBRATION)]
        steps += [("emotion", Emotion.HAPPY, 0.8, 5.0)]
        self._start_plan(steps, after=self.release, priority=Priority.CELEBRATION)

    def focus_start(self) -> None:
        """Look at the user, get excited, hop, settle somewhere comfy and focus."""
        if self.mode == "held":
            return
        l, t, r, b = self.env.screen_bounds(self.x, self.y)
        home_x = r - (r - l) * self.rng.uniform(0.08, 0.16)
        self.focus_home = (home_x, b)
        self.focus_mode = True
        self.break_mode = False
        steps: list[tuple] = []
        if self.state == S.SLEEPING:
            steps.append(("state", S.WAKING, 0.6))
        steps += [("look", A.USER, None, 0.6, S.IDLE), ("emotion", Emotion.EXCITED, 1.0, 2.5),
                  ("jump", None, 0.0, 0.0, 1), ("action", "DETERMINED", None, Priority.INTERACTION),
                  ("goto", home_x, b, False, S.WALKING), ("face", (l + r) / 2),
                  ("emotion", Emotion.NEUTRAL, 1.0, None), ("state", S.FOCUS_MODE, 8.0)]
        self._start_plan(steps, mode="auto", priority=Priority.INTERACTION)

    def focus_complete(self, celebrate_seconds: float = 3.2) -> None:
        if self.mode == "held":
            return
        self.focus_mode = False
        self.mood.event("focus_complete", self._now())
        steps = [("look", A.USER, None, 0.5, S.IDLE), ("emotion", Emotion.EXCITED, 1.0, 3.0),
                 ("jump", None, 0.0, 0.0, 1), ("emotion", Emotion.CELEBRATING, 1.0, 4.0),
                 ("state", S.CELEBRATING, celebrate_seconds), ("action", "DANCE", 2.4, Priority.CELEBRATION),
                 ("action", "LAUGH", 1.8, Priority.CELEBRATION), ("action", "PROUD", 1.6, Priority.CELEBRATION),
                 ("emotion", Emotion.EXCITED, 1.0, 6.0), ("state", S.PLAYING, 2.0)]
        self._start_plan(steps, after=self.release, priority=Priority.CELEBRATION)

    def release(self) -> None:
        """End directed behaviour: head back down to the floor and live freely."""
        self.target_rect = None
        self.attention.clear(A.DISTRACTION_WINDOW)
        if self.mode == "held":
            return
        if self.emotion.current in (Emotion.SUSPICIOUS, Emotion.ALERT, Emotion.ANNOYED):
            self.emotion.reset()
        fy = self.floor()
        if self.surface is not None or abs(self.y - fy) > 2:
            fx, fy = self.clamp_to_screen(self.x, fy)
            self._start_plan([("goto", fx, fy, False, S.WALKING), ("state", S.IDLE, 0.6)], mode="auto")
        else:
            self.mode = "auto"
            self._plan = []
            self._step_until = None
            self._step_action = None
            if self.state not in (S.IDLE, S.SITTING, S.FOCUS_MODE):
                self.set_state(S.IDLE)

    def wave(self) -> None:
        self.react_click("wave")

    def react_click(self, kind: str = "wave", where: str | None = None) -> None:
        """A click on the companion. ``where`` ("head", "body", "tail") is the part that was
        clicked: it always gets an immediate physical flinch (the animation reflexes), and a poke
        on the tail or a tap on the head gets its own little reaction instead of the default one."""
        if where in ("head", "body", "tail") and self.state != S.HELD:
            self.events.append((f"poke_{where}", 1.0))      # the flinch plays in every mode
        if self.mode != "auto" or self.state in (S.HELD, S.FALLING, S.JUMPING):
            return
        now = self._now()
        if now < getattr(self, "_annoyed_busy", 0.0):
            self.memory.click(now)              # mid-tantrum: clicks are noted, the reaction plays out
            return
        self._clicks = [t for t in self._clicks if now - t < 3.0] + [now]
        self.memory.click(now)
        self.mover.stop()
        self.attention.set(A.USER_INTERACTION, None, 1.5, now=now)
        if self.state == S.SLEEPING:          # woken up by a click: grumpy for a moment
            self.events.append(("woken", 0.0))
            self.mood.event("woken", now)
            self._start_plan([("state", S.WAKING, 0.9), ("look", A.USER, None, 0.3, None),
                              ("action", "GRUMPY", 2.2, Priority.INTERACTION), ("state", S.IDLE, 0.4)],
                             mode="auto", priority=Priority.INTERACTION)
            return
        if len(self._clicks) >= 4:            # poked over and over: annoyed -> stomp -> cool down -> laugh it off
            self._clicks.clear()
            self._annoyed_at = now
            self._annoyed_busy = now + 4.0
            self.events.append(("poked", 0.0))
            self.mood.event("repeated_clicks", now)
            if self.angry_reactions:
                # first time a few comic stomps; poked again within a minute -> the full (still cartoon) ANGRY
                reaction = "STOMP" if self.actions.ready("STOMP", now) else "ANGRY"
            else:
                reaction = "DIZZY"
            self._start_plan([("look", A.USER, None, 0.2, None), ("emotion", Emotion.ANNOYED, 0.8, 1.0),
                              ("action", reaction, None, Priority.INTERACTION), ("state", S.IDLE, 0.3),
                              ("action", "SULK", 2.2, Priority.INTERACTION),
                              ("action", "COOL_DOWN", None, Priority.INTERACTION), ("state", S.IDLE, 0.4)],
                             mode="auto", priority=Priority.INTERACTION)
            return
        self.memory.interact(now)
        annoyed_at = getattr(self, "_annoyed_at", None)
        if len(self._clicks) == 2:              # second click: curious - head tilt, ears up, lean in
            self._start_plan([("look", A.USER, None, 0.2, None), ("emotion", Emotion.CURIOUS, 0.9, 2.5),
                              ("action", "CURIOUS", None, Priority.INTERACTION), ("state", S.IDLE, 0.3)],
                             mode="auto", priority=Priority.INTERACTION)
            return
        if annoyed_at is not None and 20.0 <= now - annoyed_at <= 600.0 and len(self._clicks) == 1:
            # a while after being poked too much: forgiven - playful again
            self._annoyed_at = None
            self.mood.event("petted", now)
            play = "PLAY_BOW" if self.species in ("cat", "dog", "fox") else ("BOUNCE" if self.species == "rabbit"
                                                                              else "FIST_PUMP")
            self._start_plan([("look", A.USER, None, 0.2, None), ("emotion", Emotion.PLAYFUL, 1.0, 4.0),
                              ("action", play, None, Priority.INTERACTION), ("state", S.IDLE, 0.4)],
                             mode="auto", priority=Priority.INTERACTION)
            return
        quad = self.species in ("cat", "dog", "fox", "rabbit")
        if where == "tail" and quad and kind in ("wave", "look"):
            # "hey!" - a startled look back over the shoulder at the tail, then at you
            self._start_plan([("emotion", Emotion.SURPRISED, 1.0, 1.6),
                              ("action", "LOOK_BACK", 1.1, Priority.INTERACTION),
                              ("look", A.USER, None, 0.5, None), ("state", S.IDLE, 0.3)],
                             mode="auto", priority=Priority.INTERACTION)
            return
        if where == "head" and kind in ("wave", "look") and self.life_rng.random() < 0.6:
            # a tap on the head reads as a pat: eyes squeeze happily, a small nod / slow blink
            gesture = "SLOW_BLINK" if quad else ("NOD" if self.species != "robot" else "HEAD_ROTATION")
            self._start_plan([("look", A.USER, None, 0.2, None), ("emotion", Emotion.HAPPY, 1.0, 3.0),
                              ("action", gesture, None, Priority.INTERACTION), ("state", S.IDLE, 0.4)],
                             mode="auto", priority=Priority.INTERACTION)
            return
        if kind == "jump":
            steps = [("emotion", Emotion.EXCITED, 1.0, 2.0), ("jump", None, 0.0, 0.0, 1), ("state", S.IDLE, 0.4)]
        elif kind == "spin":
            steps = [("emotion", Emotion.HAPPY, 1.0, 3.0), ("state", S.CELEBRATING, 1.1)]
        elif kind == "look":
            steps = [("look", A.USER, None, 1.4, S.IDLE)]
        else:
            steps = [("look", A.USER, None, 0.3, None), ("state", S.WAVING, 1.6)]
        self._start_plan(steps, mode="auto", priority=Priority.INTERACTION)

    def pet(self) -> None:
        if self.mode != "auto" or self.state in (S.HELD, S.FALLING, S.JUMPING, S.SLEEPING):
            return
        self.mover.stop()
        now = self._now()
        self.mood.event("petted", now)
        self.memory.interact(now)
        self.set_emotion(Emotion.HAPPY, 1.0, 6.0)
        sp = self.species
        if sp == "cat":          # eyes close, head leans into the hand, purr
            react = [("action", "PURR", 2.2, Priority.INTERACTION), ("action", "SLOW_BLINK", None, Priority.INTERACTION)]
        elif sp == "dog":        # tail wag, wiggle, excited
            self.set_emotion(Emotion.EXCITED, 1.0, 4.0)
            react = [("action", "TAIL_WAG", 2.0, Priority.INTERACTION), ("action", "BOUNCE", None, Priority.INTERACTION)]
        elif sp == "rabbit":     # calm: ears relax back, slow nose twitch
            self.set_emotion(Emotion.RELAXED, 1.0, 6.0)
            react = [("action", "NOSE_TWITCH", 1.6, Priority.INTERACTION), ("action", "SLOW_BLINK", None, Priority.INTERACTION)]
        elif sp == "fox":
            react = [("action", "TAIL_SWISH", 1.6, Priority.INTERACTION), ("action", "SLOW_BLINK", None, Priority.INTERACTION)]
        else:
            react = [("action", "LOVE", 1.8, Priority.INTERACTION)]
        steps = [("look", A.USER, None, 0.2, None), ("state", S.PETTED, 2.0)] + react
        if self.mood.value > 30:              # already in a great mood: extra affection
            steps += self.love_steps()[:1]
        self._start_plan(steps, mode="auto", priority=Priority.INTERACTION)

    def emote(self, name: str, hold: Optional[S] = None, seconds: float | None = None,
              priority: Priority = Priority.INTERACTION) -> None:
        """Play an emotional action; then hold ``hold`` (e.g. back to WARNING) or go back to life."""
        if self.mode == "held":
            return
        steps: list[tuple] = []
        if self.state == S.SLEEPING:
            steps.append(("state", S.WAKING, 0.6))
        steps.append(("action", name, seconds, priority))
        if hold is not None:
            steps.append(("state", hold, None))
            self._start_plan(steps, mode=self.mode if self.mode != "held" else "directed", priority=priority)
        else:
            self._start_plan(steps, mode="auto", after=self.release, priority=priority)

    def glance(self, point: tuple[float, float], kind: A = A.ACTIVE_WINDOW, seconds: float = 2.0) -> None:
        self.attention.set(kind, point, seconds, now=self._now())

    # ---------------------------------------------------------------- dragging
    def grab(self) -> None:
        self.mode = "held"
        self._plan = []
        self._deferred = None
        self.actions.stop_all(self._now(), below=Priority.CRITICAL_REACTION)
        self.mover.stop()
        self.mover.leap = None
        self.mover.jstate = None
        self.mover.falling = False
        self.surface = None
        self.set_emotion(Emotion.WORRIED, 1.0, 3.0)
        self.set_state(S.HELD)

    def drag_to(self, x: float, y: float) -> None:
        now = self._now()
        if self._last_drag is not None:          # shaking it around makes it dizzy
            lx, ly, lt = self._last_drag
            dt = max(1e-3, now - lt)
            v = math.hypot(x - lx, y - ly) / dt
            self._drag_shake = self._drag_shake * math.exp(-dt / 0.8) + v * dt
        self._last_drag = (x, y, now)
        dx = x - self.mover.pos.x
        if abs(dx) > 2:
            self.mover.facing = 1 if dx > 0 else -1
            self.mover.facing_blend = float(self.mover.facing)
        self.mover.teleport(x, y)

    def drop(self) -> None:
        # land on a window top if dropped above one, else fall to the floor
        self._dizzy_after_land = self._drag_shake > self.char_size * 9
        self._drag_shake = 0.0
        self._last_drag = None
        x, y = self.mover.pos.x, self.mover.pos.y
        fy = self.floor()
        land_y = fy
        land_surface = None
        for w in self.windows():
            if w.get("perchable") and w["x"] <= x <= w["x"] + w["w"] and y <= w["y"] < land_y:
                land_y = w["y"]
                land_surface = Surface("window", w["x"], w["x"] + w["w"], w["y"], w.get("handle"))
        cx, _ = self.clamp_to_screen(x, fy)
        self.mover.pos.x = cx
        if y >= land_y - 1:
            self.mover.pos.y = land_y
            self.surface = land_surface
            self.mode = "auto"
            self.set_state(S.IDLE)
            if self._dizzy_after_land:
                self._dizzy_after_land = False
                self._start_plan([("action", "DIZZY", None, Priority.INTERACTION), ("state", S.IDLE, 0.4)],
                                 mode="auto", priority=Priority.INTERACTION)
            return
        self._falling_to = land_surface
        self.set_state(S.FALLING)
        self.mover.drop(land_y)

    # ------------------------------------------------------------------ frame
    def update(self, dt: float, now: float | None = None) -> None:
        now = self._now() if now is None else now
        # energy: drains with effort, recovers at rest, follows time of day (subtle)
        tod = time_of_day(self.clock_hour) if self.time_of_day_enabled else "day"
        self.energy.offset = self.energy_setting + (TOD_ENERGY.get(tod, 0.0) if self.time_of_day_enabled else 0.0)
        self.energy.update(dt, self.state.name, self.mover.gait_level, self.mover.speed > 5)
        e_mult = 0.92 + 0.16 * self.energy.level / 100.0
        self.mover.speed_factor = self.user_speed * self.emotion.style().speed_mult * e_mult
        arrived = self.mover.update(dt)
        for name, val in self.mover.drain_events():
            self.events.append((name, val))
            if name == "land":
                if self.state == S.FALLING:
                    self.surface = getattr(self, "_falling_to", None)
                elif self.mover.leap is None and self.surface is not None and abs(self.y - self.surface.y) < 3:
                    pass                      # in-place jump: still on the same surface
                else:
                    self.surface = self._surface_at(self.x, self.y)
                self._falling_to = None
        for ev, name in self.actions.update(now):
            self.events.append((ev, name))
            if ev == "ACTION_START":
                self.memory.action(now, name)
                ad = resolve(name, self.species)
                if ad is not None and ad.energy:
                    self.energy.spend(ad.energy)
        self.emotion.update(dt, now)
        self.mood.update(dt)
        self.memory.last_emotion = self.emotion.current.value
        if self.user_idle_seconds >= 600 and not self._idle_mood_noted:
            self._idle_mood_noted = True
            self.mood.event("long_inactivity", now)
        elif self.user_idle_seconds < 30 and self._idle_mood_noted:
            self._idle_mood_noted = False
        self._ride_surface()
        if self.mover.target is not None:
            t = self.mover.target
            self.attention.set(A.MOVEMENT_TARGET, (t.x, t.y - self.char_size * 0.5), 0.4, now=now)
        eye = (self.x + self.mover.facing * self.char_size * 0.15, self.y - self.char_size * 0.55)
        self.look = self.attention.update(dt, now, eye, self.mover.facing_blend, self.rng)

        if self.state == S.FALLING:
            if arrived or not self.mover.falling:
                self.mode = "auto"
                self.set_state(S.LANDING)
                resume = getattr(self, "_resume_after_fall", None)
                self._resume_after_fall = None
                if resume is not None:
                    steps, after, prio = resume
                    self._plan = [("state", S.LANDING, 0.3)] + steps
                    self._after_plan = after
                    self.plan_priority = prio
                    self.mode = "directed"
                elif self._dizzy_after_land:
                    self._dizzy_after_land = False
                    self._plan = [("state", S.LANDING, 0.3), ("action", "DIZZY", None, Priority.INTERACTION),
                                  ("state", S.IDLE, 0.4)]
                else:
                    self._plan = [("state", S.LANDING, 0.3), ("state", S.IDLE, 0.4)]
                self._begin_step()
                if self.on_land:
                    self.on_land()
            return
        if self.mode == "held":
            return
        if self._deferred is not None:
            steps, mode, after, prio, asked = self._deferred
            cur = self.actions.current("full")
            ready = self.mover.jstate is None and (cur is None or self.actions.at_exit(cur, now) or now - asked > 0.8)
            if ready:
                self._deferred = None
                self._start_plan(steps, mode, after, prio)

        # body follows attention when the target is behind us
        if self.look is not None and self.look.behind and not self.mover.moving and now > self._turn_cooldown \
                and self.state in (S.IDLE, S.SITTING, S.WATCHING, S.WARNING, S.LOOKING_AROUND, S.FOCUS_MODE,
                                   S.CLOSING_DISTRACTION) and self.look.point is not None:
            self._turn_cooldown = now + 1.5
            self.mover.face_towards(self.look.point[0])

        if self.mode == "auto" and self.autonomous and self.plan_priority <= Priority.IDLE and self._startle(now):
            return
        if self.mode == "auto" and self.autonomous and self.plan_priority <= Priority.IDLE and \
                (self._window_awareness(now) or self._maybe_stumble(now, dt) or self._cursor_world(now)):
            return
        if self.mode == "auto" and self.autonomous:
            self._idle_variation(now)
        if self._plan:
            self._run_step(now, arrived)
            return
        if self.mode == "auto" and self.autonomous:
            self._auto_life(now)

    def _ride_surface(self) -> None:
        """Stand on windows: ride along when they move, fall when they vanish."""
        s = self.surface
        if s is None or s.kind != "window" or s.handle is None or self.mover.airborne:
            return
        if not self.world_known():
            return
        wins = self.windows()
        w = next((w for w in wins if w.get("handle") == s.handle), None)
        if w is None or w.get("minimized"):
            self.surface = None
            if self._plan and self.plan_priority >= Priority.CELEBRATION and self.target_rect is None:
                # e.g. celebrating on the title bar of the window that was just closed:
                # fall, land, then carry on with the celebration
                self._resume_after_fall = (list(self._plan), self._after_plan, self.plan_priority)
            self._plan = []
            self.mode = "auto"
            self._falling_to = None
            self.set_state(S.FALLING)
            self.mover.drop(self.floor())
            return
        dy = w["y"] - s.y
        dx = w["x"] - s.left
        now = self._now()
        if math.hypot(dx, dy) > self.char_size * 0.35 and now > self._ride_react_cd and self.mode == "auto" \
                and self.plan_priority <= Priority.IDLE and self.window_awareness:
            # the window it is standing on was yanked around: hold on! (wide eyes, arms out)
            self._ride_react_cd = now + 10.0
            self.actions.play("SURPRISED", now, Priority.INTERACTION)
            self.set_emotion(Emotion.SURPRISED, 0.8, 1.5)
        if abs(dx) > 0.5 or abs(dy) > 0.5:
            self.mover.pos.x += dx
            self.mover.pos.y += dy
            if self.mover.target is not None:
                self.mover.target.x += dx
                self.mover.target.y += dy
            self.surface = Surface("window", w["x"], w["x"] + w["w"], w["y"], s.handle)

    def _moving_state(self, chase: bool) -> S:
        m = self.mover
        if m.turning:
            return S.TURNING
        if m.airborne:
            return S.JUMPING
        if m.speed > m.walk_speed * 1.25:
            return S.CHASING if chase else S.RUNNING
        return S.WALKING

    def _run_step(self, now: float, arrived: bool) -> None:
        step = self._plan[0]
        kind = step[0]
        if kind == "move":
            if self.mover.target is None and not self.mover.turning and not self.mover.airborne:
                self._plan.pop(0)
                if not any(s[0] in ("move", "leap") for s in self._plan) and self.on_arrive:
                    self.on_arrive()
                self._begin_step()
            else:
                self.set_state(self._moving_state(step[4]))
        elif kind == "leap":
            if not self.mover.airborne:
                self._plan.pop(0)
                self._begin_step()
            elif self.mover.leap is not None and self.mover.leap.crouch <= 0:
                self.set_state(S.JUMPING)
        elif kind == "face":
            if not self.mover.turning:
                self._plan.pop(0)
                self._begin_step()
            else:
                self.set_state(S.TURNING)
        elif kind in ("state", "look", "action"):
            if self._step_until is not None and now >= self._step_until:
                a = self._step_action
                if kind == "action" and a is not None and a in self.actions.active:
                    self.actions._stop(a, now)      # a timed / looping action ends with its step (blends out)
                self._step_action = None
                self._plan.pop(0)
                self._begin_step()
        elif kind == "shuffle":
            if self.mover.shuffle_to is None:
                self._plan.pop(0)
                self._begin_step()
            else:
                self.set_state(S.IDLE)
        elif kind == "jump":
            js = self._jump
            if js["phase"] == "wait":
                if now - self._step_started >= js["delay"]:
                    self._do_jump(js)
            elif js["phase"] == "air":
                if self.mover.airborne:
                    if self.mover.jstate is not None and self.mover.jstate["crouch"] <= 0:
                        self.set_state(S.JUMPING)
                else:
                    js["count"] -= 1
                    if js["count"] > 0:
                        js["phase"] = "wait"
                        js["delay"] = 0.05
                        self._step_started = now
                        self.set_state(S.LANDING)
                    else:
                        self._plan.pop(0)
                        self._plan.insert(0, ("state", S.LANDING, 0.22))
                        self._begin_step()
        elif kind == "press":
            t = now - self._step_started
            if t >= 0.45 and not self._press_fired:
                self._press_fired = True
                self.events.append(("press", 0.0))
                try:
                    step[1]()
                except Exception:  # noqa: BLE001 - never break the animation loop
                    import logging
                    logging.getLogger("focusbuddy.companion").exception("press callback failed")
            if t >= 0.85:
                self._plan.pop(0)
                self._begin_step()

    def _begin_step(self) -> None:
        now = self._now()
        while self._plan:
            step = self._plan[0]
            kind = step[0]
            self._step_started = now
            if kind in ("goto", "goto_rect"):
                if kind == "goto":
                    _, x, y, run, moving_state = step
                    gait, chase, rect = ("run" if run else "walk"), moving_state == S.CHASING, None
                else:
                    _, x, y, gait, chase, rect = step
                self._plan.pop(0)
                route = self._route(x, y, gait, chase, rect)
                if rect is not None and route:
                    self._dest_rect = rect
                self._plan[0:0] = route
                continue
            if kind == "move":
                _, x, y, gait, chase = step
                if math.hypot(x - self.x, y - self.y) < 1.0:
                    self._plan.pop(0)
                    continue
                self.mover.move_to(x, y, gait=gait)
                self.set_state(self._moving_state(chase))
                return
            if kind == "leap":
                _, x, y = step
                if abs(x - self.x) > 4 and (1 if x > self.x else -1) != self.mover.facing:
                    self.mover.turn_to(1 if x > self.x else -1)
                    self._plan.insert(0, ("face", x))
                    continue
                self.mover.leap_to(x, y)
                self.events.append(("jump", 0.0))
                return
            if kind == "face":
                fx = step[1]
                self.mover.face_towards(fx)
                if self.mover.turning:
                    self.set_state(S.TURNING)
                    return
                self._plan.pop(0)
                continue
            if kind == "state":
                _, st, secs = step
                if self.mover.speed > 40 and not self.mover.airborne:
                    self.mover.brake()        # interrupted mid-run: decelerate naturally, no pop
                else:
                    self.mover.stop()
                self.set_state(st)
                self._step_until = None if secs is None else now + secs
                return
            if kind == "look":
                _, akind, point, secs, st = step
                self.attention.set(akind, point, secs, now=now)
                if st is not None:
                    self.mover.brake()
                    self.set_state(st)
                self._step_until = now + secs
                return
            if kind == "emotion":
                _, e, inten, hold = step
                self.set_emotion(e, inten, hold)
                self._plan.pop(0)
                continue
            if kind == "call":
                self._plan.pop(0)
                try:
                    step[1]()
                except Exception:  # noqa: BLE001
                    import logging
                    logging.getLogger("focusbuddy.companion").exception("plan callback failed")
                continue
            if kind == "press":
                self.mover.stop()
                self.set_state(S.PRESSING)
                return
            if kind == "action":
                name, secs = step[1], (step[2] if len(step) > 2 else None)
                if not self.angry_reactions:
                    name = TAME.get(str(name).upper(), name)
                prio = step[3] if len(step) > 3 else None
                pr_ = prio if prio is not None else self.plan_priority
                # explicit requests (Playground, showcase, sequences) are not blocked by idle cooldowns
                a = self.actions.play(name, now, pr_, ignore_cooldown=pr_ >= Priority.SPECIAL_ACTION)
                if a is None:
                    self._plan.pop(0)
                    continue
                ad = a.adef
                if ad.emotion:
                    try:
                        self.set_emotion(Emotion(ad.emotion), 1.0, max(1.5, ad.duration))
                    except ValueError:
                        pass
                if ad.motion in ("jump", "hop", "double_hop", "pounce", "small_jump", "high_jump"):
                    h = self.profile.jump_height * self.mover.scale
                    spec = {"jump": (h, 0.0, 0.0, 1), "hop": (h * 0.45, 0.0, 0.0, 1),
                            "double_hop": (h * 0.5, 0.0, 0.0, 2), "small_jump": (h * 0.5, 0.0, 0.0, 1),
                            "high_jump": (h * 1.6, 0.0, 0.0, 1),
                            "pounce": (h * 0.55, 300.0 * self.mover.scale, 0.8, 1)}[ad.motion]
                    self._plan[0] = ("jump",) + spec
                    continue
                if ad.motion in ("backpedal", "sidestep"):
                    lo, hi = self.span()
                    dist = (secs or 1.0) * 45.0 * self.mover.scale if ad.motion == "backpedal" else \
                        16.0 * self.mover.scale
                    d = -self.mover.facing if ad.motion == "backpedal" else self.mover.facing
                    tx = max(lo, min(hi, self.x + d * dist))
                    self.mover.shuffle(tx - self.x)
                    self._plan[0] = ("shuffle",)
                    return
                if ad.motion == "slow_walk":
                    self._plan.pop(0)
                    self._plan.insert(0, ("goto_rel", (secs or 3.0) * self.mover.walk_speed * 0.5, "slow"))
                    continue
                if ad.motion in ("walk", "jog", "run", "sprint", "sneak", "chase"):
                    self._plan.pop(0)
                    lo, hi = self.span()
                    dist = (secs or 2.5) * self.mover.level_speed("walk" if ad.motion in ("sneak",) else
                                                                  ("run" if ad.motion == "chase" else ad.motion)) * 0.7
                    d = self.mover.facing
                    if (self.x + d * dist > hi) or (self.x + d * dist < lo):
                        d = -d
                    tx = max(lo, min(hi, self.x + d * dist))
                    gait = {"sneak": "walk", "chase": "run"}.get(ad.motion, ad.motion)
                    self._plan.insert(0, ("goto", tx, self.surface.y if self.surface else self.floor(),
                                          gait in ("run", "sprint"), S.WALKING))
                    if gait in ("jog", "sprint"):
                        self._plan[0] = ("goto_gait", tx, self._plan[0][2], gait)
                    continue
                if ad.motion == "stop":
                    self.mover.brake()
                    self._plan.pop(0)
                    continue
                if ad.motion == "turn":
                    self._plan[0] = ("face", self.x - self.mover.facing * 100)
                    continue
                if ad.track != "full":
                    self._plan.pop(0)          # upper-body / additive: plays alongside the plan
                    continue
                if ad.state is not None:
                    self.mover.brake()
                    self.set_state(ad.state)
                dur = secs if secs is not None else (a.duration if a.duration > 0 else 1.0)
                if ad.loop and secs is None:
                    dur = a.duration
                self._step_until = now + dur
                self._step_action = a
                return
            if kind == "goto_rel":
                _, dist, gait = step
                lo, hi = self.span()
                d = self.mover.facing
                if not (lo <= self.x + d * dist <= hi):
                    d = -d
                tx = max(lo, min(hi, self.x + d * dist))
                self._plan[0] = ("goto_gait", tx, self.surface.y if self.surface else self.floor(), gait)
                continue
            if kind == "turn_around":
                self._plan[0] = ("face", self.x - self.mover.facing * 100)
                continue
            if kind == "goto_gait":
                _, x, y, gait = step
                self._plan.pop(0)
                route = self._route(x, y, gait, False, None)
                self._plan[0:0] = route
                continue
            if kind == "jump":
                _, h, fwd, delay, count = step
                self.mover.brake()
                self._jump = {"h": h, "fwd": fwd, "delay": delay, "count": count, "phase": "wait"}
                if delay <= 0:
                    self._do_jump(self._jump)
                return
            self._plan.pop(0)   # unknown step: skip
        # plan finished
        self._step_until = None
        self._step_action = None
        cb, self._after_plan = self._after_plan, None
        if cb:
            cb()
        elif self.mode == "auto":
            self._step_until = now   # decide next action immediately

    # ------------------------------------------------------- gen 2 directives
    def play_action(self, name: str, seconds: float | None = None, priority: Priority = Priority.SPECIAL_ACTION,
                    then_idle: bool = True) -> bool:
        """Play one library action now (playground, scripted demos, click reactions)."""
        if self.mode == "held" or resolve(name, self.species) is None:
            return False
        steps: list[tuple] = []
        if self.state == S.SLEEPING and name.upper() not in ("SLEEP", "SLEEP_MODE", "WAKE"):
            steps.append(("state", S.WAKING, 0.8))
        steps.append(("action", name, seconds, priority))
        if then_idle:
            steps.append(("state", S.IDLE, 0.3))
        self._start_plan(steps, mode="auto" if self.autonomous else "directed", priority=priority)
        return True

    def locomote(self, gait: str, distance: float, then: Optional[list] = None) -> None:
        """Travel ``distance`` px with a given gait (walk / jog / run / sprint / escalate)."""
        lo, hi = self.span()
        d = self.mover.facing if distance >= 0 else -self.mover.facing
        dist = abs(distance)
        if not (lo <= self.x + d * dist <= hi):
            d = -d
        tx = max(lo, min(hi, self.x + d * dist))
        y = self.surface.y if self.surface is not None else self.floor()
        steps: list[tuple] = []
        if self.state == S.SLEEPING:
            steps.append(("state", S.WAKING, 0.8))
        steps += [("goto_gait", tx, y, gait), ("state", S.IDLE, 0.4)] + list(then or [])
        self._start_plan(steps, mode="auto" if self.autonomous else "directed", priority=Priority.SPECIAL_ACTION)

    def sequence_steps(self, items: list[tuple[str, float | None]]) -> list[tuple]:
        """Plan steps for a playground / demo timeline [(ACTION or GAIT, seconds|None)...]."""
        steps: list[tuple] = []
        gaits = {"WALK": "walk", "JOG": "jog", "RUN": "run", "SPRINT": "sprint", "MECHANICAL_WALK": "walk",
                 "MECHANICAL_RUN": "run", "QUIET_WALK": "walk", "SNEAK": "walk"}
        for name, secs in items:
            n = name.upper().replace(" ", "_")
            if n in gaits:
                if n in ("SNEAK", "QUIET_WALK"):
                    steps.append(("action", n, None, Priority.SPECIAL_ACTION))
                g = gaits[n]
                dist = (secs or 2.5) * self.mover.level_speed(g) * 0.75
                steps.append(("goto_rel", dist, g))
            elif n == "IDLE":
                steps.append(("state", S.IDLE, secs or 1.5))
            elif n == "TURN":
                steps.append(("turn_around",))
            else:
                steps.append(("action", n, secs, Priority.SPECIAL_ACTION))
        return steps

    def play_sequence(self, items: list[tuple[str, float | None]], loop: bool = False) -> None:
        steps = self.sequence_steps(items) + [("state", S.IDLE, 0.5)]
        self.sequence_loop = list(items) if loop else None

        def again():
            if self.sequence_loop:
                self.play_sequence(self.sequence_loop, loop=True)
            elif self.autonomous:
                self.release()
        self._start_plan(steps, mode="auto" if self.autonomous else "directed", after=again,
                         priority=Priority.SPECIAL_ACTION)

    def stop_sequence(self) -> None:
        self.sequence_loop = None

    def _surprise(self, kind: str) -> None:
        """Rare, short, species-specific moments of life."""
        lo, hi = self.span()
        y = self.surface.y if self.surface is not None else self.floor()
        r = self.rng
        P_ = Priority.IDLE
        if kind == "bug_hunt":
            x1 = max(lo, min(hi, self.x + r.choice((-1, 1)) * r.uniform(90, 160)))
            steps = [("action", "LOOK_UP", None, P_), ("state", S.WATCHING, 1.0), ("goto_gait", x1, y, "run"),
                     ("action", "POUNCE", None, P_), ("state", S.LOOKING_AROUND, 1.6), ("action", "LICK_PAW", 2.0, P_)]
        elif kind == "tail_chase":
            steps = [("action", "SPIN", None, P_), ("action", "SPIN", None, P_), ("action", "PLAY_BOW", None, P_),
                     ("action", "TAIL_WAG", None, P_), ("state", S.IDLE, 0.6)]
        elif kind == "stealth":
            x1 = max(lo, min(hi, self.x + self.mover.facing * r.uniform(160, 260)))
            steps = [("action", "SNEAK", None, P_), ("goto", x1, y, False, S.WALKING), ("action", "POUNCE", None, P_),
                     ("action", "TAIL_SWING", None, P_), ("state", S.SITTING, 2.0)]
        elif kind == "binky":
            x1 = max(lo, min(hi, self.x + r.choice((-1, 1)) * r.uniform(200, 320)))
            steps = [("goto_gait", x1, y, "sprint"), ("action", "DOUBLE_HOP", None, P_), ("action", "BOUNCE", None, P_),
                     ("action", "NOSE_TWITCH", None, P_), ("state", S.IDLE, 0.8)]
        elif kind == "glitch":
            steps = [("action", "ERROR", None, P_), ("action", "RECALIBRATE", None, P_),
                     ("action", "SYSTEM_CHECK", None, P_), ("state", S.IDLE, 0.6)]
        else:   # eureka
            steps = [("action", "THINK", None, P_), ("state", S.IDLE, 1.4), ("action", "SURPRISED", None, P_),
                     ("jump", None, 0.0, 0.0, 1), ("action", "CHEER", None, P_)]
        self.events.append(("surprise", 0.0))
        self._start_plan(steps, mode="auto")

    def _do_jump(self, js: dict) -> None:
        h = js["h"] if js["h"] is not None else self.profile.jump_height * self.mover.scale
        fwd = js["fwd"]
        if fwd:
            # keep the landing inside the current surface / screen
            flight = 2 * math.sqrt(2 * 2200.0 * max(1.0, h)) / 2200.0
            lo, hi = self.span()
            room = (hi - self.x) if self.mover.facing > 0 else (self.x - lo)
            fwd = max(0.0, min(fwd, room / max(0.1, flight)))
        self.mover.jump(h, fwd)
        js["phase"] = "air"
        self.events.append(("jump", 0.0))

    # ------------------------------------------------------------- idle life
    def _auto_life(self, now: float) -> None:
        # Sleep when the user is away; wake up when they come back.
        if self.user_idle_seconds >= self.idle_sleep_after and not self.focus_mode:
            if self.state != S.SLEEPING:
                self.mover.stop()
                self.set_state(S.SLEEPING)
                self._was_sleeping_for_idle = True
            return
        if self._was_sleeping_for_idle and self.state == S.SLEEPING:
            self._was_sleeping_for_idle = False
            self.events.append(("wake", 0.0))
            self._start_plan([("state", S.WAKING, 1.0), ("state", S.STRETCHING, 2.2), ("state", S.IDLE, 0.8)],
                             mode="auto")
            return
        if self._cursor_behaviour(now):
            return
        if self._step_until is not None and now < self._step_until:
            return
        if self.mover.moving:
            return
        self._decide(now)

    def _cursor_behaviour(self, now: float) -> bool:
        mode = self.cursor_mode
        c = self.cursor
        if mode == "avoid":
            mode = "run_away"
        if mode == "look" and c is not None:
            return self._cursor_curiosity(now, c)
        if mode in ("ignore", "look") or c is None or now < self._cursor_next:
            return False
        if self.state not in IDLE_INTERRUPTIBLE and self.state != S.PLAYING:
            return False
        l, t, r, b = self.env.screen_bounds(self.x, self.y)
        if not (l <= c[0] <= r and t <= c[1] <= b):
            return False
        lo, hi = self.span()
        dx = c[0] - self.x
        dist = math.hypot(dx, c[1] - (self.y - self.char_size * 0.5))
        moved = self._last_cursor is not None and math.dist(c, self._last_cursor) > 25
        self._last_cursor = c
        self._cursor_next = now + 0.8
        if mode == "follow" and abs(dx) > 220:
            tx = max(lo, min(hi, c[0] - math.copysign(110, dx)))
            self._start_plan([("goto", tx, self.y, False, S.WALKING), ("state", S.IDLE, 1.0)], mode="auto")
            return True
        if mode == "chase" and moved and abs(dx) > 60 and dist < 700:
            tx = max(lo, min(hi, c[0]))
            self._start_plan([("goto", tx, self.y, True, S.RUNNING), ("state", S.PLAYING, 1.2)], mode="auto")
            return True
        if mode == "run_away" and dist < 140:
            away = -1 if dx > 0 else 1
            tx = max(lo, min(hi, self.x + away * self.rng.uniform(220, 380)))
            self._start_plan([("emotion", Emotion.WORRIED, 1.0, 2.0), ("goto", tx, self.y, True, S.RUNNING),
                              ("face", c[0]), ("state", S.IDLE, 1.0)], mode="auto")
            return True
        return False

    def _startle(self, now: float) -> bool:
        """A cursor whooshing right past the head startles skittish characters (rare, with a cooldown)."""
        c = self.cursor
        prev, self._cursor_prev = self._cursor_prev, ((c[0], c[1], now) if c else None)
        if c is None or prev is None or now < self._scare_cd or self.focus_mode:
            return False
        dt = max(1e-3, now - prev[2])
        speed = math.hypot(c[0] - prev[0], c[1] - prev[1]) / dt
        head = (self.x, self.y - self.char_size * 0.55)
        if speed < 2600 or math.dist(c, head) > self.char_size * 0.9:
            return False
        if self.state not in (S.IDLE, S.SITTING, S.LOOKING_AROUND, S.WALKING):
            return False
        self._scare_cd = now + 60.0
        chance = {"rabbit": 0.8, "cat": 0.5, "fox": 0.4}.get(self.species, 0.15)
        chance = min(0.95, chance * (0.6 + 0.8 * getattr(self.profile, "nervousness", 0.3)))
        if self.rng.random() > chance:
            # not a big scare: duck under it instead (everyone flinches a little)
            if self.cursor_interaction != "off" and self.actions.ready("DUCK", now):
                self._start_plan([("action", "DUCK", None, Priority.INTERACTION), ("state", S.IDLE, 0.3)],
                                 mode="auto", priority=Priority.INTERACTION)
                return True
            return False
        self._start_plan([("action", "SCARED", None, Priority.INTERACTION), ("action", "EMBARRASSED", 1.6,
                                                                               Priority.INTERACTION),
                          ("state", S.IDLE, 0.4)], mode="auto", priority=Priority.INTERACTION)
        return True

    def _cursor_curiosity(self, now: float, c: tuple[float, float]) -> bool:
        """LOOK mode with occasional curiosity: a cursor resting near the companion may get
        a closer look (and a paw / sniff / scan) - rare, with a cooldown, never during focus."""
        st = self._cursor_still
        if st is None or math.dist(c, st[:2]) > 12:
            self._cursor_still = (c[0], c[1], now)
            return False
        if self.focus_mode or self.state not in (S.IDLE, S.SITTING, S.LOOKING_AROUND) or \
                self.brain.cooling("curious_cursor", now) or now - st[2] < 2.5:
            return False
        head_y = self.y - self.char_size * 0.5
        dist = math.hypot(c[0] - self.x, c[1] - head_y)
        if dist > 360 or c[1] > self.y + 20:
            return False
        self.brain.note("curious_cursor", now)
        if self.cursor_interaction == "off":
            return False
        if self.rng.random() > 0.35 + 0.4 * self.profile.curiosity:
            return False
        lo, hi = self.span()
        tx = max(lo, min(hi, c[0] - math.copysign(self.char_size * 0.45, c[0] - self.x)))
        act = {"cat": "PAW_AT_CURSOR", "dog": "SNIFF", "fox": "PAW_AT_CURSOR", "rabbit": "SNIFF",
               "robot": "LASER_LOOK", "buddy": "POINT"}.get(self.species, "CURIOUS")
        if self.cursor_interaction == "playful" and self.species in ("cat", "fox") and \
                self.actions.ready("POUNCE", now) and self.life_rng.random() < 0.5:
            act = "POUNCE"                   # playful: wiggle ... pounce at the resting cursor
        y = self.surface.y if self.surface is not None else self.floor()
        self._start_plan([("action", "CURIOUS", None, Priority.INTERACTION), ("goto", tx, y, False, S.WALKING),
                          ("face", c[0]), ("look", A.MOUSE, c, 0.6, None), ("action", act, None, Priority.INTERACTION),
                          ("state", S.IDLE, 1.0)], mode="auto", priority=Priority.INTERACTION)
        return True

    def _base_weights(self) -> dict[str, float]:
        """Personality weights adjusted by the user's settings (before the brain's modifiers)."""
        p = self.profile
        w = dict(p.weights)
        wf = 0.5 + self.walking_frequency
        act = 0.5 + self.idle_activity
        w["walk"] = w.get("walk", 0) * wf
        w["explore"] = w.get("explore", 0) * wf * (0 if self.roam_level == "floor" else 1)
        w["other_monitor"] = w.get("other_monitor", 0) * (1 if self.roam_level == "all" else 0)
        for k in ("play", "run"):
            w[k] = w.get(k, 0) * act
        for k in ("sit", "nap", "yawn"):
            w[k] = w.get(k, 0) * (1.6 - self.idle_activity)
        w["peek"] = w.get("peek", 0) * (1 if self.edge_peeking else 0)
        # personality traits shape the probabilities (not just colours)
        soc = getattr(p, "sociability", 0.5)
        ind = getattr(p, "independence", 0.5)
        w["watch_user"] = w.get("watch_user", 0) * (0.5 + soc)
        w["explore"] = w.get("explore", 0) * (0.6 + 0.8 * ind)
        w["look"] = w.get("look", 0) * (0.8 + 0.6 * getattr(p, "precision", 0.2))
        if self.surface is not None:   # on a window: look around, peek over the edge, sit, or hop down
            w["explore"] = 0
            w["other_monitor"] = 0
            w["hop_down"] = 10.0
            w["peek_edge"] = 5.0 if self.edge_peeking else 0.0
        return w

    def _weights(self) -> list[tuple[str, float]]:
        """Weighted options: personality x settings x energy x context x time of day x cooldowns."""
        now = self._now()
        ctx = "focus" if self.focus_mode else "break" if self.break_mode else "idle"
        if self.focus_mode:
            wf = 0.5 + self.walking_frequency
            base = {"focus": 10.0, "look": 1.2, "watch_user": 1.2, "walk": 0.6 * wf, "stretch": 0.6, "yawn": 0.3}
        elif self.break_mode:
            base = {"play": 4.0, "run": 3.0, "walk": 2.0, "dance": 1.5, "sit": 0.6, "stretch": 1.0, "explore": 1.5}
        else:
            base = self._base_weights()
        if not self.focus_mode and 90 <= self.user_idle_seconds < self.idle_sleep_after:
            base["act:BORED"] = 6.0          # nobody around: gets a little bored
        tod = time_of_day(self.clock_hour) if self.time_of_day_enabled else "day"
        self.brain.use_time_of_day = self.time_of_day_enabled
        if not self.focus_mode and self.window_awareness and self.surface is None and self._hangout_window():
            base["window_hangout"] = 2.5     # lean on / sit beside / peek around a window
        if self.surface is not None and self.window_awareness:
            base["look_over"] = 3.0
        cat = self.mood_state
        for k, m in MOOD_BIAS.get(cat, {}).items():
            if k in base:
                base[k] *= m
        opts = self.brain.weights(base, context=ctx, energy=self.energy.level, now=now, tod=tod,
                                  playfulness=self.profile.playfulness, action_ready=self.actions.ready,
                                  mood=self.mood.norm)
        if self.focus_mode:   # quiet study partner: library actions stay a small minority
            opts = [(k, v * (0.25 if k.startswith("act:") else 1.0)) for k, v in opts]
        return opts

    def _decide(self, now: float) -> None:
        r = self.rng
        if self.surface is None and abs(self.y - self.floor()) > 2:
            fx, fy = self.clamp_to_screen(self.x, self.floor())
            self._start_plan([("goto", fx, fy, False, S.WALKING)], mode="auto")
            return
        ctx = "focus" if self.focus_mode else "break" if self.break_mode else "idle"
        surprise = self.brain.maybe_surprise(now, ctx, self.profile.curiosity) if self.surface is None else None
        if surprise:
            self._surprise(surprise)
            return
        options = self._weights()
        choice = self.brain.choose(options) if options else "idle"
        self.brain.note(choice, now)
        freq = max(0.3, self.profile.idle_frequency)
        if choice == "act:BORED":
            self._start_plan(self.boredom_steps(), mode="auto")
            return
        if choice == "window_hangout":
            steps = self.window_hangout_steps()
            if steps:
                self._start_plan(steps, mode="auto")
                return
            choice = "sit"
        if choice == "look_over" and self.surface is not None:
            edge = self.surface.right - self.half * 0.7 if r.random() < 0.5 else self.surface.left + self.half * 0.7
            lo_, hi_ = self.span()
            edge = max(lo_, min(hi_, edge))
            self._start_plan([("goto", edge, self.surface.y, False, S.WALKING),
                              ("face", edge + (1 if edge > self.x else -1) * 50),
                              ("action", "LOOK_OVER", None, Priority.IDLE), ("state", S.SITTING, r.uniform(2.0, 4.0))],
                             mode="auto")
            return
        if choice.startswith("act:"):
            name = choice[4:]
            if name in VARIANTS:
                name = self.variant(name)
            ad = resolve(name, self.species)
            secs = None
            if ad is not None and ad.loop:
                secs = r.uniform(5, 12) / freq
            elif ad is not None and ad.name in ("LOAF", "LIE_DOWN", "DOWN", "READ", "WRITE", "CHARGE"):
                secs = r.uniform(6, 14) / freq
            self._start_plan([("action", name, secs, Priority.IDLE), ("state", S.IDLE, r.uniform(0.4, 1.0))],
                             mode="auto")
            return
        dur = lambda lo, hi: r.uniform(lo, hi) / freq  # noqa: E731
        lo, hi = self.span()
        sy = self.surface.y if self.surface is not None else self.floor()
        l, t, rgt, btm = self.env.screen_bounds(self.x, self.y)
        if choice in ("walk", "run", "explore", "other_monitor", "peek"):
            kind_map = {"explore": DestinationKind.WINDOW_EDGE, "other_monitor": DestinationKind.OTHER_MONITOR,
                        "peek": DestinationKind.SCREEN_EDGE}
            d = self._destination(now, kind_map.get(choice), lo, hi, sy)
            run = choice == "run" or (choice == "walk" and r.random() < self.profile.run_chance * 0.5) \
                or d.kind == DestinationKind.OTHER_MONITOR
            steps: list[tuple] = [("goto", d.x, d.y, run, S.RUNNING if run else S.WALKING)]
            at_edge = d.kind == DestinationKind.SCREEN_EDGE or abs(d.x - lo) < 30 or abs(d.x - hi) < 30
            corner = at_edge and self.surface is None
            if at_edge and self.edge_peeking:
                variant = r.random()
                if corner and variant < 0.3:
                    # corner: look around, pause, turn, carry on
                    steps += [("state", S.LOOKING_AROUND, dur(1.4, 2.2)), ("state", S.IDLE, dur(0.5, 0.9)),
                              ("face", (l + rgt) / 2), ("action", "TAIL_SWISH", None, Priority.MICRO_MOVEMENT)]
                elif variant < 0.55:
                    # sit at the screen edge and look outside for a while
                    out = hi + 100 if abs(d.x - hi) < abs(d.x - lo) else lo - 100
                    steps += [("face", out), ("state", S.SITTING, dur(3.0, 6.0)),
                              ("look", A.ACTIVE_WINDOW, (out, sy - self.char_size * 0.6), 1.5, None),
                              ("face", (l + rgt) / 2)]
                else:
                    steps += [("state", S.PEEKING, dur(1.8, 2.8)), ("state", S.LOOKING_AROUND, dur(1.0, 1.8)),
                              ("face", (l + rgt) / 2)]
            elif d.kind == DestinationKind.WINDOW_EDGE:
                steps += [("state", S.LOOKING_AROUND, dur(1.5, 2.5)), ("state", S.SITTING, dur(4, 9))]
            elif r.random() < self.profile.curiosity * 0.4:
                steps += [("state", S.LOOKING_AROUND, dur(1.6, 2.6))]
            else:
                steps += [("state", S.IDLE, dur(0.6, 1.6))]
            self._start_plan(steps, mode="auto")
        elif choice == "peek_edge" and self.surface is not None:
            # walk to the edge of the window we're standing on and look over it
            edge = self.surface.right - self.half * 0.7 if r.random() < 0.5 else self.surface.left + self.half * 0.7
            edge = max(lo, min(hi, edge))
            self._start_plan([("goto", edge, sy, False, S.WALKING), ("face", edge + (1 if edge > self.x else -1) * 50),
                              ("state", S.PEEKING, dur(2.0, 3.2)), ("action", "LOOK_UP", None, Priority.IDLE),
                              ("state", S.SITTING, dur(3.0, 6.0))], mode="auto")
        elif choice == "hop_down":
            fx, fy = self.clamp_to_screen(self.x + r.uniform(-150, 150), self.floor())
            self._start_plan([("goto", fx, fy, False, S.WALKING), ("state", S.IDLE, 0.8)], mode="auto")
        elif choice == "sit":
            self._start_plan([("state", S.SITTING, dur(5, 14))], mode="auto")
        elif choice == "look":
            self._start_plan([("state", S.LOOKING_AROUND, dur(2.5, 4.0))], mode="auto")
        elif choice == "watch_user":
            self._start_plan([("look", A.USER, None, dur(2.0, 3.5), S.SITTING if r.random() < 0.5 else S.IDLE)],
                             mode="auto")
        elif choice == "stretch":
            self._start_plan([("state", S.STRETCHING, 2.2)], mode="auto")
        elif choice == "yawn":
            self._start_plan([("state", S.YAWNING, 1.8), ("state", S.IDLE, 0.6)], mode="auto")
        elif choice == "scratch":
            self._start_plan([("state", S.SCRATCHING, dur(1.4, 2.2))], mode="auto")
        elif choice == "groom":
            self._start_plan([("state", S.GROOMING, dur(2.0, 3.5))], mode="auto")
        elif choice == "read":
            self._start_plan([("state", S.READING, dur(8, 20))], mode="auto")
        elif choice == "play":
            self._start_plan([("state", S.PLAYING, dur(2.5, 4.0))], mode="auto")
        elif choice == "nap":
            self._start_plan([("state", S.SITTING, 2.5), ("state", S.YAWNING, 1.7),
                              ("state", S.SLEEPING, r.uniform(15, 40) * (0.5 + self.profile.sleepiness)),
                              ("state", S.WAKING, 1.0), ("state", S.STRETCHING, 2.2), ("state", S.IDLE, 0.8)],
                             mode="auto")
        elif choice == "focus":
            home = self.focus_home
            if home and abs(home[0] - self.x) > 60 and self.stay_near_focus:
                self._start_plan([("goto", home[0], home[1], False, S.WALKING), ("state", S.FOCUS_MODE, dur(20, 50))],
                                 mode="auto")
            else:
                self._start_plan([("state", S.FOCUS_MODE, dur(20, 60))], mode="auto")
        elif choice == "dance":
            self._start_plan([("state", S.CELEBRATING, 2.0)], mode="auto")
        else:
            self._start_plan([("state", S.IDLE, dur(2, 5))], mode="auto")

    # ================================================== living-character upgrade
    @property
    def mood_state(self) -> str:
        now = self._now()
        return self.mood.state(self.energy.level, annoyed=now < self.memory.annoyed_until or
                               len([c for c in self.memory.clicks if now - c < 3.0]) >= 3,
                               idle_seconds=self.user_idle_seconds,
                               curious=self.emotion.current in (Emotion.CURIOUS,))

    def variant(self, name: str) -> str:
        """One of the action's variants, chosen by personality, mood and energy."""
        return pick_variant(name, self.life_rng, mood=self.mood.norm, energy=self.energy.level,
                            playfulness=self.profile.playfulness, species=self.species, ready=self.actions.ready,
                            now=self._now())

    def shout_level(self) -> int:
        """1 small "hey" / 2 normal / 3 big (hop + stomp) - escalates with ignored warnings, never beyond cartoon."""
        n = self.memory.ignored_warnings
        return 1 if n <= 0 and self.mood.value > 50 else (2 if n < 2 else 3)

    def shout_steps(self, level: int = 2) -> list[tuple]:
        level = max(1, min(3, int(level)))
        name = {1: "SHOUT_SOFT", 2: "SHOUT", 3: "SHOUT_BIG"}[level]
        steps: list[tuple] = [("action", name, None, Priority.DISTRACTION_RESPONSE)]
        if level == 3:
            steps.append(("action", "HUFF", None, Priority.DISTRACTION_RESPONSE))
        return steps

    def shout_performance(self, level: int = 2) -> list[tuple]:
        """Shout as a full performance outside an intervention: shout -> breathe -> cool down."""
        return self.shout_steps(level) + [("action", "COOL_DOWN", None, Priority.INTERACTION), ("state", S.IDLE, 0.3)]

    def angry_steps(self, intensity: float = 1.0) -> list[tuple]:
        """Cartoon anger: stomp -> huff (steam) -> arms crossed -> calm down. Playful, never threatening."""
        i = max(0.0, min(1.0, intensity))
        # notice -> freeze, eyes narrow, brows lower, body stiffens, fists clench (GLARE) -> stomp ->
        # tail lashes -> short shout -> heavy breathing (HUFF) -> cool down (deep breath, shoulders drop,
        # look away) -> neutral.  Cartoon anger only.
        steps: list[tuple] = [("emotion", Emotion.ANNOYED, max(0.4, i), 1.0),
                              ("action", "GLARE", None, Priority.INTERACTION),
                              ("emotion", Emotion.ANGRY, max(0.3, i), 3.0),
                              ("action", "ANGRY" if i > 0.7 else "STOMP", None, Priority.INTERACTION)]
        if i > 0.4:
            steps.append(("action", "SHOUT_SOFT", None, Priority.INTERACTION))
        steps += [("action", "HUFF", None, Priority.INTERACTION),
                  ("action", "COOL_DOWN", None, Priority.INTERACTION),
                  ("emotion", Emotion.NEUTRAL, 1.0, None), ("state", S.IDLE, 0.6)]
        return steps

    def scared_steps(self) -> list[tuple]:
        """Startle (jump back) -> hide the face -> tremble -> peek -> relief."""
        return [("action", "SCARED", None, Priority.INTERACTION), ("action", "BACKPEDAL", 0.8, Priority.INTERACTION),
                ("action", "HIDE_FACE", None, Priority.INTERACTION),
                ("action", "TREMBLE", 1.0, Priority.INTERACTION), ("look", A.USER, None, 0.4, S.IDLE),
                ("action", "RELIEVED", None, Priority.INTERACTION), ("action", "COOL_DOWN", None, Priority.INTERACTION),
                ("state", S.IDLE, 0.4)]

    def surprised_steps(self, point: Optional[tuple[float, float]] = None) -> list[tuple]:
        steps: list[tuple] = [("emotion", Emotion.SURPRISED, 1.0, 2.0)]
        if point is not None:
            steps.append(("look", A.NOTIFICATION, point, 0.25, S.IDLE))
        steps += [("action", "SURPRISED", None, Priority.INTERACTION), ("state", S.IDLE, 0.4)]
        return steps

    def laugh_steps(self, intensity: float = 1.0) -> list[tuple]:
        return [("action", "LAUGH_HARD" if intensity > 0.75 else "LAUGH", None, Priority.INTERACTION),
                ("state", S.IDLE, 0.4)]

    def cry_steps(self) -> list[tuple]:
        return [("emotion", Emotion.SAD, 0.8, 4.0), ("action", "SIGH", None, Priority.INTERACTION),
                ("action", "CRY", None, Priority.INTERACTION), ("action", "RELIEVED", None, Priority.INTERACTION),
                ("state", S.IDLE, 0.4)]

    def love_steps(self) -> list[tuple]:
        """Species-specific affection: cat head-bump + purr, dog wag + bounce, others lean in."""
        if self.species == "cat":
            return [("action", "HEAD_BUMP", None, Priority.INTERACTION), ("action", "PURR", None, Priority.INTERACTION),
                    ("state", S.IDLE, 0.4)]
        if self.species == "dog":
            return [("action", "TAIL_WAG", None, Priority.INTERACTION), ("action", "BOUNCE", None, Priority.INTERACTION),
                    ("state", S.IDLE, 0.4)]
        return [("action", "LEAN_TO_CURSOR", None, Priority.INTERACTION), ("action", "LOVE", 1.6, Priority.INTERACTION),
                ("state", S.IDLE, 0.4)]

    def dizzy_steps(self) -> list[tuple]:
        return [("action", "DIZZY", None, Priority.INTERACTION), ("action", "STAGGER", 1.6, Priority.INTERACTION),
                ("state", S.IDLE, 0.6)]

    def boredom_steps(self) -> list[tuple]:
        """Nobody around: bored -> look around -> sigh -> lie down -> (doze off if it goes on)."""
        r = self.life_rng
        steps: list[tuple] = [("action", "BORED", None, Priority.IDLE), ("state", S.LOOKING_AROUND, r.uniform(1.6, 2.6)),
                              ("action", "SIGH", None, Priority.IDLE)]
        lie = "LIE_DOWN" if self.species in ("cat", "fox") else "DOWN" if self.species == "dog" else \
            "LOAF" if self.species == "rabbit" else "SIT"
        steps.append(("action", lie, r.uniform(4.0, 7.0), Priority.IDLE))
        if self.user_idle_seconds > self.idle_sleep_after * 0.6 and self.actions.ready("SLEEP", self._now()):
            steps += [("state", S.YAWNING, 1.7), ("state", S.SLEEPING, r.uniform(20, 40)), ("state", S.WAKING, 1.0)]
        steps.append(("state", S.IDLE, 0.6))
        return steps

    def sulk_steps(self) -> list[tuple]:
        """Funny, not guilt-trippy: huff, turn its back, peek over the shoulder ... then forgive with a nod."""
        return [("action", "HUFF", None, Priority.INTERACTION), ("action", "SULK", 3.2, Priority.INTERACTION),
                ("action", "COOL_DOWN", None, Priority.INTERACTION),
                ("action", "NOD", None, Priority.INTERACTION), ("emotion", Emotion.HAPPY, 0.6, 2.0),
                ("state", S.IDLE, 0.5)]

    def fetch_steps(self) -> list[tuple]:
        """Dog: dash off after an imaginary toy, pounce, trot back proudly, wag."""
        lo, hi = self.span()
        y = self.surface.y if self.surface is not None else self.floor()
        d = self.mover.facing if lo + 300 < self.x + self.mover.facing * 280 < hi - 20 else -self.mover.facing
        x1 = max(lo, min(hi, self.x + d * 280 * self.mover.scale))
        x0 = self.x
        return [("action", "PLAY_BOW", None, Priority.IDLE), ("goto_gait", x1, y, "run"),
                ("action", "POUNCE", None, Priority.IDLE), ("action", "SHAKE", None, Priority.IDLE),
                ("goto_gait", x0, y, "jog"), ("face", x0 + d * 100), ("action", "PROUD", 1.2, Priority.IDLE),
                ("action", "TAIL_WAG", None, Priority.IDLE), ("state", S.IDLE, 0.6)]

    EMOTION_SEQUENCES = ("shout", "angry", "scared", "surprised", "laughing", "crying", "love", "boredom", "sulk",
                         "dizzy", "fetch")

    def perform_sequence(self, name: str, intensity: float = 1.0) -> bool:
        """Play one of the upgraded emotional sequences (Playground / showcase / app)."""
        n = name.lower()
        steps = {"shout": lambda: self.shout_performance(1 + int(round(max(0.0, min(1.0, intensity)) * 2))),
                 "fetch": self.fetch_steps,
                 "angry": lambda: self.angry_steps(intensity), "scared": self.scared_steps,
                 "surprised": self.surprised_steps, "laughing": lambda: self.laugh_steps(intensity),
                 "crying": self.cry_steps, "love": self.love_steps, "boredom": self.boredom_steps,
                 "sulk": self.sulk_steps, "dizzy": self.dizzy_steps}.get(n)
        if steps is None or self.mode == "held":
            return False
        plan = steps()
        if self.state == S.SLEEPING and n != "boredom":
            plan.insert(0, ("state", S.WAKING, 0.6))
        self._start_plan(plan + [("state", S.IDLE, 0.3)], mode="auto" if self.autonomous else "directed",
                         priority=Priority.SPECIAL_ACTION)
        return True

    def warning_ignored(self, target: str = "") -> None:
        """The user waved a warning away. Mood dips a little; after repeated ignored warnings for the
        same thing the companion sulks for a few seconds (comic, then it forgives). Otherwise just roam."""
        now = self._now()
        n = self.memory.warned(now, target or "?")
        self.mood.event("ignored", now)
        if n >= 2 and not self.focus_mode and self.mode != "held":
            self.target_rect = None
            self.attention.clear(A.DISTRACTION_WINDOW)
            self._start_plan(self.sulk_steps(), mode="auto", after=self.release, priority=Priority.INTERACTION)
            return
        self.release()

    # -------------------------------------------------- environment awareness
    def _hangout_window(self) -> Optional[dict]:
        """A window whose side edge reaches down near the floor close by (to lean on / sit beside / peek)."""
        if not self.world_known():
            return None
        fy = self.floor()
        lo, hi = self.span()
        best, bd = None, 1e9
        for w in self.windows():
            if w.get("minimized"):
                continue
            bottom = w["y"] + w["h"]
            if bottom < fy - self.char_size * 1.6 or w["y"] > fy - self.char_size * 0.5:
                continue                     # its lower edge isn't down where the companion walks
            for ex in (w["x"], w["x"] + w["w"]):
                if lo + self.half < ex < hi - self.half:
                    d = abs(ex - self.x)
                    if d < bd and d < 900:
                        best, bd = w, d
        return best

    def window_hangout_steps(self) -> list[tuple]:
        """Sit beside / lean against / peek around the side of a nearby window. Never touches it."""
        w = self._hangout_window()
        if w is None:
            return []
        r = self.life_rng
        fy = self.floor()
        left, right = w["x"], w["x"] + w["w"]
        edge = left if abs(left - self.x) < abs(right - self.x) else right
        outside = -1 if edge == left else 1                  # the side away from the window
        x = edge + outside * self.half * 0.9
        lo, hi = self.span()
        x = max(lo, min(hi, x))
        kind = r.choice(("lean", "sit", "peek"))
        steps: list[tuple] = [("goto", x, fy, False, S.WALKING)]
        if kind == "lean":
            steps += [("face", x + outside * 80), ("action", "LEAN", r.uniform(4.0, 7.0), Priority.IDLE)]
        elif kind == "sit":
            steps += [("face", edge), ("state", S.SITTING, r.uniform(3.0, 6.0))]
        else:
            steps += [("face", edge), ("action", "PEEK_BEHIND", None, Priority.IDLE)]
        steps.append(("state", S.IDLE, 0.6))
        return steps

    def _window_awareness(self, now: float) -> bool:
        """React when a window is dragged quickly toward the companion: surprised hop back out of the way."""
        if not self.window_awareness or not self.world_known() or self.mover.airborne:
            self._win_prev = {}
            return False
        cur = {}
        hit = None
        head_y = self.y - self.char_size * 0.5
        for w in self.windows():
            h = w.get("handle")
            if h is None or w.get("minimized"):
                continue
            cur[h] = (w["x"], w["y"], w["w"], w["h"], now)
            prev = self._win_prev.get(h)
            if prev is None or now - prev[4] <= 1e-3 or hit is not None:
                continue
            dt = now - prev[4]
            vx = (w["x"] - prev[0]) / dt
            if abs(vx) < 700 or not (w["y"] - 10 <= head_y <= w["y"] + w["h"] + 10):
                continue
            edge = w["x"] + w["w"] if vx > 0 else w["x"]            # leading edge
            gap = (self.x - edge) * (1 if vx > 0 else -1)
            if 0 < gap < self.char_size * 1.2:
                hit = (vx, edge)
        new = [h for h in cur if h not in self._win_prev]
        if self._win_prev and new and now >= self._open_notice_cd and not self.focus_mode:
            # a window just opened: ears perk and the eyes / head flick toward it (no interruption)
            w = next((w for w in self.windows() if w.get("handle") == new[0]), None)
            if w is not None and w["w"] * w["h"] > 60_000:
                self._open_notice_cd = now + 10.0
                self.attention.set(A.NOTIFICATION, (w["x"] + w["w"] / 2, w["y"] + w["h"] / 2), 1.4, now=now)
                self.actions.play("NOTICE", now, Priority.MICRO_MOVEMENT)
        self._win_prev = cur
        if hit is None or now < self._win_react_cd or self.surface is not None:
            return False
        self._win_react_cd = now + 8.0
        vx, edge = hit
        lo, hi = self.span()
        away = 1 if vx > 0 else -1
        tx = max(lo, min(hi, self.x + away * self.char_size * 1.3))
        self.memory.interact(now)
        self._start_plan([("look", A.NOTIFICATION, (edge, self.y - self.char_size * 0.5), 0.15, S.IDLE),
                          ("action", "SURPRISED", None, Priority.INTERACTION),
                          ("goto", tx, self.y, True, S.RUNNING), ("face", edge),
                          ("action", "HUFF" if self.angry_reactions else "NOD", None, Priority.INTERACTION),
                          ("state", S.IDLE, 0.6)], mode="auto", priority=Priority.INTERACTION)
        return True

    def _cursor_world(self, now: float) -> bool:
        """The cursor as something in the world: it approaches -> notice; it circles the character ->
        curious; it keeps coming back fast -> duck. Subtle by default, never during focus."""
        c = self.cursor
        if c is None or self.cursor_interaction == "off" or self.focus_mode:
            self._cursor_hist = []
            return False
        head = (self.x, self.y - self.char_size * 0.5)
        self._cursor_hist.append((now, c[0], c[1]))
        self._cursor_hist = [h for h in self._cursor_hist if now - h[0] < 4.0]
        if now < self._cursor_world_cd or self.state not in (S.IDLE, S.SITTING, S.LOOKING_AROUND):
            return False
        dist = math.hypot(c[0] - head[0], c[1] - head[1])
        # circling: unwrapped angle around the head while it stays close
        if dist < self.char_size * 2.2:
            ang = math.atan2(c[1] - head[1], c[0] - head[0])
            if self._circle_prev is None or now - self._circle_t0 > 6.0:
                self._circle_acc, self._circle_t0 = 0.0, now
            else:
                da = (ang - self._circle_prev + math.pi) % (2 * math.pi) - math.pi
                self._circle_acc += da
            self._circle_prev = ang
            if abs(self._circle_acc) > 2 * math.pi * 0.95:
                self._circle_acc, self._circle_prev = 0.0, None
                self._cursor_world_cd = now + 30.0
                steps = [("look", A.MOUSE, c, 0.3, None), ("action", "CURIOUS", None, Priority.INTERACTION)]
                if self.species == "dog" and self.cursor_interaction == "playful":
                    steps += [("action", "SPIN", None, Priority.INTERACTION)]
                elif self.species in ("cat", "fox") and self.cursor_interaction == "playful":
                    steps += [("action", "PAW_AT_CURSOR", None, Priority.INTERACTION)]
                self._start_plan(steps + [("state", S.IDLE, 0.4)], mode="auto", priority=Priority.INTERACTION)
                return True
        else:
            self._circle_prev = None
        # approach: closed a lot of distance quickly -> notice (ears perk, eyes on it)
        old = next((h for h in self._cursor_hist if now - h[0] <= 0.6), None)
        if old is not None and dist < self.char_size * 1.4:
            d0 = math.hypot(old[1] - head[0], old[2] - head[1])
            if d0 - dist > self.char_size * 1.0:
                self._cursor_world_cd = now + 12.0
                self.attention.set(A.MOUSE, c, 1.2, now=now)
                self.actions.play("NOTICE", now, Priority.MICRO_MOVEMENT)
                return False
        return False

    def _maybe_stumble(self, now: float, dt: float) -> bool:
        """Very rarely trip while walking / running (never in focus mode, never while carrying out a task)."""
        if not self.stumbles or self.focus_mode or self.state not in (S.WALKING, S.RUNNING) or self.mover.airborne:
            return False
        if self.life_rng.random() > self.stumble_rate * dt * (1.0 + self.mover.gait_level * 0.5):
            return False
        roll = self.life_rng.random()
        fall = roll < 0.08 and self.actions.ready("FALL_OVER", now)
        slip = not fall and roll < 0.35 and self.actions.ready("SLIP", now)
        name = "FALL_OVER" if fall else ("SLIP" if slip else "STUMBLE")
        if not self.actions.ready(name, now):
            return False
        # remember where it was going: walking -> fall -> recover -> walking
        dest = next((st for st in reversed(self._plan) if st[0] == "move"), None)
        rest = [st for st in self._plan if st[0] not in ("move", "leap")]
        self.mover.brake()
        steps: list[tuple] = [("action", name, None, Priority.INTERACTION)]
        if fall:
            steps += [("action", "RECOVER", None, Priority.INTERACTION), ("action", "SHAKE_OFF", None, Priority.INTERACTION)]
        elif slip:
            steps.append(("action", "SHAKE_OFF", None, Priority.INTERACTION))
        if dest is not None:
            steps.append(("goto", dest[1], dest[2], dest[3] in ("run", "sprint"), S.WALKING))
            steps += rest
        else:
            steps.append(("state", S.IDLE, 0.5))
        self._start_plan(steps, mode="auto", priority=Priority.INTERACTION)
        self.events.append(("stumble", 0.0))
        return True

    def _destination(self, now: float, forced: Optional[DestinationKind], lo: float, hi: float, sy: float):
        from .roaming import Destination
        l, t, r, b = self.env.screen_bounds(self.x, self.y)
        wins = [w for w in self.windows() if w.get("perchable")]
        cursor = self.cursor
        if forced == DestinationKind.SCREEN_EDGE:
            edge = 1 if abs(self.x - hi) < abs(self.x - lo) else -1
            if self.rng.random() < 0.4:
                edge = -edge
            return Destination(forced, hi if edge > 0 else lo, sy, edge=edge)
        if forced is None and self.edge_bias > 0 and self.surface is None and \
                self.life_rng.random() < self.edge_bias:
            # "where they spend time": near a screen edge / corner rather than mid-desktop
            near_right = abs(self.x - hi) < abs(self.x - lo)
            side = (1 if near_right else -1) if self.life_rng.random() < 0.65 else (-1 if near_right else 1)
            off = self.life_rng.uniform(0.3, 1.8) * self.char_size
            x = max(lo, min(hi, (hi - off) if side > 0 else (lo + off)))
            return Destination(DestinationKind.RANDOM_SAFE_POINT, x, sy, edge=side)
        if forced is not None:
            d = self.roaming._make(forced, (self.x, self.y), lo, hi, sy, (l, t, r, b), self.env.all_bounds(),
                                   wins, cursor, self.active_window, self.half, self.char_size)
            if d is not None:
                self.roaming.remember(d, now)
                return d
        return self.roaming.choose(now=now, pos=(self.x, self.y), span=(lo, hi), surface_y=sy, screen=(l, t, r, b),
                                   screens=self.env.all_bounds(), windows=wins, cursor=cursor,
                                   active_window=self.active_window, level=self.roam_level, half=self.half,
                                   char_h=self.char_size)
