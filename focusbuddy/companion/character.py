"""Companion controller: one coordinated 60 FPS loop for the desktop creature.

Per frame (single precise QTimer - no per-part timers, no threads):

    delta time -> input (cursor, idle, petting) -> Behavior (state, plan, locomotion,
    emotion, attention) -> AnimationController (gait, emotion, springs, blink)
    -> place window with sub-pixel offset -> render -> bubble / debug follow

Sub-pixel placement: the overlay window can only move by whole pixels, so the
fractional remainder is applied while painting.  Together with the precise
timer this removes the judder of slow walks.
"""

from __future__ import annotations

import math
import random
import sys
import time
from typing import Callable, Optional

from PySide6.QtCore import QObject, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QGuiApplication

from ..utils.logging_setup import get_logger
from .animation import AnimationController
from .attention import AttentionTarget
from .behavior import Behavior, Rect
from .life import Life
from .characters import CharacterSpec
from .overlay import BubbleWindow, CompanionWindow
from .particles import ParticleSystem
from .profile import get_profile
from .renderer import GROUND_MARGIN, UNITS, make_renderer
from .states import MOVING_STATES
from .states import CompanionState as S

log = get_logger("companion")

BASE_SIZE = 124
PAD = 2   # legacy constant (minimum padding for sub-pixel drawing)

# Extra room around the character inside its transparent window, as fractions of the
# character size: shadow while airborne, dust, confetti, the robot's scan beam.
# (Fully transparent pixels are click-through on Windows.)
PAD_SIDE, PAD_TOP, PAD_BOTTOM = 0.32, 0.22, 0.14


def window_pads(px: int) -> tuple[int, int, int, int]:
    return (PAD + int(px * PAD_SIDE), PAD + int(px * PAD_TOP), PAD + int(px * PAD_SIDE), PAD + int(px * PAD_BOTTOM))



QUADRUPEDS = ("cat", "dog", "fox", "rabbit")


def classify_click(alpha, x: float, y: float, species: str, facing: int = 1) -> Optional[str]:
    """The part of the companion under (x, y) of a frame whose alpha channel is ``alpha`` (H, W):
    people and the robot have their head on top; the animals stand side-on, so the front of the
    body (where it faces) is the head and the far back is the tail. None when nothing was hit."""
    import numpy as np
    m = np.asarray(alpha) > 40
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    fx = (x - x0) / max(1.0, x1 - x0)
    fy = (y - y0) / max(1.0, y1 - y0)
    if species in QUADRUPEDS:
        front = fx if facing >= 0 else 1.0 - fx
        if front > 0.6 and fy < 0.62:
            return "head"
        if front < 0.28:
            return "tail"
        return "body"
    return "head" if fy < 0.24 else "body"

class QtEnvironment:
    """Screen geometry (multi-monitor aware) + known windows, in Qt logical coordinates."""

    def __init__(self):
        self._windows: list[dict] | None = None    # None = world not observed

    def _screens(self):
        return QGuiApplication.screens() or []

    def all_bounds(self) -> list[tuple[float, float, float, float]]:
        out = []
        for s in self._screens():
            g = s.availableGeometry()
            out.append((g.left(), g.top(), g.left() + g.width(), g.top() + g.height()))
        return out or [(0, 0, 1280, 720)]

    def _screen_at(self, x: float, y: float):
        screen = QGuiApplication.screenAt(QPoint(int(x), int(y) - 2))
        if screen is None:
            best, bd = None, 1e18
            for s in self._screens():
                c = s.availableGeometry().center()
                d = (c.x() - x) ** 2 + (c.y() - y) ** 2
                if d < bd:
                    best, bd = s, d
            screen = best
        return screen

    def screen_bounds(self, x: float, y: float) -> tuple[float, float, float, float]:
        screen = self._screen_at(x, y)
        if screen is None:
            return (0, 0, 1280, 720)
        g = screen.availableGeometry()
        return (g.left(), g.top(), g.left() + g.width(), g.top() + g.height())

    def screen_info(self, x: float, y: float) -> dict:
        screens = self._screens()
        s = self._screen_at(x, y)
        if s is None:
            return {"index": 0, "name": "?", "dpr": 1.0, "primary": True}
        return {"index": screens.index(s) + 1 if s in screens else 0, "name": s.name(),
                "dpr": s.devicePixelRatio(), "primary": s == QGuiApplication.primaryScreen(),
                "count": len(screens)}

    def windows(self) -> list[dict] | None:
        return self._windows

    def set_windows(self, windows: list[dict] | None) -> None:
        self._windows = windows


def cadence_fps(base: int, state: str, visible: bool, busy: bool) -> int:
    """Render frequency policy: full rate while something happens, ~30 FPS idle, 12 FPS asleep,
    nothing while hidden / minimised (an event - waking, a reaction - makes it busy again)."""
    if not visible:
        return 0
    if busy:
        return base
    if state == "SLEEPING":
        return min(base, 12)
    return min(base, 30)


class Companion(QObject):
    clicked = Signal()
    double_clicked = Signal()
    context_menu = Signal(QPoint)
    bubble_action = Signal(str)
    bubble_dismissed = Signal()
    dropped = Signal()
    grabbed = Signal()
    petted = Signal()
    resize_requested = Signal(int)
    sound = Signal(str)
    emote = Signal(str)        # dialogue cue from the emotion pack: SHOUT, POKED, WOKEN, DIZZY ...

    def __init__(self, spec: CharacterSpec, *, size_factor: float = 1.0, always_on_top: bool = True,
                 idle_provider: Callable[[], float] | None = None, env=None, seed: int | None = None,
                 personality: str = ""):
        super().__init__()
        self.spec = spec
        self.render_mode = "2d"
        self.render_quality = "normal"
        self.renderer = make_renderer(spec, self.render_mode, self.render_quality)
        self.env = env or QtEnvironment()
        self.personality = personality
        self.profile = self._profile_for(spec, personality)
        self.animator = AnimationController(seed, self.profile)
        self._attach_rig()
        self.size_factor = size_factor
        self.px = int(BASE_SIZE * size_factor)
        self.behavior = Behavior(self.env, char_size=self.px, seed=seed, profile=self.profile)
        self.behavior.set_species(self.animator.species)
        self.behavior.press_tip = self.renderer.press_point()
        self.particles = ParticleSystem(seed)
        self.debug_draw: dict | None = None     # skeleton / IK / contacts drawn on the character
        self._last_step_sound = 0.0
        self._tick_errors = 0
        self._last_paint = 0.0
        self.auto_quality = True        # step 3D quality down when frames keep missing their budget
        self.render_fallback = ""       # "2d" after a runtime 3D failure
        self.shadows = True
        self.behavior.on_land = self._on_land
        self.idle_provider = idle_provider
        self.speech_enabled = True
        self.visible_wanted = True
        self.intensity = 1.0
        self.click_reaction = "random"
        self.petting_enabled = True
        self.footsteps = False
        self.target_fps = 60
        self.debug = None
        self._hovered = False
        self._pose = None
        self._last = time.monotonic()
        self._last_raise = 0.0
        self._idle_check = 0.0
        self._fps = 60.0
        self._frac = (0.0, 0.0)
        self._size_anim: Optional[tuple[float, float, float]] = None   # from, to, t0
        self._pet_dir = 0
        self._pet_flips: list[float] = []
        self._pet_last_x: Optional[float] = None
        self._press_timeout: Optional[QTimer] = None
        self._rng = random.Random(seed)
        self._timer_res = False

        self.window = CompanionWindow(self._win_size(), always_on_top)
        self.window.painter_cb = self._paint
        self.window.clicked.connect(self._on_click)
        self.window.double_clicked.connect(self.double_clicked.emit)
        self.window.context_menu.connect(self.context_menu.emit)
        self.window.resize_steps.connect(self.resize_requested.emit)
        self.window.drag_started.connect(self._on_drag_start)
        self.window.dragged.connect(self._on_drag)
        self.window.drag_finished.connect(self._on_drop)
        self.window.hovered.connect(self._on_hover)

        self.bubble = BubbleWindow()
        self.bubble.action.connect(self.bubble_action.emit)
        self.bubble.dismissed.connect(self.bubble_dismissed.emit)

        # the companion's life: routine, entrances / exits, antics, the close scene, memory (life.py)
        self.life_hidden = False            # off-screen for a while / hidden for a full-screen app
        self.life = Life(self.behavior, char_id=spec.id, seed=seed)
        self.life.say = lambda text, secs=None: self.say(text, duration=secs)
        self.life.ask = lambda text, buttons: self.ask(text, buttons)
        self.life.hide_bubble = lambda: self.bubble.fade_out() if self.bubble.isVisible() else None
        self.life.set_shown = self._life_show

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self._place_window()

    def _win_size(self) -> tuple[int, int]:
        l, t, r, b = window_pads(self.px)
        return self.px + l + r, self.px + t + b

    def _attach_rig(self) -> None:
        if hasattr(self.renderer, "shadows"):
            self.renderer.shadows = getattr(self, "shadows", True)
        rig = getattr(self.renderer, "rig", None)
        self.animator.set_rig(rig)

    @staticmethod
    def _profile_for(spec: CharacterSpec, personality: str = ""):
        base = get_profile(spec.id if spec.id in ("cat", "dog", "fox", "rabbit", "robot", "buddy")
                           else spec.personality)
        if personality and personality != base.id:
            p = get_profile(personality)
            # behaviour from the chosen personality, body mechanics from the character
            return p.scaled(gait=base.gait, stride_walk=base.stride_walk, stride_run=base.stride_run,
                            tail_swing=base.tail_swing, ear_response=base.ear_response, id=base.id)
        return base

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.visible_wanted and not self.life_hidden:
            self.window.show()
        self._last = time.monotonic()
        self._set_timer_resolution(True)
        self.timer.start(int(1000 / self.target_fps))

    def stop(self) -> None:
        self.timer.stop()
        self._set_timer_resolution(False)
        self.window.hide()
        self.bubble.hide_now()
        if self.debug:
            self.debug.close_all()

    def _set_timer_resolution(self, on: bool) -> None:
        """Windows' default 15.6 ms timer granularity makes 60 FPS stutter; request 1 ms while running."""
        if not sys.platform.startswith("win") or on == self._timer_res:
            return
        try:
            import ctypes
            winmm = ctypes.WinDLL("winmm")
            (winmm.timeBeginPeriod if on else winmm.timeEndPeriod)(1)
            self._timer_res = on
        except Exception:  # noqa: BLE001
            pass

    def set_visible(self, on: bool) -> None:
        self.visible_wanted = on
        if on:
            if not self.life_hidden:
                self.window.show()
            if not self.timer.isActive():
                self.start()
        else:
            self.window.hide()
            self.bubble.hide_now()

    def _life_show(self, on: bool) -> None:
        """Life hides the companion while it's off-screen / a full-screen app runs (the user's own
        show / hide choice stays in ``visible_wanted``)."""
        self.life_hidden = not on
        if on and self.visible_wanted:
            self.window.show()
        elif not on:
            self.window.hide()
            if not self.life.asking:
                self.bubble.hide_now()

    # ------------------------------------------------------------- settings
    def set_character(self, spec: CharacterSpec) -> None:
        self.spec = spec
        self.renderer = make_renderer(spec, self.render_mode, self.render_quality)
        self.set_personality(self.personality)
        self._attach_rig()
        self.behavior.set_species(self.animator.species)
        self.behavior.press_tip = self.renderer.press_point()
        self.life.set_character(spec.id)
        self.window.update()
        log.info("Character changed id=%s", spec.id)

    def set_personality(self, personality: str) -> None:
        self.personality = personality
        self.profile = self._profile_for(self.spec, personality)
        self.animator.set_profile(self.profile)
        self.behavior.set_profile(self.profile)

    def set_size(self, factor: float, animate: bool = True) -> None:
        from ..config import SIZE_RANGE
        factor = max(min(0.5, SIZE_RANGE[0]), min(SIZE_RANGE[1], factor))      # (it was capped at 2.0)
        if animate and self.timer.isActive():
            self._size_anim = (self.size_factor, factor, time.monotonic())
        else:
            self._apply_size(factor)

    def _apply_size(self, factor: float) -> None:
        self.size_factor = factor
        self.px = int(round(BASE_SIZE * factor))
        self.behavior.set_size(self.px)
        self.window.set_size(self._win_size())
        self._place_window()

    def set_opacity(self, value: float) -> None:
        self.window.setWindowOpacity(max(0.2, min(1.0, value)))

    def set_speed(self, factor: float) -> None:
        self.behavior.user_speed = max(0.3, min(2.5, factor))

    def set_intensity(self, value: float) -> None:
        self.intensity = max(0.2, min(1.6, value))

    def set_click_through(self, on: bool) -> None:
        self.window.set_click_through(on)

    def set_always_on_top(self, on: bool) -> None:
        self.window.set_always_on_top(on)

    def set_fps(self, fps: int) -> None:
        self.target_fps = max(20, min(144, int(fps)))
        if self.timer.isActive():
            self.timer.setInterval(int(1000 / self.target_fps))

    # GPU tiers render every frame the window moves (smooth motion); adaptive quality steps down
    # if a machine can't keep up
    RENDER_FPS = {"gpu": {"low": 30, "medium": 60, "normal": 60, "high": 60, "ultra": 60},
                  "cpu": {"low": 20, "medium": 24, "normal": 24, "high": 30, "ultra": 30}}
    QUALITY_ORDER = ("low", "medium", "high", "ultra")

    def set_render_mode(self, mode: str, quality: str = "normal") -> None:
        """"production" (skinned GLB characters), "3d" (procedural 3D) or "2d". Same skeleton / animation.
        Quality "normal" = automatic: medium on the CPU renderer, high when the GPU renderer is available."""
        mode = mode if mode in ("illustrated", "production", "3d") else "2d"
        if mode == self.render_mode and quality == self.render_quality:
            return
        self.render_mode, self.render_quality = mode, quality
        self.renderer = make_renderer(self.spec, mode, quality)
        if quality in ("normal", "auto") and getattr(self.renderer, "backend", None) == "gpu" and \
                hasattr(self.renderer, "set_quality"):
            self.renderer.set_quality("high")
        self._attach_rig()
        self.behavior.press_tip = self.renderer.press_point()
        self.window.update()
        log.info("Renderer mode=%s quality=%s backend=%s", mode, quality, getattr(self.renderer, "backend", "2d"))

    @property
    def render_fps(self) -> int:
        """3D frames are rendered at most this often; the window moves together with each frame."""
        backend = getattr(self.renderer, "backend", None)
        if backend is None:
            return self.target_fps
        table = self.RENDER_FPS["gpu" if str(backend).startswith("gpu") else "cpu"]
        return min(self.target_fps, table.get(self.render_quality, 24))

    @property
    def paint_fps(self) -> int:
        """Actual repaint rate now: ``render_fps`` while busy, lower idle / asleep, 0 while hidden."""
        base = self.render_fps
        b = self.behavior
        busy = b.state in MOVING_STATES or b.mover.moving or bool(b.actions.active) or self._hovered
        visible = self.window.isVisible() and not self.window.isMinimized()
        return cadence_fps(base, b.state.name, visible, busy)

    LEVELS = {"low": 0, "normal": 1, "high": 2}

    def set_motion_settings(self, *, animation_smoothness: str = "normal", movement_smoothness: str = "normal",
                            secondary_motion: str = "normal", physics_strength: str = "normal",
                            character_energy: str = "normal", particles: bool = True,
                            time_of_day: bool = True, surprises: bool = True) -> None:
        """Advanced animation settings (Low / Normal / High; Normal is the default)."""
        lv = lambda v: self.LEVELS.get(str(v).lower(), 1)  # noqa: E731
        a = self.animator
        a.smoothness = (0.65, 1.0, 1.35)[lv(animation_smoothness)]
        a.secondary_amount = (0.5, 1.0, 1.4)[lv(secondary_motion)]
        a.physics = (0.6, 1.0, 1.3)[lv(physics_strength)]
        b = self.behavior
        b.mover.movement_smoothness = (0.75, 1.0, 1.3)[lv(movement_smoothness)]
        b.energy_setting = (-15.0, 0.0, 15.0)[lv(character_energy)]
        b.time_of_day_enabled = bool(time_of_day)
        b.brain.surprises = bool(surprises)
        self.particles.enabled = bool(particles)
        if not particles:
            self.particles.clear()

    def set_windows(self, windows: list[dict] | None) -> None:
        if hasattr(self.env, "set_windows"):
            self.env.set_windows(windows)

    def set_debug(self, on: bool) -> None:
        if on and self.debug is None:
            from .debug import DebugOverlay
            self.debug = DebugOverlay(self)
        elif not on and self.debug is not None:
            self.debug.close_all()
            self.debug = None

    # -------------------------------------------------------------- speech
    def say(self, text: str, duration: float | None = None, title: str = "") -> None:
        if not text or not self.speech_enabled or not self.visible_wanted or self.life_hidden:
            return
        self.bubble.show_message(text, None, duration, title=title)
        self._place_bubble()
        self.behavior.attention.set(AttentionTarget.NOTIFICATION, None, 0.0)
        self._talk(text, duration)

    def _talk(self, text: str, duration: float | None = None) -> None:
        """Lip motion, nods and brow accents that follow the words in the bubble."""
        try:
            self.animator.talk(text, time.monotonic(), duration)
        except Exception:  # noqa: BLE001 - talking is decoration and must never break speech
            pass

    def ask(self, text: str, buttons: list[tuple[str, str, bool]], title: str = "", accent: str | None = None) -> None:
        """Bubbles with buttons are always shown - even if chatter is off or the companion
        is hidden - because they need an answer (the user must always be able to override)."""
        self.bubble.show_message(text, buttons, None, title=title, accent=accent)
        self._place_bubble()
        if self.visible_wanted and not self.life_hidden:
            self._talk(text, 4.0)

    def countdown(self, remaining: float, total: float, text: str | None = None) -> None:
        if text:
            self.bubble.set_text(text)
        self.bubble.set_countdown(remaining, total)

    def hide_bubble(self) -> None:
        self.bubble.fade_out()

    # ------------------------------------------------------------ directives
    def notice(self, rect: Optional[Rect]) -> None:
        self.behavior.notice(rect)

    def investigate(self, rect: Optional[Rect], run: bool, arrive_state: S = S.WARNING,
                    on_ready: Callable[[], None] | None = None, gait: str | None = None, mood: str = "neutral") -> None:
        self.behavior.investigate(rect, run, arrive_state, on_ready=on_ready, gait=gait, mood=mood)

    def prepare_press(self, rect: Optional[Rect], shout: bool = False) -> None:
        self.behavior.prepare_press(rect, shout=shout)

    def play_emotion(self, name: str, hold: Optional[S] = None, seconds: float | None = None) -> None:
        """Play an emotion-pack action (ANGRY, LOVE, DIZZY ...), optionally returning to a held state."""
        self.behavior.emote(name, hold, seconds)

    def press_close(self, rect: Optional[Rect], on_impact: Callable[[], None], timeout: float = 7.0) -> None:
        """Animate pressing the window's close button; ``on_impact`` runs exactly once
        (at paw contact, or after ``timeout`` if the animation cannot complete)."""
        fired = {"done": False}

        def fire():
            if fired["done"]:
                return
            fired["done"] = True
            if self._press_timeout is not None:
                self._press_timeout.stop()
            on_impact()

        if not self.visible_wanted or not self.timer.isActive() or self.behavior.mode == "held":
            fire()
            return
        self._press_timeout = QTimer(self)
        self._press_timeout.setSingleShot(True)
        self._press_timeout.timeout.connect(fire)
        self._press_timeout.start(int(timeout * 1000))
        self.behavior.press_close(rect, fire)

    def hold(self, state: S) -> None:
        self.behavior.hold(state)

    def perform(self, state: S, seconds: float, then_home: bool = True) -> None:
        self.behavior.perform(state, seconds, then_home)

    def release(self) -> None:
        self.behavior.release()

    def set_shadows(self, on: bool) -> None:
        self.shadows = bool(on)
        if hasattr(self.renderer, "shadows"):
            self.renderer.shadows = self.shadows

    def warning_ignored(self, target: str = "") -> None:
        self.behavior.warning_ignored(target)

    def perform_sequence(self, name: str, intensity: float = 1.0) -> bool:
        return self.behavior.perform_sequence(name, intensity)

    def celebrate_return(self) -> None:
        self.behavior.celebrate_return()

    def focus_start(self) -> None:
        self.behavior.focus_start()

    def focus_complete(self) -> None:
        self.behavior.focus_complete()

    def glance(self, x: float, y: float, seconds: float = 2.0) -> None:
        self.behavior.glance((x, y), AttentionTarget.ACTIVE_WINDOW, seconds)

    def set_focus_mode(self, on: bool) -> None:
        self.behavior.focus_mode = on
        if on:
            self.behavior.break_mode = False

    def set_break_mode(self, on: bool) -> None:
        self.behavior.break_mode = on

    @property
    def state(self) -> S:
        return self.behavior.state

    def feet(self) -> QPointF:
        return QPointF(self.behavior.x, self.behavior.y)

    # ------------------------------------------------------------- events
    def _clicked_part(self) -> Optional[str]:
        """Which part of the companion the last click hit ("head", "body", "tail"), from the
        pressed point and the opaque pixels of the frame on screen."""
        try:
            import numpy as np
            from PySide6.QtGui import QImage
            fr = self.window.frame
            if fr is None or fr.isNull():
                return None
            img = fr.convertToFormat(QImage.Format_ARGB32)
            w, h = img.width(), img.height()
            a = np.frombuffer(img.constBits(), np.uint8, count=img.bytesPerLine() * h)
            a = a.reshape(h, img.bytesPerLine())[:, :w * 4].reshape(h, w, 4)[:, :, 3]
            dpr = fr.devicePixelRatio() or 1.0
            pos = self.window._grab_offset
            return classify_click(a, pos.x() * dpr, pos.y() * dpr, self.animator.species,
                                  getattr(self.behavior.mover, "facing", 1))
        except Exception:  # noqa: BLE001 - a missed hit test only loses the flinch
            return None

    def _on_click(self) -> None:
        kind = self.click_reaction
        if kind == "random":
            kind = self._rng.choice(["wave", "jump", "spin", "look"])
        where = self._clicked_part()
        if kind != "speech":
            self.behavior.react_click(kind, where)
        else:
            self.behavior.react_click("look", where)
        if kind == "jump":
            self.sound.emit("jump")
        self.life.on_click()
        self.clicked.emit()

    def _on_drag_start(self) -> None:
        self.behavior.grab()
        self.life.on_drag(True)
        self.bubble.hide_now()
        self.grabbed.emit()

    def _on_drag(self, gp: QPoint) -> None:
        # Hold the character by its "scruff" (upper middle)
        self.behavior.drag_to(gp.x(), gp.y() + self.px * 0.45)

    def _on_drop(self) -> None:
        self.behavior.drop()

    def _on_land(self) -> None:
        self.dropped.emit()

    def _on_hover(self, on: bool) -> None:
        self._hovered = on
        if on:
            self.behavior.attention.set(AttentionTarget.USER_INTERACTION, None, 0.0)
        self._pet_last_x = None
        self._pet_flips.clear()

    def _petting(self, cur: QPoint, now: float) -> None:
        """Rubbing the cursor back and forth over the companion = petting."""
        if not (self._hovered and self.petting_enabled):
            return
        x = cur.x()
        if self._pet_last_x is not None:
            dx = x - self._pet_last_x
            if abs(dx) >= 3:
                d = 1 if dx > 0 else -1
                if self._pet_dir and d != self._pet_dir:
                    self._pet_flips.append(now)
                self._pet_dir = d
        self._pet_last_x = x
        self._pet_flips = [t for t in self._pet_flips if now - t < 1.6]
        if len(self._pet_flips) >= 4 and self.behavior.state != S.PETTED:
            self._pet_flips.clear()
            self.behavior.pet()
            self.life.on_pet()
            self.petted.emit()

    # -------------------------------------------------------------- frame
    def _tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last
        self._last = now
        if dt > 0:
            self._fps += (1.0 / dt - self._fps) * 0.08
        b = self.behavior
        if self.idle_provider and now - self._idle_check > 1.0:
            self._idle_check = now
            try:
                b.user_idle_seconds = float(self.idle_provider())
            except Exception:  # noqa: BLE001
                b.user_idle_seconds = 0.0
        if self._size_anim is not None:
            f0, f1, t0 = self._size_anim
            u = min(1.0, (now - t0) / 0.3)
            e = u * u * (3 - 2 * u)
            self._apply_size(f0 + (f1 - f0) * e)
            if u >= 1.0:
                self._size_anim = None

        cur = QCursor.pos()
        cx, cy = cur.x(), cur.y()
        if b.cursor_mode != "ignore":
            b.cursor = (cx, cy)
            head = (b.x, b.y - self.px * 0.55)
            reach = {"off": 0.0, "playful": 560.0}.get(b.cursor_interaction, 420.0)
            if (math.hypot(cx - head[0], cy - head[1]) < reach or self._hovered) and b.cursor_interaction != "off":
                b.attention.set(AttentionTarget.MOUSE, (cx, cy), 0.3, now=now)
        else:
            b.cursor = None
        self._petting(cur, now)

        try:
            b.update(dt, now)
        except Exception:  # noqa: BLE001 - a bug in one directive must never freeze the companion
            self._tick_errors += 1
            if self._tick_errors <= 3:
                log.exception("Companion behaviour update failed - resetting to idle")
            b._plan = []
            b._deferred = None
            b.mode = "auto" if b.mode != "held" else b.mode
        try:
            self.life.update(dt)
        except Exception:  # noqa: BLE001 - the routine must never freeze the companion
            self._tick_errors += 1
            if self._tick_errors <= 3:
                log.exception("Companion life update failed")
        u = self.px / UNITS
        for name, val in b.events:
            self.animator.on_event(name, val)
            if name in ("jump", "press", "wake"):
                self.sound.emit({"jump": "jump", "press": "click", "wake": "wake"}[name])
            elif name == "land":
                if val > 500:
                    self.sound.emit("land")
                if val > 250:
                    self.particles.landing(b.x, b.y, val, u)
            elif name == "CELEBRATION_PEAK":
                self.particles.confetti(b.x, b.y - self.px * 0.6, u)
            elif name == "SHOUT":
                self.sound.emit("shout")
                self.emote.emit("SHOUT")
            elif name == "STOMP":
                self.sound.emit("stomp")
                self.particles.landing(b.x + b.mover.facing * self.px * 0.12, b.y, 300, u)
            elif name in ("SIGH", "SNEEZE", "HICCUP", "HUM"):
                self.sound.emit(name.lower())
            elif name in ("woken", "poked"):
                self.emote.emit(name.upper())
            elif name == "ACTION_START" and val in ("LOVE", "DIZZY", "LAUGH"):
                self.sound.emit({"LOVE": "love", "DIZZY": "dizzy", "LAUGH": "laugh"}[val])
                if val == "DIZZY":
                    self.emote.emit("DIZZY")
            elif name == "SPARKS":
                self.particles.sparks(b.x + b.mover.facing * self.px * 0.05, b.y - self.px * 0.7, u)
        b.events.clear()

        st = b.state
        loco = b.mover.sample()
        self._pose = self.animator.update(st, now, dt, loco, b.emotion.style(), b.look, self.intensity,
                                          b.actions.layers(), b.energy.level)
        if st == S.HELD:
            self._pose.turn = float(b.mover.facing)
        self._animation_events(now, loco, u)
        self.particles.update(dt)
        if self.bubble.isVisible():
            self._place_bubble()
        rfps = self.paint_fps
        cost = getattr(self, "_paint_cost", 0.0)
        if rfps > 0 and cost > 0:
            # the frames are drawn on the UI thread: keep their share of it bounded (smaller while one of
            # FocusBuddy's own windows is in use), so the dashboard never stutters behind the companion
            from PySide6.QtWidgets import QApplication
            aw = QApplication.activeWindow()
            duty = 0.28 if aw is not None and aw is not self.window else 0.5
            rfps = max(8, min(rfps, int(duty / cost)))
        if rfps > 0 and now - self._last_paint >= 1.0 / rfps - 0.002:
            self._last_paint = now
            self._present()
        elif rfps <= 0:
            self._place_window()                       # hidden: keep the window where the body is
        if self.debug is not None:
            self.debug.update_info()

        # Keep above other "topmost" windows now and then
        if now - self._last_raise > 6.0 and self.window.isVisible():
            self._last_raise = now
            self.window.raise_()

        # Adaptive frame rate: full rate while anything moves or the cursor is near
        busy = (st in MOVING_STATES or b.mover.moving or st in (S.HELD, S.FALLING, S.CELEBRATING, S.PRESSING,
                S.CLOSING_DISTRACTION, S.PLAYING, S.LANDING, S.PETTED, S.WAKING) or self._hovered
                or self._size_anim is not None or self.bubble.isVisible() or len(self.particles) > 0
                or bool(b.actions.active)
                or (b.look is not None and b.look.kind == AttentionTarget.MOUSE))
        if busy:
            interval = int(1000 / self.target_fps)
        elif st == S.SLEEPING:
            interval = 66
        else:
            interval = max(int(1000 / self.target_fps), 33)
        if self.timer.interval() != interval:
            self.timer.setInterval(interval)

    def _animation_events(self, now: float, loco, u: float) -> None:
        """FOOT_CONTACT -> footsteps (synced to real contacts) + sprint dust; jumps -> dust."""
        b = self.behavior
        rig = getattr(self._pose, "rig", None)
        xs = 1.0 if b.mover.facing_blend >= 0 else -1.0
        for ev, arg in self.animator.drain_events():
            if ev == "FOOT_CONTACT":
                if self.footsteps and loco.speed > loco.walk_speed * 0.6 and now - self._last_step_sound > 0.11:
                    self._last_step_sound = now
                    self.sound.emit("step")
                if loco.gait_level > 2.3 and rig is not None and arg in rig.legs:
                    fx, fy = rig.legs[arg].end
                    self.particles.dust(b.x + xs * fx * u, b.y, int(xs), u, n=1)
            elif ev == "JUMP_START":
                self.particles.dust(b.x, b.y, int(xs), u, n=3)
            elif ev == "LAND_HEAVY":                    # robot: clunky landing throws a few sparks
                self.particles.sparks(b.x, b.y - 4 * u, u, n=5)

    @property
    def fps(self) -> float:
        return self._fps

    def _paint(self, g, rect: QRectF) -> None:
        t0 = time.perf_counter()
        try:
            self._paint_frame(g, rect)
        finally:
            c = time.perf_counter() - t0
            self._paint_cost = c if not getattr(self, "_paint_cost", 0.0) else self._paint_cost * 0.9 + c * 0.1

    def _paint_frame(self, g, rect: QRectF) -> None:
        if self._pose is None:
            self._pose = self.animator.update(self.behavior.state, time.monotonic(), 0.0)
        fx, fy = self._frac
        l, t, _, _ = window_pads(self.px)
        r = QRectF(l + fx, t + fy, self.px, self.px)
        drv = getattr(self.renderer, "driver", None)
        if drv is not None:
            try:
                from .ground import from_behavior
                drv.surface = from_behavior(self.behavior, self.px)
            except Exception:  # noqa: BLE001
                pass
        ctx = getattr(self.renderer, "set_context", None)
        if ctx is not None:
            emo = self.behavior.emotion
            ctx(getattr(emo.current, "value", str(emo.current)), float(getattr(emo, "intensity", 0.0)))
        try:
            self.renderer.paint(g, r, self._pose, 1, self.behavior.state.value, debug=self.debug_draw)
        except Exception:  # noqa: BLE001 - never crash the companion over a frame: step down the chain
            if self.render_mode not in ("illustrated", "production", "3d"):
                raise
            nxt = {"illustrated": "production", "production": "3d"}.get(self.render_mode, "2d")
            log.exception("%s rendering failed - switching to the %s renderer", self.render_mode, nxt)
            self.render_fallback = nxt
            self.render_mode = nxt
            self.renderer = make_renderer(self.spec, nxt, self.render_quality)
            self._attach_rig()
            try:
                self.renderer.paint(g, r, self._pose, 1, self.behavior.state.value, debug=self.debug_draw)
            except Exception:  # noqa: BLE001
                if nxt == "2d":
                    raise
                log.exception("procedural 3D failed too - switching to 2D")
                self.render_fallback = self.render_mode = "2d"
                self.renderer = make_renderer(self.spec, "2d", self.render_quality)
                self._attach_rig()
                self.renderer.paint(g, r, self._pose, 1, self.behavior.state.value, debug=self.debug_draw)
        self._budget_check()
        # ROOT_MOTION clips move the character through the desktop world (HYBRID / IN_PLACE don't)
        rm = getattr(self.renderer, "root_motion_px", None)
        if rm is not None:
            try:
                dx = float(rm(self.px))
                if abs(dx) > 1e-3:
                    self.behavior.apply_root_motion(dx)
            except Exception:  # noqa: BLE001
                pass
        if len(self.particles):
            self.particles.paint(g, self.window.x(), self.window.y())

    def _budget_check(self) -> None:
        """Adaptive quality: if 3D frames keep blowing their budget, step the quality down once in a while
        (never forces 60 FPS on a GPU that can't do it; the window itself keeps moving at full rate)."""
        ms = getattr(self.renderer, "last_ms", 0.0)
        if not ms or not self.auto_quality:
            return
        self._render_ms_avg = getattr(self, "_render_ms_avg", ms) * 0.95 + ms * 0.05
        budget = 1000.0 / max(1, self.render_fps)
        if self._render_ms_avg > budget * 0.85:
            self._over_budget = getattr(self, "_over_budget", 0) + 1
        else:
            self._over_budget = 0
        if self._over_budget > 90:
            from .render3d import Renderer3D
            q = Renderer3D.tier_name(self.render_quality)
            i = self.QUALITY_ORDER.index(q)
            self._over_budget = 0
            if i > 0 and hasattr(self.renderer, "set_quality"):
                new = self.QUALITY_ORDER[i - 1]
                log.info("3D frame time %.1f ms over budget %.1f ms - quality %s -> %s",
                         self._render_ms_avg, budget, q, new)
                self.render_quality = new
                self.renderer.set_quality(new)
                self._render_ms_avg = budget * 0.5

    def _window_target(self) -> tuple[int, int, float, float]:
        margin = GROUND_MARGIN * self.px / UNITS
        l, t, _, _ = window_pads(self.px)
        xf = self.behavior.x - self.px / 2 - l
        yf = self.behavior.y - self.px + margin - t
        xi, yi = math.floor(xf), math.floor(yf)
        return xi, yi, xf - xi, yf - yi

    def _present(self) -> None:
        """Render this moment's frame off-screen, then move the window there and show it together."""
        from PySide6.QtGui import QImage, QPainter as _QP
        w = self.window
        xi, yi, fx, fy = self._window_target()
        self._frac = (fx, fy)
        dpr = max(1.0, float(w.devicePixelRatioF()))
        W, H = w.width(), w.height()
        img = getattr(self, "_frame_img", None)
        need = (int(round(W * dpr)), int(round(H * dpr)))
        if img is None or (img.width(), img.height()) != need or abs(img.devicePixelRatio() - dpr) > 1e-3:
            img = QImage(need[0], need[1], QImage.Format_ARGB32_Premultiplied)
            img.setDevicePixelRatio(dpr)
            # two buffers, alternated: the one on screen is never drawn into
            self._frame_bufs = [img, QImage(img)]
            self._frame_i = 0
        self._frame_i = (getattr(self, "_frame_i", 0) + 1) % 2
        img = self._frame_bufs[self._frame_i]
        self._frame_img = img
        img.fill(0)
        g = _QP(img)
        g.setRenderHint(_QP.Antialiasing, True)
        try:
            self._paint(g, QRectF(0, 0, W, H))
        except Exception:  # noqa: BLE001 - a broken frame must not stop the companion; draw live next time
            log.exception("Companion frame failed")
            g.end()
            w.frame = None
            w.update()
            return
        g.end()
        w.present(img, xi, yi)

    def _place_window(self) -> None:
        margin = GROUND_MARGIN * self.px / UNITS
        l, t, _, _ = window_pads(self.px)
        xf = self.behavior.x - self.px / 2 - l
        yf = self.behavior.y - self.px + margin - t
        xi, yi = math.floor(xf), math.floor(yf)
        self._frac = (xf - xi, yf - yi)
        if self.window.x() != xi or self.window.y() != yi:
            self.window.move(xi, yi)

    def _place_bubble(self) -> None:
        b = self.behavior
        head_top = b.y - self.px * 0.72
        anchor = QPointF(b.x + b.mover.facing * self.px * 0.08, head_top)
        l, t, r, btm = self.env.screen_bounds(b.x, b.y)
        self.bubble.place(anchor, QRectF(l, t, r - l, btm - t), below_offset=self.px * 0.8)
