"""Companion Playground - trigger any action, build sequences, mix emotions, orbit the
3D camera and inspect the skeleton / IK / physics.

It runs the *real* companion stack (Behavior -> ActionPlayer -> LocomotionController
-> AnimationController -> Rig/IK -> Renderer) on a scrolling stage instead of the
desktop, so whatever you see here is exactly what the desktop companion does.

    python main.py --playground          (standalone window)
    Dashboard -> Playground               (page)

Playground-only extras (the desktop companion is unaffected):

* orbit camera   drag to rotate, wheel to zoom, Front / Side / Back / 3/4 / Reset
* action browser searchable, grouped by category, double-click to play
* sequencer      loop, reverse, speed, delay between actions, blend time
* emotion mixer  blend Happiness / Anger / Fear / ... with sliders, plus energy
* debug          skeleton, bone names, IK targets, foot contacts, root, velocity,
                 attention target, emotion values, mood, energy, current / next action
* performance    FPS, frame time, CPU / GPU render time, animation and physics
                 update time, draw calls
* validation run the full living-character sequence (sleep ... rest)
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
                               QLineEdit, QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..companion import actions as act
from ..companion.animation import AnimationController
from ..companion.attention import AttentionTarget
from ..companion.behavior import Behavior
from ..companion.characters import all_characters, get_character
from ..companion.emotion import Emotion
from ..companion.life import Life
from ..companion.particles import ParticleSystem
from ..companion.profile import get_profile
from ..companion.renderer import GROUND_MARGIN, UNITS, make_renderer
from ..companion.showcase import living_steps
from ..companion.states import CompanionState as S
from .theme import C, stylesheet
from .widgets import label

SPEEDS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
BUTTONS = [
    ("Locomotion", ["Idle", "Walk", "Jog", "Run", "Sprint"]),
    ("Rest", ["Sit", "Sleep", "Wake", "Stretch", "Yawn"]),
    ("Play", ["Jump", "Hop", "Dance", "Spin", "Tumble", "Skip", "Roll", "Bounce", "Pounce"]),
    ("Social", ["Wave", "Nod", "Shake Head", "Point", "Beckon", "Clap", "High Five", "Thumbs Up", "Bow", "Greet",
                "Farewell"]),
    ("Emotion", ["Happy", "Curious", "Alert", "Surprised", "Celebrate"]),
    ("Feelings", ["Angry", "Shout", "Grumpy", "Tantrum", "Sulk", "Scared", "Cry", "Embarrassed", "Love", "Laugh",
                  "Bored", "Smug", "Disgusted", "Shocked", "Dizzy", "Determined", "Relieved", "Proud", "Stomp",
                  "Hide Face", "Tremble", "Panic", "Stumble"]),
]
GAITS = {"Walk": "walk", "Jog": "jog", "Run": "run", "Sprint": "sprint"}
SEQUENCES = ["Shout", "Angry", "Scared", "Surprised", "Laughing", "Crying", "Love", "Boredom", "Sulk", "Dizzy"]
MIXER = [("Happiness", Emotion.HAPPY), ("Anger", Emotion.ANGRY), ("Fear", Emotion.SCARED), ("Sadness", Emotion.SAD),
         ("Excitement", Emotion.EXCITED), ("Sleepiness", Emotion.SLEEPY), ("Curiosity", Emotion.CURIOUS),
         ("Confidence", Emotion.CONFIDENT), ("Surprise", Emotion.SURPRISED), ("Love", Emotion.LOVE)]
MODES = [("3D Production (GLB)", "production"), ("3D Procedural", "3d"), ("2D Fallback", "2d")]
BODY_TEST = ["NEUTRAL", "HAPPY", "ANGRY", "SAD", "SCARED", "EXCITED", "SLEEPY"]
QUALITIES = [("Low", "low"), ("Medium", "medium"), ("High", "high"), ("Ultra", "ultra")]
SKINNING = [("Renderer: GPU Skinning", "gpu"), ("Renderer: CPU Skinning", "cpu")]
CAMERA_PRESETS = {"Front": 90.0, "Side": 0.0, "Back": 270.0, "3/4": 45.0}
DEBUG = [("skeleton", "Skeleton"), ("names", "Bone names"), ("ik", "IK targets"), ("contacts", "Foot contacts"),
         ("root", "Root position"), ("velocity", "Velocity vector"), ("attention", "Attention target"),
         ("emotion", "Emotion values"), ("mood", "Mood"), ("energy", "Energy"), ("current", "Current action"),
         ("next", "Next action"), ("perf", "Performance")]


LIFE_EVENTS = ("Wake-up scene", "Enter", "Leave & return", "Close dashboard", "Answer: nap", "Answer: stay",
               "Quit (goodbye)", "Nap / wake", "Focus start", "Focus done", "Distraction", "Back to work",
               "Ignore a wave", "Full-screen app")


class StageEnv:
    """A long flat stage: the ground is y = 0, plenty of room left and right (a desktop-sized
    stage while the daily life is simulated, so entrances / exits use the real screen edges)."""

    half_width = 6000.0

    def all_bounds(self):
        w = self.half_width
        return [(-w, -3000.0, w, 0.0)]

    def screen_bounds(self, x, y):
        return self.all_bounds()[0]

    def windows(self):
        return None


class PlaygroundSim:
    """Qt-free simulation used by the playground (and its tests)."""

    def __init__(self, character: str = "cat", char_px: float = 300.0, seed: int = 7, mode: str | None = None):
        from ..companion.renderer import DEFAULT_MODE
        self.char_px = char_px
        self.seed = seed
        self.t = 0.0
        self.mode = mode or (DEFAULT_MODE["mode"] if DEFAULT_MODE["mode"] != "2d" else "production")
        self.effects = True
        self.lights = {"key": True, "fill": True, "rim": True}
        self.quality = "medium"
        self.orbit = [0.0, 0.0, 1.0]          # yaw offset, pitch offset, zoom (Playground camera only)
        self.orbit_on = False
        self.timing = {"anim_ms": 0.0, "physics_ms": 0.0}
        self.validation: list[tuple[str, float]] = []
        self.cursor: tuple[float, float] | None = None
        self.skinning = "gpu"
        self.set_character(character)

    def set_character(self, cid: str) -> None:
        self.spec = get_character(cid)
        self.renderer = make_renderer(self.spec, self.mode, self.quality)
        pid = cid if cid in ("cat", "dog", "fox", "rabbit", "robot", "buddy") else self.spec.personality
        self.profile = get_profile(pid)
        self.anim = AnimationController(seed=self.seed, profile=self.profile)
        if hasattr(self.renderer, "rig"):
            self.anim.set_rig(self.renderer.rig)
        self.b = Behavior(StageEnv(), char_size=self.char_px, seed=self.seed, now=lambda: self.t,
                          profile=self.profile)
        self.b.autonomous = False
        self.b.stumbles = False
        self.b.set_species(self.anim.species)
        self.b.press_tip = self.renderer.press_point()
        self.b.mover.teleport(0.0, 0.0)
        self.b.mode = "directed"
        # the companion's daily life (wake-up, entrances, antics, close scene ...): simulated on demand
        self.life_on = False
        self.bubble: tuple[str, float] = ("", 0.0)
        self.life_shown = True
        self.life = Life(self.b, char_id=cid, seed=self.seed, now=lambda: self.t)
        self.life.say = lambda text, secs=None: self._bubble(text, secs or 4.0)
        self.life.ask = lambda text, buttons: self._bubble(text + "   [" + " / ".join(t for _a, t, _p in buttons) + "]", 25.0)
        self.life.hide_bubble = lambda: self._bubble("", 0.0)
        self.life.set_shown = lambda on: setattr(self, "life_shown", bool(on))
        self.life.on_quit_ready = lambda: self.log.append(f"{self.t:6.1f}s  (FocusBuddy would quit now)")
        self.particles = ParticleSystem(self.seed)
        self.particles.enabled = getattr(self, "effects", True)
        self.pose = None
        self.log: list[str] = []
        self._apply_orbit()
        self._renderer_opts()

    @property
    def species(self) -> str:
        return self.anim.species

    # ------------------------------------------------------------- daily life
    def _bubble(self, text: str, secs: float) -> None:
        self.bubble = (text, self.t + secs)
        if text:
            self.log.append(f"{self.t:6.1f}s  \"{text}\"")
            del self.log[:-6]

    def set_life(self, on: bool) -> None:
        """Simulate the daily life: autonomous routine on a desktop-sized stage."""
        self.life_on = on
        env = self.b.env
        env.half_width = max(900.0, self.char_px * 4.5) if on else 6000.0
        b = self.b
        b.stop_sequence()
        b._plan = []
        b.autonomous = on
        b.mode = "auto" if on else "directed"
        if on:
            b.mover.teleport(0.0, 0.0)
            self.life.skip_intro()
        else:
            self.life_shown = True
            self.bubble = ("", 0.0)

    def life_event(self, name: str) -> None:
        """Playground buttons: lifecycle scenes and app events."""
        if not self.life_on:
            self.set_life(True)
        L_ = self.life
        if name == "Wake-up scene":
            L_.wake_up()
        elif name == "Leave & return":
            L_.leave(minutes=0.25)
        elif name == "Enter":
            L_.enter()
        elif name == "Close dashboard":
            L_.close_scene()
        elif name == "Answer: nap":
            L_.answer("life:nap")
        elif name == "Answer: stay":
            L_.answer("life:stay")
        elif name == "Quit (goodbye)":
            L_.goodbye(L_.on_quit_ready)
        elif name == "Nap / wake":
            L_.on_click() if L_.napping else L_.nap()
        elif name == "Focus start":
            self.b.focus_start()
            L_.on_focus("focus_started")
        elif name == "Focus done":
            L_.on_focus("focus_completed")
            self.b.focus_complete()
        elif name == "Distraction":
            L_.distraction_started("youtube.com")
            self.b._start_plan([("action", "SURPRISED", None, act.Priority.DISTRACTION_RESPONSE),
                                ("state", S.WARNING, 3.0)], mode="directed",
                               priority=act.Priority.DISTRACTION_RESPONSE, after=self.b.release)
        elif name == "Back to work":
            L_.distraction_ended(returned=True)
            self.b.celebrate_return()
        elif name == "Ignore a wave":
            L_.awaiting_user_until = self.t + 0.1
            L_.awaiting_gag = "wave_hello"
        elif name == "Full-screen app":
            L_.set_fullscreen(not L_.fullscreen)

    def set_skinning(self, mode: str) -> None:
        """GPU (vertex shader) or CPU (NumPy) skinning for the production renderer."""
        self.skinning = mode if mode in ("gpu", "cpu") else "gpu"
        self._renderer_opts()

    def _renderer_opts(self) -> None:
        r = self.renderer
        r.effects = self.effects
        if hasattr(r, "skinning"):
            r.skinning = getattr(self, "skinning", "gpu")
        if hasattr(r, "lights"):
            r.lights = dict(self.lights)

    def set_effects(self, on: bool) -> None:
        """Emotion icons, particles and speech: off = the body-language test (acting only)."""
        self.effects = bool(on)
        self.particles.enabled = self.effects
        self._renderer_opts()

    def set_light(self, name: str, on: bool) -> None:
        self.lights[name] = bool(on)
        self._renderer_opts()

    def set_emotion(self, name: str, intensity: float = 1.0) -> None:
        self.b.emotion.override_style = None
        self.b.emotion.set_mix(None)
        self.b.emotion.routing = False
        self.b.emotion.set(Emotion(name), intensity, None, now=self.t)

    def set_pose(self, name: str | None) -> None:
        from ..companion.emotion import pose_style
        self.b.emotion.override_style = pose_style(name) if name else None

    def set_mode(self, mode: str) -> None:
        """Switch 2D / 3D without restarting the simulation."""
        self.mode = mode
        self.renderer = make_renderer(self.spec, mode, self.quality)
        if hasattr(self.renderer, "rig"):
            self.anim.set_rig(self.renderer.rig)
        self.b.press_tip = self.renderer.press_point()
        self._apply_orbit()
        self._renderer_opts()

    def set_quality(self, quality: str) -> None:
        self.quality = quality
        if hasattr(self.renderer, "set_quality"):
            self.renderer.set_quality(quality)

    # ------------------------------------------------------------ camera
    def _apply_orbit(self) -> None:
        r = self.renderer
        if not hasattr(r, "set_orbit"):
            return
        if self.orbit_on:
            r.set_orbit(*self.orbit)
        else:
            r.reset_orbit()

    def orbit_by(self, dyaw: float, dpitch: float) -> None:
        self.orbit_on = True
        self.orbit[0] = (self.orbit[0] + dyaw) % 360.0
        self.orbit[1] = max(-60.0, min(70.0, self.orbit[1] + dpitch))
        self._apply_orbit()

    def zoom_by(self, factor: float) -> None:
        self.orbit_on = True
        self.orbit[2] = max(0.4, min(2.5, self.orbit[2] * factor))
        self._apply_orbit()

    def orbit_preset(self, name: str) -> None:
        """Absolute views relative to the way the character currently faces."""
        if name == "Reset":
            self.orbit = [0.0, 0.0, 1.0]
            self.orbit_on = False
            self._apply_orbit()
            return
        turn = self.pose.turn if self.pose is not None else 1.0
        base = 90.0 - (90.0 - 22.0) * max(-1.0, min(1.0, turn))
        want = CAMERA_PRESETS.get(name, 22.0)
        if turn < 0 and name in ("Side", "3/4"):
            want = 180.0 - want
        self.orbit_on = True
        self.orbit[0] = (want - base) % 360.0
        self.orbit[1] = 0.0
        self._apply_orbit()

    # ------------------------------------------------------------ commands
    def trigger(self, name: str) -> None:
        b = self.b
        b.stop_sequence()
        if name == "Idle":
            b._start_plan([("state", S.IDLE, None)], mode="directed", priority=act.Priority.SPECIAL_ACTION)
        elif name in GAITS:
            gait = GAITS[name]
            secs = 3.0 if gait in ("walk", "jog") else 2.2
            b.play_sequence([(name.upper(), secs), ("TURN", None)], loop=True)
        elif name == "Alert":
            b._start_plan([("action", "SURPRISED", None, act.Priority.DISTRACTION_RESPONSE),
                           ("state", S.ALERT, 1.2), ("state", S.WATCHING, None)], mode="directed",
                          priority=act.Priority.DISTRACTION_RESPONSE)
        elif name == "Sleep":
            b._start_plan([("state", S.SITTING, 0.8), ("action", "YAWN", None, act.Priority.SPECIAL_ACTION),
                           ("state", S.SLEEPING, None)], mode="directed", priority=act.Priority.SPECIAL_ACTION)
        elif name == "Wake":
            b._start_plan([("state", S.WAKING, 1.0), ("action", "STRETCH", None, act.Priority.SPECIAL_ACTION),
                           ("state", S.IDLE, None)], mode="directed", priority=act.Priority.SPECIAL_ACTION)
        elif name == "Sit":
            b._start_plan([("state", S.SITTING, None)], mode="directed", priority=act.Priority.SPECIAL_ACTION)
        else:
            b.play_action(name.upper().replace(" ", "_"), priority=act.Priority.SPECIAL_ACTION)

    def perform(self, sequence: str, intensity: float = 1.0) -> bool:
        """One of the upgraded emotional sequences (shout levels, angry, scared, ...)."""
        self.b.stop_sequence()
        return self.b.perform_sequence(sequence, intensity)

    def play_sequence(self, items: list[tuple[str, float | None]], loop: bool, reverse: bool = False,
                      delay: float = 0.0, blend: float = 1.0) -> None:
        seq = list(reversed(items)) if reverse else list(items)
        if delay > 0:
            spaced: list[tuple[str, float | None]] = []
            for it in seq:
                spaced += [it, ("IDLE", delay)]
            seq = spaced
        self.b.actions.blend_scale = max(0.1, blend)
        self.b.play_sequence(seq, loop=loop)

    def run_validation(self) -> None:
        """The living-character validation sequence on the stage (sleep ... rest)."""
        self.validation = []
        b = self.b
        b.stop_sequence()
        x0 = b.x - 300.0
        steps = living_steps(self.species, x0, 2400.0, 0.0, self.char_px,
                             mark=lambda lbl: self.validation.append((lbl, round(self.t, 2))),
                             cursor=self.set_cursor)
        b._start_plan(steps + [("call", lambda: setattr(self, "cursor", None))], mode="directed",
                      priority=act.Priority.SPECIAL_ACTION)

    def set_cursor(self, x: float, y: float) -> None:
        self.cursor = (x, y)

    def look_at(self, x: float, y: float) -> None:
        """Make the character look at a stage point (eyes -> head -> neck -> torso)."""
        self.cursor = (x, y)
        self.b.attention.set(AttentionTarget.MOUSE, (x, y), 2.5, now=self.t)

    def set_mix(self, weights: dict | None, energy: float | None = None) -> None:
        self.b.emotion.set_mix(weights)
        if energy is not None:
            self.b.energy.level = max(0.0, min(100.0, energy))

    def next_action(self) -> str:
        plan = self.b._plan
        for st in plan[1:3]:
            k = st[0]
            if k == "action":
                return str(st[1])
            if k == "state":
                return st[1].value
            if k in ("goto", "goto_gait", "goto_rel", "move"):
                return "move"
            if k in ("jump", "turn_around", "face"):
                return k
        return "-"

    # --------------------------------------------------------------- frame
    def step(self, dt: float):
        b = self.b
        self.t += dt
        t0 = time.perf_counter()
        b.update(dt, self.t)
        if self.life_on:
            self.life.update(dt)
        t1 = time.perf_counter()
        u = self.char_px / UNITS
        for name, val in b.events:
            self.anim.on_event(name, val)
            if name == "land" and val > 250:
                self.particles.landing(b.x, b.y, val, u)
            elif name == "CELEBRATION_PEAK":
                self.particles.confetti(b.x, b.y - self.char_px * 0.6, u)
            elif name == "SPARKS":
                self.particles.sparks(b.x, b.y - self.char_px * 0.7, u)
            elif name == "ACTION_START":
                self.log.append(f"{self.t:6.1f}s  {val}")
                del self.log[:-6]
        b.events.clear()
        loco = b.mover.sample()
        t2 = time.perf_counter()
        self.pose = self.anim.update(b.state, self.t, dt, loco, b.emotion.style(), b.look, 1.0, b.actions.layers(),
                                     b.energy.level)
        t3 = time.perf_counter()
        a = 0.1
        self.timing["physics_ms"] += ((t1 - t0) * 1000 - self.timing["physics_ms"]) * a
        self.timing["anim_ms"] += ((t3 - t2) * 1000 - self.timing["anim_ms"]) * a
        xs = 1.0 if b.mover.facing_blend >= 0 else -1.0
        for ev, arg in self.anim.drain_events():
            if ev == "FOOT_CONTACT" and loco.gait_level > 2.3 and self.pose.rig is not None and arg in self.pose.rig.legs:
                fx, _ = self.pose.rig.legs[arg].end
                self.particles.dust(b.x + xs * fx * u, b.y, int(xs), u, n=1)
            elif ev == "JUMP_START":
                self.particles.dust(b.x, b.y, int(xs), u, n=3)
            elif ev == "LAND_HEAVY":
                self.particles.sparks(b.x, b.y, u, n=5)
        self.particles.update(dt)
        return self.pose

    def info(self, opts: dict | None = None) -> list[str]:
        opts = opts or {}
        b = self.b
        m = b.mover
        rig = self.pose.rig if self.pose is not None else None
        feet = ""
        if rig is not None:
            feet = "  ".join(f"{k}:{'▼' if l.contact else '△'}" for k, l in rig.legs.items())
        cur = b.actions.current("full")
        layers = ", ".join(a.name for a in b.actions.layers()) or "-"
        lines = [f"State: {b.state.value}    Action: {cur.name if cur else '-'}",
                 f"Layers: {layers}",
                 f"Gait: {m.gait_name}  level {m.gait_level:.2f}   {m.speed:.0f} px/s   {m.mode}",
                 f"Phase: {self.anim.phase_name}",
                 f"Feet: {feet}",
                 f"Emotion: {b.emotion.current.value}   Energy: {b.energy.level:.0f}"]
        if opts.get("current"):
            lines.append(f"Current action: {cur.name if cur else '-'}  ({cur.u:.0%})" if cur else
                         "Current action: -")
        if opts.get("next"):
            lines.append(f"Next action: {self.next_action()}")
        if opts.get("mood"):
            lines.append(f"Mood: {b.mood.value:+.0f} ({b.mood.label})   Memory: {b.memory.summary(self.t)['recent']}")
        if opts.get("energy"):
            lines.append(f"Energy: {b.energy.level:.0f}/100 ({b.energy.mood})")
        if opts.get("emotion") and self.pose is not None:
            st = b.emotion.style()
            p = self.pose
            lines.append(f"Emotion values: valence {st.valence:+.2f} arousal {st.arousal:.2f} "
                         f"intensity {b.emotion.intensity:.2f}")
            lines.append(f"Face: smile {p.smile:+.2f} open {p.mouth_open:.2f} jaw {p.jaw:.2f} lid {p.lid:.2f} "
                         f"tilt {p.lid_tilt:+.2f} pupil {p.pupil:.2f} brow {p.brow_raise:+.2f}")
        if self.life_on:
            lines += self.life.explain()
            if not self.life_shown:
                lines.append("(off-screen / hidden)")
        if opts.get("attention") and b.look is not None:
            lk = b.look
            lines.append(f"Attention: {lk.kind.value}  eyes {lk.look_x:+.2f}  head {lk.head_x:+.2f}  "
                         f"neck {lk.neck_x:+.2f}  torso {lk.torso_x:+.2f}  converge {lk.converge:.2f}")
        return lines

    def perf(self, fps: float, frame_ms: float, render_ms: float = 0.0) -> list[str]:
        st = getattr(self.renderer, "stats", None) or {}
        backend = st.get("backend", "2d")
        rast = st.get("raster_ms", render_ms)
        gpu = backend.startswith("gpu")
        drv = getattr(self.renderer, "driver", None)
        clip = getattr(drv, "clip_info", ("", "")) if drv is not None else ("", "")
        return [f"FPS: {fps:.0f}   frame {frame_ms:.1f} ms   renderer {backend.upper()} {st.get('quality', '')} "
                f"LOD {st.get('lod', '-')}",
                f"CPU render: {rast if not gpu else st.get('build_ms', 0.0):.2f} ms   "
                f"GPU render: {st.get('gpu_ms', rast if gpu else 0.0) if gpu else 0.0:.2f} ms   "
                f"decals {st.get('decal_ms', 0.0):.2f} ms",
                f"CPU skinning: {st.get('skin_ms', 0.0):.2f} ms   IK / rig: {st.get('rig_ms', 0.0):.2f} ms   "
                f"skinning {st.get('skinning', '-')}",
                f"Animation update: {self.timing['anim_ms']:.2f} ms   Physics update: {self.timing['physics_ms']:.2f} ms",
                f"Draw calls: {st.get('draw_calls', 1)}   primitives {st.get('prims', 0)}   "
                f"clip {clip[0] or '-'}{'+' + clip[1] if clip[1] else ''}"]


class PlaygroundCanvas(QWidget):
    def __init__(self, sim: PlaygroundSim):
        super().__init__()
        self.sim = sim
        self.setMinimumSize(560, 360)
        self.setMouseTracking(True)
        self.cam_x = 0.0
        self.debug = {k: False for k, _ in DEBUG}
        self.lines: list[str] = []
        self.perf_lines: list[str] = []
        self._drag: QPointF | None = None
        self._fps = 0.0
        self._last_paint = time.perf_counter()
        self.frame_ms = 0.0
        self.render_ms = 0.0
        self.show_hint = True
        self.caption = ""

    # ------------------------------------------------------- orbit camera
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.position()
        elif e.button() == Qt.RightButton:
            self.look_at_screen(e.position())

    def mouseMoveEvent(self, e):
        if self._drag is not None and self.sim.mode in ("3d", "production"):
            d = e.position() - self._drag
            self._drag = e.position()
            self.sim.orbit_by(d.x() * 0.5, d.y() * 0.35)
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        self.look_at_screen(e.position())

    def wheelEvent(self, e):
        if self.sim.mode in ("3d", "production"):
            self.sim.zoom_by(1.1 if e.angleDelta().y() > 0 else 1 / 1.1)
            self.update()

    def ground(self) -> float:
        return self.height() - 60

    def look_at_screen(self, pos: QPointF) -> None:
        """Right-click / double-click: the character looks at that spot (watch the look-at chain)."""
        x = pos.x() - self.width() / 2 + self.cam_x
        y = pos.y() - self.ground()
        self.sim.look_at(x, y)

    def to_screen(self, x: float, y: float) -> QPointF:
        return QPointF(self.width() / 2 + (x - self.cam_x), self.ground() + y)

    # -------------------------------------------------------------- paint
    def paintEvent(self, _):
        t0 = time.perf_counter()
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0, QColor("#EEF2FF"))
        grad.setColorAt(1, QColor("#F9FAFE"))
        g.fillRect(self.rect(), grad)
        ground = self.ground()
        g.setPen(QPen(QColor("#C9D0E0"), 2))
        g.drawLine(0, ground, self.width(), ground)
        g.setPen(QPen(QColor("#D9DEEA"), 1))
        step = 60
        x = -(self.cam_x % step)
        while x < self.width():
            g.drawLine(QPointF(x, ground), QPointF(x - 14, ground + 14))
            x += step
        sim = self.sim
        dbg = self.debug
        if sim.pose is not None and sim.life_shown:
            b = sim.b
            cx = self.width() / 2 + (b.x - self.cam_x)
            px = sim.char_px
            r = QRectF(cx - px / 2, ground + b.y - px + GROUND_MARGIN * px / UNITS, px, px)
            rdbg = {k: bool(dbg.get(k)) for k in ("skeleton", "ik", "contacts", "names")}
            tr = time.perf_counter()
            sim.renderer.paint(g, r, sim.pose, 1, debug=rdbg if any(rdbg.values()) else None)
            self.render_ms = (time.perf_counter() - tr) * 1000
            if len(sim.particles):
                sim.particles.paint(g, self.cam_x - self.width() / 2, -ground)
            self._overlays(g, b)
            text, until = sim.bubble
            if text and sim.t < until:
                f = QFont()
                f.setPixelSize(13)
                g.setFont(f)
                br = QRectF(cx - 190, r.top() - 46, 380, 38)
                g.setPen(QPen(QColor("#C9D0E0"), 1))
                g.setBrush(QColor("#FFFFFF"))
                g.drawRoundedRect(br, 12, 12)
                g.setPen(QColor("#1D2433"))
                g.drawText(br.adjusted(10, 0, -10, 0), Qt.AlignCenter | Qt.TextWordWrap, text)
        if sim.cursor is not None:
            c = self.to_screen(*sim.cursor)
            g.setPen(QPen(QColor("#1D2433"), 1.5))
            g.setBrush(QColor("#FFFFFF"))
            arrow = [c, c + QPointF(0, 16), c + QPointF(4, 12), c + QPointF(8, 19), c + QPointF(10, 18),
                     c + QPointF(6, 11), c + QPointF(11, 11)]
            g.drawPolygon(arrow)
        g.setPen(QColor(C["muted"]))
        f = QFont()
        f.setPixelSize(12)
        g.setFont(f)
        for i, ln in enumerate(self.lines):
            g.drawText(QPointF(14, 22 + i * 17), ln)
        if dbg.get("perf"):
            self.perf_lines = sim.perf(self._fps, self.frame_ms, self.render_ms)
            g.setPen(QColor("#2E3A8C"))
            for i, ln in enumerate(self.perf_lines):
                g.drawText(QPointF(14, self.height() - 14 - (len(self.perf_lines) - 1 - i) * 17), ln)
        if self.caption:
            f2 = QFont()
            f2.setPixelSize(18)
            f2.setBold(True)
            g.setFont(f2)
            g.setPen(QColor("#1D2433"))
            g.drawText(QRectF(0, 10, self.width(), 30), Qt.AlignCenter, self.caption)
            g.setFont(f)
        if self.show_hint:
            hint = "drag: orbit (3D) · wheel: zoom · right-click: look here"
            g.setPen(QColor("#A0A8BC"))
            g.drawText(QPointF(self.width() - 330, self.height() - 14), hint)
        g.end()
        now = time.perf_counter()
        self.frame_ms = (now - t0) * 1000
        dt = max(1e-4, now - self._last_paint)
        self._last_paint = now
        self._fps += (1.0 / dt - self._fps) * 0.1

    def _overlays(self, g: QPainter, b) -> None:
        dbg = self.debug
        root = self.to_screen(b.x, b.y)
        if dbg.get("root"):
            g.setPen(QPen(QColor("#E5484D"), 2))
            g.drawLine(root + QPointF(-8, 0), root + QPointF(8, 0))
            g.drawLine(root + QPointF(0, -8), root + QPointF(0, 8))
            g.drawText(root + QPointF(6, 16), f"root ({b.x:.0f}, {b.y:.0f})")
        if dbg.get("velocity"):
            v = b.mover.vel
            end = root + QPointF(v.x * 0.25, v.y * 0.25 - 20)
            g.setPen(QPen(QColor("#22B07D"), 2.5))
            start = root + QPointF(0, -20)
            g.drawLine(start, end)
            ang = math.atan2(end.y() - start.y(), end.x() - start.x())
            for da in (2.6, -2.6):
                g.drawLine(end, end + QPointF(8 * math.cos(ang + da), 8 * math.sin(ang + da)))
        if dbg.get("attention") and b.look is not None and b.look.point is not None:
            head = self.to_screen(b.x + b.mover.facing * self.sim.char_px * 0.15, b.y - self.sim.char_px * 0.55)
            tgt = self.to_screen(*b.look.point)
            pen = QPen(QColor("#F2994A"), 1.5)
            pen.setStyle(Qt.DashLine)
            g.setPen(pen)
            g.drawLine(head, tgt)
            g.drawEllipse(tgt, 6, 6)


class CompanionPlayground(QWidget):
    def __init__(self, character: str = "cat", standalone: bool = False):
        super().__init__()
        if standalone:
            self.setWindowTitle("FocusBuddy - Companion Playground")
            self.setObjectName("Root")
            self.setStyleSheet(stylesheet())
            self.resize(1280, 800)
        self.sim = PlaygroundSim(character)
        self.speed = 1.0
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

        side_w = QWidget()
        side = QVBoxLayout(side_w)
        side.setContentsMargins(0, 0, 8, 0)
        side.addWidget(label("Companion Playground", "H2"))
        side.addWidget(label("Trigger actions, chain sequences, mix emotions, orbit the camera.", "Muted", wrap=True))
        self.char = QComboBox()
        for spec in all_characters():
            self.char.addItem(f"{spec.emoji}  {spec.display_name}", spec.id)
        self.char.setCurrentIndex(max(0, self.char.findData(character)))
        self.char.currentIndexChanged.connect(self._character_changed)
        side.addWidget(self.char)
        mrow = QHBoxLayout()
        self.mode = QComboBox()
        for text, m in MODES:
            self.mode.addItem(text, m)
        self.mode.setCurrentIndex(max(0, self.mode.findData(self.sim.mode)))
        self.mode.currentIndexChanged.connect(lambda _i: self.sim.set_mode(self.mode.currentData()))
        mrow.addWidget(self.mode, 1)
        self.quality = QComboBox()
        for text, q in QUALITIES:
            self.quality.addItem(f"Quality: {text}", q)
        self.quality.setCurrentIndex(1)
        self.quality.currentIndexChanged.connect(lambda _i: self.sim.set_quality(self.quality.currentData()))
        mrow.addWidget(self.quality, 1)
        side.addLayout(mrow)
        self.skin_mode = QComboBox()
        for text, m in SKINNING:
            self.skin_mode.addItem(text, m)
        self.skin_mode.setToolTip("Production renderer: skin the mesh in the vertex shader or in NumPy "
                                  "(compare the two; the GPU path falls back to CPU on any driver error)")
        self.skin_mode.currentIndexChanged.connect(lambda _i: self.sim.set_skinning(self.skin_mode.currentData()))
        side.addWidget(self.skin_mode)

        # ---- camera
        side.addWidget(label("Camera (3D)", "H3"))
        crow = QHBoxLayout()
        crow.setSpacing(4)
        for name in ("Front", "Side", "Back", "3/4", "Reset"):
            b = self._chip(name)
            b.clicked.connect(lambda _=False, n=name: self.sim.orbit_preset(n))
            crow.addWidget(b)
        side.addLayout(crow)

        # ---- character studio
        side.addWidget(label("Character studio", "H3"))
        st = QGridLayout()
        self.emotion_sel = QComboBox()
        self.emotion_sel.addItem("Emotion…", "")
        for e in Emotion:
            self.emotion_sel.addItem(e.value.title(), e.value)
        self.emotion_sel.currentIndexChanged.connect(
            lambda _i: self.emotion_sel.currentData() and self.sim.set_emotion(self.emotion_sel.currentData()))
        st.addWidget(self.emotion_sel, 0, 0)
        from ..companion.emotion import POSES
        self.pose_sel = QComboBox()
        self.pose_sel.addItem("Pose library…", "")
        for name in POSES:
            self.pose_sel.addItem(name.replace("_", " "), name)
        self.pose_sel.currentIndexChanged.connect(lambda _i: self.sim.set_pose(self.pose_sel.currentData() or None))
        st.addWidget(self.pose_sel, 0, 1)
        st.addWidget(label("Mood", "Muted"), 1, 0)
        self.mood_sl = QSlider(Qt.Horizontal)
        self.mood_sl.setRange(-100, 100)
        self.mood_sl.setValue(int(self.sim.b.mood.value))
        self.mood_sl.valueChanged.connect(lambda v: setattr(self.sim.b.mood, "value", float(v)))
        st.addWidget(self.mood_sl, 1, 1)
        st.addWidget(label("Physics strength", "Muted"), 2, 0)
        self.phys_sl = QSlider(Qt.Horizontal)
        self.phys_sl.setRange(30, 160)
        self.phys_sl.setValue(100)
        self.phys_sl.valueChanged.connect(lambda v: setattr(self.sim.anim, "physics", v / 100.0))
        st.addWidget(self.phys_sl, 2, 1)
        side.addLayout(st)
        lrow = QHBoxLayout()
        self.light_boxes = {}
        for key in ("key", "fill", "rim"):
            cb = QCheckBox(f"{key.title()} light")
            cb.setChecked(True)
            cb.toggled.connect(lambda on, k=key: self.sim.set_light(k, on))
            lrow.addWidget(cb)
            self.light_boxes[key] = cb
        side.addLayout(lrow)
        erow = QHBoxLayout()
        self.effects_box = QCheckBox("Effects (icons, particles)")
        self.effects_box.setChecked(True)
        self.effects_box.toggled.connect(self.sim.set_effects)
        erow.addWidget(self.effects_box)
        bt = QPushButton("Body-language test")
        bt.setToolTip("Effects off; cycles neutral / happy / angry / sad / scared / excited / sleepy")
        bt.clicked.connect(self.body_language_test)
        erow.addWidget(bt)
        side.addLayout(erow)

        # ---- daily life
        side.addWidget(label("Daily life", "H3"))
        side.addWidget(label("Simulate the routine on a desktop-sized stage; the overlay explains every decision.",
                             "Muted", wrap=True))
        lrow2 = QHBoxLayout()
        self.life_box = QCheckBox("Autonomous life")
        self.life_box.toggled.connect(self.sim.set_life)
        lrow2.addWidget(self.life_box)
        self.quiet_box = QCheckBox("Quiet")
        self.quiet_box.toggled.connect(lambda on: self.sim.life.apply_settings(quiet=bool(on)))
        lrow2.addWidget(self.quiet_box)
        self.focus_beh = QComboBox()
        for key in ("balanced", "strict", "playful"):
            self.focus_beh.addItem(f"Focus: {key}", key)
        self.focus_beh.currentIndexChanged.connect(
            lambda _i: self.sim.life.apply_settings(focus_behavior=self.focus_beh.currentData()))
        lrow2.addWidget(self.focus_beh)
        side.addLayout(lrow2)
        pgrid = QGridLayout()
        for i, (text, key) in enumerate((("Playfulness", "playfulness"), ("Roaming", "roaming_frequency"),
                                         ("Chatter", "chatter"))):
            pgrid.addWidget(label(text, "Muted"), i, 0)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 100)
            sl.setValue(int(getattr(self.sim.life.settings, key) * 100))
            sl.valueChanged.connect(lambda v, k=key: self.sim.life.apply_settings(**{k: v / 100.0}))
            pgrid.addWidget(sl, i, 1)
        side.addLayout(pgrid)
        lgrid = QGridLayout()
        lgrid.setSpacing(6)
        for i, n in enumerate(LIFE_EVENTS):
            b = self._chip(n)
            b.clicked.connect(lambda _=False, name=n: self.sim.life_event(name))
            lgrid.addWidget(b, i // 3, i % 3)
        side.addLayout(lgrid)
        grow = QHBoxLayout()
        self.gag_sel = QComboBox()
        for name in self.sim.life.engine.gags:
            self.gag_sel.addItem(name.replace("_", " "), name)
        grow.addWidget(self.gag_sel, 1)
        gb = QPushButton("Run antic")
        gb.clicked.connect(lambda: (self.sim.life_on or self.life_box.setChecked(True),
                                    self.sim.life.trigger(self.gag_sel.currentData())))
        grow.addWidget(gb)
        side.addLayout(grow)

        # ---- quick buttons
        self.buttons: dict[str, QPushButton] = {}
        for title, names in BUTTONS:
            side.addWidget(label(title, "H3"))
            grid = QGridLayout()
            grid.setSpacing(6)
            for i, n in enumerate(names):
                b = self._chip(n)
                b.clicked.connect(lambda _=False, name=n: self.trigger(name))
                grid.addWidget(b, i // 3, i % 3)
                self.buttons[n] = b
            side.addLayout(grid)

        # ---- emotional sequences
        side.addWidget(label("Emotional sequences", "H3"))
        irow = QHBoxLayout()
        irow.addWidget(label("Intensity", "Muted"))
        self.seq_intensity = QSlider(Qt.Horizontal)
        self.seq_intensity.setRange(0, 100)
        self.seq_intensity.setValue(70)
        irow.addWidget(self.seq_intensity, 1)
        side.addLayout(irow)
        sgrid = QGridLayout()
        sgrid.setSpacing(6)
        for i, n in enumerate(SEQUENCES):
            b = self._chip(n)
            b.clicked.connect(lambda _=False, name=n: self.sim.perform(name, self.seq_intensity.value() / 100))
            sgrid.addWidget(b, i // 3, i % 3)
        side.addLayout(sgrid)
        vb = QPushButton("Validation run (sleep … rest)")
        vb.clicked.connect(self.sim_validation)
        side.addWidget(vb)

        # ---- action browser
        side.addWidget(label("Action browser", "H3"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search actions…")
        self.search.textChanged.connect(self._fill_actions)
        side.addWidget(self.search)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumHeight(200)
        self.tree.itemDoubleClicked.connect(lambda it, _c: self.trigger(it.data(0, Qt.UserRole) or ""))
        side.addWidget(self.tree)
        arow = QHBoxLayout()
        pb = QPushButton("Play")
        pb.clicked.connect(lambda: self.trigger(self.selected_action()))
        arow.addWidget(pb)
        self.dur = QDoubleSpinBox()
        self.dur.setRange(0.0, 20.0)
        self.dur.setSingleStep(0.5)
        self.dur.setValue(2.0)
        self.dur.setSuffix(" s")
        self.dur.setToolTip("Duration (0 = the action's natural length)")
        arow.addWidget(self.dur)
        add = QPushButton("Add to sequence")
        add.clicked.connect(self.add_action)
        arow.addWidget(add)
        side.addLayout(arow)

        # ---- sequencer
        side.addWidget(label("Sequence timeline", "H3"))
        self.timeline = QListWidget()
        self.timeline.setMinimumHeight(120)
        side.addWidget(self.timeline, 1)
        brow = QHBoxLayout()
        for text, fn in (("Remove", self.remove_action), ("Move Up", lambda: self.move(-1)),
                         ("Move Down", lambda: self.move(1))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            brow.addWidget(b)
        side.addLayout(brow)
        orow = QHBoxLayout()
        self.reverse = QCheckBox("Reverse")
        orow.addWidget(self.reverse)
        self.delay = self._spin(0.0, 5.0, 0.0, " s delay", 0.25)
        orow.addWidget(self.delay)
        side.addLayout(orow)
        orow2 = QHBoxLayout()
        self.blend = self._spin(0.2, 3.0, 1.0, "x blend", 0.1)
        orow2.addWidget(self.blend)
        self.seq_speed = self._spin(0.25, 2.0, 1.0, "x speed", 0.25)
        orow2.addWidget(self.seq_speed)
        side.addLayout(orow2)
        prow = QHBoxLayout()
        self.play_seq = QPushButton("Play Sequence")
        self.play_seq.setObjectName("Primary")
        self.play_seq.clicked.connect(self.play_sequence)
        prow.addWidget(self.play_seq)
        self.loop = QCheckBox("Loop")
        prow.addWidget(self.loop)
        stop = QPushButton("Stop")
        stop.clicked.connect(lambda: self.trigger("Idle"))
        prow.addWidget(stop)
        side.addLayout(prow)

        # ---- emotion mixer
        side.addWidget(label("Emotion mixer", "H3"))
        self.mix_on = QCheckBox("Use mixer (overrides the emotion system)")
        self.mix_on.toggled.connect(lambda _on: self._mix_changed())
        side.addWidget(self.mix_on)
        self.mix: dict[str, QSlider] = {}
        mgrid = QGridLayout()
        for i, (text, _e) in enumerate(MIXER + [("Energy", None)]):
            mgrid.addWidget(label(text, "Muted"), i, 0)
            sl = QSlider(Qt.Horizontal)
            sl.setRange(0, 100)
            sl.setValue(60 if text == "Energy" else 0)
            sl.valueChanged.connect(lambda _v: self._mix_changed())
            mgrid.addWidget(sl, i, 1)
            self.mix[text] = sl
        side.addLayout(mgrid)
        rb = QPushButton("Reset mixer")
        rb.clicked.connect(self.reset_mixer)
        side.addWidget(rb)

        # ---- speed
        side.addWidget(label("Speed", "H3"))
        srow = QHBoxLayout()
        srow.setSpacing(4)
        grp = QButtonGroup(self)
        self.speed_buttons = {}
        for sp in SPEEDS:
            b = self._chip(f"{sp:g}x")
            b.setCheckable(True)
            b.setChecked(sp == 1.0)
            b.clicked.connect(lambda _=False, v=sp: self.set_speed(v))
            grp.addButton(b)
            srow.addWidget(b)
            self.speed_buttons[sp] = b
        side.addLayout(srow)

        # ---- debug
        side.addWidget(label("Debug", "H3"))
        dgrid = QGridLayout()
        self.toggles = {}
        for i, (key, text) in enumerate(DEBUG):
            cb = QCheckBox(text)
            cb.toggled.connect(lambda on, k=key: self._debug(k, on))
            dgrid.addWidget(cb, i // 2, i % 2)
            self.toggles[key] = cb
        self.part = QCheckBox("Particles")
        self.part.setChecked(True)
        self.part.toggled.connect(lambda on: setattr(self.sim.particles, "enabled", on))
        dgrid.addWidget(self.part, len(DEBUG) // 2 + 1, 0)
        side.addLayout(dgrid)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(side_w)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(460)
        side_w.setMaximumWidth(440)
        root.addWidget(scroll, 0)
        self.canvas = PlaygroundCanvas(self.sim)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.canvas, 1)
        self._fill_actions()

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self._last = time.perf_counter()
        if standalone:
            self.timer.start(16)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _chip(text: str) -> QPushButton:
        b = QPushButton(text)
        b.setObjectName("Chip")
        b.setMinimumWidth(0)
        b.setStyleSheet("padding: 6px 6px;")
        return b

    @staticmethod
    def _spin(lo: float, hi: float, val: float, suffix: str, step: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    # -------------------------------------------------------------- page API
    def refresh(self) -> None:
        pass

    def tick(self) -> None:
        pass

    def showEvent(self, ev):
        self._last = time.perf_counter()
        self.timer.start(16)
        super().showEvent(ev)

    def hideEvent(self, ev):
        self.timer.stop()        # no animation cost while the page is hidden
        super().hideEvent(ev)

    # ------------------------------------------------------------- controls
    def _fill_actions(self, *_):
        """Searchable, categorised list of everything this character can do (★ = species special)."""
        sp = self.sim.species
        q = self.search.text().strip().lower().replace(" ", "_") if hasattr(self, "search") else ""
        self.tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        for n in sorted(act.available(sp)):
            a = act.ACTIONS[n]
            if q and q not in n.lower() and q not in a.category:
                continue
            cat = a.category.title()
            if cat not in groups:
                groups[cat] = QTreeWidgetItem([cat])
                groups[cat].setFlags(Qt.ItemIsEnabled)
                self.tree.addTopLevelItem(groups[cat])
            tag = " ★" if a.species is not None and a.category == "species" else ""
            it = QTreeWidgetItem([f"{n.replace('_', ' ').title()}{tag}"])
            it.setData(0, Qt.UserRole, n)
            groups[cat].addChild(it)
        self.tree.expandAll()

    def action_names(self) -> list[str]:
        out = []
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i)
            out += [g.child(j).data(0, Qt.UserRole) for j in range(g.childCount())]
        return out

    def selected_action(self) -> str:
        it = self.tree.currentItem()
        return (it.data(0, Qt.UserRole) or "") if it is not None else ""

    def _character_changed(self, *_):
        self.sim.mode = self.mode.currentData() or "production"
        self.sim.set_character(self.char.currentData() or "cat")
        self.sim.set_quality(self.quality.currentData() or "medium")
        self.canvas.sim = self.sim
        self.sim.particles.enabled = self.part.isChecked()
        if self.life_box.isChecked():
            self.sim.set_life(True)
        self._fill_actions()
        self._mix_changed()

    def body_language_test(self) -> None:
        """Required quality test: every icon / particle off, emotions must read from body + face alone."""
        self.effects_box.setChecked(False)
        self._bl_index = 0
        self._bl_next = 0.0
        self._bl_on = True

    def _body_language_tick(self) -> None:
        if not getattr(self, "_bl_on", False) or self.sim.t < self._bl_next:
            return
        if self._bl_index >= len(BODY_TEST):
            self._bl_on = False
            self.sim.set_emotion("NEUTRAL")
            return
        self.sim.set_emotion(BODY_TEST[self._bl_index])
        self.canvas.caption = f"Body-language test: {BODY_TEST[self._bl_index].lower()} (no icons)"
        self._bl_index += 1
        self._bl_next = self.sim.t + 3.0

    def _debug(self, key: str, on: bool) -> None:
        self.canvas.debug[key] = on
        self.canvas.update()

    def _mix_changed(self) -> None:
        if not self.mix_on.isChecked():
            self.sim.set_mix(None)
            return
        weights = {e: self.mix[text].value() / 100 for text, e in MIXER if self.mix[text].value() > 0}
        self.sim.set_mix(weights or {Emotion.NEUTRAL: 1.0}, energy=float(self.mix["Energy"].value()))

    def reset_mixer(self) -> None:
        for text, sl in self.mix.items():
            sl.setValue(60 if text == "Energy" else 0)
        self.mix_on.setChecked(False)
        self.sim.set_mix(None)

    def set_speed(self, v: float) -> None:
        self.speed = v
        b = self.speed_buttons.get(v)
        if b is not None and not b.isChecked():
            b.setChecked(True)

    def trigger(self, name: str) -> None:
        if name:
            self.sim.trigger(name)

    def sim_validation(self) -> None:
        self.sim.run_validation()

    def add_action(self) -> None:
        n = self.selected_action()
        if not n:
            return
        secs = self.dur.value()
        self.add_named(n, secs if secs > 0 else None)

    def add_named(self, name: str, secs: float | None) -> None:
        item = QListWidgetItem(name.replace("_", " ").title() + (f"  ·  {secs:g} s" if secs else ""))
        item.setData(Qt.UserRole, (name.upper(), secs))
        self.timeline.addItem(item)

    def remove_action(self) -> None:
        row = self.timeline.currentRow()
        if row >= 0:
            self.timeline.takeItem(row)

    def move(self, d: int) -> None:
        row = self.timeline.currentRow()
        new = row + d
        if row < 0 or not (0 <= new < self.timeline.count()):
            return
        it = self.timeline.takeItem(row)
        self.timeline.insertItem(new, it)
        self.timeline.setCurrentRow(new)

    def items(self) -> list[tuple[str, float | None]]:
        return [self.timeline.item(i).data(Qt.UserRole) for i in range(self.timeline.count())]

    def play_sequence(self) -> None:
        items = self.items()
        if items:
            if abs(self.seq_speed.value() - 1.0) > 1e-3:
                self.set_speed(min(SPEEDS, key=lambda s: abs(s - self.seq_speed.value())))
            self.sim.play_sequence(items, self.loop.isChecked(), reverse=self.reverse.isChecked(),
                                   delay=self.delay.value(), blend=self.blend.value())

    # ----------------------------------------------------------------- frame
    def step(self, dt: float) -> None:
        self._body_language_tick()
        self.sim.step(dt)
        b = self.sim.b
        self.canvas.cam_x += (b.x - self.canvas.cam_x) * min(1.0, dt * 6)
        self.canvas.lines = self.sim.info(self.canvas.debug) + [f"x{self.speed:g}"]

    def _tick(self) -> None:
        now = time.perf_counter()
        real = min(0.1, now - self._last)
        self._last = now
        self.step(real * self.speed)
        self.canvas.update()
