"""Animation Studio - developer preview for every character animation.

Runs the *real* LocomotionController + AnimationController + EmotionController
(the same engine the desktop companion uses) on a scrolling stage, with
play / pause / restart and 0.25x-2x speed.

    python main.py --studio
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QHBoxLayout, QListWidget, QPushButton, QVBoxLayout,
                               QWidget)

from ..companion.animation import AnimationController
from ..companion.characters import all_characters, get_character
from ..companion.emotion import Emotion, EmotionController
from ..companion.movement import Mover
from ..companion.profile import get_profile
from ..companion.renderer import make_renderer
from ..companion.states import CompanionState as S
from .theme import C, stylesheet
from .widgets import label

ANIMATIONS = ["Idle", "Walk", "Run", "Turn", "Sit", "Sleep", "Wake", "Jump", "Celebrate", "Alert", "Curious",
              "Happy", "Sad", "Angry", "Stretch", "Yawn", "Groom", "Scratch", "Peek", "Read", "Press close button",
              "Petted", "Focus mode", "Warning"]

STATIC = {"Idle": (S.IDLE, None), "Sit": (S.SITTING, None), "Sleep": (S.SLEEPING, Emotion.SLEEPY),
          "Celebrate": (S.CELEBRATING, Emotion.CELEBRATING), "Alert": (S.ALERT, Emotion.ALERT),
          "Curious": (S.LOOKING_AROUND, Emotion.CURIOUS), "Happy": (S.IDLE, Emotion.HAPPY),
          "Sad": (S.IDLE, Emotion.SAD), "Angry": (S.WARNING, Emotion.ANNOYED), "Stretch": (S.STRETCHING, None),
          "Yawn": (S.YAWNING, Emotion.SLEEPY), "Groom": (S.GROOMING, None), "Scratch": (S.SCRATCHING, None),
          "Peek": (S.PEEKING, Emotion.CURIOUS), "Read": (S.READING, None), "Petted": (S.PETTED, Emotion.HAPPY),
          "Focus mode": (S.FOCUS_MODE, None), "Warning": (S.WARNING, Emotion.SUSPICIOUS)}

LOOPED = {"Stretch": 2.6, "Yawn": 2.2, "Alert": 1.6, "Celebrate": 3.0}


class StudioCanvas(QWidget):
    CHAR = 220.0

    def __init__(self):
        super().__init__()
        self.setMinimumSize(620, 380)
        self.renderer = None
        self.pose = None
        self.cam_x = 0.0
        self.char_x = 0.0
        self.char_y = 0.0
        self.info: list[str] = []

    def paintEvent(self, _):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing)
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0, QColor("#EEF2FF"))
        grad.setColorAt(1, QColor("#F9FAFE"))
        g.fillRect(self.rect(), grad)
        ground = self.height() - 70
        g.setPen(QPen(QColor("#C9D0E0"), 2))
        g.drawLine(0, ground, self.width(), ground)
        g.setPen(QPen(QColor("#D9DEEA"), 1))
        step = 60
        off = -(self.cam_x % step)
        x = off
        while x < self.width():
            g.drawLine(QPointF(x, ground), QPointF(x - 14, ground + 14))
            x += step
        if self.renderer is not None and self.pose is not None:
            cx = self.width() / 2 + (self.char_x - self.cam_x)
            feet_y = ground + self.char_y
            r = QRectF(cx - self.CHAR / 2, feet_y - self.CHAR + 6 * self.CHAR / 140, self.CHAR, self.CHAR)
            self.renderer.paint(g, r, self.pose, 1)
        g.setPen(QColor(C["muted"]))
        f = QFont()
        f.setPixelSize(12)
        g.setFont(f)
        for i, ln in enumerate(self.info):
            g.drawText(QPointF(14, 22 + i * 17), ln)


class AnimationStudio(QWidget):
    def __init__(self, character: str = "cat"):
        super().__init__()
        self.setWindowTitle("FocusBuddy - Animation Studio")
        self.setObjectName("Root")
        self.setStyleSheet(stylesheet())
        self.resize(980, 560)
        root = QHBoxLayout(self)
        side = QVBoxLayout()
        side.addWidget(label("Animation Studio", "H2"))
        self.char = QComboBox()
        for spec in all_characters():
            self.char.addItem(f"{spec.emoji}  {spec.display_name}", spec.id)
        self.char.setCurrentIndex(max(0, self.char.findData(character)))
        self.char.currentIndexChanged.connect(self._restart)
        side.addWidget(self.char)
        self.list = QListWidget()
        self.list.addItems(ANIMATIONS)
        self.list.setCurrentRow(1)
        self.list.currentRowChanged.connect(self._restart)
        side.addWidget(self.list, 1)
        self.emotion = QComboBox()
        self.emotion.addItem("Emotion: automatic", "")
        for e in Emotion:
            self.emotion.addItem(f"Emotion: {e.value.title()}", e.value)
        self.emotion.currentIndexChanged.connect(self._emotion_changed)
        side.addWidget(self.emotion)
        speeds = QHBoxLayout()
        self.speed = 1.0
        grp = QButtonGroup(self)
        for sp in (0.25, 0.5, 1.0, 2.0):
            b = QPushButton(f"{sp:g}x")
            b.setObjectName("Chip")
            b.setCheckable(True)
            b.setChecked(sp == 1.0)
            b.clicked.connect(lambda _=False, v=sp: setattr(self, "speed", v))
            grp.addButton(b)
            speeds.addWidget(b)
        side.addLayout(speeds)
        ctl = QHBoxLayout()
        self.play_btn = QPushButton("Pause")
        self.play_btn.setObjectName("Primary")
        self.play_btn.clicked.connect(self._toggle)
        rst = QPushButton("Restart")
        rst.clicked.connect(self._restart)
        ctl.addWidget(self.play_btn)
        ctl.addWidget(rst)
        side.addLayout(ctl)
        root.addLayout(side, 0)
        self.canvas = StudioCanvas()
        root.addWidget(self.canvas, 1)
        self.playing = True
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self._restart()
        self.timer.start(16)

    # ----------------------------------------------------------------
    def _restart(self, *_):
        cid = self.char.currentData() or "cat"
        spec = get_character(cid)
        self.profile = get_profile(cid if cid in ("cat", "dog", "fox", "rabbit", "robot", "buddy") else spec.personality)
        self.canvas.renderer = make_renderer(spec)
        self.anim = AnimationController(seed=7, profile=self.profile)
        if hasattr(self.canvas.renderer, "rig"):
            self.anim.set_rig(self.canvas.renderer.rig)
        self.mover = Mover(0, 0, profile=self.profile, scale=StudioCanvas.CHAR / 124.0)
        self.emo = EmotionController(self.profile.baseline)
        self.t = 0.0
        self.state = S.IDLE
        self.cycle_t = 0.0
        self.turn_dir = 1
        self.canvas.cam_x = 0.0
        self._last = time.perf_counter()
        self._emotion_changed()

    def _emotion_changed(self, *_):
        v = self.emotion.currentData()
        if v:
            self.emo.set(Emotion(v), 1.0, None, now=self.t)

    def _toggle(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")

    def current(self) -> str:
        item = self.list.currentItem()
        return item.text() if item else "Idle"

    def step(self, dt: float) -> None:
        """Advance the simulation by dt seconds (scaled time)."""
        name = self.current()
        self.t += dt
        self.cycle_t += dt
        m = self.mover
        if name in ("Walk", "Run"):
            if m.target is None or m.distance_to_target() < 400:
                m.move_to(m.pos.x + 2000, 0, gait="run" if name == "Run" else "walk")
            self.state = S.RUNNING if m.speed > m.walk_speed * 1.25 else S.WALKING
        elif name == "Turn":
            if m.target is None and not m.turning:
                self.turn_dir *= -1
                m.move_to(m.pos.x + self.turn_dir * 420, 0, gait="walk_then_run")
            self.state = S.TURNING if m.turning else (S.RUNNING if m.speed > m.walk_speed * 1.25 else
                                                      (S.WALKING if m.target is not None else S.IDLE))
        elif name == "Jump":
            if not m.airborne and self.cycle_t > 1.4:
                self.cycle_t = 0.0
                m.leap_to(m.pos.x, 0, apex=70)
            self.state = S.JUMPING if m.airborne else S.IDLE
        elif name == "Wake":
            c = self.cycle_t % 5.0
            self.state = S.SLEEPING if c < 1.6 else S.WAKING if c < 2.6 else S.STRETCHING if c < 4.6 else S.IDLE
        elif name == "Press close button":
            c = self.cycle_t % 1.6
            self.state = S.PRESSING if c < 0.9 else S.CLOSING_DISTRACTION
            if c < dt:  # restart the press timeline
                self.anim._state = None
        else:
            st, emo = STATIC.get(name, (S.IDLE, None))
            loop = LOOPED.get(name)
            if loop and self.cycle_t > loop:
                self.cycle_t = 0.0
                self.anim._state = None
            self.state = st
            if emo is not None and not self.emotion.currentData() and self.emo.current != emo:
                self.emo.set(emo, 1.0, None, now=self.t)
        m.update(dt)
        for ev, val in m.drain_events():
            self.anim.on_event(ev, val)
        self.emo.update(dt, self.t)
        pose = self.anim.update(self.state, self.t, dt, m.sample(), self.emo.style(), None, 1.0)
        self.canvas.pose = pose
        self.canvas.char_x = m.pos.x
        self.canvas.char_y = m.pos.y
        self.canvas.cam_x += (m.pos.x - self.canvas.cam_x) * min(1.0, dt * 12)
        self.canvas.info = [f"State: {self.state.value}", f"Emotion: {self.emo.current.value}",
                            f"Speed: {m.speed:.0f} px/s ({m.mode})", f"Cycle: {self.anim.phase_name}",
                            f"Phase: {math.fmod(m.phase, 2 * math.pi):.2f}   run blend {m.run_blend:.2f}",
                            f"Facing: {m.facing_blend:+.2f}   x{self.speed:g}"]

    def _tick(self):
        now = time.perf_counter()
        real = now - self._last
        self._last = now
        self.step(real * self.speed if self.playing else 0.0)
        self.canvas.update()
