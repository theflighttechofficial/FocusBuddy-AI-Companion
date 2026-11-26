"""Character showcase window (``python main.py --character-showcase``), the body-language test
(``python main.py --body-language-test``) and the emotion showcase (``--emotion-showcase``).

Runs the real companion stack on the Playground stage: one character performs walk / jog / run /
sprint / turn / jump / land / sit / sleep / wake / stretch / wave / play and the full-body emotional
performances (laugh, angry, shout, scared, sad, happy, love, bored, celebrate) with captions.  The
body-language test switches every icon, particle and bubble off and cycles the core emotions: they
must read from posture, face and movement alone.  The emotion showcase holds 22 emotions in turn
(effects off by default).  The camera cuts between side, three-quarter, front, close-up and full-body
shots as the beats change.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from ..companion import actions as act
from ..companion.characters import all_characters
from ..companion.showcase import character_showcase_steps
from .playground import BODY_TEST, MODES, PlaygroundCanvas, PlaygroundSim
from .theme import stylesheet
from .widgets import label


EMOTION_SHOWCASE = ["NEUTRAL", "HAPPY", "CURIOUS", "SURPRISED", "CONFUSED", "SLEEPY", "SAD", "LONELY", "ANGRY",
                    "GRUMPY", "SCARED", "NERVOUS", "LOVE", "PLAYFUL", "CONFIDENT", "SMUG", "FOCUSED", "RELAXED",
                    "GRATEFUL", "JEALOUS", "IMPATIENT", "EXCITED"]
# camera shot per showcase beat: preset name (Playground orbit presets) or Close / Full
CAMERA_SHOTS = {"WALK": "Side", "JOG": "Side", "RUN": "3/4", "SPRINT": "Side", "TURN": "Front", "JUMP": "3/4",
                "LAND": "3/4", "SIT": "Front", "SLEEP": "Close", "WAKE": "Close", "STRETCH": "Side", "WAVE": "Front",
                "PLAY": "3/4", "LAUGH": "Close", "ANGRY": "3/4", "SHOUT": "Front", "SCARED": "Side", "SAD": "Close",
                "HAPPY": "3/4", "LOVE": "Close", "BORED": "Side", "CELEBRATE": "Full"}


class CharacterShowcase(QWidget):
    def __init__(self, character: str = "cat", body_language: bool = False, standalone: bool = True,
                 emotions: bool = False, cameras: bool = True):
        super().__init__()
        self.emotions = emotions
        self.cameras = cameras
        self.camera = "Full"
        if emotions:
            body_language = True
        self.setWindowTitle("FocusBuddy - Emotion showcase" if emotions else
                            "FocusBuddy - Character Showcase" if not body_language else
                            "FocusBuddy - Body-language test (no icons, no particles)")
        self.setObjectName("Root")
        self.setStyleSheet(stylesheet())
        self.resize(1100, 680)
        self.body_language = body_language
        self.sim = PlaygroundSim(character, char_px=320.0)
        self.sim.b.stumbles = False
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(label("Character Showcase" if not body_language else "Body-language test", "H2"))
        self.char = QComboBox()
        for spec in all_characters():
            self.char.addItem(f"{spec.emoji}  {spec.display_name}", spec.id)
        self.char.setCurrentIndex(max(0, self.char.findData(character)))
        self.char.currentIndexChanged.connect(lambda _i: self.restart())
        top.addWidget(self.char)
        self.mode = QComboBox()
        for text, m in MODES:
            self.mode.addItem(text, m)
        self.mode.setCurrentIndex(max(0, self.mode.findData(self.sim.mode)))
        self.mode.currentIndexChanged.connect(lambda _i: self.sim.set_mode(self.mode.currentData()))
        top.addWidget(self.mode)
        self.effects = QCheckBox("Effects (icons, particles)")
        self.effects.setChecked(not body_language)
        self.effects.toggled.connect(self.sim.set_effects)
        top.addWidget(self.effects)
        self.cycle = QCheckBox("Cycle all characters")
        self.cycle.setChecked(True)
        top.addWidget(self.cycle)
        again = QPushButton("Restart")
        again.clicked.connect(self.restart)
        top.addWidget(again)
        top.addStretch(1)
        lay.addLayout(top)
        self.canvas = PlaygroundCanvas(self.sim)
        self.canvas.show_hint = False
        lay.addWidget(self.canvas, 1)
        self.beats: list[tuple[str, float]] = []
        self.done = False
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self._last = time.perf_counter()
        self.restart()
        if standalone:
            self.timer.start(16)

    def restart(self) -> None:
        cid = self.char.currentData() or "cat"
        self.sim.mode = self.mode.currentData() or self.sim.mode
        self.sim.set_character(cid)
        self.sim.b.stumbles = False
        self.sim.set_effects(self.effects.isChecked())
        self.canvas.sim = self.sim
        self.beats = []
        self.done = False
        b = self.sim.b
        self.sim.orbit_preset("Reset")
        if self.body_language:
            self._bl = list(EMOTION_SHOWCASE if self.emotions else BODY_TEST)
            self._bl_next = self.sim.t
        else:
            steps = character_showcase_steps(b, mark=self._mark)
            steps.append(("call", self._finished))
            b._start_plan(steps, mode="directed", priority=act.Priority.SPECIAL_ACTION)

    def _mark(self, beat: str) -> None:
        self.beats.append((beat, round(self.sim.t, 2)))
        self.canvas.caption = f"{self.sim.spec.display_name}: {beat.replace('_', ' ').lower()}"
        if self.cameras:
            self.shot(CAMERA_SHOTS.get(beat, "Full"))

    def shot(self, name: str) -> None:
        """Cut the camera: an orbit preset, a close-up (3/4, zoomed) or the full-body default view."""
        sim = self.sim
        if name == "Full":
            sim.orbit_preset("Reset")
        elif name == "Close":
            sim.orbit_preset("3/4")
            sim.orbit[2] = 1.4
            sim._apply_orbit()
        else:
            sim.orbit_preset(name)
        self.camera = name

    def _finished(self) -> None:
        self.done = True
        if self.cycle.isChecked():
            QTimer.singleShot(800, self._next_character)

    def _next_character(self) -> None:
        i = (self.char.currentIndex() + 1) % self.char.count()
        self.char.setCurrentIndex(i)          # restart() via the signal

    def step(self, dt: float) -> None:
        if self.body_language and self.sim.t >= self._bl_next:
            if self._bl:
                e = self._bl.pop(0)
                self.sim.set_emotion(e)
                self.beats.append((e, round(self.sim.t, 2)))
                self.canvas.caption = f"{e.lower()}  -  no icons, no particles, no text"
                if self.emotions and self.cameras:
                    self.shot(("3/4", "Close", "Front", "Side")[len(self.beats) % 4])
                self._bl_next = self.sim.t + (2.8 if self.emotions else 3.2)
            elif not self.done:
                self._finished()
        self.sim.step(dt)
        b = self.sim.b
        self.canvas.cam_x += (b.x - self.canvas.cam_x) * min(1.0, dt * 4)
        self.canvas.lines = []

    def _tick(self) -> None:
        now = time.perf_counter()
        real = min(0.1, now - self._last)
        self._last = now
        self.step(real)
        self.canvas.update()
