"""Illustrated characters: the companions drawn from their character-sheet art.

Each built-in character can carry an *illustrated pack* - the poses, turnaround views and
body-language poses cut out of its character sheet (``tools/build_illustrated_packs.py``)::

    assets/characters/<id>/illustrated/
        pack.json            frame list + metadata (see below)
        frames/<name>.webp   one transparent image per pose
        portrait.webp        neutral face (tray / window icon)

``pack.json``::

    {"version": 1, "kind": "quadruped", "stand_height": 0.6,
     "frames": {"idle": {"file": "frames/idle.webp", "facing": -1, "h": 1.0, "w": 1.31,
                         "cx": 0.52, "head": [0.22, 0.12]}, ...},
     "cycles": {"walk": ["walk", "right"], "run": ["run", "sprint"]},
     "map": {"SITTING": ["sit"]}}

``h`` / ``w`` are the pose's size relative to the standing height (so a lying pose is
shorter than a standing one), ``cx`` the ground-contact point as a fraction of the image
width, ``head`` the head position (fractions of the image) for the emotion marks, and
``facing`` the way the pose looks (1 right, -1 left, 0 the viewer); poses are mirrored to
face the way the companion moves.

The renderer is driven by the same state machine, emotions, locomotion and pose channels as
the 3D characters. It picks the pose for the state / emotion, plays walk and run as pose
cycles locked to the leg phase, cross-fades between poses, and layers procedural motion on
top: breathing, bob, lean, squash & stretch, jumps, a contact shadow and the emotion marks.
It cannot turn the art in 3D - turning shows the front view for a moment.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QRadialGradient, QBrush

from .animation import Pose
from .characters import CharacterSpec
from .renderer import GROUND_MARGIN, UNITS, BaseRenderer

PACK_DIRNAME = "illustrated"

# state -> preferred poses (first one the pack has wins)
DEFAULT_MAP: dict[str, list[str]] = {
    "IDLE": ["idle"],
    "LOOKING_AROUND": ["curious", "attentive", "idle"],
    "SITTING": ["sit"],
    "FOCUS_MODE": ["study", "read", "laptop", "think", "sit"],
    "READING": ["read", "study", "laptop", "sit"],
    "SLEEPING": ["sleep", "lie_down"],
    "WAKING": ["lie_down", "sleepy", "sit"],
    "YAWNING": ["yawn", "sleepy", "sit"],
    "STRETCHING": ["stretch", "play_bow"],
    "GROOMING": ["groom", "sit"],
    "SCRATCHING": ["groom", "shake", "sit"],
    "PLAYING": ["play_bow", "playful", "roll_over", "dance", "spin", "excited"],
    "WAVING": ["wave", "happy"],
    "CELEBRATING": ["excited", "dance", "happy"],
    "WATCHING": ["attentive", "curious", "idle"],
    "ALERT": ["attentive", "curious", "idle"],
    "WARNING": ["angry", "confident", "attentive"],
    "CLOSING_DISTRACTION": ["point", "confident", "angry"],
    "PRESSING": ["point", "confident", "angry"],
    "CONFUSED": ["curious", "think"],
    "PEEKING": ["curious", "attentive"],
    "PETTED": ["love", "relaxed", "happy"],
    "TURNING": ["front", "front34", "idle"],
    "JUMPING": ["jump_air"],
    "LANDING": ["jump_land", "idle"],
    "FALLING": ["jump_air", "scared"],
    "HELD": ["scared", "jump_air"],
}
CYCLES = {"WALKING": "walk", "RUNNING": "run", "CHASING": "run"}

# emotion -> body-language pose while the companion is calm (idle / sitting / watching)
EMOTION_POSES: dict[str, list[str]] = {
    "HAPPY": ["happy"], "CURIOUS": ["curious"], "ALERT": ["attentive", "curious"],
    "SUSPICIOUS": ["grumpy"], "EXCITED": ["excited"], "SLEEPY": ["sleepy"], "WORRIED": ["sad", "scared"],
    "ANNOYED": ["grumpy", "angry"], "CELEBRATING": ["excited", "happy"], "SAD": ["sad"],
    "ANGRY": ["angry"], "GRUMPY": ["grumpy"], "SCARED": ["scared"], "EMBARRASSED": ["love", "sad"],
    "LOVE": ["love"], "BORED": ["sleepy", "relaxed"], "SMUG": ["confident"], "DISGUSTED": ["grumpy"],
    "SHOCKED": ["scared"], "LAUGHING": ["happy", "excited"], "DETERMINED": ["confident"],
    "DIZZY": ["curious"], "RELIEVED": ["relaxed", "happy"], "PROUD": ["confident"],
    "SURPRISED": ["attentive", "scared"], "CONFUSED": ["curious", "think"], "CALMING": ["relaxed"],
    "PLAYFUL": ["playful"], "FOCUSED": ["study", "attentive"],
}
CALM_STATES = {"IDLE", "SITTING", "LOOKING_AROUND", "WATCHING", "ALERT", "PEEKING", "CONFUSED"}
MOVING = {"WALKING", "RUNNING", "CHASING", "JUMPING", "FALLING", "LANDING"}

FADE_S = 0.14            # cross-fade between poses
MIN_HOLD_S = 0.35        # a pose stays at least this long (no flicker between states)
EMOTION_HOLD_S = 1.2     # an emotion pose stays at least this long


def pack_dir(char_id: str) -> Optional[Path]:
    """The illustrated pack folder for a character, if one is installed."""
    try:
        from .character_asset import search_dirs
        dirs = list(search_dirs())
    except Exception:  # noqa: BLE001
        dirs = []
    from ..config import ASSETS_DIR
    dirs.append(ASSETS_DIR / "characters")
    for d in dirs:
        p = Path(d) / char_id / PACK_DIRNAME
        if (p / "pack.json").is_file():
            return p
    return None


def has_pack(char_id: str) -> bool:
    return pack_dir(char_id) is not None


def portrait_path(char_id: str) -> Optional[Path]:
    d = pack_dir(char_id)
    if d is None:
        return None
    p = d / "portrait.webp"
    return p if p.is_file() else None


class IllustratedPack:
    """pack.json + lazily decoded frames (cached per drawn height for crisp, cheap scaling)."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "pack.json").read_text(encoding="utf-8"))
        self.frames: dict = self.meta.get("frames", {})
        if "idle" not in self.frames:
            raise ValueError(f"illustrated pack {folder} has no idle pose")
        self.kind = self.meta.get("kind", "quadruped")
        self.stand_height = float(self.meta.get("stand_height", 0.6 if self.kind != "biped" else 0.8))
        self.cycles = {k: [f for f in v if f in self.frames] for k, v in (self.meta.get("cycles") or {}).items()}
        self.map = {k.upper(): list(v) for k, v in (self.meta.get("map") or {}).items()}
        self._images: dict[str, QImage] = {}
        self._scaled: dict[tuple, QPixmap] = {}

    def image(self, name: str) -> QImage:
        im = self._images.get(name)
        if im is None:
            im = QImage(str(self.folder / self.frames[name]["file"]))
            if im.isNull():
                raise ValueError(f"cannot read frame {name} in {self.folder}")
            im = im.convertToFormat(QImage.Format_ARGB32_Premultiplied)
            self._images[name] = im
        return im

    def pixmap(self, name: str, height_px: int) -> QPixmap:
        """The frame pre-scaled (smooth, high quality) to ``height_px`` device pixels."""
        h = max(4, int(height_px))
        key = (name, h)
        pm = self._scaled.get(key)
        if pm is None:
            im = self.image(name)
            pm = QPixmap.fromImage(im.scaledToHeight(h, Qt.SmoothTransformation))
            if len(self._scaled) > 96:                       # sizes change rarely; keep the cache small
                self._scaled.clear()
            self._scaled[key] = pm
        return pm

    def has(self, name: str) -> bool:
        return name in self.frames


class IllustratedRenderer(BaseRenderer):
    """Draws a character from its illustrated pack, animated by the companion's pose."""

    backend = None          # not a 3D renderer: no GPU / CPU quality budget

    def __init__(self, spec: CharacterSpec, folder: Path | None = None):
        super().__init__(spec)
        folder = folder or pack_dir(spec.id)
        if folder is None:
            raise FileNotFoundError(f"no illustrated pack for {spec.id}")
        self.pack = IllustratedPack(folder)
        self.emotion = "NEUTRAL"
        self.emotion_intensity = 0.0
        self._cur: Optional[str] = None
        self._cur_since = 0.0
        self._prev: Optional[str] = None
        self._prev_t = 0.0
        self._prev_flip = False
        self._emotion_pose: Optional[str] = None
        self._emotion_since = 0.0
        self.last_frame = ""
        self.last_ms = 0.0
        self.stats = {"mode": "illustrated"}

    # ------------------------------------------------------------ context
    def set_context(self, emotion: str = "NEUTRAL", intensity: float = 0.0) -> None:
        self.emotion = str(emotion or "NEUTRAL").upper()
        self.emotion_intensity = float(intensity or 0.0)

    def set_quality(self, _quality: str) -> None:       # images are pre-scaled; nothing to change
        pass

    def press_point(self) -> tuple[float, float]:
        return (34.0, -30.0) if self.pack.kind == "biped" else (38.0, -22.0)

    # ------------------------------------------------------------ selection
    def _first(self, names) -> Optional[str]:
        """The first pose the pack has; a pose whose feet were covered by the sheet's label
        (``"cut"``) only when no complete one is listed."""
        fallback = None
        for n in names:
            if self.pack.has(n):
                if not self.pack.frames[n].get("cut"):
                    return n
                fallback = fallback or n
        return fallback

    def select(self, state: str, pose: Pose, now: float) -> str:
        """The pose to show for this state / emotion / pose (cycles return the current cycle frame)."""
        st = (state or "IDLE").upper()
        cyc = CYCLES.get(st)
        if cyc and self.pack.cycles.get(cyc):
            frames = self.pack.cycles[cyc]
            phase = float(getattr(pose, "leg_phase", 0.0))
            if phase == 0.0:
                phase = now * (9.0 if cyc == "run" else 6.0)
            return frames[int((phase % (2 * math.pi)) / (2 * math.pi) * len(frames)) % len(frames)]
        if cyc:
            return self._first([cyc, "walk", "idle"]) or "idle"
        if float(getattr(pose, "back", 0.0)) > 0.5:
            b = self._first(["back34", "back"])
            if b:
                return b
        lie = float(getattr(pose, "lie", 0.0))
        if st in ("IDLE", "SITTING") and lie > 0.6:
            return self._first(["lie_down", "relaxed", "sleep"]) or "idle"
        if st in CALM_STATES:
            emo = None
            if self.emotion not in ("NEUTRAL", "") and self.emotion_intensity >= 0.25:
                emo = self._first(EMOTION_POSES.get(self.emotion, []))
            if emo:
                if emo != self._emotion_pose:
                    self._emotion_pose, self._emotion_since = emo, now
                return emo
            if self._emotion_pose and now - self._emotion_since < EMOTION_HOLD_S:
                return self._emotion_pose               # let a short emotion finish its beat
            self._emotion_pose = None
            if float(getattr(pose, "front", 0.0)) > 0.6 and st in ("IDLE", "WATCHING"):
                f = self._first(["front", "front34"])
                if f:
                    return f
            if st == "IDLE" and float(getattr(pose, "sit", 0.0)) > 0.6:
                return self._first(["sit"]) or "idle"
        names = self.pack.map.get(st) or DEFAULT_MAP.get(st) or ["idle"]
        return self._first(names) or "idle"

    # ------------------------------------------------------------ painting
    def paint(self, g: QPainter, rect: QRectF, pose: Pose, facing: float = 1, state: str | None = None,
              debug: dict | None = None) -> None:
        t0 = time.perf_counter()
        now = time.perf_counter()
        st = (state or "IDLE").upper()
        want = self.select(st, pose, now)
        in_cycle = st in CYCLES
        if want != self._cur:
            if self._cur is None:
                self._cur, self._cur_since = want, now
            elif in_cycle or now - self._cur_since >= MIN_HOLD_S or st in MOVING:
                if not in_cycle or self._cur not in self.pack.cycles.get(CYCLES[st], []):
                    self._prev, self._prev_t = self._cur, now      # cross-fade from the old pose
                self._cur, self._cur_since = want, now
        name = self._cur or "idle"
        self.last_frame = name

        s = rect.width() / UNITS                                   # px per unit
        ground = rect.bottom() - GROUND_MARGIN * s
        stand_px = rect.height() * self.pack.stand_height          # standing height on screen
        turn = float(getattr(pose, "turn", 1.0))
        dir_ = 1 if turn >= 0 else -1
        dpr = g.device().devicePixelRatioF() if g.device() is not None else 1.0

        g.save()
        g.setRenderHint(QPainter.SmoothPixmapTransform, True)
        g.setRenderHint(QPainter.Antialiasing, True)
        lift = max(0.0, float(getattr(pose, "lift", 0.0))) * s
        if self.shadows:
            self._shadow(g, rect.center().x(), ground, stand_px, name, lift)
        a = 1.0
        if self._prev and self._prev != name:
            a = min(1.0, (now - self._prev_t) / FADE_S)
            if a < 1.0:
                self._draw_frame(g, self._prev, rect, ground, stand_px, pose, dir_, dpr, 1.0 - a)
            else:
                self._prev = None
        self._draw_frame(g, name, rect, ground, stand_px, pose, dir_, dpr, a, marks=True)
        g.restore()
        if debug:
            g.save()
            g.setPen(QColor(229, 72, 77))
            g.drawText(rect.adjusted(4, 4, 0, 0), Qt.AlignLeft | Qt.AlignTop, f"{st} · {name}")
            g.restore()
        self.last_ms = (time.perf_counter() - t0) * 1000

    def _draw_frame(self, g: QPainter, name: str, rect: QRectF, ground: float, stand_px: float, pose: Pose,
                    dir_: int, dpr: float, opacity: float, marks: bool = False) -> None:
        fr = self.pack.frames[name]
        h = stand_px * float(fr.get("h", 1.0))
        pm = self.pack.pixmap(name, int(round(h * dpr)))
        w = pm.width() / dpr
        cx = float(fr.get("cx", 0.5))
        native = int(fr.get("facing", 0))
        flip = native != 0 and native != dir_
        s = rect.width() / UNITS
        moving = abs(float(getattr(pose, "leg_amp", 0.0))) > 0.05
        breath = float(getattr(pose, "breath", 0.0))
        squash = max(-1.0, min(1.0, float(getattr(pose, "squash", 0.0))))
        lift = max(0.0, float(getattr(pose, "lift", 0.0))) * s
        bob = float(getattr(pose, "bob", 0.0)) * s * 0.55
        lean = max(-7.0, min(7.0, float(getattr(pose, "tilt", 0.0)) * 0.3)) if moving else 0.0
        roll = float(getattr(pose, "roll", 0.0))
        sx = 1.0 + 0.07 * squash
        sy = 1.0 - 0.07 * squash + (0.012 * breath if not moving else 0.0)

        g.save()
        g.setOpacity(opacity)
        g.translate(rect.center().x(), ground - lift + bob)
        if roll:
            g.translate(0, -h * 0.45)
            g.rotate(roll)
            g.translate(0, h * 0.45)
        if lean:
            g.rotate(lean * dir_)
        g.scale(sx, sy)
        if flip:
            g.scale(-1, 1)
        g.drawPixmap(QRectF(-cx * w, -h, w, h), pm, QRectF(pm.rect()))
        g.restore()
        if marks:
            # head position (for !, ?, zzz ...) in units around the feet, facing-right frame
            hx, hy = fr.get("head", [0.5, 0.15])
            px = (float(hx) - cx) * w * (-1 if flip else 1)
            py = -(1.0 - float(hy)) * h
            d = dir_ if native != 0 else (1 if px >= 0 else -1)
            self.head_pos = (px * d / s, py / s)
            self.head_r = max(10.0, 0.18 * h / s)
            g.save()
            g.setOpacity(opacity)
            g.translate(rect.center().x(), ground)
            g.scale(s, s)
            self._marks(g, pose, d)
            g.restore()

    def _shadow(self, g: QPainter, x: float, ground: float, stand_px: float, name: str, lift: float) -> None:
        fr = self.pack.frames[name]
        w = stand_px * float(fr.get("w", 1.0)) * (0.62 if self.pack.kind != "biped" else 0.8)
        k = max(0.25, 1.0 - lift / max(1.0, stand_px * 0.8))
        rx, ry = max(8.0, w * 0.5 * k), max(3.0, stand_px * 0.045 * k)
        grad = QRadialGradient(QPointF(x, ground), rx)
        grad.setColorAt(0.0, QColor(10, 12, 24, int(90 * k)))
        grad.setColorAt(1.0, QColor(10, 12, 24, 0))
        g.save()
        g.setPen(Qt.NoPen)
        g.setBrush(QBrush(grad))
        g.translate(x, ground)
        g.scale(1.0, ry / rx)
        g.translate(-x, -ground)
        g.drawEllipse(QPointF(x, ground), rx, rx)
        g.restore()

    def draw(self, g, pose):  # BaseRenderer API (procedural drawing) - not used
        pass
