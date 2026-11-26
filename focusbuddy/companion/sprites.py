"""Sprite-pack characters (PNG frames / sprite sheets).

Drop a folder into ``assets/characters/<id>/`` (or the user data folder
``characters/<id>/``) containing ``manifest.json``::

    {
      "id": "pixelcat",
      "name": "Pixel Cat",
      "default_name": "Pixel",
      "emoji": "🐈",
      "personality": "cat",
      "faces": "right",
      "animations": {
        "IDLE":    {"frames": ["idle_0.png", "idle_1.png"], "fps": 4},
        "WALKING": {"sheet": "walk.png", "count": 8, "fps": 10},
        "SLEEPING": {"frames": ["sleep_0.png"], "fps": 1}
      }
    }

Frames are drawn bottom-centred on the companion canvas.  Missing states
fall back along FALLBACKS (e.g. CHASING -> RUNNING -> WALKING -> IDLE), so a
pack only needs an IDLE animation to work.  Effects (!, ?, zzz, sparkles)
are still drawn on top procedurally.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from PySide6.QtCore import QRectF
from PySide6.QtGui import QPainter, QPixmap

from ..utils.logging_setup import get_logger
from .animation import Pose
from .characters import BUILTIN, CharacterSpec, register
from .renderer import GROUND_MARGIN, UNITS, BaseRenderer

log = get_logger("sprites")

FALLBACKS = {
    "RUNNING": "WALKING", "CHASING": "RUNNING", "WALKING": "IDLE",
    "SITTING": "IDLE", "FOCUS_MODE": "SITTING", "SLEEPING": "SITTING",
    "LOOKING_AROUND": "IDLE", "STRETCHING": "IDLE", "PLAYING": "CELEBRATING",
    "WAVING": "CELEBRATING", "CELEBRATING": "IDLE", "WATCHING": "ALERT",
    "ALERT": "IDLE", "WARNING": "ALERT", "CLOSING_DISTRACTION": "WARNING",
    "CONFUSED": "IDLE", "HELD": "ALERT", "FALLING": "HELD",
}


class SpriteAnimation:
    def __init__(self, frames: list[QPixmap], fps: float):
        self.frames = [f for f in frames if not f.isNull()]
        self.fps = max(0.5, fps)

    def frame(self, t: float) -> QPixmap | None:
        if not self.frames:
            return None
        return self.frames[int(t * self.fps) % len(self.frames)]


def load_manifest(folder: Path) -> dict:
    with open(folder / "manifest.json", encoding="utf-8") as fh:
        return json.load(fh)


def _load_animation(folder: Path, spec: dict) -> SpriteAnimation:
    fps = float(spec.get("fps", 8))
    if "sheet" in spec:
        sheet = QPixmap(str(folder / spec["sheet"]))
        n = max(1, int(spec.get("count", 1)))
        w = sheet.width() // n
        frames = [sheet.copy(i * w, 0, w, sheet.height()) for i in range(n)]
    else:
        frames = [QPixmap(str(folder / f)) for f in spec.get("frames", [])]
    return SpriteAnimation(frames, fps)


class SpriteRenderer(BaseRenderer):
    def __init__(self, spec: CharacterSpec):
        super().__init__(spec)
        folder = Path(spec.sprite_dir or "")
        manifest = load_manifest(folder)
        self.faces_right = manifest.get("faces", "right") == "right"
        self.anims: dict[str, SpriteAnimation] = {}
        for state, aspec in manifest.get("animations", {}).items():
            anim = _load_animation(folder, aspec)
            if anim.frames:
                self.anims[state.upper()] = anim
        if "IDLE" not in self.anims:
            raise ValueError(f"sprite pack {folder} has no IDLE animation")
        self.state = "IDLE"
        self._t0 = time.perf_counter()
        self._last_state = None
        self.head_pos = (10.0, -70.0)
        self.head_r = 22.0

    def resolve(self, state: str) -> SpriteAnimation:
        seen = set()
        s = state
        while s not in self.anims and s not in seen:
            seen.add(s)
            s = FALLBACKS.get(s, "IDLE")
        return self.anims.get(s, self.anims["IDLE"])

    def paint(self, g: QPainter, rect: QRectF, pose: Pose, facing: int = 1, state: str | None = None) -> None:
        state = state or self.state
        if state != self._last_state:
            self._last_state = state
            self._t0 = time.perf_counter()
        anim = self.resolve(state)
        pm = anim.frame(time.perf_counter() - self._t0)
        g.save()
        g.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if pm is not None:
            s = rect.width() / UNITS
            target_h = rect.height() - GROUND_MARGIN * s
            ratio = pm.width() / max(1, pm.height())
            h = target_h * 0.92
            w = h * ratio
            x = rect.center().x() - w / 2
            y = rect.bottom() - GROUND_MARGIN * s - h - pose.lift * s
            flip = (facing < 0) == self.faces_right
            if flip:
                g.translate(rect.center().x(), 0)
                g.scale(-1, 1)
                g.translate(-rect.center().x(), 0)
            g.drawPixmap(QRectF(x, y, w, h), pm, QRectF(pm.rect()))
        g.restore()
        # procedural effect marks on top
        g.save()
        s = rect.width() / UNITS
        g.translate(rect.center().x(), rect.bottom() - GROUND_MARGIN * s)
        g.scale(s, s)
        self._marks(g, pose, facing)
        g.restore()

    def draw(self, g, pose):  # not used
        pass


def discover_sprite_packs(*folders: Path) -> list[CharacterSpec]:
    """Register every valid sprite pack found in ``folders``."""
    found = []
    for root in folders:
        if not root or not root.exists():
            continue
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            mf = sub / "manifest.json"
            if not mf.exists():
                continue
            try:
                m = load_manifest(sub)
                base = BUILTIN.get(m.get("personality", "cat"), BUILTIN["cat"])
                spec = CharacterSpec(
                    id=m.get("id", sub.name), display_name=m.get("name", sub.name),
                    default_name=m.get("default_name", m.get("name", sub.name)),
                    emoji=m.get("emoji", base.emoji), skeleton="sprite", species=base.species,
                    personality=m.get("personality", base.personality),
                    traits=tuple(m.get("traits", base.traits)), tagline=m.get("tagline", "Custom character"),
                    palette=base.palette, sprite_dir=str(sub))
                register(spec)
                found.append(spec)
                log.info("Sprite pack registered id=%s", spec.id)
            except Exception as exc:  # noqa: BLE001 - a bad pack must not crash startup
                log.warning("Invalid sprite pack folder=%s err=%s", sub.name, exc)
    return found
