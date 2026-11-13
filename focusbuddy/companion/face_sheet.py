"""Sheet faces: the companion's facial expressions, painted from its character sheet.

The detailed character sheets draw every character's face in twelve expressions. The build
(``tools/face_expressions.py``) lines those panels up with the 3D model's face and stores them as
small "face canvases" (``face/<expression>.webp``) plus a *face map*: the texels of the model's
texture that show the face, each with its spot on the canvas and a blend weight. At runtime
:class:`SheetFace` blends the wanted canvas into the renderer's own copy of the texture (only the
face texels change), and :class:`ExpressionPicker` chooses the expression from the pose: the
companion's emotion, its blinks, sleep and a few mouth / eye cues.

Asset metadata::

    "face": {"mode": "sheet", "map": "face/face_map.npz", "expressions": [...], "default": "neutral"}
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("focusbuddy.companion")

# emotion (focusbuddy.companion.emotion.Emotion value) -> wanted sheet expressions, best first
EMOTION_EXPR = {
    "NEUTRAL": ["neutral"], "HAPPY": ["happy"], "CURIOUS": ["curious"], "ALERT": ["curious", "surprised"],
    "SUSPICIOUS": ["mischievous", "annoyed", "confident"], "EXCITED": ["excited", "cheerful", "happy"],
    "SLEEPY": ["sleepy"], "WORRIED": ["sad"], "ANNOYED": ["annoyed", "angry"], "CELEBRATING": ["excited", "cheerful"],
    "SAD": ["sad"], "ANGRY": ["angry", "annoyed"], "GRUMPY": ["annoyed", "angry"], "SCARED": ["surprised"],
    "EMBARRASSED": ["affection", "playful", "happy"], "LOVE": ["affection", "cheerful", "happy"],
    "BORED": ["sleepy", "sad"], "SMUG": ["mischievous", "confident"], "DISGUSTED": ["annoyed", "angry"],
    "SHOCKED": ["surprised"], "LAUGHING": ["excited", "cheerful", "happy"], "DETERMINED": ["focused", "confident"],
    "DIZZY": ["surprised", "curious"], "RELIEVED": ["happy"], "PROUD": ["confident"], "SURPRISED": ["surprised"],
    "CONFUSED": ["curious"], "CALMING": ["neutral"], "CONFIDENT": ["confident"], "LONELY": ["sad"],
    "PLAYFUL": ["playful", "mischievous"], "JEALOUS": ["annoyed", "angry"], "IMPATIENT": ["annoyed", "angry"],
    "RELAXED": ["happy", "neutral"], "FOCUSED": ["focused", "confident"], "NERVOUS": ["sad", "surprised"],
    "GRATEFUL": ["affection", "happy"],
}
# what a missing expression falls back to
FALLBACK = {"affection": ["happy"], "focused": ["confident"], "cheerful": ["excited", "happy"],
            "excited": ["cheerful", "happy"], "annoyed": ["angry"], "angry": ["annoyed"],
            "mischievous": ["playful", "happy"], "playful": ["mischievous", "happy"], "blink": ["sleepy"],
            "confident": ["happy"], "curious": ["surprised", "neutral"], "surprised": ["curious"]}
# mouth / eye cues when the emotion itself is neutral (actions, petting, talking)
MOUTH_EXPR = {"grin": "mischievous", "w": "playful", "o": "surprised", "open": "happy", "frown": "sad"}
EYE_EXPR = {"happy": "happy", "wide": "surprised", "focused": "focused"}


class ExpressionPicker:
    """Pose -> expression name, with a little hold so the face never flickers between two
    expressions (blinks are instant)."""

    HOLD = 0.18   # seconds an expression must be wanted before it shows

    def __init__(self, names, default: str = "neutral"):
        self.names = set(names)
        self.default = default if default in self.names else (sorted(self.names)[0] if self.names else "")
        self.shown = self.default
        self._want = self.default
        self._since = 0.0

    def resolve(self, name: str) -> str:
        if name in self.names:
            return name
        for alt in FALLBACK.get(name, []):
            if alt in self.names:
                return alt
        return self.default

    def wanted(self, pose, state: Optional[str] = None) -> str:
        g = lambda k, d=None: getattr(pose, k, d)  # noqa: E731
        st = str(state or "").upper()
        eye_style = g("eye_style", "normal") or "normal"
        mouth = g("mouth", "smile") or "smile"
        if st == "SLEEPING" or eye_style == "closed" or mouth in ("sleep", "yawn"):
            return self.resolve("sleepy")
        if float(g("eye_open", 1.0)) < 0.4 and eye_style in ("normal", "wide", "focused"):
            return self.resolve("blink")
        emo = str(g("emotion", "") or "").upper()
        if emo and emo != "NEUTRAL":
            for name in EMOTION_EXPR.get(emo, []):
                if name in self.names:
                    return name
        if eye_style in EYE_EXPR:
            return self.resolve(EYE_EXPR[eye_style])
        if mouth in MOUTH_EXPR:
            return self.resolve(MOUTH_EXPR[mouth])
        if float(g("smile", 0.0)) > 0.55:
            return self.resolve("happy")
        return self.default

    def pick(self, pose, state: Optional[str], now: float) -> str:
        want = self.wanted(pose, state)
        if want != self._want:
            self._want, self._since = want, now
        instant = want == self.resolve("blink") or self.shown == self.resolve("blink")
        if want != self.shown and (instant or now - self._since >= self.HOLD):
            self.shown = want
        return self.shown


class SheetFace:
    """Blends the face canvases into a texture (an (H, W, 4) uint8 array the renderer samples)."""

    def __init__(self, folder: Path, meta: dict, cache: int = 5):
        z = np.load(Path(folder) / meta.get("map", "face/face_map.npz"))
        self.folder = Path(folder)
        self.idx = z["idx"].astype(np.int64)
        uv = z["uv"].astype(np.float32) - 0.5
        self.a = (z["a"].astype(np.float32) / 255.0)[:, None]
        self.size = tuple(int(x) for x in z["size"]) if "size" in z else None
        # bilinear taps into the 256 x 256 canvases, precomputed once
        self.cs = 256
        x0 = np.clip(np.floor(uv[:, 0]), 0, self.cs - 2).astype(np.int64)
        y0 = np.clip(np.floor(uv[:, 1]), 0, self.cs - 2).astype(np.int64)
        self.fx = np.clip(uv[:, 0] - x0, 0, 1)[:, None].astype(np.float32)
        self.fy = np.clip(uv[:, 1] - y0, 0, 1)[:, None].astype(np.float32)
        self.i00 = y0 * self.cs + x0
        self.names = [n for n in meta.get("expressions", []) if (self.folder / "face" / f"{n}.webp").exists()]
        self.default = meta.get("default", "neutral")
        self.current = self.default
        self.tex: Optional[np.ndarray] = None
        self.base: Optional[np.ndarray] = None
        self._cache: OrderedDict = OrderedDict()
        self._keep = cache

    def bind(self, tex: np.ndarray) -> None:
        """Paint into ``tex`` (the renderer's own copy of the base-colour texture)."""
        if self.size is not None and tuple(tex.shape[:2]) != self.size:
            raise ValueError(f"face map is for a {self.size} texture, got {tex.shape[:2]}")
        self.tex = tex
        self.base = tex.reshape(-1, tex.shape[2])[self.idx, :3].copy()
        self.current = self.default             # the default face is baked into the texture

    def _canvas(self, name: str) -> np.ndarray:
        from PIL import Image
        with Image.open(self.folder / "face" / f"{name}.webp") as im:
            return np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3)

    def colors(self, name: str) -> np.ndarray:
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        c = self._canvas(name)
        i = self.i00
        top = c[i] * (1 - self.fx) + c[i + 1] * self.fx
        bot = c[i + self.cs] * (1 - self.fx) + c[i + self.cs + 1] * self.fx
        col = top * (1 - self.fy) + bot * self.fy
        out = np.clip(self.base * (1 - self.a) + col * self.a + 0.5, 0, 255).astype(np.uint8)
        self._cache[name] = out
        while len(self._cache) > self._keep:
            self._cache.popitem(last=False)
        return out

    def show(self, name: str) -> bool:
        """Show an expression; True when the texture changed."""
        if self.tex is None or name == self.current or name not in self.names:
            return False
        flat = self.tex.reshape(-1, self.tex.shape[2])
        flat[self.idx, :3] = self.base if name == self.default else self.colors(name)
        self.current = name
        return True


def make_sheet_face(asset, meta: dict) -> Optional[SheetFace]:
    """A SheetFace for a loaded asset (None when the asset has no usable sheet face)."""
    folder = getattr(asset, "path", None)
    if folder is None:
        return None
    folder = Path(folder)
    if folder.is_file():
        folder = folder.parent
    try:
        return SheetFace(folder, meta)
    except Exception as e:  # noqa: BLE001 - a broken face never breaks the companion
        log.warning("Sheet face for %s could not be loaded (%s)", getattr(asset, "id", "?"), e)
        return None
