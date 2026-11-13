"""Render the README pictures of the 3D companions with the app's own renderer:

    python tools/render_companion_docs.py            # docs/companions_3d.png + docs/companion_faces_3d.png

companions_3d.png: every companion in its main states (idle, walk, run, sit, wave, celebrate,
sleep). companion_faces_3d.png: the face rig of the generated companions (neutral, blink, happy squint,
smile, open mouth, sad, angry), close up.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from PySide6.QtCore import QRectF  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from focusbuddy.companion.animation import Animator  # noqa: E402
from focusbuddy.companion.characters import BUILTIN  # noqa: E402
from focusbuddy.companion.profile import get_profile  # noqa: E402
from focusbuddy.companion.renderer import make_renderer  # noqa: E402
from focusbuddy.companion.states import CompanionState as S  # noqa: E402

BG = "#EEF1F6"
ALL = ("cat", "dog", "fox", "rabbit", "robot", "buddy", "study_girl", "saree_girl")
FACED = ("cat", "dog", "fox", "rabbit", "buddy", "study_girl", "saree_girl")
PEOPLE = ("buddy", "study_girl", "saree_girl")
STATES = (("Idle", S.IDLE), ("Walk", S.WALKING), ("Run", S.RUNNING), ("Sit", S.SITTING),
          ("Wave", S.WAVING), ("Celebrate", S.CELEBRATING), ("Sleep", S.SLEEPING))
FACES = (("Neutral", {}), ("Blink", {"eye_open": 0.0}), ("Happy", {"lid_low": 1.0, "smile": 1.0}),
         ("Smile", {"smile": 1.0}), ("Open mouth", {"jaw": 1.0, "mouth_open": 1.0}),
         ("Sad", {"smile": -0.8, "brow_raise": 1.0}), ("Angry", {"smile": -0.5, "brow_raise": -1.0, "brow": -1.0}))


def qimage_to_pil(img: QImage) -> Image.Image:
    w, h = img.width(), img.height()
    a = np.frombuffer(img.constBits(), np.uint8).reshape(h, w, 4)[..., [2, 1, 0]].copy()
    return Image.fromarray(a)


def settle(r, cid, state, seconds=0.6):
    an = Animator(seed=1, profile=get_profile(cid))
    an.set_rig(r.rig)
    an._next_blink = 1e9
    p = an.pose(state, 100.0, 0.016, 60, 75)
    for k in range(1, int(seconds / 0.016) + 1):
        p = an.pose(state, 100.0 + k * 0.016, 0.016, 60, 75)
    return p


def paint(r, pose, cell, state=None) -> Image.Image:
    img = QImage(cell, cell, QImage.Format_ARGB32)
    img.fill(QColor(BG))
    g = QPainter(img)
    g.setRenderHint(QPainter.Antialiasing)
    r.paint(g, QRectF(0, 0, cell, cell), pose, 1, state)
    g.end()
    return qimage_to_pil(img)


def paint_debug(r, pose, cell) -> Image.Image:
    img = QImage(cell, cell, QImage.Format_ARGB32)
    img.fill(QColor(BG))
    g = QPainter(img)
    g.setRenderHint(QPainter.Antialiasing)
    r.paint(g, QRectF(0, 0, cell, cell), pose, 1, None, debug={"probe": True})
    g.end()
    return qimage_to_pil(img)


def font(size):
    for f in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def companions(out: Path, cell=190):
    head = 34
    sheet = Image.new("RGB", (cell * len(STATES), head + cell * len(ALL)), BG)
    d = ImageDraw.Draw(sheet)
    for i, (name, _s) in enumerate(STATES):
        d.text((i * cell + cell // 2, 8), name, fill=(60, 64, 80), font=font(18), anchor="ma")
    for j, cid in enumerate(ALL):
        r = make_renderer(BUILTIN[cid], "production")
        for i, (_n, st) in enumerate(STATES):
            sheet.paste(paint(r, settle(r, cid, st), cell, st.name), (i * cell, head + j * cell))
    sheet.save(out)


def face_tile(r, cid, pose, size=260) -> Image.Image:
    if cid in PEOPLE:
        # the people: head and shoulders, slightly turned
        r.set_orbit(20.0, -5.0, 2.5)
        crop = paint(r, pose, 900).crop((265, 0, 615, 350))
    else:
        r.set_orbit(70.0, -5.0, 2.5)
        y0 = 170 if cid == "rabbit" else 60                     # the rabbit's face sits under long ears
        crop = paint(r, pose, 700).crop((0, y0, 520, y0 + 520))
    r.reset_orbit()
    return crop.resize((size, size), Image.LANCZOS)


def faces(out: Path, size=230):
    head = 34
    sheet = Image.new("RGB", (size * len(FACES), head + size * len(FACED)), BG)
    d = ImageDraw.Draw(sheet)
    for i, (name, _kw) in enumerate(FACES):
        d.text((i * size + size // 2, 8), name, fill=(60, 64, 80), font=font(18), anchor="ma")
    for j, cid in enumerate(FACED):
        r = make_renderer(BUILTIN[cid], "production")
        base = settle(r, cid, S.IDLE)
        base = replace(base, smile=0.0, lid=0.0, lid_low=0.0, brow=0.0, brow_raise=0.0, jaw=0.0,
                       mouth_open=0.0, eye_open=1.0)
        for i, (_n, kw) in enumerate(FACES):
            sheet.paste(face_tile(r, cid, replace(base, **kw), size), (i * size, head + j * size))
    sheet.save(out)


if __name__ == "__main__":
    app = QApplication.instance() or QApplication([])
    docs = ROOT / "docs"
    which = sys.argv[1:] or ["companions", "faces"]
    if "companions" in which:
        companions(docs / "companions_3d.png")
    if "faces" in which:
        faces(docs / "companion_faces_3d.png")
