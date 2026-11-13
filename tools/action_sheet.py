"""Render every character performing its signature actions into one PNG (visual QA).

    python tools/action_sheet.py out.png [2d|3d] [actions|emotions|faces]

``faces`` renders the parametric face system: every character showing a set of
emotions driven only by the EmotionController (no action recipe).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from focusbuddy.companion.actions import ActionPlayer, resolve  # noqa: E402
from focusbuddy.companion.animation import AnimationController  # noqa: E402
from focusbuddy.companion.characters import BUILTIN  # noqa: E402
from focusbuddy.companion.profile import get_profile  # noqa: E402
from focusbuddy.companion.renderer import make_renderer  # noqa: E402
from focusbuddy.companion.states import CompanionState as S  # noqa: E402

ROWS = {
    "cat": ["LOAF", "LICK_PAW", "POUNCE", "ARCH_BACK", "LOOK_UP", "ROLL", "STRETCH", "PAW_AT_CURSOR", "SLEEP", "HAPPY"],
    "dog": ["PLAY_BOW", "PANT", "PAW", "DOWN", "SHAKE", "SNIFF", "TAIL_WAG", "SCRATCH", "CHEER", "CURIOUS"],
    "fox": ["SNEAK", "ALERT", "TAIL_SWING", "SNIFF", "LOOK_AROUND", "STRETCH", "POUNCE", "SIT", "PROUD", "WORRIED"],
    "rabbit": ["FREEZE", "NOSE_TWITCH", "GROOM", "BOUNCE", "LOAF", "DOUBLE_HOP", "STRETCH", "SIT", "SURPRISED", "CLAP"],
    "robot": ["BOOT", "SCAN", "PROCESSING", "CHARGE", "LOW_BATTERY", "ERROR", "RECALIBRATE", "SYSTEM_CHECK",
              "HEAD_ROTATION", "DANCE"],
    "buddy": ["THINK", "READ", "WRITE", "CLAP", "THUMBS_UP", "POINT", "WAVE", "CHEER", "STRETCH", "SHAKE_HEAD"],
}


EMOTIONS = ["ANGRY", "SHOUT", "GRUMPY", "TANTRUM", "SULK", "SCARED", "CRY", "EMBARRASSED", "LOVE", "LAUGH",
            "BORED", "SMUG", "DISGUSTED", "SHOCKED", "DIZZY", "DETERMINED", "RELIEVED", "PROUD"]


FACES = ["NEUTRAL", "HAPPY", "CURIOUS", "SURPRISED", "CONFUSED", "SLEEPY", "SAD", "LONELY", "ANGRY", "GRUMPY",
         "SCARED", "NERVOUS", "LOVE", "PLAYFUL", "CONFIDENT", "SMUG", "FOCUSED", "RELAXED", "GRATEFUL", "JEALOUS",
         "IMPATIENT", "EXCITED"]


def main(out: str, cell: int = 150, mode: str = "2d", which: str = "actions"):
    if which == "faces":
        return faces(out, cell, mode)
    rows = ROWS if which == "actions" else {cid: EMOTIONS for cid in ROWS}
    QApplication.instance() or QApplication([])
    cols = max(len(v) for v in rows.values())
    img = QImage(cell * cols, (cell + 18) * len(rows), QImage.Format_ARGB32)
    img.fill(QColor("#EEF1F6"))
    g = QPainter(img)
    g.setRenderHint(QPainter.Antialiasing)
    f = QFont()
    f.setPixelSize(11)
    g.setFont(f)
    for ri, (cid, names) in enumerate(rows.items()):
        r = make_renderer(BUILTIN[cid], mode)
        for ci, name in enumerate(names):
            an = AnimationController(seed=1, profile=get_profile(cid))
            an.set_rig(r.rig)
            an._next_blink = 1e9
            pl = ActionPlayer(cid)
            a = pl.play(name, 0.0)
            ad = resolve(name, cid)
            st = ad.state if ad is not None and ad.state else S.IDLE
            t_at = (0.55 if name in ("SHOUT", "SULK") else 0.45) * (ad.duration if ad and ad.duration else 1.0)
            p = None
            k = 0
            while k * 0.02 <= t_at:
                pl.update(k * 0.02)
                p = an.update(st, k * 0.02, 0.02, None, None, None, 1.0, pl.layers())
                k += 1
            y = ri * (cell + 18)
            rect = QRectF(ci * cell + 5, y + 18 + 5, cell - 10, cell - 10)
            g.fillRect(rect, QColor("#FFFFFF") if (ri + ci) % 2 else QColor("#F7F8FB"))
            r.paint(g, rect, p, 1)
            g.setPen(QColor("#333"))
            g.drawText(QRectF(ci * cell, y + 2, cell, 16), Qt.AlignCenter, f"{cid}: {name}")
    g.end()
    img.save(out)
    print("saved", out)


def faces(out: str, cell: int, mode: str):
    from focusbuddy.companion.attention import LookOutput, AttentionTarget
    from focusbuddy.companion.emotion import Emotion, EmotionController
    QApplication.instance() or QApplication([])
    cols = len(FACES)
    img = QImage(cell * cols, (cell + 18) * len(ROWS), QImage.Format_ARGB32)
    img.fill(QColor("#EEF1F6"))
    g = QPainter(img)
    g.setRenderHint(QPainter.Antialiasing)
    f = QFont()
    f.setPixelSize(11)
    g.setFont(f)
    look = LookOutput(AttentionTarget.USER, None, 0.0, 0.0, 0.0, 0.0, 0.6, False, 0.0, 0.0, 0.2, 0.0)
    for ri, cid in enumerate(ROWS):
        r = make_renderer(BUILTIN[cid], mode)
        for ci, name in enumerate(FACES):
            an = AnimationController(seed=1, profile=get_profile(cid))
            an.set_rig(r.rig)
            an._next_blink = 1e9
            ec = EmotionController()
            ec.routing = False
            ec.set(Emotion(name), 1.0, None, now=0.0)
            p = None
            for k in range(60):
                ec.update(0.02, k * 0.02)
                p = an.update(S.IDLE, k * 0.02, 0.02, None, ec.style(), look, 1.0, [])
            y = ri * (cell + 18)
            rect = QRectF(ci * cell + 5, y + 18 + 5, cell - 10, cell - 10)
            g.fillRect(rect, QColor("#FFFFFF") if (ri + ci) % 2 else QColor("#F7F8FB"))
            r.paint(g, rect, p, 1)
            g.setPen(QColor("#333"))
            g.drawText(QRectF(ci * cell, y + 2, cell, 16), Qt.AlignCenter, f"{cid}: {name.lower()}")
    g.end()
    img.save(out)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "action_sheet.png", 150,
         sys.argv[2] if len(sys.argv) > 2 else "2d", sys.argv[3] if len(sys.argv) > 3 else "actions")
