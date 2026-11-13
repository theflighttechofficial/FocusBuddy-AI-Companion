"""Render every character in every state into one PNG (visual QA).

    python tools/contact_sheet.py out.png
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from focusbuddy.companion.animation import Animator  # noqa: E402
from focusbuddy.companion.characters import all_characters  # noqa: E402
from focusbuddy.companion.renderer import make_renderer  # noqa: E402
from focusbuddy.companion.profile import get_profile  # noqa: E402
from focusbuddy.companion.states import CompanionState as S  # noqa: E402

STATES = [S.IDLE, S.WALKING, S.RUNNING, S.SITTING, S.SLEEPING, S.LOOKING_AROUND, S.STRETCHING,
          S.WAVING, S.WATCHING, S.ALERT, S.WARNING, S.CLOSING_DISTRACTION, S.CELEBRATING,
          S.CONFUSED, S.FOCUS_MODE, S.HELD, S.PRESSING, S.YAWNING, S.GROOMING, S.SCRATCHING, S.PEEKING,
          S.READING, S.PETTED, S.WAKING]
TIMES = {S.PRESSING: 0.5, S.YAWNING: 0.85, S.WAKING: 0.5}


def main(out: str, cell: int = 150, t: float = 0.55):
    QApplication.instance() or QApplication([])
    chars = all_characters()
    img = QImage(cell * len(STATES), cell * len(chars) + 24, QImage.Format_ARGB32)
    img.fill(QColor("#EEF1F6"))
    g = QPainter(img)
    g.setRenderHint(QPainter.Antialiasing)
    f = QFont()
    f.setPixelSize(11)
    g.setFont(f)
    for ci, st in enumerate(STATES):
        g.setPen(QColor("#333"))
        g.drawText(QRectF(ci * cell, 2, cell, 20), Qt.AlignCenter, st.value)
    for ri, spec in enumerate(chars):
        r = make_renderer(spec)
        for ci, st in enumerate(STATES):
            an = Animator(seed=1, profile=get_profile(spec.id))
            if hasattr(r, "rig"):
                an.set_rig(r.rig)
            an._next_blink = 1e9
            tt = TIMES.get(st, t)
            pose = an.pose(st, 100.0, 0.016, 60, 75)
            for k in range(1, int(tt / 0.016) + 1):
                pose = an.pose(st, 100.0 + k * 0.016, 0.016, 60, 75)
            rect = QRectF(ci * cell + 5, 24 + ri * cell + 5, cell - 10, cell - 10)
            g.fillRect(rect, QColor("#FFFFFF") if (ri + ci) % 2 else QColor("#F7F8FB"))
            r.paint(g, rect, pose, 1)
    g.end()
    img.save(out)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "contact_sheet.png")
