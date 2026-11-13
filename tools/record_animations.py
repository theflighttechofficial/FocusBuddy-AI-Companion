"""Record animation showcase frames with the real engine (deterministic, headless).

    python tools/record_animations.py OUT_DIR
    ffmpeg -framerate 25 -i OUT_DIR/grid_%04d.png showcase.gif
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

TILES = [("cat", "Walk", "Walk cycle"), ("dog", "Run", "Run / gallop"), ("fox", "Turn", "Brake, turn, go"),
         ("cat", "Wake", "Sleep → wake → stretch"), ("robot", "Press close button", "Presses the X"),
         ("rabbit", "Jump", "Hop & land"), ("buddy", "Read", "Idle: reading"), ("dog", "Celebrate", "Celebrate")]


def main(out: str, seconds: float = 6.0, fps: int = 25):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    QApplication.instance() or QApplication([])
    from focusbuddy.ui.studio import ANIMATIONS, AnimationStudio
    studios = []
    for cid, anim, _ in TILES:
        st = AnimationStudio(cid)
        st.timer.stop()
        st.canvas.CHAR = 150.0
        st.canvas.resize(300, 230)
        st.canvas.setFixedSize(300, 230)
        st.list.setCurrentRow(ANIMATIONS.index(anim))
        st._restart()
        st.mover.scale = 150 / 124
        studios.append(st)
    cols = 4
    w, h = 300, 230
    for f in range(int(seconds * fps)):
        img = QImage(w * cols, h * ((len(TILES) + cols - 1) // cols), QImage.Format_RGB32)
        img.fill(QColor("#FFFFFF"))
        g = QPainter(img)
        for i, st in enumerate(studios):
            st.step(1 / fps)
            st.canvas.info = []
            tile = st.canvas.grab()
            x, y = (i % cols) * w, (i // cols) * h
            g.drawPixmap(x, y, tile)
            g.setPen(QColor("#1D2433"))
            font = QFont()
            font.setPixelSize(14)
            font.setBold(True)
            g.setFont(font)
            g.drawText(QPointF(x + 12, y + 22), TILES[i][2])
            g.setPen(QColor("#E4E8F1"))
            g.drawRect(x, y, w - 1, h - 1)
        g.end()
        img.save(str(out / f"grid_{f:04d}.png"))
    print("frames:", int(seconds * fps))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "anim_frames")
