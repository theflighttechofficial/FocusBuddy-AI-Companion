"""Record gen-2 animation showcase tiles with the real engine (deterministic, headless).

    python tools/record_playground.py OUT_DIR [2d|3d] [gaits|emotions|living]

``living`` runs the living-character validation sequence (sleep ... rest) on every character.
    ffmpeg -framerate 25 -i OUT_DIR/grid_%04d.png -vf "split[a][b];[a]palettegen[p];[b][p]paletteuse" gaits.gif
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

# (character, sequence, loop, caption, skeleton overlay)
TILES = [
    ("cat", [("WALK", 3.0), ("TURN", None)], True, "Walk - planted feet (IK)", True),
    ("dog", [("JOG", 2.5), ("TURN", None)], True, "Jog - trot", False),
    ("fox", [("RUN", 2.0), ("TURN", None)], True, "Run - gallop", False),
    ("cat", [("SPRINT", 1.6), ("TURN", None)], True, "Sprint", False),
    ("rabbit", [("DOUBLE_HOP", None), ("NOSE_TWITCH", None), ("BOUNCE", None), ("IDLE", 0.6)], True,
     "Double hop + binky", False),
    ("robot", [("MECHANICAL_WALK", 2.5), ("SCAN", None), ("TURN", None)], True, "Mechanical walk + scan", True),
    ("buddy", [("RUN", 2.0), ("JUMP", None), ("THUMBS_UP", None), ("TURN", None)], True, "Run, jump, thumbs up", False),
    ("dog", [("PLAY_BOW", None), ("JUMP", None), ("TAIL_WAG", None), ("SHAKE", None)], True, "Play bow, jump, shake",
     False),
]


EMOTION_TILES = [
    ("cat", [("ANGRY", None), ("HUFF", None), ("GRUMPY", None)], True, "Angry stomp -> grumpy", False),
    ("dog", [("SHOUT", None), ("IDLE", 0.6), ("LAUGH", None)], True, "Shout \"HEY!\" -> laugh", False),
    ("rabbit", [("SCARED", None), ("EMBARRASSED", None), ("IDLE", 0.5)], True, "Scared -> embarrassed", False),
    ("fox", [("CRY", None), ("RELIEVED", None), ("LOVE", None)], True, "Cry -> relieved -> love", False),
    ("robot", [("ERROR", None), ("DIZZY", None), ("SHOCKED", None)], True, "Error -> dizzy -> shocked", False),
    ("buddy", [("TANTRUM", None), ("SULK", None), ("SMUG", None)], True, "Tantrum -> sulk -> smug", False),
    ("cat", [("SPRINT", 1.4), ("TURN", None), ("JUMP", None), ("DETERMINED", None)], True, "Sprint, jump, determined",
     False),
    ("dog", [("BORED", None), ("DISGUSTED", None), ("PROUD", None)], True, "Bored -> disgusted -> proud", False),
]


def main(out: str, seconds: float = 8.0, fps: int = 25, w: int = 320, h: int = 240, mode: str = "2d",
         which: str = "gaits"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    QApplication.instance() or QApplication([])
    from focusbuddy.ui.playground import PlaygroundCanvas, PlaygroundSim
    tiles = []
    if which == "living":
        return living(out, seconds, fps, w, h, mode)
    for cid, seq, loop, cap, skel in (TILES if which == "gaits" else EMOTION_TILES):
        sim = PlaygroundSim(cid, char_px=150.0, seed=3, mode=mode)
        sim.b.brain.surprises = False
        sim.play_sequence(seq, loop)
        cv = PlaygroundCanvas(sim)
        cv.setMinimumSize(10, 10)
        cv.resize(w, h)
        cv.debug = {"skeleton": skel, "ik": skel, "contacts": skel}
        cv.show_hint = False
        tiles.append((sim, cv, cap))
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    for f in range(int(seconds * fps)):
        img = QImage(w * cols, h * rows, QImage.Format_RGB32)
        img.fill(QColor("#FFFFFF"))
        g = QPainter(img)
        for i, (sim, cv, cap) in enumerate(tiles):
            for _ in range(2):
                sim.step(1 / (fps * 2))
            cv.cam_x += (sim.b.x - cv.cam_x) * 0.25
            cv.lines = []
            x, y = (i % cols) * w, (i // cols) * h
            g.drawPixmap(x, y, cv.grab())
            g.setPen(QColor("#1D2433"))
            font = QFont()
            font.setPixelSize(14)
            font.setBold(True)
            g.setFont(font)
            g.drawText(QPointF(x + 12, y + 22), cap)
            g.setPen(QColor("#E4E8F1"))
            g.drawRect(x, y, w - 1, h - 1)
        g.end()
        img.save(str(out / f"grid_{f:04d}.png"))
    print("frames:", int(seconds * fps))


def living(out: Path, seconds: float, fps: int, w: int, h: int, mode: str) -> None:
    from focusbuddy.ui.playground import PlaygroundCanvas, PlaygroundSim
    tiles = []
    for cid in ("cat", "dog", "fox", "rabbit", "robot", "buddy"):
        sim = PlaygroundSim(cid, char_px=150.0, seed=3, mode=mode)
        sim.run_validation()
        cv = PlaygroundCanvas(sim)
        cv.setMinimumSize(10, 10)
        cv.resize(w, h)
        cv.show_hint = False
        tiles.append((sim, cv, cid))
    cols = 3
    rows = 2
    for f in range(int(seconds * fps)):
        img = QImage(w * cols, h * rows, QImage.Format_RGB32)
        img.fill(QColor("#FFFFFF"))
        g = QPainter(img)
        for i, (sim, cv, cid) in enumerate(tiles):
            for _ in range(2):
                sim.step(1 / (fps * 2))
            cv.cam_x += (sim.b.x - cv.cam_x) * 0.25
            cv.lines = []
            x, y = (i % cols) * w, (i // cols) * h
            g.drawPixmap(x, y, cv.grab())
            beat = sim.validation[-1][0] if sim.validation else ""
            g.setPen(QColor("#1D2433"))
            font = QFont()
            font.setPixelSize(14)
            font.setBold(True)
            g.setFont(font)
            g.drawText(QPointF(x + 12, y + 22), f"{cid}: {beat.lower()}")
            g.setPen(QColor("#E4E8F1"))
            g.drawRect(x, y, w - 1, h - 1)
        g.end()
        img.save(str(out / f"grid_{f:04d}.png"))
    print("frames:", int(seconds * fps))


if __name__ == "__main__":
    which_ = sys.argv[3] if len(sys.argv) > 3 else "gaits"
    main(sys.argv[1] if len(sys.argv) > 1 else "pg_frames", 31.0 if which_ == "living" else 9.0,
         12 if which_ == "living" else 25, 320, 240,
         sys.argv[2] if len(sys.argv) > 2 else "2d", sys.argv[3] if len(sys.argv) > 3 else "gaits")
