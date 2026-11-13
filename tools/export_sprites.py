"""Export a procedural character as a sprite pack (template for custom art).

    python tools/export_sprites.py fox assets/characters/pixel-fox

Produces PNG frames + manifest.json in the sprite-pack format, so artists
(or an image generator) can replace the frames with their own drawings.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from focusbuddy.companion.animation import Animator  # noqa: E402
from focusbuddy.companion.characters import BUILTIN  # noqa: E402
from focusbuddy.companion.renderer import make_renderer  # noqa: E402
from focusbuddy.companion.states import CompanionState as S  # noqa: E402

ANIMS = {"IDLE": (S.IDLE, 4, 4), "WALKING": (S.WALKING, 8, 10), "RUNNING": (S.RUNNING, 8, 14),
         "SITTING": (S.SITTING, 4, 3), "SLEEPING": (S.SLEEPING, 4, 2), "ALERT": (S.ALERT, 4, 8),
         "WARNING": (S.WARNING, 4, 6), "CELEBRATING": (S.CELEBRATING, 8, 12), "CONFUSED": (S.CONFUSED, 4, 4)}


def export(char_id: str, out: Path, size: int = 160) -> Path:
    QApplication.instance() or QApplication([])
    base = BUILTIN[char_id]
    out.mkdir(parents=True, exist_ok=True)
    r = make_renderer(base)
    manifest = {"id": out.name, "name": f"{base.display_name} (sprites)", "default_name": base.default_name,
                "emoji": base.emoji, "personality": base.personality, "faces": "right",
                "tagline": "Sprite-pack version - replace the PNGs with your own art.", "animations": {}}
    for name, (state, frames, fps) in ANIMS.items():
        files = []
        an = Animator(seed=1)
        an._next_blink = 1e12
        t0 = 100.0
        an.pose(state, t0, 0.0, 60, 75)
        for i in range(frames):
            t = t0 + 0.3 + i / fps
            pose = an.pose(state, t, 1 / fps, 60, 75)
            pose.mark = ""  # effects are drawn by the app on top
            img = QImage(size, size, QImage.Format_ARGB32)
            img.fill(Qt.transparent)
            g = QPainter(img)
            r.paint(g, QRectF(0, 0, size, size), pose, 1)
            g.end()
            fn = f"{name.lower()}_{i}.png"
            img.save(str(out / fn))
            files.append(fn)
        manifest["animations"][name] = {"frames": files, "fps": fps}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "fox"
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("assets/characters") / f"{cid}-sprites"
    print("exported to", export(cid, dest))
