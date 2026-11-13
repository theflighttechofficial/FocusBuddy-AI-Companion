"""Generate app + browser-extension icons from the procedural cat.

    python tools/make_icons.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from focusbuddy.companion.characters import BUILTIN  # noqa: E402
from focusbuddy.ui.icons import head_pixmap  # noqa: E402


def main():
    QApplication.instance() or QApplication([])
    spec = BUILTIN["cat"]
    ext = ROOT / "browser_extension" / "icons"
    ext.mkdir(parents=True, exist_ok=True)
    for s in (16, 32, 48, 128):
        head_pixmap(spec, s).save(str(ext / f"icon{s}.png"))
    icons = ROOT / "assets" / "icons"
    icons.mkdir(parents=True, exist_ok=True)
    head_pixmap(spec, 256).save(str(icons / "focusbuddy.png"))
    if True:
        # multi-size ICO (PNG-compressed entries, Windows Vista+) - small and crisp at every size
        import struct
        from PySide6.QtCore import QBuffer, QByteArray, QIODevice
        blobs = []
        for s in (16, 24, 32, 48, 64, 128, 256):
            ba = QByteArray()
            buf = QBuffer(ba)
            buf.open(QIODevice.WriteOnly)
            head_pixmap(spec, s).save(buf, "PNG")
            blobs.append((s, bytes(ba)))
        header = struct.pack("<HHH", 0, 1, len(blobs))
        offset = 6 + 16 * len(blobs)
        entries, data = b"", b""
        for s, blob in blobs:
            entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(blob), offset + len(data))
            data += blob
        (icons / "focusbuddy.ico").write_bytes(header + entries + data)
    print("icons written")


if __name__ == "__main__":
    main()
