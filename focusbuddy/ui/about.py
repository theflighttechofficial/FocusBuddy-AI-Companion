"""About FocusBuddy (F1): version, credits, privacy, data folder."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from .. import __version__
from .icons import avatar_pixmap
from .motion import AuroraBackground, GlassCard, fade_in
from .theme import fluent_window

CREDITS = (
    "Companion art: the FocusBuddy character sheets (cat, dog, fox, rabbit, robot, Study Buddy, "
    "Study Girl, Saree Girl).\n"
    "3D models: cat, dog, fox, rabbit, Study Buddy, Study Girl and Saree Girl generated from that art "
    "with Hunyuan3D (Tencent, Tencent Hunyuan 3D Community License); the robot reconstructed with "
    "TripoSR (Stability AI & Tripo AI, MIT licence). Rigged, given faces and animated by FocusBuddy.\n"
    "Motion capture: the data used in this project was obtained from mocap.cs.cmu.edu. The database "
    "was created with funding from NSF EIA-0196217 (BVH conversion by Bruce Hahne).\n"
    "Built with Python, Qt for Python (PySide6), NumPy."
)


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About FocusBuddy")
        self.resize(620, 560)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        bg = AuroraBackground()
        outer.addWidget(bg)
        lay = QVBoxLayout(bg)
        lay.setContentsMargins(24, 24, 24, 24)
        card = GlassCard(hover=False, strong=True, radius=28)
        cl = QVBoxLayout(card)
        m = GlassCard.MARGIN
        cl.setContentsMargins(m + 28, m + 24, m + 28, m + 22)
        cl.setSpacing(8)
        top = QHBoxLayout()
        pic = QLabel()
        pic.setPixmap(avatar_pixmap("cat", "FB", 72))
        top.addWidget(pic)
        col = QVBoxLayout()
        col.setSpacing(2)
        t = QLabel("FocusBuddy")
        t.setObjectName("H1")
        col.addWidget(t)
        v = QLabel(f"Version {__version__}  ·  AI Distraction Helper")
        v.setObjectName("Muted")
        col.addWidget(v)
        top.addLayout(col, 1)
        cl.addLayout(top)
        cl.addSpacing(8)
        for head, body in (("Privacy", "Everything FocusBuddy learns about your day stays on this PC. There is no "
                                        "account server, no upload and no tracking. Profile PINs lock the window; "
                                        "they do not encrypt your files."),
                           ("Credits", CREDITS)):
            h = QLabel(head)
            h.setObjectName("H3")
            cl.addWidget(h)
            b = QLabel(body)
            b.setObjectName("Muted")
            b.setWordWrap(True)
            cl.addWidget(b)
            cl.addSpacing(4)
        cl.addStretch(1)
        row = QHBoxLayout()
        data = QPushButton("Open data folder")
        data.clicked.connect(self._open_data)
        row.addWidget(data)
        row.addStretch(1)
        ok = QPushButton("Close")
        ok.setObjectName("Primary")
        ok.setMinimumWidth(120)
        ok.clicked.connect(self.accept)
        row.addWidget(ok)
        cl.addLayout(row)
        lay.addWidget(card)

    def showEvent(self, ev):
        super().showEvent(ev)
        fluent_window(self)
        fade_in(self)

    def _open_data(self):
        from ..config import data_dir
        path = str(data_dir())
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606 - opens Explorer on the app's own data folder
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
