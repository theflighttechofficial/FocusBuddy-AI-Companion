"""Shared test setup: a private data folder, the project on the path, one Qt application."""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FOCUSBUDDY_DATA_DIR", tempfile.mkdtemp(prefix="fb-tests-"))
os.environ.setdefault("FOCUSBUDDY_FAKE_HOUR", "14")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def gl_available() -> bool:
    try:
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            QApplication([])
        from focusbuddy.companion.render3d_gl import GLSkinRaster
        return GLSkinRaster.create() is not None
    except Exception:  # noqa: BLE001
        return False
