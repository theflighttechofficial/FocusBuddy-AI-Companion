"""The running companion: renderer reuse, the governor, frame-cost clipping, pop-free skeletons,
the lazy dashboard and background previews."""

import time

import numpy as np
import pytest

from conftest import gl_available


@pytest.fixture
def companion(qapp):
    from focusbuddy.companion.character import Companion
    from focusbuddy.companion.characters import get_character
    c = Companion(get_character("cat"), seed=3)
    yield c
    c.stop()
    c.window.close()
    c.bubble.close()


def test_same_settings_keep_the_renderer(companion):
    c = companion
    c.set_performance("auto")
    c.set_render_mode("production", "normal")
    r = c.renderer
    c.set_render_mode("production", "normal")       # any other setting change re-applies these
    assert c.renderer is r
    c._apply_tier("low")                            # the governor lowered it
    c.set_render_mode("production", "normal")
    assert c.renderer is r and c.render_quality == "low"


def test_performance_modes_reach_the_renderer(companion):
    c = companion
    c.set_render_mode("production", "normal")
    c.set_performance("battery")
    assert c.render_quality == "low" and c.render_fps <= 30 and not c.particles.enabled
    c.set_performance("quality")
    assert c.render_quality == c.perf["ceiling"] and not c.perf["adaptive"]
    c.set_performance("auto")
    assert c.device is not None and c.perf["mode"] == "auto"
    assert c.render_quality == c.perf["start"]


def test_one_slow_frame_does_not_throttle(companion):
    c = companion
    c._reset_paint_cost()
    c._note_paint_cost(0.9)                         # first frame uploads meshes: 900 ms
    assert c._paint_cost <= 0.02
    for _ in range(30):
        c._note_paint_cost(0.006)
    c._note_paint_cost(0.5)                         # a hitch later on
    assert c._paint_cost < 0.012


def test_governor_lowers_quality_live(companion):
    c = companion
    c.set_performance("auto")
    c.set_render_mode("production", "high")
    if getattr(c.renderer, "backend", None) is None:
        pytest.skip("no 3D renderer")
    c.perf["ceiling"] = "high"
    c._apply_tier("high")
    c.governor.reset(0.0)
    t = time.perf_counter()
    for k in range(400):                            # frames far over budget
        c._budget_check(60.0, now=t + k / 60)
    assert c.render_quality in ("medium", "low")


def test_companion_runs_and_presents(companion, qapp):
    from PySide6.QtCore import QTimer
    c = companion
    c.set_performance("auto")
    c.set_render_mode("production", "normal")
    n = {"present": 0}
    orig = c._present

    def present():
        n["present"] += 1
        orig()
    c._present = present
    c.start()
    c.behavior.play_sequence([("WALK", 2.0)], loop=True)
    QTimer.singleShot(1500, qapp.quit)
    qapp.exec()
    c.stop()
    assert n["present"] >= 20                       # ~40-90 depending on the machine; never stuck
    assert c._tick_errors == 0


def test_skeleton_never_teleports():
    """The living sequence on the cat and a person: no free bone (ears, tail, head, a person's
    arms) jumps more than a few units from one frame to the next."""
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from focusbuddy.companion.character_asset import LIMB_BONES
    from focusbuddy.ui.playground import PlaygroundSim
    for cid in ("cat", "study_girl"):
        sim = PlaygroundSim(cid, char_px=200, mode="production")
        sim.set_effects(False)
        r = sim.renderer
        drv = getattr(r, "driver", None)
        if drv is None:
            pytest.skip("no production renderer")
        limb = set()
        for nm, bones in LIMB_BONES[drv.kind].items():
            if drv.kind == "biped" and nm.startswith("arm"):
                continue
            for b in bones:
                i = drv.canon.get(b, -1)
                if i >= 0:
                    limb.update(drv._subtree(i))
        free = [j for j in range(drv.n) if j not in limb]
        img = QImage(240, 240, QImage.Format_ARGB32_Premultiplied)
        sim.run_validation()
        P = []
        for _ in range(int(32 * 60)):
            sim.step(1 / 60)
            img.fill(QColor(0, 0, 0, 0))
            g = QPainter(img)
            r.paint(g, QRectF(0, 0, 240, 240), sim.pose, 1)
            g.end()
            P.append(drv.world_T[free].copy())
        P = np.stack(P)
        acc = np.linalg.norm(np.diff(P, 2, axis=0), axis=2).max(axis=1)
        # before 2.4: 35-50 units (an ear / tail tip / hand teleporting); now a fast motion at most
        assert acc.max() < 22.0, (cid, float(acc.max()))


def test_lazy_dashboard_pages(qapp):
    from PySide6.QtWidgets import QWidget

    from focusbuddy.ui.dashboard import _PageSlot
    built = []

    class P(QWidget):
        def refresh(self):
            built.append("refresh")

        def send(self, text):
            return text.upper()
    slot = _PageSlot(lambda: built.append("built") or P())
    assert not slot.built and not built
    slot.tick()                                     # not built yet: nothing happens
    assert not built
    assert slot.send("hi") == "HI"                  # first real use builds it
    assert built == ["built"] and slot.built
    slot.refresh()
    assert built == ["built", "refresh"]


@pytest.mark.skipif(not gl_available(), reason="needs OpenGL")
def test_background_preview_build(qapp):
    from PySide6.QtCore import QTimer

    from focusbuddy.companion.characters import get_character
    from focusbuddy.ui.preview_loader import PreviewLoader
    got = []
    PreviewLoader.get().request(get_character("dog"), "production", "normal", got.append)
    deadline = time.perf_counter() + 30
    while not got and time.perf_counter() < deadline:
        QTimer.singleShot(50, lambda: None)
        qapp.processEvents()
        time.sleep(0.01)
    assert got and got[0] is not None and getattr(got[0], "mode", "") == "production"
