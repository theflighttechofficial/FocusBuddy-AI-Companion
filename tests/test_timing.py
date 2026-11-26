"""Frame timing: a high-resolution clock everywhere, display-synchronised pulses, focus timer."""

import re
import sys
import time
from pathlib import Path

import pytest

from focusbuddy.core.focus_manager import FocusTimer, Phase
from focusbuddy.utils.clock import FakeClock

ROOT = Path(__file__).resolve().parent.parent


def test_no_coarse_monotonic_clock():
    """time.monotonic() ticks every 15.6 ms on Windows: animation dt jumped between 0 and 31 ms
    (judder). Everything frame-related uses time.perf_counter()."""
    offenders = []
    for p in (ROOT / "focusbuddy").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        if re.search(r"\b(time|_time|_t)\.monotonic\s*\(", text) or re.search(r"\bmonotonic\s+as\b", text):
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, offenders


def test_perf_counter_is_fine_grained():
    a = time.perf_counter()
    b = a
    while b == a:
        b = time.perf_counter()
    assert b - a < 0.001


@pytest.mark.skipif(not sys.platform.startswith("win"), reason="compositor clock is Windows-only")
def test_vsync_pulses(qapp):
    from PySide6.QtCore import QTimer

    from focusbuddy.companion.pacing import VsyncPacer
    p = VsyncPacer()
    if not p.available:
        pytest.skip("no compositor")
    got = []

    def on_pulse():
        got.append(time.perf_counter())
        p.consumed()
    p.pulse.connect(on_pulse)
    assert p.start()
    QTimer.singleShot(600, qapp.quit)
    qapp.exec()
    p.stop()
    if not p.healthy:
        pytest.skip("compositor does not pace here (remote session?)")
    assert len(got) >= 15                        # ~36 at 60 Hz; never zero
    gaps = sorted(b - a for a, b in zip(got, got[1:]))
    assert gaps[len(gaps) // 2] < 0.04


def test_focus_timer_cycle():
    c = FakeClock()
    t = FocusTimer(c)
    assert t.start(25)
    c.advance(600)
    assert t.pause()
    c.advance(300)                               # paused time does not count
    assert abs(t.remaining() - 900) < 1e-6
    assert t.resume()
    c.advance(900)
    ev = [e.kind for e in t.tick()]
    assert "focus_completed" in ev and t.phase == Phase.IDLE
    assert t.start_break(5)
    c.advance(301)
    kinds = [e.kind for e in t.tick()]
    assert kinds[-1] == "break_completed" and t.phase == Phase.IDLE


def test_focus_timer_survives_clock_going_back():
    c = FakeClock()
    t = FocusTimer(c)
    t.start(25)
    c.advance(-120)                              # the user set the clock back
    assert 0.0 <= t.elapsed() <= t.planned
    assert t.remaining() <= t.planned
