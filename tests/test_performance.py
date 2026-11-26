"""Device profile, performance modes and the live quality governor (companion/performance.py)."""

from focusbuddy.companion.performance import (DeviceProfile, QualityGovernor, TIERS, classify_gpu, plan,
                                              tier_index)


def test_gpu_classes():
    assert classify_gpu("NVIDIA GeForce RTX 3060/PCIe/SSE2") == "discrete"
    assert classify_gpu("AMD Radeon RX 6700 XT") == "discrete"
    assert classify_gpu("Intel(R) Iris(R) Xe Graphics") == "igpu"
    assert classify_gpu("Intel(R) UHD Graphics 630") == "igpu"
    assert classify_gpu("Intel(R) HD Graphics 620") == "igpu"
    assert classify_gpu("AMD Radeon(TM) Graphics") == "igpu"
    assert classify_gpu("Intel(R) HD Graphics 4000") == "old_igpu"
    assert classify_gpu("Intel(R) HD Graphics") == "old_igpu"
    assert classify_gpu("llvmpipe (LLVM 15.0.7, 256 bits)") == "software"
    assert classify_gpu("GDI Generic") == "software"
    assert classify_gpu("Microsoft Basic Render Driver") == "software"
    assert classify_gpu("") == "none"


def test_tier_index_accepts_aliases():
    assert tier_index("normal") == tier_index("medium") == 1
    assert tier_index("ultra") == 3
    assert tier_index("nonsense") == 1


def _dev(cls="mid", **kw):
    return DeviceProfile(device_class=cls, **kw)


def test_auto_mode_follows_the_device():
    assert plan("auto", "normal", _dev("low"))["start"] == "low"
    assert plan("auto", "normal", _dev("low"))["ceiling"] == "medium"
    p = plan("auto", "normal", _dev("mid"))
    assert (p["start"], p["ceiling"], p["adaptive"]) == ("medium", "high", True)
    p = plan("auto", "normal", _dev("high"))
    assert (p["start"], p["ceiling"]) == ("high", "ultra")
    # an explicit choice caps the ceiling
    assert plan("auto", "low", _dev("high"))["ceiling"] == "low"


def test_battery_rules():
    p = plan("auto", "normal", _dev("high", battery_saver=True))
    assert (p["start"], p["ceiling"], p["fps_cap"], p["particles"]) == ("low", "low", 30, False)
    assert plan("auto", "normal", _dev("high", on_battery=True))["ceiling"] == "medium"
    p = plan("battery", "ultra", _dev("high"))
    assert p["ceiling"] == "low" and p["fps_cap"] == 30 and not p["adaptive"]


def test_quality_and_balanced_modes():
    p = plan("quality", "ultra", _dev("low"))
    assert p["start"] == "ultra" and not p["adaptive"]
    assert plan("quality", "normal", _dev("mid"))["start"] == "high"
    p = plan("balanced", "ultra", _dev("high"))
    assert p["ceiling"] == "medium" and p["fps_cap"] == 60 and p["adaptive"]


def _feed(g, cost, budget, t0, secs, fps=60, **kw):
    out = []
    t = t0
    while t < t0 + secs:
        r = g.feed(cost, budget, t, **kw)
        if r:
            out.append((round(t, 2), r))
        t += 1.0 / fps
    return out, t


def test_governor_steps_down_after_sustained_overload():
    g = QualityGovernor()
    g.reset(0.0)
    steps, _ = _feed(g, 15.5, 16.7, 10.0, 3.0)      # p90 above 0.9 x budget, but only one window
    assert not steps
    steps, _ = _feed(g, 15.5, 16.7, 13.0, 3.0)
    assert steps and steps[0][1] == "down"


def test_governor_severe_overload_is_immediate():
    g = QualityGovernor()
    g.reset(0.0)
    steps, _ = _feed(g, 40.0, 16.7, 10.0, 2.2)
    assert steps and steps[0][1] == "down" and steps[0][0] <= 12.2


def test_governor_steps_up_only_after_a_long_calm():
    g = QualityGovernor()
    g.reset(0.0)
    steps, t = _feed(g, 3.0, 16.7, 10.0, 15.0)
    assert not steps                                 # calm, but not for long enough
    steps, _ = _feed(g, 3.0, 16.7, t, 10.0)
    assert steps and steps[0][1] == "up"


def test_governor_never_flaps():
    """An upgrade that immediately fails doubles the wait before the next try."""
    g = QualityGovernor()
    g.reset(0.0)
    steps, t = _feed(g, 3.0, 16.7, 10.0, 30.0)
    assert steps[0][1] == "up"
    g.changed("up", t)
    steps, t = _feed(g, 40.0, 16.7, t, 7.0)          # (no decision in the first COOLDOWN seconds)
    assert steps[0][1] == "down"
    g.changed("down", t)
    assert g.up_wait == QualityGovernor.UP_AFTER * 2


def test_governor_respects_limits():
    g = QualityGovernor()
    g.reset(0.0)
    steps, _ = _feed(g, 40.0, 16.7, 10.0, 6.0, can_down=False)
    assert not any(s == "down" for _t, s in steps)
    g.reset(0.0)
    steps, _ = _feed(g, 2.0, 16.7, 10.0, 60.0, can_up=False)
    assert not any(s == "up" for _t, s in steps)


def test_tiers_order():
    assert TIERS == ("low", "medium", "high", "ultra")
