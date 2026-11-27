"""Companion fixes: short greetings, the rabbit's real hop, the people's bigger head and natural hands."""

import math

import numpy as np
import pytest


def _sim(cid, mode="production", px=220):
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from focusbuddy.ui.playground import PlaygroundSim
    sim = PlaygroundSim(cid, char_px=px, mode=mode)
    sim.set_effects(False)
    return sim


def _render(sim):
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter
    img = QImage(260, 260, QImage.Format_ARGB32_Premultiplied)
    img.fill(QColor(0, 0, 0, 0))
    g = QPainter(img)
    sim.renderer.paint(g, QRectF(0, 0, 240, 240), sim.pose, 1)
    g.end()


# ------------------------------------------------------------------ greetings
@pytest.mark.parametrize("cid", ["cat", "study_girl", "rabbit"])
def test_greeting_wave_is_short_and_life_goes_on(cid):
    """Switching companions used to leave it waving forever (a plan without a release)."""
    from focusbuddy.companion.actions import Priority
    from focusbuddy.companion.states import CompanionState as S
    sim = _sim(cid, mode="2d")
    b = sim.b
    b.autonomous, b.mode = True, "auto"
    sim.set_life(True)
    for _ in range(60):
        sim.step(1 / 30)
    b._start_plan([("state", S.WAVING, 1.1)] + b.followup_steps(), mode="auto", priority=Priority.INTERACTION)
    waving = 0.0
    for _ in range(int(6 * 30)):
        sim.step(1 / 30)
        if b.state == S.WAVING:
            waving += 1 / 30
    assert waving < 1.4
    assert b.state != S.WAVING and b.mode == "auto"


def test_wave_action_is_quick():
    from focusbuddy.companion.actions import ACTIONS
    assert ACTIONS["WAVE"].duration <= 1.2


def test_people_do_not_wave_for_a_tail_wag():
    from focusbuddy.companion.actions import resolve
    assert resolve("TAIL_WAG", "buddy").name != "WAVE"


# ------------------------------------------------------------------ the rabbit hops
def test_hop_flight_only_when_no_paw_is_planted():
    from focusbuddy.companion.gait import HOP, hop_flight, hop_stance
    for g in HOP:
        for k in range(200):
            c = k / 200
            f, s = hop_flight(g, c), hop_stance(g, c)
            assert (f < 0) != (s < 0)                       # exactly one of ground / air
            planted = any(((c + off) % 1.0) < g.duty for off in g.offsets.values())
            if f > 0.02:
                assert not planted, (g.name, c)             # in flight no paw touches the ground


def test_rabbit_legs_really_move():
    from focusbuddy.companion.character_asset import LIMB_BONES
    sim = _sim("rabbit")
    d = sim.renderer.driver
    sim.trigger("Walk")
    xs = {k: [] for k in LIMB_BONES[d.kind]}
    knees = {k: [] for k in LIMB_BONES[d.kind]}
    for i in range(150):
        sim.step(1 / 60)
        _render(sim)
        if i < 30:
            continue
        for k, (u, lo, e) in LIMB_BONES[d.kind].items():
            iu, il, ie = d.canon[u], d.canon[lo], d.canon[e]
            xs[k].append(float(d.world_T[ie][0] - d.world_T[iu][0]))
            a, b_ = d.world_T[il] - d.world_T[iu], d.world_T[ie] - d.world_T[il]
            knees[k].append(math.degrees(math.acos(np.clip(a @ b_ / np.linalg.norm(a) / np.linalg.norm(b_), -1, 1))))
    for k in ("BL", "BR"):
        assert np.ptp(xs[k]) > 12.0                         # was 8.5: paws shuffling in place
        assert min(knees[k]) > 20.0                         # hind legs fold, never lock straight


# ------------------------------------------------------------------ the people's bigger head
def test_head_grow_is_baked_rigidly(monkeypatch):
    """Head and its joints 1.5x about the Head joint; shoulders, collar and long hair untouched."""
    from focusbuddy.companion.character_asset import CharacterAsset, asset_dir
    folder = asset_dir("study_girl")
    grown = CharacterAsset.from_folder("study_girl", folder)
    assert getattr(grown, "head_grown", 1.0) == pytest.approx(1.5)
    monkeypatch.setattr(CharacterAsset, "_grow_head", lambda self, k: None)
    plain = CharacterAsset.from_folder("study_girl", folder)
    hi = plain.joint_index("Head")
    H = plain.joints[hi].bind_world[:3, 3]
    assert np.allclose(grown.joints[hi].bind_world, plain.joints[hi].bind_world)    # the pivot stays
    for i, (jg, jp) in enumerate(zip(grown.joints, plain.joints)):
        if i != hi and plain._descends(i, hi):
            assert np.allclose(jg.bind_world[:3, 3], H + 1.5 * (jp.bind_world[:3, 3] - H), atol=1e-6)
            assert np.allclose(jg.inverse_bind @ jg.bind_world, np.eye(4), atol=1e-6)
            if jp.rest_t is not None:
                assert np.allclose(jg.rest_t, 1.5 * np.asarray(jp.rest_t), atol=1e-9)
        elif not plain._descends(i, hi):
            assert np.allclose(jg.bind_world, jp.bind_world)                       # body untouched
    for name, mp in plain.lods.items():
        mg = grown.lods[name]
        w = plain.head_grow_weight(mp.positions)
        body = w == 0.0
        assert body.sum() > len(w) * 0.5
        assert np.allclose(mg.positions[body], mp.positions[body])
        head = w == 1.0
        assert head.sum() > 100
        assert np.allclose(mg.positions[head], H + 1.5 * (mp.positions[head] - H), atol=1e-6)


def test_head_grow_weight_profile():
    from focusbuddy.companion.characters import get_character
    from focusbuddy.companion.renderer import load_asset
    from focusbuddy.companion.rig import Rig, load_skeleton
    spec = get_character("buddy")
    a = load_asset(spec, Rig(load_skeleton(spec.id, spec.species, "biped")))
    H = a.joints[a.joint_index("Head")].bind_world[:3, 3]
    N = a.joints[a.joint_index("Neck")].bind_world[:3, 3]
    assert a.head_grow_weight(np.array([H]))[0] == pytest.approx(1.0)
    assert a.head_grow_weight(np.array([N]))[0] == pytest.approx(0.0)
    below = N + (N - H) * 2.0                               # chest / long hair level
    assert a.head_grow_weight(np.array([below]))[0] == 0.0
    # monotonic up the neck: no bulge, no fold
    pts = np.array([N + (H - N) * t for t in np.linspace(0, 1, 30)])
    w = a.head_grow_weight(pts)
    assert (np.diff(w) >= -1e-9).all()


# ------------------------------------------------------------------ the people's hands
def test_people_wave_with_a_bent_elbow_beside_the_head():
    sim = _sim("study_girl", px=260)
    d = sim.renderer.driver
    for _ in range(40):
        sim.step(1 / 60)
        _render(sim)
    sim.trigger("Wave")
    bends, highs = [], []
    for i in range(int(1.0 * 60)):
        sim.step(1 / 60)
        _render(sim)
        arm = getattr(d, "_hwave_arm", "")
        if i > 30 and arm:
            side = "Right" if arm == "arm_R" else "Left"
            sh, el, ha = (d.world_T[d.canon[f"{side}{b}"]] for b in ("UpperArm", "LowerArm", "Hand"))
            a, b_ = el - sh, ha - el
            bends.append(math.degrees(math.acos(np.clip(a @ b_ / np.linalg.norm(a) / np.linalg.norm(b_), -1, 1))))
            highs.append(float(sh[1] - ha[1]))               # +y is down: hand above the shoulder > 0
    assert bends, "the wave started"
    assert min(bends) > 35.0                                # never a straight "zombie" arm
    assert np.median(highs) > 0.0                           # the hand is up, beside the head
