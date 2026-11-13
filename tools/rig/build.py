"""Rig one companion: skeleton, straightened head, bone-heat weights, clean-up; writes
build/rig/<id>_rig.npz (V, W, names) and <id>_joints.json, plus check views.

    python tools/rig/build.py fox
"""

from __future__ import annotations

import json
import sys

import numpy as np

import skeleton as S
import weights as Wt
from core import BUILD, Model
from joints import parents_for


def rig(cid: str, views: bool = True) -> dict:
    m = Model(cid)
    lm = S.LANDMARKS[cid]
    if m.cfg["kind"] == "quadruped":
        J = S.quadruped_skeleton(m, lm)
        yaw, roll, plane = lm.get("head_turn") or S.head_turn(m.V, J)
        m.V = S.straighten_lbs(m, J, yaw, roll, plane, cid)
        J.update(S.ears_and_top(m.V, J, lm.get("ears", True)))
        J.update(S.tail_chain(m, J, lm["tail_base"], lm.get("tail_n", 4), lm.get("tail_tip")))
    else:
        J = S.biped_skeleton(m, lm)
        yaw = roll = 0.0
        J.update(S.ears_and_top(m.V, J, False))
    P = parents_for(J)
    names, W = Wt.bone_heat(cid, m.V, m.F, J, P)
    W = Wt.clean(m.V, m.F, W, names, J, m.cfg["kind"], lm, m.vcol)
    # a few smoothing passes: the clean-up gates leave hard edges between bones
    W = Wt.smooth_weights(m.V, m.F, W, 4 if m.cfg["kind"] == "biped" else 3)
    np.savez_compressed(BUILD / f"{cid}_rig.npz", V=m.V, W=W, names=np.array(names))
    (BUILD / f"{cid}_joints.json").write_text(json.dumps(
        {"joints": {k: [float(x) for x in v] for k, v in J.items()}, "parents": P, "head_turn": [yaw, roll]},
        indent=1), encoding="utf-8")
    if views:
        from PIL import Image
        from grid import render
        from wview import views as wv
        ims = [render(m, v, 800, None, J, P) for v in ("side_left", "front", "top")]
        x = Image.new("RGB", (2400, 800))
        for k, i in enumerate(ims):
            x.paste(i, (800 * k, 0))
        x.save(BUILD / f"check_{cid}_skeleton.png")
        wv(m.V, m.F, W, names, BUILD / f"check_{cid}_weights.png")
    print(f"{cid}: {len(J)} joints, head turn {yaw:.1f} / {roll:.1f}")
    return J


if __name__ == "__main__":
    for c in sys.argv[1:]:
        rig(c)
