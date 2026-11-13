"""Build the *pipeline test asset* ``assets/characters/pipeline_test`` - a procedural asset that
exercises every feature of the production character pipeline in one file:

* textured mesh: baseColor (checker + gradient), normal map (bumped tiles), metallicRoughness
  (roughness stripes, a metallic belt), occlusion (darkened seams) and emissive (glowing visor)
  textures, all embedded PNGs with TEXCOORD_0 and TANGENT attributes
* PBR material factors (metallic / roughness / emissive)
* skeleton + skin weights (4 influences), inverse bind matrices
* animation clips: ``Idle`` (breathing, head nod, blink via a morph ``weights`` channel),
  ``Walk`` (leg / arm swing, pelvis bob and *root translation* - root motion) and ``Wave`` (upper body)
* facial morph targets with ARKit names: jawOpen, eyeBlinkLeft, eyeBlinkRight, mouthSmile,
  browDownLeft, cheekPuff

It is NOT production art (it is generated from the procedural demo buddy), and its metadata says
so (``"production": false``, ``"selectable": false`` - it stays out of the character list). The
artist-made references are ``fox_sample`` and ``robot_expressive``.

    python tools/build_reference_asset.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from focusbuddy.companion import gltf  # noqa: E402


def _tex_base(n=256):
    y, x = np.mgrid[0:n, 0:n] / n
    check = ((np.floor(x * 8) + np.floor(y * 8)) % 2).astype(float)
    r = 0.55 + 0.35 * y
    g = 0.62 + 0.2 * check
    b = 0.95 - 0.35 * y
    img = np.stack([r, g, b, np.ones_like(r)], axis=2)
    img[(x > 0.45) & (x < 0.55)] = (0.95, 0.5, 0.35, 1.0)          # a stripe to see seams / UV direction
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _tex_normal(n=256):
    y, x = np.mgrid[0:n, 0:n] / n
    h = np.sin(x * 2 * math.pi * 8) * np.sin(y * 2 * math.pi * 8) * 0.5
    dx = np.gradient(h, axis=1) * n / 64
    dy = np.gradient(h, axis=0) * n / 64
    nrm = np.stack([-dx, -dy, np.ones_like(h)], axis=2)
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    img = np.concatenate([(nrm * 0.5 + 0.5), np.ones_like(h)[..., None]], axis=2)
    return (img * 255).astype(np.uint8)


def _tex_mr(n=128):
    y, x = np.mgrid[0:n, 0:n] / n
    rough = 0.35 + 0.55 * ((np.floor(y * 6) % 2) > 0)
    metal = ((y > 0.40) & (y < 0.48)).astype(float)                  # metallic belt
    img = np.stack([np.ones_like(x), rough, metal, np.ones_like(x)], axis=2)
    return (img * 255).astype(np.uint8)


def _tex_occlusion(n=128):
    y, x = np.mgrid[0:n, 0:n] / n
    ao = 1.0 - 0.45 * np.exp(-((x - 0.5) ** 2) / 0.002) - 0.3 * np.exp(-((y - 0.44) ** 2) / 0.001)
    img = np.stack([ao, ao, ao, np.ones_like(ao)], axis=2)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _tex_emissive(n=128):
    y, x = np.mgrid[0:n, 0:n] / n
    band = ((y > 0.80) & (y < 0.86)).astype(float)                   # visor band on the head
    img = np.stack([band * 0.3, band * 0.95, band * 1.0, np.ones_like(x)], axis=2)
    return (img * 255).astype(np.uint8)


def build(out: Path) -> dict:
    src = gltf.load(ROOT / "assets" / "characters" / "buddy" / "model_low.glb")
    skin = src.skins[0]
    parents = src.parents()
    world = src.world_matrices()
    jnames = [src.nodes[j].name for j in skin.joints]
    # merge every primitive (bind space)
    P, N, C, J, W, F = [], [], [], [], [], []
    off = 0
    for nd in src.nodes:
        if nd.mesh is None:
            continue
        for pr in src.meshes[nd.mesh].primitives:
            P.append(pr.positions)
            N.append(pr.normals)
            C.append(pr.colors[:, :3] if pr.colors is not None else np.ones((len(pr.positions), 3)))
            J.append(pr.joints)
            W.append(pr.weights)
            F.append(pr.indices + off)
            off += len(pr.positions)
    P, N, C, J, W, F = (np.concatenate(a) for a in (P, N, C, J, W, F))
    # UVs: cylindrical around +Y (u = angle, v = height)
    lo, hi = P.min(axis=0), P.max(axis=0)
    ang = np.arctan2(P[:, 0], P[:, 2])
    uv = np.stack([(ang / (2 * math.pi)) % 1.0, 1.0 - (P[:, 1] - lo[1]) / (hi[1] - lo[1])], axis=1)
    from focusbuddy.companion.character_asset import _tangents
    T = _tangents(P.astype(float), N.astype(float), uv, F.reshape(-1, 3))
    # morph targets (ARKit names) around the head / face
    head = world[src.node_index("Head")][:3, 3]
    rel = P - head
    d_head = np.linalg.norm(rel, axis=1)
    face = np.clip(1.0 - d_head / (np.ptp(P[:, 1]) * 0.25), 0.0, 1.0) * (rel[:, 2] > 0)
    lower = face * (rel[:, 1] < 0)
    left, right = face * (rel[:, 0] > 0), face * (rel[:, 0] <= 0)
    up = face * (rel[:, 1] > 0)
    s = np.ptp(P[:, 1])
    targets = {
        "jawOpen": np.stack([0 * lower, -0.05 * s * lower, 0.01 * s * lower], axis=1),
        "eyeBlinkLeft": np.stack([0 * left, -0.01 * s * left * up, 0 * left], axis=1),
        "eyeBlinkRight": np.stack([0 * right, -0.01 * s * right * up, 0 * right], axis=1),
        "mouthSmile": np.stack([0.02 * s * np.sign(rel[:, 0]) * lower, 0.01 * s * lower, 0 * lower], axis=1),
        "browDownLeft": np.stack([0 * left, -0.015 * s * left * up, 0.005 * s * left * up], axis=1),
        "cheekPuff": N * (0.02 * s * face)[:, None],
    }
    w = gltf.GLBWriter("FocusBuddy build_reference_asset.py")
    # skeleton (same nodes / rest TRS as the source)
    node_of = {}
    for ji, nidx in enumerate(skin.joints):
        nd = src.nodes[nidx]
        p = parents[nidx]
        node_of[nidx] = w.add_node(nd.name, translation=tuple(nd.translation), rotation=tuple(nd.rotation),
                                   parent=node_of.get(p), root=p not in node_of)
    tex = [w.add_texture(_tex_base()), w.add_texture(_tex_normal()), w.add_texture(_tex_mr()),
           w.add_texture(_tex_occlusion()), w.add_texture(_tex_emissive())]
    mat = w.add_material("TestSuit", (1.0, 1.0, 1.0, 1.0), 1.0, 1.0, (1.0, 1.0, 1.0), base_tex=tex[0],
                         normal_tex=tex[1], normal_scale=1.0, mr_tex=tex[2], occlusion_tex=tex[3],
                         emissive_tex=tex[4])
    names = list(targets)
    mesh = w.add_mesh("TestBody", P, F, N, None, J, W, mat,
                      [{"POSITION": targets[k].astype(np.float32)} for k in names], names, uvs=uv.astype(np.float32),
                      tangents=T.astype(np.float32))
    sk = w.add_skin([node_of[j] for j in skin.joints], skin.inverse_bind, node_of[skin.joints[0]])
    mesh_node = w.add_node("TestBody", root=True, mesh=mesh, skin=sk)
    # ---- clips
    def node(name):
        return node_of[src.node_index(name)]

    def rest_q(name):
        return np.asarray(src.nodes[src.node_index(name)].rotation, dtype=float)

    def qrot(axis, deg, base):
        a = math.radians(deg) / 2
        q = np.array([*(np.asarray(axis, float) * math.sin(a)), math.cos(a)])
        x1, y1, z1, w1 = base
        x2, y2, z2, w2 = q
        return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    times = np.linspace(0, 2.0, 31)
    idle = [(node("Chest"), "rotation", times, np.array([qrot((1, 0, 0), 2.5 * math.sin(t * math.pi), rest_q("Chest"))
                                                          for t in times])),
            (node("Head"), "rotation", times, np.array([qrot((1, 0, 0), 4 * math.sin(t * math.pi + 1), rest_q("Head"))
                                                         for t in times]))]
    blink = np.zeros((len(times), len(names)))
    for k, t in enumerate(times):
        b = max(0.0, 1.0 - abs(t - 1.2) / 0.08)
        blink[k, names.index("eyeBlinkLeft")] = b
        blink[k, names.index("eyeBlinkRight")] = b
    w.add_animation("Idle", idle + [(mesh_node, "weights", times, blink.reshape(-1, 1))])
    # walk: 1 s cycle, the pelvis moves forward 0.6 m per cycle (root motion)
    wt = np.linspace(0, 1.0, 25)
    pel = src.nodes[src.node_index("Pelvis")]
    pel_t = np.asarray(pel.translation, float)
    stride = 0.6
    walk = [(node("Pelvis"), "translation", wt,
             np.array([pel_t + np.array([0.0, 0.02 * abs(math.sin(2 * math.pi * t)), stride * t]) for t in wt]))]
    for side, sgn in (("Left", 1), ("Right", -1)):
        walk.append((node(f"{side}UpperLeg"), "rotation", wt,
                     np.array([qrot((1, 0, 0), 28 * sgn * math.sin(2 * math.pi * t), rest_q(f"{side}UpperLeg"))
                               for t in wt])))
        walk.append((node(f"{side}LowerLeg"), "rotation", wt,
                     np.array([qrot((1, 0, 0), 30 * max(0.0, -sgn * math.sin(2 * math.pi * t + 0.6)),
                                    rest_q(f"{side}LowerLeg")) for t in wt])))
        walk.append((node(f"{side}UpperArm"), "rotation", wt,
                     np.array([qrot((1, 0, 0), -22 * sgn * math.sin(2 * math.pi * t), rest_q(f"{side}UpperArm"))
                               for t in wt])))
    w.add_animation("Walk", walk)
    vt = np.linspace(0, 1.6, 25)
    w.add_animation("Wave", [(node("RightUpperArm"), "rotation", vt,
                              np.array([qrot((0, 0, 1), -120 * min(1.0, t * 3), rest_q("RightUpperArm")) for t in vt])),
                             (node("RightLowerArm"), "rotation", vt,
                              np.array([qrot((0, 0, 1), 25 * math.sin(t * 9), rest_q("RightLowerArm")) for t in vt]))])
    out.mkdir(parents=True, exist_ok=True)
    w.save(out / "model.glb", extras={"focusbuddy": {"id": "pipeline_test"}})
    meta = {
        "id": "pipeline_test", "kind": "biped", "species": "human", "personality": "buddy",
        "display_name": "Pipeline test asset", "production": False, "selectable": False,
        "description": "Generated test asset (tools/build_reference_asset.py) that exercises every pipeline "
                       "feature: all five PBR texture slots, tangents, skin weights, clips with root motion and "
                       "a morph weights channel, ARKit-named morph targets. Not production art.",
        "face": {"mode": "morph"},
        "clips": {"idle": "Idle", "walk": "Walk", "wave": "Wave"},
        "clip_settings": {"Walk": {"root_motion": "HYBRID"}, "Wave": {"mask": "upper"}},
        "root_motion": "HYBRID",
        "transitions": {"default": 0.2},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"vertices": int(len(P)), "triangles": int(len(F)), "morphs": names, "clips": ["Idle", "Walk", "Wave"]}


if __name__ == "__main__":
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication
    app = QGuiApplication(sys.argv[:1])
    print(build(ROOT / "assets" / "characters" / "pipeline_test"))
