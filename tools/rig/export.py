"""Write a rigged companion as a FocusBuddy production GLB (+ metadata.json, LICENSE.md).

Rig frame (Z up, facing -Y) -> glTF (Y up, facing +Z): (x, y, z) -> (x, z, -y).
The skeleton is written translation-only (rest rotations identity), like the other built-ins, so
the RigDriver retargets it by bone directions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "assets" / "characters"

LICENSE = """# {name} - 3D model

Generated for FocusBuddy by the project owner with **Hunyuan3D** (Tencent, image-to-3D) from
the project's own character art, then rigged for FocusBuddy by `tools/rig/`: a skeleton placed
from landmarks and checked per character, Blender bone-heat skin weights with per-character
clean-up, and a face rig - eyelids and an opening mouth painted into the model's own texture at
run time, plus blink / squint / brow / mouth-corner / jaw morph targets. Textures resized.

The surface and the painted texture are the generator's output, not hand-modelled; the rig is
automatic with per-character corrections, not hand-painted. Replace `model.glb` with an
artist-made model for final production quality.

Licence: Hunyuan3D is released under the Tencent Hunyuan 3D Community License
(https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE); models made on Tencent's
hosted sites follow those sites' terms. Read them before any commercial use - among other
things that licence does not cover use in the EU, the UK or South Korea, and large services
need a separate licence. The input artwork belongs to the project owner.
"""


def to_gltf(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float)
    return np.stack([P[..., 0], P[..., 2], -P[..., 1]], axis=-1)


def split_by_uv(F, FUV):
    """glTF vertices = unique (vertex, uv) corners. Returns (vertex index per new vertex,
    uv per new vertex, new faces)."""
    corners = np.concatenate([F.reshape(-1, 1).astype(np.float64), np.round(FUV.reshape(-1, 2), 6)], axis=1)
    uniq, inv = np.unique(corners, axis=0, return_inverse=True)
    return uniq[:, 0].astype(np.int64), uniq[:, 1:].astype(np.float32), inv.reshape(-1, 3).astype(np.uint32)


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


# morphs that reshape the surface enough to need their own normals (the others are small)
NORMAL_MORPHS = ("eyeBlink", "eyeSquint")


def write(cid: str, cfg: dict, V, F, FUV, tex, J: dict, par: dict, names: list, W: np.ndarray,
          morphs: dict | None = None, parts: list | None = None, meta_extra: dict | None = None,
          tex_size: int = 2048, lod: str | None = None, normals: np.ndarray | None = None):
    """``morphs``: name -> (n, 3) rig-frame offsets per welded vertex. ``parts``: extra meshes
    [{"name", "V", "F", "UV", "tex" | "color", "bone" (rigid) | "W", "morphs": {name: offsets}}]."""
    from PIL import Image
    from focusbuddy.companion.gltf import GLBWriter
    from weights import limit4
    order = ["Root"] + [n for n in names if n != "Root"]
    tips = [n for n in J if n not in order]
    order += tips
    oi = {n: i for i, n in enumerate(order)}
    w = GLBWriter("FocusBuddy tools/rig (Hunyuan3D model, rigged)")
    node = {}
    Jg = {n: to_gltf(J[n]) for n in order}

    def add(n):
        if n in node:
            return
        p = par.get(n)
        if p is not None and p not in node:
            add(p)
        rel = Jg[n] - (Jg[p] if p else np.zeros(3))
        node[n] = w.add_node(n, translation=tuple(rel), parent=node.get(p), root=p is None)
    for n in order:
        add(n)
    ibm = np.repeat(np.eye(4)[None], len(order), axis=0)
    for i, n in enumerate(order):
        ibm[i, :3, 3] = -Jg[n]
    skin = w.add_skin([node[n] for n in order], ibm, node["Root"])
    # body
    vid, uv, Fn = split_by_uv(F, FUV)
    Vg = to_gltf(V)
    Ng = to_gltf(vertex_normals(V, F) if normals is None else normals)
    Wfull = np.zeros((len(V), len(order)), np.float32)
    for k, n in enumerate(names):
        Wfull[:, oi[n]] = W[:, k]
    jt, wt = limit4(Wfull)
    th, tw = tex.shape[:2]
    out_size = (tex_size, int(round(tex_size * th / tw)))            # (width, height): the atlas' aspect
    img = Image.fromarray(tex).resize(out_size, Image.LANCZOS) if (tw, th) != out_size else Image.fromarray(tex)
    ti = w.add_texture(np.asarray(img.convert("RGB")), "image/jpeg")
    mi = w.add_material(f"{cid}_body", (1, 1, 1, 1), 0.0, 0.85, extras={"shading": cfg.get("shading", "cloth")},
                        base_tex=ti)
    targets, tnames = [], []
    for name, off in (morphs or {}).items():
        # normal deltas too: a morph that smooths a shape away (the blink flattens the sculpted eye)
        # must also smooth its shading
        moved = np.abs(off).sum(1) > 1e-7
        dN = np.zeros_like(V)
        if moved.any() and name in NORMAL_MORPHS:
            dN = vertex_normals(V + off, F) - vertex_normals(V, F)
            near = np.zeros(len(V), bool)
            near[F[moved[F].any(1)].reshape(-1)] = True
            dN[~near] = 0.0
        t = {"POSITION": to_gltf(off)[vid].astype(np.float32)}
        if name in NORMAL_MORPHS:
            t["NORMAL"] = to_gltf(dN)[vid].astype(np.float32)
        targets.append(t)
        tnames.append(name)
    mesh = w.add_mesh(cid, Vg[vid].astype(np.float32), Fn, Ng[vid].astype(np.float32), None,
                      jt[vid].astype(np.uint16), wt[vid], mi, targets or None, tnames or None, uvs=uv)
    w.add_node(f"{cid}_mesh", mesh=mesh, skin=skin, root=True)
    for part in parts or []:
        Pv = to_gltf(part["V"]).astype(np.float32)
        Np = to_gltf(vertex_normals(part.get("N_from", part["V"]), part["F"])).astype(np.float32)
        if "tex" in part:
            pt = w.add_texture(part["tex"], "image/png")
            pm = w.add_material(part["name"], (1, 1, 1, 1), 0.0, part.get("roughness", 0.8),
                                extras={"shading": part.get("shading", "skin")}, base_tex=pt)
        else:
            pm = w.add_material(part["name"], tuple(part["color"]) + (1.0,), 0.0, part.get("roughness", 0.8),
                                extras={"shading": part.get("shading", "skin")})
        if "W" in part:
            Wp = np.zeros((len(Pv), len(order)), np.float32)
            for k, n in enumerate(names):
                Wp[:, oi[n]] = part["W"][:, k]
            pj, pw = limit4(Wp)
        else:
            pj = np.zeros((len(Pv), 4), np.uint16)
            pj[:, 0] = oi[part["bone"]]
            pw = np.zeros((len(Pv), 4), np.float32)
            pw[:, 0] = 1.0
        ptg = [{"POSITION": to_gltf(o).astype(np.float32)} for o in part.get("morphs", {}).values()]
        pm_ = w.add_mesh(part["name"], Pv, part["F"].astype(np.uint32), Np, None, pj.astype(np.uint16), pw, pm,
                         ptg or None, list(part.get("morphs", {})) or None, uvs=part.get("UV"))
        w.add_node(part["name"], mesh=pm_, skin=skin, root=True)
    out = ASSETS / cid
    out.mkdir(parents=True, exist_ok=True)
    fname = "model.glb" if lod is None else f"model_{lod}.glb"
    w.save(out / fname, extras={"focusbuddy": {"id": cid, "source": "Hunyuan3D model, rigged by tools/rig",
                                                "lod": lod or "high"}})
    if lod is not None:                         # a level of detail: the metadata belongs to model.glb
        return out / fname
    bone_map = {n: n for n in order if n not in tips}
    meta = {
        "id": cid, "kind": cfg["kind"], "species": cfg["species"], "production": True, "source": "hunyuan3d",
        "production_note": f"{cfg['name']}: 3D model generated with Hunyuan3D from the character art, rigged "
                           "for FocusBuddy (skeleton, bone-heat weights, face rig) by tools/rig; animated by "
                           "the procedural rig.",
        "license": "Model generated with Hunyuan3D (Tencent Hunyuan 3D community licence) from the project's "
                   "own character art; rig by tools/rig. See LICENSE.md.",
        "bone_map": bone_map, "normals": "asset", "pelvis_lift": "auto",
        "face": {"mode": "morph"} if morphs else {"mode": "none"},
        "morph_map": {},
        "transitions": {"default": 0.25, "idle->loco": 0.25, "loco->idle": 0.3},
    }
    if cfg["kind"] == "quadruped":
        meta["tail_gain"] = 0.6
        # the tail's thickness below its centre line (internal units: 1 = the rig's unit, the model's
        # height maps to the rig height in the renderer, so store it as a fraction and convert there)
        tails = [k for k, n in enumerate(names) if n.startswith("Tail")]
        if tails:
            dom = np.argmax(W, axis=1)
            r = 0.0
            for k in tails:
                vs = V[dom == k]
                if len(vs):
                    r = max(r, float(J[names[k]][2] - vs[:, 2].min()))
            meta["tail_floor_frac"] = round(r, 4)
    meta["pelvis_lift_k"] = 1.0
    if cfg["kind"] == "biped":                 # free-standing arms: they may rise high (a real wave)
        meta.update({"arm_limit": 175.0, "wave_up": True, "arm_out": 0.06, "arm_fwd": 0.04})
    meta.update(meta_extra or {})
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "LICENSE.md").write_text(LICENSE.format(name=cfg["name"]), encoding="utf-8")
    for stale in ("skeleton.json",):
        if (out / stale).exists():
            (out / stale).unlink()
    import shutil
    if (out / "face").exists():
        shutil.rmtree(out / "face")
    return out / "model.glb"
