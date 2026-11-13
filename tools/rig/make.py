"""Rig and export companions: python tools/rig/make.py fox cat ...  (see tools/rig/README.md)."""

from __future__ import annotations

import json
import sys

import numpy as np

import export
from build import rig
from core import BUILD, CHARS, SOURCE, Model

SHADING = {"quadruped": "fur", "biped": "cloth"}


def make(cid: str, rebuild: bool = True) -> None:
    if rebuild:
        rig(cid)
    m = Model(cid)
    z = np.load(BUILD / f"{cid}_rig.npz")
    names = [str(n) for n in z["names"]]
    d = json.loads((BUILD / f"{cid}_joints.json").read_text())
    J = {k: np.array(v) for k, v in d["joints"].items()}
    cfg = dict(CHARS[cid])
    cfg.setdefault("shading", SHADING[cfg["kind"]])
    normals = None
    head_mask = None
    Vr, Wr = z["V"], z["W"]
    if cfg["kind"] == "biped" and (SOURCE / f"{cid}_head.glb").is_file():
        # the people: a separately generated, detailed head replaces the body model's head (only the
        # head: skeleton, body, weights and animations stay as rigged)
        import face
        import head_swap
        log = []
        Vr, m.F, m.FUV, m.tex, Wr, info = head_swap.swap(
            Vr, m.F, m.FUV, m.tex, Wr, names, J, head_swap.prep_head(cid), face.FACE[cid], head_swap.HEADS[cid],
            log=log, **head_swap.OPTIONS.get(cid, {}))
        print(f"{cid}: new head - " + "; ".join(log))
        head_mask = np.zeros(len(m.F), bool)
        head_mask[len(m.F) - info["head_faces"]:] = True
        # the face rig follows the new head's eyes and mouth
        ed = abs(info["eyes"][1][0] - info["eyes"][0][0])
        # the new heads have big, drawn (anime) eyes: the face rig treats them like the animals'
        # (round lids over the painted eye, a "^^" squint, the eyeball flattening when it closes)
        face.FACE[cid] = dict(face.FACE[cid], human=False,
                              eyes=[(x, z_, info["eye_r"][0], info["eye_r"][1], 0.0) for x, z_ in info["eyes"]],
                              mouth=((info["mouth"][0] - 0.2 * ed, info["mouth"][1]),
                                     (info["mouth"][0] + 0.2 * ed, info["mouth"][1]), info["mouth"][1]))
    elif cfg["kind"] == "biped":
        # no detailed head: a sharp, retouched face tile (texture coordinates only)
        import face_tile
        log = []
        tile_info = face_tile.apply(cid, m, Vr, J, log)
        normals = face_tile.smooth_normals(Vr, m.F, tile_info)
        print(f"{cid}: face tile - " + "; ".join(log))
    morphs = parts = None
    meta = {}
    try:
        import face
        morphs, parts, meta = face.build(cid, m, Vr, J, names, Wr)
    except ImportError:
        pass
    painted = meta.pop("_painted", None)
    p = export.write(cid, cfg, Vr, m.F, m.FUV, m.tex, J, d["parents"], names, Wr, morphs, parts, meta,
                     normals=normals)
    if painted is not None:
        (p.parent / "face").mkdir(exist_ok=True)
        np.savez_compressed(p.parent / "face" / "painted.npz", **painted)
    sizes = [f"{p.stat().st_size // 1024} KB"]
    # levels of detail: the same skeleton, weights / morphs carried over to a decimated mesh
    import lod
    for name, target in lod.LEVELS.items():
        V2, F2, FUV2, W2, m2, N2 = lod.level(Vr, m.F, m.FUV, Wr, morphs, target, normals, keep=head_mask)
        q = export.write(cid, cfg, V2, F2, FUV2, m.tex, J, d["parents"], names, W2, m2, parts, meta, lod=name,
                         normals=N2)
        sizes.append(f"{name} {len(F2)} tris {q.stat().st_size // 1024} KB")
    print(f"{cid}: " + ", ".join(sizes))


if __name__ == "__main__":
    for c in [a for a in sys.argv[1:] if not a.startswith("--")]:
        make(c, "--no-rig" not in sys.argv)
