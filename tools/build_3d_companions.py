"""Stage 2 of the 3D companions: orient, auto-rig, skin and export the meshes reconstructed from the
character sheets (``tools/reconstruct_3d.py``) as the production GLB characters.

Input  ``art/meshes/<id>.npz`` + ``<id>.png`` (TripoSR frame, z up, textured)
Output ``assets/characters/<id>/model.glb`` + ``metadata.json`` + ``LICENSE.md`` (+ ``skeleton.json``
for a tail pose of the procedural rig)

Steps for each companion

1. orientation   TripoSR's frame -> glTF (+Y up, facing +Z); the facing is found by a mirror-symmetry
                 search around the vertical axis (the sheet characters are left / right symmetric),
                 then the feet are put on Y = 0 and the body centred
2. skeleton      the canonical FocusBuddy bones are placed from the shape itself: horizontal slices
                 through the mesh find the legs (separate blobs near the ground), the crotch / belly,
                 the neck (narrowest slice under the head), the arms, the ears and the tail; a few
                 per-character hints cover what the shape cannot tell (long hair hiding the neck ...)
3. skin weights  every vertex is bound to the bones whose segments are nearest *relative to the
                 bone's thickness* (an arm resting on the body stays with the body), blended near the
                 joints, smoothed over the surface and limited to 4 influences; a dress / saree is
                 bound to both legs and the hips so it sways instead of splitting
4. morphs        a "breath" morph target (chest / belly along the normals)
5. export        one skinned, textured mesh (baked TripoSR colour texture) + the skeleton, in the
                 bind pose, with no clips: the procedural animation system (RigDriver retargeting,
                 leg IK, look-at, ears, tail springs) animates it; ``"pelvis_lift": "auto"`` makes
                 it stand at its own hip height (long / short legs for its size)

    python tools/build_3d_companions.py               # all companions with a reconstructed mesh
    python tools/build_3d_companions.py cat buddy     # some of them
    python tools/build_3d_companions.py --preview out.png   # + a turnaround preview sheet
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

MESHES = ROOT / "art" / "meshes"
ASSETS = ROOT / "assets" / "characters"

# Per-character settings. ``facing``: the rough yaw (deg, about +Y, applied after z-up -> y-up) that
# turns TripoSR's input view to face +Z (front / 3-4 views: about 0; the fox is a side view).
# Fractions are of the character's height.
CHARS = {
    # the animals: standing bodies from the first sheets, detailed heads from the detailed sheets
    # (``graft``: the source, the sheet, the collar's top edge in the front cut-out, the size of the
    # new head relative to the body's own, the collar / charm colours)
    "cat": {"kind": "quadruped", "species": "cat", "facing": 0, "shading": "fur", "tail": True, "ears": True,
            "name": "Cat", "mirror": True,
            "graft": {"source": "cat_front", "sheet": "cat", "collar_y": 452, "scale": 1.15,
                     "collar_rgb": (150, 40, 38), "charm_rgb": (215, 165, 60)}},
    "dog": {"kind": "quadruped", "species": "dog", "facing": 0, "shading": "fur", "tail": True, "ears": True,
            "name": "Dog", "mirror": True, "pelvis_y": 0.5, "chest_y": 0.52,
            "graft": {"source": "dog_front", "sheet": "dog", "collar_y": 425, "scale": 1.35,
                     "collar_rgb": (125, 35, 35), "charm_rgb": (215, 170, 70)}},
    "fox": {"kind": "quadruped", "species": "fox", "facing": 90, "shading": "fur", "tail": True, "ears": True,
            "name": "Fox", "rig_tail": {"rest": [222, -12, -14, -10]},
            "graft": {"source": "fox_front", "sheet": "fox", "collar_y": 525, "scale": 1.3, "t": 0.3,
                     "collar_rgb": (120, 40, 30), "charm_rgb": (200, 150, 70)}},
    "rabbit": {"kind": "quadruped", "species": "rabbit", "facing": 0, "shading": "fur", "tail": True,
               "ears": True, "name": "Rabbit", "pelvis": (0.17, -0.12), "chest": (0.175, 0.08),
               "head": (0.52, 0.12), "tail_min_y": 0.05, "shoulder_y": 0.17, "paw_y": 0.0,
               "graft": {"source": "rabbit_front", "sheet": "rabbit", "collar_y": 645, "scale": 1.0,
                         "collar_rgb": (130, 40, 35), "charm_rgb": (235, 120, 40)}},
    "robot": {"kind": "biped", "species": "robot", "facing": 270, "shading": "plastic", "name": "Robot"},
    # the people: the chibi bodies of the first sheets (free arms and hands), detailed heads
    "buddy": {"kind": "biped", "species": "human", "facing": 270, "shading": "cloth", "name": "Buddy",
              "graft": {"source": "boy_front", "sheet": "boy", "collar_y": 185, "scale": 1.05, "recolor": 0, "pitch": -15, "sink": 0.015, "t": 0.2}},
    "study_girl": {"kind": "biped", "species": "human", "facing": 270, "shading": "cloth", "name": "Study Girl",
                   "neck": 0.63, "hair_lum": 0.3,
                   "graft": {"source": "girl_front", "sheet": "girl", "collar_y": 198, "scale": 1.0, "recolor": 0, "pitch": -15, "sink": 0.0, "t": 0.2}},
    "saree_girl": {"kind": "biped", "species": "human", "facing": 270, "shading": "cloth", "skirt": True,
                   "name": "Saree Girl", "hair_lum": 0.3, "smooth": 10, "neck": 0.69,
                   "graft": {"source": "saree_front", "sheet": "saree", "collar_y": 238, "scale": 1.12, "sink": 0.0, "t": 0.15, "dz": 0.02,
                             "recolor": 0, "pitch": -15}},
}

LICENSE = """# {name} - 3D model

Reconstructed for FocusBuddy from the {name} character sheets (art supplied by the project owner)
with **TripoSR** (Stability AI and Tripo AI, MIT licence, https://github.com/VAST-AI-Research/TripoSR):
mesh and colour texture by `tools/reconstruct_3d.py` (with the sheet's front view projected onto
the visible surface), the detailed head grafted on by `tools/head_graft.py`, skeleton, skin
weights and export by `tools/build_3d_companions.py`. The facial expressions in `face/` (where
present) are the sheet's own expression panels, lined up with the model's face by
`tools/face_expressions.py`.

The model is an automatic image-to-3D reconstruction, not a hand-modelled asset: the sides the
sheet views do not show (mostly the back) are the network's guess, the surface is softer than a
sculpt, and the skin weights are computed, not painted. Replace `model.glb` with an artist-made
model for final production quality.

TripoSR is MIT licensed (Copyright (c) 2024 Tripo AI & Stability AI). The input artwork belongs
to the project owner.
"""


# ------------------------------------------------------------------ geometry helpers
def ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def zup_to_yup(v: np.ndarray) -> np.ndarray:
    """TripoSR (x, y, z-up) -> (x, z, -y): +Y up, a proper rotation."""
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def weld(v: np.ndarray, f: np.ndarray, eps: float = 1e-6):
    """UV seams split vertices; weld by position for normals / adjacency. Returns (ids, unique_pos)."""
    key = np.round(v / eps).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return inv.reshape(-1), v[first]


def symmetry_error(p: np.ndarray, tree_cls) -> float:
    c = 0.5 * (p[:, 0].min() + p[:, 0].max())
    m = p.copy()
    m[:, 0] = 2 * c - m[:, 0]
    d, _ = tree_cls(p).query(m)
    return float(np.mean(d))


def find_facing(v: np.ndarray, hint: float) -> float:
    """Yaw (deg) with the best left / right mirror symmetry, of the two facings the nearer the hint."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    p = v[rng.choice(len(v), min(len(v), 2500), replace=False)]
    errs = {}
    for a in np.arange(0, 180, 3.0):
        errs[a] = symmetry_error(p @ ry(a).T, cKDTree)
    best = min(errs, key=errs.get)
    fine = {a: symmetry_error(p @ ry(a).T, cKDTree) for a in np.arange(best - 3, best + 3.01, 0.5)}
    best = min(fine, key=fine.get)
    cands = [best % 360, (best + 180) % 360]

    def dist(a):
        return abs((a - hint + 180) % 360 - 180)
    return float(min(cands, key=dist))


def components_2d(pts: np.ndarray, cell: float):
    """Connected blobs of points in the XZ plane (occupancy grid). Returns a list of index arrays."""
    from scipy import ndimage
    if len(pts) == 0:
        return []
    lo = pts.min(axis=0) - cell
    ij = np.floor((pts - lo) / cell).astype(int)
    grid = np.zeros(ij.max(axis=0) + 2, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    grid = ndimage.binary_closing(grid, iterations=1) | grid
    lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
    ids = lab[ij[:, 0], ij[:, 1]]
    return [np.flatnonzero(ids == k) for k in range(1, n + 1)]


class Shape:
    """The oriented mesh, densely sampled on its surface, with slice queries."""

    def __init__(self, V: np.ndarray, F: np.ndarray, n: int = 160000):
        rng = np.random.default_rng(1)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        f = rng.choice(len(F), n, p=area / area.sum())
        r1, r2 = rng.random(n), rng.random(n)
        s1 = np.sqrt(r1)
        self.P = (1 - s1)[:, None] * a[f] + (s1 * (1 - r2))[:, None] * b[f] + (s1 * r2)[:, None] * c[f]
        self.H = float(V[:, 1].max())
        self.cell = self.H * 0.012

    def slice(self, y: float, half: float | None = None) -> np.ndarray:
        h = half if half is not None else self.H * 0.012
        return np.flatnonzero(np.abs(self.P[:, 1] - y) < h)

    def blobs(self, y: float, min_frac: float = 0.02):
        idx = self.slice(y)
        comps = components_2d(self.P[idx][:, [0, 2]], self.cell)
        comps = [idx[c] for c in comps]
        tot = sum(len(c) for c in comps) or 1
        return [c for c in comps if len(c) >= min_frac * tot]

    def centroid(self, idx) -> np.ndarray:
        return self.P[idx].mean(axis=0)


# ------------------------------------------------------------------ skeleton detection
def biped_joints(S: Shape, cfg: dict, mesh=None) -> dict:
    P, H = S.P, S.H
    J: dict = {}
    # crotch: lowest height whose slice is one blob around the middle (legs merged)
    crotch = None
    for f in np.arange(0.08, 0.6, 0.005):
        b = S.blobs(f * H)
        sides = {np.sign(S.centroid(c)[0]) for c in b if abs(S.centroid(c)[0]) > 0.02 * H}
        center = [c for c in b if P[c, 0].min() < -0.01 * H and P[c, 0].max() > 0.01 * H]
        if center and not (len(sides) == 2 and len(b) >= 2 and not center):
            crotch = f * H
            break
    if cfg.get("skirt") or crotch is None or crotch < 0.12 * H:
        crotch = cfg.get("crotch", 0.36) * H
    # neck: narrowest slice (x extent of the middle blob) between the chest and the head
    best, neck = 1e9, cfg.get("neck", 0.6) * H
    if "neck" not in cfg:
        for f in np.arange(0.45, 0.82, 0.005):
            b = S.blobs(f * H)
            mid = [c for c in b if P[c, 0].min() < 0 < P[c, 0].max()]
            if not mid:
                continue
            c = max(mid, key=len)
            w = P[c, 0].max() - P[c, 0].min()
            if w < best:
                best, neck = w, f * H
    hip = crotch + 0.04 * H
    # the body's depth centre: the hips (long hair hangs behind the chest, never at the hips)
    torso = S.slice(hip + 0.03 * H)
    tz = float(0.5 * (np.percentile(P[torso, 2], 5) + np.percentile(P[torso, 2], 95))) if len(torso) else 0.0
    J["Root"] = np.array([0.0, 0.0, tz])
    J["Pelvis"] = np.array([0.0, hip + 0.02 * H, tz])
    J["Spine"] = np.array([0.0, hip + 0.38 * (neck - hip), tz])
    J["Chest"] = np.array([0.0, hip + 0.72 * (neck - hip), tz])
    J["Neck"] = np.array([0.0, neck - 0.01 * H, tz])
    hz = tz + cfg.get("head_dz", 0.0) * H
    J["Head"] = np.array([0.0, neck + 0.03 * H, hz])
    J["HeadTop"] = np.array([0.0, H, hz])
    # legs: blob centroids per side
    for side, s in (("Left", 1.0), ("Right", -1.0)):
        def leg_at(y, s=s):
            b = [c for c in S.blobs(y) if np.sign(S.centroid(c)[0]) == s and abs(S.centroid(c)[0]) > 0.015 * H]
            if not b or cfg.get("skirt"):
                return np.array([s * cfg.get("leg_x", 0.075) * H, y, tz])
            c = S.centroid(max(b, key=len))
            return np.array([c[0], y, c[2]])
        ank = leg_at(0.055 * H)
        knee = leg_at(0.5 * (hip + 0.055 * H))
        up = leg_at(crotch - 0.03 * H)
        up = np.array([up[0], hip, tz])
        J[f"{side}UpperLeg"], J[f"{side}LowerLeg"], J[f"{side}Foot"] = up, knee, ank
        foot = np.flatnonzero((P[:, 1] < 0.05 * H) & (np.sign(P[:, 0]) == s))
        tipz = float(np.percentile(P[foot, 2], 97)) if len(foot) else ank[2] + 0.06 * H
        J[f"{side}FootTip"] = np.array([ank[0], 0.02 * H, max(tipz, ank[2] + 0.03 * H)])
    # arms: shoulder just under the neck, hand = the lowest point of the outer blob on that side
    sh_y = neck - cfg.get("shoulder_drop", 0.06) * H
    for side, s in (("Left", 1.0), ("Right", -1.0)):
        sl = S.slice(sh_y)
        xs = P[sl, 0] * s
        sh_x = float(np.percentile(xs[xs > 0], 80)) * 0.78 if np.any(xs > 0) else 0.12 * H
        shoulder = np.array([s * sh_x, sh_y, tz])
        # the arm: points outside the torso width at mid-arm height, followed down
        hand = None
        for f in np.arange(sh_y / H - 0.06, max(0.12, crotch / H - 0.06), -0.01):
            b = [c for c in S.blobs(f * H) if np.sign(S.centroid(c)[0]) == s and
                 abs(S.centroid(c)[0]) > sh_x * 0.75 and P[c, 0].min() * s > 0 and
                 abs(S.centroid(c)[2] - tz) < 0.07 * H and              # not a lock of hair behind
                 np.ptp(P[c, 0]) < 0.12 * H]                            # nor the leg it rests on
            if b:
                hand = S.centroid(max(b, key=len))
            elif hand is not None:
                break
        if hand is None or "hand_y" in cfg:
            hy = cfg.get("hand_y", 0.36) * H
            hand = np.array([s * cfg.get("hand_x", sh_x / H + 0.02) * H, hy, tz])
        hand = hand + np.array([0.0, 0.02 * H, 0.0])
        elbow = 0.5 * (shoulder + hand) + np.array([s * 0.01 * H, 0.0, -0.01 * H])
        J[f"{side}UpperArm"], J[f"{side}LowerArm"], J[f"{side}Hand"] = shoulder, elbow, hand
        J[f"{side}HandTip"] = hand + (hand - elbow) / max(np.linalg.norm(hand - elbow), 1e-6) * 0.05 * H
    # the sheet characters are symmetric: average the two arms (mirror in x)
    for a, b_ in (("LeftLowerArm", "RightLowerArm"), ("LeftHand", "RightHand"), ("LeftHandTip", "RightHandTip")):
        m = 0.5 * (J[a] * np.array([1, 1, 1]) + J[b_] * np.array([-1, 1, 1]))
        J[a], J[b_] = m.copy(), m * np.array([-1, 1, 1])
    return J


def quadruped_joints(S: Shape, cfg: dict, mesh=None) -> dict:
    P, H = S.P, S.H
    J: dict = {}
    L = float(P[:, 2].max() - P[:, 2].min())
    # four legs: the four blobs where the body touches the ground (k-means, k = 4, when paws merge)
    y0 = cfg.get("leg_probe", 0.06) * H
    g = P[(P[:, 1] < y0) & (P[:, 2] > P[:, 2].min() + cfg.get("tail_skip", 0.0) * L)]
    comps = sorted(components_2d(g[:, [0, 2]], S.cell), key=len, reverse=True)
    if len(comps) >= 4 and len(comps[3]) > 0.08 * len(comps[0]):
        cents = np.array([[g[c, 0].mean(), y0, g[c, 2].mean()] for c in comps[:4]])
    else:
        from scipy.cluster.vq import kmeans2
        base = g[:, [0, 2]]
        zm = float(np.median(base[:, 1]))
        init = np.array([[sx * 0.08 * H, zm + sz * 0.15 * L] for sx in (1, -1) for sz in (1, -1)])
        cents2, _ = kmeans2(base, init, minit="matrix", seed=2)
        cents = np.array([[c[0], y0, c[1]] for c in cents2])
    front = cents[np.argsort(-cents[:, 2])[:2]]
    back = cents[np.argsort(cents[:, 2])[:2]]
    zf, zb = float(front[:, 2].mean()), float(back[:, 2].mean())

    def body_span(z):
        """(belly, back) heights at z: the surface crossings of a vertical line through the middle;
        the top inside-interval is the body (the samples are on the surface, so the crossings pair up)."""
        found = []
        for dz in (-0.04, 0.0, 0.04):
            for x in (-0.04, -0.02, 0.0, 0.02, 0.04):
                sel = np.flatnonzero((np.abs(P[:, 2] - z - dz * L) < 0.012 * L) & (np.abs(P[:, 0] - x * H) < 0.012 * H))
                ys = np.sort(P[sel, 1])
                if len(ys) < 4:
                    continue
                cuts = np.flatnonzero(np.diff(ys) > 0.02 * H)
                cross = [float(np.mean(g)) for g in np.split(ys, cuts + 1)]
                if len(cross) >= 2:
                    found.append((cross[-2], cross[-1]))
        if not found:
            return cfg.get("belly", 0.35) * H, 0.8 * H
        # a leg under a line lowers it, a chin over it raises it: the median line is the body
        f = np.array(found)
        return float(np.median(f[:, 0])), float(np.median(f[:, 1]))

    fb, ft = body_span(zf)
    bb, bt = body_span(zb)
    fb = min(fb, cfg.get("belly_max", 0.7) * H)
    bb = min(bb, cfg.get("belly_max", 0.7) * H)
    pel_y = bb + 0.5 * (min(bt, bb + 0.4 * H) - bb)
    che_y = fb + 0.5 * (min(ft, fb + 0.4 * H) - fb)
    pel = np.array([0.0, pel_y, zb])
    che = np.array([0.0, che_y, zf])
    if "pelvis_y" in cfg:
        pel[1] = pel_y = cfg["pelvis_y"] * H
    if "chest_y" in cfg:
        che[1] = che_y = cfg["chest_y"] * H
    if "pelvis" in cfg:                      # (y, z) in heights: a sitting / crouching body
        pel = np.array([0.0, cfg["pelvis"][0] * H, cfg["pelvis"][1] * H])
        pel_y, zb = pel[1], pel[2]
    if "chest" in cfg:
        che = np.array([0.0, cfg["chest"][0] * H, cfg["chest"][1] * H])
        che_y, zf = che[1], che[2]
    J["Root"] = np.array([0.0, 0.0, 0.5 * (zf + zb)])
    J["Pelvis"] = pel
    J["Spine"] = 0.5 * (pel + che)
    J["Chest"] = che
    # head: the mass above the middle of the chest and in front of the spine (the ears are the
    # top tenth); the head joint sits at its base, toward the chest
    _, mid_top0 = body_span(0.5 * (zf + zb))
    reg = np.flatnonzero((P[:, 1] > 0.5 * (fb + min(ft, mid_top0))) & (P[:, 2] > zf - 0.1 * L))
    ry_ = P[reg, 1]
    core = reg[(ry_ > np.percentile(ry_, 40)) & (ry_ < np.percentile(ry_, 90))]
    hc = P[core].mean(axis=0)
    hc[0] = 0.0
    if "head" in cfg:
        hc = np.array([0.0, cfg["head"][0] * H, cfg["head"][1] * H])
    top = reg[P[reg, 1] > hc[1]]
    _, mid_top = body_span(0.5 * (zf + zb))
    chest_top = np.array([0.0, min(ft, mid_top), zf])
    J["Head"] = hc + cfg.get("head_base", 0.45) * (chest_top - hc)
    J["HeadTop"] = hc + (hc - J["Head"]) * 1.2
    J["Neck"] = 0.5 * (J["Chest"] + J["Head"])
    # legs
    for side, s in (("Left", 1.0), ("Right", -1.0)):
        for fr, (ub, lb, eb), zz, belly, joint_y in (
                (True, ("UpperArm", "LowerArm", "Hand"), zf, fb, che_y),
                (False, ("UpperLeg", "LowerLeg", "Foot"), zb, bb, pel_y)):
            grp = front if fr else back
            paw = grp[np.argmax(grp[:, 0])] if s > 0 else grp[np.argmin(grp[:, 0])]   # left = +x of the pair
            ank = np.array([paw[0], 0.05 * H, paw[2]])
            top_y = belly + 0.35 * (joint_y - belly)
            if fr and "shoulder_y" in cfg:
                top_y = cfg["shoulder_y"] * H
            up = np.array([paw[0] * 0.85, top_y, zz])
            mid_y = 0.5 * (top_y + ank[1])
            sl = S.slice(mid_y)
            near = sl[(np.abs(P[sl, 0] - paw[0]) < 0.08 * H) & (np.abs(P[sl, 2] - paw[2]) < 0.12 * L)]
            knee = P[near].mean(axis=0) if len(near) else 0.5 * (up + ank)
            knee[1] = mid_y
            pre = "" if fr else ""
            J[f"{side}{ub}"], J[f"{side}{lb}"], J[f"{side}{eb}"] = up, knee, ank
            J[f"{side}{eb}Tip"] = ank + np.array([0.0, -0.03 * H, 0.06 * H])
            del pre
    # ears: on each side, the highest points of the head region; the base is part-way down to the head
    if cfg.get("ears"):
        for side, s in (("Left", 1.0), ("Right", -1.0)):
            e = top[P[top, 0] * s > 0.02 * H]
            if len(e) < 30:
                continue
            ys = P[e, 1]
            tip = P[e[ys > np.percentile(ys, 96)]].mean(axis=0)
            J[f"{side}Ear"] = hc + cfg.get("ear_base", 0.55) * (tip - hc)
            J[f"{side}EarTip"] = tip
    # tail: the geodesic path over the surface from the tail tip (the hindmost point) to the hips
    if cfg.get("tail") and mesh is not None:
        V, Fw = mesh
        hi = np.flatnonzero(V[:, 1] > cfg.get("tail_min_y", 0.12) * H)
        tip = int(hi[np.argmin(V[hi, 2])])
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        e = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]])
        wgt = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
        G = coo_matrix((wgt, (e[:, 0], e[:, 1])), shape=(len(V), len(V))).tocsr()
        G = G.maximum(G.T)
        _, pred = dijkstra(G, indices=tip, return_predecessors=True)
        pel = J["Pelvis"]
        goal = int(np.argmin(np.linalg.norm(V - (pel + np.array([0.0, 0.1 * H, -0.05 * L])), axis=1)))
        path = [goal]
        while path[-1] != tip and pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        path = np.array(path[::-1])                                    # tip -> hips
        r_body = max(0.5 * (bt - bb), 0.08 * H)
        dpel = np.linalg.norm(V[path] - pel, axis=1)
        out_k = np.flatnonzero(dpel > r_body * cfg.get("tail_base", 0.9))
        if len(out_k) >= 2:
            seg_pts = V[path[:out_k[-1] + 1]]                          # tip ... base (outside the body)
            seglen = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg_pts, axis=0), axis=1))]
            total = float(seglen[-1])
            n = 4 if total > 0.3 * H else 3 if total > 0.15 * H else 2
            rr = max(0.03 * H, 0.25 * total / n)
            chain = []
            for t in np.linspace(total, 0.0, n + 1):                  # base -> tip
                q = seg_pts[np.searchsorted(seglen, t).clip(0, len(seg_pts) - 1)]
                near = np.linalg.norm(V - q, axis=1) < rr
                chain.append(V[near].mean(axis=0) if near.sum() > 3 else q)
            base = chain[0] + 0.35 * (pel - chain[0])
            chain = [base] + chain[1:-1]
            for k, p in enumerate(chain):
                J[f"Tail{k}"] = p
            J["TailTip"] = V[tip]
    return J


PARENTS = {
    "Pelvis": "Root", "Spine": "Pelvis", "Chest": "Spine", "Neck": "Chest", "Head": "Neck", "HeadTop": "Head",
    "LeftEar": "Head", "RightEar": "Head", "LeftEarTip": "LeftEar", "RightEarTip": "RightEar",
    "LeftUpperArm": "Chest", "LeftLowerArm": "LeftUpperArm", "LeftHand": "LeftLowerArm", "LeftHandTip": "LeftHand",
    "RightUpperArm": "Chest", "RightLowerArm": "RightUpperArm", "RightHand": "RightLowerArm",
    "RightHandTip": "RightHand",
    "LeftUpperLeg": "Pelvis", "LeftLowerLeg": "LeftUpperLeg", "LeftFoot": "LeftLowerLeg", "LeftFootTip": "LeftFoot",
    "RightUpperLeg": "Pelvis", "RightLowerLeg": "RightUpperLeg", "RightFoot": "RightLowerLeg",
    "RightFootTip": "RightFoot",
}
# a bone's segment for skinning ends at this joint
SEG_END = {"Pelvis": "Spine", "Spine": "Chest", "Chest": "Neck", "Neck": "Head", "Head": "HeadTop",
           "LeftEar": "LeftEarTip", "RightEar": "RightEarTip",
           "LeftUpperArm": "LeftLowerArm", "LeftLowerArm": "LeftHand", "LeftHand": "LeftHandTip",
           "RightUpperArm": "RightLowerArm", "RightLowerArm": "RightHand", "RightHand": "RightHandTip",
           "LeftUpperLeg": "LeftLowerLeg", "LeftLowerLeg": "LeftFoot", "LeftFoot": "LeftFootTip",
           "RightUpperLeg": "RightLowerLeg", "RightLowerLeg": "RightFoot", "RightFoot": "RightFootTip"}
TIPS = {"HeadTop", "LeftEarTip", "RightEarTip", "LeftHandTip", "RightHandTip", "LeftFootTip", "RightFootTip",
        "TailTip"}


def hierarchy(J: dict) -> tuple[list, dict, dict]:
    par = dict(PARENTS)
    seg = dict(SEG_END)
    tails = sorted((int(k[4:]), k) for k in J if k.startswith("Tail") and k[4:].isdigit())
    for i, (_, k) in enumerate(tails):
        par[k] = "Pelvis" if i == 0 else tails[i - 1][1]
        seg[k] = tails[i + 1][1] if i + 1 < len(tails) else "TailTip"
    if tails:
        par["TailTip"] = tails[-1][1]
    names = ["Root"] + [n for n in par if n in J and n != "Root"]
    # topological order
    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        if n != "Root":
            visit(par[n])
        seen.add(n)
        order.append(n)
    for n in names:
        visit(n)
    return order, {n: par.get(n) for n in order}, seg


# ------------------------------------------------------------------ skin weights
def seg_dist(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def torso_band(S: "Shape", J: dict, H: float) -> list:
    """Biped torso extent per height between the hands and the shoulders: the middle blob of each
    slice, when the arms hang free of it (a gap); [(y, x_lo, x_hi)]."""
    out = []
    sh_x = abs(J["LeftUpperArm"][0])
    for y in np.arange(J["LeftHand"][1] - 0.02 * H, J["LeftUpperArm"][1] + 0.01 * H, 0.01 * H):
        mids = [c for c in S.blobs(y) if S.P[c, 0].min() < 0 < S.P[c, 0].max()]
        if not mids:
            continue
        c = max(mids, key=len)
        lo, hi = float(S.P[c, 0].min()), float(S.P[c, 0].max())
        if max(-lo, hi) < 1.05 * sh_x:                  # the arms are not part of this blob
            out.append((float(y), lo, hi))
    return out


def skin_weights(P: np.ndarray, faces: np.ndarray, J: dict, order: list, par: dict, seg: dict, H: float,
                 cfg: dict, col: np.ndarray | None = None, band: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    bones = [n for n in order if n in seg and n in J and seg[n] in J]
    idx = {n: i for i, n in enumerate(order)}
    D = np.stack([seg_dist(P, J[b], J[seg[b]]) for b in bones], axis=1)          # (V, B)
    near = np.argmin(D, axis=1)
    # thickness of each bone: the median distance of the vertices nearest to it
    rad = np.array([np.median(D[near == k, k]) if np.any(near == k) else 0.03 * H for k in range(len(bones))])
    rad = np.clip(rad, 0.015 * H, None)
    limb = np.array([("Arm" in b or "Leg" in b or "Hand" in b or "Foot" in b) for b in bones])
    rad[limb] = np.minimum(rad[limb], 0.045 * H)       # a thick thigh never claims the paw below it
    Dn = D / rad[None, :]
    # gates: legs / arms only take vertices on their own side; limbs never take the far body
    gate = np.ones_like(Dn)
    for k, b in enumerate(bones):
        if b.startswith("Left"):
            gate[:, k] = 1.0 / (1.0 + np.exp(-(P[:, 0] + 0.0 * H) / (0.01 * H)))
        elif b.startswith("Right"):
            gate[:, k] = 1.0 / (1.0 + np.exp((P[:, 0]) / (0.01 * H)))
    if cfg["kind"] == "biped":
        tz = J["Pelvis"][2]
        hip_y = J["Pelvis"][1]
        sh_x = abs(J["LeftUpperArm"][0])
        for k, b in enumerate(bones):
            if "Arm" in b or "Hand" in b:
                # arms: not the hair behind the back, not the middle of the torso
                gate[:, k] *= 1.0 / (1.0 + np.exp(-(P[:, 2] - (tz - 0.07 * H)) / (0.01 * H)))
                gate[:, k] *= 1.0 / (1.0 + np.exp(-(np.abs(P[:, 0]) - 0.6 * sh_x) / (0.01 * H)))
                gate[:, k] *= 1.0 / (1.0 + np.exp(-(P[:, 1] - (hip_y + 0.01 * H)) / (0.01 * H)))   # not the thighs
                if band:                                    # nor the side of the torso it hangs next to
                    ys = np.array([b_[0] for b_ in band])
                    j = np.clip(np.searchsorted(ys, P[:, 1]), 0, len(ys) - 1)
                    near = np.abs(ys[j] - P[:, 1]) < 0.012 * H
                    lo = np.array([b_[1] for b_ in band])[j] + 0.01 * H
                    hi = np.array([b_[2] for b_ in band])[j] - 0.01 * H
                    inside = near & (P[:, 0] > lo) & (P[:, 0] < hi)
                    gate[inside, k] = 0.0
            elif "Leg" in b or "Foot" in b:
                gate[:, k] *= 1.0 / (1.0 + np.exp((P[:, 1] - (hip_y + 0.04 * H)) / (0.01 * H)))
    if cfg["kind"] == "quadruped":
        for k, b in enumerate(bones):
            if any(t in b for t in ("UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot")):
                side = "Left" if b.startswith("Left") else "Right"
                top = J[f"{side}UpperArm"] if ("Arm" in b or "Hand" in b) else J[f"{side}UpperLeg"]
                # a leg never takes the body above its top joint, nor the chest / belly between the legs
                gate[:, k] *= 1.0 / (1.0 + np.exp((P[:, 1] - (top[1] + 0.02 * H)) / (0.01 * H)))
                gate[:, k] *= 1.0 / (1.0 + np.exp(-(np.abs(P[:, 0]) - 0.45 * abs(top[0])) / (0.008 * H)))
        # ... and the body never takes the legs below the belly
        low = min(J["LeftUpperArm"][1], J["LeftUpperLeg"][1]) - cfg.get("leg_zone", 0.08) * H
        for k, b in enumerate(bones):
            if b in ("Pelvis", "Spine", "Chest", "Neck", "Head", "LeftEar", "RightEar"):
                gate[:, k] *= 1.0 / (1.0 + np.exp(-(P[:, 1] - low) / (0.01 * H)))
    W = np.exp(-(Dn ** 2) * 1.6) * gate + 1e-12
    if cfg["kind"] == "quadruped":
        # paws on the ground (also when the two front paws touch in the middle) follow their leg only
        legs = [k for k, b in enumerate(bones) if any(t in b for t in ("Hand", "Foot", "LowerArm", "LowerLeg"))]
        if legs:
            dl = D[:, legs]
            near_leg = np.array(legs)[np.argmin(dl, axis=1)]
            paw = (P[:, 1] < cfg.get("paw_y", 0.09) * H) & (dl.min(axis=1) < 0.09 * H)
            W[paw] = 1e-12
            W[paw, near_leg[paw]] = 1.0
    # keep only the bones of the nearest bone's family (itself, parent, children) - no bleeding
    fam = np.zeros((len(bones), len(bones)), bool)
    for k, b in enumerate(bones):
        fam[k, k] = True
        for j, c in enumerate(bones):
            if par.get(c) == b or par.get(b) == c:
                fam[k, j] = True
            if par.get(c) == par.get(b) and b.startswith(("Pelvis", "Spine", "Chest")) is False and \
                    c.startswith("Tail") and b.startswith("Tail"):
                fam[k, j] = True
            # a biped's upper arm swings far (waves): blending it with the chest collapses the sleeve,
            # so the arm stays its own family (the chest keeps a soft band from the smoothing only)
            if cfg["kind"] == "biped" and "UpperArm" in b and c == "Chest":
                fam[k, j] = False
    # the family comes from the geometrically nearest allowed bone (thickness-normalised weights alone
    # would let a thick torso far away outweigh a thin, distant shin)
    nearn = np.argmin(D + (gate < 0.5) * 1e3, axis=1)
    if cfg["kind"] == "quadruped" and legs:
        nearn[paw] = near_leg[paw]
    W = W * fam[nearn]
    W[np.arange(len(W)), nearn] += 1e-9
    bi = {b: k for k, b in enumerate(bones)}
    if col is not None and "hair_lum" in cfg:
        # long hair (dark, above the hands) follows the head / neck / chest, never an arm or a leg
        lum = col @ np.array([0.299, 0.587, 0.114])
        hair = (lum < cfg["hair_lum"]) & (P[:, 1] > J["LeftHand"][1] + 0.03 * H)
        limb = np.array([("Arm" in b or "Hand" in b or "Leg" in b or "Foot" in b) for b in bones])
        W[np.ix_(hair, limb)] = 0.0
        body = [bi[b] for b in ("Head", "Neck", "Chest", "Spine") if b in bi]
        W[np.ix_(hair, body)] += 1e-4
    if cfg.get("skirt"):
        # a dress / saree: below the hips the cloth follows the hips and, more and more toward the
        # hem, the thigh on its side (softly across the middle) - it sways instead of splitting
        hip_y = J["Pelvis"][1]
        below = np.flatnonzero(P[:, 1] < hip_y)
        t = np.clip((hip_y - P[below, 1]) / max(hip_y, 1e-6), 0.0, 1.0)
        side = 1.0 / (1.0 + np.exp(-P[below, 0] / (0.06 * H)))
        W[below] = 0.0
        W[below, bi["Pelvis"]] = 1.0 - 0.55 * t
        W[below, bi["LeftUpperLeg"]] = 0.55 * t * side
        W[below, bi["RightUpperLeg"]] = 0.55 * t * (1.0 - side)
        feet = P[below, 1] < 0.06 * H
        for sd, sg in (("Left", 1.0), ("Right", -1.0)):
            f = below[feet & (P[below, 0] * sg > 0)]
            W[f] = 0.0
            W[f, bi[f"{sd}Foot"]] = 1.0
    # smooth over the surface (welded adjacency) so joints bend softly
    ids, U = weld(P, faces)
    Wu = np.zeros((len(U), W.shape[1]))
    np.add.at(Wu, ids, W)
    cnt = np.bincount(ids, minlength=len(U)).astype(float)
    Wu /= cnt[:, None]
    e = np.concatenate([ids[faces[:, [0, 1]]], ids[faces[:, [1, 2]]], ids[faces[:, [2, 0]]]])
    e = np.concatenate([e, e[:, ::-1]])
    from scipy.sparse import coo_matrix
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(U), len(U))).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    for _ in range(cfg.get("smooth", 6)):
        Wu = 0.5 * Wu + 0.5 * (A @ Wu) / np.maximum(deg[:, None], 1)
    W = Wu[ids]
    W /= W.sum(axis=1, keepdims=True)
    top = np.argsort(-W, axis=1)[:, :4]
    tw = np.take_along_axis(W, top, axis=1)
    tw[tw < 0.03] = 0.0
    tw /= np.maximum(tw.sum(axis=1, keepdims=True), 1e-12)
    joints = np.array([idx[bones[k]] for k in range(len(bones))])[top]
    return joints.astype(np.uint16), tw.astype(np.float32)


def restrict_weights(joints, weights, order, mask, allowed, fallback):
    """Keep only the ``allowed`` bones on the masked vertices (renormalised; ``fallback`` when
    nothing is left)."""
    joints, weights = joints.copy(), weights.copy()
    ok = np.isin(np.array(order)[joints.astype(int)], allowed) & mask[:, None]
    w = np.where(ok, weights, 0.0)
    tot = w.sum(axis=1)
    empty = mask & (tot < 1e-6)
    w[empty] = 0.0
    w[empty, 0] = 1.0
    joints[empty, 0] = order.index(fallback)
    w[mask & ~empty] /= tot[mask & ~empty, None]
    weights[mask] = w[mask]
    return joints, weights.astype(np.float32)


def cut_bridges(F, P, UV, joints, weights, ids, order, par, kind="biped"):
    dom = joints[np.arange(len(joints)), np.argmax(weights, axis=1)].astype(int)
    n = len(order)
    depth = {}

    def chain(i):
        out = [i]
        while par.get(order[out[-1]]):
            out.append(order.index(par[order[out[-1]]]))
        return out
    dist = np.zeros((n, n), int)
    chains = [chain(i) for i in range(n)]
    for a in range(n):
        for b in range(n):
            ca, cb = chains[a], chains[b]
            common = next(x for x in ca if x in cb)
            dist[a, b] = ca.index(common) + cb.index(common)
    distal = np.array([("Hand" in order[i] and kind == "biped") for i in range(n)])
    del depth
    d = dom[F]
    bad = np.zeros(len(F), bool)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        dd = dist[d[:, a], d[:, b]]
        bad |= (distal[d[:, a]] | distal[d[:, b]]) & (dd > 1)       # a hand / foot fused to something else
    if not bad.any():
        return F, P, UV, joints, weights, ids
    F = F[~bad]
    used = np.unique(F)
    remap = np.full(len(P), -1, np.int64)
    remap[used] = np.arange(len(used))
    return remap[F], P[used], UV[used], joints[used], weights[used], ids[used]


# ------------------------------------------------------------------ build
def grade(tex: np.ndarray, g: tuple) -> np.ndarray:
    """TripoSR's colour field comes out a little dull: lift brightness / saturation."""
    bright, sat = g
    t = tex.astype(np.float32) / 255.0
    lum = (t @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    t = np.clip(lum + (t - lum) * sat, 0.0, 1.0)
    t = 1.0 - (1.0 - t) ** bright if bright >= 1 else t * bright
    return (np.clip(t, 0, 1) * 255 + 0.5).astype(np.uint8)


def load_source(src: str, cfg: dict) -> "Part":
    """A reconstructed source, oriented into the build frame (+Y up, facing +Z, feet on Y = 0, the
    mirror plane at x = 0, height 1)."""
    from PIL import Image
    from head_graft import Part
    z = np.load(MESHES / f"{src}.npz")
    V0, F, UV = z["verts"].astype(np.float64), z["faces"].astype(np.int64), z["uvs"].astype(np.float32)
    tex = np.array(Image.open(MESHES / f"{src}.png").convert("RGB"))
    fpath = MESHES / f"{src}_fmap.npz"
    fmap = None
    if fpath.exists():
        fmap = np.load(fpath)["fmap"].astype(np.float32)
    else:
        import reconstruct_3d
        if reconstruct_3d.SOURCES.get(src, (None,) * 5)[4]:
            fmap = reconstruct_3d.front_map(src)            # rebuilt from the mesh + the cut-out
    P = zup_to_yup(V0)
    if cfg.get("mirror"):
        # the sheet view shows the character's left side with the head turned to the viewer; the
        # companion is seen from its right side, so mirror it (the characters are symmetric)
        P[:, 0] *= -1.0
        F = F[:, ::-1].copy()
    ids, U = weld(P, F)
    if "yaw" in cfg:
        yaw = float(cfg["yaw"])              # a front view: TripoSR's input camera faces it exactly
    else:
        yaw = find_facing(U, cfg.get("facing", 0)) + cfg.get("yaw_adjust", 0.0)
    R = ry(yaw)
    if cfg["kind"] == "quadruped" and "yaw" not in cfg:
        # the head end is the front: the highest points (head, ears) must be on the +Z side
        Ur = U @ R.T
        hi = Ur[:, 1] > np.percentile(Ur[:, 1], 92)
        if Ur[hi, 2].mean() < 0.5 * (Ur[:, 2].min() + Ur[:, 2].max()):
            yaw = (yaw + 180.0) % 360.0
            R = ry(yaw)
    P = P @ R.T
    off = np.array([0.5 * (P[:, 0].min() + P[:, 0].max()), P[:, 1].min(), 0.5 * (P[:, 2].min() + P[:, 2].max())])
    P = (P - off) / float((P - off)[:, 1].max())
    return Part(P, F, UV, tex, fmap, tag=src, extra={"yaw": yaw})


def graft_head(body, cfg: dict, hcfg: dict):
    """Put the detailed head (``hcfg["source"]``, a sitting front view) on a standing body; returns
    the composed (P, F, UV, tex, fmap). See tools/head_graft.py."""
    import head_graft as G
    ids, U = weld(body.P, body.F)
    S0 = Shape(U, ids[body.F])
    J0 = (quadruped_joints if cfg["kind"] == "quadruped" else biped_joints)(S0, cfg, (U, ids[body.F]))
    head = load_source(hcfg["source"], {"kind": "biped", "yaw": 270})
    piece, cut, ring = G.collar_cut(head, hcfg["collar_y"])
    body2, q, _n, old_h = G.remove_head(body, J0, hcfg.get("t", 0.45))
    # recolour the old body to the detailed sheet's palette (its own body, under the collar)
    from PIL import Image
    img = np.array(Image.open(ROOT / "art" / "cells" / "v2" / f"{hcfg['sheet']}_front.png").convert("RGBA"))
    below = img[int(hcfg["collar_y"]) + 40:]
    if hcfg.get("recolor", 0.85) > 0:
        body2.tex = G.recolor(body2.tex, below[..., :3][below[..., 3] > 250], hcfg.get("recolor", 0.85))
    c = q - np.array([0.0, 0.03, 0.0])                     # the collar line on the body
    placed = G.place_head(piece, ring, cut, q + np.array([0.0, 0.0, hcfg.get("dz", 0.0)]),
                          hcfg.get("scale", 1.15) * old_h, sink=hcfg.get("sink", 0.02),
                          pitch=hcfg.get("pitch", 0.0))
    if cfg["kind"] == "quadruped":         # the chest fur cut with an animal head hugs the neck
        placed = G.taper_collar(placed, body2, c, 0.12 * hcfg.get("scale", 1.15) * old_h, slack=1.0)
    placed.density = hcfg.get("density", 2.5)
    # close both cuts: no background through the seam, whatever the angle
    if cfg["kind"] == "quadruped":         # (a person's head cut runs through long hair: a cap would
        placed = G.cap_holes(placed)       # show as a plate; the body's collar / hood hides it instead)
    body2 = G.cap_holes(body2, max_extent=0.8)
    rx, rz = G.neck_radii(body2, c)
    # the collar hugs the neck, but never wider than the head's own collar line would be
    hw = 0.5 * float(np.ptp(placed.P[:, 0]))
    rx, rz = min(rx, 0.42 * hw), min(rz, 0.42 * hw)
    parts = [body2, placed]
    if hcfg.get("collar_rgb"):
        parts.append(G.collar_ring(c, rx * 1.04, rz * 1.04, hcfg.get("collar_tube", 0.014), hcfg["collar_rgb"]))
    if hcfg.get("charm_rgb"):
        parts.append(G.ball(c + np.array([0.0, -0.035, rz * 1.04 + 0.012]), 0.028, hcfg["charm_rgb"]))
    P, F, UV, tex, fmap, pid = G.compose(parts, 2048)
    # the body keeps its own skeleton (found with its own head on); the head joints come from the
    # new head: the pivot just above the collar, the top, and the ears (the highest point per side)
    J = {k: np.asarray(v, float).copy() for k, v in J0.items() if "Ear" not in k}
    hp = placed.P
    hh = float(hp[:, 1].max() - c[1])
    J["Head"] = np.array([0.0, c[1] + 0.15 * hh, c[2]])
    centre = np.array([0.0, c[1] + 0.45 * hh, float(np.median(hp[:, 2]))])
    if cfg["kind"] == "biped":
        J["Neck"] = 0.5 * (J["Chest"] + J["Head"]) if J["Neck"][1] >= J["Head"][1] else J["Neck"]
    else:
        J["Neck"] = 0.5 * (J["Chest"] + J["Head"])
    J["HeadTop"] = np.array([0.0, float(hp[:, 1].max()), centre[2]])
    if cfg.get("ears"):
        for side, sg in (("Left", 1.0), ("Right", -1.0)):
            e = hp[(hp[:, 0] * sg > 0.15 * hh) & (hp[:, 1] > centre[1])]
            if len(e) < 30:
                continue
            tip = e[e[:, 1] > np.percentile(e[:, 1], 96)].mean(axis=0)
            J[f"{side}Ear"] = centre + cfg.get("ear_base", 0.55) * (tip - centre)
            J[f"{side}EarTip"] = tip
            J["HeadTop"] = np.array([0.0, centre[1] + 0.5 * hh, centre[2]])     # the skull, not an ear
    # the bigger head made the model taller: back to height 1
    k = 1.0 / float(P[:, 1].max())
    return P * k, F, UV, tex, fmap, {n: v * k for n, v in J.items()}, pid


def build(cid: str) -> dict:
    from focusbuddy.companion.gltf import GLBWriter
    cfg = CHARS[cid]
    src = load_source(cfg.get("mesh", cid), cfg)
    yaw = src.extra["yaw"]
    J = pid = None
    if cfg.get("graft"):
        P, F, UV, tex, fmap, J, pid = graft_head(src, cfg, cfg["graft"])
    else:
        P, F, UV, tex, fmap = src.P, src.F, src.UV, src.tex, src.fmap
    # TripoSR's colour field comes out dull: lift it, but not the texels painted from the sheet
    graded = grade(tex, cfg.get("grade", (1.1, 1.12)))
    if fmap is not None:
        w = np.clip(fmap[..., 2:3], 0, 1)
        tex = (graded * (1 - w) + tex * w + 0.5).astype(np.uint8)
    else:
        tex = graded
    out = ASSETS / cid
    out.mkdir(parents=True, exist_ok=True)
    face_meta = {"mode": "none"}
    if cfg.get("graft") and fmap is not None:
        # facial expressions from the detailed sheet's panels (neutral baked into the texture)
        import face_expressions
        from build_detail_sheets import EXPR_NAMES
        sheet = cfg["graft"]["sheet"]
        tex, face_meta = face_expressions.build(sheet, EXPR_NAMES[sheet], fmap, tex, out)
    elif (out / "face").exists():
        import shutil
        shutil.rmtree(out / "face")
    ids, U = weld(P, F)
    S = Shape(U, ids[F])
    if J is None:
        J = (biped_joints if cfg["kind"] == "biped" else quadruped_joints)(S, cfg, (U, ids[F]))
    order, par, seg = hierarchy(J)
    ti_ = np.clip((UV * tex.shape[0]).astype(int), 0, tex.shape[0] - 1)
    vcol = tex[ti_[:, 1], ti_[:, 0]].astype(np.float64) / 255.0
    band = torso_band(S, J, S.H) if cfg["kind"] == "biped" else None
    joints, weights = skin_weights(P, F, J, order, par, seg, S.H, cfg, vcol, band)
    if pid is not None:
        # the grafted head is one rigid piece: the head and its ears only (the collar follows the neck)
        joints, weights = restrict_weights(joints, weights, order, pid == 1, ("Head", "LeftEar", "RightEar",
                                                                               "LeftEarTip", "RightEarTip"), "Head")
        joints, weights = restrict_weights(joints, weights, order, pid >= 2, ("Neck",), "Neck")
    # contact bridges: where the reconstruction fused a hand to the hip, the triangles between them
    # stretch into a sheet when the arm moves; cut them (the joints of one chain stay connected)
    F, P, UV, joints, weights, ids = cut_bridges(F, P, UV, joints, weights, ids, order, par, cfg["kind"])
    # normals on the welded surface (no shading seams at UV seams)
    Fu = ids[F]
    Nu = vertex_normals(U, Fu)
    N = Nu[ids]
    bad = np.linalg.norm(N, axis=1) < 0.5             # degenerate (a cap's centre): point outward
    if bad.any():
        d = P[bad] - P.mean(axis=0)
        N[bad] = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    # breath morph: chest / belly along the normals
    torso = np.isin(np.array(order)[joints[:, 0]], ["Spine", "Chest", "Pelvis"]) * weights[:, 0]
    breath = (N * (0.012 * torso)[:, None]).astype(np.float32)
    w = GLBWriter("FocusBuddy build_3d_companions.py (TripoSR reconstruction)")
    node = {}
    for n in order:
        p = par[n]
        rel = J[n] - (J[p] if p else np.zeros(3))
        node[n] = w.add_node(n, translation=tuple(rel), parent=node.get(p), root=p is None)
    ibm = np.repeat(np.eye(4)[None], len(order), axis=0)
    for i, n in enumerate(order):
        ibm[i, :3, 3] = -J[n]
    skin = w.add_skin([node[n] for n in order], ibm, node["Root"])
    ti = w.add_texture(tex, "image/jpeg") if cfg.get("jpeg", True) else w.add_texture(tex)
    shading = cfg.get("shading", "cloth")
    mi = w.add_material(f"{cid}_skin", (1, 1, 1, 1), 0.0, 0.45 if shading == "plastic" else 0.85,
                        extras={"shading": shading}, base_tex=ti)
    mesh = w.add_mesh(cid, P.astype(np.float32), F.astype(np.uint32), N.astype(np.float32), None, joints, weights,
                      mi, [{"POSITION": breath}], ["breath"], uvs=UV)
    w.add_node(f"{cid}_mesh", mesh=mesh, skin=skin, root=True)
    w.save(out / "model.glb", extras={"focusbuddy": {"id": cid, "source": "TripoSR reconstruction"}})
    for stale in ("model_low.glb", "model_medium.glb", "model_ultra.glb", "LICENSE.txt"):
        if (out / stale).exists():
            (out / stale).unlink()
    bone_map = {n: n for n in order if n not in TIPS and n != "HeadTop"}
    meta = {
        "id": cid, "kind": cfg["kind"], "species": cfg["species"], "production": True,
        "source": "triposr",
        "production_note": f"3D model reconstructed from the {cfg['name']} character sheet with TripoSR "
                           "(image-to-3D), auto-rigged and skinned by tools/build_3d_companions.py; animated by "
                           "the procedural rig (no baked clips).",
        "license": "Model reconstructed with TripoSR (MIT) from the project's own character art. See LICENSE.md.",
        "bone_map": bone_map,
        "face": face_meta,
        "morph_map": {"breath": {"param": "breath", "scale": 1.0}},
        "normals": "asset",
        "pelvis_lift": "auto",
        "transitions": {"default": 0.25, "idle->loco": 0.25, "loco->idle": 0.3},
    }
    meta.update(cfg.get("meta", {}))
    if cfg["kind"] == "quadruped":
        meta.setdefault("tail_gain", 0.5)
    else:                                   # hands held a little away from the body: they read
        meta.setdefault("arm_out", 0.12)
        meta.setdefault("arm_fwd", 0.1)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if cfg.get("rig_tail"):
        (out / "skeleton.json").write_text(json.dumps({"tail": cfg["rig_tail"]}, indent=2), encoding="utf-8")
    elif (out / "skeleton.json").exists():
        (out / "skeleton.json").unlink()
    (out / "LICENSE.md").write_text(LICENSE.format(name=cfg["name"]), encoding="utf-8")
    return {"id": cid, "yaw": round(yaw, 1), "verts": len(P), "tris": len(F), "bones": len(order),
            "joints": {k: [round(float(x), 3) for x in v] for k, v in J.items()}}


TRIPOSR_BUILT = ("robot",)


def main(argv: list[str]) -> int:
    preview = None
    if "--preview" in argv:
        i = argv.index("--preview")
        preview = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    # the other companions are generated with Hunyuan3D and rigged by tools/rig/make.py now: by
    # default only the robot is rebuilt here (name an id to rebuild an old reconstruction anyway)
    ids = argv or [c for c in CHARS if c in TRIPOSR_BUILT and (MESHES / f"{c}.npz").exists()]
    for cid in ids:
        r = build(cid)
        print(f"{cid}: yaw {r['yaw']}  {r['verts']} verts  {r['tris']} tris  {r['bones']} bones")
    if preview:
        from tools.contact_sheet import main as sheet  # noqa: F401 - documented companion tool
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
