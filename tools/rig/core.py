"""Shared helpers for rigging the companion models (plain NumPy; no Blender needed).

Frame used by every rigging stage ("rig frame"): Blender's - +Z up, the character faces -Y, its
left side is +X; feet on Z = 0, the body centred on X = 0, height 1.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "rig"
SOURCE = ROOT / "art" / "source_glb"

# per character: kind, yaw (deg about +Z that turns the imported model to face -Y)
CHARS = {
    "cat": {"kind": "quadruped", "species": "cat", "yaw": 90.0, "name": "Cat"},
    "dog": {"kind": "quadruped", "species": "dog", "yaw": 90.0, "name": "Dog"},
    "fox": {"kind": "quadruped", "species": "fox", "yaw": 90.0, "name": "Fox"},
    "rabbit": {"kind": "quadruped", "species": "rabbit", "yaw": 90.0, "name": "Rabbit"},
    "buddy": {"kind": "biped", "species": "human", "yaw": 0.0, "name": "Study Buddy"},
    "study_girl": {"kind": "biped", "species": "human", "yaw": 0.0, "name": "Study Girl"},
    "saree_girl": {"kind": "biped", "species": "human", "yaw": 0.0, "name": "Saree Girl"},
}


def rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ground_blobs(V: np.ndarray, z_max: float, cell: float = 0.01) -> list:
    """Connected blobs (XY) of the points below ``z_max``, largest first, as point arrays."""
    from scipy import ndimage
    g = V[V[:, 2] < z_max]
    lo = g[:, :2].min(0)
    ij = ((g[:, :2] - lo) / cell).astype(int)
    grid = np.zeros(ij.max(0) + 2, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    grid = ndimage.binary_closing(grid, iterations=1) | grid
    lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
    ids = lab[ij[:, 0], ij[:, 1]]
    return sorted([g[ids == k] for k in range(1, n + 1)], key=len, reverse=True)


def body_yaw(V: np.ndarray) -> float:
    """Yaw (deg) that turns a quadruped to face -Y: from its four paws (the two nearest the head
    are the front pair)."""
    H = float(V[:, 2].max() - V[:, 2].min())
    V = V - np.array([0, 0, V[:, 2].min()])
    blobs = ground_blobs(V, 0.03 * H)
    big = [b for b in blobs if len(b) > 0.3 * len(blobs[0])]
    head = V[V[:, 2] > 0.85 * H][:, :2].mean(0)
    paws = sorted(big, key=lambda b: np.linalg.norm(b[:, :2].mean(0) - head))
    if len(paws) < 4:
        raise RuntimeError("could not find four paws")
    # the four paws nearest to forming two pairs: the two nearest the head are the front pair;
    # a tail tip resting on the ground is farther than all four
    cand = paws[:5]
    import itertools
    best = None
    for four in itertools.combinations(range(len(cand)), 4):
        c = np.array([cand[i][:, :2].mean(0) for i in four])
        d = np.linalg.norm(c - head, axis=1)
        o = np.argsort(d)
        front, back = c[o[:2]].mean(0), c[o[2:]].mean(0)
        # a body: front pair and back pair about equally wide, legs parallel
        w1, w2 = np.linalg.norm(c[o[0]] - c[o[1]]), np.linalg.norm(c[o[2]] - c[o[3]])
        score = abs(w1 - w2) + 0.5 * abs(np.linalg.norm(front - back) - 0.0) * 0 + d.sum() * 0.1
        if best is None or score < best[0]:
            best = (score, front, back)
    _, front, back = best
    d = front - back
    ang = math.degrees(math.atan2(d[1], d[0]))
    return (270.0 - ang) % 360.0


class Model:
    """A prepared model in the rig frame."""

    def __init__(self, cid: str):
        from PIL import Image
        self.cid = cid
        self.cfg = CHARS[cid]
        z = np.load(BUILD / f"{cid}_prep.npz")
        V = z["V"]
        yaw = self.cfg["yaw"] if self.cfg["kind"] == "biped" else self.cfg.get("yaw_fixed", body_yaw(V))
        self.yaw = yaw
        V = V @ rz(yaw).T
        lo, hi = V.min(0), V.max(0)
        V = V - np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), lo[2]])
        if self.cfg["kind"] == "quadruped":
            # the mirror plane between the paws (a tail curled to one side moves the bounding box)
            Hh = float(V[:, 2].max())
            blobs = ground_blobs(V, 0.03 * Hh)
            big = [b for b in blobs if len(b) > 0.3 * len(blobs[0])][:5]
            head = V[V[:, 2] > 0.85 * Hh][:, :2].mean(0)
            big = sorted(big, key=lambda b: np.linalg.norm(b[:, :2].mean(0) - head))[:4]
            V[:, 0] -= float(np.mean([b[:, 0].mean() for b in big]))
        self.scale = 1.0 / float(V[:, 2].max())
        self.V = V * self.scale
        F, FUV = z["F"], z["FUV"]
        # faces on edges shared by more (or fewer) than two faces break Blender's bone-heat solve:
        # drop them (a pin-hole; the generator leaves a handful at most)
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        _, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
        bad = (cnt[inv.reshape(3, -1).T] > 2).any(1)
        self.F, self.FUV = F[~bad], FUV[~bad]
        self.tex = np.asarray(Image.open(BUILD / f"{cid}_base.png").convert("RGB"))
        self._vcol = None
        self._n = None

    @property
    def H(self) -> float:
        return 1.0

    @property
    def vcol(self) -> np.ndarray:
        """Per-vertex texture colour (mean over the vertex's corners), 0..1."""
        if self._vcol is None:
            T = self.tex.shape[0]
            uv = self.FUV.reshape(-1, 2)
            px = self.tex[np.clip((uv[:, 1] * T).astype(int), 0, T - 1), np.clip((uv[:, 0] * T).astype(int), 0, T - 1)]
            acc = np.zeros((len(self.V), 3))
            np.add.at(acc, self.F.reshape(-1), px.astype(np.float64))
            cnt = np.bincount(self.F.reshape(-1), minlength=len(self.V))[:, None]
            self._vcol = acc / np.maximum(cnt, 1) / 255.0
        return self._vcol

    @property
    def normals(self) -> np.ndarray:
        if self._n is None:
            V, F = self.V, self.F
            fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
            n = np.zeros_like(V)
            for k in range(3):
                np.add.at(n, F[:, k], fn)
            self._n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return self._n


# ------------------------------------------------------------------ quick views with joints
VIEWS = {"front": (0, 2, -1.0), "side": (1, 2, 1.0), "top": (0, 1, 1.0), "back": (0, 2, 1.0)}


def raster(V, F, colors, axes, size=700, depth_axis=None, depth_sign=1.0, box=None):
    """Z-buffered flat-colour triangle raster (orthographic). ``axes`` = (u axis, v axis, u sign);
    returns (image, (lo, scale)) for placing overlays."""
    ua, va, us = axes
    da = depth_axis if depth_axis is not None else ({0, 1, 2} - {ua, va}).pop()
    U = V[:, ua] * us
    W = V[:, va]
    D = V[:, da] * depth_sign
    if box is None:
        lo = np.array([U.min(), W.min()])
        span = max(U.max() - U.min(), W.max() - W.min()) * 1.04
        lo = lo - 0.02 * span
    else:
        lo, span = box
    sc = size / span
    X = (U - lo[0]) * sc
    Y = size - (W - lo[1]) * sc
    zb = np.full((size, size), -np.inf)
    img = np.full((size, size, 3), 235.0)
    tri = F
    xs, ys, zs = X[tri], Y[tri], D[tri]
    x0 = np.clip(np.floor(xs.min(1)).astype(int), 0, size - 1)
    x1 = np.clip(np.ceil(xs.max(1)).astype(int), 0, size - 1)
    y0 = np.clip(np.floor(ys.min(1)).astype(int), 0, size - 1)
    y1 = np.clip(np.ceil(ys.max(1)).astype(int), 0, size - 1)
    for k in np.argsort(zs.mean(1)):
        if x1[k] < x0[k] or y1[k] < y0[k]:
            continue
        gy, gx = np.mgrid[y0[k]:y1[k] + 1, x0[k]:x1[k] + 1] + 0.5
        a, b, c = xs[k], ys[k], zs[k]
        d = (b[1] - b[2]) * (a[0] - a[2]) + (a[2] - a[1]) * (b[0] - b[2])
        if abs(d) < 1e-12:
            continue
        l1 = ((b[1] - b[2]) * (gx - a[2]) + (a[2] - a[1]) * (gy - b[2])) / d
        l2 = ((b[2] - b[0]) * (gx - a[2]) + (a[0] - a[2]) * (gy - b[2])) / d
        l3 = 1 - l1 - l2
        ins = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        if not ins.any():
            continue
        zz = l1 * c[0] + l2 * c[1] + l3 * c[2]
        iy, ix = (gy[ins] - 0.5).astype(int), (gx[ins] - 0.5).astype(int)
        up = zz[ins] >= zb[iy, ix]
        zb[iy[up], ix[up]] = zz[ins][up]
        img[iy[up], ix[up]] = colors[k]
    return img, (lo, sc, size, axes)


def shaded_colors(model: Model, light=(0.3, -0.6, 0.75)) -> np.ndarray:
    V, F = model.V, model.F
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    L = np.array(light) / np.linalg.norm(light)
    sh = 0.55 + 0.45 * np.abs(fn @ L)
    col = model.vcol[F].mean(1) * 255
    return np.clip(col * 0.35 + 150 * 0.65, 0, 255) * sh[:, None]


def overlay(img, frame, joints: dict, parents: dict, labels=True):
    from PIL import Image, ImageDraw
    lo, sc, size, (ua, va, us) = frame
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)

    def p2(p):
        return ((p[ua] * us - lo[0]) * sc, size - (p[va] - lo[1]) * sc)
    for n, p in joints.items():
        par = parents.get(n)
        if par in joints:
            d.line([p2(joints[par]), p2(p)], fill=(20, 90, 255), width=2)
    for n, p in joints.items():
        x, y = p2(p)
        c = (230, 30, 30) if n.startswith("Left") else (20, 160, 40) if n.startswith("Right") else (0, 0, 0)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=c)
        if labels:
            d.text((x + 5, y - 6), n.replace("Left", "L").replace("Right", "R"), fill=(80, 0, 120))
    return pil


def save_json(path: Path, obj) -> None:
    def conv(o):
        if isinstance(o, np.ndarray):
            return [round(float(x), 5) for x in o]
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=1, default=conv), encoding="utf-8")
