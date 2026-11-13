"""Bake ambient occlusion for the companion models (2.2 "refined models").

For every level of detail of a character, each vertex's ambient occlusion is estimated by
looking at the bind-pose mesh from many directions (a Fibonacci sphere): a depth map is rendered
along each direction, and the vertex counts as lit from there when nothing of the model lies in
front of it. Directions below the horizon are mostly blocked by the floor the companion stands
on. The result is weighted by the vertex normal (cosine), smoothed over the mesh a little, and
written to ``refine.npz`` next to the model:

    ao_<lod>   (N,) float16   0 = fully occluded .. 1 = open

At load time ``CharacterAsset`` darkens the creases with it (under the chin, inside the ears,
between the legs, the folds of clothes), which gives the models much more depth.

    python -m tools.refine.bake_ao cat dog ...        (no names = every built-in companion)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUILTIN = ("cat", "dog", "fox", "rabbit", "robot", "buddy", "study_girl", "saree_girl")


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], axis=1)


def _basis(d: np.ndarray):
    """Orthonormal (u, v) spanning the plane perpendicular to view direction d."""
    a = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, d)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v


def depth_map(P2: np.ndarray, z: np.ndarray, F: np.ndarray, res: int):
    """Z-buffer (larger z = nearer the viewer) of triangles F over 2D points P2 in [0, res)."""
    depth = np.full(res * res, -np.inf, dtype=np.float32)
    x, y = P2[:, 0], P2[:, 1]
    x0, x1, x2 = x[F[:, 0]], x[F[:, 1]], x[F[:, 2]]
    y0, y1, y2 = y[F[:, 0]], y[F[:, 1]], y[F[:, 2]]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    ok = np.abs(area) > 1e-9
    F, x0, x1, x2, y0, y1, y2, area = F[ok], x0[ok], x1[ok], x2[ok], y0[ok], y1[ok], y2[ok], area[ok]
    xmin = np.clip(np.floor(np.minimum(np.minimum(x0, x1), x2)).astype(np.int64), 0, res - 1)
    xmax = np.clip(np.ceil(np.maximum(np.maximum(x0, x1), x2)).astype(np.int64), 0, res - 1)
    ymin = np.clip(np.floor(np.minimum(np.minimum(y0, y1), y2)).astype(np.int64), 0, res - 1)
    ymax = np.clip(np.ceil(np.maximum(np.maximum(y0, y1), y2)).astype(np.int64), 0, res - 1)
    bw, bh = xmax - xmin + 1, ymax - ymin + 1
    cnt = bw * bh
    tri = np.repeat(np.arange(len(F)), cnt)
    local = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    cx = xmin[tri] + local % bw[tri]
    cy = ymin[tri] + local // bw[tri]
    px, py = cx + 0.5, cy + 0.5
    ax, ay = x0[tri], y0[tri]
    d = area[tri]
    b1 = ((px - ax) * (y2[tri] - ay) - (py - ay) * (x2[tri] - ax)) / d
    b2 = ((x1[tri] - ax) * (py - ay) - (y1[tri] - ay) * (px - ax)) / d
    b0 = 1 - b1 - b2
    ins = (b0 >= -1e-4) & (b1 >= -1e-4) & (b2 >= -1e-4)
    f = F[tri[ins]]
    zz = (b0[ins] * z[f[:, 0]] + b1[ins] * z[f[:, 1]] + b2[ins] * z[f[:, 2]]).astype(np.float32)
    np.maximum.at(depth, cy[ins] * res + cx[ins], zz)
    return depth


def bake(pos: np.ndarray, nrm: np.ndarray, F: np.ndarray, n_dirs: int = 72, res: int = 320,
         floor_block: float = 0.75) -> np.ndarray:
    """Per-vertex ambient occlusion (0..1) of a mesh. Internal frame: +y is down (floor below)."""
    N = len(pos)
    span = float(np.max(np.ptp(pos, axis=0)))
    bias = span * 0.006
    dirs = fibonacci_sphere(n_dirs)
    vis_sum = np.zeros(N)
    w_sum = np.zeros(N)
    nn = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    for d in dirs:
        w = np.clip(nn @ d, 0.0, None)                  # cosine weight (light arriving from d)
        if not np.any(w > 0):
            continue
        u, v = _basis(d)
        pu, pv, pz = pos @ u, pos @ v, pos @ d          # viewer sits along +d: larger pz = nearer
        lo_u, lo_v = pu.min(), pv.min()
        s = (res - 2) / max(pu.max() - lo_u, pv.max() - lo_v, 1e-6)
        P2 = np.stack([(pu - lo_u) * s + 1, (pv - lo_v) * s + 1], axis=1)
        dm = depth_map(P2, pz, F, res)
        ix = np.clip(P2[:, 0].astype(np.int64), 0, res - 1)
        iy = np.clip(P2[:, 1].astype(np.int64), 0, res - 1)
        # sample a 2x2 neighbourhood and keep the farthest blocker (robust against raster noise)
        best = dm[iy * res + ix]
        for ox, oy in ((1, 0), (0, 1), (1, 1)):
            j = np.clip(iy + oy, 0, res - 1) * res + np.clip(ix + ox, 0, res - 1)
            best = np.minimum(best, np.where(np.isfinite(dm[j]), dm[j], best))
        vis = (pz >= best - bias).astype(float)
        if d[1] > 0.05:                                 # pointing down into the floor
            vis *= 1.0 - floor_block * min(1.0, (d[1] - 0.05) / 0.4)
        vis_sum += vis * w
        w_sum += w
    ao = np.where(w_sum > 1e-9, vis_sum / np.maximum(w_sum, 1e-9), 1.0)
    return smooth(ao, F, 2)


def smooth(values: np.ndarray, F: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Average each vertex with its mesh neighbours (removes sampling noise)."""
    N = len(values)
    v = values.astype(float)
    a = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    b = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    deg = np.bincount(a, minlength=N).astype(float)
    for _ in range(iterations):
        acc = np.bincount(a, weights=v[b], minlength=N)
        v = 0.5 * v + 0.5 * np.where(deg > 0, acc / np.maximum(deg, 1), v)
    return v


def bake_character(cid: str, n_dirs: int = 72, res: int = 320) -> Path:
    from focusbuddy.companion.character_asset import CharacterAsset, asset_dir
    folder = asset_dir(cid)
    a = CharacterAsset.from_folder(cid, folder, refine="geometry")      # re-sculpted shapes, original vertices
    out = {}
    for lod, m in a.lods.items():
        t0 = time.time()
        # welded positions: generated models are often split at UV seams; bake on positions so
        # both sides of a seam get the same value
        key = np.round(m.positions / 1e-4).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        nw = int(inv.max()) + 1
        P = np.zeros((nw, 3)); Nn = np.zeros((nw, 3)); c = np.zeros(nw)
        np.add.at(P, inv, m.positions); np.add.at(Nn, inv, m.normals); np.add.at(c, inv, 1)
        P /= c[:, None]
        Fw = inv[m.indices]
        ao_w = bake(P, Nn, Fw, n_dirs, res)
        out[f"ao_{lod}"] = ao_w[inv].astype(np.float16)
        print(f"{cid:11s} {lod:6s} {len(m.positions):7d} verts  ao mean {ao_w.mean():.2f}  {time.time() - t0:5.1f}s",
              flush=True)
    path = folder / "refine.npz"
    old = {}
    if path.is_file():
        with np.load(path) as z:
            old = {k: z[k] for k in z.files}
    old.update(out)
    np.savez_compressed(path, **old)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--dirs", type=int, default=72)
    ap.add_argument("--res", type=int, default=320)
    a = ap.parse_args(argv)
    for cid in a.ids or BUILTIN:
        print(bake_character(cid, a.dirs, a.res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
