"""Textured orthographic renders of a model (rig frame) with per-pixel 3D position / normal / face
id - used to find the painted eyes, brows and mouth precisely and map them back to the surface."""

from __future__ import annotations

import numpy as np


def ortho(V, F, FUV, tex, box, size=1024, view="front"):
    """``box`` = (u0, u1, v0, v1) in the view plane. front: camera at -Y looking +Y, u = x, v = z.
    Returns dict(img (S,S,3) uint8, pos (S,S,3), fid (S,S) int, nrm (S,S,3), scale, box)."""
    u0, u1, v0, v1 = box
    span = max(u1 - u0, v1 - v0)
    sc = size / span
    if view == "front":
        U, Vv, D = V[:, 0], V[:, 2], -V[:, 1]          # depth: larger = nearer the camera
    else:
        raise ValueError(view)
    X = (U - u0) * sc
    Y = size - (Vv - v0) * sc
    zb = np.full((size, size), -np.inf)
    fid = np.full((size, size), -1, np.int64)
    bar = np.zeros((size, size, 3))
    xs, ys, ds = X[F], Y[F], D[F]
    keep = (xs.max(1) >= 0) & (xs.min(1) < size) & (ys.max(1) >= 0) & (ys.min(1) < size)
    for k in np.flatnonzero(keep):
        a, b, d3 = xs[k], ys[k], ds[k]
        x0, x1 = int(max(np.floor(a.min()), 0)), int(min(np.ceil(a.max()), size - 1))
        y0, y1 = int(max(np.floor(b.min()), 0)), int(min(np.ceil(b.max()), size - 1))
        if x1 < x0 or y1 < y0:
            continue
        gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1] + 0.5
        den = (b[1] - b[2]) * (a[0] - a[2]) + (a[2] - a[1]) * (b[0] - b[2])
        if abs(den) < 1e-12:
            continue
        l1 = ((b[1] - b[2]) * (gx - a[2]) + (a[2] - a[1]) * (gy - b[2])) / den
        l2 = ((b[2] - b[0]) * (gx - a[2]) + (a[0] - a[2]) * (gy - b[2])) / den
        l3 = 1 - l1 - l2
        ins = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        if not ins.any():
            continue
        zz = l1 * d3[0] + l2 * d3[1] + l3 * d3[2]
        iy, ix = (gy[ins] - 0.5).astype(int), (gx[ins] - 0.5).astype(int)
        up = zz[ins] > zb[iy, ix]
        iy, ix = iy[up], ix[up]
        zb[iy, ix] = zz[ins][up]
        fid[iy, ix] = k
        bar[iy, ix] = np.stack([l1[ins][up], l2[ins][up], l3[ins][up]], -1)
    m = fid >= 0
    f = F[fid[m]]
    b = bar[m]
    pos = np.zeros((size, size, 3))
    pos[m] = sum(b[:, i, None] * V[f[:, i]] for i in range(3))
    uv = sum(b[:, i, None] * FUV[fid[m], i] for i in range(3))
    TH, TW = tex.shape[:2]
    img = np.full((size, size, 3), 255, np.uint8)
    img[m] = tex[np.clip((uv[:, 1] * TH).astype(int), 0, TH - 1), np.clip((uv[:, 0] * TW).astype(int), 0, TW - 1)]
    uvimg = np.full((size, size, 2), -1.0)
    uvimg[m] = uv
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    nrm = np.zeros((size, size, 3))
    nrm[m] = fn[fid[m]]
    return {"img": img, "pos": pos, "fid": fid, "nrm": nrm, "scale": sc, "box": box, "size": size, "uv": uvimg}


def to_px(r, x, z):
    u0, u1, v0, v1 = r["box"]
    return (x - u0) * r["scale"], r["size"] - (z - v0) * r["scale"]


def to_world(r, px, py):
    """Surface point under pixel (px, py) (nearest covered pixel)."""
    ix, iy = int(round(px)), int(round(py))
    ix = min(max(ix, 0), r["size"] - 1)
    iy = min(max(iy, 0), r["size"] - 1)
    if r["fid"][iy, ix] >= 0:
        return r["pos"][iy, ix], r["nrm"][iy, ix]
    ys, xs = np.nonzero(r["fid"] >= 0)
    k = np.argmin((xs - ix) ** 2 + (ys - iy) ** 2)
    return r["pos"][ys[k], xs[k]], r["nrm"][ys[k], xs[k]]
