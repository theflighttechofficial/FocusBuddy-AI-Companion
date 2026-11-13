"""Natural faces for the people (2.3): re-sculpt the chin and jaw and re-bake each face into one
continuous, retouched image.

Why: the people's heads were generated (Hunyuan3D) with the face cut into dozens of small texture
islands. Up close every island edge shows as a faint seam or a flat patch, hair colour from the
generator is smeared on the cheeks, and the weld between the generated head and the body left a
long sloping wedge under the chin that reads as a pointed jaw. Together that made the faces look
like stickers. This tool, run once per character (the result ships in ``refine.npz``):

1. **Chin and jaw** - the underside of the jaw is lifted toward the chin (so it runs back to the
   neck almost level, like a real jaw, instead of sloping down to a point), the chin is rounded a
   little, and the facial skin is lightly smoothed (Taubin smoothing: removes the generator's
   bumps without shrinking the nose or lips). New positions and normals for every level of detail.
2. **One face image** - the front of the head (cheek to cheek, forehead to chin) is mapped with a
   cylindrical projection onto a free corner of the texture (``face_rect``), and the old texture
   is sampled into it texel by texel, so the face becomes one continuous picture.
3. **Retouch** - the old island seams are blended away, smudges of hair colour on the skin are
   removed, and the skin is evened out with an edge-preserving filter that leaves the eyes, brows,
   lips and nostrils sharp; a soft natural blush, a little shading in the eye sockets and under the
   lower lip give the face depth that does not depend on the old paint.
4. **Painted face** - the blink / squint / open-mouth data (``face/painted.npz``) is rebuilt for
   the new image: the eye and mouth coordinates are exact linear maps of the 3D position (fitted
   to the old data), and the closed lid is a fresh, seamless fill of the skin around each eye.

    python -m tools.refine.faces buddy study_girl saree_girl      (then re-bake the occlusion:)
    python -m tools.refine.bake_ao buddy study_girl saree_girl
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

PEOPLE = ("buddy", "study_girl", "saree_girl")
TH = math.radians(80.0)          # half the angular width of the face image (cylindrical projection)
TH_FACE = math.radians(66.0)     # the face proper: cheek to cheek, in front of the ears


# ====================================================================== mesh helpers
def welded(P: np.ndarray):
    key = np.round(P / 1e-4).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.reshape(-1)


def head_weight(asset, m) -> np.ndarray:
    names = [getattr(j, "name", j) for j in asset.joints]
    hi = [i for i, n in enumerate(names) if str(n) in ("Head", "HeadTop")]
    w = np.zeros(len(m.positions))
    for k in range(4):
        for h in hi:
            w += m.weights[:, k] * (m.joints[:, k] == h)
    return w


def face_normals(P, F):
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    a = np.linalg.norm(fn, axis=1)
    return fn / np.maximum(a[:, None], 1e-12), a


def vertex_normals(P, F, inv, sign: float):
    """Smooth area-weighted outward normals, shared across welded duplicates (``sign`` = +1 when
    the winding's normals point outward, -1 when they point inward)."""
    fn, a = face_normals(P, F)
    acc = np.zeros((int(inv.max()) + 1, 3))
    for k in range(3):
        np.add.at(acc, inv[F[:, k]], fn * a[:, None])
    n = acc[inv]
    return sign * n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def neighbours(Fw, nw):
    """Vertex adjacency (welded ids) as a CSR pair."""
    e = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    e = np.unique(e, axis=0)
    order = np.argsort(e[:, 0], kind="stable")
    e = e[order]
    start = np.searchsorted(e[:, 0], np.arange(nw + 1))
    return start, e[:, 1]


def laplacian(X, start, nb, mask):
    """Umbrella operator (mean of neighbours - self) for the masked vertices."""
    deg = np.diff(start)
    s = np.zeros_like(X)
    np.add.at(s, np.repeat(np.arange(len(deg)), deg), X[nb])
    mean = s / np.maximum(deg, 1)[:, None]
    L = mean - X
    L[~mask] = 0.0
    return L


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ====================================================================== landmarks
def uv_to_pos(m, q: np.ndarray):
    """3D bind-pose points of texture coordinates ``q`` (K, 2) (first triangle that holds each)."""
    uv, F, P = m.uvs, m.indices, m.positions
    T = uv[F]
    out = np.full((len(q), 3), np.nan)
    a_, b_, c_ = T[:, 0], T[:, 1], T[:, 2]
    v0, v1 = c_ - a_, b_ - a_
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den[np.abs(den) < 1e-16] = 1e-16
    lo, hi = T.min(1), T.max(1)
    for i, p in enumerate(q):
        cand = np.flatnonzero((lo[:, 0] <= p[0]) & (p[0] <= hi[:, 0]) & (lo[:, 1] <= p[1]) & (p[1] <= hi[:, 1]))
        if not len(cand):
            continue
        v2 = p[None, :] - a_[cand]
        d02, d12 = (v0[cand] * v2).sum(1), (v1[cand] * v2).sum(1)
        u_ = (d11[cand] * d02 - d01[cand] * d12) / den[cand]
        w_ = (d00[cand] * d12 - d01[cand] * d02) / den[cand]
        ok = np.flatnonzero((u_ >= -1e-6) & (w_ >= -1e-6) & (u_ + w_ <= 1 + 1e-6))
        if not len(ok):
            continue
        k = ok[0]
        tri = F[cand[k]]
        out[i] = P[tri[0]] * (1 - u_[k] - w_[k]) + P[tri[2]] * u_[k] + P[tri[1]] * w_[k]
    return out


def texel_uv(idx, size):
    H, W = size
    return np.stack([(idx % W + 0.5) / W, (idx // W + 0.5) / H], 1)


def landmarks(m, pf, rng):
    eyes = []
    for k in np.unique(pf.e_eye):
        sel = np.flatnonzero(pf.e_eye == k)
        pick = rng.choice(sel, min(400, len(sel)), replace=False)
        p = uv_to_pos(m, texel_uv(pf.e_idx[pick], pf.size))
        ok = ~np.isnan(p[:, 0])
        r = np.hypot(pf.e_xy[pick, 0], pf.e_xy[pick, 1])
        ok &= r < 0.6
        eyes.append(np.median(p[ok], 0))
    pick = rng.choice(len(pf.m_idx), min(300, len(pf.m_idx)), replace=False)
    pm = uv_to_pos(m, texel_uv(pf.m_idx[pick], pf.size))
    ok = ~np.isnan(pm[:, 0]) & (np.abs(pf.m_xy[pick, 0]) < 0.4) & (np.abs(pf.m_xy[pick, 1]) < 0.15)
    mouth = np.median(pm[ok], 0) if ok.sum() > 5 else np.nanmedian(pm, 0)
    eyes.sort(key=lambda e: e[2])
    return eyes, mouth


# ====================================================================== 1. chin and jaw
def sculpt(m, w, eyes, mouth, sign, strength=1.0):
    """New positions (N, 3) for the high-detail bind pose: a level jaw underside, a rounder chin
    and lightly smoothed facial skin. +x is the face's forward direction, +y points down."""
    P = m.positions.astype(float).copy()
    E = (eyes[0] + eyes[1]) / 2
    iod = float(np.linalg.norm(eyes[0] - eyes[1]))
    cz = float(E[2])
    near = (np.abs(P[:, 2] - cz) < 0.35 * iod)
    # chin: the most forward point below the mouth, and where the front drops back to the neck
    below = near & (P[:, 1] > mouth[1] + 0.15 * iod) & (P[:, 1] < mouth[1] + 1.6 * iod) & (w > 0.3)
    chin_x = float(P[below, 0].max())
    prof = []
    for yy in np.arange(mouth[1], mouth[1] + 2.5 * iod, 0.05 * iod):
        band = near & (np.abs(P[:, 1] - yy) < 0.04 * iod)
        if band.sum() > 2:
            prof.append((yy, P[band, 0].max()))
    prof = np.array(prof)
    # the neck: the profile's deepest point between just under the mouth and the collar
    win = prof[(prof[:, 0] > mouth[1] + 0.3 * iod) & (prof[:, 0] < mouth[1] + 1.7 * iod)]
    j = int(np.argmin(win[:, 1]))
    neck_y, neck_x = float(win[j, 0]), float(win[j, 1])
    # chin bottom: where the profile falls below 75 % of the way from the neck to the chin
    lim = neck_x + 0.75 * (chin_x - neck_x)
    after = prof[(prof[:, 0] > mouth[1] + 0.2 * iod) & (prof[:, 1] < lim)]
    chin_y = float(after[0, 0]) if len(after) else float(mouth[1] + 0.6 * iod)
    depth = max(0.3 * iod, neck_y - chin_y)
    # head half-width at the mouth (cheek to cheek) and where the face's front ends
    band = (w > 0.5) & (np.abs(P[:, 1] - mouth[1]) < 0.3 * iod)
    hw = float(np.percentile(np.abs(P[band, 2] - cz), 98))
    lat = 1.0 - smoothstep(0.72 * hw, 1.02 * hw, np.abs(P[:, 2] - cz))
    front = smoothstep(neck_x - 0.6 * iod, neck_x - 0.1 * iod, P[:, 0])
    # --- a shorter lower face, as drawn in the character art: the chin moves up toward the mouth
    #     (the generated jaw is about a third too long), the jaw underside and the top of the neck
    #     follow and fade out down the neck
    s_ = 1.0 - 0.28 * strength
    shorten = (chin_y - mouth[1]) * (1.0 - s_)
    yy = P[:, 1]
    dy1 = np.where(yy <= mouth[1], 0.0, -(np.minimum(yy, chin_y) - mouth[1]) * (1.0 - s_))
    tail = 1.0 - smoothstep(chin_y, chin_y + 1.2 * depth, yy)
    dy1 = np.where(yy > chin_y, -shorten * tail, dy1)
    dy1 *= lat * front
    P1 = P.copy()
    P1[:, 1] += dy1
    chin_y1 = chin_y - shorten
    # --- the jaw underside: lifted toward the chin's level, most where it is furthest forward,
    #     so it runs back to the neck almost level instead of sloping down to a point
    fwd = smoothstep(neck_x - 0.05 * iod, chin_x - 0.05 * iod, P1[:, 0])
    t = (P1[:, 1] - chin_y1) / depth
    zone = (t > -0.25) & (t < 1.3) & (P1[:, 0] > neck_x - 0.2 * iod)
    k = 0.75 * strength * lat * fwd * smoothstep(-0.25, 0.1, t) * (1.0 - smoothstep(0.85, 1.3, t))
    dy = -(P1[:, 1] - chin_y1) * k
    dy[~zone] = 0.0
    dy = np.minimum(dy, 0.0)
    # --- a rounder chin (the narrow, pointed lower face widened a little)
    u = np.clip((P1[:, 1] - mouth[1]) / max(1e-3, chin_y1 - mouth[1]), 0.0, 1.3)
    widen = 0.10 * strength * smoothstep(0.2, 1.0, u) * (1.0 - smoothstep(1.0, 1.3, u))
    widen *= smoothstep(neck_x, chin_x, P1[:, 0])
    dz = (P1[:, 2] - cz) * widen * lat
    Pn = P1.copy()
    Pn[:, 1] += dy
    Pn[:, 2] += dz
    # --- the throat: the top of the neck comes forward and a little wider under the jaw, so the
    #     jaw meets the neck in a soft curve instead of narrowing to a point
    neck_y1 = neck_y - shorten * (1.0 - smoothstep(chin_y, chin_y + 1.2 * depth, neck_y))
    yy = Pn[:, 1]
    hump = smoothstep(chin_y1 - 0.1 * iod, chin_y1 + 0.45 * depth, yy) * \
        (1.0 - smoothstep(neck_y1 - 0.1 * depth, neck_y1 + 0.9 * iod, yy))
    nfront = smoothstep(neck_x - 1.0 * iod, neck_x - 0.2 * iod, Pn[:, 0]) * (Pn[:, 0] < chin_x - 0.3 * iod)
    nlat = 1.0 - smoothstep(0.55 * hw, 0.85 * hw, np.abs(Pn[:, 2] - cz))
    g = hump * nfront * nlat * strength
    Pn[:, 0] += 0.3 * (chin_x - neck_x) * g * smoothstep(neck_x + 0.6 * (chin_x - neck_x), neck_x - 0.4 * iod, Pn[:, 0])
    Pn[:, 2] += (Pn[:, 2] - cz) * 0.15 * g
    chin_y = chin_y1
    neck_x1 = neck_x
    # --- light Taubin smoothing on the facial skin (front of the head, below the hairline)
    inv = welded(P)
    nw = int(inv.max()) + 1
    Pw = np.zeros((nw, 3))
    cnt = np.zeros(nw)
    np.add.at(Pw, inv, Pn)
    np.add.at(cnt, inv, 1)
    Pw /= cnt[:, None]
    ww = np.zeros(nw)
    np.maximum.at(ww, inv, w)
    Fw = inv[m.indices]
    start, nb = neighbours(Fw, nw)
    th = np.arctan2(Pw[:, 2] - cz, Pw[:, 0] - (E[0] - 1.6 * iod))
    face = (ww > 0.6) & (np.abs(th) < math.radians(70)) & (Pw[:, 1] > E[1] - 1.1 * iod) & \
        (Pw[:, 1] < chin_y + 0.6 * iod)
    # keep the eyes and the lips crisp (only very light smoothing there)
    dE = np.minimum(np.linalg.norm(Pw - eyes[0], axis=1), np.linalg.norm(Pw - eyes[1], axis=1))
    dM = np.linalg.norm(Pw - mouth, axis=1)
    amt = face * (0.35 + 0.65 * smoothstep(0.35 * iod, 0.6 * iod, np.minimum(dE, dM * 1.3)))
    for _ in range(int(6 * strength)):
        Pw += 0.5 * amt[:, None] * laplacian(Pw, start, nb, face)
        Pw -= 0.53 * amt[:, None] * laplacian(Pw, start, nb, face)
    Pn = Pw[inv]
    info = dict(chin_x=chin_x, chin_y=chin_y, neck_x=neck_x, neck_y=neck_y, iod=iod, cz=cz, E=E,
                neck_y1=neck_y1, neck_x1=neck_x1 + 0.3 * (chin_x - neck_x) * 0.5)
    return Pn, info


def transfer(P_old_hi, P_new_hi, P_lod, radius):
    """Carry the high-detail displacement over to another level of detail (inverse-distance
    weighted over the nearest high-detail vertices)."""
    from scipy.spatial import cKDTree
    d = P_new_hi - P_old_hi
    moved = np.linalg.norm(d, axis=1) > 1e-7
    if not moved.any():
        return P_lod.copy()
    tree = cKDTree(P_old_hi)
    dist, idx = tree.query(P_lod, k=8)
    wgt = 1.0 / np.maximum(dist, 1e-4) ** 2
    wgt /= wgt.sum(1, keepdims=True)
    disp = (d[idx] * wgt[..., None]).sum(1)
    disp[dist[:, 0] > radius] = 0.0
    return P_lod + disp


def reskin(asset, m, P, info, strength=1.0):
    """Skin weights for the jaw and neck: the jaw underside follows the head completely and the
    neck blends Head -> Neck -> Chest down its length. (The generated head was joined to the body
    with the blend squeezed into the jaw underside, so every head turn sheared it into a flat
    wedge.) Returns (vertex ids, joints (K, 4), weights (K, 4))."""
    names = [str(getattr(j, "name", j)) for j in asset.joints]
    try:
        jh, jn, jc = names.index("Head"), names.index("Neck"), names.index("Chest")
    except ValueError:
        return np.zeros(0, np.int64), np.zeros((0, 4), np.int64), np.zeros((0, 4))
    iod, cz = info["iod"], info["cz"]
    chin_y, neck_y, neck_x = info["chin_y"], info["neck_y1"], info["neck_x1"]
    r_n = 0.75 * iod                                   # neck radius
    ax = neck_x - r_n                                  # neck axis (x)
    d = np.hypot(P[:, 0] - ax, P[:, 2] - cz)
    y_j = neck_y                                       # where the jaw underside meets the neck
    y_c = neck_y + 1.3 * iod                           # the collar
    hw = 1.9 * iod
    # jaw underside: all head
    under = (P[:, 1] > chin_y - 0.15 * iod) & (P[:, 1] <= y_j) & (P[:, 0] > ax - 0.2 * r_n) & \
        (np.abs(P[:, 2] - cz) < 0.95 * hw)
    under_w = (1.0 - smoothstep(0.65 * hw, 0.95 * hw, np.abs(P[:, 2] - cz))) * \
        smoothstep(chin_y - 0.15 * iod, chin_y + 0.1 * iod, P[:, 1])
    # the neck column
    neck = (P[:, 1] > y_j - 0.2 * iod) & (P[:, 1] < y_c) & (d < 1.25 * r_n)
    neck_w = 1.0 - smoothstep(1.0 * r_n, 1.25 * r_n, d)
    t = np.clip((P[:, 1] - y_j) / (y_c - y_j), 0.0, 1.0)
    wh = 1.0 - smoothstep(0.0, 0.65, t)
    wc = smoothstep(0.35, 1.0, t)
    wn = 1.0 - wh - wc
    J0, W0 = m.joints.astype(np.int64), m.weights.astype(float)
    # only vertices that the head / neck / chest carry today (never arms, hair ribbons ...)
    allowed = np.isin(J0, [jh, jn, jc, names.index("HeadTop") if "HeadTop" in names else jh]) | (W0 < 1e-4)
    ok = allowed.all(1)
    sel = np.flatnonzero(ok & (under | neck))
    Jn = np.zeros((len(sel), 4), np.int64)
    Wn = np.zeros((len(sel), 4))
    for r, v in enumerate(sel):
        old = {}
        for j, w_ in zip(J0[v], W0[v]):
            if w_ > 0:
                old[j] = old.get(j, 0.0) + w_
        if neck[v] and P[v, 1] > y_j:
            tgt = {jh: wh[v], jn: wn[v], jc: wc[v]}
            a_ = neck_w[v] * strength
        else:
            tgt = {jh: 1.0}
            a_ = under_w[v] * strength
        mix = {}
        for j, w_ in old.items():
            mix[j] = mix.get(j, 0.0) + w_ * (1 - a_)
        for j, w_ in tgt.items():
            mix[j] = mix.get(j, 0.0) + w_ * a_
        items = sorted(mix.items(), key=lambda kv: -kv[1])[:4]
        tot = sum(w_ for _, w_ in items) or 1.0
        for c, (j, w_) in enumerate(items):
            Jn[r, c] = j
            Wn[r, c] = w_ / tot
    return sel, Jn, Wn


# ====================================================================== 2. projection
class Projection:
    """Cylindrical coordinates around a vertical axis behind the face -> a texture rectangle."""

    def __init__(self, E, iod, y_top, y_bot, rect, tex_size):
        self.cx = float(E[0] - 1.6 * iod)        # the axis: about the middle of the head
        self.cz = float(E[2])
        self.y_top, self.y_bot = float(y_top), float(y_bot)
        self.rect = rect                          # x, y, w, h in texture pixels
        self.H, self.W = tex_size

    def theta(self, P):
        return np.arctan2(P[..., 2] - self.cz, P[..., 0] - self.cx)

    def radius(self, P):
        return np.hypot(P[..., 2] - self.cz, P[..., 0] - self.cx)

    def px(self, P):
        x, y, w, h = self.rect
        u = (self.theta(P) + TH) / (2 * TH)
        v = (P[..., 1] - self.y_top) / (self.y_bot - self.y_top)
        return np.stack([u * (w - 1), v * (h - 1)], -1)     # pixel coords inside the rectangle

    def uv(self, P):
        x, y, w, h = self.rect
        q = self.px(P)
        return np.stack([(x + q[..., 0] + 0.5) / self.W, (y + q[..., 1] + 0.5) / self.H], -1)


def raster(V2, F, size, z=None):
    """Rasterise triangles (pixel coords V2) into an id buffer (front-most by ``z`` = larger) and
    barycentrics. Returns (tri id (H, W) or -1, b1, b2)."""
    H, W = size
    best = np.full(H * W, -np.inf)
    tid = np.full(H * W, -1, dtype=np.int64)
    B1 = np.zeros(H * W)
    B2 = np.zeros(H * W)
    x, y = V2[:, 0], V2[:, 1]
    zz = np.zeros(len(V2)) if z is None else z
    for s0 in range(0, len(F), 20000):
        f = F[s0:s0 + 20000]
        x0, x1, x2 = x[f[:, 0]], x[f[:, 1]], x[f[:, 2]]
        y0, y1, y2 = y[f[:, 0]], y[f[:, 1]], y[f[:, 2]]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        ok = np.abs(area) > 1e-12
        xmin = np.clip(np.floor(np.minimum(np.minimum(x0, x1), x2)), 0, W - 1).astype(np.int64)
        xmax = np.clip(np.ceil(np.maximum(np.maximum(x0, x1), x2)), 0, W - 1).astype(np.int64)
        ymin = np.clip(np.floor(np.minimum(np.minimum(y0, y1), y2)), 0, H - 1).astype(np.int64)
        ymax = np.clip(np.ceil(np.maximum(np.maximum(y0, y1), y2)), 0, H - 1).astype(np.int64)
        bw = xmax - xmin + 1
        bh = ymax - ymin + 1
        n = np.where(ok, bw * bh, 0)
        if n.sum() == 0:
            continue
        ti = np.repeat(np.arange(len(f)), n)
        local = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        cx = xmin[ti] + local % bw[ti]
        cy = ymin[ti] + local // bw[ti]
        px, py = cx + 0.5, cy + 0.5
        ax, ay = x0[ti], y0[ti]
        d = area[ti]
        b1 = ((px - ax) * (y2[ti] - ay) - (py - ay) * (x2[ti] - ax)) / d
        b2 = ((x1[ti] - ax) * (py - ay) - (y1[ti] - ay) * (px - ax)) / d
        b0 = 1 - b1 - b2
        ins = (b0 >= -1e-4) & (b1 >= -1e-4) & (b2 >= -1e-4)
        ti, cx, cy, b1, b2, b0 = ti[ins], cx[ins], cy[ins], b1[ins], b2[ins], b0[ins]
        fz = zz[f[ti, 0]] * b0 + zz[f[ti, 1]] * b1 + zz[f[ti, 2]] * b2
        pid = cy * W + cx
        order = np.argsort(fz, kind="stable")          # nearest last -> wins
        pid, fz, ti, b1, b2 = pid[order], fz[order], ti[order], b1[order], b2[order]
        upd = fz >= best[pid]
        pid, fz, ti, b1, b2 = pid[upd], fz[upd], ti[upd], b1[upd], b2[upd]
        best[pid] = fz
        tid[pid] = ti + s0
        B1[pid] = b1
        B2[pid] = b2
    return tid.reshape(H, W), B1.reshape(H, W), B2.reshape(H, W)


def face_region(m, w, proj: Projection, sign, res=(420, 600)):
    """Triangles that make up the visible front of the face: inside the projection window, facing
    outward and the outermost surface there (hair in front of the forehead counts, the skin under
    it does not)."""
    P, F = m.positions, m.indices
    th = proj.theta(P)
    r = proj.radius(P)
    chin = getattr(proj, "chin_y", proj.y_bot)
    low = P[F, 1].min(1) > chin - 0.2                       # under the chin: jaw and top of the neck
    cand = ((w[F].min(1) > 0.3) | (low & (w[F].max(1) > 0.02))) & (np.abs(th[F]).max(1) < TH_FACE) & \
        (P[F, 1].min(1) > proj.y_top - 0.3) & (P[F, 1].max(1) < proj.y_bot)
    fn, _ = face_normals(P, F)
    fn = sign * fn                                  # outward
    cen = P[F].mean(1)
    out = np.stack([cen[:, 0] - proj.cx, np.zeros(len(cen)), cen[:, 2] - proj.cz], 1)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    facing = (fn * out).sum(1)
    cand &= np.where(low, facing > -0.15, facing > 0.12)
    ci = np.flatnonzero(cand)
    H, W = res
    V2 = np.stack([(th + TH) / (2 * TH) * (W - 1), (P[:, 1] - proj.y_top) / (proj.y_bot - proj.y_top) * (H - 1)], 1)
    allF = np.flatnonzero((np.abs(th[F]).max(1) < TH * 1.1) & (w[F].max(1) > 0.05))   # occluders: any head surface
    tid, _, _ = raster(V2, F[allF], res, r)
    hit = np.zeros(len(F), dtype=np.int64)
    ids = tid[tid >= 0]
    np.add.at(hit, allF[ids], 1)
    tid_c, _, _ = raster(V2, F[ci], res, None)
    cover = np.zeros(len(F), dtype=np.int64)
    ids = tid_c[tid_c >= 0]
    np.add.at(cover, ci[ids], 1)
    # visible: wins the depth test where it covers pixels (tiny triangles: keep if not beaten)
    vis = np.zeros(len(F), bool)
    vis[ci] = hit[ci] >= 0.5 * np.maximum(cover[ci], 1)
    vis[ci[(cover[ci] == 0)]] = True
    region = cand & vis
    # drop specks: keep the big connected piece(s) of the region
    return np.flatnonzero(region)


def bake_image(m, tris, proj: Projection, old_tex):
    """Sample the old texture into the face rectangle through the region's triangles."""
    x, y, w, h = proj.rect
    F = m.indices[tris]
    V2 = proj.px(m.positions)
    tid, b1, b2 = raster(V2, F, (h, w), None)
    mask = tid >= 0
    t = np.where(mask, tid, 0)
    f = F[t]
    b0 = 1 - b1 - b2
    uv = m.uvs
    ouv = uv[f[..., 0]] * b0[..., None] + uv[f[..., 1]] * b1[..., None] + uv[f[..., 2]] * b2[..., None]
    pos = m.positions[f[..., 0]] * b0[..., None] + m.positions[f[..., 1]] * b1[..., None] + \
        m.positions[f[..., 2]] * b2[..., None]
    from focusbuddy.companion import pbr
    col = pbr.sample(old_tex, ouv.reshape(-1, 2))[:, :3].reshape(h, w, 3) * 255.0
    col[~mask] = 0
    return col, mask, ouv, pos, tid


# ====================================================================== 3. retouch
def seams(ouv, mask, tex_size, k=6.0):
    """Pixels where neighbouring face pixels come from far-apart places in the old texture."""
    H, W = tex_size
    q = ouv * np.array([W, H])
    s = np.zeros(mask.shape, bool)
    gx = np.linalg.norm(q[:, 1:] - q[:, :-1], axis=2)
    gy = np.linalg.norm(q[1:] - q[:-1], axis=2)
    both_x = mask[:, 1:] & mask[:, :-1]
    both_y = mask[1:] & mask[:-1]
    med = np.median(np.concatenate([gx[both_x], gy[both_y]]))
    jx = both_x & (gx > k * max(med, 0.5))
    jy = both_y & (gy > k * max(med, 0.5))
    s[:, 1:] |= jx
    s[:, :-1] |= jx
    s[1:] |= jy
    s[:-1] |= jy
    return s


def push_pull(img, known):
    """Smoothly fill the unknown pixels from the known ones around them (a pyramid "push-pull"
    membrane: coarse levels bridge big holes, finer levels keep the colours local)."""
    import cv2
    levels = []
    c = img.astype(np.float32) * known[..., None]
    wgt = known.astype(np.float32)
    while min(wgt.shape) > 4:
        levels.append((c, wgt))
        c = cv2.pyrDown(c)
        wgt = cv2.pyrDown(wgt)
    out = c / np.maximum(wgt[..., None], 1e-6)
    for c, wgt in reversed(levels):
        up = cv2.resize(out, (wgt.shape[1], wgt.shape[0]), interpolation=cv2.INTER_LINEAR)
        a = np.clip(wgt * 4.0, 0.0, 1.0)[..., None]
        out = (c / np.maximum(wgt[..., None], 1e-6)) * a + up * (1 - a)
    return out


def pad(img, mask, iters=24):
    """Grow the image's colours outward over the empty pixels (no dark bleeding at the edges)."""
    import cv2
    out = img.copy()
    m = mask.copy()
    k = np.ones((3, 3), np.uint8)
    for _ in range(iters):
        if m.all():
            break
        grown = cv2.dilate(m.astype(np.uint8), k).astype(bool)
        ring = grown & ~m
        if not ring.any():
            break
        s = cv2.blur(out * m[..., None], (3, 3))
        c = cv2.blur(m.astype(np.float32), (3, 3))
        out[ring] = s[ring] / np.maximum(c[ring, None], 1e-6)
        m = grown
    out[~m] = out[m].mean(0) if m.any() else 0
    return out


def remove_seams(col, mask, seam_pairs_x, seam_pairs_y, scale=2):
    """Gradient-domain seam removal: find a smooth additive correction that cancels the colour
    jumps across the old island edges and changes nothing else (solved at 1/``scale`` resolution,
    the correction is smooth anyway). ``seam_pairs_*`` mark horizontally / vertically adjacent
    pixel pairs that straddle a seam."""
    import cv2
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import cg
    h, w = mask.shape
    hs, ws = h // scale, w // scale
    # the jump each seam pair carries, at full resolution
    jx = np.zeros((h, w - 1, 3))
    jy = np.zeros((h - 1, w, 3))
    jx[seam_pairs_x] = (col[:, 1:] - col[:, :-1])[seam_pairs_x]
    jy[seam_pairs_y] = (col[1:] - col[:-1])[seam_pairs_y]
    # down-sample: a coarse pixel pair takes the mean jump of the fine pairs it covers
    ms = cv2.resize(mask.astype(np.float32), (ws, hs), interpolation=cv2.INTER_AREA) > 0.5
    idx = -np.ones((hs, ws), dtype=np.int64)
    idx[ms] = np.arange(ms.sum())
    n = int(ms.sum())

    def coarse(j, sx, sy):
        hh, ww = j.shape[:2]
        jj = cv2.resize(j.astype(np.float32), (ws - sx, hs - sy), interpolation=cv2.INTER_AREA) * scale
        return jj
    cjx = coarse(jx, 1, 0)
    cjy = coarse(jy, 0, 1)
    rows, cols_, vals, rhs = [], [], [], []
    r = 0
    lam_seam, lam_smooth = 1.0, 0.25
    for (a, b, jj) in ((idx[:, :-1], idx[:, 1:], cjx), (idx[:-1], idx[1:], cjy)):
        ok = (a >= 0) & (b >= 0)
        aa, bb, jv = a[ok], b[ok], jj[ok]
        strong = np.linalg.norm(jv, axis=1) > 1e-3
        wgt = np.where(strong, lam_seam, lam_smooth)
        k = len(aa)
        rows += [np.arange(r, r + k), np.arange(r, r + k)]
        cols_ += [bb, aa]
        vals += [wgt, -wgt]
        rhs.append(-jv * np.where(strong, lam_seam, 0.0)[:, None])
        r += k
    # a weak pull to zero keeps the system anchored
    rows.append(np.arange(r, r + n))
    cols_.append(np.arange(n))
    vals.append(np.full(n, 0.02))
    rhs.append(np.zeros((n, 3)))
    r += n
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols_))), shape=(r, n)).tocsr()
    B = np.concatenate(rhs)
    AtA = (A.T @ A).tocsr()
    corr = np.zeros((hs, ws, 3), np.float32)
    for c in range(3):
        x, _ = cg(AtA, A.T @ B[:, c], maxiter=600)
        corr[..., c][ms] = x
    corr = pad(corr, ms, 8)
    full = cv2.resize(corr, (w, h), interpolation=cv2.INTER_CUBIC)
    out = col + full
    return np.clip(out, 0, 255)


def skin_mask(col, mask, ref):
    """Skin-coloured pixels (near the reference skin tone in Lab)."""
    import cv2
    lab = cv2.cvtColor(np.clip(col, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float)
    rl = cv2.cvtColor(np.clip(ref, 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2LAB).astype(float)[0, 0]
    d = np.sqrt(((lab[..., 1:] - rl[1:]) ** 2).sum(-1) * 1.6 + ((lab[..., 0] - rl[0]) * 0.45) ** 2)
    return mask & (d < 26), lab


def affine_fit(P, XY):
    A = np.concatenate([P, np.ones((len(P), 1))], 1)
    coef, *_ = np.linalg.lstsq(A, XY, rcond=None)
    res = np.abs(A @ coef - XY).mean()
    return coef, float(res)


def rebuild_painted(pf, m_old, ouv, pos_old, mask, rect, tex_size, img, rng, skin=None, mouth_px=None,
                    px_iod=None):
    """New ``painted.npz`` arrays for the face image: eye / mouth membership from the old data
    (looked up through each new pixel's old texture coordinate), coordinates from exact affine
    maps of the 3D position, and a fresh seamless closed-lid fill."""
    import cv2
    H, W = tex_size
    x0, y0, w, h = rect
    old_e = np.full(H * W, -1, dtype=np.int64)
    old_e[pf.e_idx] = pf.e_eye
    old_m = np.zeros(H * W, bool)
    old_m[pf.m_idx] = True
    q = ouv * np.array([W, H])
    oi = (np.clip(np.floor(q[..., 1]), 0, H - 1).astype(np.int64) * W +
          np.clip(np.floor(q[..., 0]), 0, W - 1).astype(np.int64))
    eye_of = np.where(mask, old_e[oi], -1)
    in_m = mask & old_m[oi]
    # the eye coordinates: affine in the old 3D position, fitted per eye on the old texels
    out = {}
    e_idx, e_xy, e_eye = [], [], []
    ring, r_lid = 0.3, float(pf.R)
    for k in np.unique(pf.e_eye):
        sel = np.flatnonzero(pf.e_eye == k)
        pick = rng.choice(sel, min(1500, len(sel)), replace=False)
        p = uv_to_pos(m_old, texel_uv(pf.e_idx[pick], pf.size))
        ok = ~np.isnan(p[:, 0])
        coef, res = affine_fit(p[ok], pf.e_xy[pick][ok].astype(float))
        rmax = float(np.hypot(pf.e_xy[sel, 0], pf.e_xy[sel, 1]).max())
        if skin is not None:
            # a person's eye: the old area plus a ring around it, so a closing lid also covers the
            # painted liner and lashes at the eye's edge (no outlined disc on a closed eye)
            allpix = np.argwhere(mask)
            xy_all = np.concatenate([pos_old[allpix[:, 0], allpix[:, 1]], np.ones((len(allpix), 1))], 1) @ coef
            r_all = np.hypot(xy_all[:, 0], xy_all[:, 1])
            mine = eye_of[allpix[:, 0], allpix[:, 1]] == k
            keep = (mine & (r_all <= rmax + 0.05)) | ((r_all > rmax - 0.1) & (r_all <= rmax + ring) &
                                                      (eye_of[allpix[:, 0], allpix[:, 1]] < 0))
            pix, xy = allpix[keep], xy_all[keep]
            r_lid = max(r_lid, rmax + ring)
        else:
            pix = np.argwhere(eye_of == k)
            xy = np.concatenate([pos_old[pix[:, 0], pix[:, 1]], np.ones((len(pix), 1))], 1) @ coef
            # the eye's area is the old one: keep what lies inside the old radius
            keep = np.hypot(xy[:, 0], xy[:, 1]) <= rmax + 0.05
            pix, xy = pix[keep], xy[keep]
        e_idx.append((y0 + pix[:, 0]) * W + x0 + pix[:, 1])
        e_xy.append(xy)
        e_eye.append(np.full(len(pix), k))
        print(f"    eye {k}: {len(pix)} texels (was {len(sel)}), fit error {res:.3f}")
    e_idx = np.concatenate(e_idx)
    e_xy = np.concatenate(e_xy).astype(np.float32)
    e_eye = np.concatenate(e_eye)
    # closed lids: the skin around each eye continued over it (inpainted), a touch darker / warmer
    em = np.zeros((h, w), np.uint8)
    rr = (e_idx // W - y0, e_idx % W - x0)
    em[rr] = 255
    em = cv2.dilate(em, np.ones((5, 5), np.uint8))
    src = np.clip(img, 0, 255).astype(np.uint8)
    if skin is not None and px_iod:
        # a person's lid is skin: the colour of the skin around the eye only (never hair or the
        # painted eye-liner), smoothly continued over it - no visible patch when the eye closes
        lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(float)
        core = skin & mask & (em == 0)
        refc = np.median(src[core], 0) if core.any() else np.array([230.0, 180.0, 160.0])
        rlab = cv2.cvtColor(refc.astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
        core &= (np.abs(lab[..., 0] - rlab[0]) < 22) & (np.hypot(lab[..., 1] - rlab[1], lab[..., 2] - rlab[2]) < 14)
        sk = core.astype(np.float32)
        sig = 0.18 * px_iod
        num = cv2.GaussianBlur(src.astype(np.float32) * sk[..., None], (0, 0), sig)
        den = cv2.GaussianBlur(sk, (0, 0), sig)[..., None]
        wide = cv2.GaussianBlur(src.astype(np.float32) * sk[..., None], (0, 0), 3 * sig) / \
            np.maximum(cv2.GaussianBlur(sk, (0, 0), 3 * sig)[..., None], 1e-4)
        wden = cv2.GaussianBlur(sk, (0, 0), 3 * sig)[..., None]
        wide = np.where(wden > 0.02, wide, refc[None, None, :].astype(np.float32))
        smooth_skin = np.where(den > 0.05, num / np.maximum(den, 1e-4), wide)
        # the lid continues the skin right around the eye (with its socket shading), filled in
        # from the edge; hair or holes around it are replaced by the smooth skin first, so they
        # never leak into the lid
        hairy = ~(mask & (lab[..., 0] > rlab[0] - 58) & (np.hypot(lab[..., 1] - rlab[1], lab[..., 2] - rlab[2]) < 40))
        src2 = np.where(hairy[..., None], smooth_skin, src.astype(np.float32))
        known = mask & (em == 0)
        fill = push_pull(src2, known)
        fill = cv2.GaussianBlur(fill, (0, 0), 1.5)
        fill *= np.array([0.97, 0.95, 0.945], np.float32)           # lids: a shade darker and warmer
    else:
        fill = cv2.inpaint(src, em, 9, cv2.INPAINT_TELEA).astype(np.float32)
        fill = cv2.GaussianBlur(fill, (0, 0), 2.0) * 0.6 + fill * 0.4
        fill *= np.array([0.96, 0.93, 0.92], np.float32)
    e_fill = fill[rr]
    # texel size in eye units
    texel = []
    for k in np.unique(e_eye):
        s = e_eye == k
        g = np.abs(np.diff(e_xy[s][np.argsort(e_idx[s])][:, 0]))
        texel.append(float(np.median(g[g > 0])) if (g > 0).any() else 0.02)
    if mouth_px is not None:
        # a person's mouth: coordinates from the lip line found in the image itself
        pix, mxy = mouth_px
        print(f"    mouth: {len(pix)} texels from the lip line")
        out.update(size=np.array(pf.size), R=np.array(r_lid, np.float32), R_eye=np.array(pf.R, np.float32),
                   lash_r=np.array(pf.LASH_R, np.float32),
                   squint_style=np.array(pf.squint_style), e_idx=e_idx, e_xy=e_xy.astype(np.float16),
                   e_fill=e_fill.astype(np.float16), e_eye=e_eye.astype(np.int8), lash=pf.lash.astype(np.float32),
                   texel=np.array(texel, np.float32), m_idx=(y0 + pix[:, 0]) * W + x0 + pix[:, 1],
                   m_xy=mxy.astype(np.float16), m_cols=pf.m_cols, m_shape=np.array([0.78, 0.3], np.float32),
                   m_open=np.array(0.30, np.float32), m_style=np.array("human"), lid_style=np.array("skin"))
        return out
    # the mouth
    pick = rng.choice(len(pf.m_idx), min(1200, len(pf.m_idx)), replace=False)
    p = uv_to_pos(m_old, texel_uv(pf.m_idx[pick], pf.size))
    ok = ~np.isnan(p[:, 0])
    coef, res = affine_fit(p[ok], pf.m_xy[pick][ok].astype(float))
    pix = np.argwhere(in_m)
    mxy = np.concatenate([pos_old[pix[:, 0], pix[:, 1]], np.ones((len(pix), 1))], 1) @ coef
    print(f"    mouth: {len(pix)} texels (was {len(pf.m_idx)}), fit error {res:.3f}")
    out.update(size=np.array(pf.size), R=np.array(pf.R, np.float32), lash_r=np.array(pf.LASH_R, np.float32),
               squint_style=np.array(pf.squint_style), e_idx=e_idx, e_xy=e_xy.astype(np.float16),
               e_fill=e_fill.astype(np.float16), e_eye=e_eye.astype(np.int8), lash=pf.lash.astype(np.float32),
               texel=np.array(texel, np.float32), m_idx=(y0 + pix[:, 0]) * W + x0 + pix[:, 1],
               m_xy=mxy.astype(np.float16), m_cols=pf.m_cols, m_shape=pf.m_shape)
    return out


def lip_line(img, mask, eyes_px, px_iod, debug=None):
    """Find the line between the lips below the eyes (pixel coords ``eyes_px``): the lips are the
    reddest blob on the lower middle of the face, the line is the darkest curve through them.
    Returns the pixels of the mouth area and their mouth coordinates (X = +-1 at the corners,
    Y = 1 one mouth-width below the lip line, which follows the smile's curve)."""
    import cv2
    h, w = mask.shape
    lab = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float)
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.2)
    A = cv2.GaussianBlur(lab[..., 1], (0, 0), 2.0)
    ex = float(np.mean([e[0] for e in eyes_px]))
    ey = float(np.mean([e[1] for e in eyes_px]))
    yy, xx = np.mgrid[0:h, 0:w]
    area = mask & (np.abs(xx - ex) < 0.55 * px_iod) & (yy > ey + 0.55 * px_iod) & (yy < ey + 1.6 * px_iod)
    thr = np.percentile(A[area], 93)
    blob = (area & (A >= thr)).astype(np.uint8)
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, st, cen = cv2.connectedComponentsWithStats(blob)
    best, bscore = 0, -1e9
    for k in range(1, n):
        score = st[k, cv2.CC_STAT_AREA] - 3.0 * abs(cen[k, 0] - ex)
        if score > bscore:
            best, bscore = k, score
    bx, by, bw, bh = st[best, :4]
    # the parting line: the darkest smooth path through the lips (dynamic programming, the line
    # may climb or drop one pixel per column), a little beyond the lips on both sides
    x0, x1 = int(max(0, bx - 0.3 * bw)), int(min(w - 1, bx + bw * 1.3))
    y0, y1 = int(max(0, by - 0.6 * bh)), int(min(h - 1, by + bh * 1.2))
    win = L[y0:y1, x0:x1]
    nh, nw = win.shape
    cost = win.copy()
    back = np.zeros((nh, nw), np.int64)
    for i in range(1, nw):
        prev = cost[:, i - 1]
        cand = np.stack([np.r_[np.inf, prev[:-1]], prev, np.r_[prev[1:], np.inf]])
        k = np.argmin(cand, axis=0)
        cost[:, i] = win[:, i] + cand[k, np.arange(nh)]
        back[:, i] = np.arange(nh) + (k - 1)
    path = np.zeros(nw, np.int64)
    path[-1] = int(np.argmin(cost[:, -1]))
    for i in range(nw - 1, 0, -1):
        path[i - 1] = back[path[i], i]
    ref = np.percentile(win, 70)
    dark = ref - win[path, np.arange(nw)]
    dark = np.convolve(dark, np.ones(5) / 5, mode="same")
    cols = np.stack([x0 + np.arange(nw), y0 + path, dark], 1).astype(float)
    strong = dark > 0.07 * ref
    mid = int(np.argmin(np.abs(cols[:, 0] - (bx + bw / 2))))
    if not strong[mid]:
        near = np.flatnonzero(strong)
        mid = int(near[np.argmin(np.abs(near - mid))]) if len(near) else mid
    lo = mid
    while lo > 0 and (strong[lo - 1] or strong[max(0, lo - 3):lo].any()):
        lo -= 1
    hi = mid
    while hi < nw - 1 and (strong[hi + 1] or strong[hi + 1:hi + 4].any()):
        hi += 1
    run = cols[lo:hi + 1]
    xl, xr = run[0, 0], run[-1, 0]
    half = max(4.0, (xr - xl) / 2.0)
    xc = (xl + xr) / 2.0
    coef = np.polyfit(run[:, 0] - xc, run[:, 1], 2, w=np.maximum(run[:, 2], 1e-3))
    X = (xx - xc) / half
    Y = (yy - np.polyval(coef, np.clip(xx - xc, -half, half))) / (2 * half)
    marea = mask & (np.abs(X) <= 1.2) & (Y >= -0.3) & (Y <= 0.55)
    pix = np.argwhere(marea)
    xy = np.stack([X[marea], Y[marea]], 1)
    if debug is not None:
        debug.update(xl=xl, xr=xr, xc=xc, coef=coef, half=half, blob=(bx, by, bw, bh))
    return pix, xy


def eye_coords(painted, pos_old, rect, W):
    """(X, Y) in each eye's ellipse units for every face pixel (affine in the 3D position)."""
    x0, y0 = rect[0], rect[1]
    out = []
    for k in np.unique(painted["e_eye"]):
        s = painted["e_eye"] == k
        ii = painted["e_idx"][s]
        py, px = ii // W - y0, ii % W - x0
        coef, _ = affine_fit(pos_old[py, px], painted["e_xy"][s].astype(float))
        flat = pos_old.reshape(-1, 3)
        xy = (np.concatenate([flat, np.ones((len(flat), 1))], 1) @ coef).reshape(pos_old.shape[:2] + (2,))
        out.append(xy)
    return out


def retouch(col, mask, ouv, tex_size, pos, pos_old, eyes, mouth, iod, painted, rect, proj, blush=0.5):
    """Seamless, even, natural skin; the eyes, brows, lips and nostrils stay sharp."""
    import cv2
    H, W = tex_size
    h, w = mask.shape
    q = ouv * np.array([W, H])
    gx = np.linalg.norm(q[:, 1:] - q[:, :-1], axis=2)
    gy = np.linalg.norm(q[1:] - q[:-1], axis=2)
    bx = mask[:, 1:] & mask[:, :-1]
    by = mask[1:] & mask[:-1]
    med = np.median(np.concatenate([gx[bx], gy[by]]))
    sx = bx & (gx > 6 * max(med, 0.5))
    sy = by & (gy > 6 * max(med, 0.5))
    out = remove_seams(col, mask, sx, sy)
    # --- features to protect: eyes + lashes, brows, lips, nostrils
    exy = eye_coords(painted, pos_old, rect, W)
    feat = np.zeros((h, w), bool)
    for xy in exy:
        X, Y = xy[..., 0], xy[..., 1]
        feat |= (X / 1.55) ** 2 + (Y / 1.45) ** 2 <= 1.0                      # eye, lid, lashes
        feat |= (np.abs(X) < 1.9) & (Y > 0.6) & (Y < 3.4)                     # the brow above it
    mp = np.zeros((h, w), np.uint8)
    mi = painted["m_idx"]
    mp[mi // W - rect[1], mi % W - rect[0]] = 255
    px_iod = np.linalg.norm(proj.px(eyes[0]) - proj.px(eyes[1]))
    mp = cv2.dilate(mp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(px_iod * 0.25) | 1,) * 2))
    feat |= mp > 0
    # the nose: the most forward point between the eyes and the mouth, near the middle
    E = (eyes[0] + eyes[1]) / 2
    midz = np.abs(pos[..., 2] - E[2]) < 0.3 * iod
    band = mask & midz & (pos[..., 1] > E[1] + 0.3 * iod) & (pos[..., 1] < mouth[1] - 0.15 * iod)
    if band.any():
        k = np.argmax(np.where(band, pos[..., 0], -1e9))
        ny, nx = np.unravel_index(k, band.shape)
        yy, xx = np.mgrid[0:h, 0:w]
        feat |= ((xx - nx) / (0.32 * px_iod)) ** 2 + ((yy - ny - 0.05 * px_iod) / (0.22 * px_iod)) ** 2 <= 1.0
    lum = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    L, A, B = (lum[..., c].astype(float) for c in range(3))
    # --- skin: light, warm pixels (hair is darker, clothes are cooler or paler)
    ref_m = mask & (pos[..., 1] > mouth[1] - 0.9 * iod) & (pos[..., 1] < mouth[1] - 0.1 * iod) & ~feat
    ref = np.median(out[ref_m], 0)
    rl = cv2.cvtColor(np.clip(ref, 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    skin = mask & (L > rl[0] - 32) & (A > 132) & (B > 132) & (np.hypot(A - rl[1], B - rl[2]) < 22)
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(skin)
    big = np.zeros(n, bool)
    big[1:] = st[1:, cv2.CC_STAT_AREA] > 0.05 * st[1:, cv2.CC_STAT_AREA].max() if n > 1 else False
    skin = big[lbl] & mask
    # --- strands of hair colour painted onto the skin: thin dark lines (black-hat) away from features
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(px_iod * 0.16) | 1,) * 2)
    bh = cv2.morphologyEx(L.astype(np.uint8), cv2.MORPH_BLACKHAT, ker).astype(float)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(px_iod * 0.22) | 1,) * 2)
    region = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_CLOSE, close_k)
    region = cv2.erode(region, np.ones((5, 5), np.uint8)) > 0
    cheeks = (pos[..., 1] > E[1] - 0.35 * iod) & (pos[..., 1] < mouth[1] + 0.25 * iod) & \
        (np.abs(pos[..., 2] - E[2]) < 1.75 * iod)                      # the face proper, not the hair
    smudge = region & cheeks & ~cv2.dilate(feat.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & (bh > 12)
    smudge = cv2.dilate(smudge.astype(np.uint8), np.ones((5, 5), np.uint8))
    if smudge.any():
        fixed = cv2.inpaint(np.clip(out, 0, 255).astype(np.uint8), smudge * 255, 6, cv2.INPAINT_TELEA)
        out = np.where(smudge[..., None] > 0, fixed.astype(float), out)
        skin = skin | ((smudge > 0) & mask)
    # --- even skin tone: the broad colour of the skin drifts halfway toward the face's own tone
    #     (removes pale or grey patches left by the generator; blush and shading survive)
    # pale or greyish patches of the face proper count as skin here (the generator left some)
    zone = mask & (pos[..., 1] > E[1] - 0.3 * iod) & (pos[..., 1] < mouth[1] + 0.9 * iod) & \
        (np.abs(pos[..., 2] - E[2]) < 1.75 * iod) & ~feat
    lab0 = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float)
    pale = zone & (lab0[..., 0] > rl[0] - 20) & (np.hypot(lab0[..., 1] - 128, lab0[..., 2] - 128) <
                                               np.hypot(rl[1] - 128, rl[2] - 128) * 0.8)
    skin = skin | pale
    soft_t = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 3.0)
    soft = soft_t * (1.0 - cv2.GaussianBlur(feat.astype(np.float32), (0, 0), 4.0))
    lab = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    sk = skin.astype(np.float32)
    sig = 0.35 * px_iod
    num = cv2.GaussianBlur(lab * sk[..., None], (0, 0), sig)
    den = cv2.GaussianBlur(sk, (0, 0), sig)[..., None]
    low = num / np.maximum(den, 1e-3)
    tgt = rl.astype(np.float32)
    corr = (tgt - low) * np.array([0.30, 0.60, 0.60], np.float32)
    corr[..., 0] = np.clip(corr[..., 0], -12, 12)
    lab = lab + corr * soft_t[..., None]
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(float)
    # --- gentle edge-preserving smoothing of the skin (not the features)
    lum2 = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(lum2, 40, 90) > 0
    protect = cv2.GaussianBlur((feat | cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool))
                               .astype(np.float32), (0, 0), 3.0)
    even = cv2.bilateralFilter(np.clip(out, 0, 255).astype(np.uint8), 0, 9, 3).astype(float)
    k = (soft * (1.0 - protect))[..., None] * 0.6
    out = out * (1 - k) + even * k
    # --- a soft, natural blush on the cheeks (below and outside each eye)
    if blush > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        for e in eyes:
            c3 = e + np.array([0.15 * iod, 0.95 * iod, (e[2] - E[2]) * 0.35])
            cx, cy = proj.px(c3)
            g = np.exp(-(((xx - cx) / (0.55 * px_iod)) ** 2 + ((yy - cy) / (0.38 * px_iod)) ** 2))
            a = (blush * 0.16 * g * soft)[..., None]
            out = out * (1 - a) + (out * np.array([1.0, 0.80, 0.80])) * a
    # --- toward the edge of the face image the retouch fades out, so where the face meets the
    #     rest of the head (temples, neck) the colours are exactly the old ones: no visible step
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    wb = np.clip((dist - 2.0) / (0.12 * px_iod), 0.0, 1.0)[..., None]
    out = col * (1 - wb) + out * wb
    # --- crisp features: the old face had far fewer texels, so eyes, brows and lips were soft;
    #     a gentle unsharp mask restores their edges (the skin is left alone)
    fs = cv2.GaussianBlur(feat.astype(np.float32), (0, 0), 3.0)[..., None]
    blur = cv2.GaussianBlur(out.astype(np.float32), (0, 0), 1.4)
    out = out + 0.55 * fs * (out - blur)
    return np.clip(out, 0, 255), skin, ref, feat, smudge


def bake_character(cid: str, strength: float = 1.0, debug: Path | None = None) -> Path:
    import copy
    import cv2
    from focusbuddy.companion.character_asset import CharacterAsset, asset_dir
    from focusbuddy.companion.face_paint import PaintedFace
    folder = asset_dir(cid)
    rng = np.random.default_rng(7)
    a = CharacterAsset.from_folder(cid, folder, refine="none")
    hi = a.lods["high"]
    md = [x for x in a.materials if "baseColor" in x.textures][0]
    tex = md.textures["baseColor"]
    old_painted = folder / "face" / "painted_v22.npz"
    if not old_painted.is_file():                  # keep the original face data as the source
        import shutil
        shutil.copy2(folder / "face" / "painted.npz", old_painted)
    pf = PaintedFace(old_painted)
    w_hi = head_weight(a, hi)
    eyes, mouth = landmarks(hi, pf, rng)
    P_hi_new, info = sculpt(hi, w_hi, eyes, mouth, -1, strength)
    iod = info["iod"]
    E = info["E"]
    rect = (1032, 2048 + 8, 1008, 872)
    if tex.shape[0] < 3072 or tex.shape[1] < 2048:
        raise SystemExit(f"{cid}: unexpected texture size {tex.shape}")
    proj = Projection(E, iod, E[1] - 1.25 * iod, info["neck_y"] + 0.15 * iod, rect, tex.shape[:2])
    proj.chin_y = info["chin_y"]
    out = {}
    # --- geometry and face triangles per level of detail
    hi_new = copy.copy(hi)
    hi_new.positions = P_hi_new
    for lod, m in a.lods.items():
        if lod == "high":
            Pn = P_hi_new
        else:
            Pn = transfer(hi.positions, P_hi_new, m.positions, 0.25 * iod)
        moved = np.linalg.norm(Pn - m.positions, axis=1) > 1e-6
        inv = welded(m.positions)
        N = vertex_normals(Pn, m.indices, inv, -1)
        # normals change on the moved vertices and two rings around them
        ring = moved.copy()
        F = m.indices
        for _ in range(2):
            t = ring[F].any(1)
            ring[F[t].ravel()] = True
            rw = np.zeros(int(inv.max()) + 1, bool)
            np.logical_or.at(rw, inv, ring)
            ring = rw[inv]
        idx = np.flatnonzero(ring)
        out[f"shape_idx_{lod}"] = idx.astype(np.int32)
        out[f"shape_pos_{lod}"] = Pn[idx].astype(np.float32)
        out[f"shape_nrm_{lod}"] = N[idx].astype(np.float16)
        # the jaw's own shadow: its underside and the top of the neck sit out of the light
        wm = head_weight(a, m)
        depth2 = max(0.3 * iod, info["neck_y"] - info["chin_y"])
        t = (Pn[:, 1] - info["chin_y"]) / depth2
        bump = smoothstep(-0.05, 0.3, t) * (1.0 - smoothstep(1.0, 1.8, t))
        hw = 1.9 * iod
        bump *= 1.0 - smoothstep(0.75 * hw, 1.1 * hw, np.abs(Pn[:, 2] - info["cz"]))
        bump *= smoothstep(info["neck_x"] - 1.2 * iod, info["neck_x"] - 0.3 * iod, Pn[:, 0])
        down = np.clip(N[:, 1] * 1.5, 0.0, 1.0)
        tint = 1.0 - 0.42 * strength * bump * (0.35 + 0.65 * down)
        ti = np.flatnonzero(tint < 0.995)
        si, sj, sw = reskin(a, m, Pn, info, strength)
        out[f"skin_idx_{lod}"] = si.astype(np.int32)
        out[f"skin_j_{lod}"] = sj.astype(np.uint8)
        out[f"skin_w_{lod}"] = sw.astype(np.float32)
        out[f"tint_idx_{lod}"] = ti.astype(np.int32)
        out[f"tint_{lod}"] = tint[ti].astype(np.float16)
        mm = copy.copy(m)
        mm.positions = Pn
        tris = face_region(mm, wm, proj, -1)
        verts = np.unique(m.indices[tris])
        out[f"face_tris_{lod}"] = tris.astype(np.int32)
        out[f"face_verts_{lod}"] = verts.astype(np.int32)
        out[f"face_uv_{lod}"] = proj.uv(Pn[verts]).astype(np.float32)
        print(f"  {lod:6s} moved {moved.sum():6d} verts, face {len(tris)} triangles / {len(verts)} vertices")
    # --- the face image (from the high level of detail)
    tris = out["face_tris_high"].astype(np.int64)
    col, mask, ouv, pos, tid = bake_image(hi_new, tris, proj, tex)
    F = hi.indices[tris]
    t = np.where(mask, tid, 0)
    V2 = proj.px(P_hi_new)
    # old positions at the same surface points (the painted data was fitted to them)
    _, b1, b2 = raster(V2, F, (rect[3], rect[2]), None)
    b0 = 1 - b1 - b2
    f = F[t]
    pos_old = hi.positions[f[..., 0]] * b0[..., None] + hi.positions[f[..., 1]] * b1[..., None] + \
        hi.positions[f[..., 2]] * b2[..., None]
    if debug:
        cv2.imwrite(str(debug / f"{cid}_0_baked.png"), cv2.cvtColor(np.clip(col, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    painted = rebuild_painted(pf, hi, ouv, pos_old, mask, rect, tex.shape[:2], col, rng)
    img, skin, ref, feat, smudge = retouch(col, mask, ouv, tex.shape[:2], pos, pos_old, eyes, mouth, iod, painted,
                                           rect, proj)
    if debug:
        dbg = np.clip(img, 0, 255).astype(np.uint8).copy()
        dbg[feat] = dbg[feat] // 2 + np.array([0, 60, 0], np.uint8)
        dbg[smudge > 0] = [255, 0, 255]
        cv2.imwrite(str(debug / f"{cid}_masks.png"), cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
    # bare skin everywhere on the body (face, ears, neck, arms, hands): the "skin" material
    import cv2 as _cv
    from focusbuddy.companion import pbr as _pbr
    rl = _cv.cvtColor(np.clip(ref, 0, 255).astype(np.uint8).reshape(1, 1, 3), _cv.COLOR_RGB2LAB)[0, 0].astype(float)
    names = [str(getattr(j, "name", j)) for j in a.joints]
    bare = [i for i, n in enumerate(names) if n in ("Head", "HeadTop", "Neck", "Chest", "Spine") or "Arm" in n
            or "Hand" in n or "Finger" in n or "Thumb" in n]
    for lod, m in a.lods.items():
        c = _pbr.sample(tex, m.uvs)[:, :3] * 255.0
        lab = _cv.cvtColor(np.clip(c, 0, 255).astype(np.uint8).reshape(-1, 1, 3), _cv.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
        dom = m.joints[np.arange(len(m.joints)), np.argmax(m.weights, axis=1)]
        sk = (lab[:, 0] > rl[0] - 40) & (np.hypot(lab[:, 1] - rl[1], lab[:, 2] - rl[2]) < 20) & \
            (lab[:, 1] > 131) & np.isin(dom, bare)
        out[f"skinmat_{lod}"] = sk
        print(f"  {lod:6s} skin vertices {sk.sum()}")
    # the lids are re-made from the retouched image (seamless with it)
    px_iod = float(np.linalg.norm(proj.px(eyes[0]) - proj.px(eyes[1])))
    dbg_m = {}
    mpix = lip_line(img, mask, [proj.px(e) for e in eyes], px_iod, dbg_m)
    print(f"    lip line: corners {dbg_m['xl']:.0f}..{dbg_m['xr']:.0f} px, centre {dbg_m['xc']:.0f}")
    painted = rebuild_painted(pf, hi, ouv, pos_old, mask, rect, tex.shape[:2], img, rng, skin=skin,
                              mouth_px=mpix, px_iod=px_iod)
    if debug:
        dd = np.clip(img, 0, 255).astype(np.uint8).copy()
        xs = np.arange(int(dbg_m["xl"]), int(dbg_m["xr"]) + 1)
        ys = np.polyval(dbg_m["coef"], xs - dbg_m["xc"]).astype(int)
        dd[np.clip(ys, 0, dd.shape[0] - 1), xs] = [0, 255, 0]
        cv2.imwrite(str(debug / f"{cid}_lips.png"), cv2.cvtColor(dd, cv2.COLOR_RGB2BGR))
    img = pad(img, mask, 40)
    out["face_rgb"] = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    out["face_rect"] = np.array(rect, np.int32)
    if debug:
        cv2.imwrite(str(debug / f"{cid}_1_final.png"), cv2.cvtColor(out["face_rgb"], cv2.COLOR_RGB2BGR))
    np.savez_compressed(folder / "face" / "painted.npz", **painted)
    path = folder / "refine.npz"
    old = {}
    if path.is_file():
        with np.load(path) as z:
            old = {k: z[k] for k in z.files if not k.startswith(("shape_", "face_", "tint", "skin"))}
    old.update(out)
    np.savez_compressed(path, **old)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--debug", type=Path, default=None, help="write the intermediate face images here")
    a = ap.parse_args(argv)
    for cid in a.ids or PEOPLE:
        t0 = time.time()
        print(cid)
        print(" ", bake_character(cid, a.strength, a.debug), f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
