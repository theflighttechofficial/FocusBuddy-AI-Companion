"""Detailed heads for the 3D companions: graft the head of a detailed-sheet reconstruction onto a body,
and compose several textured parts into one mesh with one texture atlas.

The detailed character sheets draw the animals sitting, while the companions walk on all fours;
their standing bodies come from the first sheets. ``tools/build_3d_companions.py`` therefore takes
the *head* - the part that carries the face, the expressions and most of the character - from the
big front view's reconstruction and puts it on the standing body:

1. the head is cut off its sitting body just under the collar (the collar line is given in the
   front cut-out's pixels and found on the mesh through the front map); the collar and its charm
   come with the head, so the seam is hidden under the collar like on the sheet;
2. the standing body loses its own head above the neck;
3. the detailed head is scaled to the chibi proportions of the sheet, placed on the neck (the
   collar sits around it) and kept upright, facing forward;
4. both parts (and any extra part, such as hands) are re-packed into one UV atlas and re-baked
   from their own textures; the front map (the sheet coordinates of every texel the front view
   painted) is carried along, so the expressions can be painted on the new atlas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Part:
    """A textured triangle mesh in the build frame (+Y up, facing +Z)."""
    P: np.ndarray                        # (V, 3) positions
    F: np.ndarray                        # (T, 3) faces
    UV: np.ndarray                       # (V, 2) uvs into ``tex``
    tex: np.ndarray                      # (S, S, 3) uint8
    fmap: np.ndarray | None = None       # (S, S, 3) front map (sheet x, sheet y, weight) or None
    density: float = 1.0                 # texel density relative to the other parts
    tag: str = "part"
    extra: dict = field(default_factory=dict)


def subset(part: Part, keep_faces: np.ndarray) -> Part:
    F = part.F[keep_faces]
    used = np.unique(F)
    remap = np.full(len(part.P), -1, np.int64)
    remap[used] = np.arange(len(used))
    return Part(part.P[used].copy(), remap[F], part.UV[used].copy(), part.tex, part.fmap, part.density, part.tag,
                dict(part.extra))


def big_components(part: Part, min_frac: float = 0.03) -> Part:
    """Drop small floating bits (welded adjacency)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    key = np.round(part.P / 1e-6).astype(np.int64)
    _, ids = np.unique(key, axis=0, return_inverse=True)
    ids = ids.reshape(-1)
    Fw = ids[part.F]
    n = int(ids.max()) + 1
    e = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    fl = lab[Fw[:, 0]]
    sizes = np.bincount(fl)
    return subset(part, sizes[fl] >= min_frac * sizes.max())


def sample(img: np.ndarray, uv: np.ndarray, order: int = 1) -> np.ndarray:
    """Sample an (S, S, C) image at uvs (glTF convention: row = v * S)."""
    from scipy import ndimage
    S = img.shape[0]
    x = uv[:, 0] * S - 0.5
    y = uv[:, 1] * S - 0.5
    return np.stack([ndimage.map_coordinates(img[..., c].astype(np.float32), [y, x], order=order, mode="nearest")
                     for c in range(img.shape[2])], 1)


# ------------------------------------------------------------------ the head
def collar_cut(head: Part, collar_y: float, charm_box=None, band: float = 8.0) -> tuple[Part, float, np.ndarray]:
    """Cut a sitting reconstruction under its collar. ``collar_y``: the collar's lower edge in the
    front cut-out (pixels); ``charm_box``: (x0, y0, x1, y1) of the charm hanging from it, kept too.
    Returns (head part, cut height, collar-ring centre)."""
    fm = sample(head.fmap, head.UV, order=0)              # per vertex: sheet x, sheet y, weight
    seen = fm[:, 2] > 0.6
    ring = seen & (np.abs(fm[:, 1] - collar_y) < band)
    if ring.sum() < 10:
        raise RuntimeError("collar line not found on the mesh (check collar_y)")
    cut = float(np.median(head.P[ring, 1]))
    keep_v = head.P[:, 1] > cut
    if charm_box is not None:
        x0, y0, x1, y1 = charm_box
        charm = (fm[:, 2] > 0.3) & (fm[:, 0] > x0) & (fm[:, 0] < x1) & (fm[:, 1] > y0) & (fm[:, 1] < y1)
        keep_v |= charm
    piece = big_components(subset(head, keep_v[head.F].all(axis=1)), 0.02)
    near = np.abs(piece.P[:, 1] - cut) < 0.02 * np.ptp(piece.P[:, 1])
    centre = piece.P[near].mean(axis=0) if near.any() else piece.P.mean(axis=0)
    centre[0] = 0.0
    return piece, cut, centre


def remove_head(body: Part, J: dict, t: float = 0.45, reach: float = 3.0) -> tuple[Part, np.ndarray, np.ndarray, float]:
    """Remove a body's own head: everything past the plane across the neck (``t`` of the way from
    the neck joint to the head joint), near the head. Returns (body, plane point, plane normal,
    height of the removed head)."""
    a, b = np.asarray(J["Neck"], float), np.asarray(J["Head"], float)
    n = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    q = a + t * (b - a)
    top = np.asarray(J.get("HeadTop", b + (b - a)), float)
    R = reach * max(np.linalg.norm(top - q), np.linalg.norm(b - a))
    d = (body.P - q) @ n
    gone = (d > 0) & (np.linalg.norm(body.P - q, axis=1) < R)
    old_h = float(body.P[gone, 1].max() - q[1]) if gone.any() else float(np.linalg.norm(top - q))
    keep = ~gone[body.F].any(axis=1)
    return big_components(subset(body, keep), 0.05), q, n, old_h


def place_head(piece: Part, ring: np.ndarray, cut: float, at: np.ndarray, height: float,
               sink: float = 0.0, pitch: float = 0.0) -> Part:
    """Scale the head piece to ``height`` (collar line to the top) and put its collar ring at ``at``
    (lowered by ``sink``), upright; ``pitch`` (deg) nods it forward."""
    h0 = float(piece.P[:, 1].max() - cut)
    k = height / max(h0, 1e-9)
    P = (piece.P - ring) * k
    if pitch:
        a = np.radians(-pitch)                  # + nods forward (the front goes down)
        c, s = np.cos(a), np.sin(a)
        P = P @ np.array([[1, 0, 0], [0, c, s], [0, -s, c]]).T
    P = P + at - np.array([0.0, sink, 0.0])
    out = Part(P, piece.F, piece.UV, piece.tex, piece.fmap, piece.density, piece.tag, dict(piece.extra))
    out.extra["scale"] = k
    return out


# ------------------------------------------------------------------ one mesh, one atlas
def compose(parts: list[Part], size: int = 2048, dilate_steps: int = 8):
    """Merge parts into one mesh and re-bake one texture atlas (and front map) from the parts' own
    textures. Returns (P, F, UV, tex, fmap, part index per vertex)."""
    import xatlas
    from reconstruct_3d import dilate, raster_bary
    offs = np.cumsum([0] + [len(p.P) for p in parts])
    P = np.concatenate([p.P for p in parts])
    F = np.concatenate([p.F + o for p, o in zip(parts, offs[:-1])])
    pid = np.concatenate([np.full(len(p.P), i) for i, p in enumerate(parts)])
    oldUV = np.concatenate([p.UV for p in parts])
    # texel density: xatlas sizes charts by 3D area, so give each part its share by scaling it
    Pd = P * np.concatenate([np.full(len(p.P), p.density) for p in parts])[:, None]
    vmap, idx, uvs = xatlas.parametrize(Pd.astype(np.float32), F.astype(np.uint32))
    idx = idx.astype(np.int64)
    fid, bary = raster_bary(uvs, idx, size)
    m = fid >= 0
    f = idx[fid[m]]
    b = bary[m]
    src_v = vmap[f]                                          # (n, 3) original vertex ids
    part_of = pid[src_v[:, 0]]
    ouv = sum(b[:, k, None] * oldUV[src_v[:, k]] for k in range(3))
    tex = np.zeros((size, size, 3), np.float32)
    fmap = np.zeros((size, size, 3), np.float32)
    flat_t = np.zeros((int(m.sum()), 3), np.float32)
    flat_f = np.zeros((int(m.sum()), 3), np.float32)
    for i, p in enumerate(parts):
        sel = part_of == i
        if not sel.any():
            continue
        flat_t[sel] = sample(p.tex, ouv[sel])
        if p.fmap is not None:
            flat_f[sel] = sample(p.fmap, ouv[sel], order=0)
    tex[m] = flat_t
    fmap[m] = flat_f
    tex = dilate(np.clip(tex, 0, 255).astype(np.uint8), m, dilate_steps)
    return P[vmap], idx, uvs.astype(np.float32), tex, fmap, pid[vmap]


def neck_radii(body: Part, at: np.ndarray, reach: float = 0.13) -> tuple[float, float]:
    """The neck's half width / half depth just under ``at`` (only the body near the neck: the
    back and the shoulders at the same height do not count)."""
    near = (np.abs(body.P[:, 1] - (at[1] - 0.01)) < 0.03) & (np.abs(body.P[:, 2] - at[2]) < reach) & \
        (np.abs(body.P[:, 0]) < 2 * reach)
    if near.sum() < 10:
        return reach, reach
    rel = body.P[near] - at
    return float(np.percentile(np.abs(rel[:, 0]), 92)), float(np.percentile(np.abs(rel[:, 2]), 92))


def taper_collar(head: Part, body: Part, at: np.ndarray, band: float, slack: float = 1.12) -> Part:
    """Pull the bottom of a placed head piece (the chest fur cut with it) in to the body's neck, so
    the collar sits on the neck instead of flaring out like a skirt."""
    rx, rz = neck_radii(body, at)
    P = head.P.copy()
    y0 = float(P[:, 1].min())
    t = np.clip((P[:, 1] - y0) / max(band, 1e-6), 0, 1)
    rel = P[:, [0, 2]] - at[[0, 2]]
    # elliptic radius: 1 on the neck's outline (times the slack)
    e = np.sqrt((rel[:, 0] / (rx * slack)) ** 2 + (rel[:, 1] / (rz * slack)) ** 2)
    tgt = 1.0 + (e - 1.0) * t ** 1.5           # the rim lands on the neck's outline (in or out)
    tgt = np.where(e > 1e-6, tgt * np.minimum(1.0, e / 0.85), tgt)    # the middle stays inside
    P[:, [0, 2]] = at[[0, 2]] + rel * (tgt / np.maximum(e, 1e-9))[:, None]
    return Part(P, head.F, head.UV, head.tex, head.fmap, head.density, head.tag, dict(head.extra))


def recolor(tex: np.ndarray, ref: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Match a texture's colours to reference pixels (mean / spread per Lab channel)."""
    from skimage import color
    lab = color.rgb2lab(tex.astype(np.float32) / 255.0)
    rl = color.rgb2lab(ref.reshape(-1, 1, 3).astype(np.float32) / 255.0).reshape(-1, 3)
    valid = tex.reshape(-1, 3).sum(1) > 15
    src = lab.reshape(-1, 3)[valid]
    mu_s, sd_s = src.mean(0), src.std(0) + 1e-6
    mu_r, sd_r = rl.mean(0), rl.std(0) + 1e-6
    out = (lab - mu_s) / sd_s * sd_r + mu_r
    out = lab + (out - lab) * strength
    return (np.clip(color.lab2rgb(out), 0, 1) * 255 + 0.5).astype(np.uint8)


def collar_ring(centre: np.ndarray, rx: float, rz: float, tube: float, rgb, tilt: float = 0.0,
                n: int = 72, m: int = 12) -> Part:
    """A collar (elliptic torus) around the neck at the graft line: it covers the seam all round
    like the sheet's collar. ``tilt`` (deg) lowers its front."""
    u = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = np.linspace(0, 2 * np.pi, m, endpoint=False)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    flat = 1.6                                   # a band: taller than it is thick
    x = (rx + tube * np.cos(Vv)) * np.sin(U)
    z = (rz + tube * np.cos(Vv)) * np.cos(U)
    y = tube * flat * np.sin(Vv)
    P = np.stack([x, y, z], -1).reshape(-1, 3)
    if tilt:
        a = np.radians(tilt)
        c, s = np.cos(a), np.sin(a)
        P = P @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]]).T
    P = P + centre
    i, j = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    a = (i * m + j).ravel()
    b = (((i + 1) % n) * m + j).ravel()
    c = (((i + 1) % n) * m + (j + 1) % m).ravel()
    d = (i * m + (j + 1) % m).ravel()
    F = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    # leather: the colour with a lighter top edge and a darker bottom edge (v around the tube)
    S = 32
    tex = np.zeros((S, S, 3), np.float32)
    shade = 0.85 + 0.25 * np.sin(np.linspace(0, 2 * np.pi, S, endpoint=False))[None, :, None]
    tex[:] = np.asarray(rgb, np.float32)[None, None, :] * shade
    UV = np.stack([(i / n).ravel(), (j / m).ravel()], 1).astype(np.float32)
    UV = np.stack([np.full(len(UV), 0.5), UV[:, 1] * 0.98 + 0.01], 1).astype(np.float32)
    return Part(P, F[:, ::-1].copy(), UV, np.clip(tex, 0, 255).astype(np.uint8), None, 0.2, "collar")


def ball(centre: np.ndarray, r: float, rgb, n: int = 16, tag: str = "charm") -> Part:
    """A small shiny ball (a collar bell / tag) - a UV sphere with a soft highlight band."""
    th = np.linspace(0, np.pi, n // 2 + 1)
    ph = np.linspace(0, 2 * np.pi, n, endpoint=False)
    T, Ph = np.meshgrid(th, ph, indexing="ij")
    P = np.stack([np.sin(T) * np.sin(Ph), np.cos(T), np.sin(T) * np.cos(Ph)], -1).reshape(-1, 3) * r + centre
    rows, cols = T.shape
    i, j = np.meshgrid(np.arange(rows - 1), np.arange(cols), indexing="ij")
    a = (i * cols + j).ravel()
    b = (i * cols + (j + 1) % cols).ravel()
    c = ((i + 1) * cols + (j + 1) % cols).ravel()
    d = ((i + 1) * cols + j).ravel()
    F = np.concatenate([np.stack([a, c, b], 1), np.stack([a, d, c], 1)])
    S = 32
    g = np.linspace(0, 1, S)[:, None, None]
    tex = np.asarray(rgb, np.float32)[None, None, :] * (1.25 - 0.55 * g) * np.ones((S, S, 1), np.float32)
    UV = np.stack([np.full(len(P), 0.5), (T / np.pi).reshape(-1) * 0.98 + 0.01], 1).astype(np.float32)
    return Part(P, F, UV, np.clip(tex, 0, 255).astype(np.uint8), None, 0.2, tag)


def cap_holes(part: Part, min_loop: int = 8, lift: float = 0.0, max_extent: float | None = 0.45) -> Part:
    """Close the open cut(s) of a part with triangle fans, so the background never shows through a
    seam. Each fan triangle takes its texture from the boundary edge it closes (its centre vertex
    reuses the edge's first uv), so the cap is coloured like the rim."""
    key = np.round(part.P / 1e-6).astype(np.int64)
    _, ids = np.unique(key, axis=0, return_inverse=True)
    ids = ids.reshape(-1)
    Fw = ids[part.F]
    e = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]])
    ev = np.concatenate([part.F[:, [0, 1]], part.F[:, [1, 2]], part.F[:, [2, 0]]])
    k = np.sort(e, axis=1)
    _, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    border = np.flatnonzero(cnt[inv.reshape(-1)] == 1)
    if not len(border):
        return part
    nxt = {}
    for i in border:
        nxt.setdefault(int(e[i, 0]), []).append(i)
    used = set()
    P, F, UV = [part.P], [part.F], [part.UV]
    n = len(part.P)
    for start in border:
        if start in used:
            continue
        loop = []
        cur = int(start)
        while cur not in used:
            used.add(cur)
            loop.append(cur)
            cands = [j for j in nxt.get(int(e[cur, 1]), []) if j not in used]
            if not cands:
                break
            cur = cands[0]
        closed = int(e[loop[-1], 1]) == int(e[loop[0], 0])
        if len(loop) < min_loop or not closed:
            continue
        # only a compact, roughly flat cut: never a "loop" wandering over the model
        pts = part.P[ev[loop, 0]]
        ext = np.ptp(pts, axis=0)
        if max_extent is not None and float(ext.max()) > max_extent:
            continue
        a = ev[loop, 0]
        b = ev[loop, 1]
        c = part.P[a].mean(axis=0) + np.array([0.0, lift, 0.0])
        m = len(loop)
        # one centre vertex per triangle (its uv = the edge's first uv)
        P.append(np.repeat(c[None], m, axis=0))
        UV.append(part.UV[a])
        F.append(np.stack([b, a, n + np.arange(m)], 1))
        n += m
    return Part(np.concatenate(P), np.concatenate(F), np.concatenate(UV).astype(np.float32), part.tex, part.fmap,
                part.density, part.tag, dict(part.extra))
