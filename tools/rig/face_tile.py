"""A sharp face for the people: their own texture tile, retouched.

The generator spends very little of its texture atlas on a person's face (an eye is about 25
texels wide at 4096², 12 after export), and its texture bake leaves a horizontal seam across the
face where two views meet - on Study Girl it cuts the eyes in half. So the front of each person's
face gets **its own texture tile** appended under the atlas:

1. the faces of the front of the face that the front view sees completely (not hidden behind a
   strand of hair) are mapped onto a square tile by a planar front projection (they face the
   viewer, so the stretch is small);
2. the tile is the front view of that region rendered from the full-resolution source texture
   with bilinear sampling (2x super-sampled) - eight times the texels the face had after export;
3. the tile is retouched as a 2D picture (``beauty.py``): the bake seam blended away, cut eyes
   completed, skin evened out, eyes / lashes / lips made crisper;
4. the atlas grows from 4096 x 4096 to 4096 x 6144 (2048 x 3072 after export) with the tile in
   the new bottom rows; every other texture coordinate is rescaled, nothing else moves.

The geometry, skeleton, weights and morphs are untouched: only texture coordinates change.
"""

from __future__ import annotations

import numpy as np

from facerender import ortho

TILE = 2048          # tile size at the source resolution (4096 atlas) -> 1024 after export


def region(J: dict, spec: dict) -> tuple:
    """Square box (x0, x1, z0, z1) around the face, from the eyes and the mouth."""
    eyes = spec["eyes"]
    ex = float(np.mean([e[0] for e in eyes]))
    ez = float(np.mean([e[1] for e in eyes]))
    eye_dist = abs(eyes[1][0] - eyes[0][0])
    mz = spec["mouth"][2]
    em = ez - mz                                      # eyes to mouth
    top = ez + 1.05 * em                              # brows / lower forehead
    bot = mz - 0.95 * em                              # chin
    half_w = 1.25 * eye_dist
    side = max(top - bot, 2 * half_w)
    cz = 0.5 * (top + bot)
    return (ex - side / 2, ex + side / 2, cz - side / 2, cz + side / 2)


def bilinear(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    H, W = tex.shape[:2]
    x = np.clip(uv[:, 0] * W - 0.5, 0, W - 1.001)
    y = np.clip(uv[:, 1] * H - 0.5, 0, H - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    t = tex.astype(np.float32)
    a = t[y0, x0] * (1 - fx) + t[y0, x0 + 1] * fx
    b = t[y0 + 1, x0] * (1 - fx) + t[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


def select_faces(V, F, box, r, n_max: float = -0.35, visible: float = 0.97) -> np.ndarray:
    """Faces inside the box, facing the viewer, fully visible in the front view ``r``."""
    x0, x1, z0, z1 = box
    cen = V[F].mean(1)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    area = np.linalg.norm(fn, axis=1)
    fn /= np.maximum(area[:, None], 1e-12)
    inside = (cen[:, 0] > x0) & (cen[:, 0] < x1) & (cen[:, 2] > z0) & (cen[:, 2] < z1)
    cand = np.flatnonzero(inside & (fn[:, 1] < n_max))
    # visible fraction: pixels showing the face / its projected area in pixels
    sc = r["scale"]
    shown = np.bincount(r["fid"][r["fid"] >= 0].ravel(), minlength=len(F)).astype(float)
    P = V[F][:, :, [0, 2]] * sc
    proj = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) -
                        (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1]))
    ok = shown[cand] >= visible * proj[cand] - 2.0          # small faces: a couple of pixels of slack
    sel = cand[ok]
    # keep the main connected patch (the face), drop stray bits (an ear tip, a strand end)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    Fs = F[sel]
    E = np.concatenate([Fs[:, [0, 1]], Fs[:, [1, 2]], Fs[:, [2, 0]]])
    n = len(V)
    G = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    _, lab = connected_components(G, directed=False)
    fl = lab[Fs[:, 0]]
    vals, cnt = np.unique(fl, return_counts=True)
    return sel[fl == vals[np.argmax(cnt)]]


def build(V, F, FUV, tex, J, spec, tile: int = TILE, ss: int = 2):
    """Returns (new texture (H + tile, W, 3), new FUV, info dict with the tile picture and its
    region) - the tile is the plain front view; ``beauty.retouch`` edits it afterwards."""
    box = region(J, spec)
    S = tile * ss
    r = ortho(V, F, FUV, tex, box, S)
    sel = select_faces(V, F, box, r)
    keep = np.zeros(len(F), bool)
    keep[sel] = True
    # the tile picture: bilinear samples of the source at every pixel of the selected faces
    m = r["fid"] >= 0
    m &= keep[np.maximum(r["fid"], 0)]
    img = np.zeros((S, S, 3), np.float32)
    img[m] = bilinear(tex, r["uv"][m])
    cov = m.astype(np.float32)
    if ss > 1:
        img = img.reshape(tile, ss, tile, ss, 3).sum((1, 3))
        cov = cov.reshape(tile, ss, tile, ss).sum((1, 3))
        img = img / np.maximum(cov[..., None], 1e-6)
        covered = cov > 0
    else:
        covered = m
    img = fill_gutter(img, covered)
    H, W = tex.shape[:2]
    out = np.zeros((H + tile, W, 3), np.uint8)
    out[:H] = tex
    out[H:, :tile] = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    out[H:, tile:] = tex[H - tile:H, tile:].mean((0, 1)).astype(np.uint8)    # unused: a neutral colour
    # texture coordinates: the atlas squeezed into the top rows, the tile's faces into the tile
    NH = H + tile
    FUV2 = FUV.copy()
    FUV2[..., 1] = FUV[..., 1] * H / NH
    x0, x1, z0, z1 = r["box"]
    span = max(x1 - x0, z1 - z0)
    P = V[F[sel]]
    u = (P[..., 0] - x0) / span                      # 0..1 across the tile
    v = 1.0 - (P[..., 2] - z0) / span                # down the tile (glTF v down)
    FUV2[sel, :, 0] = u * tile / W
    FUV2[sel, :, 1] = (H + v * tile) / NH
    info = {"box": tuple(float(b) for b in r["box"]), "faces": sel, "tile": tile, "atlas": (H, W),
            "covered": covered}
    return out, FUV2, info


def fill_gutter(img: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Texels outside the selected faces take their nearest covered colour (no dark bleed when the
    renderer filters across a tile edge)."""
    from scipy import ndimage
    _, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    return img[iy, ix]


def tile_to_world(info: dict, px: float, py: float) -> tuple:
    """World (x, z) of tile pixel (px, py)."""
    x0, x1, z0, z1 = info["box"]
    span = max(x1 - x0, z1 - z0)
    t = info["tile"]
    return x0 + (px + 0.5) / t * span, z0 + (1.0 - (py + 0.5) / t) * span


def world_to_tile(info: dict, x: float, z: float) -> tuple:
    x0, x1, z0, z1 = info["box"]
    span = max(x1 - x0, z1 - z0)
    t = info["tile"]
    return (x - x0) / span * t - 0.5, (1.0 - (z - z0) / span) * t - 0.5


def apply(cid: str, model, V, J: dict, log: list | None = None) -> dict:
    """Give a person's model its face tile, retouched: replaces ``model.tex`` and ``model.FUV``.
    Returns the tile info."""
    import beauty
    import face
    spec = face.FACE[cid]
    hz = J["HeadTop"][2]
    h = hz - J["Neck"][2]
    hx = J["Head"][0]
    r0 = ortho(V, model.F, model.FUV, model.tex, (hx - 0.6 * h, hx + 0.6 * h, hz - 1.25 * h, hz - 0.05 * h), 1400)
    eyes = [face.refine_eye(r0, e, True) for e in spec["eyes"]]
    mouth = face.refine_mouth(r0, spec["mouth"])
    spec2 = dict(spec, eyes=eyes, mouth=mouth)
    tex, FUV, info = build(V, model.F, model.FUV, model.tex, J, spec2)
    span = info["box"][1] - info["box"][0]
    s = info["tile"] / span
    eyes_px = []
    for cx, cz, a, b, _ang in eyes:
        px, py = world_to_tile(info, cx, cz)
        eyes_px.append((px, py, a * s, b * s))
    (lx, _lz), (rx, _rz), mz = mouth
    mpx, mpy = world_to_tile(info, 0.5 * (lx + rx), mz)
    H, _W = info["atlas"]
    t = info["tile"]
    img = tex[H:, :t].astype(np.float32)
    out = beauty.retouch(cid, img, info["covered"], eyes_px, (mpx, mpy, abs(rx - lx) / 2 * s), log)
    out = fill_gutter(out, info["covered"])
    tex[H:, :t] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    model.tex, model.FUV = tex, FUV
    return info


def smooth_normals(V, F, info: dict, iters: int = 30) -> np.ndarray:
    """Vertex normals with the face's shading smoothed: the generated face is a few large flat
    triangles, so plain vertex normals shade it in hard facets (a "low-poly" nose and cheeks).
    Inside the face tile's region the normals are averaged with their neighbours (positions never
    move - only shading), fading to the plain normals at the region's edge."""
    from scipy.sparse import coo_matrix, diags
    from export import vertex_normals
    N = vertex_normals(V, F)
    n = len(V)
    sel = np.zeros(n, bool)
    sel[F[info["faces"]].ravel()] = True
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    P = diags(1.0 / np.maximum(deg, 1)) @ A
    # weight: 1 well inside the region, fading over a few rings to 0 outside
    w = sel.astype(float)
    for _ in range(4):
        w = np.minimum(w, P @ w + 0.0)
    w = np.clip(w, 0, 1)
    S = N.copy()
    for _ in range(iters):
        S = 0.5 * S + 0.5 * (P @ S)
        S /= np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
    out = N * (1 - w[:, None]) + S * w[:, None]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
