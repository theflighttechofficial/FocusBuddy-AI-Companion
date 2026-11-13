"""Stage 1 of the 3D companions: reconstruct a textured 3D mesh from character-sheet artwork with
**TripoSR** (Stability AI / Tripo AI, MIT licence) - a feed-forward image-to-3D model - and
project the sheet art itself back onto the visible side of the mesh.

Two generations of sources live side by side:

* the first character sheets (``art/cells/<cell>.png``): small whole-body views, reconstructed into
  the standing / walking bodies (``cat``, ``dog``, ``fox``, ``robot``, ``buddy``, ``study_girl``,
  ``saree_girl``; ``rabbit`` is kept for reference);
* the detailed sheets (``art/cells/v2/<sheet>_front.png``, cut out by ``tools/build_detail_sheets.py``):
  the big front view of every character. These give the detailed heads (and the whole rabbit, which
  sits on both sheets); ``tools/build_3d_companions.py`` grafts each detailed head onto its body.

For each source

* TripoSR predicts a density / colour field; a triangle mesh is extracted (marching cubes),
  cleaned (largest parts only), smoothed and decimated to a real-time budget;
* xatlas unwraps it and the colour texture is baked by querying TripoSR's colour field at the 3D
  position of every texel (this covers the back and the sides);
* **projection**: the input view is projected back onto the mesh from TripoSR's input camera (the
  camera is fitted to the cut-out silhouette, a depth buffer decides what is visible), so every
  surface facing the viewer gets the sheet's own pixels - sharp eyes, fur strokes, fabric prints -
  instead of the blurrier colour field. The per-texel image coordinates and weights are saved as the
  *front map*, which the build stage uses to paint the facial expressions from the sheet panels.

Outputs ``art/meshes/<id>.npz`` (vertices, faces, uvs, TripoSR frame, z up), ``<id>.png`` (texture)
and ``<id>_fmap.npz`` (front map). Only stage 2 runs in normal development; this stage needs PyTorch
and a TripoSR checkout and downloads the weights from Hugging Face:

    pip install torch "transformers<5" "huggingface_hub<1" omegaconf einops xatlas fast-simplification scikit-image opencv-python
    git clone https://github.com/VAST-AI-Research/TripoSR   (or download the source)
    set TRIPOSR_DIR=path\\to\\TripoSR
    python tools/reconstruct_3d.py                      # every source
    python tools/reconstruct_3d.py cat_front boy_front  # some of them
    python tools/reconstruct_3d.py --project-only cat   # re-project an existing mesh (no TripoSR)

TripoSR's ``torchmcubes`` dependency is replaced by scikit-image's marching cubes (CPU), so no
compiler or GPU is needed; on a laptop CPU one source takes one to two minutes.
"""

from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CELLS = ROOT / "art" / "cells"
OUT = ROOT / "art" / "meshes"

# source id -> (input cut-out under art/cells, triangle budget, marching-cubes resolution, texture size,
#               project the input view onto the mesh)
SOURCES = {
    # first sheets: whole bodies
    "cat": ("cat_act_idle", 16000, 256, 1024, False),
    "dog": ("dog_act_idle", 16000, 256, 1024, False),
    "fox": ("fox_turn_right", 16000, 256, 1024, False),
    "rabbit": ("rabbit_turn_front34", 16000, 256, 1024, False),
    "robot": ("robot_turn_front", 18000, 256, 1024, False),
    "buddy": ("boy_turn_front", 18000, 256, 1024, False),
    "study_girl": ("girl_turn_front", 18000, 256, 1024, False),
    "saree_girl": ("saree_turn_front", 18000, 256, 1024, False),
    # detailed sheets: big front views (heads, and the whole rabbit)
    "cat_front": ("v2/cat_front", 30000, 320, 2048, True),
    "dog_front": ("v2/dog_front", 30000, 320, 2048, True),
    "fox_front": ("v2/fox_front", 30000, 320, 2048, True),
    "rabbit_front": ("v2/rabbit_front", 30000, 320, 2048, True),
    "boy_front": ("v2/boy_front", 30000, 320, 2048, True),
    "girl_front": ("v2/girl_front", 30000, 320, 2048, True),
    "saree_front": ("v2/saree_front", 30000, 320, 2048, True),
}
RATIO = 0.85          # foreground size in TripoSR's input square
CAM_DIST, FOVY = 1.9, 40.0


def _install_mcubes_shim() -> None:
    """``torchmcubes`` stand-in (scikit-image). Returns vertices in z, y, x order like torchmcubes."""
    import torch
    from skimage.measure import marching_cubes as _mc

    def marching_cubes(vol, thresh):
        a = vol.detach().cpu().numpy().astype(np.float32)
        v, f, _n, _ = _mc(a, float(thresh))
        return (torch.from_numpy(np.ascontiguousarray(v[:, ::-1]).astype(np.float32)),
                torch.from_numpy(np.ascontiguousarray(f[:, ::-1]).astype(np.int64)))

    mod = types.ModuleType("torchmcubes")
    mod.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = mod


def square_frame(path: Path, ratio: float = RATIO):
    """The cut-out on TripoSR's input square: (RGBA square array, crop x0, crop y0, pad x0, pad y0)."""
    from PIL import Image
    a = np.array(Image.open(path).convert("RGBA"))
    ys, xs = np.where(a[..., 3] > 16)
    x0, y0 = xs.min(), ys.min()
    a = a[y0:ys.max() + 1, x0:xs.max() + 1]
    h, w = a.shape[:2]
    side = int(max(h, w) / ratio)
    canvas = np.zeros((side, side, 4), np.uint8)
    py, px = (side - h) // 2, (side - w) // 2
    canvas[py:py + h, px:px + w] = a
    return canvas, int(x0), int(y0), px, py


def prepare_image(path: Path, ratio: float = RATIO):
    """Cut-out artwork -> TripoSR input: centred on a 50% grey square, foreground at ``ratio``."""
    from PIL import Image
    canvas = square_frame(path, ratio)[0]
    rgba = canvas.astype(np.float32) / 255
    rgb = rgba[..., :3] * rgba[..., 3:4] + 0.5 * (1 - rgba[..., 3:4])
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize((512, 512), Image.LANCZOS)


def clean_mesh(mesh, faces_target: int):
    """Keep the big parts, smooth the voxel steps away, decimate to the budget."""
    import fast_simplification
    import trimesh
    parts = mesh.split(only_watertight=False)
    big = max(len(p.faces) for p in parts)
    mesh = trimesh.util.concatenate([p for p in parts if len(p.faces) > 0.02 * big])
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=6)
    ratio = 1.0 - min(1.0, faces_target / len(mesh.faces))
    v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int64), ratio)
    out = trimesh.Trimesh(v, f, process=True)
    out.remove_unreferenced_vertices()
    return out


def raster_bary(uvs: np.ndarray, faces: np.ndarray, size: int):
    """Rasterise a UV atlas: for every texel the triangle covering it (-1: none) and its barycentrics."""
    fid = np.full((size, size), -1, np.int32)
    bary = np.zeros((size, size, 3), np.float32)
    P = uvs * size - 0.5
    for k, f in enumerate(faces):
        a, b, c = P[f]
        x0, y0 = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        x1, y1 = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        l1 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / d
        l2 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / d
        l3 = 1 - l1 - l2
        eps = -0.02
        ins = (l1 >= eps) & (l2 >= eps) & (l3 >= eps)
        if not ins.any():
            continue
        fid[gy[ins], gx[ins]] = k
        bary[gy[ins], gx[ins]] = np.stack([l1[ins], l2[ins], l3[ins]], -1)
    return fid, bary


def interp(fid: np.ndarray, bary: np.ndarray, faces: np.ndarray, attr: np.ndarray) -> np.ndarray:
    """Per-texel interpolation of a per-vertex attribute (zeros where no triangle)."""
    out = np.zeros(fid.shape + attr.shape[1:], np.float32)
    m = fid >= 0
    f = faces[fid[m]]
    b = bary[m]
    out[m] = sum(b[:, k, None] * attr[f[:, k]] for k in range(3))
    return out


def raster_uv(uvs: np.ndarray, faces: np.ndarray, verts: np.ndarray, size: int):
    """Rasterise the UV atlas: the 3D position (and a coverage mask) of every texel."""
    fid, bary = raster_bary(uvs, faces, size)
    return interp(fid, bary, faces, verts), fid >= 0


def dilate(tex: np.ndarray, mask: np.ndarray, steps: int = 8) -> np.ndarray:
    """Grow the islands outwards so bilinear filtering / mip levels never pick up the background."""
    tex = tex.copy()
    m = mask.copy()
    for _ in range(steps):
        acc = np.zeros_like(tex, dtype=np.float32)
        cnt = np.zeros(m.shape, np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            sm = np.roll(np.roll(m, dy, 0), dx, 1)
            st = np.roll(np.roll(tex, dy, 0), dx, 1)
            acc += st * sm[..., None]
            cnt += sm
        grow = (~m) & (cnt > 0)
        tex[grow] = (acc[grow] / cnt[grow, None]).astype(tex.dtype)
        m |= grow
    return tex


def vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


# ------------------------------------------------------------------ projection of the input view
def camera_project(p: np.ndarray, size: int, azimuth: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """TripoSR's cameras: azimuth 0 is the input view (at +x looking at the origin, +z up, image
    right = +y); other azimuths orbit about +z. Returns pixel coordinates on a ``size`` square and
    the depth."""
    if azimuth:
        a = np.radians(azimuth)
        c, s = np.cos(a), np.sin(a)
        p = p @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]).T     # rotate the world by -azimuth
    f = 0.5 * size / np.tan(np.radians(FOVY) / 2)
    z = CAM_DIST - p[:, 0]
    return np.stack([size / 2 + f * p[:, 1] / z, size / 2 - f * p[:, 2] / z], 1), z


def zbuffer(xy: np.ndarray, z: np.ndarray, faces: np.ndarray, size: int) -> np.ndarray:
    """Depth buffer of the projected mesh (inf where empty)."""
    zb = np.full((size, size), np.inf, np.float32)
    for f in faces:
        a, b, c = xy[f]
        x0, y0 = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        x1, y1 = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, size - 1), min(y1, size - 1)
        if x1 < x0 or y1 < y0:
            continue
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1] + 0.5
        l1 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / d
        l2 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / d
        l3 = 1 - l1 - l2
        ins = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not ins.any():
            continue
        zz = l1 * z[f[0]] + l2 * z[f[1]] + l3 * z[f[2]]
        iy, ix = (gy[ins] - 0.5).astype(int), (gx[ins] - 0.5).astype(int)
        zb[iy, ix] = np.minimum(zb[iy, ix], zz[ins])
    return zb


def fit_camera(verts: np.ndarray, faces: np.ndarray, alpha512: np.ndarray, azimuth: float = 0.0):
    """TripoSR does not reproduce the input framing exactly (and the other sheet views are drawn at
    their own scale): find the similarity (scale about the centre + shift, in 512-px units) that
    best lays the mesh silhouette over the cut-out. Returns (scale, dx, dy, IoU)."""
    import cv2
    xy, z = camera_project(verts, 512, azimuth)
    sil = np.isfinite(zbuffer(xy, z, faces, 512)).astype(np.float32)
    tgt = (alpha512 > 0.5).astype(np.float32)
    cm = np.array(np.nonzero(sil)[::-1]).mean(1)
    ct = np.array(np.nonzero(tgt)[::-1]).mean(1)

    def iou(s, tx, ty):
        M = np.array([[s, 0, 256 + tx - s * 256], [0, s, 256 + ty - s * 256]], np.float32)
        w = cv2.warpAffine(sil, M, (512, 512), flags=cv2.INTER_NEAREST)
        return float((w * tgt).sum() / max(((w + tgt) > 0).sum(), 1))

    best = (-1.0, 1.0, 0.0, 0.0)
    for s in np.arange(0.6, 1.45, 0.01):
        t = ct - 256 - s * (cm - 256)
        v = iou(s, t[0], t[1])
        if v > best[0]:
            best = (v, s, t[0], t[1])
    _, s, tx, ty = best
    for step in (2.0, 1.0, 0.5):
        improved = True
        while improved:
            improved = False
            for ds, dx, dy in ((0.005, 0, 0), (-0.005, 0, 0), (0, step, 0), (0, -step, 0), (0, 0, step), (0, 0, -step)):
                v = iou(s + ds, tx + dx, ty + dy)
                if v > best[0] + 1e-5:
                    best = (v, s + ds, tx + dx, ty + dy)
                    _, s, tx, ty = best
                    improved = True
    return s, tx, ty, best[0]


def project_one(verts, faces, pos, nrm, image_path: Path, azimuth: float):
    """One sheet view seen from ``azimuth``: per texel (colour, weight, image x, image y)."""
    import cv2
    from PIL import Image
    from scipy import ndimage
    sq, cx0, cy0, px, py = square_frame(image_path)
    side = sq.shape[0]
    a512 = cv2.resize(sq[..., 3], (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    s, tx, ty, iou = fit_camera(verts, faces, a512, azimuth)
    print(f"  {image_path.stem} @ {azimuth:g} deg: scale {s:.3f} shift ({tx:.1f}, {ty:.1f}) IoU {iou:.3f}", flush=True)
    R = 1024                                    # depth buffer resolution

    def to_px(p):                               # 3D -> depth-buffer pixels (R square) + depth
        xy, z = camera_project(p, 512, azimuth)
        return (256 + s * (xy - 256) + np.array([tx, ty])) * (R / 512), z
    xy, z = to_px(verts)
    zb = ndimage.minimum_filter(zbuffer(xy, z, faces, R), 3)    # a little slack at depth edges
    pxy, pz = to_px(pos)
    ix = np.clip(pxy[:, 0].astype(int), 0, R - 1)
    iy = np.clip(pxy[:, 1].astype(int), 0, R - 1)
    span = float(np.ptp(verts, axis=0).max())
    visible = pz <= zb[iy, ix] + 0.012 * span
    a = np.radians(azimuth)
    to_cam = np.array([CAM_DIST * np.cos(a), CAM_DIST * np.sin(a), 0.0]) - pos
    to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
    facing = (nrm * to_cam).sum(1)
    w = np.clip((facing - 0.12) / 0.33, 0, 1) * visible * np.clip((iou - 0.75) / 0.12, 0, 1)
    sx = pxy[:, 0] * side / R - px + cx0        # depth-buffer px -> input square -> cut-out pixels
    sy = pxy[:, 1] * side / R - py + cy0
    img = np.array(Image.open(image_path).convert("RGBA")).astype(np.float32)
    core = ndimage.binary_erosion(img[..., 3] > 250, iterations=3).astype(np.float32)
    core = ndimage.gaussian_filter(core, 1.5)
    rgb = np.stack([ndimage.map_coordinates(img[..., k], [sy, sx], order=1, mode="nearest") for k in range(3)], 1)
    w = w * ndimage.map_coordinates(core, [sy, sx], order=1, mode="constant", cval=0.0)
    return rgb, w, sx, sy


def project_view(verts, faces, uvs, tex, views: list, size: int):
    """Project sheet views onto the texture. ``views``: [(cut-out path, azimuth)], the first one the
    input (front) view, which wins wherever it sees the surface well; the others fill in the sides
    and the back. Returns (texture, front map [x, y, weight] per texel in the front cut-out's
    pixel coordinates)."""
    fid, bary = raster_bary(uvs, faces, size)
    m = fid >= 0
    pos = interp(fid, bary, faces, verts)[m]
    nrm = interp(fid, bary, faces, vertex_normals(verts, faces))[m]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    field = tex.astype(np.float32)[m]
    rgb0, w0, sx, sy = project_one(verts, faces, pos, nrm, views[0][0], views[0][1])
    # the colour field is duller / darker than the art: fit an affine colour map field -> art on the
    # texels the front view sees well and apply it to the whole field (back and sides then match)
    sure = w0 > 0.8
    if sure.sum() > 500:
        X = np.concatenate([field[sure], np.ones((int(sure.sum()), 1), np.float32)], 1)
        A, *_ = np.linalg.lstsq(X, rgb0[sure], rcond=None)
        field = np.concatenate([field, np.ones((len(field), 1), np.float32)], 1) @ A
    acc = np.zeros_like(field)
    wsum = np.zeros(len(field), np.float32)
    for path, az in views[1:]:
        if not Path(path).exists():
            continue
        rgb, w, _x, _y = project_one(verts, faces, pos, nrm, Path(path), az)
        acc += rgb * w[:, None]
        wsum += w
    other = np.where(wsum[:, None] > 1e-6, acc / np.maximum(wsum, 1e-6)[:, None], field)
    w_other = np.clip(wsum, 0, 1) * (1 - w0)
    flat = field * (1 - w0 - w_other)[:, None] + rgb0 * w0[:, None] + other * w_other[:, None]
    out = np.zeros(tex.shape, np.float32)
    out[m] = flat
    fmap = np.zeros((size, size, 3), np.float32)
    fmap[m] = np.stack([sx, sy, w0], 1)
    tex_out = dilate(np.clip(out, 0, 255).astype(np.uint8), m)
    print(f"  front view covers {100 * float((w0 > 0.5).mean()):.0f}% of the surface, the other views "
          f"{100 * float((w_other > 0.5).mean()):.0f}%", flush=True)
    return tex_out, fmap


def front_map(cid: str) -> np.ndarray:
    """The front map of a projected source, recomputed from its saved mesh and the cut-out (no
    TripoSR needed); cached as ``art/meshes/<id>_fmap.npz`` (a build cache, not committed)."""
    cell, _b, _r, size, _p = SOURCES[cid]
    z = np.load(OUT / f"{cid}.npz")
    verts, faces, uvs = z["verts"].astype(np.float32), z["faces"].astype(np.int64), z["uvs"]
    fid, bary = raster_bary(uvs, faces, size)
    m = fid >= 0
    pos = interp(fid, bary, faces, verts)[m]
    nrm = interp(fid, bary, faces, vertex_normals(verts, faces))[m]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    _rgb, w, sx, sy = project_one(verts, faces, pos, nrm, CELLS / f"{cell}.png", 0.0)
    fmap = np.zeros((size, size, 3), np.float32)
    fmap[m] = np.stack([sx, sy, w], 1)
    np.savez_compressed(OUT / f"{cid}_fmap.npz", fmap=fmap.astype(np.float16), image=np.array(cell))
    return fmap


def views_for(cell: str) -> list:
    """The sheet views projected onto a source: its own input view. (The detailed sheets' side and
    back views were tried as extra projections; a single-view reconstruction's depth is too far off
    for them to line up - silhouette IoU 0.6 - 0.78 against 0.95+ for the front - so the sides and
    the back keep the colour-corrected TripoSR field.)"""
    return [(CELLS / f"{cell}.png", 0.0)]


# ------------------------------------------------------------------ reconstruction
def reconstruct(cid: str, model, device: str = "cpu") -> dict:
    import torch
    import xatlas
    cell, budget, mc_res, tex_size, project = SOURCES[cid]
    t0 = time.time()
    img = prepare_image(CELLS / f"{cell}.png")
    with torch.no_grad():
        codes = model([img], device=device)
        raw = model.extract_mesh(codes, False, resolution=mc_res)[0]
    mesh = clean_mesh(raw, budget)
    vmap, idx, uvs = xatlas.parametrize(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    verts = mesh.vertices[vmap].astype(np.float32)
    idx = idx.astype(np.int64)
    fid, bary = raster_bary(uvs, idx, tex_size)
    mask = fid >= 0
    pos = interp(fid, bary, idx, verts)
    # query the colour field a hair inside the surface (the field is sharpest just below it)
    npos = interp(fid, bary, idx, mesh.vertex_normals[vmap].astype(np.float32))
    q = (pos - 0.004 * npos)[mask]
    cols = []
    with torch.no_grad():
        for s in range(0, len(q), 65536):
            pts = torch.from_numpy(q[s:s + 65536]).to(device)
            cols.append(model.renderer.query_triplane(model.decoder, pts, codes[0])["color"].cpu().numpy())
    tex = np.zeros((tex_size, tex_size, 3), np.float32)
    tex[mask] = np.concatenate(cols)
    tex = dilate(tex, mask)
    tex = (np.clip(tex, 0, 1) * 255 + 0.5).astype(np.uint8)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{cid}.npz", verts=verts, faces=idx.astype(np.int32), uvs=uvs.astype(np.float32),
                        source=np.array(cell))
    from PIL import Image
    Image.fromarray(tex).save(OUT / f"{cid}_field.png", optimize=True)   # the colour field alone
    if project:
        tex, fmap = project_view(verts, idx, uvs, tex, views_for(cell), tex_size)
        np.savez_compressed(OUT / f"{cid}_fmap.npz", fmap=fmap.astype(np.float16), image=np.array(cell))
    Image.fromarray(tex).save(OUT / f"{cid}.png", optimize=True)        # row = v * size (glTF: top-left origin)
    return {"id": cid, "cell": cell, "faces": int(len(idx)), "verts": int(len(verts)),
            "seconds": round(time.time() - t0, 1)}


def reproject(cid: str) -> dict:
    """Projection only, on an existing mesh + colour field (no TripoSR needed)."""
    from PIL import Image
    cell, _b, _r, tex_size, _p = SOURCES[cid]
    z = np.load(OUT / f"{cid}.npz")
    field = OUT / f"{cid}_field.png"
    tex = np.array(Image.open(field if field.exists() else OUT / f"{cid}.png").convert("RGB"))
    if not field.exists():
        Image.fromarray(tex).save(field, optimize=True)
    tex, fmap = project_view(z["verts"].astype(np.float32), z["faces"].astype(np.int64), z["uvs"], tex,
                             views_for(cell), tex.shape[0])
    np.savez_compressed(OUT / f"{cid}_fmap.npz", fmap=fmap.astype(np.float16), image=np.array(cell))
    Image.fromarray(tex).save(OUT / f"{cid}.png", optimize=True)
    return {"id": cid, "projected": True}


def main(argv: list[str]) -> int:
    if argv[:1] == ["--project-only"]:
        for cid in argv[1:]:
            print(reproject(cid), flush=True)
        return 0
    tsr_dir = os.environ.get("TRIPOSR_DIR")
    if not tsr_dir or not Path(tsr_dir, "tsr").is_dir():
        print("Set TRIPOSR_DIR to a TripoSR source checkout (see the module docstring).")
        return 2
    sys.path.insert(0, tsr_dir)
    _install_mcubes_shim()
    import torch
    from tsr.system import TSR
    torch.set_num_threads(os.cpu_count() or 4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(device).eval()
    for cid in (argv or list(SOURCES)):
        print(reconstruct(cid, model, device), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
