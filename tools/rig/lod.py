"""Levels of detail for a rigged companion: the mesh decimated in Blender (``blender_decimate.py``),
then every per-vertex attribute of the full mesh - skin weights, morph offsets - carried over by
the closest point on the full surface (barycentric interpolation inside its triangle).

The painted face needs nothing: it lives in the texture, which every level shares.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# triangles per level (the full mesh is "high"): the CPU renderer draws "medium" at the default
# quality, "low" at the low quality tier
LEVELS = {"medium": 20000, "low": 9000}


def decimate(V, F, FUV, target: int, lock: np.ndarray | None = None):
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in.npz", Path(d) / "out.npz"
        extra = {} if lock is None else {"lock": lock}
        np.savez(src, V=V, F=F, FUV=FUV, target=np.array(target), **extra)
        r = subprocess.run([sys.executable, str(Path(__file__).with_name("blender_decimate.py")), str(src), str(dst)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not dst.is_file():
            raise RuntimeError(f"decimation failed: {r.stderr[-800:]}")
        z = np.load(dst)
        return z["V"], z["F"], z["FUV"]


def closest(V, F, P):
    """(triangle index, barycentric (k, 3)) of the closest point on the mesh (V, F) to each P."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    pts, _, tri = trimesh.proximity.closest_point(mesh, P)
    bary = trimesh.triangles.points_to_barycentric(V[F[tri]], pts)
    bary = np.clip(bary, 0.0, 1.0)
    bary /= np.maximum(bary.sum(1, keepdims=True), 1e-12)
    return tri, bary


def carry(F, tri, bary, A: np.ndarray) -> np.ndarray:
    """Per-vertex attribute A (n, ...) of the full mesh, at the closest points."""
    c = F[tri]
    shape = (-1,) + (1,) * (A.ndim - 1)
    return sum(bary[:, i].reshape(shape) * A[c[:, i]] for i in range(3))


def decimate_parts(V, F, FUV, target: int, keep: np.ndarray, share: float):
    """Decimate two face groups separately: ``keep`` (e.g. a detailed head) gets ``share`` of the
    triangle budget, the rest the remainder - the face stays smooth on the lower levels. Vertices
    the two groups share (a welded seam) stay where they are in both, and are merged again: the
    level has no crack along the seam."""
    used_by = [np.unique(F[~keep]), np.unique(F[keep])]
    seam = np.intersect1d(*used_by)
    outs = []
    for mask, t, used in ((~keep, int(target * (1 - share)), used_by[0]), (keep, int(target * share), used_by[1])):
        Fm = F[mask]
        rm = np.full(len(V), -1, np.int64)
        rm[used] = np.arange(len(used))
        outs.append(decimate(V[used], rm[Fm], FUV[mask], min(t, len(Fm)), lock=np.isin(used, seam)))
    (Va, Fa, Ua), (Vb, Fb, Ub) = outs
    Fb = Fb + len(Va)
    Vall = np.concatenate([Va, Vb])
    if len(seam):
        from scipy.spatial import cKDTree
        d, nn = cKDTree(Va).query(Vb, k=1)
        same = d < 1e-7
        remap = np.arange(len(Vall))
        remap[len(Va) + np.flatnonzero(same)] = nn[same]
        Fb = remap[Fb]
    F2 = np.concatenate([Fa, Fb])
    used = np.unique(F2)
    rm = np.full(len(Vall), -1, np.int64)
    rm[used] = np.arange(len(used))
    return Vall[used], rm[F2], np.concatenate([Ua, Ub])


def level(V, F, FUV, W, morphs: dict | None, target: int, normals: np.ndarray | None = None,
          keep: np.ndarray | None = None, share: float = 0.45):
    """(V, F, FUV, W, morphs, normals or None) of one level. Custom ``normals`` (smoothed shading)
    are carried over like the weights; without them the level computes its own. ``keep``: a face
    group decimated on its own with ``share`` of the budget."""
    if keep is not None and keep.any():
        V2, F2, FUV2 = decimate_parts(V, F, FUV, target, keep, share)
    else:
        V2, F2, FUV2 = decimate(V, F, FUV, target)
    tri, bary = closest(V, F, V2)
    W2 = carry(F, tri, bary, W)
    W2 = np.clip(W2, 0.0, None)
    W2 /= np.maximum(W2.sum(1, keepdims=True), 1e-12)
    m2 = {k: carry(F, tri, bary, off) for k, off in (morphs or {}).items()}
    N2 = None
    if normals is not None:
        from export import vertex_normals
        own = vertex_normals(V2, F2)
        N2 = carry(F, tri, bary, normals)
        # where the full mesh kept its plain normals, the level keeps its own (decimation moves
        # creases); in the smoothed region the carried normals win
        plain = carry(F, tri, bary, vertex_normals(V, F))
        changed = np.clip(np.linalg.norm(N2 - plain, axis=1) / 0.05, 0, 1)[:, None]
        N2 = own * (1 - changed) + N2 * changed
        N2 /= np.maximum(np.linalg.norm(N2, axis=1, keepdims=True), 1e-12)
    return V2, F2, FUV2, W2, m2, N2
