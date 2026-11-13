"""Weld a new head's neck onto a body's neck: one continuous surface, one skin tone.

``head_swap.swap`` places the new head; these are the geometric pieces of the join:

- ``first_hits`` / ``neck_axis`` / ``clean_depth``: rays from a vertical axis find a neck's
  centre and radius at a height, and how far under the jaw the new head's neck stays a clean
  tube (where hair or a collar joins it, the section jumps);
- ``deform_neck``: bends a neck's cross-section toward another's, fading out along the neck and
  away from it (hair, collar and shoulders further out keep still);
- ``clip``: splits faces exactly on a plane, so each cut leaves a clean outline (``outline``
  walks it into an ordered ring);
- ``tube``: joins the head's ring to the body's with a short band, the rings matched by their
  length - it follows each ring's real shape;
- ``shell``: the outer surface of the new head, to tell what it covers.
"""

from __future__ import annotations

import numpy as np


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def clip(V, F, FUV, W, mask, z, keep_above: bool, return_src: bool = False, edges: list | None = None):
    """Split the faces in ``mask`` at the plane Z = z, keeping the part above (or below). Faces not in
    the mask are untouched. Returns (V, F, FUV, W, cut vertex indices) and, with ``return_src``,
    the index of the input face each output face comes from. ``edges``: a list that receives the
    new edges along the cut (pairs of cut vertices), the cut's outline."""
    V = V.copy()
    side = (V[:, 2] >= z) if keep_above else (V[:, 2] <= z)
    near = np.abs(V[:, 2] - z) < 1e-7
    V[near, 2] += 2e-7 if keep_above else -2e-7
    side = (V[:, 2] >= z) if keep_above else (V[:, 2] <= z)
    newV, newW, cut_key = [], [], {}

    def cut_vertex(i, j):
        key = (min(i, j), max(i, j))
        if key not in cut_key:
            t = (z - V[i, 2]) / (V[j, 2] - V[i, 2])
            p = V[i] + t * (V[j] - V[i])
            p[2] = z
            cut_key[key] = len(V) + len(newV)
            newV.append(p)
            newW.append(W[i] + t * (W[j] - W[i]))
        return cut_key[key]

    outF, outUV, outS = [], [], []
    keepF = np.zeros(len(F), bool)
    for fi in range(len(F)):
        f = F[fi]
        if not mask[fi]:
            keepF[fi] = True
            continue
        s = side[f]
        if s.all():
            keepF[fi] = True
            continue
        if not s.any():
            continue
        # polygon clip (Sutherland-Hodgman against one plane), uv interpolated per corner
        poly, puv, cuts = [], [], []
        uv = FUV[fi]
        for k in range(3):
            a, b = k, (k + 1) % 3
            ia, ib = f[a], f[b]
            if s[a]:
                poly.append(ia)
                puv.append(uv[a])
            if s[a] != s[b]:
                t = (z - V[ia, 2]) / (V[ib, 2] - V[ia, 2])
                poly.append(cut_vertex(ia, ib))
                cuts.append(poly[-1])
                puv.append(uv[a] + t * (uv[b] - uv[a]))
        if edges is not None and len(cuts) == 2:
            edges.append((cuts[0], cuts[1]))
        for k in range(1, len(poly) - 1):
            outF.append([poly[0], poly[k], poly[k + 1]])
            outUV.append([puv[0], puv[k], puv[k + 1]])
            outS.append(fi)
    if newV:
        V = np.concatenate([V, np.array(newV)])
        W = np.concatenate([W, np.array(newW)])
    F2 = np.concatenate([F[keepF], np.array(outF, np.int64).reshape(-1, 3)])
    UV2 = np.concatenate([FUV[keepF], np.array(outUV).reshape(-1, 3, 2)])
    cut = np.array(sorted(cut_key.values()), np.int64)
    if return_src:
        return V, F2, UV2, W, cut, np.concatenate([np.flatnonzero(keepF), np.array(outS, np.int64)])
    return V, F2, UV2, W, cut


def first_hits(V, F, p, zs, nth: int = 48, rmax: float = 0.06, last: bool = False):
    """Distance from the vertical axis through p = (x, y) to the first surface hit by horizontal
    rays (the last one with ``last``: the outer shell), for every height in zs and nth angles:
    (angles, R (len(zs), nth) with NaN = no hit)."""
    import trimesh
    m = trimesh.Trimesh(V, F, process=False)
    th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    zs = np.atleast_1d(zs)
    O = np.array([[p[0], p[1], z] for z in zs for _ in th])
    D = np.array([[np.cos(t), np.sin(t), 0.0] for _ in zs for t in th])
    loc, ri, _ti = m.ray.intersects_location(O, D, multiple_hits=True)
    R = np.full(len(O), np.nan)
    if len(ri):
        d = np.hypot(loc[:, 0] - O[ri, 0], loc[:, 1] - O[ri, 1])
        o = np.lexsort((-d if last else d, ri))
        ri, d = ri[o], d[o]
        first = np.r_[True, ri[1:] != ri[:-1]]
        R[ri[first]] = d[first]
    R[R > rmax] = np.nan
    return th, R.reshape(len(zs), len(th))


def neck_axis(V, F, starts, z, rmax: float = 0.06):
    """Centre (x, y), median radius and ray hit fraction of the neck at height z: from each start
    point, rays out to the first surface, re-centred on the hits a few times. The first start
    giving a mostly closed section wins. None if no start is inside a neck."""
    best = None
    for p in starts:
        p = np.array(p[:2], float)
        ok = True
        for _ in range(4):
            th, R = first_hits(V, F, p, [z], 48, rmax)
            r = R[0]
            hit = ~np.isnan(r)
            if hit.sum() < 20:
                ok = False
                break
            P = np.stack([p[0] + r[hit] * np.cos(th[hit]), p[1] + r[hit] * np.sin(th[hit])], 1)
            p = 0.5 * (P.min(0) + P.max(0))
        if not ok:
            continue
        th, R = first_hits(V, F, p, [z], 48, rmax)
        frac = float(np.mean(~np.isnan(R[0])))
        cand = (p, float(np.nanmedian(R[0])), frac)
        if frac >= 0.6:
            return cand
    return best


def deform_neck(V, centre, radius, target_c, target_r, z_full, z_none, amount: float = 1.0):
    """Move a neck's vertices toward another neck: at z_full (and beyond, away from z_none) its
    cross-section is moved to target_c and scaled to target_r by ``amount``; the change fades out
    toward z_none and away from the neck (hair, collar and shoulders further out keep still)."""
    V = V.copy()
    w = smooth((V[:, 2] - z_none) / (z_full - z_none)) * amount
    d = V[:, :2] - np.asarray(centre)[None]
    rad = np.hypot(d[:, 0], d[:, 1])
    f = 1 - smooth((rad - 1.25 * radius) / (0.6 * radius))
    w = w * f
    k = 1 + w * (target_r / radius - 1)
    V[:, :2] = np.asarray(centre)[None] + d * k[:, None] + w[:, None] * (np.asarray(target_c) - np.asarray(centre))[None]
    return V


def angular_colour(px_pos, px_rgb, c, nth: int = 64):
    """Median colour per angle bin about c (x, y) of texels at positions px_pos; empty bins filled
    from their neighbours around the circle, then smoothed. Returns (nth, 3)."""
    ang = np.arctan2(px_pos[:, 1] - c[1], px_pos[:, 0] - c[0])
    b = ((ang + np.pi) / (2 * np.pi) * nth).astype(int) % nth
    out = np.full((nth, 3), np.nan)
    for k in range(nth):
        s = b == k
        if s.sum() >= 3:
            out[k] = np.median(px_rgb[s], axis=0)
    have = ~np.isnan(out[:, 0])
    if not have.any():
        return np.tile(np.median(px_rgb, axis=0) if len(px_rgb) else np.array([210.0, 160, 140]), (nth, 1))
    idx = np.flatnonzero(have)
    xs = np.arange(nth)
    for ch in range(3):
        xp = np.concatenate([idx - nth, idx, idx + nth])
        fp = np.tile(out[idx, ch], 3)
        out[:, ch] = np.interp(xs, xp, fp)
    # circular smoothing
    kern = np.array([1, 2, 3, 2, 1], float)
    kern /= kern.sum()
    pad = np.concatenate([out[-2:], out, out[:2]])
    return np.stack([np.convolve(pad[:, ch], kern, mode="valid") for ch in range(3)], 1)


def clean_depth(V, F, starts, top, depths):
    """The deepest of ``depths`` below ``top`` where the neck is still one clean tube: closed (rays
    hit all round), not thicker than near the top, and not jumping sideways (where hair or a collar
    joins it, the section suddenly grows / moves). Returns (depth, axis result)."""
    first = None
    best = None
    for dpt in depths:
        a = neck_axis(V, F, starts, top - dpt)
        if a is None or a[2] < 0.9:
            break
        if first is None:
            first = a
        elif np.hypot(*(a[0] - first[0])) > 0.008 or a[1] > 1.3 * first[1]:
            break
        best = (dpt, a)
        starts = [a[0]]
    return best


def shell(V, F, p, z0, z1, step: float = 0.004, nth: int = 48):
    """The outer surface of a head around the vertical axis through p, as a function of angle and
    height: returns inside(points, margin) -> bool, true where a point is under that surface."""
    zs = np.arange(z0, z1 + step, step)
    th, R = first_hits(V, F, p, zs, nth, rmax=0.2, last=True)

    def inside(P, margin: float = 0.003):
        ang = np.arctan2(P[:, 1] - p[1], P[:, 0] - p[0])
        k = np.round((ang + np.pi) / (2 * np.pi) * nth).astype(int) % nth
        j = np.clip(np.round((P[:, 2] - z0) / step).astype(int), 0, len(zs) - 1)
        r = np.hypot(P[:, 0] - p[0], P[:, 1] - p[1])
        lim = R[j, k]
        ok = (P[:, 2] >= z0) & (P[:, 2] <= z1)
        return ok & ~np.isnan(lim) & (r < np.nan_to_num(lim) - margin)
    return inside


def outline(V, edges, keep, c):
    """The cut's main loop: cut edges between vertices in ``keep`` (vertices at the same place - a
    texture seam - count as one), walked into an ordered ring counter-clockwise about c (x, y).
    Returns vertex indices (one per place)."""
    E = np.array([e for e in edges if keep[e[0]] and keep[e[1]]], np.int64).reshape(-1, 2)
    if len(E) == 0:
        return np.zeros(0, np.int64)
    ids = np.unique(E)
    key = np.round(V[ids] / 1e-7).astype(np.int64)
    _, rep_of = np.unique(key, axis=0, return_inverse=True)
    rep_of = rep_of.ravel()
    first = np.full(rep_of.max() + 1, -1, np.int64)
    for i in range(len(ids) - 1, -1, -1):
        first[rep_of[i]] = ids[i]
    lookup = dict(zip(ids.tolist(), rep_of.tolist()))
    adj = {}
    for a, b in E.tolist():
        ra, rb = lookup[a], lookup[b]
        if ra != rb:
            adj.setdefault(ra, set()).add(rb)
            adj.setdefault(rb, set()).add(ra)
    # components; the main one has the most vertices
    seen, comps = set(), []
    for r0 in adj:
        if r0 in seen:
            continue
        stack, comp = [r0], []
        seen.add(r0)
        while stack:
            r = stack.pop()
            comp.append(r)
            for n in adj[r]:
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        comps.append(comp)
    comp = max(comps, key=len)
    # walk counter-clockwise about c: from each vertex to the unvisited neighbour a small step
    # further round (where other surfaces touch the cut, the loop branches)
    ang = {r: float(np.arctan2(V[first[r], 1] - c[1], V[first[r], 0] - c[0])) for r in comp}
    ends = [r for r in comp if len(adj[r]) == 1]
    cur = ends[0] if ends else min(comp, key=lambda r: ang[r])
    order, visited = [cur], {cur}
    while True:
        nxt = [n for n in adj[cur] if n not in visited]
        if not nxt:
            break
        d = [((ang[n] - ang[cur]) % (2 * np.pi)) for n in nxt]
        fwd = [(dd, n) for dd, n in zip(d, nxt) if dd < np.pi]
        cur = min(fwd)[1] if fwd else min(zip([2 * np.pi - dd for dd in d], nxt))[1]
        order.append(cur)
        visited.add(cur)
    ring = first[np.array(order)]
    P = V[ring, :2] - np.asarray(c)[None]
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - P[:, 1] * np.roll(P[:, 0], -1))
    return ring if area >= 0 else ring[::-1]


def _arc(P):
    d = np.linalg.norm(np.diff(np.concatenate([P, P[:1]]), axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)[:-1]]) / max(d.sum(), 1e-12)


def zip_rings(a, ua, b, ub):
    """Triangles joining two closed rings (vertex indices a, b in order, with parameters ua, ub in
    [0, 1) increasing, both starting near the same place)."""
    n, m = len(a), len(b)
    ua = np.append(ua, 1.0)
    ub = np.append(ub, 1.0)
    i = j = 0
    tris = []
    while i < n or j < m:
        if j >= m or (i < n and ua[i + 1] <= ub[j + 1]):
            tris.append([a[i % n], a[(i + 1) % n], b[j % m]])
            i += 1
        else:
            tris.append([a[i % n], b[(j + 1) % m], b[j % m]])
            j += 1
    return np.array(tris, np.int64)


def tube(V, A, B, c, rings: int = 2, samples: int = 96):
    """Vertices and triangles joining ring A (the head's cut, top) to ring B (the body's, below):
    both ordered loops (see ``outline``). In-between rings are blends of the two at equal
    fractions of their length, so the join follows each ring's real shape. Returns (new vertices,
    triangles with indices >= len(V) for them, the blend parameter t of each new vertex)."""
    PA, PB = V[A], V[B]
    # B starts where A starts (the same direction from the centre)
    a0 = np.arctan2(PA[0, 1] - c[1], PA[0, 0] - c[0])
    angB = np.arctan2(PB[:, 1] - c[1], PB[:, 0] - c[0])
    k0 = int(np.argmin(np.abs(np.angle(np.exp(1j * (angB - a0))))))
    B, PB = np.roll(B, -k0), np.roll(PB, -k0, axis=0)
    ua, ub = _arc(PA), _arc(PB)
    s = np.arange(samples) / samples

    def at(P, u):
        Pc = np.concatenate([P, P[:1]])
        uu = np.append(u, 1.0)
        return np.stack([np.interp(s, uu, Pc[:, k]) for k in range(3)], 1)
    SA, SB = at(PA, ua), at(PB, ub)
    n0 = len(V)
    newV, ts, ids = [], [], []
    for k in range(1, rings + 1):
        t = k / (rings + 1)
        ids.append(n0 + len(newV) + np.arange(samples))
        newV.extend((SA * (1 - t) + SB * t).tolist())
        ts.extend([t] * samples)
    tris = [zip_rings(A, ua, ids[0], s)]
    for k in range(rings - 1):
        tris.append(zip_rings(ids[k], s, ids[k + 1], s))
    tris.append(zip_rings(ids[-1], s, B, ub))
    T = np.concatenate(tris)
    # outward-facing
    Vall = np.concatenate([V, np.array(newV)])
    nrm = np.cross(Vall[T[:, 1]] - Vall[T[:, 0]], Vall[T[:, 2]] - Vall[T[:, 0]])
    cen = Vall[T].mean(axis=1)
    inward = (nrm[:, 0] * (cen[:, 0] - c[0]) + nrm[:, 1] * (cen[:, 1] - c[1])) < 0
    if inward.mean() > 0.5:                     # one winding for all (rings are both ordered)
        T = T[:, [0, 2, 1]]
    return np.array(newV), T, np.array(ts)
