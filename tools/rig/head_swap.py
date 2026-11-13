"""New heads for the people: a detailed head generated on its own replaces the full-body model's head.

The full-body models were generated with the whole figure in one picture, so the face got very
few triangles and texels. A second Hunyuan3D generation from a head-and-shoulders close-up of the
same character art (``art/source_glb/<id>_head.glb``) has a properly modelled face. This module
puts that head on the rigged body:

1. the new head is turned upright, scaled so its eyes and mouth span the old ones and placed eye
   on eye, its face's front on the old face's front;
2. the join (``weld.py``): just under the new jaw, where the new head's neck is still a clean
   tube, both necks are measured, bent toward each other (each moves part of the way, fading out
   over the neck), cut by two close planes and stitched with a short band - one continuous
   neck, no step, no gap;
3. the old head goes: in the neck what is above the cut; around it the old hair, the old chin
   and jaw; the hood lining and collar are cut straight under the new jaw; the old hair behind
   the neck narrows into the new head and ends inside it. The new head drops its generated
   clothes and collar pieces under the jaw; its long hair stays;
4. colour: the body's neck takes the new head's skin tone (an offset, its detail stays), the
   band's texture runs from one ring's colours to the other's, and both cuts fade into them;
5. the new head's texture becomes a tile under the body's atlas (the atlas grows to 2:3, every
   body texture coordinate is rescaled);
6. skin weights: the body's neck weights at the join, blending into the Head bone over the jaw.

The skeleton, the body's weights and every animation are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core import BUILD, SOURCE


def prep_head(cid: str) -> dict:
    """The new head's mesh (Blender frame, as generated) via blender_prep (cached)."""
    import subprocess
    import sys
    src = SOURCE / f"{cid}_head.glb"
    npz = BUILD / f"{cid}_head_prep.npz"
    png = BUILD / f"{cid}_head_base.png"
    if not npz.is_file() or npz.stat().st_mtime < src.stat().st_mtime:
        r = subprocess.run([sys.executable, str(Path(__file__).with_name("blender_prep.py")), str(src), str(npz),
                            str(png)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-1500:])
    from PIL import Image
    z = np.load(npz)
    return {"V": z["V"], "F": z["F"], "FUV": z["FUV"], "tex": np.asarray(Image.open(png).convert("RGB"))}


def decimate(V, F, FUV, target: int):
    if len(F) <= target:
        return V, F, FUV
    import lod
    return lod.decimate(V, F, FUV, target)


HEADS = {
    "buddy": {"eyes": ((-0.019, 0.573), (0.125, 0.571)), "eye_r": (0.032, 0.018), "mouth": (0.050, 0.445),
              "chin": 0.365},
    "study_girl": {"eyes": ((-0.063, 0.483), (0.089, 0.503)), "eye_r": (0.039, 0.021), "mouth": (0.020, 0.373),
                   "chin": 0.315},
    "saree_girl": {"eyes": ((-0.037, 0.490), (0.115, 0.452)), "eye_r": (0.034, 0.026), "mouth": (0.023, 0.345),
                   "chin": 0.275},
}


# per character swap options
OPTIONS = {
    "buddy": {"head_w": 0.15, "cloth_to_mouth": True},
    "study_girl": {"head_w": 0.15, "keep_hair": True},
    "saree_girl": {"head_w": 0.15, "keep_hair": True},
}


def front_y(V, x, z, r):
    """Depth (y) of the frontmost surface near (x, z) (the face's front)."""
    m = (np.abs(V[:, 0] - x) < r) & (np.abs(V[:, 2] - z) < r)
    return float(np.percentile(V[m, 1], 3)) if m.sum() > 3 else None


def match_neck(V, F, FUV, atlas, HV, HF, HFUV, cut_z, ax, ay, log, head_z=None):
    from skimage import color
    from face import atlas_texels
    NH, Wd = atlas.shape[:2]

    def skin_med(Vx, Fx, FUVx, lo, hi, rad):
        c = Vx[Fx].mean(axis=1)
        sel = np.flatnonzero((c[:, 2] > lo) & (c[:, 2] < hi) & (np.hypot(c[:, 0] - ax, c[:, 1] - ay) < rad))
        if len(sel) < 5:
            return None
        idx, _pos, _f = atlas_texels(Vx, Fx, FUVx, (NH, Wd), sel)
        px = atlas.reshape(-1, 3)[idx].astype(np.float32)
        lab = color.rgb2lab(px[None] / 255)[0]
        ok = (lab[:, 1] > 8) & (lab[:, 2] > 6) & (lab[:, 0] > 40)
        return np.median(lab[ok], axis=0) if ok.sum() > 20 else None
    hz = cut_z if head_z is None else head_z
    new = skin_med(HV, HF, HFUV, hz, hz + 0.012, 0.045)
    old = skin_med(V, F, FUV, cut_z - 0.012, cut_z, 0.045)
    if new is None or old is None:
        log.append("neck colour match: no skin found")
        return atlas
    delta = new - old
    log.append(f"neck colour shift (Lab) {np.round(delta, 1).tolist()}")
    c = V[F].mean(axis=1)
    sel = np.flatnonzero((c[:, 2] > cut_z - 0.07) & (c[:, 2] < cut_z + 0.01) & (np.hypot(c[:, 0] - ax, c[:, 1] - ay) < 0.06))
    idx, pos, _f = atlas_texels(V, F, FUV, (NH, Wd), sel)
    flat = atlas.reshape(-1, 3)
    lab = color.rgb2lab(flat[idx][None].astype(np.float32) / 255)[0]
    skin = np.clip((lab[:, 1] - 6) / 6, 0, 1) * np.clip((lab[:, 0] - 35) / 10, 0, 1)
    w = np.clip((pos[:, 2] - (cut_z - 0.07)) / 0.06, 0, 1) * skin
    lab2 = lab + delta[None] * w[:, None]
    return _commit(atlas, idx, np.clip(color.lab2rgb(lab2[None])[0] * 255, 0, 255))


def neck_axis(HV, HF, HFUV, tex, chin, s):
    """The new head's neck under the chin: the skin-coloured surface in a band just below the jaw
    (the generated hood / headphones / straps around it are other colours). Returns (x, y, radius)."""
    from skimage import color
    T = tex.shape[0]
    cen = HV[HF].mean(axis=1)
    uv = HFUV.mean(axis=1)
    band = (cen[:, 2] < chin - 0.004) & (cen[:, 2] > chin - 0.02)
    px = tex[np.clip((uv[band, 1] * T).astype(int), 0, T - 1), np.clip((uv[band, 0] * T).astype(int), 0, T - 1)]
    lab = color.rgb2lab(px[None].astype(np.float32) / 255)[0]
    skin = (lab[:, 0] > 45) & (lab[:, 1] > 13) & (lab[:, 2] > 12) & (lab[:, 1] < 40)
    P = cen[band][skin]
    if len(P) < 10:
        raise RuntimeError("no neck found under the chin")
    c = np.median(P[:, :2], axis=0)
    r = float(np.percentile(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]), 85))
    return float(c[0]), float(c[1]), r


def swap(V, F, FUV, tex, W, names, J, head: dict, old: dict, new: dict, scale_k: float = 1.0,
         tile: int = 2048, budget: int = 30000, head_w: float = 0.5, keep_hair: bool = False,
         weld_depth: float = 0.016, weld_gap: float = 0.003, weld_rings: int = 2, weld_fade: float = 0.003, weld_blend: float = 0.03,
         head_shift: float = 0.4, body_blend: float = 0.025, back_hair: float = 0.015, cloth_to_mouth: bool = False,
         log: list | None = None):
    """Put the new head on the body. ``old``: the body's face landmarks (FACE spec form: eyes
    [(x, z, ...)], mouth ((x, z), (x, z), z)); ``new``: HEADS[cid]. Returns (V, F, FUV, atlas, W,
    info); faces are ordered body | head | weld band (``info["head_faces"]`` counts the last two).
    ``J`` is updated in place (HeadTop)."""
    log = log if log is not None else []
    HV = head["V"].astype(np.float64).copy()
    HF, HFUV = head["F"], head["FUV"]
    HV, HF, HFUV = decimate(HV, HF, HFUV, budget)
    (oel, oer), (oml, omr, omz) = [(e[0], e[1]) for e in old["eyes"]], old["mouth"]
    o_eye = np.array([(oel[0] + oer[0]) / 2, (oel[1] + oer[1]) / 2])
    o_mouth = np.array([(oml[0] + omr[0]) / 2, omz])
    (nel, ner), nm = new["eyes"], np.array(new["mouth"])
    n_eye = np.array([(nel[0] + ner[0]) / 2, (nel[1] + ner[1]) / 2])
    # the generated heads lean (the art's head tilt): turn them upright about the eyes' midpoint,
    # so the eyes are level - landmarks turn with them
    roll = np.arctan2(ner[1] - nel[1], ner[0] - nel[0])
    cr, sr = np.cos(-roll), np.sin(-roll)

    def turn(x, z):
        dx, dz = x - n_eye[0], z - n_eye[1]
        return n_eye[0] + cr * dx - sr * dz, n_eye[1] + sr * dx + cr * dz
    HV[:, 0], HV[:, 2] = turn(HV[:, 0], HV[:, 2])
    nel, ner = turn(*nel), turn(*ner)
    nm = np.array(turn(*nm))
    log.append(f"head roll {np.degrees(roll):.1f} deg straightened")
    s_eyes = abs(oer[0] - oel[0]) / abs(ner[0] - nel[0])
    s_face = abs(o_eye[1] - o_mouth[1]) / abs(n_eye[1] - nm[1])
    s = scale_k * 0.5 * (s_eyes + s_face)
    log.append(f"scale eyes {s_eyes:.3f} face {s_face:.3f} -> {s:.3f}")
    ny = front_y(HV, n_eye[0], n_eye[1] - 0.3 * (n_eye[1] - nm[1]), 0.04)
    oy = front_y(V, o_eye[0], o_eye[1] - 0.3 * (o_eye[1] - o_mouth[1]), 0.012)
    HV = (HV - np.array([n_eye[0], ny, n_eye[1]])) * s + np.array([o_eye[0], oy, o_eye[1]])
    chin_x, chin_z = turn(new["mouth"][0], new["chin"])
    chin_new = (chin_z - n_eye[1]) * s + o_eye[1]
    chin_new = min(chin_new, (nm[1] - n_eye[1]) * s + o_eye[1] - 0.6 * abs(nm[1] - new["chin"]) * s)
    # ---- the join (see weld.py): a little under the jaw, where the new head's neck is still a clean
    # tube. Both necks are measured there, bent toward each other, cut and stitched together
    import weld
    from skimage import color as _color
    ix = {n: i for i, n in enumerate(names)}

    def face_lab(tex_, FUVx):
        Th, Tw = tex_.shape[:2]
        uv = FUVx.mean(axis=1)
        px = tex_[np.clip((uv[:, 1] * Th).astype(int), 0, Th - 1), np.clip((uv[:, 0] * Tw).astype(int), 0, Tw - 1)]
        return _color.rgb2lab(px[None].astype(np.float32) / 255)[0]

    def is_skin(lab):
        return (lab[:, 0] > 45) & (lab[:, 1] > 9) & (lab[:, 2] > 7) & (lab[:, 1] < 40)
    jx, jy = float(J["Neck"][0]), float(J["Neck"][1])
    try:
        guess = neck_axis(HV, HF, HFUV, head["tex"], chin_new, s)[:2]
    except RuntimeError:
        guess = (jx, jy)
    # the new head's neck: as deep under the jaw as it stays a clean tube (at most weld_depth)
    depths = np.arange(0.004, weld_depth + 1e-6, 0.002)
    found = weld.clean_depth(HV, HF, [(jx, jy), guess], chin_new, depths)
    zj = chin_new - (found[0] if found else 0.006)
    nh_ = found[1] if found else weld.neck_axis(HV, HF, [(jx, jy), guess], zj)
    nb_ = weld.neck_axis(V, F, [(jx, jy)], zj)
    cb, rb = (np.array(nb_[0]), nb_[1]) if nb_ else (np.array([jx, jy]), 0.024)
    ch, rh = (np.array(nh_[0]), nh_[1]) if nh_ else (np.array(guess), 0.02)
    log.append(f"join {zj:.4f}: body neck ({cb[0]:.4f}, {cb[1]:.4f}) r {rb:.4f}, head neck ({ch[0]:.4f}, {ch[1]:.4f}) r {rh:.4f}")
    # part of the offset: the whole head moves (it stays on the old face within a few mm); the rest:
    # both necks bend toward the meeting point
    shift = head_shift * (cb - ch)
    HV[:, :2] += shift
    ch = ch + shift
    cm, rm = 0.5 * (cb + ch), 0.5 * (rb + rh)
    HV = weld.deform_neck(HV, ch, rh, cm, rm, z_full=zj, z_none=chin_new + 0.004)
    V = weld.deform_neck(V, cb, rb, cm, rm, z_full=zj, z_none=zj - body_blend)
    zh, zb = zj, zj - weld_gap
    cut_z = zb
    # ---- body: the old head off, the neck cut at zb
    blab = face_lab(tex, FUV)
    hw = W[:, ix["Head"]][F].mean(axis=1)
    bc = V[F].mean(axis=1)
    r_cut = 1.6 * rm
    rad_b = np.hypot(bc[:, 0] - cm[0], bc[:, 1] - cm[1])
    # the old head: what moves with the head bone. In the neck, faces across the cut are split by
    # it, never dropped whole (that would leave holes). Behind the neck the old hair stays up to a
    # little above the jaw (it falls on from under the new head's hair, which hides its top edge);
    # beside and in front, the old hair and the old jaw's skin go and the rest (hood lining,
    # collar) is cut straight just under the new jaw
    zmin = V[F][:, :, 2].min(axis=1)
    inside = rad_b < r_cut
    behind = ~inside & (bc[:, 1] > cm[1] - 0.3 * rm)
    old_hair = ~inside & ~behind & (blab[:, 0] < 45) & (bc[:, 2] > chin_new + 0.01)
    keep_b = ~old_hair
    keep_b &= ~(~inside & ~behind & (hw > 0.5) & (zmin > zb) & is_skin(blab) & (rad_b < 2.2 * rm))  # old jaw
    keep_b &= ~(inside & (zmin > zb))                                            # the old neck
    keep_b &= ~(behind & is_skin(blab) & (bc[:, 2] > chin_new))                  # the old nape
    gold = (blab[:, 2] > 30) & (blab[:, 2] > 1.5 * blab[:, 1])                    # earrings stay
    collar = ~inside & ~behind & ((hw > head_w) | (rad_b < 0.07)) & ~gold
    keep_b &= ~(collar & (hw > head_w) & is_skin(blab) & (zmin > zb) & (bc[:, 1] < cm[1]))  # old chin
    keep_b &= ~(collar & (blab[:, 0] < 55) & (bc[:, 2] > zb))                     # old locks of hair
    back = behind & (hw > head_w)
    # the old hair behind narrows toward the new head above the jaw, so its cut top ends inside it
    zt, z0 = chin_new + back_hair, chin_new - 0.03
    hlab0 = face_lab(head["tex"], HFUV)
    hood0 = (HV[HF][:, :, 2].mean(axis=1) < chin_new) & (((hlab0[:, 1] < 9) & (hlab0[:, 0] > 50)) |
                                                         ((hlab0[:, 2] < 4) & (hlab0[:, 0] > 22)) | (hlab0[:, 2] < -3))
    th_s, R_s = weld.first_hits(HV, HF[~hood0], cm, [zt], 48, rmax=0.2)
    R_s = R_s[0]
    R_s = np.where(np.isnan(R_s), np.nanmedian(R_s) if (~np.isnan(R_s)).any() else 0.05, R_s)
    bv = np.unique(F[back])
    bv = bv[V[bv, 2] > z0]
    if len(bv):
        V = V.copy()
        d = V[bv, :2] - cm[None]
        r = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
        k = np.round((np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2 * np.pi) * len(th_s)).astype(int) % len(th_s)
        rt = np.minimum(r, 0.9 * R_s[k])
        w = weld.smooth((V[bv, 2] - z0) / (zt - z0))
        V[bv, :2] = cm[None] + d * (1 + w * (rt / r - 1))[:, None]
    F1, FUV1 = F[keep_b], FUV[keep_b]
    V1, F1, FUV1, W1, _, src = weld.clip(V, F1, FUV1, W, collar[keep_b], chin_new - 0.004, keep_above=False,
                                         return_src=True)
    V1, F1, FUV1, W1, _, src_b = weld.clip(V1, F1, FUV1, W1, back[keep_b][src], chin_new + back_hair, keep_above=False,
                                           return_src=True)
    src = src[src_b]
    inside1 = inside[keep_b][src]
    edges_b = []
    Vb, Fb, FUVb, Wb, cut_b = weld.clip(V1, F1, FUV1, W1, inside1, zb, keep_above=False, edges=edges_b)
    if len(cut_b):
        cut_b = cut_b[np.hypot(Vb[cut_b, 0] - cm[0], Vb[cut_b, 1] - cm[1]) < r_cut * 1.05]
    Bw = Wb[cut_b].mean(axis=0) if len(cut_b) else np.eye(W.shape[1])[ix.get("Neck", ix["Head"])]
    mk = np.zeros(len(Vb), bool)
    mk[cut_b] = True
    ring_b = weld.outline(Vb, edges_b, mk, cm)
    # ---- head: weights from the body's neck (Bw) blending into the Head bone over the jaw
    Wh = np.zeros((len(HV), W.shape[1]))
    t = smooth01((HV[:, 2] - zh) / max(1e-6, chin_new + 0.012 - zh))
    Wh += (1 - t)[:, None] * Bw[None]
    Wh[:, ix["Head"]] += t
    hc = HV[HF].mean(axis=1)
    hmask = np.hypot(hc[:, 0] - cm[0], hc[:, 1] - cm[1]) < r_cut
    edges_h = []
    HV, HF, HFUV, Wh, cut_h = weld.clip(HV, HF, HFUV, Wh, hmask, zh, keep_above=True, edges=edges_h)
    if len(cut_h):
        cut_h = cut_h[np.hypot(HV[cut_h, 0] - cm[0], HV[cut_h, 1] - cm[1]) < r_cut * 1.05]
    mk = np.zeros(len(HV), bool)
    mk[cut_h] = True
    ring_h = weld.outline(HV, edges_h, mk, cm)
    log.append(f"neck rings: body {len(ring_b)}, head {len(ring_h)} vertices")
    cen = HV[HF].mean(axis=1)
    radial = np.hypot(cen[:, 0] - cm[0], cen[:, 1] - cm[1])
    keep_h = cen[:, 2] > zh - 1e-5
    plab = face_lab(head["tex"], HFUV)
    hairy = (plab[:, 0] < 55) & (plab[:, 2] > 4) & (plab[:, 1] > 2)
    if keep_hair:
        # long hair below the cut: kept beside and behind the neck where the body itself has hair
        # or head (the generated bust's straps, collar and hood strings sit on the chest)
        from scipy.spatial import cKDTree
        hw_v = W[:, ix["Head"]] + W[:, ix.get("Neck", ix["Head"])]
        d, nn = cKDTree(V).query(cen, k=4)
        near_head = (hw_v[nn].mean(axis=1) > 0.15) & (d.min(axis=1) < 0.05)
        chest = (cen[:, 1] < cm[1] - 0.6 * rm) & (np.abs(cen[:, 0] - cm[0]) < 0.055)
        keep_h |= hairy & (radial > r_cut * 1.2) & near_head & ~chest
    # below the mouth, the generated clothes (hood: light, colourless; headphones / straps: blue,
    # grey, black-blue) go - the body has its own; skin and hair (warm colours) stay
    # (``cloth_to_mouth``: from the mouth down, for a hood worn up around the face; otherwise only
    # under the jaw - above it, the face's own shadowed or lilac-tinted patches look the same)
    low = cen[:, 2] < ((nm[1] - n_eye[1]) * s + o_eye[1] if cloth_to_mouth else chin_new + 0.003)
    cloth = low & (((plab[:, 1] < 9) & (plab[:, 0] > 50)) | ((plab[:, 2] < 4) & (plab[:, 0] > 22)) | (plab[:, 2] < -3))
    keep_h &= ~cloth
    log.append(f"dropped {int(cloth.sum())} clothing faces of the new head")
    # under the jaw and away from the neck, the generated bust's hood rim and collar are the same
    # warm white as skin: pieces there stay only when they are mostly hair
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    under = keep_h & (cen[:, 2] < chin_new - 0.003) & (radial > r_cut)
    uf = np.flatnonzero(under)
    if len(uf):
        Fu = HF[uf]
        E = np.concatenate([Fu[:, [0, 1]], Fu[:, [1, 2]]])
        G = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(HV), len(HV)))
        _, lab_ = connected_components(G, directed=False)
        fl = lab_[Fu[:, 0]]
        hairish = ((plab[uf, 0] < 58) & (plab[uf, 2] > 4) & (plab[uf, 1] > 2)).astype(float)
        ids, inv = np.unique(fl, return_inverse=True)
        frac = np.bincount(inv, weights=hairish) / np.bincount(inv)
        drop = frac[inv] < 0.5
        keep_h[uf[drop]] = False
        log.append(f"dropped {int(drop.sum())} faces of collar pieces under the jaw")
    # small loose bits left after the cuts (corners of the generated collar) go too
    kf = np.flatnonzero(keep_h)
    Fk = HF[kf]
    E = np.concatenate([Fk[:, [0, 1]], Fk[:, [1, 2]]])
    G = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(HV), len(HV)))
    _, lab_ = connected_components(G, directed=False)
    fl = lab_[Fk[:, 0]]
    ids, cnt = np.unique(fl, return_counts=True)
    small = np.isin(fl, ids[cnt < 0.01 * len(kf)])
    keep_h[kf[small]] = False
    HF2, HFUV2 = HF[keep_h], HFUV[keep_h]
    used = np.unique(HF2)
    ring_h = ring_h[np.isin(ring_h, used)]
    remap = np.full(len(HV), -1, np.int64)
    remap[used] = np.arange(len(used))
    HV, HF2, Wh = HV[used], remap[HF2], Wh[used]
    ring_h = remap[ring_h]
    log.append(f"body faces {len(Fb)}, head faces {len(HF2)}")
    # ---- atlas
    H, Wd = tex.shape[:2]
    from PIL import Image
    ht = np.asarray(Image.fromarray(head["tex"]).resize((tile, tile), Image.LANCZOS))
    NH = H + tile
    atlas = np.zeros((NH, Wd, 3), np.uint8)
    atlas[:H] = tex
    atlas[H:, :tile] = ht
    atlas[H:, tile:] = np.median(ht.reshape(-1, 3), axis=0).astype(np.uint8)
    FUVb = FUVb.copy()
    FUVb[..., 1] = FUVb[..., 1] * H / NH
    HFUV2 = HFUV2.copy()
    HFUV2[..., 0] = HFUV2[..., 0] * tile / Wd
    HFUV2[..., 1] = (H + HFUV2[..., 1] * tile) / NH
    # the body's neck takes the new head's skin tone (a soft colour shift fading down the neck)
    try:
        atlas = match_neck(Vb, Fb, FUVb, atlas, HV, HF2, HFUV2, zb, cm[0], cm[1], log, head_z=zh)
    except Exception as e:  # noqa: BLE001 - a colour tweak never breaks the build
        log.append(f"neck colour match skipped: {e}")
    # ---- the weld
    nb = len(Vb)
    Vn = np.concatenate([Vb, HV])
    Fn = np.concatenate([Fb, HF2 + nb])
    FUVn = np.concatenate([FUVb, HFUV2])
    Wn = np.concatenate([Wb, Wh])
    TF = np.zeros((0, 3), np.int64)
    if len(ring_h) >= 8 and len(ring_b) >= 8:
        TV, TF, tt = weld.tube(Vn, ring_h + nb, ring_b, cm, rings=weld_rings)
        nbf = len(Fb)
        colA = ring_colours(atlas, Vn, Fn[nbf:], FUVn[nbf:], zh, zh + 0.005, cm, r_cut)
        colB = ring_colours(atlas, Vn, Fn[:nbf], FUVn[:nbf], zb - 0.005, zb, cm, r_cut)
        # the body's neck shifts toward the head's colours at the seam, fading out down the neck
        # (an offset: its texture's own detail stays)
        atlas = blend_offset(atlas, Vn, Fn[:nbf], FUVn[:nbf], zb, weld_blend, cm, r_cut, colA, colB)
        colB = ring_colours(atlas, Vn, Fn[:nbf], FUVn[:nbf], zb - 0.005, zb, cm, r_cut)
        atlas, patch = angular_patch(atlas, colA, colB, tile)
        # texels right at each cut fade into the ring colours (no line where the textures meet)
        atlas = fade_to(atlas, Vn, Fn[nbf:], FUVn[nbf:], zh, zh + weld_fade, cm, r_cut, colA, top=False)
        atlas = fade_to(atlas, Vn, Fn[:nbf], FUVn[:nbf], zb - weld_fade, zb, cm, r_cut, colB, top=True)
        n0 = len(Vn)
        tall = np.concatenate([np.zeros(n0), tt])
        tall[ring_b] = 1.0
        Vn = np.concatenate([Vn, TV])
        Wn = np.concatenate([Wn, np.repeat(Bw[None], len(TV), axis=0)])
        FUVn = np.concatenate([FUVn, patch_uv(patch, Vn, TF, tall, cm)])
        Fn = np.concatenate([Fn, TF])
        log.append(f"welded: tube {len(TF)} triangles")
    else:
        log.append("weld skipped: a neck ring was not found")
    usedv = np.unique(Fn)
    rm_ = np.full(len(Vn), -1, np.int64)
    rm_[usedv] = np.arange(len(usedv))
    J["HeadTop"] = np.array([J["HeadTop"][0], J["HeadTop"][1], float(HV[:, 2].max())])
    sx = float(shift[0])
    info = {"head_faces": int(len(HF2) + len(TF)), "tube_faces": int(len(TF)), "scale": s, "cut": cut_z,
            "chin": chin_new,
            "eyes": [((x - n_eye[0]) * s + o_eye[0] + sx, (z - n_eye[1]) * s + o_eye[1]) for x, z in (nel, ner)],
            "mouth": ((nm[0] - n_eye[0]) * s + o_eye[0] + sx, (nm[1] - n_eye[1]) * s + o_eye[1]),
            "eye_r": (new["eye_r"][0] * s, new["eye_r"][1] * s)}
    # head faces last (the levels of detail keep a share of their budget for them): body | head | tube
    return Vn[usedv], rm_[Fn], FUVn, atlas, Wn[usedv], info


def smooth01(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _commit(atlas, idx, new, steps: int = 2):
    """Write new colours to texels idx, and the same change to the few texels around them (the
    gutter across UV seams that bilinear filtering reads, which no face covers)."""
    from face import grow
    flat = atlas.reshape(-1, 3)
    delta = new.astype(np.float32) - flat[idx].astype(np.float32)
    gi, gd = grow(idx, [delta], atlas.shape[:2], steps)
    out = atlas.copy()
    o = out.reshape(-1, 3)
    o[gi] = np.clip(o[gi].astype(np.float32) + gd + 0.5, 0, 255).astype(np.uint8)
    return out


def _texels(atlas, V, F, FUV, sel):
    from face import atlas_texels
    return atlas_texels(V, F, FUV, atlas.shape[:2], sel)


NTH = 64       # angle bins of the weld's colours


def ring_colours(atlas, V, F, FUV, z0, z1, c, rad):
    """Colour per angle about c of the neck's texels between heights z0 and z1: (NTH, 3)."""
    import weld
    cen = V[F].mean(axis=1)
    sel = np.flatnonzero((cen[:, 2] > z0) & (cen[:, 2] < z1) & (np.hypot(cen[:, 0] - c[0], cen[:, 1] - c[1]) < rad))
    if len(sel) == 0:
        return np.tile(np.array([210.0, 160.0, 140.0]), (NTH, 1))
    idx, pos, _f = _texels(atlas, V, F, FUV, sel)
    return weld.angular_colour(pos, atlas.reshape(-1, 3)[idx].astype(np.float32), c, NTH)


def angular_patch(atlas, colA, colB, tile, cell: int = 6, h: int = 192, pad: int = 12):
    """The weld's texture in the unused corner right of the head tile: columns = angle around the
    neck (one period, plus ``pad`` columns of the next one so faces across the wrap stay smooth),
    rows = top ring (colA) to bottom ring (colB). Returns (atlas, (x0, period, y0, h) in pixels)."""
    NH, Wd = atlas.shape[:2]
    period = NTH * cell
    x0, y0 = tile + 64, NH - h - 64
    ang = (np.arange(period + pad * cell) + 0.5) / cell - 0.5            # in bins
    k0 = np.floor(ang).astype(int)
    f = (ang - k0)[:, None]
    A = colA[k0 % NTH] * (1 - f) + colA[(k0 + 1) % NTH] * f
    B = colB[k0 % NTH] * (1 - f) + colB[(k0 + 1) % NTH] * f
    t = np.linspace(0, 1, h)[:, None, None]
    img = A[None] * (1 - t) + B[None] * t
    img = np.concatenate([np.repeat(img[:1], 8, 0), img, np.repeat(img[-1:], 8, 0)])
    img = np.concatenate([np.repeat(img[:, :1], 8, 1), img, np.repeat(img[:, -1:], 8, 1)], 1)
    atlas = atlas.copy()
    atlas[y0 - 8:y0 + h + 8, x0 - 8:x0 + img.shape[1] - 8] = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    return atlas, (x0, period, y0, h, NH, Wd)


def patch_uv(patch, V, T, tt, c):
    """Texture coordinates of the weld's triangles T: u by angle about c (unwrapped per triangle),
    v by the blend parameter tt of each vertex (0 at the head's ring, 1 at the body's)."""
    x0, period, y0, h, NH, Wd = patch
    ang = np.arctan2(V[T, 1] - c[1], V[T, 0] - c[0])                       # (n, 3)
    wrap = (ang.max(1) - ang.min(1)) > np.pi
    ang = np.where(wrap[:, None] & (ang < 0), ang + 2 * np.pi, ang)
    u = (x0 + (ang + np.pi) / (2 * np.pi) * period) / Wd
    v = (y0 + 0.5 + (h - 1) * tt[T]) / NH
    return np.stack([u, v], -1)


def _skin_w(lab):
    return np.clip((lab[..., 1] - 6) / 6, 0, 1) * np.clip((lab[..., 0] - 35) / 10, 0, 1)


def blend_offset(atlas, V, F, FUV, z_top, depth, c, rad, colA, colB):
    """Shift the texels of the neck below z_top by the colour difference colA - colB (per angle,
    in Lab), fully at z_top, fading out over ``depth`` and away from the neck; skin only (hair or
    cloth at the seam keeps its colour)."""
    from skimage import color
    P = V[F]
    zmax, zmin = P[:, :, 2].max(axis=1), P[:, :, 2].min(axis=1)
    cen = P.mean(axis=1)
    sel = np.flatnonzero((zmax > z_top - depth) & (zmin < z_top + 0.001) &
                         (np.hypot(cen[:, 0] - c[0], cen[:, 1] - c[1]) < rad * 1.4))
    if len(sel) == 0:
        return atlas
    idx, pos, _f = _texels(atlas, V, F, FUV, sel)
    labA = color.rgb2lab(colA[None] / 255)[0]
    labB = color.rgb2lab(colB[None] / 255)[0]
    dl = (labA - labB) * (_skin_w(labA) * np.clip((labB[:, 1] - 4) / 6, 0, 1))[:, None]
    ang = np.arctan2(pos[:, 1] - c[1], pos[:, 0] - c[0])
    b = (ang + np.pi) / (2 * np.pi) * NTH - 0.5
    k0 = np.floor(b).astype(int)
    f = (b - k0)[:, None]
    off = dl[k0 % NTH] * (1 - f) + dl[(k0 + 1) % NTH] * f
    r = np.hypot(pos[:, 0] - c[0], pos[:, 1] - c[1])
    w = smooth01(1 - (z_top - pos[:, 2]) / depth) * (1 - smooth01((r - rad) / (0.3 * rad)))
    flat = atlas.reshape(-1, 3)
    lab = color.rgb2lab(flat[idx][None].astype(np.float32) / 255)[0]
    w = w * np.clip((lab[:, 1] - 4) / 6, 0, 1)          # skin, also in shadow (hair / cloth keep theirs)
    lab = lab + off * w[:, None]
    return _commit(atlas, idx, np.clip(color.lab2rgb(lab[None])[0] * 255, 0, 255))


def fade_to(atlas, V, F, FUV, z0, z1, c, rad, cols, top: bool):
    """Texels of the neck faces between z0 and z1 blend toward the ring colours ``cols`` (per angle):
    fully at the cut (z1 when ``top`` - the body's cut is above its faces - else z0), fading out
    across the band."""
    cen = V[F].mean(axis=1)
    sel = np.flatnonzero((cen[:, 2] > z0 - 0.002) & (cen[:, 2] < z1 + 0.002) &
                         (np.hypot(cen[:, 0] - c[0], cen[:, 1] - c[1]) < rad * 1.1))
    if len(sel) == 0:
        return atlas
    idx, pos, _f = _texels(atlas, V, F, FUV, sel)
    d = (z1 - pos[:, 2]) if top else (pos[:, 2] - z0)
    w = 1 - smooth01(d / max(1e-6, z1 - z0))
    ang = np.arctan2(pos[:, 1] - c[1], pos[:, 0] - c[0])
    b = (ang + np.pi) / (2 * np.pi) * NTH - 0.5
    k0 = np.floor(b).astype(int)
    f = (b - k0)[:, None]
    col = cols[k0 % NTH] * (1 - f) + cols[(k0 + 1) % NTH] * f
    old = atlas.reshape(-1, 3)[idx].astype(np.float32)
    return _commit(atlas, idx, old * (1 - w[:, None]) + col * w[:, None])
