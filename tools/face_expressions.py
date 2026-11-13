"""Facial expressions for the 3D companions, painted from the detailed sheets' expression panels.

Every detailed sheet draws the character's face twelve times (neutral, happy, curious, surprised,
playful, sleepy, angry / annoyed, sad, excited / cheerful, confident, mischievous, affection /
focused). The build paints those faces onto the model's own head:

1. **registration** - the NEUTRAL panel is matched to the big front view (SIFT features on the head,
   a RANSAC similarity, refined by ECC on the face), every other panel to the NEUTRAL panel (same
   drawing scale, so only small shifts / tilts);
2. **face canvas** - every panel is warped into one square around the face in the front view's
   pixels (the "canvas"), so all expressions line up with each other and with the model;
3. **face map** - the texels of the texture atlas that the front view painted (the front map from
   ``tools/reconstruct_3d.py``) and that fall inside the face oval get their canvas coordinate and a
   blend weight (the oval's soft edge times how squarely the texel faced the sheet);
4. a synthetic **blink** canvas: the neutral face with the eye band of the sleepy panel.

The asset gets ``face/<expression>.webp`` (256 x 256 canvases), ``face/face_map.npz`` and a
``"face": {"mode": "sheet", ...}`` metadata block; ``focusbuddy/companion/face_sheet.py`` swaps the
expressions into the texture at runtime from the companion's emotion. The neutral face is baked
into the model's texture, so the GLB alone shows a calm face.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CELLS = ROOT / "art" / "cells" / "v2"
CANVAS = 256
HUMAN_SHEETS = ("boy", "girl", "saree")
# the face oval in the NEUTRAL panel (centre x, centre y, radius x, radius y as panel fractions)
OVAL = {"animal": (0.5, 0.56, 0.37, 0.31), "human": (0.5, 0.5, 0.30, 0.30)}
# the eye band (panel fractions: centre y, half height) for the blink, and the panel it comes from
EYES = {"animal": (0.47, 0.11), "human": (0.46, 0.09)}
BLINK_FROM = {"animal": "sleepy", "human": "playful"}
# panels the automatic match can't place: the head drawn turned away (skipped - the companion falls
# back to a related expression), or with other face proportions (placed from hand-picked landmarks:
# left eye, right eye, nose tip in panel pixels, fitted to the neutral panel's by least squares)
SKIP = {"dog": ["confident"], "rabbit": ["mischievous"]}
LANDMARKS = {"dog": {"neutral": [(40, 42), (73, 37), (59, 53)], "angry": [(37, 54), (77, 52), (57, 79)],
                      "sleepy": [(40, 45), (73, 51), (51, 71)], "playful": [(40, 53), (75, 52), (56, 71)],
                      "affection": [(39, 60), (79, 44), (69, 64)]}}


def _load(p: Path) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(p).convert("RGBA"))


def _gray(a: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_RGB2GRAY)


def sift_similarity(src: np.ndarray, dst: np.ndarray, mask_dst=None, ratio: float = 0.8):
    """Similarity src px -> dst px from SIFT matches (RANSAC); (2x3 or None, inliers)."""
    import cv2
    s = cv2.SIFT_create(6000)
    k1, d1 = s.detectAndCompute(_gray(src), None)
    k2, d2 = s.detectAndCompute(_gray(dst), mask_dst)
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return None, 0
    good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < ratio * b.distance]
    if len(good) < 6:
        return None, len(good)
    p1 = np.float32([k1[a.queryIdx].pt for a in good])
    p2 = np.float32([k2[a.trainIdx].pt for a in good])
    A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=10000,
                                         confidence=0.999)
    return A, int(inl.sum()) if inl is not None else 0


def ecc_refine(src: np.ndarray, dst: np.ndarray, A: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Refine ``A`` (src -> dst) by ECC (rigid motion) on the masked part of ``dst``."""
    import cv2
    h, w = dst.shape[:2]
    ws = cv2.warpAffine(src, A, (w, h), flags=cv2.INTER_LINEAR)
    t = cv2.GaussianBlur(_gray(dst).astype(np.float32), (0, 0), 2.0)
    i = cv2.GaussianBlur(_gray(ws).astype(np.float32), (0, 0), 2.0)
    m = ((mask > 0) & (ws[..., 3] > 200)).astype(np.uint8) * 255
    W = np.eye(2, 3, dtype=np.float32)
    try:
        cc, W = cv2.findTransformECC(t, i, W, cv2.MOTION_EUCLIDEAN,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7), m, 5)
    except cv2.error:
        return A, -1.0
    M = np.vstack([cv2.invertAffineTransform(W), [0, 0, 1]]) @ np.vstack([A, [0, 0, 1]])
    return M[:2], float(cc)


def template_search(e: np.ndarray, neu: np.ndarray, kind: str):
    """Global search for a panel's placement against the neutral panel: the neutral face (the
    middle of the oval) as a template, matched over rotations and scales of the panel. Returns
    a 2x3 (panel px -> neutral px)."""
    import cv2
    ph, pw = neu.shape[:2]
    cx, cy, rx, ry = OVAL[kind]
    x0, x1 = int((cx - 0.8 * rx) * pw), int((cx + 0.8 * rx) * pw)
    y0, y1 = int((cy - 0.8 * ry) * ph), int((cy + 0.8 * ry) * ph)
    tpl = cv2.GaussianBlur(_gray(neu)[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.5)
    src = cv2.GaussianBlur(_gray(e).astype(np.float32), (0, 0), 1.5)
    best = (-2.0, None)
    for rot in range(-24, 25, 4):
        for sc in (0.85, 0.92, 1.0, 1.08, 1.16):
            R = cv2.getRotationMatrix2D((pw / 2, ph / 2), rot, sc)
            w = cv2.warpAffine(src, R, (pw, ph), borderMode=cv2.BORDER_REPLICATE)
            res = cv2.matchTemplate(w, tpl, cv2.TM_CCOEFF_NORMED)
            _, v, _, loc = cv2.minMaxLoc(res)
            if v > best[0]:
                T = np.array([[1, 0, x0 - loc[0]], [0, 1, y0 - loc[1]], [0, 0, 1]], np.float64)
                best = (v, (T @ np.vstack([R, [0, 0, 1]]))[:2])
    return best[1]


def register(sheet: str, names: list) -> dict:
    """{expression: 2x3 panel px -> front px}."""
    import cv2
    front = _load(CELLS / f"{sheet}_front.png")
    neu = _load(CELLS / f"{sheet}_expr_neutral.png")
    kind = "human" if sheet in HUMAN_SHEETS else "animal"
    H = front.shape[0]
    head = np.zeros(front.shape[:2], np.uint8)
    head[:int(H * (0.3 if kind == "human" else 0.6))] = 255
    up = cv2.resize(neu, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)     # match the front's detail
    A, n = sift_similarity(up, front, head)
    if A is None:
        raise RuntimeError(f"{sheet}: the neutral panel could not be matched to the front view")
    A = A.copy()
    A[:, :2] *= 2.0
    ph, pw = neu.shape[:2]
    cx, cy, rx, _ry = OVAL[kind]
    c = A @ np.array([cx * pw, cy * ph, 1.0])
    r = float(np.hypot(*A[:, 0])) * pw * rx
    face = np.zeros(front.shape[:2], np.uint8)
    cv2.ellipse(face, (int(c[0]), int(c[1])), (int(r), int(r * 0.85)), 0, 0, 360, 255, -1)
    An, cc = ecc_refine(neu, front, A, face)
    out = {"neutral": An}
    # the face oval of the neutral panel: expressions are matched on the face, not the ears / hair
    face_mask = np.zeros(neu.shape[:2], np.uint8)
    ocx, ocy, orx, ory = OVAL[kind]
    cv2.ellipse(face_mask, (int(ocx * pw), int(ocy * ph)), (int(orx * pw * 1.1), int(ory * ph * 1.15)), 0, 0, 360,
                255, -1)
    for name in names:
        if name == "neutral":
            continue
        e = _load(CELLS / f"{sheet}_expr_{name}.png")
        if e.shape != neu.shape:
            e = cv2.resize(e, (pw, ph), interpolation=cv2.INTER_CUBIC)
        lm = LANDMARKS.get(sheet, {})
        if name in lm:
            Ae, _ = cv2.estimateAffinePartial2D(np.float32(lm[name]), np.float32(lm["neutral"]), method=cv2.LMEDS)
            out[name] = (np.vstack([An, [0, 0, 1]]) @ np.vstack([Ae, [0, 0, 1]]))[:2]
            continue
        Ae, ne = sift_similarity(e, neu)
        cands = [A for A in (Ae if ne >= 8 else None, template_search(e, neu, kind)) if A is not None]
        best = (-2.0, np.eye(2, 3))
        for A0 in cands:
            A1, cc = ecc_refine(e, neu, A0, face_mask)
            if cc > best[0]:
                best = (cc, A1)
        Ae = best[1]
        out[name] = (np.vstack([An, [0, 0, 1]]) @ np.vstack([Ae, [0, 0, 1]]))[:2]
    return out


def _face_area(panel: np.ndarray) -> np.ndarray:
    """1 where a panel shows the character, 0 on its dark background (dark, unsaturated areas
    connected to the panel's edge), feathered."""
    import cv2
    from scipy import ndimage
    rgb = panel[..., :3].astype(np.float32) / 255.0
    v = rgb.max(axis=2)
    sat = (v - rgb.min(axis=2)) / np.maximum(v, 1e-6)
    dark = (v < 0.22) & (sat < 0.45)
    lab, n = ndimage.label(dark)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(edge))
    bg = ndimage.binary_opening(bg, iterations=1)
    valid = (~bg).astype(np.float32)
    valid[:2] = valid[-2:] = 0.0
    valid[:, :2] = valid[:, -2:] = 0.0
    return cv2.GaussianBlur(valid, (0, 0), 2.0)


def build_canvases(sheet: str, names: list, reg: dict):
    """Warp every panel into the face canvas. Returns (canvases {name: (C, C, 3) uint8}, canvas
    oval mask (C, C) float, canvas box (x0, y0, size) in front px)."""
    import cv2
    neu = _load(CELLS / f"{sheet}_expr_neutral.png")
    kind = "human" if sheet in HUMAN_SHEETS else "animal"
    ph, pw = neu.shape[:2]
    cx, cy, rx, ry = OVAL[kind]
    An = reg["neutral"]
    centre = An @ np.array([cx * pw, cy * ph, 1.0])
    k = float(np.hypot(*An[:, 0]))
    size = 2.25 * k * pw * max(rx, ry)
    x0, y0 = centre[0] - size / 2, centre[1] - size / 2
    T = np.array([[CANVAS / size, 0, -x0 * CANVAS / size], [0, CANVAS / size, -y0 * CANVAS / size], [0, 0, 1]])
    oval = np.zeros((ph, pw), np.float32)
    cv2.ellipse(oval, (int(cx * pw), int(cy * ph)), (int(rx * pw), int(ry * ph)), 0, 0, 360, 1.0, -1)
    oval = cv2.GaussianBlur(oval, (0, 0), pw * 0.05)
    mask = cv2.warpAffine(oval, (T @ np.vstack([An, [0, 0, 1]]))[:2], (CANVAS, CANVAS), flags=cv2.INTER_LINEAR)
    canv = {}
    for name in ["neutral"] + [n for n in names if n != "neutral"]:
        e = _load(CELLS / f"{sheet}_expr_{name}.png")
        if e.shape[:2] != (ph, pw):
            e = cv2.resize(e, (pw, ph), interpolation=cv2.INTER_CUBIC)
        M = (T @ np.vstack([reg[name], [0, 0, 1]]))[:2]
        warped = cv2.warpAffine(np.ascontiguousarray(e[..., :3]), M, (CANVAS, CANVAS), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        # where the panel has no face - outside the panel, or its dark studio background (dark
        # areas touching the panel's edge) - the neutral face shows instead
        valid = _face_area(e)
        vm = cv2.warpAffine(valid, M, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR, borderValue=0.0)[..., None]
        if name == "neutral":
            canv[name] = warped
        else:
            canv[name] = (warped * vm + canv["neutral"] * (1 - vm) + 0.5).astype(np.uint8)
    # blink: the neutral face with the eye band of a closed-eye panel (feathered): the animals'
    # sleepy face, the people's playful smile (their sleepy panels rest a hand on the cheek)
    src = BLINK_FROM.get(kind, "sleepy")
    if src in canv:
        ey, eh = EYES[kind]
        band = np.zeros((ph, pw), np.float32)
        band[int((ey - eh) * ph):int((ey + eh) * ph), int(0.12 * pw):int(0.88 * pw)] = 1.0
        band = cv2.GaussianBlur(band, (0, 0), ph * 0.025)
        bm = cv2.warpAffine(band, (T @ np.vstack([An, [0, 0, 1]]))[:2], (CANVAS, CANVAS))[..., None]
        canv["blink"] = (canv["neutral"] * (1 - bm) + canv[src] * bm + 0.5).astype(np.uint8)
    return canv, mask, (float(x0), float(y0), float(size))


def face_map(fmap: np.ndarray, mask: np.ndarray, box: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Texels of the atlas that get the face: (flat texel index, canvas uv in px, weight 0..1)."""
    from scipy import ndimage
    x0, y0, size = box
    w = np.clip(fmap[..., 2], 0, 1)
    sel = np.flatnonzero(w.reshape(-1) > 0.02)
    xy = fmap.reshape(-1, 3)[sel, :2]
    cu = (xy[:, 0] - x0) * CANVAS / size
    cv = (xy[:, 1] - y0) * CANVAS / size
    a = ndimage.map_coordinates(mask, [cv - 0.5, cu - 0.5], order=1, mode="constant", cval=0.0)
    a = a * w.reshape(-1)[sel]
    keep = a > 1.0 / 255
    return sel[keep].astype(np.int32), np.stack([cu[keep], cv[keep]], 1).astype(np.float32), \
        np.clip(a[keep], 0, 1).astype(np.float32)


def paint(tex: np.ndarray, idx: np.ndarray, uv: np.ndarray, a: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """Blend one canvas into an (S, S, 3|4) texture at the face texels (a copy)."""
    from scipy import ndimage
    out = tex.copy()
    flat = out.reshape(-1, out.shape[2])
    col = np.stack([ndimage.map_coordinates(canvas[..., c].astype(np.float32), [uv[:, 1] - 0.5, uv[:, 0] - 0.5],
                                            order=1, mode="nearest") for c in range(3)], 1)
    base = flat[idx, :3].astype(np.float32)
    flat[idx, :3] = np.clip(base * (1 - a[:, None]) + col * a[:, None] + 0.5, 0, 255).astype(np.uint8)
    return out


def build(sheet: str, names: list, fmap: np.ndarray, tex: np.ndarray, out_dir: Path) -> tuple[np.ndarray, dict]:
    """Write the face canvases + map for one character; returns (texture with the neutral face
    baked in, metadata ``face`` block)."""
    from PIL import Image
    names = [n for n in names if n not in SKIP.get(sheet, [])]
    reg = register(sheet, names)
    canv, mask, box = build_canvases(sheet, names, reg)
    idx, uv, a = face_map(fmap, mask, box)
    face_dir = out_dir / "face"
    face_dir.mkdir(parents=True, exist_ok=True)
    for old in face_dir.glob("*.webp"):
        old.unlink()
    for name, c in canv.items():
        Image.fromarray(c).save(face_dir / f"{name}.webp", quality=92, method=6)
    np.savez_compressed(face_dir / "face_map.npz", idx=idx, uv=uv.astype(np.float16),
                        a=(a * 255 + 0.5).astype(np.uint8), size=np.array(tex.shape[:2]))
    baked = paint(tex, idx, uv, a, canv["neutral"])
    meta = {"mode": "sheet", "map": "face/face_map.npz", "expressions": sorted(canv),
            "default": "neutral", "texels": int(len(idx)),
            "source": f"art/sheets/v2/{sheet}.webp (FACIAL EXPRESSIONS)"}
    (face_dir / "registration.json").write_text(json.dumps({k: np.round(v, 5).tolist() for k, v in reg.items()},
                                                           indent=1), encoding="utf-8")
    return baked, meta
