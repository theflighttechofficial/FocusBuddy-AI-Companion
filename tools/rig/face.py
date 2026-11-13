"""Face rig for the generated companions: eyelids that blink, squint and close, a mouth that
opens, brows and mouth corners that move - driven by the companion's emotion.

The generated models have their eyes and mouth painted in the texture, so:

* **eyelids and the open mouth are painted** at run time (``focusbuddy/companion/face_paint.py``)
  into the renderer's own copy of the texture. This module fits an ellipse to each painted eye
  (``refine_eye``) and the people's lips (``refine_mouth``) on a front render, works out what a
  closed lid shows (``fill_eye``: the surrounding fur / skin continued over the eye by Poisson
  blending, with the fur detail of the band above the eye) and stores, for every texel of the eyes
  and the mouth, its place on the eye / mouth (``painted_data`` -> ``face/painted.npz``). Only
  texels the eye's own surface owns are kept (``surface_patch``: measured along the surface);
* **eyeBlink / eyeSquint** morphs flatten the sculpted eyeball and socket onto a shell fitted to
  the face around the eye (the animals; the people's eyes are flat already);
* **mouth corners / brows / jaw** are morphs of the head's own vertices (a smooth falloff on the
  face's front surface - lips and the mouth's inside move together);
* a **breath** morph swells the chest and belly.

Landmarks (eye centres and sizes, mouth corners) are read off grid renders of every face
(``tools/rig/grid.py`` / ``facerender.py``) and refined automatically; check views are written to
``build/rig/check_<id>_face.png``.
"""

from __future__ import annotations

import math

import numpy as np

from core import BUILD
from facerender import ortho, to_px

# per character (rig frame, front view: x right = the character's left, z up):
#   eyes: [(cx, cz, rx, rz, angle deg)] - right eye (x < centre) first; fitted to the outline
#   mouth: (left corner (x, z), right corner (x, z)) in image left -> right order, centre z
#   jaw: "patch" (dark interior unfolds) | "drop" (an already open mouth opens wider) | None
FACE = {
    "fox": {"eyes": [(-0.078, 0.684, 0.040, 0.035, 0.0), (0.086, 0.681, 0.040, 0.035, 0.0)],
            "mouth": ((-0.026, 0.596), (0.068, 0.596), 0.590), "jaw": "patch"},
    "cat": {"eyes": [(-0.128, 0.670, 0.049, 0.039, 0.0), (0.047, 0.669, 0.042, 0.040, 0.0)],
            "mouth": ((-0.077, 0.571), (0.037, 0.571), 0.568), "jaw": "patch"},
    "dog": {"eyes": [(-0.1000, 0.8076, 0.030, 0.027, 0.0), (0.0435, 0.8006, 0.025, 0.029, 0.0)],
            "mouth": ((-0.114, 0.712), (0.0685, 0.705), 0.70), "jaw": "drop"},
    "rabbit": {"eyes": [(-0.0833, 0.557, 0.034, 0.031, 0.0), (0.0777, 0.5617, 0.032, 0.030, 0.0)],
               "mouth": ((-0.0265, 0.478), (0.0379, 0.478), 0.476), "jaw": "patch"},
    "buddy": {"eyes": [(-0.022, 0.9085, 0.0095, 0.0048, 0.0), (0.021, 0.909, 0.0095, 0.0048, 0.0)],
              "mouth": ((-0.0129, 0.8719), (0.0129, 0.8719), 0.8705), "jaw": "patch", "human": True},
    "study_girl": {"eyes": [(-0.0224, 0.9143, 0.0080, 0.0038, 0.0), (0.023, 0.9143, 0.0080, 0.0038, 0.0)],
                   "mouth": ((-0.0127, 0.875), (0.0104, 0.875), 0.8745), "jaw": "patch", "human": True},
    "saree_girl": {"eyes": [(-0.0206, 0.9112, 0.0094, 0.0052, 0.0), (0.0216, 0.9112, 0.0094, 0.0052, 0.0)],
                   "mouth": ((-0.0144, 0.8737), (0.0125, 0.8737), 0.873), "jaw": "patch", "human": True},
}

MORPH_MAP = {"eyeBlink": "lid", "eyeSquint": "lid_low", "mouthSmile": "happy", "mouthFrown": "frown",
             "jawOpen": "jaw", "browInnerUp": "brow_up", "browDown": "brow_down", "breath": "breath"}


def smooth01(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ mouth
def refine_mouth(r, est):
    """Lips (people): the slit between the lips is the darkest row across the lips' middle, the
    corners are where the lips' red ends along it. ``est`` / result: ((lx, lz), (rx, rz), mz)."""
    from skimage import color
    (lx, lz), (rx_, rz_), mz = est
    mw = abs(rx_ - lx)
    cx = 0.5 * (lx + rx_)
    sc = r["scale"]
    x0, x1 = to_px(r, cx - 1.0 * mw, 0)[0], to_px(r, cx + 1.0 * mw, 0)[0]
    y0, y1 = to_px(r, 0, mz + 0.6 * mw)[1], to_px(r, 0, mz - 0.6 * mw)[1]
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
    crop = r["img"][y0:y1, x0:x1].astype(np.float32) / 255
    lab = color.rgb2lab(crop)
    L, A = lab[..., 0], lab[..., 1]
    W = crop.shape[1]
    mid = slice(int(W * 0.4), int(W * 0.6))
    # the lips: redder than the skin around them
    skin_a = np.median(A[:, :int(W * 0.12)])
    lips = A > skin_a + 6
    rows = np.nonzero(lips[:, mid].mean(1) > 0.5)[0]
    if len(rows) < 3:
        return est
    band = slice(rows.min(), rows.max() + 1)
    prof = np.median(L[band, mid], axis=1)
    k = int(np.argmin(prof[1:-1]) + 1) + rows.min()        # the slit's row
    row = lips[max(0, k - 2):k + 3].any(0)
    cols = np.nonzero(row)[0]
    c_mid = W // 2
    # the connected run of lip columns through the middle
    left = c_mid
    while left > 0 and row[left - 1]:
        left -= 1
    right = c_mid
    while right < W - 1 and row[right + 1]:
        right += 1
    if not len(cols) or right - left < W * 0.15:
        return est
    zk = mz + 0.6 * mw - (k + 0.5) / sc
    xl = cx - 1.0 * mw + (left + 0.5) / sc
    xr = cx - 1.0 * mw + (right + 0.5) / sc
    return ((xl, zk), (xr, zk), zk)


# ------------------------------------------------------------------ eye ellipses
def refine_eye(r, est, human=False):
    """Fit the ellipse to the painted eye: the dark outline / pupil, the white and everything unlike
    the surrounding skin / fur, closed and filled, the part over the estimate's centre; OpenCV
    ellipse fit. ``est`` and the result are (cx, cz, rx, rz, angle) in world units / degrees; a fit
    that is ragged (solidity) or far from the estimate is not trusted."""
    import cv2
    from scipy import ndimage
    from skimage import color
    cx, cz, rx, rz, _a = est
    px, py = to_px(r, cx, cz)
    sc = r["scale"]
    R = int(max(rx, rz) * sc * 2.0)
    x0, y0 = int(px - R), int(py - R)
    pad = np.pad(r["img"], ((R, R), (R, R), (0, 0)), mode="edge")
    crop = pad[y0 + R:y0 + 3 * R, x0 + R:x0 + 3 * R].astype(np.float32) / 255
    lab = color.rgb2lab(crop)
    L, C = lab[..., 0], np.hypot(lab[..., 1], lab[..., 2])
    hh, ww = L.shape
    yy, xx = np.mgrid[:hh, :ww]
    ox, oy = px - x0, py - y0
    d = np.sqrt(((xx - ox) / (rx * sc)) ** 2 + ((yy - oy) / (rz * sc)) ** 2)
    ring = (d > 1.4) & (d < 1.9)
    ref = np.median(lab[ring], axis=0)
    diff = np.linalg.norm(lab - ref, axis=2)
    dark = L < 0.9 * np.percentile(L[d < 1.0], 25)
    white = (L > 80) & (C < 14)
    mask = (dark | white | (diff > 30)) & (d < 1.45)
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=2)
    lab_, n = ndimage.label(mask)
    if n == 0:
        return est
    cen = d < 0.5
    k = 1 + int(np.argmax([(lab_ == i + 1)[cen].sum() for i in range(n)]))
    blob = (lab_ == k).astype(np.uint8)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 6:
        return est
    (ex, ey), (w2, h2), ang = cv2.fitEllipse(cnt)
    solidity = float(blob.sum()) / max(math.pi * w2 * h2 / 4, 1.0)
    a, b = w2 / 2 / sc, h2 / 2 / sc
    ang = -ang                                    # image y is down, world z is up
    if a < b:
        a, b, ang = b, a, ang + 90.0
    ang = (ang + 90.0) % 180.0 - 90.0
    wx = r["box"][0] + (ex + x0) / sc
    wz = r["box"][2] + (r["size"] - (ey + y0)) / sc
    ok = (solidity > 0.82 and abs(wx - cx) < 0.5 * rx and abs(wz - cz) < 0.5 * rz and
          0.7 < a / max(rx, rz) < 1.5 and 0.6 < b / min(rx, rz) < 1.6)
    return (float(wx), float(wz), float(a), float(b), float(ang)) if ok else est


# the painted eye's outline reaches a little past the fitted ellipse: the closed lid covers
# EYE_LID_R x the ellipse, the fill (what the lid shows) reaches EYE_FILL_K x, and the texels
# kept for the runtime reach EYE_SEL_K x (so the whole lid area is redrawn)
EYE_LID_R = 1.38
EYE_LASH_R = 1.1
EYE_FILL_K = 1.40
EYE_SEL_K = 1.42
# the blink morph replaces the sculpted eye inside BLINK_FIT[0] x the ellipse by a smooth shell fitted
# between BLINK_FIT[0] and BLINK_FIT[1] x, fading out toward BLINK_FIT[1]
BLINK_FIT = (1.55, 1.9)


def ellipse_pt(e, u, v, k=1.0):
    """World (x, z) of the elliptic-grid point (u in [-1, 1] across, v in [-1, 1] bottom..top)."""
    cx, cz, a, b, ang = e
    x = a * k * u
    y = b * k * v * np.sqrt(np.clip(1 - u * u, 0, 1))
    t = math.radians(ang)
    return cx + x * math.cos(t) - y * math.sin(t), cz + x * math.sin(t) + y * math.cos(t)


# ------------------------------------------------------------------ eyelids
def inpainted(r, cache={}):
    """The face render with every eye filled in from its surroundings (what a closed lid shows)."""
    key = id(r)
    if key not in cache:
        cache.clear()
        cache[key] = r["img"].copy()
    return cache[key]


def lid_ellipses(eyes, human):
    """The lid areas. A person's painted eye is a narrow almond with lashes and shadow painted
    above and below it, so the lid area is made rounder (height at least 0.62 x the width) and the
    closed eye's lash line moves in to stay on the eye's real bottom edge. Returns (ellipses,
    lash line radius in lid-area units)."""
    if not human:
        return list(eyes), EYE_LASH_R
    out, ratio = [], []
    for cx, cz, a, b, ang in eyes:
        nb = max(b, 0.5 * a)
        out.append((cx, cz, a, nb, ang))
        ratio.append(b / nb)
    return out, float(EYE_LASH_R * np.mean(ratio))


def fill_eye(r, e, k=1.12, squash=0.22, skin_only=False):
    """Paint the eye ``e`` over with what a closed upper lid shows: the skin / fur just above the
    eye, mirrored down over it (so the fur keeps its strands and stripes instead of a flat
    smear), then Poisson-blended (``cv2.seamlessClone``) into the face so the patch meets the
    surrounding colours on every side. ``squash`` < 1 samples only the band just above the eye
    (never the brow). Updates the shared inpainted image."""
    import cv2
    img = inpainted(r)
    S = r["size"]
    cx, cz, a, b, ang = e
    t = math.radians(ang)
    u0, u1, v0, v1 = r["box"]
    sc = r["scale"]
    yy, xx = np.mgrid[:S, :S].astype(np.float32) + 0.5
    wx = u0 + xx / sc
    wz = v0 + (S - yy) / sc
    dx, dz = wx - cx, wz - cz
    X = (dx * math.cos(t) + dz * math.sin(t)) / (a * k)
    Y = (-dx * math.sin(t) + dz * math.cos(t)) / (b * k)
    h = np.sqrt(np.clip(1 - X * X, 0, 1))
    inside = (X * X + Y * Y) < 1.0
    # mirror about the top edge of the (enlarged) eye: Y -> h + (h - Y) * squash
    Ys = np.where(inside, h + (h - Y) * squash, Y)
    sx = cx + (X * a * k) * math.cos(t) - (Ys * b * k) * math.sin(t)
    sz = cz + (X * a * k) * math.sin(t) + (Ys * b * k) * math.cos(t)
    mapx = ((sx - u0) * sc - 0.5).astype(np.float32)
    mapy = ((S - (sz - v0) * sc) - 0.5).astype(np.float32)
    src = cv2.remap(np.ascontiguousarray(img), mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = (inside.astype(np.uint8) * 255)
    ys_, xs_ = np.nonzero(mask)
    if not len(ys_):
        return
    # low frequencies: a membrane between the colours around the eye (no stripe or brow pulled
    # in); high frequencies: the fur / skin detail of the band above the eye, mirrored down
    sig = max(2.0, b * k * sc * 0.12)
    detail = src.astype(np.float32) - cv2.GaussianBlur(src.astype(np.float32), (0, 0), sig)
    border = img
    if skin_only:
        # skin: a lid is skin even where hair, a brow or the lashes touch the eye's surroundings,
        # so the border colours unlike the cheek's skin under the eye are replaced by it
        from skimage import color
        below = ((X * X + Y * Y) > 1.3 ** 2) & ((X * X + Y * Y) < 2.2 ** 2) & (Y < -0.3)
        lab = color.rgb2lab(img.astype(np.float32) / 255)
        ref = np.median(lab[below], axis=0)
        unlike = lab[..., 0] < ref[0] - 35          # hair and lashes; the eye shadow stays
        ring = ((X * X + Y * Y) < 1.6 ** 2) & ~inside
        border = img.copy()
        skin = np.median(img[below], axis=0)
        border[ring & unlike] = skin
        detail = detail * 0.0
    flat = poisson_patch(border, np.zeros_like(img), inside)
    from scipy import ndimage
    fade = np.clip(ndimage.distance_transform_edt(inside) / (0.3 * b * k * sc), 0, 1)
    fade = (fade * fade * (3 - 2 * fade))[..., None]
    out = flat + 0.9 * detail * fade
    if skin_only:
        # the outer ring of the lid area lies outside the painted eye (eye shadow, the lid's crease):
        # there the closed lid keeps the face's own colours, fading into the membrane inward - no
        # pale patch with an edge where the lid area ends
        rr = np.sqrt(X * X + Y * Y)
        keep = np.clip((rr - 0.72) / 0.22, 0, 1)
        keep = (keep * keep * (3 - 2 * keep))[..., None]
        out = out * (1 - keep) + img.astype(np.float32) * keep
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 1.5)[..., None]
    img[:] = np.clip(img * (1 - soft) + out * soft, 0, 255).astype(np.uint8)


def poisson_patch(dst, src, inside):
    """``src`` inside the mask with its colours shifted by a harmonic (membrane) offset so the patch
    meets ``dst`` exactly on the mask's border: Poisson image editing with the source gradients.
    Solved on the mask's bounding box with one sparse factorisation for the three channels."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu
    ys, xs = np.nonzero(inside)
    y0, y1, x0, x1 = ys.min() - 1, ys.max() + 2, xs.min() - 1, xs.max() + 2
    m = inside[y0:y1, x0:x1]
    d = dst[y0:y1, x0:x1].astype(np.float64)
    s_ = src[y0:y1, x0:x1].astype(np.float64)
    diff = d - s_                                   # the offset on the border
    H, W = m.shape
    idx = -np.ones((H, W), np.int64)
    iy, ix = np.nonzero(m)
    n = len(iy)
    idx[iy, ix] = np.arange(n)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    rhs = np.zeros((n, 3))
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = np.clip(iy + dy, 0, H - 1), np.clip(ix + dx, 0, W - 1)
        j = idx[ny, nx]
        inn = j >= 0
        rows.append(np.arange(n)[inn]); cols.append(j[inn]); vals.append(-np.ones(inn.sum()))
        rhs[~inn] += diff[ny[~inn], nx[~inn]]
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsc()
    off = splu(A).solve(rhs)
    out = dst.astype(np.float64).copy()
    out[y0 + iy, x0 + ix] = s_[iy, ix] + off
    return np.clip(out, 0, 255)


def lash_colour(r, e, k, upper):
    """The darkest colours along the painted eye's outline (the lashes / eyeliner)."""
    img = r["img"].astype(np.float32)
    samples = []
    for u in np.linspace(-0.8, 0.8, 21):
        for kk in (0.9, 0.97, 1.03):
            x, z = ellipse_pt(e, u, 1.0, k * kk)
            px, py = to_px(r, x, z)
            samples.append(img[int(np.clip(py, 0, r["size"] - 1)), int(np.clip(px, 0, r["size"] - 1))])
    samples = np.array(samples)
    lum = samples @ np.array([0.299, 0.587, 0.114])
    c = samples[lum <= np.percentile(lum, 15)].mean(0)
    return np.minimum(c, 70.0) if upper else c * 0.8 + 30


# ------------------------------------------------------------------ body morphs
def front_mask(V, N, J, centre_z, radius):
    """Head vertices that face the camera near the face (morph regions never touch the back of the
    head or the ears)."""
    hx = J["Head"][0]
    return (N[:, 1] < -0.15) & (np.abs(V[:, 2] - centre_z) < radius) & (np.abs(V[:, 0] - hx) < radius * 1.5)


def depth_mask(V, J, x, z, radius, depth):
    """The face's front surface near (x, z), whichever way it faces (lips, mouth corners and the
    mouth's inside move together - a normal test would tear them apart): vertices at most
    ``depth`` behind the frontmost point near (x, z)."""
    near = ((V[:, 0] - x) ** 2 + (V[:, 2] - z) ** 2) < radius ** 2
    if not near.any():
        return np.zeros(len(V))
    y0 = float(V[near, 1].min())
    return (near & (V[:, 1] < y0 + depth)).astype(np.float64)


def falloff(V, c, rx, rz):
    d2 = ((V[:, 0] - c[0]) / rx) ** 2 + ((V[:, 2] - c[1]) / rz) ** 2
    return np.exp(-2.0 * d2) * (d2 < 4.0)


def build(cid: str, model, V: np.ndarray, J: dict, names: list, W: np.ndarray):
    """Returns (body morphs {name: (n, 3)}, parts [...], metadata)."""
    from PIL import Image, ImageDraw
    spec = FACE.get(cid)
    if spec is None:
        return None, None, {}
    human = spec.get("human", False)
    hz = J["HeadTop"][2]
    h = hz - J["Neck"][2]
    hx = J["Head"][0]
    box = (hx - 0.6 * h, hx + 0.6 * h, hz - 1.25 * h, hz - 0.05 * h)
    r = ortho(V, model.F, model.FUV, model.tex, box, 1400)
    eyes = [refine_eye(r, e, human) for e in spec["eyes"]]
    if human:
        spec = dict(spec, mouth=refine_mouth(r, spec["mouth"]))
    fn = np.cross(V[model.F[:, 1]] - V[model.F[:, 0]], V[model.F[:, 2]] - V[model.F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, model.F[:, k], fn)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    morphs = {}
    parts = []
    # --- eyelids and mouth, painted into the texture at runtime (focusbuddy/companion/face_paint.py)
    inpainted(r)
    lid_eyes, lash_r = lid_ellipses(eyes, human)
    for e in lid_eyes:
        fill_eye(r, e, EYE_FILL_K, skin_only=human)
    # --- the closing lid smooths the bulging painted eyeball into a lid's curve (the texture paints
    # the lid itself): eye vertices move onto a shell fitted to the skin around the eye
    blink = np.zeros_like(V)
    for e in eyes:
        cx, cz, a, b, ang = e
        t = math.radians(ang)
        dx, dz = V[:, 0] - cx, V[:, 2] - cz
        X = (dx * math.cos(t) + dz * math.sin(t)) / a
        Y = (-dx * math.sin(t) + dz * math.cos(t)) / b
        rr = X * X + Y * Y
        yfront = float(V[(rr < 1.0), 1].min())
        frontal = V[:, 1] < yfront + 0.12                  # the face, not the back of the head
        # the sculpted eye (the socket rim, the lid crease) is smoothed away too: the shell is fitted
        # to the face just outside the rim
        ring = (rr > BLINK_FIT[0] ** 2) & (rr < BLINK_FIT[1] ** 2) & (N[:, 1] < -0.2) & frontal
        P = V[ring]
        A = np.stack([np.ones(len(P)), P[:, 0], P[:, 2], P[:, 0] ** 2, P[:, 0] * P[:, 2], P[:, 2] ** 2], 1)
        coef, *_ = np.linalg.lstsq(A, P[:, 1], rcond=None)
        # every vertex of the eye goes onto the shell, whichever way it faces (sculpted lashes stick
        # out in front, the socket's walls face sideways): only the back of the head is left alone
        dy = V[:, 1] - coef @ np.stack([np.ones(len(V)), V[:, 0], V[:, 2], V[:, 0] ** 2, V[:, 0] * V[:, 2],
                                        V[:, 2] ** 2])
        sel = frontal & (rr < BLINK_FIT[1] ** 2) & (dy > -0.15) & (dy < 0.08)
        Q = V[sel]
        shell = coef @ np.stack([np.ones(len(Q)), Q[:, 0], Q[:, 2], Q[:, 0] ** 2, Q[:, 0] * Q[:, 2], Q[:, 2] ** 2])
        r1 = np.sqrt(rr[sel])
        shell = shell - 0.25 * b * np.clip(1 - r1 * r1 / 1.2 ** 2, 0, 1)     # a gentle lid dome
        wgt = smooth01((BLINK_FIT[1] - 0.05 - r1) / 0.35)
        blink[sel, 1] += (shell - Q[:, 1]) * wgt
    if human:
        # the people's eyes are painted on a nearly flat, coarse surface (a triangle or two per eye):
        # a shell fitted around them extrapolates badly and would push the lids out of the face, so
        # their blink is the painted lid alone
        blink[:] = 0.0
    morphs["eyeBlink"] = blink
    morphs["eyeSquint"] = blink * 0.45
    # --- brows: the region above each eye (inner half more)
    up = np.zeros_like(V)
    down = np.zeros_like(V)
    for e in eyes:
        cx, cz, a, b, _ = e
        inner = 1.0 if cx < hx else -1.0              # direction toward the face's middle
        c_br = (cx + inner * 0.25 * a, cz + b * (2.0 if human else 1.6))
        m = front_mask(V, N, J, c_br[1], h * 0.6)
        f = falloff(V, c_br, a * 1.3, b * (1.6 if human else 1.1)) * m
        # never the eye itself (the lid patches cover it): fade out inside the eye ellipse
        dz = (V[:, 2] - cz) / (b * 1.2)
        f *= smooth01((dz - 0.9) / 0.6)
        up[:, 2] += f * b * (0.70 if human else 0.45)
        down[:, 2] -= f * b * (0.50 if human else 0.35)
        down[:, 0] += f * inner * a * 0.12
    morphs["browInnerUp"] = up
    morphs["browDown"] = down
    # --- mouth corners
    (lx, lz), (rx_, rz_), mz = spec["mouth"]
    mw = abs(rx_ - lx)
    smile = np.zeros_like(V)
    frown = np.zeros_like(V)
    for (x, z), sgn in (((lx, lz), -1.0), ((rx_, rz_), 1.0)):
        m = depth_mask(V, J, x, z, h * 0.5, mw * 0.6)
        f = falloff(V, (x, z), mw * 0.32, mw * 0.30) * m
        smile[:, 2] += f * mw * (0.12 if human else 0.12)
        smile[:, 0] += f * sgn * mw * 0.05
        frown[:, 2] -= f * mw * (0.10 if human else 0.10)
    morphs["mouthSmile"] = smile
    morphs["mouthFrown"] = frown
    # --- jaw: the lower lip / chin drops; an interior patch unfolds from the lip line
    jaw = np.zeros_like(V)
    cxm = 0.5 * (lx + rx_)
    open_h = mw * (0.45 if human else 0.75)
    if spec["jaw"] == "patch":
        below = (V[:, 2] < mz) & front_mask(V, N, J, mz, h * 0.6)
        f = falloff(V, (cxm, mz - 0.3 * open_h), mw * 0.6, open_h * 1.4) * below
        # the people's lower face is a few big triangles: a small drop, the painted mouth does the rest
        jaw[:, 2] -= f * open_h * (0.12 if human else 0.35)
    else:
        # an already open mouth opens wider: the lower jaw (lip, tongue, chin) drops, blending in
        # smoothly under the mouth line - a hard cut there would tear the muzzle
        below = smooth01((mz - V[:, 2]) / (0.35 * open_h)) * depth_mask(V, J, cxm, mz, h * 0.6, mw * 0.8)
        f = falloff(V, (cxm, mz - 0.3 * open_h), mw * 0.6, open_h * 1.4) * below
        jaw[:, 2] -= f * open_h * 0.3
    morphs["jawOpen"] = jaw
    # --- breath: chest / belly along the normals
    ix = {n: i for i, n in enumerate(names)}
    torso = sum(W[:, ix[n]] for n in ("Spine", "Chest", "Pelvis") if n in ix)
    morphs["breath"] = N * (0.008 * torso)[:, None]
    # --- check view
    im = Image.fromarray(r["img"]).convert("RGB")
    d = ImageDraw.Draw(im)
    for e in eyes:
        pts = [to_px(r, *ellipse_pt(e, u, s, 1.05)) for s in (1, -1) for u in (np.linspace(-1, 1, 40) * s)]
        d.line(pts + [pts[0]], fill=(0, 255, 0), width=2)
        pts = [to_px(r, *ellipse_pt(e, u, s)) for s in (1, -1) for u in (np.linspace(-1, 1, 40) * s)]
        d.line(pts + [pts[0]], fill=(255, 0, 255), width=1)
    for p in ((lx, lz), (rx_, rz_)):
        q = to_px(r, *p)
        d.ellipse([q[0] - 6, q[1] - 6, q[0] + 6, q[1] + 6], outline=(255, 0, 0), width=2)
    im.save(BUILD / f"check_{cid}_face.png")
    painted = painted_data(r, model, V, lid_eyes, spec, human, lash_r=lash_r)
    meta = {"face": {"mode": "painted", "data": "face/painted.npz",
                     "eyes": [[round(x, 5) for x in e] for e in eyes]},
            "morph_map": {k: {"param": v, "scale": 1.0} for k, v in MORPH_MAP.items() if k in morphs},
            "_painted": painted}
    return morphs, parts, meta


def _shape(size) -> tuple:
    """(rows, cols) of an atlas given as one int (square) or (rows, cols)."""
    return (int(size), int(size)) if np.isscalar(size) else (int(size[0]), int(size[1]))


def atlas_texels(V, F, FUV, size, faces):
    """Texels of the atlas (``size``: n for a square one, or (rows, cols)) covered by the given
    faces: (flat index row * cols + col, 3D position, face id)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reconstruct_3d import raster_bary
    Hh, Ww = _shape(size)
    n = max(Hh, Ww)
    uvs = FUV[faces].reshape(-1, 2).astype(np.float64).copy()
    uvs[:, 0] *= Ww / n                         # rasterise on an n x n grid: the atlas is its top-left
    uvs[:, 1] *= Hh / n
    tri = np.arange(len(faces) * 3).reshape(-1, 3)
    fid, bary = raster_bary(uvs, tri, n)
    fid, bary = fid[:Hh, :Ww], bary[:Hh, :Ww]
    m = fid >= 0
    f = faces[fid[m]]
    pos = sum(bary[m][:, i, None] * V[F[f, i]] for i in range(3))
    return np.flatnonzero(m.reshape(-1)), pos, f


def grow(idx, data: list, size: int, steps: int = 3, forbid=None):
    """Extend a texel set by ``steps`` texels in the atlas (copying the nearest member's data), so
    bilinear filtering across UV seams never mixes in an unpainted texel."""
    from scipy import ndimage
    Hh, Ww = _shape(size)
    mask = np.zeros(Hh * Ww, bool)
    mask[idx] = True
    mask = mask.reshape(Hh, Ww)
    rowof = np.full(Hh * Ww, -1, np.int64)
    rowof[idx] = np.arange(len(idx))
    grown = ndimage.binary_dilation(mask, iterations=steps)
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    add = grown & ~mask
    if forbid is not None:
        add &= ~forbid                       # only the gutter: never another surface's texels
    new = np.flatnonzero(add.reshape(-1))
    src = rowof[iy.reshape(-1)[new] * Ww + ix.reshape(-1)[new]]
    all_idx = np.concatenate([idx, new])
    out = [np.concatenate([d, d[src]]) for d in data]
    return [all_idx] + out


def surface_patch(V, F, r, x, z, limit):
    """Faces within ``limit`` of the visible surface point at (x, z), measured along the surface
    (Dijkstra over the mesh edges). A strand of hair hanging in front of an eye is close to it in
    space but far along the surface, so it is never painted as the eye."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    px, py = to_px(r, x, z)
    k = int(r["fid"][int(np.clip(py, 0, r["size"] - 1)), int(np.clip(px, 0, r["size"] - 1))])
    if k < 0:
        return np.ones(len(F), bool)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    # the generated surface has small cracks (unwelded seams): vertices closer than a fraction of
    # the patch's size count as connected too
    from scipy.spatial import cKDTree
    pairs = cKDTree(V).query_pairs(0.08 * limit, output_type="ndarray")
    if len(pairs):
        E = np.concatenate([E, pairs])
    w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    G = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(len(V), len(V))).tocsr()
    d = dijkstra(G, indices=F[k], limit=limit, min_only=True)
    return (d[F] < limit).any(1)          # big triangles: any corner near enough


def painted_data(r, model, V, eyes, spec, human, size=None, lash_r=None):
    """Per-texel data for face_paint.PaintedFace: the eye texels (their place in each eye's
    ellipse, the closed-lid colour), the mouth texels (mouth coordinates), the colours."""
    F = model.F
    if size is None:                            # the exported atlas: 2048 wide, the model's aspect
        th, tw = model.tex.shape[:2]
        size = (int(round(2048 * th / tw)), 2048)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    cen = V[F].mean(1)
    (lx, lz), (rx_, rz_), mz = spec["mouth"]
    mw = abs(rx_ - lx)
    cxm = 0.5 * (lx + rx_)
    # candidate faces: front-facing, near an eye or the mouth, on the visible front surface
    near = np.zeros(len(F), bool)
    for e in eyes:
        cx, cz, a, b, _ = e
        near |= ((cen[:, 0] - cx) ** 2 / (1.5 * a) ** 2 + (cen[:, 2] - cz) ** 2 / (1.5 * b) ** 2) < 1.0
    mouth_near = ((cen[:, 0] - cxm) ** 2 / (0.8 * mw) ** 2 + (cen[:, 2] - (mz - 0.3 * mw)) ** 2 / (0.7 * mw) ** 2) < 1.0
    near |= mouth_near
    near &= fn[:, 1] < 0.35                       # the whole eyeball, its sides too (seen from above / aside)
    faces = np.flatnonzero(near)
    idx, pos, f = atlas_texels(V, F, model.FUV, size, faces)
    # only texels on the surface the front view sees (not a hidden fold behind it)
    px = np.array([to_px(r, p[0], p[2]) for p in pos])
    ix = np.clip(px[:, 0].astype(int), 0, r["size"] - 1)
    iy = np.clip(px[:, 1].astype(int), 0, r["size"] - 1)
    ys = r["pos"][iy, ix, 1]
    front = (r["fid"][iy, ix] >= 0) & (np.abs(pos[:, 1] - ys) < 0.015)
    fill_img = inpainted(r)
    Hh, Ww = _shape(size)
    used = np.zeros(Hh * Ww, bool)                     # texels any surface uses
    used[atlas_texels(V, F, model.FUV, size, np.arange(len(F)))[0]] = True
    used = used.reshape(Hh, Ww)
    out = {"size": np.array([Hh, Ww]), "R": np.array(1.25 if human else EYE_LID_R, np.float32),
           "lash_r": np.array(EYE_LASH_R if lash_r is None else lash_r, np.float32),
           "squint_style": np.array("arch" if human else "rise")}
    e_idx, e_xy, e_fill, e_eye, e_pos = [], [], [], [], []
    lash = []
    for k, e in enumerate(eyes):
        cx, cz, a, b, ang = e
        t = math.radians(ang)
        dx, dz = pos[:, 0] - cx, pos[:, 2] - cz
        X = (dx * math.cos(t) + dz * math.sin(t)) / a
        Y = (-dx * math.sin(t) + dz * math.cos(t)) / b
        sel = (r["fid"][iy, ix] >= 0) & (pos[:, 1] - ys < 0.05) & (X * X + Y * Y <= EYE_SEL_K ** 2)
        sel &= surface_patch(V, F, r, cx, cz, 2.6 * max(a, b))[f]
        e_idx.append(idx[sel])
        e_xy.append(np.stack([X[sel], Y[sel]], 1))
        e_fill.append(fill_img[iy[sel], ix[sel]].astype(np.float32))
        e_eye.append(np.full(int(sel.sum()), k))
        e_pos.append(pos[sel])
        lash.append(lash_colour(r, e, 1.0, True))
    # a texel can be near both eyes only in theory; keep the first
    E = np.concatenate(e_idx)
    _, first = np.unique(E, return_index=True)
    ei, exy, efill, eeye = grow(E[first], [np.concatenate(e_xy)[first], np.concatenate(e_fill)[first],
                                           np.concatenate(e_eye)[first][:, None].astype(np.float32)], size,
                                  forbid=used)
    out["e_idx"] = ei.astype(np.int32)
    # the texture's resolution at each eye (one texel in the eye's height units): the lash line is
    # never drawn thinner than that allows
    tex_u = []
    for k, e in enumerate(eyes):
        m = np.concatenate(e_eye)[first] == k
        ii = E[first][m]
        P = np.concatenate(e_pos)[first][m]
        lut = dict(zip(ii.tolist(), range(len(ii))))
        nb = [(j, lut.get(int(t) + 1)) for j, t in enumerate(ii)]
        d = [np.linalg.norm(P[j] - P[q]) for j, q in nb if q is not None]
        tex_u.append(float(np.median(d)) / e[3] if d else 0.02)
    out["texel"] = np.array(tex_u, np.float32)
    out["e_xy"] = exy.astype(np.float16)
    out["e_fill"] = np.clip(efill, 0, 255).astype(np.uint8)
    out["e_eye"] = eeye[:, 0].astype(np.uint8)
    out["lash"] = np.array(lash, np.float32)
    if spec.get("jaw") == "patch":
        # mouth coordinates: X = 1 at the corners, Y down from the lip line in mouth widths
        X = (pos[:, 0] - cxm) / (0.5 * mw)
        Y = (mz - pos[:, 2]) / mw * 2.0
        sel = front & (np.abs(X) < 1.2) & (Y > -0.3) & (Y < 1.6)
        sel &= ~np.isin(idx, out["e_idx"])
        mi, mxy = grow(idx[sel], [np.stack([X[sel], Y[sel]], 1)], size, forbid=used)
        keep = ~np.isin(mi, out["e_idx"])
        out["m_idx"] = mi[keep].astype(np.int32)
        out["m_xy"] = mxy[keep].astype(np.float16)
        lipc = r["img"][int(to_px(r, cxm, mz)[1]), int(to_px(r, cxm, mz)[0])].astype(np.float32)
        out["m_cols"] = np.array([[60, 18, 26], [205, 95, 105] if not human else [200, 100, 110],
                                  list(lipc * 0.55)], np.float32)
        # a person's mouth opens between the lips (smaller, centred near the line); an animal's
        # small painted mouth hangs open below its line
        out["m_shape"] = np.array([0.42, 0.45] if human else [0.62, 1.0], np.float32)
    return out
