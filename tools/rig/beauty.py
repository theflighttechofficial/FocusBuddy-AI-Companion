"""Retouch a person's face tile (``face_tile.py``) as a 2D picture - the front view of the face.

* **bake seam**: the generator's texture bake leaves a horizontal seam across the face (two views
  meet); the colours on both sides are matched column by column (a smooth correction that fades
  away from the seam) and the seam line itself is softened;
* **cut eyes** (``"complete_eyes"``): where the seam cut the lower half of the eyes away (Study
  Girl), the lower half is rebuilt from the upper half mirrored, with a softer lower lid line and
  the catch-lights left only on top;
* **skin**: evened out with an edge-preserving filter (blemishes and bake noise, not features);
* **eyes, lashes, brows, lips**: crisper (unsharp mask) and a little more contrast; a heavy dark
  ring painted under the eyes (``"lift_under_eyes"``) is lifted toward the skin colour.

Every step works on the tile picture only; ``face_tile.build`` maps the tile onto the face.
"""

from __future__ import annotations

import numpy as np

# per character: what the retouch does beyond the common steps
RECIPES = {
    "buddy": {"skin": 0.55, "under_eye": 0.15},
    "study_girl": {"skin": 0.6, "complete_eyes": True, "under_eye": 0.3},
    "saree_girl": {"skin": 0.6, "under_eye": 0.35, "clean_lashes": True},
}


def lab(img: np.ndarray) -> np.ndarray:
    from skimage import color
    return color.rgb2lab(np.clip(img, 0, 255).astype(np.float32) / 255.0)


def skin_mask(img: np.ndarray, covered: np.ndarray, ref_pts: list) -> tuple:
    """Pixels whose colour is close to the skin sampled at ``ref_pts`` (cheeks); returns (mask,
    soft weight 0..1, the reference Lab colour)."""
    L = lab(img)
    samples = np.concatenate([L[max(0, y - 12):y + 12, max(0, x - 12):x + 12].reshape(-1, 3) for x, y in ref_pts])
    ref = np.median(samples, axis=0)
    d = np.linalg.norm((L - ref) * np.array([0.6, 1.0, 1.0]), axis=2)
    w = np.clip(1.0 - (d - 10.0) / 14.0, 0.0, 1.0) * covered
    return w > 0.5, w, ref


# ------------------------------------------------------------------ seam
def find_seam(img: np.ndarray, covered: np.ndarray, rows: tuple, cols: tuple) -> int:
    """Row of the strongest horizontal colour break across the face (between ``rows``)."""
    y0, y1 = rows
    x0, x1 = cols
    best, by = -1.0, -1
    for y in range(y0 + 10, y1 - 10):
        a = img[y - 9, x0:x1]
        b = img[y + 9, x0:x1]
        ok = covered[y - 9, x0:x1] & covered[y + 9, x0:x1]
        if ok.sum() < 20:
            continue
        jump = (np.abs(a - b).sum(1) > 30)[ok].mean()
        # a seam is a straight line: its break is sharp, unlike a soft shadow
        if jump > best:
            best, by = jump, y
    return by if best > 0.25 else -1


def heal_seam_line(img: np.ndarray, covered: np.ndarray, y: int, half: int = 6, skip_cols=None) -> np.ndarray:
    """The rows right at the seam carry sampling noise from both texture islands (a saw-tooth of
    light / dark texels): repaint them from the rows around."""
    import cv2
    H, W = covered.shape
    mask = np.zeros((H, W), np.uint8)
    mask[max(0, y - half):y + half + 1] = 255
    mask &= (covered * 255).astype(np.uint8)
    if skip_cols is not None:
        mask[:, skip_cols] = 0
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    fixed = cv2.inpaint(u8, mask, 5, cv2.INPAINT_TELEA).astype(np.float32)
    return np.where(mask[..., None] > 0, fixed, img)


def blend_seam(img: np.ndarray, covered: np.ndarray, y: int, fade: float = 70.0) -> np.ndarray:
    from scipy import ndimage
    H, W = covered.shape
    out = img.copy()
    up = np.median(img[y - 16:y - 6], axis=0)          # (W, 3) the colours just above
    dn = np.median(img[y + 6:y + 16], axis=0)          # just below
    okc = covered[y - 16:y + 16].all(0)
    Lu, Ld = lab(up[None])[0], lab(dn[None])[0]
    same = okc & (np.linalg.norm(Lu - Ld, axis=1) < 22)          # the same kind of surface on both sides
    diff = np.zeros((W, 3), np.float32)
    if same.sum() < 10:
        return out
    xs = np.arange(W)
    for c in range(3):
        d = dn[:, c] - up[:, c]
        diff[:, c] = np.interp(xs, xs[same], d[same])
    diff = ndimage.gaussian_filter1d(diff, 18, axis=0)
    yy = np.arange(H)[:, None, None]
    w_up = np.where(yy < y, np.exp(-(y - yy) / fade), 0.0)
    w_dn = np.where(yy >= y, np.exp(-(yy - y) / fade), 0.0)
    out = out + 0.5 * diff[None] * w_up - 0.5 * diff[None] * w_dn
    # soften the line itself
    band = slice(max(0, y - 12), min(H, y + 12))
    soft = ndimage.gaussian_filter1d(out[band], 4.0, axis=0)
    k = np.exp(-((np.arange(band.start, band.stop) - y) / 7.0) ** 2)[:, None, None]
    out[band] = out[band] * (1 - k) + soft * k
    return np.where(covered[..., None], out, img)


# ------------------------------------------------------------------ eyes
def complete_eye(img: np.ndarray, cx: float, half_w: float, y_s: int) -> np.ndarray:
    """Rebuild the lower half of an eye the seam at row ``y_s`` cut away: the lower part of the
    opening above the seam (never the lashes) is flipped, squashed into a lower half that is flatter
    than the top and pasted under the seam inside a lower-lid arc, which gets a soft lid line.
    Catch-lights in the flipped part are painted out (one highlight, on top)."""
    import cv2
    from scipy import ndimage
    H, W = img.shape[:2]
    y_s = y_s - 5                                          # above the seam's noisy rows
    lum = ndimage.gaussian_filter(img.mean(2), 1.5)
    x0, x1 = int(cx - half_w), int(cx + half_w) + 1
    xs = np.arange(x0, x1)
    reach = int(1.2 * half_w)
    lash = np.array([y_s - reach + int(np.argmin(lum[y_s - reach:y_s - 2, x])) for x in xs], float)
    lash = ndimage.gaussian_filter1d(ndimage.median_filter(lash, 9), 4)
    mid = (np.abs(xs - cx) < 0.4 * half_w)
    h_c = float(np.median((y_s - lash)[mid]))              # the visible opening at the middle
    take = max(4, int(0.72 * h_c))                         # rows of opening to flip (no lashes)
    depth_max = max(4, int(0.8 * h_c) + 5)
    patch = img[y_s - take:y_s, x0:x1][::-1]
    patch = cv2.resize(patch, (x1 - x0, depth_max), interpolation=cv2.INTER_LINEAR)
    # catch-lights in the flipped part: painted out from their surroundings
    hl = (patch.mean(2) > 185).astype(np.uint8) * 255
    hl = cv2.dilate(hl, np.ones((5, 5), np.uint8))
    if hl.any():
        patch = cv2.inpaint(np.clip(patch, 0, 255).astype(np.uint8), hl, 4, cv2.INPAINT_TELEA).astype(np.float32)
    # darker toward the lower lid (its shadow)
    fy = (np.arange(depth_max) / max(depth_max - 1, 1))[:, None, None]
    patch = patch * (1.0 - 0.2 * fy ** 2)
    t = np.clip((xs - cx) / half_w, -1, 1)
    arc = depth_max * np.clip(1 - t * t, 0, 1) ** 0.75                  # the lower lid, per column
    yy = np.arange(depth_max)[:, None]
    inside = np.clip(arc[None, :] - yy + 0.5, 0, 1)                    # anti-aliased arc
    region = np.zeros((H, W), np.float32)
    region[y_s:y_s + depth_max, x0:x1] = inside
    region = ndimage.gaussian_filter(region, 0.8)
    out = img.copy()
    out[y_s:y_s + depth_max, x0:x1] = patch
    res = img * (1 - region[..., None]) + out * region[..., None]
    # the lower lid line along the arc
    below = img[y_s + depth_max + 4:y_s + depth_max + 14, x0:x1].reshape(-1, 3)
    lid_col = np.median(below, axis=0) if len(below) else np.array([200.0, 150.0, 130.0])
    line_col = 0.5 * lid_col + 0.5 * np.array([95.0, 60.0, 55.0])
    line = np.zeros((H, W), np.float32)
    for i, x in enumerate(xs):
        d = arc[i]
        if d < 1.0:
            continue
        for y in range(y_s, min(H, int(y_s + d + 4))):
            line[y, x] = max(0.0, 1.0 - abs(y - y_s - d) / 2.0) * min(1.0, d / 6.0) * 0.7
    line = ndimage.gaussian_filter(line, 0.7)
    return res * (1 - line[..., None]) + line_col * line[..., None]


def clean_lashes(img: np.ndarray, cx: float, rx: float, cy: float, ry: float, y_s: int) -> np.ndarray:
    """The seam and an atlas-island edge run along the upper lid: the lash band is peppered with
    specks of the wrong colour (light and dark). A median filter (it removes specks of either
    polarity and keeps the band's edges) is applied in a band along the eye's upper outline only -
    the iris, the white and the skin around stay untouched."""
    import cv2
    from scipy import ndimage
    H, W = img.shape[:2]
    pad = int(1.6 * max(rx, ry))
    x0, x1 = max(0, int(cx - pad)), min(W, int(cx + pad))
    y0, y1 = max(0, int(cy - pad)), min(H, int(cy + 0.6 * pad))
    roi = np.clip(img[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    med = cv2.medianBlur(cv2.medianBlur(roi, 7), 5).astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    band = np.clip((1.45 - d) / 0.12, 0, 1) * np.clip((d - 0.78) / 0.12, 0, 1)
    band *= np.clip((cy + 0.15 * ry - yy) / (0.2 * ry), 0, 1)          # the upper outline and the corners
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 1.5)
    out = img.copy()
    out[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - band[..., None]) + med * band[..., None]
    return out


def unsharp(img: np.ndarray, mask: np.ndarray, radius: float, amount: float) -> np.ndarray:
    from scipy import ndimage
    blur = ndimage.gaussian_filter(img, (radius, radius, 0))
    sharp = img + amount * (img - blur)
    return img * (1 - mask[..., None]) + sharp * mask[..., None]


def ellipse_mask(shape, cx, cy, rx, ry, soft: float = 0.25) -> np.ndarray:
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    return np.clip((1.0 + soft - d) / soft, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ all of it
def retouch(cid: str, img: np.ndarray, covered: np.ndarray, eyes_px: list, mouth_px: tuple,
            log: list | None = None) -> np.ndarray:
    """``eyes_px``: [(cx, cy, rx, ry)] tile pixels; ``mouth_px``: (cx, cy, half width)."""
    import cv2
    from scipy import ndimage
    rec = RECIPES.get(cid, {})
    log = log if log is not None else []
    img = img.astype(np.float32)
    H, W = covered.shape
    (ex0, ey0, erx, ery), (ex1, ey1, _, _) = eyes_px[0], eyes_px[1]
    eye_dist = abs(ex1 - ex0)
    mcx, mcy, mhw = mouth_px
    cheeks = [(int(ex0), int(0.5 * (ey0 + mcy))), (int(ex1), int(0.5 * (ey1 + mcy))),
              (int(0.5 * (ex0 + ex1)), int(0.62 * mcy + 0.38 * ey0))]
    # 1. the bake seam
    rows = (int(min(ey0, ey1) - 1.2 * eye_dist), int(mcy - 0.2 * (mcy - ey0)))
    cols = (int(min(ex0, ex1) - 0.6 * eye_dist), int(max(ex0, ex1) + 0.6 * eye_dist))
    y_s = find_seam(img, covered, rows, cols)
    log.append(f"seam row {y_s}")
    keep_cols = np.zeros(W, bool)
    # 2. eyes the seam cut in half (before the seam is healed: they are rebuilt from above it)
    if rec.get("complete_eyes") and y_s > 0:
        for (cx, cy, rx, ry) in eyes_px:
            if cy - ry < y_s < cy + 2.5 * ry:
                img = complete_eye(img, cx, 0.85 * rx, y_s)
                keep_cols[int(cx - 0.85 * rx):int(cx + 0.85 * rx) + 1] = True
                log.append(f"completed eye at x {cx:.0f}")
    if y_s > 0:
        img = heal_seam_line(img, covered, y_s, skip_cols=keep_cols)
        img = blend_seam(img, covered, y_s)
        if rec.get("clean_lashes"):
            for (cx, cy, rx, ry) in eyes_px:
                if cy - 1.6 * ry < y_s < cy:
                    img = clean_lashes(img, cx, rx, cy, ry, y_s)
                    log.append(f"cleaned lashes at x {cx:.0f}")
    _, skin_w, ref = skin_mask(img, covered, cheeks)
    # 3. skin: edge-preserving smoothing, only on skin
    feat = np.zeros((H, W), np.float32)
    for (cx, cy, rx, ry) in eyes_px:
        feat = np.maximum(feat, ellipse_mask((H, W), cx, cy, 1.5 * rx, 2.6 * ry))
        feat = np.maximum(feat, ellipse_mask((H, W), cx, cy - 2.4 * ry, 1.7 * rx, 1.6 * ry))   # brows
    feat = np.maximum(feat, ellipse_mask((H, W), mcx, mcy, 1.35 * mhw, 0.55 * mhw))
    amount = float(rec.get("skin", 0.5))
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    smooth = cv2.bilateralFilter(u8, 13, 18, 9).astype(np.float32)
    smooth = cv2.bilateralFilter(np.clip(smooth, 0, 255).astype(np.uint8), 9, 12, 6).astype(np.float32)
    w = amount * skin_w * (1.0 - feat)
    img = img * (1 - w[..., None]) + smooth * w[..., None]
    # 4. a heavy dark ring under the eyes, lifted toward the skin
    lift = float(rec.get("under_eye", 0.0))
    if lift > 0:
        L = lab(img)
        for (cx, cy, rx, ry) in eyes_px:
            m = ellipse_mask((H, W), cx, cy + 2.1 * ry, 1.2 * rx, 1.2 * ry, soft=0.6)
            m *= 1.0 - ellipse_mask((H, W), cx, cy, 1.05 * rx, 1.2 * ry, soft=0.3)    # never the eye
            m *= np.clip((ref[0] - L[..., 0]) / 25.0, 0, 1)            # only where it is darker than skin
            m *= covered
            target = img * 0.0 + np.array(lab_to_rgb(ref))[None, None]
            img = img + (target - img) * (lift * m)[..., None]
    # 5. crisper eyes, brows and lips
    eyes_m = np.zeros((H, W), np.float32)
    for (cx, cy, rx, ry) in eyes_px:
        eyes_m = np.maximum(eyes_m, ellipse_mask((H, W), cx, cy, 1.25 * rx, 1.9 * ry))
    img = unsharp(img, eyes_m * covered, 3.0, 0.45)
    img = unsharp(img, eyes_m * covered, 1.2, 0.3)
    brow_m = np.zeros((H, W), np.float32)
    for (cx, cy, rx, ry) in eyes_px:
        brow_m = np.maximum(brow_m, ellipse_mask((H, W), cx, cy - 2.6 * ry, 1.6 * rx, 1.3 * ry))
    img = unsharp(img, brow_m * covered * (1 - skin_w), 2.5, 0.6)
    lip_m = ellipse_mask((H, W), mcx, mcy, 1.25 * mhw, 0.5 * mhw) * covered
    img = unsharp(img, lip_m, 2.5, 0.5)
    # lips: a touch more colour
    g = img.mean(2, keepdims=True)
    img = img + (img - g) * (0.12 * lip_m)[..., None]
    img = np.where(covered[..., None], img, img)
    return np.clip(img, 0, 255)


def lab_to_rgb(L) -> list:
    from skimage import color
    return list(np.clip(color.lab2rgb(np.array(L, np.float64)[None, None])[0, 0] * 255, 0, 255))
