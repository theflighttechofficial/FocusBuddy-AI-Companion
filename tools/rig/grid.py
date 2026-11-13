"""Annotation views: orthographic renders with a labelled coordinate grid (rig frame), used to
place and check joints by eye.

    python tools/rig/grid.py fox side_left|side_right|front|top|back [joints.json] [crop x0 x1 z0 z1]
"""

from __future__ import annotations

import json
import sys

import numpy as np
from PIL import Image, ImageDraw

from core import BUILD, Model, raster, shaded_colors

# view -> (u axis, v axis, u sign, depth axis, depth sign, vertex filter)
VIEWS = {
    "side_left": (1, 2, -1.0, 0, 1.0, lambda V: V[:, 0] > -0.01),     # camera at +X: the left half, front to the right
    "side_right": (1, 2, -1.0, 0, -1.0, lambda V: V[:, 0] < 0.01),    # camera at -X looking +X (mirrored to match)
    "front": (0, 2, 1.0, 1, -1.0, None),
    "back": (0, 2, 1.0, 1, 1.0, None),
    "top": (0, 1, 1.0, 2, 1.0, None),
}


def render(model: Model, view: str, size=1000, box=None, joints=None, parents=None):
    ua, va, us, da, ds, flt = VIEWS[view]
    F = model.F
    col = shaded_colors(model)
    if flt is not None:
        keep = flt(model.V)[F].all(1)
        F, col = F[keep], col[keep]
    if box is not None:
        (u0, u1, v0, v1) = box
        b = (np.array([u0, v0]), max(u1 - u0, v1 - v0))
    else:
        b = None
    img, frame = raster(model.V, F, col, (ua, va, us), size, depth_axis=da, depth_sign=ds, box=b)
    lo, sc, size, _ = frame
    pil = Image.fromarray(img.astype(np.uint8))
    d = ImageDraw.Draw(pil)
    span = size / sc
    step = 0.05 if span > 0.5 else 0.02 if span > 0.2 else 0.01
    for k in np.arange(np.ceil(lo[0] / step) * step, lo[0] + span, step):
        x = (k - lo[0]) * sc
        major = abs(round(k / (step * 2)) * step * 2 - k) < 1e-9
        d.line([(x, 0), (x, size)], fill=(120, 160, 230) if major else (200, 210, 235), width=1)
        if major:
            d.text((x + 2, 2), f"{k * us:+.2f}", fill=(30, 60, 160))
    for k in np.arange(np.ceil(lo[1] / step) * step, lo[1] + span, step):
        y = size - (k - lo[1]) * sc
        major = abs(round(k / (step * 2)) * step * 2 - k) < 1e-9
        d.line([(0, y), (size, y)], fill=(120, 160, 230) if major else (200, 210, 235), width=1)
        if major:
            d.text((2, y + 1), f"{k:.2f}", fill=(30, 60, 160))
    if joints:
        def p2(p):
            return ((p[ua] * us - lo[0]) * sc, size - (p[va] - lo[1]) * sc)
        for n, p in joints.items():
            par = (parents or {}).get(n)
            if par in joints:
                d.line([p2(joints[par]), p2(p)], fill=(250, 120, 0), width=3)
        for n, p in joints.items():
            x, y = p2(p)
            c = (220, 0, 0) if n.startswith("Left") else (0, 150, 0) if n.startswith("Right") else (0, 0, 0)
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=c, outline=(255, 255, 255))
            d.text((x + 7, y - 7), n.replace("Left", "L").replace("Right", "R"), fill=(120, 0, 120))
    return pil


if __name__ == "__main__":
    cid, view = sys.argv[1], sys.argv[2]
    m = Model(cid)
    J = P = None
    if len(sys.argv) > 3 and sys.argv[3] != "-":
        data = json.loads(open(sys.argv[3]).read())
        J = {k: np.array(v) for k, v in data["joints"].items()}
        P = data.get("parents")
    box = tuple(float(x) for x in sys.argv[4:8]) if len(sys.argv) > 7 else None
    render(m, view, 1000, box, J, P).save(BUILD / f"grid_{cid}_{view}.png")
