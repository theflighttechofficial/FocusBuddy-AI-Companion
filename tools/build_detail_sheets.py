"""Cut the detailed character sheets (``art/sheets/v2/<sheet>.webp``) into the pieces the 3D build uses.

Every detailed sheet (1536 x 1024) has a big FRONT VIEW, a SIDE VIEW (RIGHT), a 6 x 2 grid of FACIAL
EXPRESSIONS, a row of TURNAROUND VIEWS (front, 3/4 front, side right, 3/4 back, back, side left) and
detail panels. This tool writes to ``art/cells/v2/``:

* ``<sheet>_front.png``, ``<sheet>_side.png``: the two big views, background removed (ISNet), the
  interior made opaque (dark eyes / hair must not turn see-through) and stray annotation text dropped;
* ``<sheet>_turn_<view>.png``: the turnaround figures, cut out the same way;
* ``<sheet>_expr_<name>.png``: the 12 expression panels as plain crops (the build registers them to
  the front view's face, so no cut-out is needed);
* ``layout.json``: every box used, so the cut is reproducible and reviewable.

    python tools/build_detail_sheets.py            # all seven sheets
    python tools/build_detail_sheets.py cat boy    # some of them

Needs ``rembg`` + onnxruntime (the ISNet model downloads on first use), Pillow, NumPy, SciPy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SHEETS = ROOT / "art" / "sheets" / "v2"
OUT = ROOT / "art" / "cells" / "v2"

# boxes (x0, y0, x1, y1) on the 1536 x 1024 sheets
FRONT = {"cat": (15, 40, 440, 560), "dog": (20, 40, 420, 555), "rabbit": (90, 15, 420, 555),
         "fox": (20, 10, 480, 545), "boy": (80, 15, 330, 590), "saree": (80, 20, 330, 625),
         "girl": (80, 15, 340, 595)}
SIDE = {"cat": (420, 30, 820, 516), "dog": (410, 35, 830, 555), "rabbit": (490, 40, 830, 555),
        "fox": (420, 40, 860, 545), "boy": (440, 30, 680, 595), "saree": (430, 25, 830, 632),
        "girl": (470, 25, 720, 595)}
TURN_ROW = {"cat": (0, 608, 862, 818), "dog": (0, 603, 862, 818), "rabbit": (0, 592, 862, 815),
            "fox": (0, 598, 862, 815), "boy": (0, 622, 762, 826), "saree": (0, 662, 772, 846),
            "girl": (0, 622, 766, 826)}
TURN_VIEWS = ["front", "front34", "right", "back34", "back", "left"]
# the expression grid (found by detect_panels inside this block)
EXPR_BLOCK = {"cat": (862, 38, 1532, 362), "dog": (860, 38, 1532, 358), "rabbit": (862, 36, 1532, 358),
              "fox": (878, 38, 1532, 360), "boy": (810, 42, 1532, 364), "saree": (826, 38, 1532, 360),
              "girl": (822, 40, 1532, 366)}
# panel labels, row by row, as printed on each sheet
_ANIMAL_A = "neutral happy curious surprised playful sleepy angry sad excited confident mischievous affection"
_ANIMAL_B = "neutral happy curious surprised playful sleepy excited angry sad confident mischievous affection"
EXPR_NAMES = {
    "cat": _ANIMAL_A.split(), "fox": _ANIMAL_A.split(),
    "dog": _ANIMAL_B.split(), "rabbit": _ANIMAL_B.split(),
    "boy": "neutral happy excited curious surprised playful sleepy focused confident sad angry mischievous".split(),
    "saree": "neutral happy cheerful curious surprised playful sleepy focused confident sad annoyed mischievous".split(),
    "girl": "neutral happy cheerful curious surprised playful sleepy focused confident sad annoyed mischievous".split(),
}


def _runs(mask):
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        if not m and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def detect_panels(sheet: np.ndarray, block) -> list:
    """The 6 x 2 expression panels: panel image areas are bright somewhere in every column / row,
    the gaps between them stay dark. Panels share one width (dark hair can hide a panel's left edge,
    so short panels are re-measured from their right edge)."""
    x0, y0, x1, y1 = block
    g = sheet[y0:y1, x0:x1].astype(np.float32).mean(2)
    rows = [r for r in _runs(np.percentile(g, 90, 1) > 60) if r[1] - r[0] > 60][:2]
    cols_per_row = []
    for r0, r1 in rows:
        cols = [c for c in _runs(np.percentile(g[r0:r1], 92, 0) > 55) if c[1] - c[0] > 50]
        cols_per_row.append(cols)
    W = max(int(np.percentile([c1 - c0 for c0, c1 in cols], 85)) for cols in cols_per_row)
    panels = []
    for (r0, r1), cols in zip(rows, cols_per_row):
        for c0, c1 in cols:
            if c1 - c0 < 0.93 * W:
                c0 = c1 - W
            panels.append((x0 + c0, y0 + r0, x0 + c1, y0 + r1))
    return panels


def cutout(img, session, scale: int = 2) -> np.ndarray:
    """Background removal + clean-up: keep the big parts, make the interior opaque."""
    from PIL import Image
    from rembg import remove
    from scipy import ndimage as ndi
    big = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    a = np.array(remove(big, session=session).convert("RGBA"))
    m = a[..., 3] > 128
    lab, n = ndi.label(ndi.binary_dilation(m, iterations=3))
    if n:
        sz = ndi.sum(m, lab, range(1, n + 1))
        keep = np.isin(lab, 1 + np.flatnonzero(sz > 0.05 * sz.max()))
        a[..., 3] = np.where(keep, a[..., 3], 0)
    solid = ndi.binary_fill_holes(a[..., 3] > 40)
    d = ndi.distance_transform_edt(solid)
    a[..., 3] = np.where(d > 5, 255, np.where(solid, np.maximum(a[..., 3], (d / 5 * 255).astype(np.uint8)),
                                               a[..., 3]))
    return a


def split_row(a: np.ndarray, n: int = 6) -> list:
    """Split a cut-out turnaround row into its ``n`` figures at the emptiest columns."""
    occ = (a[..., 3] > 64).sum(0).astype(np.float32)
    W = a.shape[1]
    cuts = [0]
    for k in range(1, n):
        c = int(W * k / n)
        lo, hi = c - W // (3 * n), c + W // (3 * n)
        cuts.append(lo + int(np.argmin(occ[lo:hi])))
    cuts.append(W)
    out = []
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        piece = a[:, c0:c1].copy()
        from scipy import ndimage as ndi
        m = piece[..., 3] > 64
        lab, k = ndi.label(ndi.binary_dilation(m, iterations=2))
        if k > 1:       # drop slivers of the neighbouring figure (a tail tip crossing the cut)
            sz = ndi.sum(m, lab, range(1, k + 1))
            piece[..., 3] = np.where(np.isin(lab, 1 + np.flatnonzero(sz > 0.2 * sz.max())), piece[..., 3], 0)
        ys, xs = np.nonzero(piece[..., 3] > 16)
        out.append(piece[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(xs) else piece)
    return out


def build(sheet: str, session) -> dict:
    from PIL import Image
    im = Image.open(SHEETS / f"{sheet}.webp").convert("RGB")
    arr = np.asarray(im)
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, boxes in (("front", FRONT), ("side", SIDE)):
        Image.fromarray(cutout(im.crop(boxes[sheet]), session)).save(OUT / f"{sheet}_{tag}.png")
    row = cutout(im.crop(TURN_ROW[sheet]), session)
    for view, piece in zip(TURN_VIEWS, split_row(row)):
        Image.fromarray(piece).save(OUT / f"{sheet}_turn_{view}.png")
    panels = detect_panels(arr, EXPR_BLOCK[sheet])
    if len(panels) != 12:
        raise RuntimeError(f"{sheet}: found {len(panels)} expression panels, expected 12")
    for name, box in zip(EXPR_NAMES[sheet], panels):
        im.crop(box).save(OUT / f"{sheet}_expr_{name}.png")
    return {"front": FRONT[sheet], "side": SIDE[sheet], "turn_row": TURN_ROW[sheet],
            "expressions": dict(zip(EXPR_NAMES[sheet], [list(map(int, b)) for b in panels]))}


def main(argv: list[str]) -> int:
    from rembg import new_session
    session = new_session("isnet-general-use")
    sheets = argv or sorted(FRONT)
    path = OUT / "layout.json"
    layout = json.loads(path.read_text()) if path.exists() else {}
    for s in sheets:
        layout[s] = build(s, session)
        print(f"{s}: done", flush=True)
    path.write_text(json.dumps(layout, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
