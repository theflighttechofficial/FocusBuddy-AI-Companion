"""Build the illustrated character packs from the character-sheet art.

The source sheets (one per character: turnaround views, facial expressions, full-body actions,
emotional body language, detail views) live in ``art/sheets/<sheet>.webp``. The art was provided
by the FocusBuddy owner. Three stages:

    python tools/build_illustrated_packs.py detect    # find the pose cells -> art/sheets/cells.json
    python tools/build_illustrated_packs.py extract   # cut every pose out    -> art/cells/ (cache)
    python tools/build_illustrated_packs.py pack      # normalise + write     -> assets/characters/<id>/illustrated/

``detect`` and ``extract`` need ``rembg`` (+ onnxruntime and the ISNet / U2-Net models, downloaded
on first use); ``pack`` needs only Pillow, NumPy and SciPy. ``cells.json`` and the packs are
committed, so the app never needs the extraction models.

Background removal runs per pose cell (3x Lanczos upscaled) with two models: ISNet with alpha
matting gives the soft fur / hair edges, U2-Net's solid mask fills bodies that ISNet loses on
the dark studio background. ``pack`` then fills interior holes, removes leftover label bars,
un-mixes the dark background from the semi-transparent edge pixels, puts every pose on one
scale (the turnaround row is drawn bigger than the action rows), finds the ground contact and
head positions, and writes WebP frames + ``pack.json`` (format: focusbuddy/companion/illustrated.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHEETS = ROOT / "art" / "sheets"
CELLS = ROOT / "art" / "cells"
OUT = ROOT / "assets" / "characters"

# sheet -> character id, body kind
CHARACTERS = {
    "cat": ("cat", "quadruped"), "dog": ("dog", "quadruped"), "rabbit": ("rabbit", "quadruped"),
    "fox": ("fox", "quadruped"), "robot": ("robot", "biped"), "boy": ("buddy", "biped"),
    "girl": ("study_girl", "biped"), "saree": ("saree_girl", "biped"),
}

# rows of each sheet (y ranges in the 1536 x 1024 sheet): turnaround / expressions / actions / body language
BANDS = {
    "cat": {"turn": (30, 300), "expr": (332, 466), "act": (520, 668), "body": (697, 852)},
    "dog": {"turn": (30, 298), "expr": (332, 458), "act": (521, 668), "body": (697, 850)},
    "rabbit": {"turn": (34, 300), "expr": (332, 462), "act": (517, 666), "body": (696, 850)},
    "fox": {"turn": (34, 283), "expr": (314, 448), "act": (508, 663), "body": (692, 848)},
    "robot": {"turn": (30, 290), "expr": (320, 455), "act": (509, 643), "body": (696, 820)},
    "boy": {"turn": (50, 298), "expr": (330, 450), "act": (505, 665), "body": (697, 850)},
    "girl": {"turn": (50, 308), "expr": (339, 460), "act": (515, 672), "body": (702, 852)},
    "saree": {"turn": (44, 304), "expr": (334, 457), "act": (486, 662), "body": (696, 853)},
}
TURN = "front back left right front34 back34 top bottom".split()
_B13 = "happy curious playful confident scared angry sad excited relaxed sleepy grumpy love attentive"
ACT = {k: v.split() for k, v in {
    "cat": "idle walk run sprint jump_start jump_air jump_land sit lie_down sleep stretch groom",
    "dog": "idle walk run sprint jump_start jump_air jump_land sit lie_down roll_over play_bow stretch sleep shake",
    "rabbit": "idle walk run sprint jump_start jump_air jump_land sit lie_down sleep stretch groom stand_up",
    "fox": "idle walk run sprint jump_start jump_air jump_land sit lie_down sleep stretch groom sneak",
    "robot": "idle walk run sprint jump_start jump_air jump_land sit lie_down sleep stretch wave point dance",
    "boy": "idle walk run sprint jump_start jump_air jump_land sit lie_down sleep stretch yawn wave point think read",
    "girl": "idle walk run sprint jump_start jump_air jump_land sit lie_down sleep stretch yawn point think read",
    "saree": "idle walk run sprint jump_start jump_air jump_land sit lie_down sleep stretch spin wave point read laptop",
}.items()}
BODY = {k: v.split() for k, v in {
    "cat": "happy curious playful confident scared angry sad excited relaxed sleepy grumpy love",
    "dog": "happy curious playful confident scared angry sad excited relaxed sleepy grumpy grumpy2 love attentive",
    "rabbit": _B13, "fox": _B13, "robot": _B13, "boy": _B13 + " study", "girl": _B13 + " study",
    "saree": _B13 + " study"}.items()}

# which way each pose looks: 1 right, -1 left, 0 the viewer (unlisted = 0). Side views are
# matched to the walk pose automatically.
FACING = {
    "cat": dict(idle=-1, walk=1, run=1, sprint=1, jump_start=1, jump_air=1, jump_land=1, lie_down=1, sleep=1,
                stretch=-1, happy=1, playful=1, confident=1, scared=1, angry=1, sad=1, excited=1, relaxed=1),
    "dog": dict(walk=1, run=1, sprint=1, jump_start=1, jump_air=1, jump_land=1, play_bow=1, sleep=1,
                happy=1, curious=1, playful=1, scared=1, angry=1, sad=1, excited=1),
    "rabbit": dict(idle=1, walk=1, run=1, sprint=1, jump_start=1, jump_air=1, jump_land=1, lie_down=1, sleep=1,
                   stretch=1, happy=1, curious=1, playful=1, scared=1, sad=1),
    "fox": dict(idle=1, walk=1, run=1, sprint=1, jump_air=1, lie_down=1, sleep=1, stretch=-1, sneak=1,
                happy=1, curious=1, playful=1, confident=1, scared=1, angry=1, sad=1, excited=1, relaxed=1,
                attentive=1),
    "robot": dict(walk=1, run=1, sprint=1, jump_start=1, lie_down=1, sleep=1, point=-1),
    "boy": dict(walk=1, run=1, sprint=1, jump_start=1, lie_down=1, sleep=-1, point=1, curious=1, playful=1),
    "girl": dict(walk=1, run=1, sprint=1, jump_start=1, sleep=-1, point=1, curious=1),
    "saree": dict(walk=1, run=1, sprint=1, jump_start=1, lie_down=-1, sleep=-1, point=1),
}
STAND_PX = {"quadruped": 300, "biped": 380}       # stored height of a standing pose (px)
STAND_HEIGHT = {"quadruped": 0.62, "biped": 0.80}  # standing height as a fraction of the companion box


# ============================================================ detect
def detect() -> None:
    """Pose cells per row: components of a whole-row segmentation, or the evenly spaced labels
    under the poses when components merge. Writes art/sheets/cells.json."""
    import numpy as np
    from PIL import Image
    from rembg import new_session, remove
    from scipy import ndimage
    sess = new_session("isnet-general-use")
    out = {}
    for sheet, bands in BANDS.items():
        im = Image.open(SHEETS / f"{sheet}.webp").convert("RGB")
        arr = np.asarray(im).astype(int)
        grey, sat = arr.mean(axis=2), arr.max(axis=2) - arr.min(axis=2)
        out[sheet] = {}
        for band, names in (("turn", TURN), ("act", ACT[sheet]), ("body", BODY[sheet])):
            y0, y1 = bands[band]
            a = np.asarray(remove(im.crop((0, y0, im.width, y1)), session=sess))[:, :, 3] > 128
            lab, n = ndimage.label(ndimage.binary_opening(a, iterations=2))
            comps = sorted((sl[1].start, sl[1].stop) for i, sl in enumerate(ndimage.find_objects(lab))
                           if (lab[sl] == i + 1).sum() >= 600)
            search = (max(0, y0 - 45), y0 + 5) if band == "turn" else (y1 - 30, y1 + 22)
            labels, label_y = _label_centres(grey, sat, len(names), search)
            widths = [b - a_ for a_, b in comps]
            comp_ok = len(comps) == len(names) and (band == "turn" or max(widths) < 1.8 * np.median(widths))
            lab_ok = len(labels) == len(names) and min(np.diff(labels)) >= 55 and max(np.diff(labels)) <= 330
            if (band == "turn" and lab_ok) or not comp_ok:
                if not lab_ok:
                    raise SystemExit(f"{sheet}/{band}: cells not found - edit art/sheets/cells.json by hand")
                centres = labels
            else:
                centres = [(a_ + b) / 2 for a_, b in comps]
            bounds = [0] + [int((centres[i] + centres[i + 1]) / 2) for i in range(len(centres) - 1)] + [im.width]
            bottom = y1 if band == "turn" or label_y < y1 - 22 else min(y1, label_y - 1)
            out[sheet][band] = {"cells": {nm: [bounds[i], y0, bounds[i + 1], bottom] for i, nm in enumerate(names)}}
            print(sheet, band, len(centres))
    (SHEETS / "cells.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def _label_centres(grey, sat, expected: int, search):
    import numpy as np
    best = None
    for y in range(search[0], search[1] - 12):
        strip = (grey[y:y + 12] > 165) & (sat[y:y + 12] < 40)
        xs = np.where(strip.any(axis=0))[0]
        if len(xs) == 0:
            continue
        cl, s, p = [], xs[0], xs[0]
        for x in xs[1:]:
            if x - p > 14:
                cl.append((s, p))
                s = x
            p = x
        cl.append((s, p))
        cl = [c for c in cl if c[1] - c[0] >= 12]
        key = (-abs(len(cl) - expected), int(strip.sum()))
        if best is None or key > best[0]:
            best = (key, y, cl)
    return [int((a + b) // 2) for a, b in best[2]], best[1]


# ============================================================ extract
def extract(sheets) -> None:
    import numpy as np
    from PIL import Image
    from rembg import new_session, remove
    from scipy import ndimage
    s1, s2 = new_session("isnet-general-use"), new_session("u2net")
    cells = json.loads((SHEETS / "cells.json").read_text(encoding="utf-8"))
    CELLS.mkdir(parents=True, exist_ok=True)
    for sheet in sheets:
        im = Image.open(SHEETS / f"{sheet}.webp").convert("RGB")
        for band in ("turn", "act", "body"):
            for name, (x0, y0, x1, y1) in cells[sheet][band]["cells"].items():
                if band == "turn" and name in ("top", "bottom"):
                    continue
                crop = im.crop((x0, max(0, y0 - 4), x1, y1))
                big = crop.resize((crop.width * 3, crop.height * 3), Image.LANCZOS)
                a1 = np.asarray(remove(big, session=s1, alpha_matting=True, alpha_matting_foreground_threshold=235,
                                       alpha_matting_background_threshold=20, alpha_matting_erode_size=6))[:, :, 3]
                m2 = np.asarray(remove(big, session=s2, only_mask=True)).astype(float)
                core = ndimage.gaussian_filter(ndimage.binary_erosion(m2 > 160, iterations=5).astype(float), 1.5)
                for suffix, a in (("", a1.astype(float)), ("_u2", np.maximum(a1.astype(float), core * 255))):
                    Image.fromarray(np.dstack([np.asarray(big), _main_parts(a)]).astype(np.uint8), "RGBA").save(
                        CELLS / f"{sheet}_{band}_{name}{suffix}.png")
            print(sheet, band, flush=True)
        _portrait_cell(sheet, im)


def _main_parts(a):
    """Keep the character (big components), drop stray bits (sparkles, neighbours' tails)."""
    import numpy as np
    from scipy import ndimage
    m = a > 128
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 0.08 * sizes.max()])
        a = np.where(ndimage.binary_dilation(keep, iterations=6), a, 0)
    return a.clip(0, 255)


def _portrait_cell(sheet: str, im) -> None:
    """The neutral face (first expression) as a square crop for icons."""
    y0, y1 = BANDS[sheet]["expr"]
    w = min(118, int((y1 - y0) * 0.95))
    crop = im.crop((4, y0 + 2, 4 + w, y0 + 2 + w))
    crop.resize((w * 2, w * 2)).save(CELLS / f"{sheet}_portrait.png")


# ============================================================ pack
def _clean(img, bg_rgb, shadows: bool = False):
    """RGBA cell -> cleaned RGBA: label-bar rows and floor shadows removed, interior holes filled,
    the dark studio background un-mixed from the semi-transparent edge pixels."""
    import numpy as np
    from scipy import ndimage
    a = img[:, :, 3].astype(float) / 255.0
    rgb = img[:, :, :3].astype(float)
    h, w = a.shape
    val = rgb.max(axis=2)
    sat = val - rgb.min(axis=2)
    bv = float(max(bg_rgb))
    greyish = (sat < 16) & (val > 18) & (val < max(110.0, 1.15 * bv))
    for y in range(h - 1, int(h * 0.8), -1):                  # label bar left at the bottom of the cell
        if ((a[y] > 0.5) & greyish[y]).mean() > 0.55:
            a[y:] = 0
    if shadows:
        # floor shadow / reflection: grey pixels under a foot / shoe, or a thin grey sliver on the
        # floor (legs and clothes continue upward into the body, a shadow does not)
        m = a > 0.3
        cand = m & (sat < 20) & (val < 1.2 * bv + 12)
        y_lim, y_foot = int(h * 0.85), int(h * 0.75)
        for x in range(w):
            col = m[:, x]
            if not col[y_lim:].any():
                continue
            solid_ng = np.where(col & ~cand[:, x])[0]
            low_ng = int(solid_ng.max()) if len(solid_ng) else -1
            if low_ng >= y_foot:                                  # a foot / shoe: grey below it is shadow
                ys = np.arange(low_ng + 1, h)
                a[ys, x] = np.where(cand[ys, x], 0.0, a[ys, x])
                continue
            y = h - 1
            while y >= y_lim:
                if not col[y]:
                    y -= 1
                    continue
                y1 = y
                while y >= 0 and col[y]:
                    y -= 1
                if y1 - y < 0.16 * h:
                    seg = slice(y + 1, y1 + 1)
                    a[seg, x] = np.where(cand[seg, x], 0.0, a[seg, x])
    solid = ndimage.binary_fill_holes(a > 0.4)
    solid = ndimage.binary_erosion(solid, iterations=2)
    a = np.maximum(a, solid.astype(float))
    edge = (a > 0.04) & (a < 0.97)
    B = np.asarray(bg_rgb, dtype=float)[None, :]
    ae = a[edge][:, None]
    rgb[edge] = np.clip((rgb[edge] - (1 - ae) * B) / np.maximum(ae, 0.25), 0, 255)
    a[a < 0.04] = 0
    return np.dstack([rgb, a * 255]).astype(np.uint8)


def _bbox(a, thr=20):
    import numpy as np
    ys, xs = np.where(a > thr)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pack(sheets) -> None:
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    cells = json.loads((SHEETS / "cells.json").read_text(encoding="utf-8"))
    for sheet in sheets:
        cid, kind = CHARACTERS[sheet]
        sheet_img = np.asarray(Image.open(SHEETS / f"{sheet}.webp").convert("RGB"))
        poses = {}
        for band, names in (("turn", [t for t in TURN if t not in ("top", "bottom")]), ("act", ACT[sheet]),
                            ("body", BODY[sheet])):
            for name in names:
                p = CELLS / f"{sheet}_{band}_{name}.png"
                if not p.is_file():
                    continue
                p = _pick_variant(p)
                x0, y0, x1, y1 = cells[sheet][band]["cells"][name]
                cell = sheet_img[max(0, y0 - 4):y1, x0:x1]
                border = np.concatenate([cell[:6].reshape(-1, 3), cell[:, :4].reshape(-1, 3),
                                         cell[:, -4:].reshape(-1, 3)])
                img = _clean(np.asarray(Image.open(p).convert("RGBA")), np.median(border, axis=0))
                bx0, by0, bx1, by1 = _bbox(img[:, :, 3])
                cov = lambda y: float((img[y, bx0:bx1, 3] > 128).mean())      # noqa: E731
                cut = band != "turn" and cov(by1 - 1) > 0.28 and cov(by1 - 1) > 0.8 * cov(max(by0, by1 - 7))
                if cut:                                         # the row label covers the feet: soften the edge
                    k = max(4, int((by1 - by0) * 0.07))
                    ramp = np.linspace(1.0, 0.0, k)[:, None]
                    img = img.copy()
                    img[by1 - k:by1, :, 3] = (img[by1 - k:by1, :, 3] * ramp).astype(np.uint8)
                key = name if band != "turn" else name
                if band == "turn" and name in ("front", "back"):
                    key = name
                poses[key] = {"img": img[by0:by1, bx0:bx1], "band": band, "cut": bool(cut)}
        # one scale for every row: the turnaround row is drawn bigger than the action rows
        hgt = {k: v["img"].shape[0] for k, v in poses.items()}
        r_turn = (np.mean([hgt["idle"] / hgt["front"], hgt["walk"] / max(hgt["left"], hgt["right"])])
                  if kind == "quadruped" else hgt["idle"] / hgt["front"])
        r_body = 1.0
        if kind == "biped" and "confident" in hgt:
            r_body = float(np.clip(hgt["idle"] / hgt["confident"], 0.8, 1.25))
        scale = {"turn": r_turn, "act": 1.0, "body": r_body}
        stand = hgt["idle"]
        # side views look the way the walk pose does (decided by silhouette overlap)
        facing = dict(FACING.get(sheet, {}))
        facing.update(_side_facing(poses, facing.get("walk", 1)))
        folder = OUT / cid / "illustrated"
        (folder / "frames").mkdir(parents=True, exist_ok=True)
        for old in (folder / "frames").glob("*.webp"):
            old.unlink()
        frames = {}
        for name, d in poses.items():
            img = d["img"]
            rel_h = img.shape[0] * scale[d["band"]] / stand
            out_h = max(24, int(round(STAND_PX[kind] * rel_h)))
            out_w = max(8, int(round(img.shape[1] * out_h / img.shape[0])))
            im = Image.fromarray(img, "RGBA").resize((out_w, out_h), Image.LANCZOS)
            im.save(folder / "frames" / f"{name}.webp", "WEBP", quality=90, method=6, alpha_quality=100)
            a = np.asarray(im)[:, :, 3]
            f = int(facing.get(name, 0))
            frames[name] = {"file": f"frames/{name}.webp", "facing": f, "h": round(rel_h, 4),
                            "w": round(rel_h * out_w / out_h, 4), "cx": round(_ground_x(a), 4),
                            "head": [round(v, 4) for v in _head(a, f, kind)]}
            if d.get("cut"):
                frames[name]["cut"] = True
        cycles = {"walk": [n for n in ("walk", _side_name(facing, facing.get("walk", 1))) if n in frames],
                  "run": [n for n in ("run", "sprint") if n in frames]}
        meta = {"version": 1, "id": cid, "kind": kind, "stand_height": STAND_HEIGHT[kind],
                "source": f"art/sheets/{sheet}.webp",
                "license": "Character art provided by the FocusBuddy owner; cut out and packed by "
                           "tools/build_illustrated_packs.py.",
                "frames": frames, "cycles": cycles}
        (folder / "pack.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        pp = CELLS / f"{sheet}_portrait.png"
        if pp.is_file():
            Image.open(pp).convert("RGB").resize((192, 192), Image.LANCZOS).save(folder / "portrait.webp", "WEBP",
                                                                                 quality=90)
        print(cid, len(frames), "frames", flush=True)


def _pick_variant(p: Path) -> Path:
    """ISNet alone (clean edges, no floor shadow) unless it lost part of the body on the dark
    background - then the ISNet + U2-Net ensemble."""
    import numpy as np
    from PIL import Image
    alt = p.with_name(p.stem + "_u2.png")
    if not alt.is_file():
        return p
    a1 = np.asarray(Image.open(p))[:, :, 3] > 128
    a2 = np.asarray(Image.open(alt))[:, :, 3] > 128
    return alt if a1.sum() < 0.88 * a2.sum() else p


def _side_facing(poses, walk_facing: int) -> dict:
    import numpy as np
    from PIL import Image
    if "walk" not in poses:
        return {"left": -1, "right": 1}

    def sil(img, flip=False):
        im = Image.fromarray(img, "RGBA").resize((96, 64))
        a = np.asarray(im)[:, :, 3] > 100
        return a[:, ::-1] if flip else a
    w = sil(poses["walk"]["img"])
    out = {}
    for side in ("left", "right"):
        if side not in poses:
            continue
        s = sil(poses[side]["img"])
        iou = lambda x, y: (x & y).sum() / max(1, (x | y).sum())      # noqa: E731
        same = iou(w, s) >= iou(w, sil(poses[side]["img"], True))
        out[side] = walk_facing if same else -walk_facing
    if out.get("left") == out.get("right"):                            # ambiguous: trust the names
        out = {"left": -1, "right": 1}
    return out


def _side_name(facing: dict, walk_facing: int) -> str:
    for side in ("right", "left"):
        if facing.get(side) == walk_facing:
            return side
    return "idle"


def _ground_x(a) -> float:
    import numpy as np
    h, w = a.shape
    rows = a[int(h * 0.92):]
    xs = np.where(rows > 100)[1]
    return float(xs.mean() / w) if len(xs) else 0.5


def _head(a, facing: int, kind: str):
    import numpy as np
    h, w = a.shape
    top = a[: max(2, int(h * (0.4 if kind == "quadruped" else 0.25)))] > 100
    ys, xs = np.where(top)
    if len(xs) == 0:
        return 0.5, 0.15
    if facing > 0:
        hx = float(np.percentile(xs, 85))
    elif facing < 0:
        hx = float(np.percentile(xs, 15))
    else:
        hx = float(xs.mean())
    col = a[:, max(0, int(hx) - 3):int(hx) + 4] > 100
    ys2 = np.where(col.any(axis=1))[0]
    head_top = int(ys2.min()) if len(ys2) else 0
    return hx / w, min(0.6, (head_top + 0.12 * h) / h)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["detect", "extract", "pack", "all"])
    ap.add_argument("sheets", nargs="*", default=list(CHARACTERS))
    a = ap.parse_args(argv)
    if a.stage in ("detect", "all"):
        detect()
    if a.stage in ("extract", "all"):
        extract(a.sheets)
    if a.stage in ("pack", "all"):
        pack(a.sheets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
