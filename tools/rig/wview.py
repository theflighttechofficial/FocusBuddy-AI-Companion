"""Weight check views: every vertex coloured by its strongest bone (and a heat map of one bone)."""
import sys

import numpy as np
from PIL import Image, ImageDraw

from core import raster

PAL = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
                [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
                [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195], [128, 128, 0], [255, 215, 180],
                [0, 0, 128], [128, 128, 128], [40, 40, 40], [255, 120, 120], [120, 255, 120], [120, 120, 255],
                [200, 100, 0], [0, 200, 100], [100, 0, 200], [180, 180, 30], [30, 180, 180], [180, 30, 180]],
               float)


def views(V, F, W, names, out, size=520, bone=None):
    if bone is None:
        dom = np.argmax(W, axis=1)
        vc = PAL[dom % len(PAL)]
    else:
        w = W[:, names.index(bone)]
        vc = np.stack([255 * w, 60 + 0 * w, 255 * (1 - w)], 1)
    fc = vc[F].mean(1)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    ims = []
    for axes, da, ds, L in (((0, 2, 1.0), 1, -1.0, (0.2, -1, 0.4)), ((1, 2, -1.0), 0, 1.0, (1, 0.2, 0.4)),
                            ((1, 2, -1.0), 0, -1.0, (-1, 0.2, 0.4)), ((0, 1, 1.0), 2, 1.0, (0.2, 0.2, 1))):
        L = np.array(L, float) / np.linalg.norm(L)
        sh = 0.55 + 0.45 * np.abs(fn @ L)
        img, _ = raster(V, F, fc * sh[:, None], axes, size, depth_axis=da, depth_sign=ds)
        ims.append(img)
    pil = Image.fromarray(np.concatenate(ims, 1).astype(np.uint8))
    if bone is None:
        d = ImageDraw.Draw(pil)
        for i, n in enumerate(names):
            d.rectangle([4 + (i % 8) * 130, size - 60 + (i // 8) * 14, 14 + (i % 8) * 130, size - 50 + (i // 8) * 14],
                        fill=tuple(int(c) for c in PAL[i % len(PAL)]))
            d.text((18 + (i % 8) * 130, size - 62 + (i // 8) * 14), n, fill=(0, 0, 0))
    pil.save(out)
