"""Painted faces: dirty-tile tracking and cached redraws (companion/face_paint.py)."""

from pathlib import Path

import numpy as np
import pytest

from focusbuddy.companion.face_paint import PaintedFace, tile_spans

ROOT = Path(__file__).resolve().parent.parent
FACES = sorted((ROOT / "assets" / "characters").glob("*/face/painted.npz"))


def test_tile_spans_cover_every_texel():
    rng = np.random.default_rng(3)
    w, h = 512, 256
    idx = rng.choice(w * h, 2000, replace=False)
    spans = tile_spans(idx, w, tile=32)
    covered = np.zeros(w * h, bool)
    for x, y, rw, rh in spans:
        assert 0 <= x and 0 <= y and x + rw <= w and y + rh <= h
        for yy in range(y, y + rh):
            covered[yy * w + x: yy * w + x + rw] = True
    assert covered[idx].all()
    assert tile_spans(np.zeros(0, np.int64), w) == []


@pytest.mark.parametrize("path", FACES, ids=lambda p: p.parent.parent.name)
def test_blink_marks_only_face_tiles_and_caches(path):
    pf = PaintedFace(path)
    h, w = pf.size
    tex = np.full((h, w, 4), 128, np.uint8)
    pf.bind(tex)
    assert not pf.dirty
    assert pf.update(1.0, 0.0, 0.0)
    assert pf.dirty and pf.dirty <= set(pf.e_spans)
    closed = tex.reshape(-1, 4)[pf.e_idx, :3].copy()
    pf.dirty.clear()
    pf.update(0.0, 0.0, 0.0)
    pf.update(1.0, 0.0, 0.0)                     # the same lid value again comes from the cache
    assert np.array_equal(tex.reshape(-1, 4)[pf.e_idx, :3], closed)
    fresh = np.clip(pf.eyes(1.0, 0.0) + 0.5, 0, 255).astype(np.uint8)
    assert np.array_equal(closed, fresh)
    if len(pf.m_idx):
        pf.dirty.clear()
        pf.update(1.0, 0.0, 0.8)
        assert pf.dirty and pf.dirty <= set(pf.m_spans)
    # pixels outside the eye and mouth tiles are never touched
    mask = np.zeros(h * w, bool)
    mask[pf.e_idx] = True
    if len(pf.m_idx):
        mask[pf.m_idx] = True
    assert (tex.reshape(-1, 4)[~mask] == 128).all()
