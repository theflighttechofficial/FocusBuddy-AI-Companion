"""Stylised PBR shading shared by the CPU rasteriser and (mirrored in GLSL) the GPU path.

The look is a "polished 3D toy": soft wrapped key light with a gentle toon step, a cool
fill, sky / ground ambient (times ambient occlusion), a rim light, per-material response
(fur sheen, cloth, skin warmth, metal reflections, emissive) and - for glTF PBR materials -
a GGX highlight driven by metallic / roughness, deliberately capped so characters never look
wet or glossy.

Inputs per sample (all NumPy, camera space):

    n        (K, 3)  normal (after normal mapping)
    base     (K, 3)  base colour = vertex colour * baseColorFactor * baseColorTexture
    row      (K, 9)  stylised params  wrap toon spec power rim sheen sss metal emissive
    metal    (K,)    metallic  (factor * texture.B)
    rough    (K,)    roughness (factor * texture.G)
    ao       (K,)    ambient occlusion (1 = open)
    emis     (K, 3)  emissive colour (factor * texture)
    pbr      (K,)    1 = metallic / roughness drive the highlight, 0 = stylised table highlight
"""

from __future__ import annotations

import numpy as np

from .render3d import HALF, LIGHT

# per shading type: wrap, toon, spec, power, rim, sheen, sss, metal, emissive
# 2.2 "studio" look: much less toon banding (soft, rounded forms), a warmer, deeper skin scatter,
# a backlit glow on fur edges, and soft broad highlights instead of small hard ones
MATERIALS = {
    "fur": (0.60, 0.18, 0.06, 10.0, 0.30, 0.28, 0.08, 0.0, 0.0),
    "cloth": (0.48, 0.16, 0.03, 8.0, 0.14, 0.14, 0.00, 0.0, 0.0),
    "skin": (0.52, 0.10, 0.10, 22.0, 0.14, 0.04, 0.42, 0.0, 0.0),
    "plastic": (0.28, 0.30, 0.42, 52.0, 0.18, 0.00, 0.00, 0.0, 0.0),
    "metal": (0.20, 0.30, 0.70, 30.0, 0.24, 0.00, 0.00, 1.0, 0.0),
    "eye": (0.15, 0.40, 0.85, 90.0, 0.10, 0.00, 0.00, 0.0, 0.0),
    "emissive": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
}
# light rig balance (shared with the GLSL shader): a stronger key and a little less flat ambient
# give the forms depth; the rim is a warm back light
AMB_K, KEY_K, FILL_K = 0.44, 0.70, 0.18
RIM_COL = np.array([1.0, 0.95, 0.88])
GRADE_CONTRAST, GRADE_SAT = 0.10, 1.07


def grade(col: np.ndarray) -> np.ndarray:
    """Gentle filmic grade (mirrored in GLSL): a soft S-curve and a touch more colour."""
    c = np.clip(col, 0.0, 1.0)
    c = c + GRADE_CONTRAST * c * (1.0 - c) * (2.0 * c - 1.0) * 2.0
    lum = (c * np.array([0.299, 0.587, 0.114])).sum(axis=-1, keepdims=True)
    return np.clip(lum + (c - lum) * GRADE_SAT, 0.0, 1.0)
FILL = np.array([0.62, 0.18, 0.55])
FILL = FILL / np.linalg.norm(FILL)
SKY = np.array([1.0, 1.0, 1.04])
GROUND = np.array([0.72, 0.66, 0.66])
KEY_COL = np.array([1.0, 0.97, 0.92])
FILL_COL = np.array([0.80, 0.86, 1.0])
SPEC_CAP = 0.55            # highest highlight a PBR material may add (keeps the toy look matte-ish)
VIEW = np.array([0.0, 0.0, 1.0])


def material_table(shading) -> np.ndarray:
    """Stylised parameter rows for a list of shading names or MaterialDesc objects."""
    names = [getattr(s, "shading", s) for s in (shading or ["plastic"])]
    return np.array([MATERIALS.get(s or "plastic", MATERIALS["plastic"]) for s in names], dtype=np.float32)


def material_inputs(materials) -> dict:
    """Per-material PBR inputs as arrays (index = material index)."""
    mats = list(materials or [])
    if not mats:
        return {"metal": np.zeros(1), "rough": np.full(1, 0.8), "pbr": np.zeros(1), "emis": np.zeros((1, 3)),
                "unlit": np.zeros(1)}
    return {"metal": np.array([float(getattr(m, "metallic", 0.0)) for m in mats]),
            "rough": np.array([float(getattr(m, "roughness", 0.8)) for m in mats]),
            "pbr": np.array([1.0 if getattr(m, "pbr", False) else 0.0 for m in mats]),
            # glTF emissive adds light for PBR materials; stylised "emissive" materials are unlit instead
            "emis": np.array([list(getattr(m, "emissive", (0, 0, 0)))[:3] if getattr(m, "pbr", False) else
                              [0.0, 0.0, 0.0] for m in mats], dtype=float),
            "unlit": np.array([1.0 if getattr(m, "unlit", False) else 0.0 for m in mats])}


def sample(tex: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear RGBA sample (0..1) of an (H, W, 4) uint8 texture at (K, 2) UVs, REPEAT wrapping.

    Float32 maths and 1-D ``take`` gathers on the flattened texture: about 3x faster than 2-D fancy
    indexing in float64, the same result to within 1e-4."""
    h, w = tex.shape[:2]
    uvf = uv.astype(np.float32, copy=False)
    u = np.mod(uvf[:, 0], 1.0) * w - 0.5
    v = np.mod(uvf[:, 1], 1.0) * h - 0.5
    x0 = np.floor(u)
    y0 = np.floor(v)
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    x0 = x0.astype(np.int64) % w
    y0 = y0.astype(np.int64) % h
    x1 = x0 + 1
    x1[x1 == w] = 0
    y1 = y0 + 1
    y1[y1 == h] = 0
    flat = tex.reshape(-1, tex.shape[2])
    r0 = y0 * w
    r1 = y1 * w
    # gather the four texels first, then convert (converting the whole texture each frame costs far
    # more than the sampling itself)
    f = (lambda q: q.astype(np.float32)) if tex.dtype != np.float32 else (lambda q: q)
    a = f(flat.take(r0 + x0, axis=0)) * (1 - fx) + f(flat.take(r0 + x1, axis=0)) * fx
    b = f(flat.take(r1 + x0, axis=0)) * (1 - fx) + f(flat.take(r1 + x1, axis=0)) * fx
    out = a * (1 - fy) + b * fy
    return out * np.float32(1.0 / 255.0) if tex.dtype == np.uint8 else out


def shade(n: np.ndarray, base: np.ndarray, params: np.ndarray, lights: dict | None = None, metal=None,
          rough=None, ao=None, emis=None, pbr=None) -> np.ndarray:
    lights = lights or {}
    key_on = 1.0 if lights.get("key", True) else 0.0
    fill_on = 1.0 if lights.get("fill", True) else 0.0
    rim_on = 1.0 if lights.get("rim", True) else 0.0
    K = len(n)
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    wrap, toon, spec, power, rim, sheen, sss, tmetal, em = (params[:, i] for i in range(9))
    pbr = np.zeros(K) if pbr is None else pbr
    metal_v = tmetal if metal is None else np.where(pbr > 0.5, metal, tmetal)
    ao = np.ones(K) if ao is None else ao
    ndl = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]
    w = np.clip((ndl + wrap) / (1.0 + wrap), 0.0, 1.0)
    t = np.clip(w / 0.6, 0.0, 1.0)
    tn = t * t * (3 - 2 * t)
    diff = (w + (tn - w) * toon) * key_on
    hemi = 0.5 - 0.5 * ny                                     # y is down: up-facing -> 1
    amb = GROUND[None, :] + (SKY - GROUND)[None, :] * hemi[:, None]
    fdl = np.clip(nx * FILL[0] + ny * FILL[1] + nz * FILL[2], 0.0, 1.0) * fill_on
    light = amb * (AMB_K * ao)[:, None] + KEY_COL[None, :] * (diff * KEY_K * (0.55 + 0.45 * ao))[:, None] + \
        FILL_COL[None, :] * (fdl * FILL_K * ao)[:, None]
    col = base * light
    if np.any(metal_v > 0):
        env = (0.55 + 0.45 * hemi)[:, None] * np.array([0.92, 0.95, 1.0])[None, :]
        mcol = base * (0.30 + 0.55 * diff)[:, None] + env * 0.32 * base.mean(axis=1, keepdims=True)
        col = col + (mcol - col) * metal_v[:, None]
    col = col + base * ((1.0 - diff) * sss)[:, None] * np.array([0.30, 0.08, 0.04])[None, :]
    edge = np.clip(1.0 - nz, 0.0, 1.0)
    col = col + base * (sheen * edge * edge)[:, None] * 0.6
    hx, hy, hz = HALF
    ndh = np.clip(nx * hx + ny * hy + nz * hz, 0.0, 1.0)
    sp_table = spec * np.power(ndh, power) * (w > 0.25) * key_on
    if np.any(pbr > 0.5):
        r = np.clip(rough if rough is not None else np.full(K, 0.8), 0.08, 1.0)
        a2 = (r * r) ** 2
        d = ndh * ndh * (a2 - 1.0) + 1.0
        D = a2 / (np.pi * d * d)
        vdh = max(1e-3, float(np.dot(HALF, VIEW)))
        fres = (1.0 - vdh) ** 5
        F0 = 0.04 + (base.mean(axis=1) - 0.04) * metal_v
        F = F0 + (1.0 - F0) * fres
        sp_pbr = np.minimum(D * F * 0.25, SPEC_CAP) * np.clip(ndl, 0.0, 1.0) * key_on * (0.6 + 0.4 * ao)
        sp = np.where(pbr > 0.5, sp_pbr, sp_table)
    else:
        sp = sp_table
    rimv = rim * edge ** 3 * rim_on * (0.6 + 0.4 * hemi)
    # the rim takes a little of the surface colour (fur glows warm, not white)
    col = col + sp[:, None] + rimv[:, None] * (RIM_COL[None, :] * 0.6 + base * 0.4)
    col = col + (base - col) * em[:, None]                     # stylised emissive: unlit
    if emis is not None:
        col = col + emis
    return grade(col)


def shade_one(n, base, params) -> tuple:
    """``shade`` for a single normal with all lights on and table (non-PBR) parameters, in plain
    floats - used for decal tints several times per frame, where NumPy's call overhead dominates."""
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    wrap, toon, spec, power, rim, sheen, sss, tmetal, em = (float(v) for v in params)
    ndl = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]
    w = min(1.0, max(0.0, (ndl + wrap) / (1.0 + wrap)))
    t = min(1.0, max(0.0, w / 0.6))
    tn = t * t * (3 - 2 * t)
    diff = w + (tn - w) * toon
    hemi = 0.5 - 0.5 * ny
    fdl = min(1.0, max(0.0, nx * FILL[0] + ny * FILL[1] + nz * FILL[2]))
    edge = min(1.0, max(0.0, 1.0 - nz))
    ndh = min(1.0, max(0.0, nx * HALF[0] + ny * HALF[1] + nz * HALF[2]))
    sp = spec * ndh ** power * (1.0 if w > 0.25 else 0.0)
    rimv = rim * edge ** 3 * (0.6 + 0.4 * hemi)
    bm = (base[0] + base[1] + base[2]) / 3.0
    out = []
    for k in range(3):
        b = float(base[k])
        amb = GROUND[k] + (SKY[k] - GROUND[k]) * hemi
        light = amb * AMB_K + KEY_COL[k] * diff * KEY_K + FILL_COL[k] * fdl * FILL_K
        c = b * light
        if tmetal > 0:
            env = (0.55 + 0.45 * hemi) * (0.92, 0.95, 1.0)[k]
            mc = b * (0.30 + 0.55 * diff) + env * 0.32 * bm
            c = c + (mc - c) * tmetal
        c += b * (1.0 - diff) * sss * (0.30, 0.08, 0.04)[k]
        c += b * sheen * edge * edge * 0.6
        c += sp + rimv * (RIM_COL[k] * 0.6 + b * 0.4)
        c += (b - c) * em
        out.append(min(1.0, max(0.0, c)))
    return tuple(float(v) for v in grade(np.array(out)))


def surface(mats, mi: np.ndarray, uv: np.ndarray, n: np.ndarray, tan: np.ndarray, col: np.ndarray, inputs: dict):
    """Per-sample texture lookups grouped by material.

    Returns (base (K,3), n (K,3), metal (K,), rough (K,), ao (K,), emis (K,3), alpha (K,))."""
    K = len(mi)
    base = col.copy()
    metal = inputs["metal"][mi].astype(float)
    rough = inputs["rough"][mi].astype(float)
    ao = np.ones(K)
    emis = inputs["emis"][mi].astype(float)
    alpha = np.ones(K)
    n = n.copy()
    for m_index in np.unique(mi):
        if m_index < 0 or m_index >= len(mats):
            continue
        md = mats[int(m_index)]
        tx = getattr(md, "textures", None) or {}
        if not tx:
            continue
        sel = np.flatnonzero(mi == m_index)
        u = uv[sel]
        if "baseColor" in tx:
            s_ = sample(tx["baseColor"], u)
            base[sel] = base[sel] * s_[:, :3]
            alpha[sel] = s_[:, 3] * float(md.base_color[3]) if len(md.base_color) > 3 else s_[:, 3]
        if "metallicRoughness" in tx:
            s_ = sample(tx["metallicRoughness"], u)
            rough[sel] = rough[sel] * s_[:, 1]
            metal[sel] = metal[sel] * s_[:, 2]
        if "occlusion" in tx:
            s_ = sample(tx["occlusion"], u)
            ao[sel] = 1.0 + (s_[:, 0] - 1.0) * float(md.occlusion_strength)
        if "emissive" in tx:
            s_ = sample(tx["emissive"], u)
            emis[sel] = emis[sel] * s_[:, :3]
        if "normal" in tx and tan is not None:
            s_ = sample(tx["normal"], u)
            tn = s_[:, :3] * 2.0 - 1.0
            tn[:, :2] *= float(md.normal_scale)
            N = n[sel]
            T = tan[sel, :3]
            T = T - N * (N * T).sum(axis=1, keepdims=True)
            T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
            B = np.cross(N, T) * tan[sel, 3:4]
            nn = T * tn[:, 0:1] + B * tn[:, 1:2] + N * tn[:, 2:3]
            ok = np.linalg.norm(T, axis=1) > 1e-6
            n[sel] = np.where(ok[:, None], nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9), N)
    return base, n, metal, rough, ao, emis, alpha
