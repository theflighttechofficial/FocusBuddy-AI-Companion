"""Motion curves: different movements deserve different timing.

    linear        mechanical (robot servos)
    ease_in       wind-ups, things starting to fall
    ease_out      arrivals, things coming to rest
    smoothstep    general blending (ease in + out)
    smootherstep  long, very soft blends (sit down, lie down)
    ease_out_back overshoot then settle (pops, happy reactions)
    spring        damped oscillation toward 1 (landings, recoils)
    anticipate    small move the other way first (wind-up before a lunge)

Every function maps t in [0, 1] to a progress value; ``CURVES`` lets actions
pick one by name.
"""

from __future__ import annotations

import math


def _c(t: float) -> float:
    return 0.0 if t <= 0.0 else 1.0 if t >= 1.0 else t


def linear(t: float) -> float:
    return _c(t)


def ease_in(t: float) -> float:
    t = _c(t)
    return t * t * t


def ease_out(t: float) -> float:
    t = 1.0 - _c(t)
    return 1.0 - t * t * t


def smoothstep(t: float) -> float:
    t = _c(t)
    return t * t * (3 - 2 * t)


def smootherstep(t: float) -> float:
    t = _c(t)
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_out_back(t: float, k: float = 1.70158) -> float:
    t = _c(t) - 1.0
    return 1 + (k + 1) * t * t * t + k * t * t


def spring(t: float, freq: float = 1.5, zeta: float = 0.35) -> float:
    """Damped spring from 0 settling at 1 (overshoots once or twice)."""
    t = _c(t)
    w = 2 * math.pi * freq
    return 1.0 - math.exp(-zeta * w * t) * math.cos(w * math.sqrt(max(1e-6, 1 - zeta * zeta)) * t) * (1 - t)


def anticipate(t: float, amount: float = 0.18) -> float:
    """Pull back a little, then go: dips below 0 before rising to 1."""
    t = _c(t)
    return t * t * ((amount * 10 + 1) * t - amount * 10)


CURVES = {"linear": linear, "ease_in": ease_in, "ease_out": ease_out, "smoothstep": smoothstep,
          "smootherstep": smootherstep, "ease_out_back": ease_out_back, "spring": spring, "anticipate": anticipate}


def curve(name: str):
    return CURVES.get(name, smoothstep)
