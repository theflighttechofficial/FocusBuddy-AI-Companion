"""Generic secondary-motion system: spring-damper "jiggle bones".

Everything loose on a character - tail, both ears, the head, hair, a hoodie
hem, the robot's antenna, the dog's collar tag - is a ``SpringBone``: an
angle that chases a target (its rest pose plus what the animation asks for)
with stiffness and damping, and is kicked by how the *body* moves:

    linear acceleration   body speeds up -> ears / tail swing back
    vertical velocity     hops and landings -> ears / antenna bounce
    turn rate             body turns right -> tail lags, swings, settles
    tilt rate             nodding / lunging -> hair and head lag

One ``SecondarySystem.update()`` per frame drives all bones (cheap: a few
floats each, fixed 1/240 s sub-steps so it is frame-rate independent).
Amount / stiffness are scaled by the "Secondary motion" and "Physics
strength" settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SpringBone:
    name: str
    freq: float = 2.5          # Hz (stiffness)
    zeta: float = 0.35         # damping ratio
    limit: float = 60.0        # +/- clamp (deg)
    k_accel: float = 0.0       # response to forward acceleration (deg per normalised accel)
    k_vy: float = 0.0          # response to vertical velocity (impulse per unit)
    k_turn: float = 0.0        # response to turn rate (impulse per facing change / s)
    k_tilt: float = 0.0        # response to body tilt rate
    k_speed: float = 0.0       # steady offset per normalised speed (streaming back while running)
    x: float = 0.0
    v: float = 0.0

    def step(self, target: float, dt: float, stiff: float = 1.0) -> float:
        w = 2 * math.pi * self.freq * stiff
        n = max(1, int(dt / (1 / 240)) + 1)
        h = dt / n
        for _ in range(n):
            a = w * w * (target - self.x) - 2 * self.zeta * w * self.v
            self.v += a * h
            self.x += self.v * h
        if self.x > self.limit or self.x < -self.limit:      # constraint: soft stop, kill velocity
            self.x = max(-self.limit, min(self.limit, self.x))
            self.v *= -0.2
        if not (self.x == self.x):                          # NaN guard
            self.x, self.v = 0.0, 0.0
        return self.x


# per-species bone sets: (name, freq, zeta, limit, k_accel, k_vy, k_turn, k_tilt, k_speed)
_BONES = {
    "quad": [("tail", 1.9, 0.38, 70, -10, 0.0, 26, 0.0, -22), ("ear_l", 2.6, 0.32, 45, -6, 0.022, 9, 0.4, -24),
             ("ear_r", 2.4, 0.30, 45, -6, 0.025, -9, 0.4, -24), ("head", 3.0, 0.55, 8, -3, 0.004, 1.5, 0.08, 0.0),
             ("tag", 1.7, 0.18, 40, -12, 0.03, 10, 0.5, 0.0)],
    "robot": [("antenna", 3.4, 0.22, 35, -14, 0.02, 14, 0.6, -10), ("head", 4.5, 0.7, 5, -2, 0.002, 1.0, 0.05, 0.0),
              ("ear_l", 5.0, 0.5, 10, -2, 0.004, 2, 0.1, 0), ("ear_r", 5.0, 0.5, 10, -2, 0.004, -2, 0.1, 0)],
    "buddy": [("hair", 2.2, 0.45, 30, -8, 0.01, 12, 0.5, -9), ("hem", 2.0, 0.4, 20, -6, 0.008, 8, 0.2, -6),
              ("head", 3.0, 0.55, 8, -3, 0.004, 1.5, 0.08, 0.0)],
}


class SecondarySystem:
    def __init__(self, species: str = "cat"):
        self.set_species(species)
        self.amount = 1.0      # "Secondary motion" setting
        self.stiff = 1.0       # "Physics strength" (sqrt-scaled)
        self._prev_facing = None
        self._prev_tilt = None

    def set_species(self, species: str) -> None:
        kind = "robot" if species == "robot" else "buddy" if species == "buddy" else "quad"
        self.kind = kind
        self.bones: dict[str, SpringBone] = {}
        for (n, f, z, lim, ka, kvy, kt, ktl, ks) in _BONES[kind]:
            self.bones[n] = SpringBone(n, f, z, lim, ka, kvy, kt, ktl, ks)

    def impulse(self, name: str, dv: float) -> None:
        b = self.bones.get(name)
        if b is not None:
            b.v += dv

    def kick_all(self, dv: float) -> None:
        for b in self.bones.values():
            b.v += dv * (1 if b.name != "head" else 0.2)

    def update(self, dt: float, accel_norm: float, vy: float, facing: float, tilt: float, speed_norm: float,
               targets: dict | None = None) -> dict:
        """Returns {bone: angle (deg)}. ``targets`` adds animation-driven rest offsets per bone."""
        dt = max(1e-4, dt)
        turn_rate = 0.0 if self._prev_facing is None else (facing - self._prev_facing) / dt
        tilt_rate = 0.0 if self._prev_tilt is None else (tilt - self._prev_tilt) / dt
        self._prev_facing, self._prev_tilt = facing, tilt
        turn_rate = max(-12.0, min(12.0, turn_rate))
        tilt_rate = max(-400.0, min(400.0, tilt_rate))
        vy = max(-600.0, min(600.0, vy))
        out = {}
        for b in self.bones.values():
            b.v += (b.k_vy * -vy + b.k_turn * turn_rate * 0.05 + b.k_tilt * tilt_rate * 0.01) * dt * 60 * \
                (0.6 + 0.4 * self.stiff)
            tgt = b.k_accel * accel_norm + b.k_speed * speed_norm + (targets or {}).get(b.name, 0.0)
            out[b.name] = b.step(tgt, dt, self.stiff ** 0.5) * self.amount
        return out
