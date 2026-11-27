"""Foot-phase system: real walk / jog / run / sprint cycles with planted feet.

Every foot runs its own cycle ``u = (cycle + offset) mod 1``:

    u in [0, duty)   STANCE   CONTACT -> PLANTED   the foot is locked to the ground
    u in [duty, 1)   SWING    LIFT -> SWING -> REACH the foot arcs forward to its next plant

``cycle`` comes from the locomotion controller and advances with *distance
travelled / stride length*, so the cycle is speed independent.  While a foot is
planted its position is stored in ground coordinates (odometer units) and
converted back to the body frame every frame - a planted foot therefore stays
exactly where it touched down while the body moves over it (no foot sliding).
Swing feet interpolate from where they lifted to their next landing spot.
When the body stops, feet that are not under their hips take a quick
settling step (one at a time), like a real animal shuffling into a stance.

Gait tables (offsets per foot, duty factor, swing height, body motion):

    quadruped  WALK  lateral sequence FL -> BR -> FR -> BL (4 beats, duty .66)
               JOG   trot: diagonal pairs FL+BR / FR+BL (2 beats, duty .5)
               RUN   rotary gallop: front pair, flight, hind pair (duty .34)
               SPRINT extended gallop, shorter contact (duty .26)
    rabbit     bound / hop: front pair lands, hind pair passes, long flight
    biped      L / R half a cycle apart; duty .62 walk -> .3 sprint (flight)
    robot      biped table with linear (mechanical) swing interpolation

Gait levels are continuous (0 walk .. 3 sprint); offsets interpolate along the
shortest way round the cycle and duty factors interpolate linearly, so speed
changes blend smoothly between gaits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

FOOT_EVENT = "FOOT_CONTACT"


@dataclass(frozen=True)
class GaitDef:
    name: str
    offsets: dict
    duty: float
    lift: float          # swing height (units)
    bob: float           # body bob amplitude (units)
    lean: float          # forward lean (deg)
    flex: float          # spine flex (gallop gather / extend)
    vault: float         # 1 = body highest mid-stance (walk), -1 = lowest mid-stance (running compression)


QUAD = (
    # offsets are phase leads: foot k touches down when the cycle reaches (1 - offset) -> FL, BR, FR, BL
    GaitDef("WALK", {"FL": 0.0, "BR": 0.75, "FR": 0.5, "BL": 0.25}, 0.66, 3.6, 1.2, 0.0, 0.0, 1.0),
    GaitDef("JOG", {"FL": 0.0, "BR": 0.0, "FR": 0.5, "BL": 0.5}, 0.5, 5.4, 2.0, 3.0, 0.2, -0.4),
    GaitDef("RUN", {"FL": 0.0, "FR": 0.1, "BL": 0.5, "BR": 0.6}, 0.34, 7.0, 3.0, 8.0, 1.0, -1.0),
    GaitDef("SPRINT", {"FL": 0.0, "FR": 0.08, "BL": 0.5, "BR": 0.58}, 0.26, 8.0, 3.6, 12.0, 1.25, -1.0),
)
# a rabbit's hop (a half-bound): the front paws land first (cycle 0), the hind feet swing up under the
# body and land beside / ahead of them while the front paws lift, the hind legs push off - and the
# rest of the cycle is flight (see ``hop_flight``: the body only rises while no paw is planted)
HOP = (
    GaitDef("HOP", {"FL": 0.0, "FR": 0.96, "BL": 0.78, "BR": 0.75}, 0.30, 4.0, 1.0, 2.0, 0.3, -1.0),
    GaitDef("HOP_JOG", {"FL": 0.0, "FR": 0.96, "BL": 0.8, "BR": 0.77}, 0.27, 5.0, 1.2, 3.0, 0.5, -1.0),
    GaitDef("BOUND", {"FL": 0.0, "FR": 0.95, "BL": 0.82, "BR": 0.79}, 0.23, 6.0, 1.4, 6.0, 0.9, -1.0),
    GaitDef("SPRINT", {"FL": 0.0, "FR": 0.95, "BL": 0.84, "BR": 0.81}, 0.2, 7.0, 1.6, 10.0, 1.1, -1.0),
)


def _hop_takeoff(g: "GaitDef") -> float:
    offs = [g.offsets.get(k, 0.0) for k in ("BL", "BR")]
    return min(0.92, max(((1.0 - o) % 1.0) + g.duty for o in offs))


def hop_flight(g: "GaitDef", cycle: float) -> float:
    """Flight position of a hop at ``cycle``: -1 while paws are on the ground, 0..1 through the
    airborne part (it starts when the last hind foot pushes off, ends when the front paws land)."""
    c = cycle % 1.0
    start = _hop_takeoff(g)
    if c < start:
        return -1.0
    return (c - start) / max(1e-3, 1.0 - start)


def hop_stance(g: "GaitDef", cycle: float) -> float:
    """Position through the ground part of a hop (0 front paws land .. 1 push-off), -1 in flight."""
    c = cycle % 1.0
    start = _hop_takeoff(g)
    return c / start if c < start else -1.0
BIPED = (
    GaitDef("WALK", {"L": 0.0, "R": 0.5}, 0.62, 3.8, 1.3, 0.0, 0.0, 1.0),
    GaitDef("JOG", {"L": 0.0, "R": 0.5}, 0.42, 6.0, 2.0, 4.0, 0.0, -0.6),
    GaitDef("RUN", {"L": 0.0, "R": 0.5}, 0.34, 8.0, 2.8, 8.0, 0.0, -1.0),
    GaitDef("SPRINT", {"L": 0.0, "R": 0.5}, 0.3, 9.5, 3.2, 12.0, 0.0, -1.0),
)
MECH = (
    GaitDef("WALK", {"L": 0.0, "R": 0.5}, 0.6, 3.4, 0.6, 0.0, 0.0, 1.0),
    GaitDef("JOG", {"L": 0.0, "R": 0.5}, 0.5, 4.6, 0.9, 2.0, 0.0, -0.5),
    GaitDef("RUN", {"L": 0.0, "R": 0.5}, 0.42, 5.6, 1.2, 4.0, 0.0, -1.0),
    GaitDef("SPRINT", {"L": 0.0, "R": 0.5}, 0.38, 6.2, 1.4, 6.0, 0.0, -1.0),
)
TABLES = {"trot": QUAD, "hop": HOP, "biped": BIPED, "mech": MECH}


def _circ_lerp(a: float, b: float, t: float) -> float:
    d = ((b - a + 0.5) % 1.0) - 0.5
    return (a + d * t) % 1.0


def blend_gait(table: tuple, level: float) -> GaitDef:
    level = max(0.0, min(3.0, level))
    i = min(2, int(level))
    t = level - i
    a, b = table[i], table[i + 1]
    if t <= 1e-6:
        return a
    offs = {k: _circ_lerp(a.offsets[k], b.offsets[k], t) for k in a.offsets}
    lerp = lambda x, y: x + (y - x) * t  # noqa: E731
    return GaitDef(a.name if t < 0.5 else b.name, offs, lerp(a.duty, b.duty), lerp(a.lift, b.lift),
                   lerp(a.bob, b.bob), lerp(a.lean, b.lean), lerp(a.flex, b.flex), lerp(a.vault, b.vault))


def _ease(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


@dataclass
class FootState:
    name: str
    planted: bool = True
    plant_world: float = 0.0     # ground position while planted (odometer units)
    local_x: float = 0.0         # body-frame position (units)
    y: float = 0.0               # height above the ankle rest height (units, negative = up)
    lift_local: float = 0.0
    lift_s0: float = 0.0
    phase: str = "PLANTED"
    settle: dict | None = None
    u: float = 0.0
    lift_u: float = 0.0
    wrapped: bool = True


@dataclass
class FootFrame:
    targets: dict = field(default_factory=dict)      # name -> (x, y) body-frame units (ground at y = -ankle)
    contacts: dict = field(default_factory=dict)     # name -> planted?
    phases: dict = field(default_factory=dict)       # name -> CONTACT / PLANTED / LIFT / SWING / REACH / SETTLE
    events: list = field(default_factory=list)       # [(FOOT_CONTACT, foot)]
    support: float = 0.0                             # 0..1 stance support (drives body bob)
    cycle_phase: str = "IDLE"                        # CONTACT / DOWN / PASS / UP ...
    gait: GaitDef | None = None


class FootPhaseSystem:
    SETTLE_DUR = 0.16
    SETTLE_GAP = 0.07

    def __init__(self, feet: tuple[str, ...], kind: str = "trot", mechanical: bool = False):
        self.kind = kind if kind in TABLES else "trot"
        self.table = TABLES[self.kind]
        self.mechanical = mechanical or self.kind == "mech"
        self.feet = {f: FootState(f) for f in feet}
        self._init = False
        self._facing = 1
        self._airborne = False
        self._settle_next = 0.0
        self._prev_cycle = 0.0
        self.gait: GaitDef = self.table[0]

    def reset(self, neutral: dict, odo: float) -> None:
        for f, st in self.feet.items():
            nx = neutral.get(f, (0.0, 0.0))[0]
            st.planted, st.local_x, st.y, st.settle = True, nx, 0.0, None
            st.plant_world = odo + nx
            st.phase = "PLANTED"
        self._init = True

    # --------------------------------------------------------------------
    def update(self, dt: float, now: float, cycle: float, level: float, stride_u: float, odo: float,
               moving_w: float, neutral: dict, reach: dict, facing: int, airborne: bool) -> FootFrame:
        out = FootFrame()
        if not self._init:
            self.reset(neutral, odo)
        if facing != self._facing:
            self._facing = facing
            self.reset(neutral, odo)
        g = blend_gait(self.table, level)
        self.gait = g
        out.gait = g
        if airborne:
            self._airborne = True
            for f, st in self.feet.items():
                st.planted = False
                st.settle = None
                st.phase = "AIR"
                # airborne paws clear the ground (they used to hang at ground level and skid along
                # it at the start / end of a hop)
                st.y += (-2.5 - st.y) * 0.5
                out.targets[f] = (st.local_x, neutral[f][1] + st.y)
                out.contacts[f] = False
                out.phases[f] = "AIR"
            out.cycle_phase = "AIRBORNE"
            return out
        if self._airborne:           # just landed: plant where we are
            self._airborne = False
            for f, st in self.feet.items():
                lo, hi = neutral[f][0] - reach.get(f, 10.0), neutral[f][0] + reach.get(f, 10.0)
                st.local_x = max(lo, min(hi, st.local_x))
                st.planted, st.plant_world, st.y, st.phase = True, odo + st.local_x, 0.0, "CONTACT"
                out.events.append((FOOT_EVENT, f))
        c = cycle % 1.0
        moving = moving_w > 0.05
        support = 0.0
        for f, st in self.feet.items():
            nx, ny = neutral[f]
            r = reach.get(f, 12.0)
            sweep = min(stride_u * g.duty, 2 * r * 0.95)
            if moving:
                st.settle = None
                u = (c + g.offsets.get(f, 0.0)) % 1.0
                wrapped = (st.u - u) > 0.5
                st.u = u
                if wrapped:
                    st.wrapped = True
                lo, hi = nx - r, nx + r
                if st.planted:
                    local = st.plant_world - odo
                    if u >= g.duty or local < lo:
                        # lift at the end of stance - or early when the leg is fully stretched
                        st.planted = False
                        st.lift_local = max(lo - r * 0.25, local)   # lift from where the paw is (clamping it
                        # forward to the reach limit made it skid on the lift-off frame)
                        st.lift_u = u
                        st.wrapped = False
                    else:
                        if local > hi:
                            st.plant_world -= local - hi
                            local = hi
                        st.local_x = local
                        st.y = 0.0
                        st.phase = "CONTACT" if u < 0.15 * g.duty else "PLANTED"
                        support += math.sin(math.pi * min(1.0, u / g.duty))
                if not st.planted:
                    if st.wrapped and u < g.duty:
                        st.planted = True
                        st.plant_world = odo + st.local_x
                        st.y = 0.0
                        st.phase = "CONTACT"
                        out.events.append((FOOT_EVENT, f))
                        support += math.sin(math.pi * min(1.0, u / g.duty))
                    else:
                        span = (1.0 - st.lift_u) if st.lift_u < 1.0 else 1.0
                        k = ((u - st.lift_u) % 1.0) / max(1e-3, span) if not st.wrapped else 1.0
                        k = max(0.0, min(1.0, k))
                        land = nx + min(sweep * 0.5, r * 0.95)    # never beyond reach: a foot planted
                        # out of reach was dragged back while "planted" (sliding in a gallop)
                        # the paw peels up off the ground before it travels, and sets down after
                        # (horizontal progress lags the lift at both ends: no skid at lift-off / touch-down)
                        kx = max(0.0, min(1.0, (k - 0.1) / 0.8))
                        e = kx if self.mechanical else _ease(kx)
                        st.local_x = st.lift_local + (land - st.lift_local) * e
                        arc = math.sin(math.pi * k)
                        if self.mechanical:
                            arc = min(1.0, arc * 2.2)      # servo: lift, carry flat, set down
                        st.y = -g.lift * arc * min(1.0, moving_w * 1.5)
                        st.phase = "LIFT" if k < 0.25 else ("SWING" if k < 0.8 else "REACH")
            else:
                self._settle_foot(st, now, odo, nx, r)
                if st.planted and st.settle is None:
                    support += 0.5
            out.targets[f] = (st.local_x, ny + st.y)
            out.contacts[f] = st.planted
            out.phases[f] = st.phase
        n = max(1, len(self.feet))
        out.support = support / (n * 0.5)
        lead = c if self.kind != "trot" or level > 1.5 else (c * 2.0) % 1.0
        if moving:
            if level < 1.5:
                names = ("CONTACT", "DOWN", "PASS", "UP")
                out.cycle_phase = names[int(((c * 2.0) % 1.0) * 4) % 4]
            else:
                names = ("CONTACT", "GATHER", "EXTEND", "FLIGHT")
                out.cycle_phase = names[int(lead * 4) % 4]
        else:
            out.cycle_phase = "SETTLE" if any(s.settle for s in self.feet.values()) else "STANCE"
        self._prev_cycle = cycle
        return out

    def _settle_foot(self, st: FootState, now: float, odo: float, nx: float, reach: float = 99.0) -> None:
        if st.settle is not None:
            s = st.settle
            k = min(1.0, (now - s["t0"]) / s["dur"])
            st.local_x = s["from"] + (nx - s["from"]) * (k if self.mechanical else _ease(k))
            st.y = s["y0"] * (1 - k) - 3.0 * math.sin(math.pi * k)
            st.phase = "SETTLE"
            if k >= 1.0:
                st.settle = None
                st.planted = True
                st.plant_world = odo + nx
                st.local_x = nx
                st.y = 0.0
                st.phase = "CONTACT"
            return
        if not st.planted:
            if now >= self._settle_next:
                self._settle_next = now + self.SETTLE_GAP
                st.settle = {"t0": now, "dur": self.SETTLE_DUR, "from": st.local_x, "y0": st.y}
            return
        st.local_x = st.plant_world - odo
        st.y = 0.0
        over = abs(st.local_x - nx) > reach * 0.9          # stretched leg: step right away
        if (abs(st.local_x - nx) > 2.5 and now >= self._settle_next) or over:
            self._settle_next = now + self.SETTLE_GAP + self.SETTLE_DUR * 0.5
            st.planted = False
            st.settle = {"t0": now, "dur": self.SETTLE_DUR, "from": st.local_x, "y0": 0.0}
        else:
            st.phase = "PLANTED"
