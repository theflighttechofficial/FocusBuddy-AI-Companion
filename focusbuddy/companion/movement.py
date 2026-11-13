"""LocomotionController: weight, momentum and natural movement.

Positions are the companion's *feet* (bottom-centre) in global logical
desktop coordinates.  Everything is integrated with delta-time (fixed
sub-steps of <= 1/240 s), so apparent speed is identical at 60/120/144 Hz.

Locomotion phases
    STOPPED -> (TURNING) -> ACCELERATING -> CRUISING -> DECELERATING -> STOPPED
    LEAPING (ballistic jump onto / off windows)   FALLING (dropped / ground vanished)

* Acceleration eases in, braking follows an arrival curve (never overshoots
  the target position - the *body* overshoots visually via the settle spring).
* Reversing direction brakes to a stop, turns (with a turn animation) and
  only then accelerates the other way.
* Gait phase advances with *distance travelled / stride length*, so feet are
  planted instead of sliding, and the walk->run blend follows real speed.

Second generation additions

* Gait levels WALK / JOG / RUN / SPRINT with their own top speed and stride.
  ``gait="escalate"`` is the intervention chase: walk -> jog -> run -> a short
  sprint burst (limited by ``profile.sprint_time``) -> arrival braking, which
  naturally passes back down through run / jog / walk.
* Anticipation: running starts from rest begin with a short wind-up
  (``sample().anticipation``) the animation turns into a lean-back + crouch.
* Physical jumps: ``jump(height)`` = crouch -> vertical velocity + gravity ->
  ground detection -> land event with impact speed.
* ``odometer`` (path length) and ``ground_y`` (support height under the body)
  let the animation plant feet and keep the shadow on the ground.

The class is still called ``Mover`` (public API unchanged); ``LocomotionController``
is an alias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .profile import CharacterProfile

# Kept for backwards compatibility (old defaults at scale 1.0)
WALK_SPEED = 80.0
RUN_SPEED = 280.0
GRAVITY = 2200.0
SUBSTEP = 1.0 / 240.0


@dataclass
class Vec:
    x: float
    y: float

    def __sub__(self, o: "Vec") -> "Vec":
        return Vec(self.x - o.x, self.y - o.y)

    def length(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass
class Leap:
    sx: float
    sy: float
    ex: float
    ey: float
    duration: float
    apex: float
    crouch: float
    t: float = 0.0


@dataclass
class LocoSample:
    """What the animation layer needs to know about locomotion each frame."""
    speed: float
    walk_speed: float
    run_speed: float
    phase: float
    run_blend: float
    accel: float           # signed, along travel direction (px/s^2)
    facing_blend: float    # -1 .. 1 (passes through 0 while turning)
    turning: bool
    airborne: bool
    crouch: float          # 0..1 pre-jump crouch
    leap_progress: float   # 0..1 while airborne
    vy: float
    mode: str              # STOPPED | ACCELERATING | CRUISING | DECELERATING | TURNING | LEAPING | FALLING | JUMPING
    gait_level: float = 0.0     # 0 walk .. 1 jog .. 2 run .. 3 sprint (continuous)
    gait_name: str = "IDLE"     # IDLE | WALK | JOG | RUN | SPRINT
    odometer: float = 0.0       # px travelled along the ground (monotonic)
    scale: float = 1.0          # px per px-at-size-1 (char_size / 124)
    stride: float = 34.0        # current stride length in px at size 1.0
    ground_dy: float = 0.0      # support surface below the feet (px, >= 0 while airborne)
    anticipation: float = 0.0   # 0..1 wind-up before a fast start / jump
    jumping: bool = False       # physical jump (vs. parametric leap)


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


class Mover:
    def __init__(self, x: float = 0.0, y: float = 0.0, profile: CharacterProfile | None = None, scale: float = 1.0):
        self.profile = profile or CharacterProfile()
        self.scale = scale
        self.pos = Vec(x, y)
        self.vel = Vec(0.0, 0.0)
        self.target: Optional[Vec] = None
        self.running = False
        self.gait = "walk"            # walk | run | walk_then_run
        self.speed_factor = 1.0
        self.facing = 1               # +1 right, -1 left (logical facing)
        self.facing_blend = 1.0       # rendered facing, animated through turns
        self.falling = False
        self.floor_y: float = y
        self.mode = "STOPPED"
        self.phase = 0.0              # gait phase (radians)
        self.run_blend = 0.0
        self.accel_along = 0.0
        self.events: list[tuple[str, float]] = []
        self.leap: Optional[Leap] = None
        self._turn_from = 1.0
        self._turn_to = 1
        self._turn_t = 0.0
        self._turn_dur = 0.0
        self._move_t = 0.0
        self._walk_lead = 0.7
        self._last_step = 0
        # gen 2
        self.odometer = 0.0
        self.gait_level = 0.0
        self.max_gait = "walk"         # walk | jog | run | sprint (current speed cap)
        self._sprint_used = 0.0
        self._antic = 0.0              # remaining anticipation (s)
        self._antic_total = 0.0
        self.jstate: Optional[dict] = None   # physical jump state
        self.shuffle_to: Optional[float] = None   # backpedal / sidestep: move without turning (x target)
        self.slow = False                    # slow walk (sad / tired / sneaky)
        self.movement_smoothness = 1.0     # 0.6 snappy .. 1.4 floaty (settings)

    # ------------------------------------------------------------- params
    @property
    def k(self) -> float:
        return self.scale * self.speed_factor

    @property
    def walk_speed(self) -> float:
        return self.profile.walk_speed * self.k

    @property
    def run_speed(self) -> float:
        return self.profile.run_speed * self.k

    @property
    def jog_speed(self) -> float:
        return self.profile.jog_speed * self.k

    @property
    def sprint_speed(self) -> float:
        return self.profile.sprint_speed * self.k

    def level_speed(self, gait: str) -> float:
        return {"walk": self.walk_speed, "jog": self.jog_speed, "run": self.run_speed,
                "sprint": self.sprint_speed}.get(gait, self.walk_speed)

    @property
    def max_speed(self) -> float:
        if self.max_gait == "walk" and self.running:
            return self.run_speed
        return self.level_speed(self.max_gait) * (0.55 if self.slow else 1.0)

    @property
    def gait_name(self) -> str:
        if self.speed < 0.15 * self.walk_speed or self.airborne:
            return "IDLE" if not self.airborne else "AIR"
        return ("WALK", "JOG", "RUN", "SPRINT")[min(3, int(self.gait_level + 0.5))]

    @property
    def ground_y(self) -> float:
        """Height of the support surface under the body (for the shadow / landing)."""
        if self.leap is not None:
            lp = self.leap
            u = min(1.0, max(0.0, lp.t))
            return lp.sy + (lp.ey - lp.sy) * u
        if self.jstate is not None:
            return self.jstate["ground"]
        if self.falling:
            return self.floor_y
        return self.pos.y

    @property
    def speed(self) -> float:
        return self.vel.length()

    @property
    def moving(self) -> bool:
        return self.target is not None or self.falling or self.leap is not None or self.turning

    @property
    def turning(self) -> bool:
        return self._turn_dur > 0

    @property
    def airborne(self) -> bool:
        return self.leap is not None or self.falling or self.jstate is not None

    def distance_to_target(self) -> float:
        return (self.target - self.pos).length() if self.target else 0.0

    # ----------------------------------------------------------- commands
    def move_to(self, x: float, y: float, run: bool = False, gait: str | None = None) -> None:
        resting = self.speed < 5 and self.target is None
        self.target = Vec(x, y)
        self.shuffle_to = None
        self.slow = gait == "slow"
        if gait == "slow":
            gait = "walk"
        self.gait = gait or ("run" if run else "walk")
        self.running = self.gait in ("run", "sprint")
        self.max_gait = {"jog": "jog", "run": "run", "sprint": "sprint"}.get(self.gait, "walk")
        self._move_t = 0.0
        self._sprint_used = 0.0
        if resting and self.gait in ("run", "sprint") and not self.airborne:
            # anticipation: a short wind-up before a fast start
            self._antic = self._antic_total = 0.12 if self.gait == "run" else 0.16
            self.events.append(("anticipate", 0.0))

    def shuffle(self, dx: float) -> None:
        """Short steps without turning: backpedal (dx against the facing) or sidestep. The feet re-plant
        with settle steps as the body moves over them, so nothing slides."""
        if self.airborne:
            return
        self.target = None
        self.vel = Vec(0.0, 0.0)
        self.shuffle_to = self.pos.x + dx

    def stop(self) -> None:
        """Immediate halt (used by directives). Natural stops use arrival braking / ``brake()``."""
        self.target = None
        self._antic = 0.0
        self.max_gait = "walk"
        self.running = False
        if self.jstate is not None or self.leap is not None or self.falling:
            return          # can't stop mid-air: gravity finishes the jump
        self.vel = Vec(0.0, 0.0)
        self.mode = "STOPPED"

    def brake(self) -> None:
        """Natural stop: drop the target and decelerate to rest (no pop)."""
        self.target = None
        self._antic = 0.0
        self.max_gait = "walk"
        self.running = False

    def teleport(self, x: float, y: float) -> None:
        self.pos = Vec(x, y)
        self.vel = Vec(0.0, 0.0)

    def drop(self, floor_y: float) -> None:
        self.target = None
        self.leap = None
        self.jstate = None
        self.falling = True
        self.floor_y = floor_y
        self.vel = Vec(self.vel.x * 0.4, 0.0)
        self.mode = "FALLING"

    def leap_to(self, x: float, y: float, apex: float | None = None) -> None:
        """Ballistic jump (onto a title bar, down from a window, or a hop in place)."""
        dx, dy = x - self.pos.x, y - self.pos.y
        rise = max(0.0, -dy)
        if apex is None:
            apex = rise + 34 * self.scale if rise > 0 else 24 * self.scale + abs(dx) * 0.08
        dist = math.hypot(dx, dy)
        dur = max(0.38, min(1.05, 0.34 + dist / 1500.0 + math.sqrt(max(apex, 1.0)) / 60.0))
        if abs(dx) > 4:
            self.facing = 1 if dx > 0 else -1
            self.facing_blend = float(self.facing)
        self.target = None
        self.vel = Vec(0.0, 0.0)
        self.leap = Leap(self.pos.x, self.pos.y, x, y, dur, apex, crouch=0.14)
        self.mode = "LEAPING"
        self.events.append(("crouch", 0.0))

    def jump(self, height: float | None = None, forward: float = 0.0, crouch: float | None = None) -> None:
        """Physical jump: crouch (anticipation) -> takeoff with vertical velocity -> gravity -> land.

        ``height`` is the apex in px (default: profile jump height x size); ``forward`` an
        optional horizontal speed (px/s, along facing) carried through the air."""
        if self.airborne:
            return
        h = (self.profile.jump_height * self.scale) if height is None else height
        if crouch is None:            # species style: quick cat crouch, slow mechanical robot wind-up
            crouch = getattr(self.profile, "jump_crouch", 0.13)
        self.target = None
        self.jstate = {"crouch": crouch, "crouch_total": crouch, "vy0": -math.sqrt(2 * GRAVITY * max(1.0, h)),
                     "vx": forward * self.facing, "ground": self.pos.y, "apex": h}
        self.vel = Vec(0.0, 0.0)
        self.mode = "JUMPING"
        self.events.append(("crouch", 0.0))

    def turn_to(self, direction: int) -> None:
        """Turn in place to face +1/-1 (animated)."""
        direction = 1 if direction >= 0 else -1
        if direction == self.facing and not self.turning:
            return
        self._turn_from = self.facing_blend
        self._turn_to = direction
        self._turn_t = 0.0
        frac = abs(direction - self.facing_blend) / 2.0   # 1 for a full 180
        self._turn_dur = max(0.08, self.profile.turn_time * max(0.35, frac))
        self.facing = direction
        self.events.append(("turn", float(direction)))

    def face_towards(self, x: float) -> None:
        if abs(x - self.pos.x) > 4:
            self.turn_to(1 if x > self.pos.x else -1)

    # ------------------------------------------------------------ update
    def update(self, dt: float) -> bool:
        """Advance by ``dt`` seconds. Returns True when a target/landing was reached."""
        dt = max(0.0, min(dt, 0.1))  # clamp huge frame gaps (suspend, drag)
        arrived = False
        while dt > 1e-9:
            h = min(SUBSTEP, dt)
            dt -= h
            if self._step(h):
                arrived = True
        return arrived

    def _step(self, h: float) -> bool:
        # --- turning in place ---------------------------------------------
        if self.turning:
            self._turn_t += h
            u = smoothstep(self._turn_t / self._turn_dur)
            self.facing_blend = self._turn_from + (self._turn_to - self._turn_from) * u
            self.mode = "TURNING"
            if self._turn_t >= self._turn_dur:
                self.facing_blend = float(self._turn_to)
                self._turn_dur = 0.0
                self.mode = "STOPPED"
            self._settle_gait(h)
            return False

        # --- shuffling (backpedal / sidestep) ------------------------------
        if self.shuffle_to is not None:
            d = self.shuffle_to - self.pos.x
            sp = 75.0 * self.scale
            if abs(d) <= sp * h:
                self.pos = Vec(self.shuffle_to, self.pos.y)
                self.odometer += d * self.facing
                self.shuffle_to = None
                self.mode = "STOPPED"
                return True
            step = math.copysign(sp * h, d)
            self.pos = Vec(self.pos.x + step, self.pos.y)
            self.odometer += step * self.facing          # signed: planted feet stay planted
            self.mode = "SHUFFLING"
            return False

        # --- physical jump -------------------------------------------------
        if self.jstate is not None:
            j = self.jstate
            if j["crouch"] > 0:
                j["crouch"] -= h
                if j["crouch"] <= 0:
                    self.vel = Vec(j["vx"], j["vy0"])
                    self.events.append(("takeoff", 0.0))
                return False
            self.vel.y += GRAVITY * h
            self.pos = Vec(self.pos.x + self.vel.x * h, self.pos.y + self.vel.y * h)
            dist = abs(self.vel.x * h)
            if dist > 0:
                self.odometer += dist
            if self.pos.y >= j["ground"] and self.vel.y > 0:
                impact = self.vel.y
                self.pos = Vec(self.pos.x, j["ground"])
                self.vel = Vec(0.0, 0.0)
                self.jstate = None
                self.mode = "STOPPED"
                self.events.append(("land", impact))
                return True
            return False

        # --- ballistic leap ------------------------------------------------
        if self.leap is not None:
            lp = self.leap
            if lp.crouch > 0:
                lp.crouch -= h
                if lp.crouch <= 0:
                    self.events.append(("takeoff", 0.0))
                return False
            lp.t += h / lp.duration
            u = min(1.0, lp.t)
            ux = u * u * (3 - 2 * u) * 0.35 + u * 0.65       # slight ease on horizontal travel
            ny = lp.sy + (lp.ey - lp.sy) * u - 4 * lp.apex * u * (1 - u)
            nx = lp.sx + (lp.ex - lp.sx) * ux
            self.vel = Vec((nx - self.pos.x) / h, (ny - self.pos.y) / h)
            self.pos = Vec(nx, ny)
            if u >= 1.0:
                impact = abs(self.vel.y)
                self.pos = Vec(lp.ex, lp.ey)
                self.vel = Vec(0.0, 0.0)
                self.leap = None
                self.mode = "STOPPED"
                self.events.append(("land", impact))
                return True
            return False

        # --- falling (dropped / surface vanished) --------------------------
        if self.falling:
            self.vel.y += GRAVITY * h
            self.pos = Vec(self.pos.x + self.vel.x * h, self.pos.y + self.vel.y * h)
            self.vel.x *= (1 - min(1.0, 3.0 * h))
            if self.pos.y >= self.floor_y:
                impact = self.vel.y
                self.pos.y = self.floor_y
                self.vel = Vec(0.0, 0.0)
                self.falling = False
                self.mode = "STOPPED"
                self.events.append(("land", impact))
                return True
            return False

        if self.target is None:
            self._brake_to_rest(h)
            return False

        delta = self.target - self.pos
        dist = delta.length()
        speed = self.speed
        if dist < 0.75 and speed < 40:
            self.pos = Vec(self.target.x, self.target.y)
            self._arrive(speed)
            return True

        # Need to reverse? brake -> turn -> go
        if abs(delta.x) > 6 * self.scale:
            want_dir = 1 if delta.x > 0 else -1
            if want_dir != self.facing:
                if speed > 12:
                    self._brake(h, self.profile.deceleration * self.k)
                    self.pos = Vec(self.pos.x + self.vel.x * h, self.pos.y + self.vel.y * h)
                    self.mode = "DECELERATING"
                    self._advance_gait(speed * h, h)
                    return False
                self.vel = Vec(0.0, 0.0)
                self.turn_to(want_dir)
                return False

        # anticipation (wind-up before a fast start)
        if self._antic > 0:
            self._antic -= h
            self.mode = "ANTICIPATING"
            self._settle_gait(h)
            return False

        # gait selection ("run the last distance")
        self._move_t += h
        smooth = max(0.5, self.movement_smoothness)
        decel = self.profile.deceleration * self.k / smooth
        if self.gait == "walk_then_run" and not self.running:
            brake_dist = self.run_speed ** 2 / (2 * decel)
            if self._move_t >= self._walk_lead and dist > brake_dist * 1.2 + 60 * self.scale:
                self.running = True
                self.max_gait = "run"
        elif self.gait == "escalate":
            self._escalate(h, dist, decel)
        max_v = self.max_speed
        desired = min(max_v, math.sqrt(2 * decel * max(0.0, dist - 0.5)))
        accel = self.profile.acceleration * self.k / smooth
        # ease-in: gentle first push, then full acceleration
        accel *= 0.45 + 0.55 * smoothstep(self._move_t / 0.35)
        dir_x, dir_y = delta.x / dist, delta.y / dist
        # steering with inertia: rotate current heading toward the target direction
        if speed > 1:
            hx, hy = self.vel.x / speed, self.vel.y / speed
            ang = math.atan2(hx * dir_y - hy * dir_x, hx * dir_x + hy * dir_y)
            max_turn = 5.0 * h
            ang = max(-max_turn, min(max_turn, ang))
            ca, sa = math.cos(ang), math.sin(ang)
            hx, hy = hx * ca - hy * sa, hx * sa + hy * ca
        else:
            hx, hy = dir_x, dir_y
        if desired > speed:
            new_speed = min(desired, speed + accel * h)
            self.mode = "ACCELERATING" if new_speed < max_v * 0.97 else "CRUISING"
        else:
            new_speed = max(desired, speed - decel * h)
            self.mode = "DECELERATING" if new_speed < speed - 1e-6 else "CRUISING"
        self.accel_along = (new_speed - speed) / h
        self.vel = Vec(hx * new_speed, hy * new_speed)
        step_x, step_y = self.vel.x * h, self.vel.y * h
        step_len = math.hypot(step_x, step_y)
        if step_len >= dist:  # never overshoot the destination
            self.pos = Vec(self.target.x, self.target.y)
            self._advance_gait(dist, h)
            self._arrive(new_speed)
            return True
        self.pos = Vec(self.pos.x + step_x, self.pos.y + step_y)
        self._advance_gait(step_len, h)
        return False

    def _escalate(self, h: float, dist: float, decel: float) -> None:
        """Intervention chase: walk -> jog -> run -> short sprint, then brake into the arrival."""
        t = self._move_t
        lead = self._walk_lead * 0.6
        if self.max_gait == "walk" and t >= lead:
            self.max_gait = "jog"
        elif self.max_gait == "jog" and t >= lead + 0.45 and dist > self.run_speed ** 2 / (2 * decel) + 40 * self.scale:
            self.max_gait = "run"
            self.running = True
        elif self.max_gait == "run" and self.speed > self.run_speed * 0.9 and self._sprint_used == 0.0:
            need = self.sprint_speed ** 2 / (2 * decel) * 1.3 + 90 * self.scale
            if dist > need:
                self.max_gait = "sprint"
                self.events.append(("sprint", self.speed))
        elif self.max_gait == "sprint":
            self._sprint_used += h
            need = self.sprint_speed ** 2 / (2 * decel) * 1.15
            if self._sprint_used >= self.profile.sprint_time or dist < need:
                self.max_gait = "run"
                self._sprint_used = max(self._sprint_used, 1e-3)

    def _arrive(self, impact_speed: float) -> None:
        self.target = None
        self.vel = Vec(0.0, 0.0)
        self.mode = "STOPPED"
        self.running = False
        self.max_gait = "walk"
        self.accel_along = 0.0
        self.events.append(("stop", impact_speed))

    def _brake(self, h: float, decel: float) -> None:
        sp = self.speed
        if sp <= 0:
            return
        ns = max(0.0, sp - decel * h)
        self.accel_along = (ns - sp) / h
        f = ns / sp
        self.vel = Vec(self.vel.x * f, self.vel.y * f)

    def _brake_to_rest(self, h: float) -> None:
        if self.speed > 0:
            self._brake(h, self.profile.deceleration * self.k * 1.5)
            self.pos = Vec(self.pos.x + self.vel.x * h, self.pos.y + self.vel.y * h)
            self._advance_gait(self.speed * h, h)
            return
        else:
            self.accel_along = 0.0
            self.mode = "STOPPED"
        self._settle_gait(h)

    def _target_level(self, sp: float) -> float:
        """Continuous gait level from real speed: 0 walk, 1 jog, 2 run, 3 sprint."""
        pts = (self.walk_speed, self.jog_speed, self.run_speed, self.sprint_speed)
        if sp <= pts[0]:
            return 0.0
        for i in range(3):
            a, b = pts[i], pts[i + 1]
            if sp <= b:
                return i + (sp - a) / max(1e-6, b - a)
        return 3.0

    def stride_px(self, level: float | None = None) -> float:
        """Stride length (px at size 1.0) for a gait level."""
        p = self.profile
        strides = (p.stride_walk, p.stride_jog, p.stride_run, p.stride_sprint)
        lv = self.gait_level if level is None else level
        lv = max(0.0, min(3.0, lv))
        i = min(2, int(lv))
        return strides[i] + (strides[i + 1] - strides[i]) * (lv - i)

    def _advance_gait(self, distance: float, h: float) -> None:
        sp = self.speed
        wr = self.walk_speed
        rr = max(wr + 1, self.run_speed)
        target_blend = smoothstep((sp - wr * 1.05) / (rr - wr * 1.05))
        self.run_blend += (target_blend - self.run_blend) * (1 - math.exp(-h / 0.18))
        self.gait_level += (self._target_level(sp) - self.gait_level) * (1 - math.exp(-h / 0.15))
        stride = self.stride_px() * self.scale
        self.odometer += distance
        self.phase += 2 * math.pi * distance / max(1.0, stride)
        steps = int(self.phase / math.pi)
        if steps != self._last_step:
            self._last_step = steps
            self.events.append(("step", sp))
        if self.phase > 1e6:
            self.phase %= 2 * math.pi
            self._last_step = int(self.phase / math.pi)

    def _settle_gait(self, h: float) -> None:
        """Ease the gait cycle back to a neutral stance and the run blend to walk."""
        self.run_blend += (0.0 - self.run_blend) * (1 - math.exp(-h / 0.25))
        self.gait_level += (0.0 - self.gait_level) * (1 - math.exp(-h / 0.25))

    def drain_events(self) -> list[tuple[str, float]]:
        ev, self.events = self.events, []
        return ev

    def sample(self) -> LocoSample:
        lp = self.leap
        j = self.jstate
        crouch = 0.0
        if lp is not None and lp.crouch > 0:
            crouch = 1 - max(0.0, lp.crouch) / 0.14
        elif j is not None and j["crouch"] > 0:
            crouch = 1 - max(0.0, j["crouch"]) / max(1e-6, j["crouch_total"])
        antic = 0.0
        if self._antic > 0 and self._antic_total > 0:
            antic = math.sin(math.pi * (1 - self._antic / self._antic_total))
        return LocoSample(
            speed=self.speed, walk_speed=self.walk_speed, run_speed=self.run_speed, phase=self.phase,
            run_blend=self.run_blend, accel=self.accel_along, facing_blend=self.facing_blend,
            turning=self.turning, airborne=self.airborne and crouch <= 0,
            crouch=crouch,
            leap_progress=min(1.0, lp.t) if lp is not None else 0.0, vy=self.vel.y, mode=self.mode,
            gait_level=self.gait_level, gait_name=self.gait_name, odometer=self.odometer, scale=self.scale,
            stride=self.stride_px(), ground_dy=max(0.0, self.ground_y - self.pos.y), anticipation=antic,
            jumping=j is not None)


LocomotionController = Mover
