"""AnimationController: turns state + actions + locomotion + emotion + attention into a Pose
and a solved skeleton.

Explicit layer pipeline (evaluated every frame inside the one companion loop):

    1. base           state pose (sit, sleep, press, celebrate ...)
    2. action         full-body / upper-body actions from the ActionPlayer,
                      each blended in / out by its own weight (no popping)
    3. locomotion     WALK / JOG / RUN / SPRINT body motion from the gait
                      tables: bob driven by real foot support, lean, spine flex,
                      anticipation (lean back before a sprint / crouch before a jump)
    4. emotion        ears, eyes, brows, mouth, tail wag, bounce
    5. attention      eyes (fast) and head (slower) toward the chosen target
    6. life           breathing (rate by state / exertion / emotion), weight
                      shifting, independent micro-movements (ear twitch, tail
                      flick, nose twitch, servo ticks) + additive actions
    7. secondary      springs: head lag, ear flop, stop settle, landing squash,
                      tail chain follow-through, antenna / hair / collar tag
    8. blending       every field eases toward its target with its own time
                      constant (eyes fast, body slow) - nothing ever pops
    9. skeleton       rig layout -> foot-phase system (planted feet) -> pose
                      overrides -> two-bone IK -> ``Pose.rig`` for the renderer

Animation events (``drain_events``): FOOT_CONTACT, JUMP_START, JUMP_LAND,
BLINK, TAIL_SWING ... the companion turns them into footsteps / particles.

``Animator`` is kept as an alias with the old ``pose(state, now, dt, speed,
max_speed)`` call used by previews and icons.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields, replace
from typing import Optional

from .gait import FOOT_EVENT, FootPhaseSystem, hop_flight, hop_stance
from .profile import CharacterProfile, get_profile
from .rig import Rig, apply_constraints, skeleton_for_profile
from .secondary import SecondarySystem
from .states import CompanionState as S

UNITS_PER_PX_AT_DEFAULT = 140.0 / 124.0   # renderer units per logical px at size 1.0
LEG_LEN = 21.0


@dataclass
class Pose:
    bob: float = 0.0          # body vertical offset (units, negative = up)
    lift: float = 0.0         # jump height (units, positive = up)
    tilt: float = 0.0         # body lean in degrees (positive = nose down / lean forward)
    leg_phase: float = 0.0    # radians
    leg_amp: float = 0.0      # stride amplitude (x24 degrees)
    sit: float = 0.0          # 0..1
    lie: float = 0.0          # 0..1
    stretch: float = 0.0      # 0..1 play-bow / stretch
    tail: float = 0.0         # tail angle offset (deg)
    ear: float = 0.0          # -1 droop .. 1 perked
    eye_open: float = 1.0     # 0..1.2
    look_x: float = 0.0       # -1..1 (pupil offset, + = facing direction)
    look_y: float = 0.0
    head_tilt: float = 0.0    # degrees
    paw: float = 0.0          # front paw / arm raise 0..1
    paw_wave: float = 0.0     # extra paw rotation (deg)
    arms_up: float = 0.0      # both arms up (bipeds) 0..1
    squash: float = 0.0       # -1 stretch .. 1 squash
    blush: float = 0.5
    # --- upgrade fields ---
    turn: float = 1.0         # rendered facing (-1..1), passes 0 mid-turn
    front: float = 0.0        # 0..1 face the viewer
    gait_run: float = 0.0     # 0 walk .. 1 run/gallop shape
    foot_lift: float = 0.0    # knee lift during swing
    flex: float = 0.0         # spine stretch (+) / gather (-) for the gallop
    head_dx: float = 0.0      # head lag offsets (units)
    head_dy: float = 0.0
    ear_flop: float = 0.0     # extra ear rotation from momentum (deg)
    brow: float = 0.0         # -1 angry .. +1 worried
    eye_scale: float = 1.0
    press: float = 0.0        # paw thrust toward a button 0..1 (negative = wind-up)
    ripple: float = 0.0       # click ripple progress 0..1 (0 = none)
    yawn: float = 0.0
    groom: float = 0.0
    scratch: float = 0.0
    read: float = 0.0
    hop: float = 0.0          # legs tucked for hop / airborne
    jump_vy: float = 0.0      # vertical speed while airborne (px/s, - = rising): motion-capture jump timing
    crouch: float = 0.0
    roll: float = 0.0         # whole-body rotation (deg) for flips
    # non-numeric
    mouth: str = "smile"      # smile grin open o flat sleep w frown yawn
    eye_style: str = "normal" # normal happy closed wide focused
    mark: str = ""            # ! ? zzz sparkle speed sweat dots heart
    emotion: str = ""         # the emotion shown (Emotion value; sheet faces pick their expression by it)
    mark_phase: float = 0.0
    headband: bool = False
    gait: str = "trot"
    # --- second generation channels ---
    breath: float = 0.0       # -1..1 breathing cycle (chest)
    shift: float = 0.0        # weight shift (body sways over planted feet)
    paw2: float = 0.0         # far front paw / far arm raise
    point: float = 0.0        # point forward
    thumb: float = 0.0        # thumbs up (bipeds)
    beckon: float = 0.0       # "come here": hand in front, palm up, fingers curling (bipeds)
    clap: float = 0.0
    chin: float = 0.0         # hand to chin (thinking)
    write: float = 0.0        # writing on a notepad
    tongue: float = 0.0
    nose: float = 0.0         # nose twitch -1..1
    scan: float = 0.0         # robot laser scan beam
    glitch: float = 0.0       # robot error glitch
    antenna: float = 0.0      # secondary physics (deg)
    hair: float = 0.0
    tag: float = 0.0
    crouch_body: float = 0.0  # lower the body (sneak, pounce wind-up, anticipation)
    tuck: float = 0.0         # paws tucked under (loaf)
    freeze: float = 0.0       # hold perfectly still
    ground: float = 0.0       # support surface below the feet (units) - shadow while airborne
    flush: float = 0.0        # red face tint (angry / embarrassed)
    cover: float = 0.0        # paws / hands over the face (embarrassed, crying)
    cross: float = 0.0        # arms crossed (grumpy)
    belly: float = 0.0        # hands on belly (laughing)
    stomp: float = 0.0        # stomp foot lift (angry)
    back: float = 0.0         # turn the back to the viewer (sulking) 0..1
    # --- living-character channels ---
    pelvis_roll: float = 0.0  # deg: hips roll side to side with each step
    torso_twist: float = 0.0  # deg: chest twist (shoulders counter-rotate against the hips)
    head_yaw: float = 0.0     # -1..1 head turn toward the look target (+ = facing direction)
    neck_yaw: float = 0.0     # -1..1 neck follows when the head runs out of range
    eye_conv: float = 0.0     # 0..0.8 eyes converge on a near target
    jaw: float = 0.0          # 0..1 jaw opening (follows the mouth shape)
    smile: float = 0.0        # -1 frown .. +1 smile (parametric mouth curve)
    mouth_open: float = 0.0   # 0..1
    mouth_width: float = 1.0  # mouth width scale
    lid: float = 0.0          # upper eyelid lowered 0..1
    lid_low: float = 0.0      # lower lid / cheek raise 0..1 (happy squint)
    lid_tilt: float = 0.0     # -1 angry (inner corners down) .. +1 sad (inner corners up)
    pupil: float = 1.0        # pupil / iris size
    brow_raise: float = 0.0   # -1 lowered .. +1 raised
    slump: float = 0.0        # drooping posture 0..1
    ear_l: float = 0.0        # per-ear spring offsets (deg) - asymmetric secondary motion
    ear_r: float = 0.0
    hem: float = 0.0          # hoodie hem / clothing spring (deg)
    brake: float = 0.0        # 0..1 braking from speed (lean back, feet dig in)
    foot_pitch: float = 0.0   # extra toe-down for all swinging feet (deg)
    # --- body language (3D character overhaul) ---
    head_roll: float = 0.0    # deg: cocked head (curious, confused, love)
    shoulders: float = 0.0    # -1 drooping .. +1 raised / tense
    fists: float = 0.0        # 0..1 clenched hands / braced paws (angry, determined)
    akimbo: float = 0.0       # 0..1 hands on hips (confident, impatient)
    hug: float = 0.0          # 0..1 arms pulled in to the chest (scared, nervous, love)
    arms_out: float = 0.0     # 0..1 arms open / spread (excited, surprised, shrug)
    stance: float = 0.0       # -1 narrow .. +1 wide, planted feet
    tail_pose: float = 0.0    # -1 tucked .. +1 held high
    puff: float = 0.0         # 0..1 chest expansion (inhale before a shout, pride)
    nod: float = 0.0          # face-life part of head_dy (talking nods, head drift) - kept on captured postures
    sway: float = 0.0         # face-life part of head_dx
    tail_chain: Optional[tuple] = None   # solved tail segment angles (secondary physics)
    rig: Optional[object] = None         # rig.RigPose (solved skeleton)
    action: str = ""                     # top action name (debug)
    action_t: float = 0.0                # seconds into the top action (GLB clip playback)
    action_w: float = 0.0                # its blend weight
    gait_level: float = 0.0              # 0 walk .. 1 jog .. 2 run .. 3 sprint (clip locomotion blending)
    gait_lift: float = 0.0               # part of ``lift`` that is the procedural hop gait (clips replace it)
    stride_u: float = 0.0                # rig stride (internal units per 2*pi of leg_phase), for clip speed matching
    moving: float = 0.0                  # 0 standing .. 1 moving (clip locomotion weight)
    clock: float = 0.0                   # animation time (s) - clip playback for idle / state clips
    state: str = ""                      # behaviour state name (clip selection)


_NUMERIC = [f.name for f in fields(Pose) if f.type in ("float", float)]

# Smoothing time constants (seconds). 0 = copy directly (driven by physics/cycles).
TAU = {name: 0.12 for name in _NUMERIC}
TAU.update({
    "look_x": 0.05, "look_y": 0.05, "eye_open": 0.06, "eye_scale": 0.1,
    "bob": 0.0, "lift": 0.025, "leg_phase": 0.0, "leg_amp": 0.07, "foot_lift": 0.06, "flex": 0.04,
    "tilt": 0.11, "squash": 0.04, "sit": 0.24, "lie": 0.32, "stretch": 0.16, "paw": 0.2,
    "paw_wave": 0.02, "press": 0.025, "ripple": 0.0, "turn": 0.0, "gait_run": 0.0, "hop": 0.03, "jump_vy": 0.0,
    "crouch": 0.0, "ear_flop": 0.0, "head_dx": 0.07, "head_dy": 0.05, "tail": 0.06, "mark_phase": 0.0,
    "yawn": 0.12, "groom": 0.14, "scratch": 0.1, "read": 0.25, "front": 0.1, "arms_up": 0.17,
    "roll": 0.0,
    "breath": 0.0, "shift": 0.3, "paw2": 0.2, "point": 0.12, "thumb": 0.12, "beckon": 0.12, "clap": 0.1, "chin": 0.2,
    "write": 0.25, "tongue": 0.1, "nose": 0.0, "scan": 0.1, "glitch": 0.0, "antenna": 0.0, "hair": 0.0, "tag": 0.0,
    "crouch_body": 0.08, "tuck": 0.3, "freeze": 0.1, "ground": 0.0,
    "flush": 0.2, "cover": 0.12, "cross": 0.15, "belly": 0.12, "stomp": 0.03, "back": 0.18,
    "pelvis_roll": 0.05, "torso_twist": 0.08, "head_yaw": 0.12, "neck_yaw": 0.25, "eye_conv": 0.1, "jaw": 0.05,
    "smile": 0.14, "mouth_open": 0.06, "mouth_width": 0.12, "lid": 0.08, "lid_low": 0.1, "lid_tilt": 0.14,
    "pupil": 0.25, "brow_raise": 0.1, "slump": 0.35, "ear_l": 0.0, "ear_r": 0.0, "hem": 0.0, "brake": 0.08,
    "foot_pitch": 0.05,
    "head_roll": 0.16, "shoulders": 0.14, "fists": 0.1, "akimbo": 0.2, "hug": 0.16, "arms_out": 0.14,
    "stance": 0.25, "tail_pose": 0.22, "puff": 0.12, "nod": 0.05, "sway": 0.07, "gait_level": 0.1, "moving": 0.08, "clock": 0.0,
    "gait_lift": 0.025, "stride_u": 0.0,
})
# how far each mouth drawing opens the jaw (the parametric jaw follows the chosen shape)
MOUTH_JAW = {"open": 0.45, "o": 0.3, "grin": 0.25, "shout": 1.0, "laugh": 0.75, "yawn": 0.9, "teeth": 0.1,
             "wobble": 0.1, "tongue": 0.2}
# per-phase foot pitch (deg, + = toe down): heel lifts, toe points through the swing, flattens to land
FOOT_PITCH = {"quad": {"LIFT": 28.0, "SWING": 18.0, "REACH": 4.0, "AIR": 32.0},
              "biped": {"LIFT": 24.0, "SWING": 10.0, "REACH": -12.0, "AIR": 26.0},
              "mech": {"LIFT": 8.0, "SWING": 4.0, "REACH": -4.0, "AIR": 10.0}}
_STRINGS = ("mouth", "eye_style", "mark", "emotion")
# channels smoothed with ease-in / ease-out (big limb and posture moves)
EASED = frozenset({"paw", "paw2", "arms_up", "point", "thumb", "beckon", "clap", "chin", "write", "cover", "cross",
                   "belly", "hug", "arms_out", "akimbo", "fists", "sit", "lie", "stretch", "groom", "scratch", "read",
                   "crouch_body", "tuck", "head_tilt", "head_roll", "shoulders", "slump", "back", "stance", "puff",
                   "tail_pose", "yawn"})


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def ease_out_back(x: float, k: float = 1.6) -> float:
    x = max(0.0, min(1.0, x)) - 1
    return 1 + (k + 1) * x * x * x + k * x * x


def _bump(t: float, duration: float) -> float:
    if t <= 0 or t >= duration:
        return 0.0
    return math.sin(math.pi * t / duration)


class Spring:
    """Damped spring for secondary motion (follow-through, overshoot, settle)."""

    def __init__(self, freq: float = 4.0, zeta: float = 0.5):
        self.freq = freq
        self.zeta = zeta
        self.x = 0.0
        self.v = 0.0

    def impulse(self, dv: float) -> None:
        self.v += dv

    def update(self, target: float, dt: float) -> float:
        w = 2 * math.pi * self.freq
        n = max(1, int(dt / (1 / 240)) + 1)
        h = dt / n
        for _ in range(n):
            a = w * w * (target - self.x) - 2 * self.zeta * w * self.v
            self.v += a * h
            self.x += self.v * h
        return self.x


@dataclass
class _LegacyLoco:
    speed: float
    walk_speed: float
    run_speed: float
    phase: float
    run_blend: float
    accel: float = 0.0
    facing_blend: float = 1.0
    turning: bool = False
    airborne: bool = False
    crouch: float = 0.0
    leap_progress: float = 0.0
    vy: float = 0.0
    mode: str = "CRUISING"
    gait_level: float = 0.0
    gait_name: str = "WALK"
    odometer: float = 0.0
    scale: float = 1.0
    stride: float = 34.0
    ground_dy: float = 0.0
    anticipation: float = 0.0
    jumping: bool = False


class TailChain:
    """Chain of springs: each segment's world angle chases (parent angle + rest bend), so a
    moving base produces natural follow-through toward the tip."""

    def __init__(self, rest: tuple, stiffness: tuple):
        self.rest = tuple(rest)
        n = len(rest)
        st = tuple(stiffness) + (2.0,) * max(0, n - len(stiffness))
        self.springs = [Spring(st[i], 0.42) for i in range(n)]
        self._init = False

    def update(self, base: float, dt: float, amount: float = 1.0, stiff: float = 1.0) -> tuple:
        n = len(self.rest)
        if n == 0:
            return ()
        if not self._init:
            ang = base
            for i, sp in enumerate(self.springs):
                if i:
                    ang += self.rest[i]
                sp.x, sp.v = ang, 0.0
            self._init = True
        world = [base]
        self.springs[0].x = base
        for i in range(1, n):
            sp = self.springs[i]
            freq = sp.freq
            sp.freq = freq * stiff
            target = world[-1] + self.rest[i]
            x = sp.update(target, dt)
            sp.freq = freq
            # amount < 1 pulls the result toward the rigid chain (less follow-through)
            x = target + (x - target) * amount
            lo, hi = target - 70.0, target + 70.0
            x = max(lo, min(hi, x))
            if not (x == x):          # NaN guard
                x = target
                sp.x, sp.v = target, 0.0
            world.append(x)
        return (0.0,) + tuple(world[i] - world[i - 1] for i in range(1, n))


class AnimationController:
    def __init__(self, seed: int | None = None, profile: CharacterProfile | None = None):
        self._rng = random.Random(seed)
        self.profile = profile or get_profile("cat")
        self._state: S | None = None
        self._state_t0 = 0.0
        self._cur = Pose()
        self._first = True
        # blinking
        self._next_blink = 2.0
        self._blink_start = -10.0
        self._blink_dur = 0.11
        self._double_at: Optional[float] = None
        self._blink_depth = 1.0
        # springs
        self.settle = Spring(2.6, 0.32)
        self.land = Spring(3.2, 0.38)
        self.tail_spring = Spring(1.8, 0.38)
        self.ear_spring = Spring(2.4, 0.4)
        self.head_spring = Spring(3.0, 0.55)
        # reflexes (2.2): a quick physical flinch when the companion is poked - head, body or tail.
        # Applied after the smoothing stage so it stays snappy, and kept out of the smoothing state
        self._reflex = {"head": Spring(2.6, 0.50), "body": Spring(2.4, 0.42), "tail": Spring(2.2, 0.30)}
        self._reflex_off: dict[str, float] = {}
        self._legacy_phase = 0.0
        self._ear_twitch_at = 3.0
        self.intensity = 1.0
        self.celebration = None   # override of profile.celebration
        self.phase_name = "IDLE_CYCLE"
        # --- gen 2 -------------------------------------------------------
        self.smoothness = 1.0       # Low 0.65 / Normal 1 / High 1.35  (blend time constants)
        self.secondary_amount = 1.0 # Low 0.5 / Normal 1 / High 1.4  (follow-through)
        self.physics = 1.0          # Low 0.6 / Normal 1 / High 1.3  (squash, springs, impacts)
        self.idle_activity = 0.5    # micro-movement frequency
        self.emotion_amount = 1.0   # Low 0.5 / Normal 1 / High 1.4 (expressiveness of the emotion pack)
        self.events: list[tuple[str, str]] = []
        self.antenna_spring = Spring(3.4, 0.22)
        self.hair_spring = Spring(2.2, 0.45)
        self.tag_spring = Spring(1.7, 0.18)
        self._breath_phase = 0.0
        self._breath_val = 0.0
        self.breath_rate = self.profile.breath_rate
        self._exertion = 0.0
        self._shift_seed = self._rng.uniform(0, 10)
        self._support = 0.5
        self._support_mean = 0.5
        self._micro: list[dict] = []
        self._micro_next: dict[str, float] = {}
        self._last_body_y = 0.0
        self._prev_airborne = False
        self.foot_frame = None
        self.secondary = SecondarySystem(self.profile.id)
        self._facing_prev = 1.0
        self._foot_ang: dict[str, float] = {}
        self._rebound_t0 = -10.0
        self._now_hint = 0.0
        self._rebound_h = 0.0
        self._stride_seed = self._rng.uniform(0, 100)
        self.ik_targets: dict = {}          # named IK target overrides (Playground / actions)
        self.face_viewer = True             # when standing still, turn the body 3/4 toward the user
        self._idle_turn = 1.0
        self._prev_look_x = 0.0
        self._last_saccade_blink = -10.0
        self._s1: dict = {}                 # first stage of the eased (two-stage) smoothing
        self._sink = 0.0                    # reach guard: how far the body sinks so legs can carry a step
        from .expression import FaceLife
        self.face = FaceLife(self._rng.randint(0, 1 << 30))   # talking, micro-expressions, sighs
        self.set_rig(None)

    # ------------------------------------------------------------- rig
    def set_rig(self, rig: Rig | None) -> None:
        """Attach the character's skeleton (defaults to the profile's species skeleton)."""
        self.rig = rig or Rig(skeleton_for_profile(self.profile.id, self.profile.gait))
        sk = self.rig.sk
        names = tuple(l.name for l in sk.legs)
        self.feet = FootPhaseSystem(names, "biped" if sk.kind == "biped" and self.profile.gait != "mech"
                                    else ("mech" if self.profile.gait == "mech" else
                                          ("hop" if self.profile.gait == "hop" else "trot")),
                                    mechanical=self.profile.mechanical)
        self.tail = TailChain(sk.tail.rest, sk.tail.stiffness)
        if hasattr(self, "secondary"):
            self.secondary.set_species(self.species)

    @property
    def species(self) -> str:
        sid = self.rig.sk.id if self.rig is not None else self.profile.id
        return sid if sid in ("cat", "dog", "fox", "rabbit", "robot", "buddy") else (
            "robot" if self.profile.gait == "mech" else "buddy" if self.profile.gait == "biped" else "cat")

    def talk(self, text: str, now: float, duration: float | None = None) -> float:
        """Move the mouth (and head / brows) along with a line of speech."""
        return self.face.talk(text, now, duration)

    def drain_events(self) -> list[tuple[str, str]]:
        ev, self.events = self.events, []
        return ev

    def set_profile(self, profile: CharacterProfile) -> None:
        old = self.profile
        self.profile = profile
        self.breath_rate = profile.breath_rate
        if old.gait != profile.gait or old.id != profile.id:
            self.set_rig(None)

    # --------------------------------------------------------------- events
    def on_event(self, name: str, value: float = 0.0) -> None:
        p = self.profile
        if name == "stop" and value > 25:
            self.settle.impulse(min(1.2, value / 260.0) * 6.0 * (0.4 + p.overshoot))
        elif name == "land":
            k = min(1.4, value / 700.0 + 0.3)
            stiff = 0.55 if p.jump_style == "mech" else 1.0
            self.land.impulse(k * 7.0 * self.physics * stiff)
            # loose parts react to the impact: ears / tail / antenna / hair snap down, then settle
            self.secondary.kick_all(-k * 60.0 * self.physics)
            self.secondary.impulse("head", k * 20.0)
            self.events.append(("JUMP_LAND", f"{value:.0f}"))
            if p.jump_style == "mech":
                self.events.append(("LAND_HEAVY", "clunk"))
            if p.land_rebound > 0 and value > 250:
                self._rebound_t0 = self._now_hint
                self._rebound_h = 7.0 * p.land_rebound * min(1.2, value / 600.0)
        elif name == "takeoff":
            self.land.impulse(-2.5 * self.physics)
            self.secondary.kick_all(25.0 * self.physics)
            self.events.append(("JUMP_START", ""))
        elif name == "sprint":
            self.settle.impulse(-5.0 * (0.5 + 0.5 * self.physics))   # lean back, then drive forward
        elif name == "anticipate":
            self.settle.impulse(-2.0)
        elif name.startswith("poke_"):
            part = name[5:]
            s = self._reflex.get(part) or self._reflex["body"]
            s.impulse(2 * math.pi * s.freq * (0.6 + 0.4 * self.physics) * max(0.3, min(1.5, value or 1.0)))
            self._next_blink = min(self._next_blink, self._now_hint)    # flinching comes with a blink
            self._double_at = None
            self.events.append(("POKE", part))

    # -------------------------------------------------------- legacy API
    def pose(self, state: S, now: float, dt: float, speed: float = 0.0, max_speed: float = 1.0) -> Pose:
        walk = max(1.0, max_speed if state == S.WALKING else max_speed * 0.3)
        run = max(walk + 1, max_speed)
        moving = state in (S.WALKING, S.RUNNING, S.CHASING)
        sp = (speed if speed > 0 else (walk * 0.9 if state == S.WALKING else run * 0.95)) if moving else 0.0
        rb = 1.0 if state in (S.RUNNING, S.CHASING) else 0.0
        stride = (self.profile.stride_walk + (self.profile.stride_run - self.profile.stride_walk) * rb)
        self._legacy_phase += 2 * math.pi * sp * dt / max(1.0, stride)
        self._legacy_odo = getattr(self, "_legacy_odo", 0.0) + sp * dt
        loco = _LegacyLoco(sp, walk, run, self._legacy_phase, rb, gait_level=2.0 * rb, odometer=self._legacy_odo,
                           stride=stride)
        return self.update(state, now, dt, loco if moving else None)

    # ------------------------------------------------------------- update
    def update(self, state: S, now: float, dt: float, loco=None, emotion=None, look=None,
               intensity: float | None = None, actions=None, energy: float | None = None) -> Pose:
        """One frame of the layer pipeline. ``actions`` = ActionPlayer.layers() (optional)."""
        if intensity is not None:
            self.intensity = intensity
        self._now_hint = now
        k = self.intensity
        if state != self._state:
            self._state = state
            self._state_t0 = now
        t = now - self._state_t0
        self._breathing(state, loco, emotion, dt, energy)
        # 1. base
        target = self._state_pose(state, t, now)
        target.gait = self.profile.gait
        target.clock = now
        target.state = getattr(state, "value", str(state))
        # 2. actions (full body, then upper body)
        layers = list(actions or ())
        if layers:
            self._apply_actions(target, layers, now, ("full", "upper"))
            top = max(layers, key=lambda a: (a.track == "full", a.weight))
            target.action = top.name
            target.action_t = top.t
            target.action_w = top.weight
        # 3. locomotion
        moving_w = 0.0
        if loco is not None:
            spin = target.turn
            moving_w = self._gait_layer(target, state, loco, now, k)
            target.turn = loco.facing_blend * spin
            if loco.turning:
                target.front = max(target.front, 1 - abs(loco.facing_blend))
        # standing still: open the body up toward the viewer (3/4) - bipeds a lot, animals a little
        calm_idle = state in (S.IDLE, S.SITTING, S.WAVING, S.PETTED, S.LOOKING_AROUND, S.CONFUSED, S.CELEBRATING,
                              S.WATCHING, S.YAWNING, S.STRETCHING)
        moving = loco is not None and (loco.speed > 8 or loco.turning or loco.airborne)
        want = (0.5 if self.rig.sk.kind == "biped" else 0.82) if (self.face_viewer and calm_idle and not moving) else 1.0
        tau = 0.45 if want < self._idle_turn else 0.18
        self._idle_turn += (want - self._idle_turn) * (1 - math.exp(-dt / tau))
        if self._idle_turn < 0.999:
            target.turn *= self._idle_turn
        # 4. emotion
        if emotion is not None:
            self._emotion_layer(target, state, emotion, now, moving_w, k)
        # 5. attention
        if look is not None:
            jump = abs(look.look_x - self._prev_look_x) + abs(look.look_y * 0.5)
            if jump > 0.45 and now - self._last_saccade_blink > 1.5 and now - self._blink_start > 0.6:
                self._next_blink = now                     # a big gaze shift comes with a blink
                self._last_saccade_blink = now
                self.face.flash(now)                       # ... and a quick eyebrow flash
            self._prev_look_x = look.look_x
        if look is not None and target.eye_style not in ("closed",):
            target.look_x = look.look_x if state not in (S.LOOKING_AROUND,) else target.look_x
            target.look_y = look.look_y if state not in (S.LOOKING_AROUND, S.PEEKING, S.READING) else target.look_y
            target.head_dx += look.head_x * 2.5 * k
            target.head_dy += look.head_y * 2.0 * k
            target.head_tilt += look.head_y * 5.0
            target.front = max(target.front, look.front)
            # look-at chain: eyes (fast) -> head -> neck -> torso, each joining in when the one
            # before it runs out of range (all clamped by the rig constraints)
            target.head_yaw += look.head_x
            target.neck_yaw += getattr(look, "neck_x", 0.0)
            target.torso_twist += getattr(look, "torso_x", 0.0) * 9.0
            target.head_dx += getattr(look, "neck_x", 0.0) * 2.0 * k
            target.eye_conv = max(target.eye_conv, getattr(look, "converge", 0.0))
        # 6. life: breathing, weight shift, micro-movements, additive actions
        self._life(target, state, moving_w, now, dt)
        if layers:
            self._apply_actions(target, layers, now, ("additive",))
        # 6b. face life: lip motion while talking, micro-saccades, brow flashes, micro-smiles, sighs
        hy0, hx0 = target.head_dy, target.head_dx
        self.face.apply(target, now, dt, emotion=target.emotion, asleep=state == S.SLEEPING, thinking=state == S.READING or target.chin > 0.3,
                        moving=moving_w, mechanical=self.profile.mechanical)
        target.nod, target.sway = target.head_dy - hy0, target.head_dx - hx0
        # 7. secondary physics
        self._secondary(target, loco, dt, k)
        self._finish(target, now)
        # 8. blending
        out = self._smooth(target, dt)
        self._blink(out, now, emotion, state)
        self._reflexes(out, dt)
        out.mark_phase = now
        # 9. skeleton (planted feet + IK)
        self._rig_layer(out, loco, dt, now, moving_w)
        cur = replace(out)
        for n, v in self._reflex_off.items():      # the smoothing state never sees the flinch
            setattr(cur, n, getattr(cur, n) - v)
        self._cur = cur
        return replace(out)

    def _reflexes(self, p: Pose, dt: float) -> None:
        h = self._reflex["head"].update(0.0, dt)
        b = self._reflex["body"].update(0.0, dt)
        t = self._reflex["tail"].update(0.0, dt)
        if abs(h) + abs(b) + abs(t) < 1e-3:
            self._reflex_off = {}
            return
        k = self.intensity
        off = {"head_dy": -4.5 * h * k,                                  # head pulls back ...
               "tilt": (-6.0 * h - 3.5 * b) * k,                          # ... the body leans away
               "squash": 0.10 * b * self.physics,                         # a poke in the belly squashes
               "crouch_body": 0.35 * max(0.0, b) * self.physics,
               "ear_flop": (-14.0 * h - 6.0 * b - 9.0 * t) * self.secondary_amount,   # ears flatten
               "tail": 40.0 * t * k}                                      # a poked tail whips
        for n, v in off.items():
            setattr(p, n, getattr(p, n) + v)
        self._reflex_off = off

    # ------------------------------------------------------- gen 2 layers
    def _apply_actions(self, p: Pose, layers, now: float, tracks: tuple) -> None:
        from .actions import UPPER_FIELDS
        for a in layers:
            if a.track not in tracks or a.adef.pose is None or a.weight <= 0.001:
                continue
            q = replace(p)
            try:
                a.adef.pose(q, a.u, a.t, now, self.species)
            except Exception:  # noqa: BLE001 - a broken recipe must never break the loop
                continue
            w = a.weight
            names = _NUMERIC if a.track != "upper" else [n for n in _NUMERIC if n in UPPER_FIELDS]
            for n in names:
                b = getattr(q, n)
                if b != getattr(p, n):
                    av = getattr(p, n)
                    setattr(p, n, av + (b - av) * w)
            if w > 0.5:
                if q.headband and not p.headband:
                    p.headband = True
                for n in _STRINGS:
                    v = getattr(q, n)
                    if v != getattr(p, n) and (a.track != "upper" or n in UPPER_FIELDS):
                        setattr(p, n, v)

    def _finish(self, p: Pose, now: float) -> None:
        """Parametric face follow-ups + landing rebound + joint constraints (before smoothing)."""
        p.jaw = max(p.jaw, p.mouth_open * 0.8, MOUTH_JAW.get(p.mouth, 0.0) * (p.yawn if p.mouth == "yawn" else 1.0))
        rt = now - self._rebound_t0
        if 0.06 < rt < 0.36 and self._rebound_h > 0:      # happy little second bounce (dog / rabbit)
            p.lift += self._rebound_h * math.sin(math.pi * (rt - 0.06) / 0.3)
        if p.slump > 0.01:                                 # sad / bored / tired posture
            p.head_dy += 4.5 * p.slump
            p.tilt += 2.5 * p.slump
            p.ear = max(-1.0, p.ear - 0.4 * p.slump)
        apply_constraints(p, self.species, self.rig.sk.kind)

    def _breathing(self, state: S, loco, emotion, dt: float, energy: float | None) -> None:
        sp = loco.speed if loco is not None else 0.0
        run = max(1.0, loco.run_speed) if loco is not None else 1.0
        # exertion builds while running and recovers slowly afterwards
        drive = min(1.0, sp / run)
        tau = 1.2 if drive > self._exertion else 6.0
        self._exertion += (drive - self._exertion) * (1 - math.exp(-dt / tau))
        rate = self.breath_rate
        if state == S.SLEEPING:
            rate *= 0.7
        rate += 0.9 * self._exertion
        if emotion is not None and getattr(emotion, "bounce", 0) > 0:
            rate += 0.15
        if energy is not None and energy < 25:
            rate *= 0.9
        rate *= getattr(self, "_breath_style", (1.0, 1.0))[0]
        # 2.2: breathing like a living thing instead of a metronome - every breath a little
        # different in pace and depth, now and then a slow deep breath while calm, a short inhale,
        # a longer exhale and a small pause before the next breath (gone when out of breath)
        c_rate, c_depth = getattr(self, "_breath_cycle", (1.0, 1.0))
        ph = self._breath_phase + 2 * math.pi * rate * c_rate * dt
        if ph >= 2 * math.pi:
            ph %= 2 * math.pi
            calm = state in (S.IDLE, S.SITTING, S.WATCHING, S.FOCUS_MODE, S.READING, S.SLEEPING)
            if calm and self._exertion < 0.2 and self._rng.random() < 1.0 / 14.0:
                self._breath_cycle = (0.7, 1.7)
            else:
                self._breath_cycle = (self._rng.uniform(0.9, 1.1), self._rng.uniform(0.86, 1.08))
        self._breath_phase = ph
        u = ph / (2 * math.pi)
        rest = 0.15 * (1.0 - min(1.0, self._exertion * 1.5))
        top = 0.38 * (1.0 - rest) / 0.85
        if u < top:
            w = -1.0 + 2.0 * _ease(u / top)
        elif u < 1.0 - rest:
            w = 1.0 - 2.0 * _ease((u - top) / max(1e-3, 1.0 - rest - top))
        else:
            w = -1.0
        self._breath_val = w * c_depth

    def _life(self, p: Pose, state: S, moving_w: float, now: float, dt: float) -> None:
        still = 1.0 - _clamp01(p.freeze)
        amp = 1.0 + 0.6 * self._exertion
        if state == S.SLEEPING:
            amp = 1.4
        p.breath = self._breath_val * amp * still * getattr(self, "_breath_style", (1.0, 1.0))[1]
        # weight shifting while standing around (feet stay planted - the IK legs adapt)
        calm = state in (S.IDLE, S.WATCHING, S.WARNING, S.LOOKING_AROUND, S.CLOSING_DISTRACTION, S.WAVING)
        if calm and moving_w < 0.05 and p.sit < 0.3 and p.lie < 0.3:
            sh = math.sin(now * 2 * math.pi / 7.3 + self._shift_seed) * 0.75 \
                + math.sin(now * 2 * math.pi / 3.1 + self._shift_seed * 2) * 0.25
            p.shift += sh * still * (0.4 if self.profile.mechanical else 1.0)
            p.tilt += sh * 0.6 * still
        # micro-movements (independent additive layers)
        self._schedule_micro(state, moving_w, now)
        keep = []
        for m in self._micro:
            t = now - m["t0"]
            if t >= m["dur"]:
                continue
            keep.append(m)
            u = t / m["dur"]
            w = still * self.secondary_amount
            fn = m["fn"]
            q = replace(p)
            fn(q, u, t, now, self.species)
            for n in ("ear_flop", "ear", "tail", "nose", "head_tilt", "head_dx", "look_x", "look_y", "head_roll",
                      "ear_l", "ear_r"):
                a, b = getattr(p, n), getattr(q, n)
                if a != b:
                    setattr(p, n, a + (b - a) * w)
        self._micro = keep

    def _schedule_micro(self, state: S, moving_w: float, now: float) -> None:
        from . import actions as act
        if state in (S.SLEEPING, S.HELD, S.FALLING, S.PRESSING):
            return
        sp = self.species
        rate = 0.4 + 1.2 * self.idle_activity
        table = []
        if sp in ("cat", "dog", "fox", "rabbit"):
            table.append(("ear", act.p_ear_twitch, 0.45, (3.5, 11.0)))
        if sp in ("cat", "fox", "dog"):
            table.append(("tail", act.p_tail_swish, 1.2, (6.0, 16.0)))
        if sp in ("cat", "dog", "fox", "rabbit"):
            table.append(("tilt", _p_head_tilt, 1.5, (14.0, 34.0)))
            table.append(("flick", _p_ear_flick, 0.25, (6.0, 15.0)))
        if sp == "rabbit":
            table.append(("nose", act.p_nose_twitch, 0.8, (1.5, 4.5)))
        if sp == "robot":
            table.append(("servo", _p_servo_tick, 0.35, (4.0, 10.0)))
        if sp == "buddy":
            table.append(("glance", _p_glance, 0.9, (5.0, 12.0)))
        for key, fn, dur, (lo, hi) in table:
            nxt = self._micro_next.get(key)
            if nxt is None:
                self._micro_next[key] = now + self._rng.uniform(lo, hi) / rate
                continue
            if now >= nxt:
                self._micro_next[key] = now + self._rng.uniform(lo, hi) / rate
                if moving_w > 0.5 and key in ("nose", "glance", "tilt"):
                    continue
                self._micro.append({"fn": fn, "t0": now, "dur": dur, "key": key})
                if key == "tail":
                    self.events.append(("TAIL_SWING", "micro"))

    def _rig_layer(self, out: Pose, loco, dt: float, now: float, moving_w: float) -> None:
        rig = self.rig
        lay = rig.layout(out)
        neutral = rig.neutral_feet(out, lay)
        reach = rig.stance_reach(out, lay)
        if loco is not None:
            units = UNITS_PER_PX_AT_DEFAULT / max(1e-6, getattr(loco, "scale", 1.0))
            odo = getattr(loco, "odometer", 0.0) * units
            stride_u = getattr(loco, "stride", self.profile.stride_walk) * UNITS_PER_PX_AT_DEFAULT
            cycle = loco.phase / (2 * math.pi)
            level = getattr(loco, "gait_level", 2.0 * loco.run_blend)
            facing = 1 if loco.facing_blend >= 0 else -1
            # the hop gait's own flight is part of the stride (the feet keep swinging); only real
            # jumps / falls count as airborne here (the gait lift used to freeze every paw in place)
            body_lift = out.lift - (getattr(out, "gait_lift", 0.0) if self.profile.gait == "hop" else 0.0)
            airborne = bool(loco.airborne) or body_lift > 4.0 or out.hop > 0.5
            out.ground = getattr(loco, "ground_dy", 0.0) * units
        else:
            odo, stride_u, cycle, level, facing = self._odo_cache if hasattr(self, "_odo_cache") else (0.0, 30.0, 0.0, 0.0, 1)
            airborne = out.lift > 4.0 or out.hop > 0.5
        self._odo_cache = (odo, stride_u, cycle, level, facing)
        # reach guard: a stance leg must be able to cover its part of the stride. When the gait's
        # bob / lean lifts the hips so high that the legs are nearly straight, the feet could only
        # touch the ground for an instant (floating paws, legs that barely move). The body then sinks
        # just enough for the legs to bend and carry the step - smoothly, and only while moving.
        if self.rig.sk.kind != "biped" and self.profile.gait != "hop":
            g = self.feet.gait
            want = min(0.5 * stride_u * g.duty / 0.95, 0.8 * min(rig.leg_reach(k_) for k_ in reach))
            short = 0.0
            if moving_w > 0.05 and not airborne:
                for k_, (hx, hy) in rig.hips(out, lay).items():
                    r = rig.leg_reach(k_) * 0.97
                    h = max(0.0, -rig.sk.ankle - hy)
                    h_max = math.sqrt(max(0.0, r * r - want * want))
                    short = max(short, h - h_max)
                short = min(short, 0.22 * rig.leg_reach(next(iter(reach)))) * moving_w
            tau = 0.12 if short > self._sink else 0.3
            self._sink += (short - self._sink) * (1 - math.exp(-max(dt, 1e-4) / tau))
            if self._sink > 0.02:
                out.bob += self._sink
                out.head_dy -= self._sink * 0.5          # the head stays nearer its height
                lay = rig.layout(out)
                neutral = rig.neutral_feet(out, lay)
                reach = rig.stance_reach(out, lay)
        frame = self.feet.update(dt, now, cycle, level, stride_u, odo, moving_w, neutral, reach, facing, airborne)
        gl = getattr(out, "gait_lift", 0.0)
        if self.profile.gait == "hop" and gl > 0.05 and not airborne:
            # in the hop's flight the paws travel with the body (legs reach / trail, they don't
            # stretch down to the floor); planted paws stay on the ground
            for f_, (tx, ty) in list(frame.targets.items()):
                if not frame.contacts.get(f_, False):
                    frame.targets[f_] = (tx, ty - gl * 0.85)
        for ev, foot in frame.events:
            self.events.append((ev, foot))
        self._support = frame.support
        self._support_mean += (frame.support - self._support_mean) * (1 - math.exp(-dt / 0.6))
        self.foot_frame = frame
        if moving_w > 0.05:
            self.phase_name = f"{frame.gait.name if frame.gait else 'WALK'}:{frame.cycle_phase}"
        sk = rig.sk
        if sk.tail.lengths:
            base = lay["tilt"] + sk.tail.rest[0] - out.tail
            out.tail_chain = self.tail.update(base, dt, self.secondary_amount if not self.profile.mechanical else 0.2,
                                              self.physics ** 0.5)
        out.rig = rig.solve(out, frame.targets, frame.contacts, frame.phases, self._foot_pitch(frame, dt, out),
                            self.ik_targets or None)

    def _foot_pitch(self, frame, dt: float, out: Pose) -> dict:
        """Per-foot pitch from its gait phase, eased so feet roll instead of snapping."""
        kind = "mech" if self.profile.mechanical else ("biped" if self.rig.sk.kind == "biped" else "quad")
        table = FOOT_PITCH[kind]
        res = {}
        a = 1 - math.exp(-dt / 0.045)
        for name, ph in (frame.phases or {}).items():
            tgt = table.get(ph, 0.0) + (out.foot_pitch if ph not in ("PLANTED", "CONTACT", "SETTLE") else 0.0)
            if out.sit > 0.5 or out.lie > 0.5:
                tgt = 0.0
            cur = self._foot_ang.get(name, 0.0)
            cur += (tgt - cur) * a
            self._foot_ang[name] = cur
            res[name] = cur
        return res

    # ------------------------------------------------------------ layers
    def _gait_layer(self, p: Pose, state: S, loco, now: float, k: float) -> float:
        prof = self.profile
        sp = loco.speed
        walk = max(1.0, loco.walk_speed)
        moving_w = smoothstep(sp / (0.3 * walk)) if not loco.airborne else 0.0
        rb = loco.run_blend
        level = getattr(loco, "gait_level", 2.0 * rb)
        phi = loco.phase
        stride_units = getattr(loco, "stride", prof.stride_walk + (prof.stride_run - prof.stride_walk) * rb) \
            * UNITS_PER_PX_AT_DEFAULT
        amp_deg = math.degrees(math.asin(min(0.8, stride_units / (4 * LEG_LEN))))
        p.gait_run = rb
        p.gait_level = level
        p.stride_u = stride_units
        p.moving = moving_w
        g = self.feet.gait
        if moving_w > 0.01:
            self.phase_name = ("SPRINT_CYCLE" if level > 2.5 else "RUN_CYCLE" if level > 1.5 else
                               "JOG_CYCLE" if level > 0.5 else "WALK_CYCLE")
            p.leg_phase = phi
            p.leg_amp = (amp_deg / 24.0) * moving_w
            p.foot_lift = (0.8 + 0.5 * rb) * moving_w
            if prof.gait == "hop":
                # the body rises only while no paw is planted (push-off -> arc -> front paws land):
                # it used to peak mid-stance, which stretched the planted legs straight and froze them
                f = hop_flight(g, phi / (2 * math.pi))
                arc = math.sin(math.pi * f) if f >= 0.0 else 0.0
                p.gait_lift = arc * (5 + 4 * level) * moving_w
                p.lift += p.gait_lift
                # nose up on the way up, nose down reaching for the landing, level on the ground
                p.tilt += (-math.sin(2 * math.pi * f) * (6 + 2 * level) if f >= 0.0 else 0.0) * moving_w * k
                if f < 0.0:      # on the ground: the hind legs gather under the body (a little squash)
                    gather = math.sin(math.pi * max(0.0, hop_stance(g, phi / (2 * math.pi))))
                    p.squash += 0.16 * gather * moving_w * self.physics
                    p.flex += -0.5 * gather * moving_w
                else:            # stretched out in the air
                    p.squash += (-0.06 - 0.03 * level) * arc * moving_w * self.physics
                    p.flex += 0.6 * arc * moving_w
            else:
                # body bob follows real foot support: vaulting walk, compressing run
                dev = self._support - self._support_mean
                bob = -g.vault * g.bob * dev * 1.8 * prof.body_bob * moving_w * k
                if level > 1.3:     # flight phase of the gallop / run
                    bob += -abs(math.sin(phi)) * (0.6 + 0.6 * (level - 1.3)) * prof.body_bob * moving_w
                if prof.gait == "mech":
                    bob *= 0.6
                p.bob += bob
                p.flex = g.flex * math.sin(phi) * moving_w
                p.head_dy += -bob * 0.55          # head stabilisation
                p.squash += max(0.0, dev) * 0.08 * min(1.0, level) * moving_w * self.physics
            acc_norm = max(-1.0, min(1.0, loco.accel / max(1.0, prof.acceleration)))
            p.tilt += (g.lean * prof.run_lean + acc_norm * 4) * moving_w * k
            # weight: hips roll and the chest counter-twists with every step; the head stays level
            mech = prof.mechanical
            biped = self.rig.sk.kind == "biped"
            roll_amp = (1.2 if mech else 4.5 if biped else 3.0) * (1.0 - 0.35 * min(1.0, level / 2.0))
            twist_amp = (1.5 if mech else 6.0 if biped else 3.0) * (0.7 + 0.3 * min(1.0, level))
            p.pelvis_roll += math.sin(phi) * roll_amp * moving_w * k
            p.torso_twist += -math.sin(phi) * twist_amp * moving_w * k
            p.head_tilt += -math.sin(phi) * roll_amp * 0.25 * moving_w   # head stabilisation (counter-roll)
            # natural variation: no two steps are identical (tiny, slow, deterministic noise)
            n = self._stride_seed + loco.odometer * 0.013
            p.tilt += (math.sin(n * 2.1) * 0.6 + math.sin(n * 5.3) * 0.3) * moving_w * (0.3 if mech else 1.0)
            # braking: lean back and sink as speed bleeds off quickly
            run_sp = max(1.0, loco.run_speed)
            if acc_norm < -0.3 and sp > walk * 0.7:
                br = smoothstep((-acc_norm - 0.3) / 0.5) * min(1.0, sp / run_sp)
                p.brake = max(p.brake, br)
                p.tilt -= 8.0 * br * k
                p.crouch_body += 0.3 * br
                p.ear_flop -= 6.0 * br
            if self.rig.sk.kind != "biped" and prof.gait != "hop" and level > 0.8:
                # running animals carry the tail out behind them (it streams in the air), not over the back
                p.tail += 125.0 * smoothstep((level - 0.8) / 1.6) * moving_w
            if level > 1.2 and p.mouth == "smile":
                p.mouth = "open"
            if level > 1.6 and not p.mark:
                p.mark = "speed"
            if level > 2.5:
                p.ear = min(p.ear, 0.2)
        else:
            if state not in (S.WALKING, S.RUNNING, S.CHASING):
                self.phase_name = f"{state.value}_POSE"
        antic = getattr(loco, "anticipation", 0.0)
        if antic > 0:
            p.tilt -= 7 * antic
            p.crouch_body += 0.45 * antic
            self.phase_name = "ANTICIPATION"
        if loco.airborne:
            self.phase_name = "JUMP_RISE" if loco.vy < 0 else "JUMP_FALL"
            p.hop = 1.0
            p.jump_vy = float(loco.vy)
            p.tilt += max(-16.0, min(16.0, loco.vy / 55.0))
            p.ear = min(p.ear, -0.2)
            p.mouth = "o"
            # stretch on the way up, slight squash anticipation on the way down
            p.squash += max(-0.3, min(0.12, loco.vy / 2400.0)) * self.physics
            style = prof.jump_style
            rising = loco.vy < 0
            if style == "agile":            # cat / fox: long stretched leap, paws reach for the landing
                p.stretch = max(p.stretch, 0.35 if rising else 0.1)
                p.foot_pitch += 14.0
            elif style == "high":           # rabbit: tight tuck, ears stream back
                p.ear_flop -= 18.0
                p.squash -= 0.06 * self.physics
            elif style == "bouncy":         # dog: front paws forward, happy open mouth
                p.paw = max(p.paw, 0.45)
                p.paw2 = max(p.paw2, 0.45)
                p.mouth = "open"
                p.tongue = max(p.tongue, 0.5)
            elif style == "mech":           # robot: stiff, upright, little arm pump
                p.tilt *= 0.25
                p.arms_up = max(p.arms_up, 0.35)
                p.squash *= 0.4
        if loco.crouch > 0:
            self.phase_name = "JUMP_ANTICIPATION"
            p.squash += 0.55 * loco.crouch * self.physics
            p.crouch = loco.crouch
            p.crouch_body += 0.6 * loco.crouch
        return moving_w

    def _emotion_layer(self, p: Pose, state: S, st, now: float, moving_w: float, k: float) -> None:
        prof = self.profile
        p.emotion = getattr(st, "name", None) or ""
        p.ear = max(-1.0, min(1.0, p.ear + st.ear * max(0.2, prof.ear_response)))
        p.eye_scale *= st.eye_scale
        p.eye_open *= st.eye_open
        p.brow += st.brow
        p.head_tilt += st.head_tilt
        if st.mouth and p.mouth == "smile":
            p.mouth = st.mouth
        if st.eye_style and p.eye_style == "normal":
            p.eye_style = st.eye_style
        amt = self.emotion_amount
        tr = getattr(st, "tremble", 0.0) * amt
        if tr > 0.01:                      # shiver: fast small jitter of body and head
            p.shift += math.sin(now * 53.0) * tr * 0.7
            p.head_dx += math.sin(now * 61.0 + 1.3) * tr * 0.5
        p.flush = max(p.flush, getattr(st, "flush", 0.0) * min(1.0, amt))
        # continuous face: composed from parameters so emotions blend smoothly
        fa = min(1.4, amt)
        p.smile += getattr(st, "smile", 0.0) * fa
        p.mouth_open += getattr(st, "mouth_open", 0.0) * fa
        p.mouth_width += getattr(st, "mouth_width", 0.0) * 0.5 * fa
        p.lid += getattr(st, "lid_upper", 0.0) * fa
        p.lid_low += max(getattr(st, "lid_lower", 0.0), getattr(st, "cheek", 0.0) * 0.5) * fa
        p.lid_tilt += getattr(st, "lid_tilt", 0.0) * fa
        p.pupil *= 1.0 + (getattr(st, "pupil", 1.0) - 1.0) * fa
        p.brow_raise += getattr(st, "brow_raise", 0.0) * fa
        p.slump += getattr(st, "slump", 0.0) * fa
        ch = getattr(st, "chest", 0.0) * fa
        if ch:                            # puffed up (proud / angry) or shrinking (scared)
            p.head_dy -= 2.0 * ch
            p.tilt -= 1.5 * ch
            p.puff = max(p.puff, 0.5 * max(0.0, ch))
        self._body_language(p, st, now, moving_w, fa)
        mk = getattr(st, "mark", None)
        if mk and not p.mark and amt > 0.3:
            p.mark = mk
        p.blush = max(p.blush, st.blush) if state not in (S.WARNING, S.ALERT) else p.blush
        wag = st.tail_amp * math.sin(now * 2 * math.pi * st.tail_hz) * prof.tail_swing * (1 - st.stillness)
        p.tail += wag * k
        calm = state in (S.IDLE, S.SITTING, S.WAVING, S.FOCUS_MODE, S.PETTED, S.LOOKING_AROUND)
        if st.bounce > 0 and calm and moving_w < 0.1:
            p.lift += st.bounce * abs(math.sin(math.pi * st.bounce_hz * now)) * k

    def _body_language(self, p: Pose, st, now: float, moving_w: float, amt: float) -> None:
        """Full-body pose from the emotion's pose-library entry: posture, head, shoulders, arms, tail,
        stance, sway. Locomotion keeps the upper-body part; stance / crouch fade while moving."""
        g = getattr
        still = 1.0 - 0.65 * moving_w
        mech = 0.6 if self.profile.mechanical else 1.0
        biped = self.rig.sk.kind == "biped"
        hunch = g(st, "hunch", 0.0) * amt
        p.tilt += g(st, "lean", 0.0) * amt * (0.8 if biped else 1.0)
        p.crouch_body += (g(st, "crouch", 0.0) + 0.25 * hunch) * amt * still
        p.head_dy += (g(st, "head_drop", 0.0) + 3.0 * hunch) * amt
        p.head_dx += (g(st, "head_fwd", 0.0) - 2.0 * hunch) * amt
        p.head_tilt += g(st, "head_pitch", 0.0) * amt
        p.head_roll += g(st, "head_roll", 0.0) * amt * mech
        p.shoulders += (g(st, "shoulders", 0.0) + 0.5 * hunch) * amt
        p.fists = max(p.fists, g(st, "fists", 0.0) * amt * still)
        p.akimbo = max(p.akimbo, g(st, "akimbo", 0.0) * amt * still)
        p.hug = max(p.hug, g(st, "hug", 0.0) * amt * still)
        p.arms_out = max(p.arms_out, g(st, "arms_out", 0.0) * amt * still)
        p.stance += g(st, "stance", 0.0) * amt * still
        p.tail_pose += g(st, "tail_pose", 0.0) * amt
        p.shift += g(st, "weight", 0.0) * 2.2 * amt * still
        asym = g(st, "ear_asym", 0.0) * amt
        if asym:
            p.ear_l += 22.0 * asym
            p.ear_r -= 22.0 * asym
        sway = g(st, "sway", 0.0) * amt
        if sway > 0.01:
            ph = now * 2 * math.pi * 0.23 + self._shift_seed
            p.shift += math.sin(ph) * 1.6 * sway * still
            p.tilt += math.sin(ph + 0.8) * 1.2 * sway
            p.head_roll += math.sin(ph * 0.9 + 1.7) * 4.0 * sway * mech
        # tail held high / low (quadrupeds): applied as an angle offset on the tail chain base
        tp = max(-1.0, min(1.0, p.tail_pose))
        p.tail += -40.0 * tp if tp > 0 else -55.0 * tp
        self._breath_style = (g(st, "breath_rate", 1.0), g(st, "breath_depth", 1.0))
        self._blink_speed = g(st, "blink_speed", 1.0)

    def _secondary(self, p: Pose, loco, dt: float, k: float) -> None:
        prof = self.profile
        sp = loco.speed if loco is not None else 0.0
        run = max(1.0, loco.run_speed) if loco is not None else 1.0
        acc = loco.accel if loco is not None else 0.0
        acc_norm = max(-1.0, min(1.0, acc / max(1.0, prof.acceleration)))
        amt = self.secondary_amount
        tail_t = (-sp / run * 22 - acc_norm * 10) * prof.tail_swing
        p.tail += self.tail_spring.update(tail_t, dt) * k * (0.5 + 0.5 * amt)
        ear_t = -sp / run * 26 * prof.ear_response - acc_norm * 6 * prof.ear_response
        p.ear_flop += self.ear_spring.update(ear_t, dt) * k * amt
        head_t = -acc_norm * 3.0 * prof.head_lag
        p.head_dx += self.head_spring.update(head_t, dt) * k * amt
        # body vertical velocity drives the dangly bits (antenna, hair, collar tag)
        body_y = p.bob - p.lift
        vy = (body_y - self._last_body_y) / max(1e-3, dt)
        self._last_body_y = body_y
        vy = max(-400.0, min(400.0, vy))
        self.antenna_spring.impulse(-vy * 0.02 * self.physics)
        p.antenna = self.antenna_spring.update(-acc_norm * 14 - sp / run * 10, dt) * amt
        self.hair_spring.impulse(-vy * 0.01 * self.physics)
        p.hair = self.hair_spring.update(-acc_norm * 8 - sp / run * 9, dt) * amt
        self.tag_spring.impulse(-acc_norm * 1.2 * self.physics)
        p.tag = self.tag_spring.update(0.0, dt) * amt
        facing = loco.facing_blend if loco is not None else self._facing_prev
        self._facing_prev = facing
        self.secondary.amount = amt
        self.secondary.stiff = self.physics
        sec = self.secondary.update(dt, acc_norm, vy, facing, p.tilt, min(1.0, sp / run))
        if "ear_l" in sec:
            p.ear_l += (sec["ear_l"] - sec.get("ear_r", 0.0)) * 0.5 * k        # asymmetry only (the
            p.ear_r += (sec.get("ear_r", 0.0) - sec["ear_l"]) * 0.5 * k        # common part is ear_flop)
        if "tail" in sec:
            p.tail += (sec["tail"] - self.secondary.bones["tail"].k_speed * min(1.0, sp / run)) * 0.35 * k
        if "hem" in sec:
            p.hem += sec["hem"]
        if "head" in sec:
            p.head_dy += sec["head"] * 0.12 * k
        s = self.settle.update(0.0, dt)
        p.tilt += s * 1.4 * k
        p.squash += -s * 0.05 * k * self.physics
        ld = self.land.update(0.0, dt)
        p.squash += max(-0.4, min(0.6, ld * 0.09)) * self.physics
        p.ear_flop += -abs(ld) * 1.5 * amt
        if ld > 0:                       # landing: legs bend, body compresses, head dips
            p.crouch_body += min(0.6, ld * 0.6) * self.physics
            p.head_dy += min(3.0, ld * 2.5)

    def _smooth(self, target: Pose, dt: float) -> Pose:
        if self._first:
            self._first = False
            return replace(target)
        cur = self._cur
        out = replace(target)
        s1 = self._s1
        for name in _NUMERIC:
            tau = TAU.get(name, 0.12) * self.smoothness
            if tau <= 0:
                continue
            a = getattr(cur, name)
            b = getattr(target, name)
            if name in EASED:
                # two cascaded filters (critically damped): the limb eases in and out instead of
                # leaving at full speed - a paw no longer snaps up when a gesture starts
                k = 1 - math.exp(-dt / (0.5 * tau))
                m = s1.get(name, a)
                m += (b - m) * k
                s1[name] = m
                setattr(out, name, a + (m - a) * k)
            else:
                setattr(out, name, a + (b - a) * (1 - math.exp(-dt / tau)))
        # a tail swings fast but never teleports: when an action ends and the tail's pose jumps
        # (curled round the body -> out behind) it sweeps over a few frames (wags stay well below this)
        lim = 650.0 * dt
        d_tail = out.tail - cur.tail
        if abs(d_tail) > lim:
            out.tail = cur.tail + math.copysign(lim, d_tail)
        # the body never drops faster than gravity would let it (no pop when a hop is cut short)
        out.lift = max(out.lift, cur.lift - 170.0 * dt) if target.lift < cur.lift else out.lift
        return out

    def _blink(self, p: Pose, now: float, emotion, state: S) -> None:
        prof = self.profile
        if p.eye_style not in ("normal", "wide", "focused"):
            return
        mult = emotion.blink_mult if emotion is not None else 1.0
        if now >= self._next_blink:
            self._blink_start = now
            self.events.append(("BLINK", ""))
            sleepy = mult > 1.6
            self._blink_dur = (0.3 if sleepy else self._rng.uniform(0.09, 0.14)) / max(0.3, getattr(self, "_blink_speed",
                                                                                                    1.0))
            lo, hi = prof.blink_interval
            self._next_blink = now + self._rng.uniform(lo, hi) / max(0.2, mult)
            if self._rng.random() < prof.double_blink:
                self._double_at = now + self._blink_dur + 0.12
            # now and then a lazy half blink, like real eyes do
            self._blink_depth = 0.55 if (not sleepy and self._rng.random() < 0.08) else 1.0
        if self._double_at is not None and now >= self._double_at:
            self._blink_start = now
            self._double_at = None
        bt = now - self._blink_start
        if 0 <= bt < self._blink_dur:
            # real lids snap shut (about a third of the blink) and open more slowly, easing out
            u = bt / self._blink_dur
            closed = _ease(u / 0.35) if u < 0.35 else 1.0 - _ease((u - 0.35) / 0.65)
            closed *= self._blink_depth
            p.eye_open = p.eye_open * (1 - closed) + 0.06 * closed

    # ------------------------------------------------------------ states
    def _celebration(self, p: Pose, t: float, now: float) -> None:
        style = self.celebration or self.profile.celebration
        p.mouth = "grin"
        p.eye_style = "happy"
        p.mark = "sparkle"
        p.blush = 1.0
        p.ear = 1.0
        if style == "spin":
            hop = abs(math.sin(t * 5.0))
            p.lift = hop * 12
            p.turn = math.cos(t * 2 * math.pi * 0.9)
            p.tail = math.sin(now * 12) * 30
        elif style == "wag":
            p.lift = abs(math.sin(t * 7.0)) * 9
            p.tail = math.sin(now * 22) * 40
            p.paw = 1.0 if int(t * 3) % 2 == 0 else 0.3
            p.squash = 0.2 if abs(math.sin(t * 7.0)) < 0.15 else -0.1
        elif style == "dance":
            sway = math.sin(t * 6.0)
            p.tilt = sway * 10
            p.lift = abs(math.cos(t * 6.0)) * 6
            p.arms_up = 0.5 + 0.5 * sway
            p.paw = 1.0
            p.paw_wave = sway * 25
            p.head_tilt = -sway * 10
        elif style == "flip":
            cyc = t % 1.3
            if cyc < 0.7:
                u = cyc / 0.7
                p.lift = 26 * math.sin(math.pi * u)
                p.roll = -360 * smoothstep(u)
            else:
                p.squash = 0.3 * _bump(cyc - 0.7, 0.3)
            p.tail = math.sin(now * 10) * 20
        elif style == "cheer":
            hop = abs(math.sin(t * 6.0))
            p.lift = hop * 14
            p.arms_up = 1.0
            p.paw = 1.0
            p.paw_wave = math.sin(t * 12) * 18
        else:  # jump (binky)
            hop = abs(math.sin(t * 6.0))
            p.lift = hop * 18
            p.squash = 0.4 if hop < 0.15 else -0.15
            p.arms_up = 1.0
            p.paw = 1.0
            p.paw_wave = math.sin(t * 12) * 18
            p.tail = math.sin(now * 12) * 30
            if hop > 0.8:
                p.tilt = math.sin(t * 3) * 12

    def _state_pose(self, st: S, t: float, now: float) -> Pose:
        breathe = self._breath_val
        p = Pose(mark_phase=now, gait=self.profile.gait)
        if st == S.IDLE or st == S.TURNING:
            p.bob = breathe * 0.8
            p.tail = math.sin(now * 1.4) * 10
            if now > self._ear_twitch_at:
                p.ear = 0.6
                if now > self._ear_twitch_at + 0.25:
                    self._ear_twitch_at = now + self._rng.uniform(3, 8)
        elif st in (S.WALKING, S.RUNNING, S.CHASING):
            p.tail = 0.0
            if st == S.CHASING:
                p.eye_style = "focused"
                p.ear = 0.8
        elif st == S.SITTING:
            p.sit = 1.0
            p.bob = breathe * 0.7
            p.tail = 85 + math.sin(now * 0.9) * 6
        elif st == S.SLEEPING:
            p.lie = 1.0
            p.bob = math.sin(now * 1.2) * 1.2
            p.eye_style = "closed"
            p.mouth = "sleep"
            p.ear = -0.7
            p.tail = 70
            p.mark = "zzz"
            p.blush = 0.8
        elif st == S.WAKING:
            u = smoothstep(t / 1.0)
            p.lie = 1 - u
            p.sit = 0.6 * u
            p.eye_open = 0.25 + 0.75 * u
            p.ear = -0.7 + 1.2 * u
            p.mouth = "o" if u < 0.6 else "smile"
        elif st == S.LOOKING_AROUND:
            s = math.sin(t * 1.7)
            p.look_x = s
            p.look_y = -0.3
            p.head_tilt = s * 9
            p.ear = 0.5
            p.tail = math.sin(now * 2.0) * 12
        elif st == S.STRETCHING:
            k = _bump(t, 2.2)
            p.stretch = k
            p.eye_open = 1 - 0.7 * k
            p.mouth = "o" if k > 0.6 else "smile"
            p.tail = 30 * k
            p.ear = -0.3 * k
            p.arms_up = k
        elif st == S.YAWNING:
            k = _bump(t, 1.7)
            p.sit = 0.6
            p.yawn = k
            p.mouth = "yawn" if k > 0.25 else "smile"
            p.eye_open = 1 - 0.9 * k
            p.head_tilt = -9 * k
            p.ear = -0.5 * k
        elif st == S.GROOMING:
            p.sit = 0.8
            p.groom = 1.0
            p.paw_wave = math.sin(t * 9) * 9
            p.head_tilt = 12 + math.sin(t * 9) * 3
            p.eye_style = "closed"
            p.mouth = "smile"
            p.tail = 80
        elif st == S.SCRATCHING:
            p.sit = 0.85
            p.scratch = 1.0
            p.head_tilt = 16
            p.eye_style = "happy"
            p.ear = -0.4
            p.tail = 80
        elif st == S.PEEKING:
            p.tilt = 14
            p.look_y = 0.9
            p.look_x = 0.7
            p.ear = 0.8
            p.paw = 0.35
            p.mouth = "o"
            p.tail = math.sin(now * 3) * 12 + 10
        elif st == S.READING:
            p.sit = 1.0
            p.read = 1.0
            p.look_y = 0.55
            p.look_x = 0.4 + 0.15 * math.sin(now * 1.3)
            p.head_tilt = 6
            p.bob = breathe * 0.5
            p.mark = ""
        elif st == S.PETTED:
            p.sit = 0.5
            p.eye_style = "happy"
            p.blush = 1.0
            p.mark = "heart"
            p.ear = -0.25
            p.tail = math.sin(now * 8) * 28
            p.head_tilt = math.sin(now * 2.5) * 10
            p.mouth = "smile"
        elif st == S.PLAYING:
            hop = max(0.0, math.sin(t * 5.5))
            p.lift = hop * 10
            p.paw = 1.0 if hop > 0.2 else 0.2
            p.stretch = 0.4 * (1 - hop)
            p.tail = math.sin(now * 9) * 25
            p.mouth = "grin"
            p.eye_style = "happy" if hop > 0.5 else "normal"
        elif st == S.WAVING:
            # animals sit up to wave; people (not the robot) wave standing
            p.sit = 0.0 if (self.rig.sk.kind == "biped" and self.profile.gait != "mech") else 0.6
            p.paw = 1.0
            p.paw_wave = math.sin(t * 11) * 22
            p.mouth = "grin"
            p.tail = math.sin(now * 6) * 18
            p.blush = 0.9
        elif st == S.WATCHING:
            p.ear = 1.0
            p.eye_open = 0.85
            p.mouth = "flat"
            p.tail = math.sin(now * 4.5) * 6 - 10
            p.mark = "dots"
            p.head_tilt = 4
        elif st == S.ALERT:
            p.lift = 12 * _bump(t, 0.35)
            p.ear = 1.0
            p.eye_style = "wide"
            p.mouth = "o"
            p.tail = -30
            p.mark = "!"
        elif st == S.WARNING:
            p.paw = 1.0
            p.paw_wave = math.sin(t * 3) * 6
            p.ear = 0.9
            p.mouth = "flat"
            p.brow = -0.4
            p.bob = abs(math.sin(t * 6)) * -1.0
            p.tail = math.sin(now * 5) * 10 - 15
            p.mark = "!"
        elif st == S.CLOSING_DISTRACTION:   # paw hovering next to the close button
            p.paw = 0.75
            p.paw_wave = math.sin(t * 8) * 3
            p.press = -0.1
            p.tilt = 4
            p.ear = 1.0
            p.brow = -0.5
            p.eye_style = "focused"
            p.mouth = "flat"
            p.bob = -abs(math.sin(t * 5)) * 0.8   # foot tap
            p.tail = math.sin(now * 6) * 14
        elif st == S.PRESSING:
            # raise (0-.30) wind-up (.30-.38) thrust (.38-.45) IMPACT hold (.45-.60) retract (.60-.85)
            if t < 0.30:
                p.paw = smoothstep(t / 0.30)
                p.press = 0.0
            elif t < 0.38:
                p.paw = 1.0
                p.press = -0.35 * smoothstep((t - 0.30) / 0.08)
            elif t < 0.45:
                u = (t - 0.38) / 0.07
                p.paw = 1.0
                p.press = -0.35 + 1.35 * u * u
            elif t < 0.60:
                p.paw = 1.0
                p.press = 1.0
                p.squash = 0.18
            else:
                u = smoothstep((t - 0.60) / 0.25)
                p.paw = 1.0 - 0.7 * u
                p.press = 1.0 - u
            if t >= 0.45:
                p.ripple = min(1.0, (t - 0.45) / 0.45)
            p.tilt = 6 if t < 0.6 else 3
            p.eye_style = "focused"
            p.mouth = "grin" if t >= 0.45 else "flat"
            p.ear = 1.0
            p.brow = -0.6 if t < 0.45 else 0.0
        elif st == S.CELEBRATING:
            self._celebration(p, t, now)
        elif st == S.CONFUSED:
            p.head_tilt = 16
            p.mouth = "w"
            p.ear = -0.3
            p.look_y = -0.6
            p.mark = "?"
            p.brow = 0.6
            p.tail = math.sin(now * 1.5) * 8
        elif st == S.FOCUS_MODE:
            p.sit = 1.0
            p.headband = True
            p.eye_style = "focused"
            p.mouth = "smile"
            p.bob = breathe * 0.6
            p.tail = 85 + math.sin(now * 0.8) * 5
            p.look_x = 0.3
            if self.profile.gait == "biped":
                p.read = 1.0
        elif st == S.HELD:
            p.leg_phase, p.leg_amp = now * 7, 0.5
            p.tilt = math.sin(now * 3.0) * 9
            p.mouth = "o"
            p.eye_style = "wide"
            p.ear = -0.5
            p.mark = "sweat"
            p.tail = math.sin(now * 8) * 25
        elif st == S.FALLING:
            p.mouth = "o"
            p.eye_style = "wide"
            p.ear = 1.0
            p.hop = 1.0
            p.squash = -0.3
        elif st == S.JUMPING:
            p.ear = -0.2
            p.mouth = "o"
            p.tail = -20
        elif st == S.LANDING:
            p.squash = 0.25 * _bump(t, 0.3)
        return p


def _clamp01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def _ease(v: float) -> float:
    v = _clamp01(v)
    return v * v * (3.0 - 2.0 * v)


def _p_servo_tick(p, u, t, now, sp):
    """Robot micro-movement: the head servo snaps to a new angle and back."""
    p.head_tilt += (5.0 if u < 0.5 else 0.0)
    p.look_x = 0.5 if u < 0.5 else p.look_x


def _p_glance(p, u, t, now, sp):
    p.look_x = math.sin(math.pi * u) * -0.8
    p.head_dx += math.sin(math.pi * u) * -1.5



def _p_head_tilt(p, u, t, now, sp):
    """Curious head tilt: the head cocks to one side, holds, comes back (the classic dog / cat 'huh?')."""
    side = 1.0 if int(now * 7.0) % 2 else -1.0
    k = math.sin(math.pi * u) ** 0.6 if 0 < u < 1 else 0.0
    p.head_tilt += 10.0 * side * k
    p.head_roll += 14.0 * side * k
    p.ear += 0.35 * k


def _p_ear_flick(p, u, t, now, sp):
    """One ear flicks on its own (a fly, a sound) - asymmetric, unlike the paired twitch."""
    k = math.sin(math.pi * min(1.0, t / 0.16)) if t < 0.16 else 0.0
    if int(now * 3.0) % 2:
        p.ear_l += 30.0 * k
    else:
        p.ear_r += 30.0 * k


# Backwards compatible name
Animator = AnimationController
