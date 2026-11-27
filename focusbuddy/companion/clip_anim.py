"""GLB clip animation: named clips, crossfades, layers, masks and root motion.

Used by ``skinning.RigDriver`` when a character asset ships skeletal animation clips (an
artist-made GLB with Idle / Walk / Run ...). The procedural rig still runs every frame and
provides what clips can't know about: ground contact and IK targets, body-language offsets,
look-at, secondary springs and every state that has no clip. Evaluation order per frame::

    base layer (locomotion blend by gait level, phase-synced to the feet | state clip | idle variation)
      -> crossfades (per-transition blend times)
      -> action layer (upper-body / full-body mask, weight from the ActionPlayer)
      -> root / world transform (root-motion policy)
      -> emotion offsets (lean, crouch, head, tail, ears)       } RigDriver._clip_post
      -> IK (feet / paws onto the ground, no skating)           }
      -> look-at (head + neck)                                   }
      -> secondary (tail / ear springs from the rig)             }
      -> blend with the procedural pose where no clip plays
      -> skinning (GPU or CPU)

metadata.json::

    "clips": {"idle": "Idle", "walk": "Walk", "run": "Run", "WAVE": "Wave", "sit": "Sitting"},
    "clip_settings": {"Walk": {"phase_offset": 0.0, "mask": "full", "loop": true, "speed": 1.0}},
    "transitions": {"default": 0.25, "idle->walk": 0.3, "walk->run": 0.2},
    "root_motion": "HYBRID"            # IN_PLACE | ROOT_MOTION | HYBRID

Clip keys are case-insensitive; common clip names (Idle, Walking, Run, Jump, Sitting, Wave,
Dance ...) are recognised without a mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .gltf import quat_to_mat, sample_channel

POLICIES = ("IN_PLACE", "ROOT_MOTION", "HYBRID")

# standard clip keys and the file names recognised for them
CLIP_KEYS = {
    "idle": ("idle", "idle_01", "breathing", "breathe", "stand_idle"),
    "idle_01": ("idle_01", "idle1", "idle_b", "survey", "lookaround"),
    "idle_02": ("idle_02", "idle2", "idle_c"),
    "idle_03": ("idle_03", "idle3", "idle_d"),
    "walk": ("walk", "walking", "walk_cycle", "walkcycle"),
    "walk_fast": ("walk_fast", "fast_walk", "brisk_walk"),
    "jog": ("jog", "jogging", "trot"),
    "run": ("run", "running", "run_cycle"),
    "sprint": ("sprint", "sprinting", "gallop"),
    "jump_start": ("jump_start", "jumpstart", "takeoff"),
    "jump_loop": ("jump_loop", "jumploop", "fall", "in_air"),
    "jump_land": ("jump_land", "land", "landing"),
    "jump": ("jump",),
    "sit": ("sit", "sitting", "sit_down"),
    "stand": ("stand", "standing", "stand_up"),
    "sleep": ("sleep", "sleeping", "lie", "lying"),
    "wake": ("wake", "wake_up", "waking"),
    "wave": ("wave", "waving"),
    "celebrate": ("celebrate", "dance", "victory", "cheer"),
    "angry": ("angry", "anger", "punch"),
    "sad": ("sad",),
    "scared": ("scared", "fear", "flinch"),
    "laugh": ("laugh", "laughing"),
    "yes": ("yes", "nod"),
    "no": ("no", "head_shake", "headshake"),
    "thumbs_up": ("thumbsup", "thumbs_up"),
    "turn_left": ("turn_left", "turnleft"),
    "turn_right": ("turn_right", "turnright"),
    "stop": ("stop",),
    "stretch": ("stretch",),
    "death": ("death", "die"),
    # motion-capture library (focusbuddy/companion/mocap.py): posture transitions and gestures
    "sit_down": ("sit_down", "sitdown"),
    "stand_up": ("stand_up", "standup", "get_up_from_floor"),
    "lie_down": ("lie_down", "liedown", "lay_down"),
}
# gait level of each locomotion clip (Pose.gait_level: 0 walk .. 1 jog .. 2 run .. 3 sprint)
LOCO_LEVEL = {"walk": 0.0, "walk_fast": 0.5, "jog": 1.0, "run": 2.0, "sprint": 3.0}
# behaviour state -> clip key (only used when the asset has that clip)
STATE_CLIPS = {"SITTING": "sit", "SLEEPING": "sleep", "WAKING": "wake", "CELEBRATING": "celebrate",
               "WAVING": "wave", "LOOKING_AROUND": "idle_01", "STRETCHING": "stretch"}
IDLE_STATES = {"IDLE", "WATCHING", "FOCUS_MODE", "LOOKING_AROUND", "WARNING", "PETTED", "CONFUSED", "THINKING",
               "WALKING", "RUNNING", "TURNING", "ALERT", ""}
# action name -> clip key
ACTION_CLIPS = {"WAVE": "wave", "WAVE_BIG": "wave", "HELLO": "wave", "DANCE": "celebrate", "CELEBRATE": "celebrate",
                "CHEER": "celebrate", "VICTORY_DANCE": "celebrate", "NOD": "yes", "YES": "yes", "HEAD_SHAKE": "no",
                "SHAKE_HEAD": "no", "THUMBS_UP": "thumbs_up", "ANGRY": "angry", "STOMP": "angry", "SAD": "sad",
                "CRY": "sad", "SCARED": "scared", "SURPRISED": "scared", "LAUGH": "laugh", "LAUGH_HARD": "laugh",
                "STRETCH": "stretch", "JUMP": "jump", "HIGH_JUMP": "jump", "SMALL_JUMP": "jump", "SIT": "sit",
                "STAND": "stand", "FALL_OVER": "death"}
UPPER_KEYS = {"wave", "yes", "no", "thumbs_up"}

# ---- motion capture (metadata "mocap": true): the clips carry the whole body in every calm state
MOCAP_STATES = {"SITTING": "sit", "READING": "sit", "FOCUS_MODE": "sit", "SLEEPING": "sleep",
                "STRETCHING": "stretch", "YAWNING": "yawn",
                "CELEBRATING": "celebrate", "CONFUSED": "shrug"}
# action -> motion-capture clip (gestures play as a layer over the idle / posture)
# (the hello WAVE is procedural for the people - skinning._human_gestures: the captured one raised a stiff
# arm and barely waved, and had to be squeezed to twice its speed to fit the action)
MOCAP_ACTIONS = {"GREET": "wave_big", "FAREWELL": "wave_big", "WAVE_BIG": "wave_big",
                 "BOW": "bow", "CELEBRATE": "celebrate", "CHEER": "celebrate", "VICTORY_DANCE": "dance",
                 "DANCE": "dance", "LAUGH": "laugh", "LAUGH_HARD": "laugh", "CRY": "cry", "SCARED": "scared",
                 "ANGRY": "angry", "TANTRUM": "angry", "STOMP": "angry", "SHRUG": "shrug",
                 "BORED": "fidget", "IMPATIENT": "fidget"}
# gestures whose clip must not move the feet (the base keeps standing / sitting)
MOCAP_UPPER = {"wave", "wave_m", "wave_big", "wave_big_m", "shrug"}
# a posture is entered and left through its own transition clips
ENTER = {"sit": "sit_down", "sleep": "lie_down"}
EXIT = {"sit": "stand_up", "sleep": "wake"}
POSTURE_FAMILY = {"sit": "sit", "sit_down": "sit", "stand_up": "sit",
                  "sleep": "sleep", "lie_down": "sleep", "wake": "sleep"}
# where lie_down passes through sitting on the floor: sit -> sleep joins it there (seconds)
SIT_TO_LIE_T = 1.6
LOCO_IDLE = {"", "loco", "idle", "idle_01", "idle_02", "idle_03"}


def _canon_key(name: str) -> str:
    return "".join(ch for ch in (name or "").lower() if ch.isalnum() or ch == "_")


@dataclass
class _Track:
    joint: int
    path: str
    channel: object


@dataclass
class ClipInfo:
    key: str
    clip: object
    tracks: list
    loop: bool = True
    phase_offset: float = 0.0
    mask: str = "full"
    speed: float = 1.0
    root_policy: str = "HYBRID"
    drift: np.ndarray = field(default_factory=lambda: np.zeros(3))     # root displacement per cycle (asset units)
    root_t0: Optional[np.ndarray] = None
    root_joint: int = -1                       # motion root: highest joint with a translation track
    morph: list = field(default_factory=list)
    stride: float = 0.0                        # locomotion: body travel per clip cycle (internal units), 0 = unknown
    bake: Optional[object] = None              # _Baked: tracks resampled on a fixed grid (built on first use)

    @property
    def duration(self) -> float:
        return max(1e-3, float(self.clip.duration or 0.0))


@dataclass
class Playing:
    info: ClipInfo
    t: float = 0.0
    weight: float = 0.0
    target: float = 1.0
    rate: float = 4.0                  # weight change per second


@dataclass
class ClipResult:
    weight: float                      # how much of the final pose comes from clips (0..1)
    rot: dict                          # joint -> local quaternion (asset space)
    trans: dict                        # joint -> local translation
    scale: dict                        # joint -> local scale
    morphs: dict                       # morph target -> weight
    root_delta: np.ndarray             # root motion this frame (asset units, ROOT_MOTION policy only)
    base: str = ""
    action: str = ""
    upper_only: bool = False
    proc_full: float = 0.0             # a procedural full-body action without a clip is playing (0..1)
    proc_upper: float = 0.0            # ... an upper-body one (arms / head / spine use the rig there)
    posture: bool = False              # a motion-capture clip owns the whole body (sit, sleep, stretch ...)


class _Ch:
    __slots__ = ("times", "values", "interpolation", "path")

    def __init__(self, times, values, interp, path):
        self.times, self.values, self.interpolation, self.path = times, values, interp, path


BAKE_HZ = 60.0


def _sample_vec(ch, ts: np.ndarray) -> np.ndarray:
    """Vectorised ``sample_channel`` for many times at once (rotation: normalised, nlerp between
    keys, which is indistinguishable from slerp at animation key spacing)."""
    times = np.asarray(ch.times, dtype=float)
    vals = np.asarray(ch.values, dtype=float)
    cubic = ch.interpolation == "CUBICSPLINE"
    if cubic:
        vals = vals.reshape(len(times), 3, -1)
    if len(times) <= 1:
        v = vals[0, 1] if cubic else vals[0]
        out = np.repeat(np.asarray(v, dtype=float)[None], len(ts), axis=0)
    else:
        tc = np.clip(ts, times[0], times[-1])
        i = np.clip(np.searchsorted(times, tc, side="right") - 1, 0, len(times) - 2)
        t0, t1 = times[i], times[i + 1]
        u = ((tc - t0) / np.maximum(t1 - t0, 1e-9))[:, None]
        if ch.interpolation == "STEP":
            out = vals[np.where(tc >= times[-1], len(times) - 1, i)]
        elif cubic:
            dt = (t1 - t0)[:, None]
            p0, m0 = vals[i, 1], vals[i, 2] * dt
            p1, m1 = vals[i + 1, 1], vals[i + 1, 0] * dt
            u2, u3 = u * u, u * u * u
            out = (2 * u3 - 3 * u2 + 1) * p0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * p1 + (u3 - u2) * m1
        elif ch.path == "rotation":
            a, b = vals[i], vals[i + 1]
            d = (a * b).sum(axis=1)
            b = np.where((d < 0)[:, None], -b, b)
            d = np.abs(d)
            th = np.arccos(np.clip(d, -1.0, 1.0))[:, None]
            sn = np.sin(th)
            near = (d > 0.9995)[:, None]
            sa = np.where(near, 1.0 - u, np.sin((1.0 - u) * th) / np.where(near, 1.0, sn))
            sb = np.where(near, u, np.sin(u * th) / np.where(near, 1.0, sn))
            out = a * sa + b * sb                                  # slerp
        else:
            out = vals[i] + (vals[i + 1] - vals[i]) * u
    if ch.path == "rotation":
        out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


class _Baked:
    """A clip's tracks resampled at BAKE_HZ (rotations retargeted and hemisphere-continuous)."""
    __slots__ = ("hz", "last", "ts", "rot_j", "R", "tr_j", "Tv", "sc_j", "Sv")


def _nlerp_acc(acc: Optional[np.ndarray], q: np.ndarray, w: float) -> np.ndarray:
    if acc is None:
        return q * w
    if float(np.dot(acc, q)) < 0:
        q = -q
    return acc + q * w


def _norm(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _qinv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]]) / max(1e-12, float(np.dot(q, q)))


class ClipAnimator:
    """Plays an asset's clips as layers. ``update(pose)`` returns a ``ClipResult`` or None."""

    def __init__(self, asset, seed: int = 11):
        self.asset = asset
        meta = asset.meta or {}
        self.policy = str(meta.get("root_motion", "HYBRID")).upper()
        if self.policy not in POLICIES:
            self.policy = "HYBRID"
        self.transitions = {str(k).lower(): float(v) for k, v in (meta.get("transitions") or {}).items()}
        self.default_blend = float(self.transitions.get("default", 0.25))
        self.joints = asset.joints
        self.rest = [(j.rest_r if j.rest_r is not None else np.array([0, 0, 0, 1.0]),
                      j.rest_t if j.rest_t is not None else np.zeros(3),
                      j.rest_s if j.rest_s is not None else np.ones(3)) for j in self.joints]
        self.root = self._root_joint()
        self.clips: dict[str, ClipInfo] = {}
        self._build(meta)
        self.loco = sorted(((LOCO_LEVEL[k], k) for k in LOCO_LEVEL if k in self.clips))
        for _, k in self.loco:
            try:
                self.clips[k].stride = float((meta.get("clip_settings") or {}).get(self.clips[k].clip.name, {})
                                             .get("stride", 0.0)) or self._measure_stride(self.clips[k])
            except Exception:  # noqa: BLE001 - an odd clip only loses stride matching
                self.clips[k].stride = 0.0
        self._loco_phase: Optional[float] = None
        self._last_leg_phase = 0.0
        self.base: Optional[Playing] = None
        self.fading: list[Playing] = []
        self.base_key = ""
        self.last_clock: Optional[float] = None
        self.loco_w = 0.0
        self.rng = np.random.default_rng(seed)
        self.idle_keys = [k for k in ("idle", "idle_01", "idle_02", "idle_03") if k in self.clips]
        self.last_root = None
        self.root_motion = np.zeros(3)
        self.enabled = True
        self.mocap = bool(meta.get("mocap"))
        self.chain: Optional[dict] = None          # transition clip playing: {"kind": in|out, "next": key}
        self._done_key = ""                        # a one-shot state clip that already played out
        self._jump: dict = {}

    # ------------------------------------------------------------ setup
    def _root_joint(self) -> int:
        for name in ("Root", "Pelvis"):
            for i, j in enumerate(self.joints):
                if j.canonical == name:
                    return i
        return 0 if self.joints else -1

    def _build(self, meta: dict) -> None:
        mapping = {str(k).lower(): v for k, v in (meta.get("clips") or {}).items()}
        settings = meta.get("clip_settings") or {}
        by_canon = {_canon_key(n): n for n in self.asset.clips}
        chosen: dict[str, str] = {}
        for key, names in CLIP_KEYS.items():
            if key in mapping and mapping[key] in self.asset.clips:
                chosen[key] = mapping[key]
                continue
            for nm in names:
                if nm in by_canon:
                    chosen[key] = by_canon[nm]
                    break
        # action-name mappings ("WAVE": "Wave") and custom keys
        for k, v in mapping.items():
            if v in self.asset.clips and k not in chosen:
                chosen[k] = v
        # per-model choice among the captured clips (metadata "mocap": {"walk": "walk_lady"})
        mc_meta = meta.get("mocap") if isinstance(meta.get("mocap"), dict) else {}
        for k, v in mc_meta.items():
            if isinstance(v, str) and v in self.asset.clips:
                chosen[k] = v
        # motion-capture clips are named by their key already ("wave_m", "cry" ...)
        used = set(chosen.values())
        for nm, clip in self.asset.clips.items():
            if getattr(clip, "mocap", None) is not None and nm not in used and nm not in chosen:
                chosen[nm] = nm
        for key, name in chosen.items():
            clip = self.asset.clips[name]
            st = settings.get(name, {}) or settings.get(key, {}) or {}
            tracks = []
            for (ji, path, times, values, interp) in clip.channels:
                tracks.append(_Track(ji, path, _Ch(times, values, interp, path)))
            mc = getattr(clip, "mocap", None)
            default_loop = bool(mc.get("loop")) if isinstance(mc, dict) else \
                key in ("idle", "idle_01", "idle_02", "idle_03", "walk", "walk_fast", "jog", "run", "sprint", "jump_loop", "sleep",
                        "celebrate")
            loop = bool(st.get("loop", default_loop))
            mask = st.get("mask") or (meta.get("clip_mask") or {}).get(name) or \
                ("upper" if key in UPPER_KEYS else "full")
            info = ClipInfo(key, clip, tracks, loop, float(st.get("phase_offset", 0.0)), mask,
                            float(st.get("speed", 1.0)), str(st.get("root_motion", self.policy)).upper())
            info.morph = [(names, _Ch(t, v, i, "weights")) for (names, t, v, i) in
                          getattr(clip, "morph_channels", [])]
            rt = sorted((tr for tr in tracks if tr.path == "translation"), key=lambda tr: self._depth(tr.joint))
            if rt:
                info.root_joint = rt[0].joint
                ch = rt[0].channel
                v0 = np.asarray(sample_channel(ch, 0.0), dtype=float)
                v1 = np.asarray(sample_channel(ch, info.duration), dtype=float)
                info.root_t0 = v0
                info.drift = v1 - v0
            self.clips[key] = info

    def _measure_stride(self, info: ClipInfo) -> float:
        """Body travel per cycle of an in-place locomotion clip, in internal units: the speed at which a
        planted foot slides backward under the body (stance phase) times the cycle length. Used to
        play the clip at the rate that keeps planted feet still on the ground."""
        C = getattr(self.asset, "_C", None)
        feet = [i for i, j in enumerate(self.joints) if j.canonical in ("LeftFoot", "RightFoot")]
        if C is None or not feet or not info.loop:
            return 0.0
        order = sorted(range(len(self.joints)), key=self._depth)
        n = 48
        dur = info.duration
        xs = {f: [] for f in feet}
        ys = {f: [] for f in feet}
        for k in range(n):
            acc_r, acc_t, acc_s = {}, {}, {}
            self._accumulate(info, dur * k / n, 1.0, acc_r, acc_t, acc_s, {})
            W = {}
            for i in order:
                j = self.joints[i]
                q = _norm(acc_r[i]) if i in acc_r else self.rest[i][0]
                t = acc_t[i][0] / max(acc_t[i][1], 1e-9) if i in acc_t else self.rest[i][1]
                sc = acc_s[i][0] / max(acc_s[i][1], 1e-9) if i in acc_s else self.rest[i][2]
                L = local_matrix(q, t, sc)
                p = j.parent
                W[i] = (W[p] if p >= 0 else (j.base if j.base is not None else np.eye(4))) @ L
            for f in feet:
                P = (C @ W[f])[:3, 3]
                xs[f].append(float(P[0]))          # internal +x = forward
                ys[f].append(float(P[1]))          # internal +y = down
        speeds = []
        dt = dur / n
        for f in feet:
            x, y = np.array(xs[f]), np.array(ys[f])
            if np.ptp(x) < 1e-6:
                continue
            ground = y >= y.max() - 0.2 * max(1e-6, np.ptp(y))      # lowest points (largest y) = stance
            for k in range(n):
                k2 = (k + 1) % n
                if ground[k] and ground[k2]:
                    speeds.append(-(x[k2] - x[k]) / dt)          # planted foot moves backward under the body
        if len(speeds) < 3:
            return 0.0
        v = float(np.median(speeds))
        return v * dur if v > 1e-3 else 0.0

    def _loco_cycle(self, pose, pick) -> float:
        """Clip cycle position (0..1). The rig's leg phase advances 2*pi per rig stride; the clip
        advances by the same distance measured in its own stride, so planted feet stay put."""
        lp = float(getattr(pose, "leg_phase", 0.0))
        rig_stride = float(getattr(pose, "stride_u", 0.0))
        if self._loco_phase is None:
            self._loco_phase = lp / (2 * math.pi)
            self._last_leg_phase = lp
        d = (lp - self._last_leg_phase + math.pi) % (2 * math.pi) - math.pi
        self._last_leg_phase = lp
        ws = sum(w for _, w in pick)
        clip_stride = sum(self.clips[k].stride * w for k, w in pick) / max(1e-6, ws)
        ratio = 1.0
        if rig_stride > 1e-6 and clip_stride > 1e-6 and all(self.clips[k].stride > 0 for k, _ in pick):
            ratio = max(0.2, min(5.0, rig_stride / clip_stride))
        self._loco_phase = (self._loco_phase + d / (2 * math.pi) * ratio) % 1.0
        return self._loco_phase

    def _depth(self, j: int) -> int:
        d = 0
        while j >= 0 and d < 64:
            j = self.joints[j].parent
            d += 1
        return d

    def has(self, key: str) -> bool:
        return key in self.clips

    @property
    def has_locomotion(self) -> bool:
        return bool(self.loco)

    @property
    def has_base(self) -> bool:
        """True when clips can carry the whole body (locomotion, idle or state clips)."""
        return bool(self.loco or self.idle_keys or any(k in self.clips for k in set(STATE_CLIPS.values())))

    def blend_time(self, a: str, b: str) -> float:
        return self.transitions.get(f"{a}->{b}", self.transitions.get(f"*->{b}",
                                    self.transitions.get(f"{a}->*", self.default_blend)))

    # ------------------------------------------------------------ per frame
    def _choose_mocap(self, pose) -> str:
        state = str(getattr(pose, "state", "") or "").upper()
        if state in ("HELD", "FALLING"):
            return ""                               # dangling / flailing: the procedural rig
        if state in ("JUMPING", "LANDING") and "jump" in self.clips:
            return "jump"
        if float(getattr(pose, "moving", 0.0)) > 0.12 and self.loco:
            return "loco"
        k = MOCAP_STATES.get(state)
        fam = POSTURE_FAMILY.get(self.base_key)
        if state == "YAWNING" and fam == "sit":
            k = "sit"                               # yawn where you sit (then lie back from there)
        elif k == "yawn" and k not in self.clips:
            k = "stretch"                           # an older clip library: the plain stretch
        if k and k in self.clips:
            if k == self._done_key:
                k = ""                              # one-shot already played: back to the idle
            else:
                return k
        if state == "WAKING" and fam == "sleep":
            return self.idle_keys[0] if self.idle_keys else ""
        if self.base_key in self.idle_keys:
            return self.base_key
        return self.idle_keys[0] if self.idle_keys else ""

    def _choose_base(self, pose) -> str:
        if self.mocap:
            return self._choose_mocap(pose)
        state = str(getattr(pose, "state", "") or "").upper()
        moving = float(getattr(pose, "moving", 0.0))
        if moving > 0.12 and self.loco:
            return "loco"
        k = STATE_CLIPS.get(state)
        if k and k in self.clips:
            return k
        if float(getattr(pose, "lie", 0.0)) > 0.5:
            return "sleep" if "sleep" in self.clips else ""
        if float(getattr(pose, "sit", 0.0)) > 0.5:
            return "sit" if "sit" in self.clips else ""
        if float(getattr(pose, "lift", 0.0)) > 6 and "jump_loop" in self.clips:
            return "jump_loop"
        if state in IDLE_STATES and self.idle_keys:
            if self.base_key in self.idle_keys:
                return self.base_key
            return self.idle_keys[0]
        return ""

    def _jump_time(self, pose, dt: float) -> float:
        """Where the captured jump is, from the physics: the crouch before takeoff, the flight
        (rise / fall by vertical speed, so any jump height fits) and the landing after."""
        info = self.clips["jump"]
        ev = (getattr(info.clip, "mocap", None) or {}).get("events") or {}
        t_off = float(ev.get("takeoff", info.duration * 0.45))
        t_land = float(ev.get("land", info.duration * 0.65))
        J = self._jump
        if J.get("t") is None:
            J.update(t=max(0.0, t_off - 0.32), vy0=0.0, air=False, landed=False)
        air = float(getattr(pose, "hop", 0.0)) > 0.5
        vy = float(getattr(pose, "jump_vy", 0.0))
        if air:
            J["air"] = True
            J["vy0"] = max(J["vy0"], abs(vy))
            v0 = max(1e-3, J["vy0"])
            u = 0.5 * (1.0 - min(1.0, abs(vy) / v0)) if vy < 0 else 0.5 + 0.5 * min(1.0, vy / v0)
            J["t"] = t_off + u * (t_land - t_off)
        elif J["air"]:
            J["t"] = max(J["t"], t_land) + dt                 # landed: absorb it
        else:
            J["t"] = min(t_off, J["t"] + dt * 2.2)            # quick crouch, then wait for the takeoff
        return min(J["t"], info.duration)

    def _start_chain(self, key: str, kind: str, nxt: str, t0: float = 0.0, blend: Optional[float] = None) -> bool:
        if key not in self.clips:
            return False
        self._switch(key, blend)
        self.base.t = t0
        self.chain = {"kind": kind, "next": nxt, "boost": 1.0}
        return True

    def _route(self, want: str, first: bool) -> None:
        """Pick the base clip, going through posture transitions (sit down / stand up, lie down /
        get up) instead of cross-fading straight between postures."""
        if want != self._done_key:
            self._done_key = ""
        ch = self.chain
        if ch is not None and self.base is not None:
            info = self.base.info
            urgent = want in ("", "jump") or (want == "loco" and ch["kind"] == "in")
            if urgent:
                self.chain = None
                self._switch(want, 0.2)
                return
            if ch["kind"] == "out":
                ch["next"] = want
                ch["boost"] = 2.4 if want == "loco" else 1.0       # hurry when it wants to walk away
            if self.base.t >= info.duration - 0.12:
                self.chain = None
                nxt = ch["next"]
                if ch["kind"] == "in" and want != nxt:
                    # changed its mind during the transition: leave the posture again
                    if not (nxt in EXIT and want != nxt and self._start_chain(EXIT[nxt], "out", want)):
                        self._switch(want)
                    return
                self._switch(nxt if ch["kind"] == "in" else want, 0.3)
            return
        if want == self.base_key:
            return
        cur = self.base_key
        if first:
            self._switch(want)
            return
        urgent = want in ("", "jump") or cur in ("", "jump")
        if not urgent and cur in EXIT and not (cur == "sit" and want == "sleep"):
            if self._start_chain(EXIT[cur], "out", want, blend=0.25):
                return
        if want in ENTER and not urgent:
            if cur == "sit" and want == "sleep":
                if self._start_chain(ENTER[want], "in", want, SIT_TO_LIE_T, blend=0.7):
                    return
            elif self._start_chain(ENTER[want], "in", want, blend=0.3):
                return
        self._switch(want)

    def _switch(self, key: str, blend: Optional[float] = None) -> None:
        if key == "jump":
            self._jump = {}
        bt = max(0.01, blend if blend is not None else self.blend_time(self.base_key or "none", key or "none"))
        if self.base is not None:
            self.base.target = 0.0
            self.base.rate = 1.0 / bt
            self.fading.append(self.base)
        if key:
            info = self.clips[key] if key != "loco" else ClipInfo("loco", None, [], True)
            self.base = Playing(info, 0.0, 0.0, 1.0, 1.0 / bt)
        else:
            self.base = None
        self.base_key = key

    def update(self, pose) -> Optional[ClipResult]:
        if not self.enabled or not self.clips:
            return None
        clock = float(getattr(pose, "clock", 0.0))
        first = self.last_clock is None
        dt = 0.0 if first else max(0.0, min(0.25, clock - self.last_clock))
        self.last_clock = clock
        want = self._choose_base(pose)
        if self.mocap:
            self._route(want, first)
            if first and self.base is not None:
                self.base.weight = 1.0
        elif want != self.base_key:
            self._switch(want)
            if first and self.base is not None:
                self.base.weight = 1.0                 # first frame: no crossfade from nothing
        # idle variations: at the end of a non-looping pass / every few loops pick another idle
        layers = ([self.base] if self.base is not None else []) + self.fading
        for pl in layers:
            diff = pl.target - pl.weight
            if diff:
                pl.weight += math.copysign(min(abs(diff), pl.rate * dt), diff)
            if pl.info.key != "loco":
                boost = self.chain.get("boost", 1.0) if (self.chain is not None and pl is self.base) else 1.0
                pl.t += dt * pl.info.speed * boost
        self.fading = [p for p in self.fading if p.weight > 1e-3]
        if self.mocap and self.chain is None and self.base is not None and not self.base.info.loop and \
                self.base.info.key not in ("loco", "jump") and self.base.t >= self.base.info.duration - 0.25:
            # a one-shot state clip (stretch, shrug ...) ends in the idle while the state lasts
            self._done_key = self.base.info.key
            self._switch(self.idle_keys[0] if self.idle_keys else "", 0.35)
        if self.base is not None and self.base.info.key in self.idle_keys and len(self.idle_keys) > 1 and \
                self.base.t > self.base.info.duration * 2.0:
            others = [k for k in self.idle_keys if k != self.base.info.key]
            self._switch(str(self.rng.choice(others)))
        rot: dict = {}
        trans: dict = {}
        scale: dict = {}
        morphs: dict = {}
        wsum = 0.0
        acc_r: dict = {}
        acc_t: dict = {}
        acc_s: dict = {}
        if self.base is not None and self.base.info.key == "jump":
            self.base.t = self._jump_time(pose, dt)
        for pl in ([self.base] if self.base is not None else []) + self.fading:
            if pl.weight <= 1e-4:
                continue
            if pl.info.key == "loco":
                for info, w, t in self._loco_parts(pose):
                    self._accumulate(info, t, pl.weight * w, acc_r, acc_t, acc_s, morphs)
            else:
                info = pl.info
                t = (pl.t % info.duration) if info.loop else min(pl.t, info.duration)
                self._accumulate(info, t, pl.weight, acc_r, acc_t, acc_s, morphs)
            wsum += pl.weight
        base_w = min(1.0, wsum)
        if wsum > 1e-6:
            for j, q in acc_r.items():
                rot[j] = _norm(q)
            for j, (v, w) in acc_t.items():
                trans[j] = v / max(w, 1e-9)
            for j, (v, w) in acc_s.items():
                scale[j] = v / max(w, 1e-9)
        # action layer
        act_name = str(getattr(pose, "action", "") or "").upper()
        if self.mocap:
            act_key = MOCAP_ACTIONS.get(act_name)
            if act_key and float(getattr(pose, "turn", 1.0)) < 0 and act_key + "_m" in self.clips:
                act_key += "_m"                      # gesture with the arm nearer the viewer
        else:
            act_key = ACTION_CLIPS.get(act_name)
            if act_key is None and act_name.lower() in self.clips:
                act_key = act_name.lower()
        act_w = float(getattr(pose, "action_w", 0.0))
        upper_only = False
        if act_key and act_key in self.clips and act_w > 1e-3:
            info = self.clips[act_key]
            t = float(getattr(pose, "action_t", 0.0)) * info.speed
            if self.mocap and not info.loop:
                # the captured gesture is fitted to the action's length (it starts and ends at rest)
                try:
                    from .actions import ACTIONS
                    ad = ACTIONS.get(act_name)
                    if ad is not None and ad.duration > 0.3:
                        t = float(getattr(pose, "action_t", 0.0)) / ad.duration * info.duration
                except Exception:  # noqa: BLE001
                    pass
            t = (t % info.duration) if info.loop else min(t, info.duration)
            if self.mocap and act_key in MOCAP_UPPER:
                info.mask = "upper"
            a_r, a_t, a_s = {}, {}, {}
            self._accumulate(info, t, 1.0, a_r, a_t, a_s, morphs, act_w)
            legs = self._leg_joints() if info.mask in ("upper", "arms") else set()
            upper_only = info.mask in ("upper", "arms")
            arms_only = upper_only and self.mocap and self.base_key not in LOCO_IDLE
            arms = self._arm_joints() if arms_only else None
            for j, q in a_r.items():
                if j in legs or j == self.root or (arms is not None and j not in arms):
                    continue
                q = _norm(q)
                base_q = rot.get(j, self.rest[j][0])
                rot[j] = self._slerp(base_q, q, act_w if base_w > 0 else 1.0)
            if not upper_only:
                for j, (v, w) in a_t.items():
                    base_t = trans.get(j, self.rest[j][1])
                    trans[j] = base_t + (v / max(w, 1e-9) - base_t) * act_w
            base_w = max(base_w, act_w if not upper_only else base_w)
            if upper_only and base_w <= 0:
                base_w = act_w
        proc_full = proc_upper = 0.0
        posture = self.mocap and self.base_key not in LOCO_IDLE
        if act_name and not (act_key and act_key in self.clips) and act_w > 1e-3 and not self.mocap:
            try:
                from .actions import ACTIONS
                ad = ACTIONS.get(act_name)
            except Exception:  # noqa: BLE001
                ad = None
            if ad is not None and ad.track == "full":
                proc_full = act_w            # e.g. STRETCH, GROOM, POUNCE: the rig performs it
            elif ad is not None and ad.track == "upper":
                proc_upper = act_w
        if base_w <= 1e-4 and not morphs:
            return None
        root_delta = self._root_policy(trans)
        return ClipResult(base_w, rot, trans, scale, morphs, root_delta, self.base_key, act_key or "", upper_only,
                          proc_full, proc_upper, posture)

    def _loco_parts(self, pose):
        """Locomotion clips around the current gait level, phase-synced to the rig's feet."""
        lvl = float(getattr(pose, "gait_level", 0.0))
        loco = self.loco
        if lvl <= loco[0][0]:
            pick = [(loco[0][1], 1.0)]
        elif lvl >= loco[-1][0]:
            pick = [(loco[-1][1], 1.0)]
        else:
            for (l0, k0), (l1, k1) in zip(loco, loco[1:]):
                if l0 <= lvl <= l1:
                    u = (lvl - l0) / max(1e-6, l1 - l0)
                    pick = [(k0, 1.0 - u), (k1, u)]
                    break
        phase = self._loco_cycle(pose, pick)
        out = []
        for k, w in pick:
            if w <= 1e-3:
                continue
            info = self.clips[k]
            u = (phase + info.phase_offset) % 1.0
            out.append((info, w, u * info.duration))
        return out

    def _arm_joints(self) -> set:
        """Both arms (upper arm down to the fingers): a gesture made while sitting or lying keeps the
        posture's own spine and head."""
        if getattr(self, "_arms", None) is None:
            out = {i for i, j in enumerate(self.joints) if any(t in (j.canonical or "") for t in
                                                                ("UpperArm", "LowerArm", "Hand"))}
            changed = True
            while changed:
                changed = False
                for i, j in enumerate(self.joints):
                    if i not in out and j.parent in out:
                        out.add(i)
                        changed = True
            self._arms = out
        return self._arms

    def _leg_joints(self) -> set:
        names = ("LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "RightUpperLeg", "RightLowerLeg", "RightFoot", "Pelvis",
                 "Root")
        out = {i for i, j in enumerate(self.joints) if j.canonical in names}
        # everything below the leg bones too (toes)
        changed = True
        while changed:
            changed = False
            for i, j in enumerate(self.joints):
                if i not in out and j.parent in out and j.canonical not in ("Spine", "Chest", "Tail0"):
                    if self.joints[j.parent].canonical not in ("Pelvis", "Root"):
                        out.add(i)
                        changed = True
        return out

    def _bake(self, info: ClipInfo) -> "_Baked":
        own = bool(getattr(info.clip, "own", False))
        rest_src = getattr(info.clip, "rest", {}) or {}
        n = max(1, int(math.ceil(info.duration * BAKE_HZ)))
        ts = np.minimum(np.arange(n + 1) / BAKE_HZ, info.duration)
        bk = _Baked()
        bk.hz, bk.last, bk.ts = BAKE_HZ, n, ts
        rj, R, tj, Tv, sj, Sv = [], [], [], [], [], []
        for tr in info.tracks:
            j = tr.joint
            if tr.path == "rotation":
                q = _sample_vec(tr.channel, ts)
                if not own and j in rest_src:
                    # rest-relative retarget: target rest * (source rest^-1 * source pose)
                    pre = _qmul(self.rest[j][0], _qinv(rest_src[j][0]))
                    q = np.array([_qmul(pre, qq) for qq in q])
                    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
                for k in range(1, len(q)):                 # one hemisphere along time (safe nlerp)
                    if float(np.dot(q[k], q[k - 1])) < 0:
                        q[k] = -q[k]
                rj.append(j)
                R.append(q)
            elif tr.path == "translation":
                if not own:
                    if j != info.root_joint:
                        continue                        # foreign clips: keep this skeleton's proportions
                v = _sample_vec(tr.channel, ts)
                if not own:
                    src = rest_src.get(j)
                    if src is not None:
                        v = self.rest[j][1] + (v - src[1])
                tj.append(j)
                Tv.append(v)
            elif tr.path == "scale" and own:
                sj.append(j)
                Sv.append(_sample_vec(tr.channel, ts))
        bk.rot_j, bk.R = rj, (np.stack(R, axis=1) if R else np.zeros((n + 1, 0, 4)))
        bk.tr_j, bk.Tv = tj, (np.stack(Tv, axis=1) if Tv else np.zeros((n + 1, 0, 3)))
        bk.sc_j, bk.Sv = sj, (np.stack(Sv, axis=1) if Sv else np.zeros((n + 1, 0, 3)))
        info.bake = bk
        return bk

    def _accumulate(self, info: ClipInfo, t: float, w: float, acc_r, acc_t, acc_s, morphs, morph_w: float = 1.0):
        bk = info.bake if info.bake is not None else self._bake(info)
        tc = min(max(t, 0.0), info.duration)
        i0 = min(int(tc * bk.hz), max(0, bk.last - 1))
        i1 = min(i0 + 1, bk.last)
        span = float(bk.ts[i1] - bk.ts[i0])
        a = min(1.0, max(0.0, (tc - float(bk.ts[i0])) / span)) if span > 1e-9 else 0.0
        if bk.rot_j:
            Q = bk.R[i0] + (bk.R[i1] - bk.R[i0]) * a
            Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
            for j, q in zip(bk.rot_j, Q):
                acc_r[j] = _nlerp_acc(acc_r.get(j), q, w)
        if bk.tr_j:
            V = bk.Tv[i0] + (bk.Tv[i1] - bk.Tv[i0]) * a
            for j, v in zip(bk.tr_j, V):
                if j == info.root_joint and info.root_t0 is not None:
                    v = self._root_shape(info, v, t)
                pv, pw = acc_t.get(j, (np.zeros(3), 0.0))
                acc_t[j] = (pv + v * w, pw + w)
        if bk.sc_j:
            S = bk.Sv[i0] + (bk.Sv[i1] - bk.Sv[i0]) * a
            for j, v in zip(bk.sc_j, S):
                pv, pw = acc_s.get(j, (np.zeros(3), 0.0))
                acc_s[j] = (pv + v * w, pw + w)
        for names, ch in info.morph:
            vals = np.asarray(sample_channel(ch, t), dtype=float).reshape(-1)
            for k, nm in enumerate(names):
                if k < len(vals):
                    morphs[nm] = max(morphs.get(nm, 0.0), float(vals[k]) * w * morph_w)

    def _root_shape(self, info: ClipInfo, v: np.ndarray, t: float) -> np.ndarray:
        """Root translation under the clip's policy (asset space: x / z horizontal, y up)."""
        pol = info.root_policy
        v = v.copy()
        rest = self.rest[info.root_joint][1]
        if pol == "IN_PLACE":
            v[0], v[2] = rest[0], rest[2]
        elif pol == "HYBRID":
            # the desktop movement owns world position: remove the per-cycle drift, keep the sway
            u = t / info.duration if info.loop else 0.0
            v = v - info.drift * u
            v[0] += rest[0] - info.root_t0[0]
            v[2] += rest[2] - info.root_t0[2]
        return v

    def _root_policy(self, trans: dict) -> np.ndarray:
        """ROOT_MOTION: report the root's horizontal movement this frame and keep it out of the pose
        (the caller moves the character in the world by that much)."""
        delta = np.zeros(3)
        info = self.base.info if (self.base is not None and self.base.info.key != "loco") else None
        r = info.root_joint if info is not None else -1
        if r < 0 or r not in trans:
            self.last_root = None
            return delta
        cur = trans[r]
        pol = info.root_policy
        if pol == "ROOT_MOTION":
            if self.last_root is not None:
                delta = cur - self.last_root
                delta[1] = 0.0
            self.last_root = cur.copy()
            rest = self.rest[r][1]
            trans[r] = np.array([rest[0], cur[1], rest[2]])
        else:
            self.last_root = None
        self.root_motion = delta
        return delta

    @staticmethod
    def _slerp(a, b, t):
        from .gltf import slerp
        return slerp(a, b, max(0.0, min(1.0, t)))


def local_matrix(q, t, s) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(q) * np.asarray(s, dtype=float)[None, :]
    m[:3, 3] = t
    return m
