"""Action library + layered, prioritised action player.

An *action* is a short, named piece of behaviour ("GROOM", "PLAY_BOW", "SCAN",
"THUMBS_UP" ...).  Each one declares

* ``track``     full (whole body), upper (head / arms / face only) or additive
                 (small offsets layered on top: ear twitch, tail swish, nod).
                 Several tracks play at once, which gives combinations such as
                 WALK + LOOK_AROUND + EAR_TWITCH + TAIL_SWING.
* ``state``     the CompanionState held while it plays (full track only)
* ``priority``  CRITICAL_REACTION > SPECIAL_ACTION > DISTRACTION_RESPONSE >
                 CELEBRATION > INTERACTION > LOCOMOTION > IDLE > MICRO_MOVEMENT
* blend in / out times, natural exit points (where it can be interrupted
  without looking broken), timed animation events, a cooldown, the species
  that can perform it (with substitutes for the others), an emotion, an energy
  cost and an optional physical motion (jump, hop, pounce ...).
* ``pose``      a function that shapes the Pose for this action.

The ``ActionPlayer`` keeps the active actions per track.  A new request with a
lower priority than what is playing is refused; an equal / higher priority
waits for the next exit point (at most ``MAX_WAIT`` s) unless it is CRITICAL.
Interrupted actions never reset - they blend out from wherever they are while
the new action blends in, so nothing pops.

Pure Python (no Qt) - unit tested.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional

from .states import CompanionState as S


class Priority(IntEnum):
    MICRO_MOVEMENT = 0
    IDLE = 1
    LOCOMOTION = 2
    INTERACTION = 3
    CELEBRATION = 4
    DISTRACTION_RESPONSE = 5
    SPECIAL_ACTION = 6
    CRITICAL_REACTION = 7


QUADS = frozenset({"cat", "dog", "fox", "rabbit"})
BIPEDS = frozenset({"robot", "buddy"})
ALL = None

# animation events
ACTION_START, ACTION_END = "ACTION_START", "ACTION_END"
CELEBRATION_PEAK, TAIL_SWING, SPARKS, BLINK = "CELEBRATION_PEAK", "TAIL_SWING", "SPARKS", "BLINK"

UPPER_FIELDS = frozenset({
    "look_x", "look_y", "head_tilt", "head_dx", "head_dy", "ear", "eye_open", "eye_scale", "brow", "paw",
    "paw_wave", "paw2", "arms_up", "point", "thumb", "clap", "chin", "write", "front", "tongue", "nose", "blush",
    "read", "scan", "mouth", "eye_style", "mark", "beckon"})


def _bump(t: float, d: float) -> float:
    return math.sin(math.pi * t / d) if 0 < t < d else 0.0


def _ss(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _env(u: float, a: float = 0.15, b: float = 0.15) -> float:
    """0 -> 1 -> 0 envelope over normalised time with soft edges."""
    return _ss(u / a) * _ss((1 - u) / b) if a > 0 and b > 0 else 1.0


PoseFn = Callable[..., None]   # fn(pose, u, t, now, species)


@dataclass(frozen=True)
class ActionDef:
    name: str
    category: str
    pose: Optional[PoseFn] = None
    track: str = "full"
    state: Optional[S] = None
    duration: float = 1.5
    loop: bool = False
    priority: Priority = Priority.IDLE
    blend_in: float = 0.25
    blend_out: float = 0.3
    exits: tuple = ()
    events: tuple = ()
    cooldown: float = 0.0
    species: Optional[frozenset] = None
    emotion: Optional[str] = None
    energy: float = 0.0            # cost (negative = restores)
    motion: Optional[str] = None   # jump | hop | double_hop | pounce | walk | jog | run | sprint | stop | turn
    tags: frozenset = frozenset()
    weight: float = 0.0            # base weight for the idle brain (0 = never chosen on its own)


# =========================================================================
#  pose recipes
# =========================================================================
def p_lie_awake(p, u, t, now, sp):
    p.lie = 1.0
    p.sit = 0.0
    p.eye_open = min(p.eye_open, 0.9)
    p.tail = 60 + math.sin(now * 1.1) * 8


def p_loaf(p, u, t, now, sp):
    p.lie = 0.75
    p.sit = 0.15
    p.tuck = 1.0
    p.eye_open = 0.55 + 0.1 * math.sin(now * 0.4)
    p.mouth = "smile"
    p.tail = 78
    p.blush = 0.8


def p_shake(p, u, t, now, sp):
    k = _env(u, 0.1, 0.25)
    p.roll = math.sin(t * 38) * 9 * k
    p.ear_flop += math.sin(t * 38 + 1) * 18 * k
    p.eye_style = "closed"
    p.tail = math.sin(t * 30) * 30 * k
    p.mouth = "w"


def p_sniff(p, u, t, now, sp):
    k = _env(u, 0.2, 0.2)
    p.tilt += 8 * k
    p.head_dy += 7 * k
    p.head_dx += 3 * k
    p.look_y = 0.9 * k
    p.nose = math.sin(t * 22) * k
    p.ear = 0.6


def p_jump_pose(p, u, t, now, sp):
    p.mouth = "open"
    p.ear = 0.5


def p_bounce(p, u, t, now, sp):
    hop = abs(math.sin(t * 7.0))
    p.lift = hop * 10 * _env(u, 0.05, 0.12)
    p.squash = 0.3 if hop < 0.12 else -0.12
    p.mouth = "grin"
    p.eye_style = "happy"
    p.ear = 0.8
    if sp == "rabbit":                       # binky: mid-air twist
        p.roll = math.sin(t * 7.0) * 14 * (hop > 0.5)


def p_spin(p, u, t, now, sp):
    p.turn = math.cos(2 * math.pi * _ss(u))
    p.lift = 6 * _bump(t, 0.6) + 6 * _bump(t - 0.6, 0.6)
    p.mouth = "grin"
    p.eye_style = "happy"
    p.tail = math.sin(now * 12) * 30


def p_dance(p, u, t, now, sp):
    sway = math.sin(t * 6.0)
    p.tilt = sway * 10
    p.lift = abs(math.cos(t * 6.0)) * 6
    p.arms_up = 0.5 + 0.5 * sway
    p.paw = 1.0
    p.paw2 = 0.5 - 0.5 * sway
    p.paw_wave = sway * 25
    p.head_tilt = -sway * 10
    p.mouth = "grin"
    p.eye_style = "happy"
    p.mark = "note"
    p.tail = sway * 35


def p_roll(p, u, t, now, sp):
    p.lie = 1.0
    p.roll = -360 * _ss((u - 0.15) / 0.7)
    p.lift = 4 * math.sin(math.pi * _ss((u - 0.15) / 0.7))
    p.eye_style = "happy"
    p.mouth = "grin"


def p_paw_at(p, u, t, now, sp):
    swipe = max(0.0, math.sin(t * 9))
    p.sit = 0.4
    p.paw = 0.5 + 0.5 * swipe
    p.paw_wave = -25 * swipe
    p.ear = 1.0
    p.eye_style = "focused"
    p.look_y = -0.4
    p.tail = math.sin(now * 6) * 20


def p_happy(p, u, t, now, sp):
    k = _env(u)
    p.lift += abs(math.sin(t * 6)) * 4 * k
    if sp in BIPEDS:                       # a happy little bounce with open arms
        p.arms_out = max(p.arms_out, 0.75 * k)
        p.head_tilt += 7 * math.sin(t * 5) * k
    p.eye_style = "happy"
    p.mouth = "grin"
    p.tail += math.sin(now * 14) * 30 * k
    p.ear = 0.8


def p_excited(p, u, t, now, sp):
    k = _env(u, 0.08, 0.15)
    p.lift += abs(math.sin(t * 9)) * 7 * k
    p.ear = 1.0
    p.eye_scale = 1.15
    p.mouth = "open"
    p.paw = 0.5 * k
    p.tail += math.sin(now * 20) * 35 * k
    p.mark = "sparkle"


def p_curious(p, u, t, now, sp):
    k = _env(u, 0.25, 0.25)
    p.head_tilt += 14 * k
    p.ear = 1.0
    p.look_y = -0.2
    p.mouth = "o"
    p.mark = "?"


def p_surprised(p, u, t, now, sp):
    """Head snaps back, body recoils (a little hop backwards), arms fly open, then settles."""
    snap = _bump(t, 0.32)
    rec = _bump(t - 0.05, 0.6)
    p.lift += 10 * snap
    p.tilt -= 11 * rec
    p.head_dx -= 4.5 * snap
    p.head_dy -= 3.0 * snap
    p.shift -= 2.5 * rec
    p.shoulders += 0.9 * rec
    p.arms_out = max(p.arms_out, 0.75 * rec if sp in BIPEDS else 0.0)
    p.paw = max(p.paw, 0.35 * rec if sp not in BIPEDS else p.paw)
    p.tail_pose += 0.9 * rec
    p.eye_style = "wide"
    p.eye_scale = 1.25 + 0.1 * snap
    p.mouth = "o"
    p.mouth_open = max(p.mouth_open, 0.6 * rec)
    p.ear = 1.0
    p.mark = "!"


def p_worried(p, u, t, now, sp):
    p.ear = -0.6
    p.brow = 0.8
    p.mouth = "frown"
    p.mark = "sweat"
    p.tail = 50
    p.head_tilt = -6


def p_annoyed(p, u, t, now, sp):
    p.brow = -0.8
    p.mouth = "flat"
    p.eye_open = 0.7
    p.tail += math.sin(t * 10) * 18
    p.ear = 0.2


def p_sleepy(p, u, t, now, sp):
    p.eye_open = 0.35 + 0.1 * math.sin(now * 0.8)
    p.ear = -0.5
    p.head_dy += 2
    p.mark = "zzz"


def p_proud(p, u, t, now, sp):
    p.tilt -= 6
    p.head_dy -= 3
    p.eye_style = "happy"
    p.mouth = "smile"
    p.ear = 1.0
    p.mark = "sparkle"
    p.tail = -25


def p_look_user(p, u, t, now, sp):
    p.front = 1.0
    p.look_x = 0.0
    p.look_y = 0.1
    p.mouth = "smile"


def p_nod(p, u, t, now, sp):
    k = _env(u, 0.1, 0.2)
    p.head_dy += math.sin(t * 11) * 3.5 * k
    p.head_tilt += math.sin(t * 11) * 4 * k


def p_shake_head(p, u, t, now, sp):
    k = _env(u, 0.1, 0.2)
    p.look_x = math.sin(t * 12) * k
    p.head_dx += math.sin(t * 12) * 3 * k
    p.front = 0.6
    p.mouth = "flat"


def p_point(p, u, t, now, sp):
    p.point = 1.0
    p.paw = 0.85 if sp not in BIPEDS else 0.0
    p.paw_wave = -30 if sp not in BIPEDS else 0.0
    p.ear = 1.0
    p.brow = -0.3
    p.mouth = "flat"


def p_beckon(p, u, t, now, sp):
    if sp in BIPEDS:
        p.beckon = _env(u, 0.15, 0.2)
        p.paw_wave = math.sin(t * 9) * 22             # the fingers / forearm curl in, "come here"
    else:
        p.paw = 0.8
        p.paw_wave = -10 + math.sin(t * 9) * 22
    p.front = 0.5
    p.mouth = "smile"


def p_cheer(p, u, t, now, sp):
    hop = abs(math.sin(t * 6.0))
    p.lift = hop * 12
    p.arms_up = 1.0
    p.paw = 1.0
    p.paw2 = 1.0
    p.paw_wave = math.sin(t * 12) * 18
    p.mouth = "grin"
    p.eye_style = "happy"
    p.mark = "sparkle"


def p_clap(p, u, t, now, sp):
    if sp in BIPEDS:
        p.clap = 1.0
    else:
        p.sit = 0.6
        p.paw = 0.7 + 0.3 * math.sin(t * 9)
        p.paw2 = 0.7 - 0.3 * math.sin(t * 9)
    p.mouth = "grin"
    p.eye_style = "happy"


def p_thumbs(p, u, t, now, sp):
    p.thumb = 1.0
    p.mouth = "grin"
    p.eye_style = "happy"
    p.front = 0.6
    p.head_tilt = 6


def p_tail_swish(p, u, t, now, sp):
    p.tail += math.sin(2 * math.pi * u * 2) * 26


def p_tail_wag(p, u, t, now, sp):
    p.tail += math.sin(t * 22) * 32 * _env(u, 0.1, 0.2)


def p_tail_swing(p, u, t, now, sp):
    p.tail += math.sin(2 * math.pi * u) * 30


def p_ear_twitch(p, u, t, now, sp):
    p.ear_flop += 22 * _bump(t, 0.12) - 10 * _bump(t - 0.14, 0.12)
    p.ear += 0.3 * _bump(t, 0.25)


def p_nose_twitch(p, u, t, now, sp):
    p.nose = math.sin(t * 30) * _env(u, 0.1, 0.1)


def p_lick_paw(p, u, t, now, sp):
    p.sit = 0.85
    p.groom = 1.0
    p.paw_wave = math.sin(t * 7) * 6
    p.head_tilt = 6 + math.sin(t * 7) * 3
    p.tongue = 0.6 + 0.4 * math.sin(t * 14)
    p.eye_style = "closed"
    p.tail = 80


def p_arch(p, u, t, now, sp):
    k = _env(u, 0.25, 0.3)
    p.flex = -1.4 * k
    p.lift += 3 * k
    p.tail = -40 * k
    p.ear = -0.4 * k
    p.eye_style = "wide"
    p.mouth = "o"


def p_look_up(p, u, t, now, sp):
    p.look_y = -1.0
    p.head_tilt -= 12
    p.head_dy -= 2
    p.ear = 1.0
    p.mouth = "o"


def p_play_bow(p, u, t, now, sp):
    p.stretch = 1.0
    p.tail += math.sin(now * 20) * 35
    p.lift += abs(math.sin(t * 5)) * 2.5
    p.eye_style = "happy"
    p.mouth = "open"
    p.ear = 0.6


def p_pant(p, u, t, now, sp):
    p.tongue = 1.0
    p.mouth = "open"
    p.bob += math.sin(t * 16) * 0.8
    p.breath = math.sin(t * 16)
    p.eye_style = "happy"


def p_give_paw(p, u, t, now, sp):
    p.sit = 0.9
    p.paw = 0.75
    p.paw_wave = math.sin(t * 3) * 4
    p.front = 0.7
    p.mouth = "smile"


def p_freeze(p, u, t, now, sp):
    p.ear = 1.0
    p.eye_style = "wide"
    p.eye_scale = 1.2
    p.mouth = "o"
    p.freeze = 1.0
    p.tail = -10


def p_sneak(p, u, t, now, sp):
    p.crouch_body = 0.6
    p.ear = 0.3
    p.eye_style = "focused"
    p.tail = -5


def p_alert(p, u, t, now, sp):
    p.ear = 1.0
    p.eye_style = "wide"
    p.mark = "!"


def p_stand_tall(p, u, t, now, sp):
    p.ear = 0.6


def p_boot(p, u, t, now, sp):
    k = _ss(u / 0.6)
    p.eye_open = 0.05 + 0.95 * k * (0.6 + 0.4 * (math.sin(t * 40) > -0.3))
    p.head_dy += (1 - k) * 4
    p.tilt += (1 - k) * 6
    p.mark = "dots" if u < 0.7 else ""
    p.mouth = "flat"


def p_scan(p, u, t, now, sp):
    p.scan = _env(u, 0.1, 0.1)
    p.look_x = math.sin(2 * math.pi * u)
    p.head_dx += math.sin(2 * math.pi * u) * 2
    p.eye_style = "focused"


def p_processing(p, u, t, now, sp):
    p.mark = "dots"
    p.head_tilt += 8 * (1 if int(t * 2.5) % 2 else -1)
    p.look_y = -0.4
    p.mouth = "flat"


def p_head_rot(p, u, t, now, sp):
    steps = int(u * 6)
    p.front = (0.0, 0.5, 1.0, 1.0, 0.5, 0.0)[min(5, steps)]
    p.look_x = (1.0, 0.4, 0.0, 0.0, -0.4, 1.0)[min(5, steps)]


def p_charge(p, u, t, now, sp):
    p.sit = 1.0
    p.mark = "battery_charge"
    p.eye_open = 0.5
    p.mouth = "smile"


def p_low_batt(p, u, t, now, sp):
    p.tilt += 9
    p.head_dy += 4
    p.eye_open = 0.4
    p.mark = "battery_low"
    p.mouth = "frown"
    p.arms_up = 0.0


def p_error(p, u, t, now, sp):
    p.glitch = _env(u, 0.05, 0.2)
    p.head_dx += (math.sin(t * 57) * 2.5) * p.glitch
    p.roll = math.sin(t * 43) * 3 * p.glitch
    p.mark = "!"
    p.eye_style = "wide"
    p.mouth = "w"


def p_recal(p, u, t, now, sp):
    seq = ((0.0, 0.0), (1.0, -0.6), (-1.0, -0.6), (1.0, 0.6), (-1.0, 0.6), (0.0, 0.0))
    i = min(5, int(u * 6))
    p.look_x, p.look_y = seq[i]
    p.head_tilt += (8, -8, 8, -8, 0, 0)[i]
    p.mark = "dots" if u < 0.85 else ""


def p_laser(p, u, t, now, sp):
    p.scan = _env(u, 0.1, 0.1)
    p.eye_style = "focused"


def p_syscheck(p, u, t, now, sp):
    i = min(3, int(u * 4))
    p.paw = 1.0 if i == 0 else 0.0
    p.paw2 = 1.0 if i == 1 else 0.0
    p.lift += 3 if i == 2 else 0
    p.head_tilt += 10 if i == 3 else 0
    p.mark = "dots"


def p_think(p, u, t, now, sp):
    p.chin = 1.0
    p.look_y = -0.7
    p.look_x = 0.3
    p.head_tilt = 9
    p.mouth = "flat"
    p.mark = "dots"


def p_write(p, u, t, now, sp):
    p.sit = 1.0
    p.write = 1.0
    p.look_y = 0.6
    p.look_x = 0.5
    p.head_tilt = 7
    p.mouth = "smile"


def p_pounce(p, u, t, now, sp):
    if u < 0.55:      # crouch + butt wiggle before the leap
        p.crouch_body = 0.9
        p.roll = math.sin(t * 24) * 2.5 * _ss(u / 0.2)
        p.eye_style = "focused"
        p.ear = 1.0
        p.tail = math.sin(now * 14) * 18
    else:
        p.mouth = "open"
        p.paw = 1.0
        p.paw2 = 1.0


def p_stop(p, u, t, now, sp):
    p.tilt += 5 * _bump(t, 0.3)


def p_reading(p, u, t, now, sp):
    p.read = 1.0
    p.sit = 1.0


# ------------------------------------------------------------ emotion pack
SHOUT, STOMP, SIGH = "SHOUT", "STOMP", "SIGH"


def _lerp(a, b, k):
    return a + (b - a) * k


def p_shout(p, u, t, now, sp):
    """A real shout: anticipation (lean back, crouch, breathe in - the chest expands) -> head forward,
    mouth opens, arms / paws gesture, the whole body lunges -> "HEY!" -> recoil -> settle."""
    inhale = _ss(u / 0.24) * (1 - _ss((u - 0.3) / 0.12))             # chest expansion
    lunge = _ss((u - 0.22) / 0.1) * (1 - _ss((u - 0.62) / 0.16))      # forward drive
    recoil = _bump(u - 0.6, 0.25)                                       # rebound after the shout
    settle = _ss((u - 0.8) / 0.2)
    p.puff = max(p.puff, inhale)
    p.tilt += -9 * inhale * (1 - lunge) + 13 * lunge - 4 * recoil
    p.crouch_body += 0.35 * inhale * (1 - lunge) + 0.15 * lunge
    p.shift += 2.5 * lunge - 1.5 * recoil
    p.head_dx += -2.5 * inhale * (1 - lunge) + 5.0 * lunge
    p.head_dy += -1.5 * inhale + 1.0 * lunge
    p.head_tilt += 6 * lunge
    p.shoulders += 0.8 * inhale + 0.4 * lunge
    p.stance += 0.8 * lunge
    if sp in BIPEDS:
        p.arms_out = max(p.arms_out, 0.8 * lunge)
        p.fists = max(p.fists, inhale * 0.8)
    else:
        p.paw = max(p.paw, 0.45 * lunge)
    shouting = lunge > 0.4
    p.mouth = "shout" if shouting else ("o" if inhale > 0.3 else "flat")
    p.mouth_open = max(p.mouth_open, lunge)
    p.eye_style = "angry"
    p.brow = -1.0
    p.ear = -0.6 * lunge + 0.5 * inhale
    p.tail_pose += 0.7 * lunge
    p.flush = 0.6 * (1 - settle)
    if shouting:
        p.mark = "shout"
        p.roll = math.sin(t * 40) * 1.2 * lunge


def p_angry(p, u, t, now, sp):
    """Stomp, stomp, stomp: steam, anger vein, gritted teeth, lashing tail."""
    beat = abs(math.sin(t * 7.5))
    p.stomp = beat
    p.squash += 0.12 * (1 - beat)
    p.bob += (1 - beat) * 0.8
    p.eye_style = "angry"
    p.brow = -1.0
    p.mouth = "teeth"
    p.flush = 0.9
    p.mark = "steam" if (int(t * 1.5) % 2) else "anger"
    p.tail += math.sin(t * 16) * 35
    p.ear = -0.4
    if sp in BIPEDS:
        p.cross = 0.0
        p.arms_up = 0.25 * beat


def p_grumpy(p, u, t, now, sp):
    p.cross = 1.0
    p.sit = 0.0 if sp in BIPEDS else 0.6
    p.front = 0.0
    p.look_x = -0.8 + 0.6 * _bump(u, 1.0) * (u > 0.6)
    p.head_tilt -= 8
    p.mouth = "pout"
    p.eye_style = "half"
    p.brow = -0.7
    p.flush = 0.35
    p.tail += math.sin(t * 6) * 14
    if 0.45 < u < 0.7:
        p.mark = "steam"


def p_tantrum(p, u, t, now, sp):
    k = _env(u, 0.08, 0.2)
    if sp in BIPEDS:
        p.stomp = abs(math.sin(t * 10)) * k
        p.arms_up = (0.5 + 0.5 * math.sin(t * 12)) * k
        p.paw_wave = math.sin(t * 14) * 30
    else:
        p.lie = 0.85 * k
        p.roll = math.sin(t * 9) * 22 * k
        p.paw = (0.5 + 0.5 * math.sin(t * 14)) * k
        p.paw2 = (0.5 - 0.5 * math.sin(t * 14)) * k
        p.paw_wave = math.sin(t * 16) * 30
    p.eye_style = "squeeze"
    p.mouth = "shout"
    p.mark = "anger"
    p.flush = 0.8
    p.tail += math.sin(t * 20) * 40


def p_sulk(p, u, t, now, sp):
    glance = 0.6 < u < 0.75
    p.sit = 0.9 if sp not in BIPEDS else 0.7
    p.back = 1.0 if not glance else 0.35            # turns its back on you ... then peeks over its shoulder
    p.ear = -0.8
    p.head_tilt -= 10
    p.mouth = "pout"
    p.eye_style = "half"
    p.mark = "gloom" if not glance else ""
    p.tail = 70


def p_scared(p, u, t, now, sp):
    p.lift += 10 * _bump(t, 0.3)
    p.tilt -= 10 * _bump(t, 0.5)
    k = _env(u, 0.1, 0.25)
    p.shift += math.sin(t * 58) * 0.9 * k
    p.crouch_body = 0.5 * k
    p.ear = -1.0
    p.eye_style = "shock"
    p.eye_scale = 1.25
    p.mouth = "wobble"
    p.mark = "sweat" if u > 0.3 else "!!"
    p.tail = 75
    p.brow = 0.9


def p_cry(p, u, t, now, sp):
    sob = max(0.0, math.sin(t * 7.0))
    p.sit = 0.85
    p.head_dy += 3 + sob * 1.2
    p.bob += sob * 0.8
    p.eye_style = "cry"
    p.mouth = "wobble" if sob < 0.6 else "open"
    p.ear = -1.0
    p.brow = 1.0
    p.mark = "gloom"
    p.cover = 0.5 if sp in BIPEDS and u > 0.5 else 0.0
    p.tail = 65


def p_embarrassed(p, u, t, now, sp):
    k = _env(u, 0.15, 0.25)
    p.cover = k
    p.flush = 1.0
    p.blush = 1.0
    p.look_y = 0.6
    p.head_tilt -= 10
    p.ear = -0.6
    p.mouth = "w"
    p.mark = "sweat"
    p.tail = 60 + math.sin(t * 5) * 8


def p_love(p, u, t, now, sp):
    sway = math.sin(t * 3.0)
    p.tilt += sway * 4
    p.head_tilt += sway * 10 + 6
    p.eye_style = "heart"
    p.mouth = "smile"
    p.blush = 1.0
    p.mark = "heart"
    p.tail += math.sin(t * 5) * 18
    p.paw = 0.25 * (sp not in BIPEDS)


def p_bored(p, u, t, now, sp):
    p.sit = 0.8 if sp not in BIPEDS else 0.0
    p.lie = 0.25 if sp not in BIPEDS else 0.0
    p.eye_style = "half"
    p.eye_open = 0.45
    p.mouth = "flat"
    p.head_tilt -= 6
    p.head_dy += 2
    if sp == "buddy":
        p.chin = 1.0
    if 0.35 < u < 0.6:
        p.mark = "sigh"
        p.bob += 1.2 * _bump(u - 0.35, 0.25)
    p.tail = 70 + math.sin(t * 1.2) * 4


def p_smug(p, u, t, now, sp):
    p.tilt -= 5
    p.head_dy -= 2
    p.head_tilt -= 6
    p.eye_style = "half"
    p.mouth = "smirk"
    p.blush = 0.8
    p.mark = "sparkle" if u > 0.5 else ""
    p.cross = 1.0 if sp in BIPEDS else 0.0
    p.tail += math.sin(t * 2) * 10


def p_disgusted(p, u, t, now, sp):
    k = _env(u, 0.12, 0.25)
    p.tilt -= 8 * k
    p.head_dx -= 3 * k
    p.head_tilt -= 8 * k
    p.shift += math.sin(t * 40) * 0.5 * _bump(t, 0.4)
    p.eye_style = "half"
    p.mouth = "tongue"
    p.brow = -0.4
    p.ear = -0.7
    p.tail = 40


def p_shocked(p, u, t, now, sp):
    p.lift += 14 * _bump(t, 0.35)
    p.squash -= 0.25 * _bump(t, 0.35)
    p.eye_style = "shock"
    p.eye_scale = 1.35
    p.mouth = "shout"
    p.mark = "!!"
    p.ear = 1.0
    p.brow = 0.7
    p.tail = -40
    p.arms_up = 0.8 * _bump(t, 0.8) if sp in BIPEDS else 0.0


def p_laugh(p, u, t, now, sp):
    shake = math.sin(t * 22)
    k = _env(u, 0.08, 0.2)
    p.squash += 0.08 * shake * k
    p.bob += shake * 0.9 * k
    p.tilt -= 8 * k
    p.head_tilt -= 8 * k
    p.eye_style = "squeeze" if int(t * 2) % 3 == 0 else "happy"
    p.mouth = "laugh"
    p.mark = "haha"
    p.blush = 1.0
    p.belly = k if sp in BIPEDS else 0.0
    p.sit = 0.0 if sp in BIPEDS else 0.5
    p.tail += math.sin(t * 18) * 30


def p_determined(p, u, t, now, sp):
    k = _env(u, 0.1, 0.25)
    p.crouch_body = 0.35 * k
    p.tilt += 6 * k
    p.eye_style = "focused"
    p.brow = -0.9
    p.mouth = "flat" if u < 0.5 else "grin"
    p.headband = True
    p.mark = "sparkle" if u > 0.5 else ""
    p.ear = 1.0
    if sp in BIPEDS:
        pump = _bump(u - 0.45, 0.35)
        p.paw = pump
        p.arms_up = pump * 0.0
        p.point = 0.0
        p.paw_wave = -40 * pump
    else:
        p.paw = 0.6 * _bump(u - 0.4, 0.4)


def p_dizzy(p, u, t, now, sp):
    k = _env(u, 0.1, 0.3)
    p.tilt += math.sin(t * 4.5) * 9 * k
    p.head_dx += math.cos(t * 4.5) * 3 * k
    p.head_tilt += math.sin(t * 4.5) * 12 * k
    p.shift += math.sin(t * 4.5) * 1.2 * k
    p.eye_style = "spiral"
    p.mouth = "w"
    p.mark = "stars"
    p.ear = -0.4


def p_relieved(p, u, t, now, sp):
    k = _bump(u, 1.0)
    p.bob += 2.0 * k
    p.tilt -= 3 * k
    p.eye_style = "closed"
    p.mouth = "smile"
    p.mark = "sigh"
    p.ear = 0.2
    p.blush = 0.8


def p_huff(p, u, t, now, sp):
    p.mark = "steam"
    p.flush = max(p.flush, 0.4)
    p.head_dy += 1.5 * _bump(t, 0.3)




# ------------------------------------------------- living-character upgrade
def p_shout_soft(p, u, t, now, sp):
    """Level 1: a small "hey..." - lean in, brows down, no shake."""
    k = _env(u, 0.2, 0.3)
    p.tilt += 4 * k
    p.head_dx += 1.5 * k
    p.mouth = "open" if 0.25 < u < 0.75 else "flat"
    p.brow = -0.6
    p.mark = "!" if u > 0.25 else ""
    p.ear = 0.3


def p_shout_big(p, u, t, now, sp):
    """Level 3: hop, stomp, big "HEY!!" with steam - still a cartoon, never scary."""
    p_shout(p, u, t, now, sp)
    p.lift += 9 * _bump(u - 0.22, 0.18)
    p.stomp = max(p.stomp, _bump(u - 0.4, 0.15))
    p.flush = 1.0
    if u > 0.7:
        p.mark = "steam"
    if sp == "robot":
        p.glitch = max(p.glitch, 0.25 * _bump(u - 0.25, 0.5))


def p_stomp(p, u, t, now, sp):
    """Three quick, comic stomps."""
    beat = abs(math.sin(t * 8.0)) * _env(u, 0.08, 0.15)
    p.stomp = beat
    p.squash += 0.1 * (1 - beat)
    p.eye_style = "angry"
    p.brow = -0.9
    p.mouth = "pout"
    p.flush = 0.6
    p.mark = "anger"
    p.tail += math.sin(t * 14) * 25


def p_high_five(p, u, t, now, sp):
    """Raise a hand / paw high, SLAP at the peak (sparkle), bounce back."""
    k = _env(u, 0.25, 0.3)
    slap = _bump(u - 0.42, 0.16)
    p.paw = k
    p.paw_wave = -18 * k - 10 * slap
    p.lift += 4 * slap
    p.squash -= 0.08 * slap
    p.eye_style = "happy" if u > 0.45 else "normal"
    p.mouth = "grin"
    p.mark = "sparkle" if 0.42 < u < 0.75 else ""
    if sp not in BIPEDS:
        p.tilt -= 10 * k          # rear up a little to reach
        p.lift += 3 * k
    p.tail += math.sin(t * 12) * 18


def p_bow(p, u, t, now, sp):
    k = _env(u, 0.3, 0.35)
    if sp in BIPEDS:
        p.tilt += 24 * k
        p.head_dy += 2 * k
        p.cross = 0.5 * k
    else:
        p.stretch = 0.65 * k
        p.head_dy += 5 * k
    p.eye_style = "closed" if k > 0.5 else "normal"
    p.mouth = "smile"


def p_greet(p, u, t, now, sp):
    hop = _bump(u, 0.35)
    p.lift += 7 * hop
    p.paw = _env(u, 0.15, 0.2)
    p.paw_wave = math.sin(t * 11) * 22
    p.eye_style = "happy"
    p.mouth = "grin"
    p.blush = 1.0
    p.ear = 1.0
    p.tail += math.sin(t * 14) * 30
    p.front = max(p.front, 0.6)


def p_farewell(p, u, t, now, sp):
    k = _env(u, 0.2, 0.3)
    p.paw = k
    p.paw_wave = math.sin(t * 6) * 18
    p.mouth = "smile"
    p.brow = 0.3
    p.head_tilt += 6 * k
    p.front = max(p.front, 0.5)
    p.tail += math.sin(t * 5) * 12


def p_tumble(p, u, t, now, sp):
    """Forward somersault and pop back up."""
    k = _ss(u / 0.75)
    p.roll = -360 * k
    p.lift += 14 * math.sin(math.pi * min(1.0, u / 0.75))
    p.hop = 1.0 if u < 0.75 else 0.0
    p.eye_style = "squeeze" if u < 0.75 else "happy"
    p.mouth = "grin"
    p.squash += 0.25 * _bump(u - 0.75, 0.2)


def p_skip(p, u, t, now, sp):
    """Additive skip on top of a jog: a happy bounce every step."""
    p.lift += abs(math.sin(t * 7.0)) * 5
    p.eye_style = "happy"
    p.mouth = "grin"
    p.arms_up = 0.25 + 0.15 * math.sin(t * 7.0) if sp in BIPEDS else p.arms_up
    p.tail += math.sin(t * 14) * 15


def p_hide_face(p, u, t, now, sp):
    k = _env(u, 0.12, 0.25)
    peek = 0.65 < u < 0.82
    p.cover = k * (0.45 if peek else 1.0)
    p.crouch_body = 0.4 * k
    p.ear = -1.0
    p.tail = 75
    p.eye_style = "wide" if peek else "closed"
    p.mouth = "wobble"
    p.shift += math.sin(t * 50) * 0.4 * k


def p_tremble(p, u, t, now, sp):
    k = _env(u, 0.1, 0.25)
    p.shift += math.sin(t * 57) * 0.8 * k
    p.head_dx += math.sin(t * 63 + 1) * 0.5 * k
    p.crouch_body = 0.35 * k
    p.eye_style = "wide"
    p.mouth = "wobble"
    p.ear = -0.9
    p.brow = 0.9
    p.mark = "sweat"
    p.tail = 75


def p_panic(p, u, t, now, sp):
    """Additive on top of a short sprint: arms flail, eyes huge, sweat drops - comic, not distressing."""
    p.arms_up = 0.5 + 0.5 * math.sin(t * 18)
    p.paw = 0.5 + 0.5 * math.sin(t * 18 + 1.5) if sp not in BIPEDS else p.paw
    p.eye_style = "shock"
    p.mouth = "wobble"
    p.mark = "sweat"
    p.ear = -1.0


def p_stumble(p, u, t, now, sp):
    """Trip on nothing, wave the arms, catch the balance, look around ("nobody saw that")."""
    trip = _bump(u, 0.45)
    p.tilt += 24 * trip
    p.head_dy += 3 * trip
    p.lift += 3 * _bump(u - 0.05, 0.25)
    p.arms_up = max(p.arms_up, 0.8 * trip)
    p.paw = max(p.paw, 0.6 * trip) if sp not in BIPEDS else p.paw
    p.eye_style = "wide" if u < 0.55 else "normal"
    p.mouth = "o" if u < 0.55 else "w"
    p.mark = "!" if u < 0.4 else ("sweat" if u > 0.6 else "")
    if u > 0.6:
        p.look_x = math.sin((u - 0.6) * 12) * 0.8
        p.flush = 0.4


def p_fall_over(p, u, t, now, sp):
    """Very rare: topple over, see stars, get up, shake it off, embarrassed smile."""
    down = _ss(u / 0.18) * (1 - _ss((u - 0.62) / 0.18))
    p.lie = max(p.lie, 0.9 * down)
    p.roll = 18 * down * (1 if sp not in BIPEDS else 0.4)
    p.tilt += 30 * down if sp in BIPEDS else 0.0
    p.eye_style = "spiral" if 0.15 < u < 0.6 else ("closed" if u > 0.85 else "normal")
    p.mark = "stars" if 0.15 < u < 0.6 else ("sweat" if u > 0.7 else "")
    p.mouth = "w"
    if u > 0.7:
        p.flush = 0.5
        p.blush = 1.0


def p_stagger(p, u, t, now, sp):
    """Additive wobble while walking dizzy."""
    p.tilt += math.sin(t * 5.0) * 7
    p.shift += math.sin(t * 5.0) * 1.4
    p.head_tilt += math.sin(t * 5.0 + 1) * 10
    p.eye_style = "spiral"
    p.mark = "stars"
    p.mouth = "w"


def p_lean(p, u, t, now, sp):
    """Lean back against a window edge, relaxed (arms crossed / tail curled)."""
    k = _env(u, 0.12, 0.12)
    if sp in BIPEDS:
        p.tilt -= 10 * k
        p.cross = k
        p.shift -= 1.5 * k
    else:
        p.sit = max(p.sit, 0.7 * k)
        p.tilt -= 6 * k
        p.tail = 80
    p.eye_open = min(p.eye_open, 0.8)
    p.mouth = "smile"
    p.head_tilt += 5 * k


def p_peek_behind(p, u, t, now, sp):
    """Peek out from behind a window's edge: lean sideways, big curious eyes, paw on the edge."""
    k = _env(u, 0.2, 0.2)
    p.tilt += 16 * k
    p.crouch_body = 0.25 * k
    p.head_dx += 4 * k
    p.paw = 0.35 * k
    p.eye_style = "wide"
    p.mouth = "o"
    p.ear = 1.0
    p.look_x = 0.6 + 0.3 * math.sin(t * 2)


def p_look_over(p, u, t, now, sp):
    """Look over the top edge of the window it stands on."""
    k = _env(u, 0.2, 0.2)
    p.tilt += 18 * k
    p.head_dy += 5 * k
    p.look_y = 0.9 * k
    p.look_x = 0.6
    p.paw = 0.25 * k
    p.ear = 0.8


def p_head_bump(p, u, t, now, sp):
    """Cat affection: lean in and bump / rub the head, eyes closed, purring."""
    bump = _bump((u * 3.0) % 1.0, 1.0)
    p.head_dx += 4.5 * bump
    p.head_tilt += 10 * math.sin(t * 6)
    p.eye_style = "closed"
    p.mouth = "smile"
    p.blush = 1.0
    p.mark = "heart" if u > 0.3 else ""
    p.tail += math.sin(t * 4) * 12
    p.front = max(p.front, 0.4)


def p_purr(p, u, t, now, sp):
    """Additive purr: a tiny fast vibration of the body and closed happy eyes."""
    p.shift += math.sin(t * 90) * 0.18
    p.breath *= 1.3
    p.eye_style = "closed"
    p.blush = max(p.blush, 0.9)


def p_laugh_hard(p, u, t, now, sp):
    """Laughing so hard: animals roll onto their back kicking, people bend over holding the belly."""
    p_laugh(p, u, t, now, sp)
    k = _env(u, 0.12, 0.2)
    if sp in BIPEDS:
        p.tilt += 16 * k + math.sin(t * 20) * 2 * k
        p.belly = k
    else:
        p.lie = max(p.lie, 0.85 * k)
        p.roll = math.sin(t * 7) * 20 * k
        p.paw = (0.5 + 0.5 * math.sin(t * 16)) * k
        p.paw2 = (0.5 - 0.5 * math.sin(t * 16)) * k


def p_jump_spin(p, u, t, now, sp):
    p_jump_pose(p, u, t, now, sp)
    p.turn = math.cos(2 * math.pi * _ss(u))
    p.eye_style = "happy"
    p.mouth = "grin"


def p_jump_star(p, u, t, now, sp):
    p_jump_pose(p, u, t, now, sp)
    k = _bump(u, 1.0)
    p.arms_up = max(p.arms_up, k)
    p.paw = max(p.paw, 0.8 * k)
    p.paw2 = max(p.paw2, 0.8 * k)
    p.mark = "sparkle" if 0.3 < u < 0.7 else ""
    p.mouth = "grin"


def p_turn_hop(p, u, t, now, sp):
    p.lift += 7 * _bump(u, 1.0)
    p.mouth = "grin"


def p_turn_look(p, u, t, now, sp):
    p.look_x = -0.9 * _env(u, 0.2, 0.3)
    p.head_dx -= 3 * _env(u, 0.2, 0.3)
    p.ear = 0.7


def p_sigh(p, u, t, now, sp):
    k = _bump(u, 1.0)
    p.bob += 1.6 * k
    p.head_dy += 1.5 * k
    p.mark = "sigh" if u > 0.2 else ""
    p.eye_open = min(p.eye_open, 1 - 0.4 * k)


def p_lean_cursor(p, u, t, now, sp):
    """Love: lean toward the cursor / user with a dreamy look."""
    k = _env(u, 0.25, 0.25)
    p.tilt += 6 * k
    p.head_dx += 3 * k
    p.eye_style = "heart"
    p.mark = "heart"
    p.blush = 1.0
    p.mouth = "smile"


def p_glare(p, u, t, now, sp):
    """The start of anger: freeze, eyes narrow, brows drop, body stiffens, fists clench, tail lashes."""
    k = _ss(u / 0.35)
    p.freeze = max(p.freeze, 0.8 * k)
    p.eye_style = "angry"
    p.lid = max(p.lid, 0.35 * k)
    p.brow = -1.0
    p.fists = max(p.fists, k)
    p.shoulders += 0.7 * k
    p.tilt += 5 * k
    p.head_dx += 2.5 * k
    p.head_tilt += 7 * k
    p.stance += 0.8 * k
    p.mouth = "pout" if u < 0.6 else "teeth"
    p.tail += math.sin(t * 9) * 22 * k
    p.ear = -0.3


def p_cool_down(p, u, t, now, sp):
    """Recovery after a big emotion: deep breath in, long breath out, shoulders drop, look away, neutral."""
    inhale = _ss(u / 0.35) * (1 - _ss((u - 0.4) / 0.3))
    exhale = _ss((u - 0.38) / 0.3)
    p.puff = max(p.puff, inhale)
    p.shoulders += 0.5 * inhale - 0.9 * exhale * (1 - _ss((u - 0.8) / 0.2))
    p.head_dy += -2.0 * inhale + 2.0 * exhale * (1 - u)
    p.look_x = -0.7 * _bump(u - 0.45, 0.4)
    p.head_dx -= 2.0 * _bump(u - 0.45, 0.4)
    if 0.45 < u < 0.7:
        p.eye_style = "closed"
        p.mouth = "o"
    p.mark = "sigh" if 0.4 < u < 0.75 else ""
    p.tail_pose -= 0.3 * exhale * (1 - u)


def p_notice(p, u, t, now, sp):
    """Something caught its attention: ears perk, head comes up, eyes widen (additive)."""
    k = _env(u, 0.12, 0.4)
    p.ear = max(p.ear, 1.0 * k)
    p.head_dy -= 2.5 * k
    p.eye_scale += 0.12 * k
    p.tail_pose += 0.4 * k


def p_duck(p, u, t, now, sp):
    """Something whooshed past the head: duck, squeeze eyes, then peek."""
    k = _env(u, 0.06, 0.45)
    p.crouch_body += 0.7 * k
    p.head_dy += 6 * k
    p.head_dx -= 2 * k
    p.ear = -1.0
    p.shoulders += 0.9 * k
    p.hug = max(p.hug, 0.6 * k if sp in BIPEDS else 0.0)
    p.eye_style = "squeeze" if u < 0.55 else "wide"
    p.mouth = "wobble"
    p.tail_pose -= 0.8 * k


def p_recover(p, u, t, now, sp):
    """Get back up after a fall: push up, shake the head, brush off, small embarrassed smile."""
    up = _ss(u / 0.45)
    p.lie = max(p.lie, 0.8 * (1 - up))
    p.crouch_body += 0.4 * (1 - _ss((u - 0.3) / 0.4))
    p.head_roll += math.sin(t * 16) * 10 * _bump(u - 0.35, 0.3)
    p.eye_style = "closed" if 0.35 < u < 0.6 else "normal"
    p.mouth = "w"
    if sp == "robot":
        p.glitch = max(p.glitch, 0.5 * _bump(u, 0.4))
        p.mark = "dots" if u < 0.5 else ""
    if u > 0.7:
        p.blush = 1.0


def p_weight_shift(p, u, t, now, sp):
    """Idle: move the weight onto the other side and back (hips sway, opposite shoulder dips)."""
    k = _bump(u, 0.5)
    p.shift += 3.5 * k * (1 if int(now) % 2 else -1)
    p.pelvis_roll += 4.0 * k
    p.head_roll -= 2.5 * k


def p_small_stretch(p, u, t, now, sp):
    """Idle: a little stretch - back arches, chin up, a quiet breath out."""
    k = _env(u, 0.3, 0.35)
    if sp in BIPEDS:
        p.arms_up = max(p.arms_up, 0.55 * k)
        p.head_tilt -= 8 * k
    else:
        p.stretch = max(p.stretch, 0.45 * k)
        p.tail_pose += 0.4 * k
    p.puff = max(p.puff, 0.5 * _bump(u - 0.35, 0.3))
    p.eye_style = "closed" if 0.3 < u < 0.7 else p.eye_style


def p_slow_blink(p, u, t, now, sp):
    """Idle: a slow, contented blink (cats do it as a friendly signal)."""
    c = _bump(u, 0.5)
    p.eye_open = min(p.eye_open, 1.0 - 0.95 * c)
    p.lid = max(p.lid, 0.9 * c)
    p.smile = max(p.smile, 0.25 * c)


def p_slip(p, u, t, now, sp):
    """Physical comedy: feet slide out, arms / paws flail, a little hop to catch the balance."""
    sl = _bump(u, 0.35)
    p.tilt -= 20 * sl
    p.lift += 4 * _bump(u - 0.15, 0.2)
    p.stance += 0.8 * sl
    p.arms_up = max(p.arms_up, 0.9 * sl) if sp in BIPEDS else p.arms_up
    p.paw = max(p.paw, 0.7 * sl) if sp not in BIPEDS else p.paw
    p.eye_style = "wide" if u < 0.5 else "normal"
    p.mouth = "o" if u < 0.5 else "w"
    p.ear_flop += 20 * sl
    if u > 0.55:
        p.look_x = math.sin((u - 0.55) * 14) * 0.7
        p.blush = max(p.blush, 0.8)


def p_shake_off(p, u, t, now, sp):
    """After a fall / slip: shake it off (head and body wobble), then stand tall."""
    k = _env(u, 0.1, 0.3)
    p.roll += math.sin(t * 30) * 5 * k
    p.head_roll += math.sin(t * 30 + 0.6) * 9 * k
    p.ear_flop += math.sin(t * 30 + 1.0) * 14 * k
    p.tail = math.sin(t * 26) * 25 * k
    p.eye_style = "closed" if u < 0.6 else "normal"
    p.mouth = "w"
    if u > 0.7:
        p.head_tilt -= 5 * _ss((u - 0.7) / 0.3)


def p_skid(p, u, t, now, sp):
    """Skid stop from a run: feet dig in forward, body leans way back, ears forward, then rebound."""
    k = _env(u, 0.05, 0.4)
    p.tilt -= 14 * k
    p.crouch_body += 0.45 * k
    p.stance += 1.0 * k
    p.head_dx -= 2 * k
    p.ear_flop += 12 * k
    p.foot_pitch -= 8 * k
    p.eye_style = "wide" if u < 0.5 else "normal"
    p.mouth = "o"


def p_backpedal(p, u, t, now, sp):
    """Backing away (additive on a backwards shuffle): weight back, head pulled in, eyes on the thing."""
    k = _env(u, 0.15, 0.2)
    p.tilt -= 6 * k
    p.crouch_body += 0.2 * k
    p.head_dx -= 3 * k
    p.shoulders += 0.4 * k
    p.ear = -0.2


def p_listen(p, u, t, now, sp):
    """Fox: freeze, ears swivel one way then the other, head cocked."""
    k = _env(u, 0.15, 0.2)
    p.freeze = max(p.freeze, 0.7 * k)
    sw = math.sin(t * 2.2)
    p.ear_l += 25 * sw * k
    p.ear_r -= 25 * sw * k
    p.head_roll += 14 * k * (1 if sw > 0 else -1) * 0.7
    p.ear = 1.0
    p.eye_style = "focused"


def p_tail_wrap(p, u, t, now, sp):
    k = _env(u, 0.2, 0.2)
    p.sit = max(p.sit, k)
    p.tail = 95 * k
    p.eye_open = min(p.eye_open, 1 - 0.3 * k)
    p.mouth = "smile"


def p_look_back(p, u, t, now, sp):
    k = _env(u, 0.2, 0.25)
    p.look_x = -1.0 * k
    p.head_dx -= 4 * k
    p.head_tilt -= 4 * k
    p.front = max(p.front, 0.45 * k)
    p.ear = 0.8


def p_curl_up(p, u, t, now, sp):
    k = _ss(u / 0.15)
    p.lie = max(p.lie, k)
    p.tuck = max(p.tuck, k)
    p.tail = 100 * k
    p.head_dy += 3 * k
    p.eye_style = "closed"
    p.mouth = "sleep"


def p_watch(p, u, t, now, sp):
    """Buddy: raise the wrist, look down at the watch, a little nod."""
    k = _env(u, 0.2, 0.25)
    p.paw = max(p.paw, 0.55 * k)
    p.paw_wave = 40 * k
    p.look_y = 0.7 * k
    p.look_x = 0.5 * k
    p.head_tilt += 10 * k
    if 0.45 < u < 0.6:
        p.head_dy += 1.0


def p_phone(p, u, t, now, sp):
    k = _env(u, 0.2, 0.25)
    p.read = 0.6 * k
    p.write = 0.5 * k * (0.5 + 0.5 * math.sin(t * 9))
    p.look_y = 0.8 * k
    p.head_tilt += 12 * k
    p.mouth = "smile" if u > 0.6 else "flat"


def p_fist_pump(p, u, t, now, sp):
    pump = abs(math.sin(t * 7.0)) * _env(u, 0.1, 0.2)
    p.arms_up = max(p.arms_up, 0.0)
    p.paw = max(p.paw, 0.6 + 0.4 * pump)
    p.paw_wave = -60 * pump
    p.fists = max(p.fists, 0.8)
    p.lift += 4 * pump
    p.eye_style = "happy"
    p.mouth = "grin"
    p.mark = "sparkle" if u > 0.4 else ""


def p_excited_jump(p, u, t, now, sp):
    p_jump_pose(p, u, t, now, sp)
    p.paw = max(p.paw, 0.7)
    p.paw2 = max(p.paw2, 0.7)
    p.tongue = 0.7
    p.tail += math.sin(t * 20) * 30
    p.eye_style = "happy"



# ----------------------------------------------------------------- 2.0 reactions
SNEEZE, HICCUP, HUM = "SNEEZE", "HICCUP", "HUM"


def p_sneeze(p, u, t, now, sp):
    """"Ah... ah... CHOO!": the head tips back with the eyes squeezing and the mouth opening in two
    hitches, snaps forward on the sneeze (whole body jolts, eyes shut), then a little shake-off."""
    big = sp in BIPEDS
    if u < 0.52:                                    # the build-up, in two hitches
        k = _ss(u / 0.52)
        hitch = 0.25 * math.sin(math.pi * min(1.0, u / 0.26)) if u < 0.26 else 0.0
        p.head_dy -= (3.0 * k + 1.2 * hitch)
        p.head_tilt -= 8.0 * k
        p.tilt -= 3.0 * k
        p.lid += 0.65 * k
        p.lid_low += 0.3 * k
        p.brow_raise += 0.6 * k
        p.mouth_open = max(p.mouth_open, 0.45 * k + hitch)
        p.mouth = "open" if k > 0.35 else p.mouth
        p.puff = max(p.puff, 0.5 * k)
        p.breath = max(p.breath, k)
        p.nose += 0.8 * k * math.sin(t * 30)
    elif u < 0.64:                                  # CHOO
        k = _bump(u - 0.52, 0.12)
        p.head_dy += 6.0 * k
        p.tilt += (10.0 if big else 6.0) * k
        p.squash += 0.25 * k
        p.eye_style = "closed"
        p.lid = max(p.lid, 1.0)
        p.mouth = "o"
        p.mouth_open = max(p.mouth_open, 0.2)
        p.ear_flop += 25 * k
        p.shoulders += 0.5 * k
        p.mark = "dots"
        if big:
            p.chin = max(p.chin, 0.9 * k)          # people sneeze into a hand
    else:                                           # recover: blink, small head shake, sniff
        k = 1.0 - _ss((u - 0.64) / 0.36)
        p.head_dx += math.sin((u - 0.64) * 40) * 1.2 * k
        p.lid += 0.3 * k
        p.nose += 0.6 * k * math.sin(t * 24)
        p.brow += 0.3 * k
        p.blush = max(p.blush, 0.8)


def p_hiccup(p, u, t, now, sp):
    """Three little hiccups: each a quick jolt up with the shoulders, eyes popping wide, a tiny "hic"."""
    for c in (0.12, 0.45, 0.78):
        k = _bump(u - c, 0.09)
        if k > 0:
            p.lift += 2.2 * k
            p.shoulders += 0.7 * k
            p.head_dy -= 1.2 * k
            p.eye_scale *= 1.0 + 0.12 * k
            p.brow_raise += 0.7 * k
            p.mouth_open = max(p.mouth_open, 0.25 * k)
            p.squash -= 0.15 * k
    e = _env(u, 0.1, 0.2)
    p.brow += 0.45 * e                              # mildly bewildered in between
    p.head_roll += 4.0 * e
    p.blush = max(p.blush, 0.75)
    if 0.3 < u < 0.95:
        p.mark = "dots"


def p_shiver(p, u, t, now, sp):
    """Brrr: a fast fine shiver, shoulders up, arms hugged in, a small huddle; eyes squeezed."""
    k = _env(u, 0.12, 0.25)
    p.shift += math.sin(t * 48) * 0.6 * k
    p.head_dx += math.sin(t * 55 + 0.7) * 0.4 * k
    p.shoulders += 0.8 * k
    p.hug = max(p.hug, 0.85 * k)
    p.crouch_body = max(p.crouch_body, 0.2 * k)
    p.slump += 0.1 * k
    p.lid += 0.35 * k
    p.mouth = "teeth" if k > 0.4 else p.mouth
    p.ear = min(p.ear, -0.6 * k)
    p.tail_pose -= 0.8 * k
    p.brow += 0.4 * k


def p_daydream(p, u, t, now, sp):
    """Drifting off into a daydream: the gaze floats up and away, a soft smile, slow sway,
    lids half down - then a little start back to the present at the end."""
    k = _env(u, 0.18, 0.12)
    p.look_x = p.look_x * (1 - k) + (0.55 + 0.1 * math.sin(t * 0.7)) * k
    p.look_y = p.look_y * (1 - k) - 0.45 * k
    p.head_yaw += 0.25 * k
    p.head_dy -= 1.6 * k
    p.head_roll += 6.0 * k * math.sin(t * 0.6)
    p.lid += 0.35 * k
    p.smile += 0.35 * k
    p.lid_low += 0.15 * k
    p.shift += math.sin(t * 0.9) * 0.6 * k
    p.chin = max(p.chin, 0.5 * k) if sp in BIPEDS else p.chin
    p.tail += math.sin(t * 1.3) * 12 * k
    if 0.15 < u < 0.85:
        p.mark = "dots" if u < 0.3 else "sparkle"
    if u > 0.9:                                     # snap out of it
        b = _bump(u - 0.9, 0.1)
        p.eye_scale *= 1.0 + 0.15 * b
        p.brow_raise += 0.6 * b


def p_hum(p, u, t, now, sp):
    """Humming a little tune: closed-mouth hum with small head bobs on the beat, eyes half closed
    and content, a music note floating up."""
    k = _env(u, 0.12, 0.15)
    beat = math.sin(t * 2 * math.pi * 1.6)
    p.head_roll += 6.0 * beat * k
    p.head_dy += 0.8 * max(0.0, beat) * k
    p.shift += 0.5 * beat * k
    p.lid += 0.3 * k
    p.smile += 0.4 * k
    p.mouth_open = max(p.mouth_open, (0.08 + 0.06 * abs(beat)) * k)
    p.tail += 18 * beat * k
    p.ear += 0.2 * k
    p.mark = "note"


def p_eat(p, u, t, now, sp):
    """Eating a treat: animals dip the head to the floor and munch (jaw chewing, happy squint, tail
    going), then sit back and lick their lips; people take a bite from the hand and chew."""
    k = _env(u, 0.15, 0.2)
    chew = max(0.0, math.sin(t * 2 * math.pi * 3.2))
    if sp in BIPEDS:
        p.chin = max(p.chin, 0.9 * k)
        p.head_dy += 1.5 * k
        p.jaw = max(p.jaw, 0.35 * chew * k)
        p.mouth_open = max(p.mouth_open, 0.25 * chew * k)
        p.smile += 0.4 * k
        p.lid_low += 0.4 * k
    else:
        dip = k if u < 0.75 else k * (1 - (u - 0.75) / 0.25)
        p.head_dy += 9.0 * dip
        p.head_dx += 3.0 * dip
        p.tilt += 6.0 * dip
        p.crouch_body = max(p.crouch_body, 0.25 * dip)
        p.jaw = max(p.jaw, 0.45 * chew * dip)
        p.lid_low += 0.45 * k
        p.ear += 0.3 * k
        p.tail += math.sin(t * 9) * 25 * k
        if u > 0.78:                                   # sits back up and licks its lips
            p.tongue = max(p.tongue, 0.7 * math.sin(math.pi * (u - 0.78) / 0.22))
    p.blush = max(p.blush, 0.8)
    if u > 0.8:
        p.mark = "heart"

# =========================================================================
def A(name, category, pose=None, **kw) -> ActionDef:  # noqa: N802
    kw.setdefault("tags", frozenset())
    if isinstance(kw["tags"], (set, tuple, list)):
        kw["tags"] = frozenset(kw["tags"])
    return ActionDef(name, category, pose, **kw)


P = Priority
_ACTIONS = [
    # ---------------------------------------------------------------- basic
    A("IDLE", "basic", None, state=S.IDLE, loop=True, duration=3.0),
    A("STAND", "basic", p_stand_tall, state=S.IDLE, duration=1.2),
    A("WALK", "basic", None, motion="walk", priority=P.LOCOMOTION, duration=0),
    A("JOG", "basic", None, motion="jog", priority=P.LOCOMOTION, duration=0, energy=1),
    A("RUN", "basic", None, motion="run", priority=P.LOCOMOTION, duration=0, energy=2),
    A("SPRINT", "basic", None, motion="sprint", priority=P.LOCOMOTION, duration=0, energy=4, cooldown=20),
    A("STOP", "basic", p_stop, motion="stop", track="additive", duration=0.4),
    A("TURN", "basic", None, motion="turn", duration=0),
    # --------------------------------------------------------------- animal
    A("SIT", "animal", None, state=S.SITTING, loop=True, duration=6, tags={"idle", "focus"}, weight=10, energy=-1),
    A("LIE_DOWN", "animal", p_lie_awake, state=S.IDLE, duration=8, species=QUADS, tags={"idle", "focus"}, weight=4,
      energy=-2,
      blend_in=0.6, blend_out=0.6),
    A("SLEEP", "animal", None, state=S.SLEEPING, loop=True, duration=30, cooldown=300, tags={"sleep"}, energy=-15),
    A("WAKE", "animal", None, state=S.WAKING, duration=1.0),
    A("STRETCH", "animal", None, state=S.STRETCHING, duration=2.2, cooldown=40, tags={"idle", "focus"}, weight=4),
    A("YAWN", "animal", None, state=S.YAWNING, duration=1.7, cooldown=45, tags={"idle", "sleep"}, weight=2),
    A("SCRATCH", "animal", None, state=S.SCRATCHING, duration=1.8, cooldown=40, species=QUADS, tags={"idle"},
      weight=2),
    A("GROOM", "animal", None, state=S.GROOMING, duration=3.0, cooldown=40, species=frozenset({"cat", "rabbit", "fox"}),
      tags={"idle", "focus"}, weight=4),
    A("SHAKE", "animal", p_shake, state=S.IDLE, duration=1.0, cooldown=60, species=QUADS, tags={"idle", "break"},
      weight=2, events=((0.5, "SHAKE"),)),
    A("SNIFF", "animal", p_sniff, state=S.IDLE, duration=2.2, cooldown=20, species=QUADS, tags={"idle", "explore"},
      weight=4),
    A("LOOK_AROUND", "animal", None, state=S.LOOKING_AROUND, duration=3.0, tags={"idle", "focus"}, weight=8),
    # -------------------------------------------------------------- playful
    A("JUMP", "playful", p_jump_pose, motion="jump", duration=0.9, cooldown=20, priority=P.INTERACTION,
      tags={"play", "break"}, weight=3, energy=4, exits=(0.0, 1.0)),
    A("HOP", "playful", p_jump_pose, motion="hop", duration=0.6, cooldown=10, tags={"play", "break"}, weight=3,
      energy=2, exits=(0.0, 1.0)),
    A("BOUNCE", "playful", p_bounce, state=S.IDLE, duration=1.6, cooldown=25, tags={"play", "break"}, weight=3,
      energy=3),
    A("SPIN", "playful", p_spin, state=S.IDLE, duration=1.2, cooldown=40, tags={"play", "break"}, weight=2, energy=2,
      exits=(0.0, 1.0)),
    A("DANCE", "playful", p_dance, state=S.IDLE, duration=2.6, cooldown=90, tags={"play", "break", "celebrate"},
      weight=2, energy=5, events=((0.5, CELEBRATION_PEAK),)),
    A("CHASE_CURSOR", "playful", None, motion="chase", priority=P.INTERACTION, duration=0, tags={"play"}, energy=4),
    A("PLAY", "playful", None, state=S.PLAYING, duration=3.0, tags={"play", "break"}, weight=5, energy=4),
    A("ROLL", "playful", p_roll, state=S.IDLE, duration=1.6, cooldown=60, species=QUADS, tags={"play", "break"},
      weight=2, energy=2, exits=(0.0, 1.0)),
    A("PAW_AT_OBJECT", "playful", p_paw_at, state=S.IDLE, duration=2.0, cooldown=30, tags={"play"}, weight=2),
    # ------------------------------------------------------------ emotional
    A("HAPPY", "emotional", p_happy, track="upper", duration=1.6, emotion="HAPPY", tags={"social"}),
    A("EXCITED", "emotional", p_excited, state=S.IDLE, duration=1.4, emotion="EXCITED", tags={"break"}, energy=2),
    A("CURIOUS", "emotional", p_curious, track="upper", duration=1.8, emotion="CURIOUS", tags={"idle", "explore"},
      weight=3),
    A("SURPRISED", "emotional", p_surprised, state=S.IDLE, duration=1.0, emotion="SURPRISED", cooldown=30,
      priority=P.INTERACTION),
    A("WORRIED", "emotional", p_worried, track="upper", duration=2.0, emotion="WORRIED"),
    A("ANNOYED", "emotional", p_annoyed, track="upper", duration=1.8, emotion="ANNOYED"),
    A("SLEEPY", "emotional", p_sleepy, track="upper", duration=2.5, emotion="SLEEPY", tags={"sleep"}),
    A("CONFUSED", "emotional", None, state=S.CONFUSED, duration=2.0, emotion="WORRIED"),
    A("PROUD", "emotional", p_proud, state=S.IDLE, duration=2.0, emotion="PROUD", tags={"celebrate"}),
    A("CELEBRATE", "emotional", None, state=S.CELEBRATING, duration=2.4, priority=P.CELEBRATION,
      emotion="CELEBRATING", events=((0.5, CELEBRATION_PEAK),), tags={"celebrate"}, energy=3),
    # ---------------------------------------------------------- emotion pack
    A("SHOUT", "emotional", p_shout, state=S.IDLE, duration=1.5, cooldown=120, priority=P.INTERACTION,
      emotion="ANGRY", events=((0.3, SHOUT),), tags={"angry"}),
    A("ANGRY", "emotional", p_angry, state=S.IDLE, duration=2.2, cooldown=20, priority=P.INTERACTION,
      emotion="ANGRY", events=((0.12, STOMP), (0.4, STOMP), (0.68, STOMP)), tags={"angry"}, energy=2),
    A("GRUMPY", "emotional", p_grumpy, state=S.IDLE, duration=3.0, emotion="GRUMPY", tags={"angry"}),
    A("TANTRUM", "emotional", p_tantrum, state=S.IDLE, duration=2.4, cooldown=120, priority=P.INTERACTION,
      emotion="ANGRY", events=((0.3, STOMP), (0.6, STOMP)), tags={"angry"}, energy=4),
    A("SULK", "emotional", p_sulk, state=S.IDLE, duration=4.0, cooldown=60, emotion="SAD", blend_in=0.4,
      blend_out=0.5),
    A("SCARED", "emotional", p_scared, state=S.IDLE, duration=1.8, cooldown=15, priority=P.INTERACTION,
      emotion="SCARED"),
    A("CRY", "emotional", p_cry, state=S.IDLE, duration=3.2, cooldown=60, emotion="SAD", blend_in=0.4),
    A("EMBARRASSED", "emotional", p_embarrassed, state=S.IDLE, duration=2.4, cooldown=20, emotion="EMBARRASSED"),
    A("LOVE", "emotional", p_love, state=S.IDLE, duration=2.6, emotion="LOVE", tags={"social"}, weight=1),
    A("BORED", "emotional", p_bored, state=S.IDLE, duration=4.0, cooldown=90, emotion="BORED", tags={"bored"},
      blend_in=0.5, blend_out=0.5, events=((0.4, SIGH),)),
    A("SMUG", "emotional", p_smug, state=S.IDLE, duration=2.2, emotion="SMUG"),
    A("DISGUSTED", "emotional", p_disgusted, state=S.IDLE, duration=1.8, emotion="DISGUSTED"),
    A("SHOCKED", "emotional", p_shocked, state=S.IDLE, duration=1.4, priority=P.INTERACTION, emotion="SHOCKED"),
    A("LAUGH", "emotional", p_laugh, state=S.IDLE, duration=2.4, cooldown=20, emotion="LAUGHING",
      tags={"social", "celebrate"}),
    A("DETERMINED", "emotional", p_determined, state=S.IDLE, duration=1.8, emotion="DETERMINED",
      tags={"focus_start"}),
    A("DIZZY", "emotional", p_dizzy, state=S.IDLE, duration=2.4, cooldown=10, priority=P.INTERACTION,
      emotion="DIZZY"),
    A("RELIEVED", "emotional", p_relieved, state=S.IDLE, duration=1.6, emotion="RELIEVED", events=((0.3, SIGH),)),
    A("HUFF", "emotional", p_huff, track="additive", duration=0.9, tags={"angry"}),
    # ------------------------------------------------- living-character upgrade
    A("SHOUT_SOFT", "emotional", p_shout_soft, state=S.IDLE, duration=1.2, cooldown=30, priority=P.INTERACTION,
      emotion="ANNOYED", tags={"angry"}),
    A("SHOUT_BIG", "emotional", p_shout_big, state=S.IDLE, duration=1.8, cooldown=120, priority=P.INTERACTION,
      emotion="ANGRY", events=((0.3, SHOUT), (0.45, STOMP)), tags={"angry"}, energy=2),
    A("STOMP", "emotional", p_stomp, state=S.IDLE, duration=1.3, cooldown=60, priority=P.INTERACTION,
      emotion="ANNOYED", events=((0.15, STOMP), (0.5, STOMP), (0.85, STOMP)), tags={"angry"}, energy=1),
    A("HIDE_FACE", "emotional", p_hide_face, state=S.IDLE, duration=2.4, cooldown=20, emotion="SCARED"),
    A("TREMBLE", "emotional", p_tremble, state=S.IDLE, duration=1.6, emotion="NERVOUS"),
    A("PANIC", "emotional", p_panic, track="additive", motion="sprint", duration=1.6, cooldown=60,
      priority=P.INTERACTION, emotion="SCARED", energy=3),
    A("SIGH", "emotional", p_sigh, track="additive", duration=1.2, events=((0.2, SIGH),)),
    A("LAUGH_HARD", "emotional", p_laugh_hard, state=S.IDLE, duration=2.6, cooldown=40, emotion="LAUGHING",
      tags={"celebrate"}),
    A("STUMBLE", "emotional", p_stumble, state=S.IDLE, duration=1.3, cooldown=600, priority=P.INTERACTION,
      emotion="EMBARRASSED"),
    A("FALL_OVER", "emotional", p_fall_over, state=S.IDLE, duration=2.6, cooldown=1800, priority=P.INTERACTION,
      emotion="DIZZY"),
    A("STAGGER", "emotional", p_stagger, track="additive", motion="walk", duration=1.8, emotion="DIZZY"),
    A("LEAN_TO_CURSOR", "emotional", p_lean_cursor, state=S.IDLE, duration=2.2, emotion="LOVE"),
    A("HIGH_FIVE", "social", p_high_five, state=S.IDLE, duration=1.4, priority=P.INTERACTION, emotion="HAPPY",
      events=((0.45, CELEBRATION_PEAK),), tags={"social"}),
    A("BOW", "social", p_bow, state=S.IDLE, duration=1.6, tags={"social"}),
    A("GREET", "social", p_greet, state=S.IDLE, duration=1.8, priority=P.INTERACTION, emotion="EXCITED",
      tags={"social"}),
    A("FAREWELL", "social", p_farewell, state=S.IDLE, duration=1.8, emotion="HAPPY", tags={"social"}),
    A("TUMBLE", "playful", p_tumble, state=S.IDLE, duration=1.1, cooldown=60, tags={"play", "break"}, weight=1,
      energy=3),
    A("SKIP", "playful", p_skip, track="additive", motion="jog", duration=2.4, cooldown=30, tags={"play"},
      emotion="PLAYFUL", energy=2),
    A("JUMP_SPIN", "playful", p_jump_spin, motion="jump", duration=0.9, cooldown=20, priority=P.INTERACTION,
      energy=4, exits=(0.0, 1.0)),
    A("JUMP_STAR", "playful", p_jump_star, motion="jump", duration=0.9, cooldown=20, priority=P.INTERACTION,
      energy=4, exits=(0.0, 1.0)),
    A("TURN_HOP", "basic", p_turn_hop, motion="turn", duration=0.45),
    A("TURN_LOOK", "basic", p_turn_look, track="upper", motion="turn", duration=0.7),
    A("LEAN", "environment", p_lean, state=S.IDLE, duration=6.0, tags={"window"}),
    A("PEEK_BEHIND", "environment", p_peek_behind, state=S.IDLE, duration=2.6, cooldown=40, tags={"window"}),
    A("LOOK_OVER", "environment", p_look_over, state=S.IDLE, duration=2.6, cooldown=30, tags={"window"}),
    A("HEAD_BUMP", "species", p_head_bump, state=S.IDLE, duration=2.2, species=frozenset({"cat"}),
      emotion="LOVE", tags={"social"}),
    A("PURR", "species", p_purr, track="additive", duration=2.5, species=frozenset({"cat"})),
    # ------------------------------------------------------------ 2.0 reactions
    A("EAT", "emotional", p_eat, state=S.IDLE, duration=3.2, priority=P.INTERACTION, emotion="HAPPY",
      blend_in=0.3, blend_out=0.4),
    A("SNEEZE", "emotional", p_sneeze, state=S.IDLE, duration=2.0, cooldown=240, priority=P.INTERACTION,
      tags={"idle", "gag"}, weight=0.4, events=((0.53, SNEEZE),), blend_in=0.2, blend_out=0.35),
    A("HICCUP", "emotional", p_hiccup, track="additive", duration=2.6, cooldown=300, tags={"idle", "gag"},
      weight=0.3, events=((0.12, HICCUP), (0.45, HICCUP), (0.78, HICCUP))),
    A("SHIVER", "emotional", p_shiver, state=S.IDLE, duration=2.2, cooldown=180, tags={"idle", "gag"}, weight=0.3,
      emotion="NERVOUS"),
    A("DAYDREAM", "emotional", p_daydream, state=S.IDLE, duration=6.0, cooldown=180, tags={"idle", "focus"},
      weight=0.8, emotion="RELAXED", blend_in=0.8, blend_out=0.6),
    A("HUM", "emotional", p_hum, track="additive", duration=4.0, cooldown=120, tags={"idle"}, weight=0.8,
      emotion="HAPPY", events=((0.1, HUM),)),
    # ----------------------------------------------------- 3D character overhaul
    A("GLARE", "emotional", p_glare, state=S.IDLE, duration=1.0, priority=P.INTERACTION, emotion="ANNOYED",
      tags={"angry"}),
    A("COOL_DOWN", "emotional", p_cool_down, state=S.IDLE, duration=2.2, emotion="CALMING", blend_in=0.3,
      blend_out=0.5, events=((0.45, SIGH),)),
    A("NOTICE", "emotional", p_notice, track="additive", duration=1.2),
    A("DUCK", "emotional", p_duck, state=S.IDLE, duration=1.1, cooldown=20, priority=P.INTERACTION,
      emotion="NERVOUS"),
    A("RECOVER", "basic", p_recover, state=S.IDLE, duration=1.6, priority=P.INTERACTION, emotion="EMBARRASSED"),
    A("SKID_STOP", "basic", p_skid, motion="stop", track="additive", duration=0.8),
    A("BACKPEDAL", "basic", p_backpedal, track="additive", motion="backpedal", duration=1.4),
    A("SIDESTEP", "basic", None, motion="sidestep", track="additive", duration=0.8),
    A("SLOW_WALK", "basic", None, motion="slow_walk", priority=P.LOCOMOTION, duration=0),
    A("SMALL_JUMP", "playful", p_jump_pose, motion="small_jump", duration=0.6, cooldown=8, priority=P.INTERACTION,
      energy=1, exits=(0.0, 1.0)),
    A("HIGH_JUMP", "playful", p_jump_star, motion="high_jump", duration=1.1, cooldown=30, priority=P.INTERACTION,
      energy=5, exits=(0.0, 1.0)),
    A("LISTEN", "species", p_listen, state=S.IDLE, duration=2.4, species=frozenset({"fox", "cat", "rabbit", "dog"}),
      tags={"idle", "explore"}, weight=1.5),
    A("TAIL_WRAP", "species", p_tail_wrap, state=S.SITTING, duration=5.0, species=frozenset({"fox", "cat"}),
      tags={"idle", "focus"}, weight=1.5),
    A("LOOK_BACK", "species", p_look_back, track="upper", duration=1.6, tags={"idle"}, weight=1.0),
    A("CURL_UP", "species", p_curl_up, state=S.IDLE, duration=10.0, species=frozenset({"cat", "fox"}),
      tags={"focus"}, weight=1.0, energy=-3),
    A("LOOK_AT_WATCH", "species", p_watch, track="upper", duration=1.8, species=frozenset({"buddy"}),
      tags={"focus", "idle"}, weight=1.0),
    A("CHECK_PHONE", "species", p_phone, state=S.IDLE, duration=3.0, species=frozenset({"buddy"}),
      tags={"idle"}, weight=1.0),
    A("WEIGHT_SHIFT", "basic", p_weight_shift, track="additive", duration=2.2, tags={"micro", "idle"}),
    A("SMALL_STRETCH", "basic", p_small_stretch, track="additive", duration=1.8, cooldown=25, tags={"micro", "idle"}),
    A("SLOW_BLINK", "basic", p_slow_blink, track="additive", duration=1.1, tags={"micro", "idle"}),
    A("SLIP", "emotional", p_slip, state=S.IDLE, duration=1.2, cooldown=600, priority=P.INTERACTION,
      emotion="EMBARRASSED"),
    A("SHAKE_OFF", "basic", p_shake_off, state=S.IDLE, duration=1.2, priority=P.INTERACTION),
    A("FIST_PUMP", "species", p_fist_pump, state=S.IDLE, duration=1.4, species=frozenset({"buddy", "robot"}),
      tags={"celebrate"}, emotion="EXCITED"),
    A("EXCITED_JUMP", "species", p_excited_jump, motion="jump", duration=0.9, cooldown=20, priority=P.INTERACTION,
      species=frozenset({"dog"}), energy=4, emotion="EXCITED", exits=(0.0, 1.0)),
    # ---------------------------------------------------------------- social
    A("WAVE", "social", None, state=S.WAVING, duration=1.6, priority=P.INTERACTION, emotion="HAPPY", tags={"social"}),
    A("LOOK_AT_USER", "social", p_look_user, track="upper", duration=1.5, tags={"social", "focus"}, weight=4),
    A("NOD", "social", p_nod, track="additive", duration=1.0, tags={"social", "focus"}),
    A("SHAKE_HEAD", "social", p_shake_head, track="upper", duration=1.1),
    A("POINT", "social", p_point, track="upper", duration=1.4, priority=P.INTERACTION),
    A("BECKON", "social", p_beckon, track="upper", duration=1.6),
    A("CHEER", "social", p_cheer, state=S.IDLE, duration=1.8, priority=P.CELEBRATION, emotion="EXCITED",
      events=((0.5, CELEBRATION_PEAK),), tags={"celebrate"}),
    A("CLAP", "social", p_clap, track="upper", duration=1.6, tags={"celebrate", "social"}),
    A("THUMBS_UP", "social", p_thumbs, track="upper", duration=1.6, species=BIPEDS, tags={"social"}),
    # ---------------------------------------------------------------- cat
    A("LOAF", "species", p_loaf, state=S.IDLE, duration=12, species=frozenset({"cat", "rabbit"}),
      tags={"idle", "focus"}, weight=5, energy=-4, blend_in=0.7, blend_out=0.6),
    A("SCRATCH_EAR", "species", None, state=S.SCRATCHING, duration=1.8, cooldown=40, species=QUADS),
    A("LICK_PAW", "species", p_lick_paw, state=S.GROOMING, duration=3.0, cooldown=40, species=frozenset({"cat"}),
      tags={"idle", "focus"}, weight=4),
    A("TAIL_SWISH", "species", p_tail_swish, track="additive", duration=1.4, species=QUADS, tags={"micro"},
      events=((0.25, TAIL_SWING), (0.75, TAIL_SWING))),
    A("PAW_AT_CURSOR", "species", p_paw_at, state=S.IDLE, duration=1.8, cooldown=60, priority=P.INTERACTION,
      species=frozenset({"cat", "fox", "dog"}), tags={"play"}),
    A("POUNCE", "species", p_pounce, motion="pounce", duration=1.5, cooldown=45,
      species=frozenset({"cat", "fox"}), tags={"play", "break"}, weight=2, energy=5, exits=(0.0, 1.0)),
    A("ARCH_BACK", "species", p_arch, state=S.IDLE, duration=1.6, cooldown=90, species=frozenset({"cat"}),
      tags={"idle"}, weight=1),
    A("LOOK_UP", "species", p_look_up, track="upper", duration=2.0, tags={"idle", "explore"}, weight=2),
    # ---------------------------------------------------------------- dog
    A("DOWN", "species", p_lie_awake, state=S.IDLE, duration=8, species=frozenset({"dog"}), tags={"idle", "focus"},
      weight=4, energy=-2, blend_in=0.6, blend_out=0.6),
    A("TAIL_WAG", "species", p_tail_wag, track="additive", duration=1.8, species=frozenset({"dog", "fox", "cat"}),
      tags={"micro", "social"}, events=((0.5, TAIL_SWING),)),
    A("PLAY_BOW", "species", p_play_bow, state=S.IDLE, duration=1.8, cooldown=40, species=frozenset({"dog", "fox"}),
      tags={"play", "break"}, weight=3, energy=1),
    A("PANT", "species", p_pant, track="upper", duration=2.5, species=frozenset({"dog"}), tags={"idle"}, weight=2),
    A("PAW", "species", p_give_paw, state=S.IDLE, duration=1.8, species=QUADS, tags={"social"}, weight=1),
    A("CHASE", "species", None, motion="chase", priority=P.INTERACTION, duration=0, energy=4),
    # ---------------------------------------------------------------- fox
    A("EAR_TWITCH", "species", p_ear_twitch, track="additive", duration=0.45, tags={"micro"}),
    A("QUIET_WALK", "species", p_sneak, track="additive", motion="walk", duration=0, species=frozenset({"fox", "cat"})),
    A("SNEAK", "species", p_sneak, track="additive", motion="sneak", duration=0, species=frozenset({"fox", "cat"}),
      tags={"explore"}),
    A("TAIL_SWING", "species", p_tail_swing, track="additive", duration=2.0, species=frozenset({"fox", "cat", "dog"}),
      tags={"micro"}, events=((0.5, TAIL_SWING),)),
    A("ALERT", "species", p_alert, state=S.ALERT, duration=1.2, priority=P.DISTRACTION_RESPONSE),
    # ---------------------------------------------------------------- rabbit
    A("DOUBLE_HOP", "species", p_jump_pose, motion="double_hop", duration=1.2, cooldown=25,
      species=frozenset({"rabbit"}), tags={"play", "break"}, weight=3, energy=3, exits=(0.0, 1.0)),
    A("NOSE_TWITCH", "species", p_nose_twitch, track="additive", duration=1.0, species=frozenset({"rabbit"}),
      tags={"micro"}),
    A("FREEZE", "species", p_freeze, state=S.IDLE, duration=2.0, cooldown=60, species=frozenset({"rabbit"}),
      tags={"idle"}, weight=2),
    # ---------------------------------------------------------------- robot
    A("BOOT", "species", p_boot, state=S.IDLE, duration=2.4, species=frozenset({"robot"})),
    A("SCAN", "species", p_scan, track="upper", duration=2.4, cooldown=20, species=frozenset({"robot"}),
      tags={"idle", "explore", "focus"}, weight=5),
    A("PROCESSING", "species", p_processing, track="upper", duration=2.4, species=frozenset({"robot"}),
      tags={"idle", "focus"}, weight=3),
    A("HEAD_ROTATION", "species", p_head_rot, track="upper", duration=2.0, cooldown=25, species=frozenset({"robot"}),
      tags={"idle"}, weight=3),
    A("MECHANICAL_WALK", "species", None, motion="walk", priority=P.LOCOMOTION, duration=0,
      species=frozenset({"robot"})),
    A("MECHANICAL_RUN", "species", None, motion="run", priority=P.LOCOMOTION, duration=0,
      species=frozenset({"robot"})),
    A("CHARGE", "species", p_charge, state=S.SITTING, duration=8, cooldown=120, species=frozenset({"robot"}),
      tags={"idle", "sleep"}, weight=2, energy=-20),
    A("LOW_BATTERY", "species", p_low_batt, state=S.IDLE, duration=2.5, species=frozenset({"robot"}),
      tags={"sleep"}),
    A("ERROR", "species", p_error, state=S.IDLE, duration=1.4, cooldown=300, species=frozenset({"robot"}),
      events=((0.2, SPARKS), (0.6, SPARKS))),
    A("RECALIBRATE", "species", p_recal, track="upper", duration=2.0, cooldown=60, species=frozenset({"robot"}),
      tags={"idle"}, weight=2),
    A("LASER_LOOK", "species", p_laser, track="upper", duration=1.6, species=frozenset({"robot"}),
      priority=P.INTERACTION),
    A("SYSTEM_CHECK", "species", p_syscheck, state=S.IDLE, duration=2.4, cooldown=90, species=frozenset({"robot"}),
      tags={"idle"}, weight=2),
    A("SLEEP_MODE", "species", None, state=S.SLEEPING, loop=True, duration=30, species=frozenset({"robot"})),
    # ---------------------------------------------------------------- human
    A("THINK", "species", p_think, track="upper", duration=2.8, cooldown=20, species=frozenset({"buddy"}),
      tags={"idle", "focus"}, weight=4),
    A("READ", "species", p_reading, state=S.READING, duration=12, species=frozenset({"buddy", "robot"}),
      tags={"idle", "focus"}, weight=6, energy=-1),
    A("WRITE", "species", p_write, state=S.SITTING, duration=8, species=frozenset({"buddy"}), tags={"focus", "idle"},
      weight=4),
]

ACTIONS: dict[str, ActionDef] = {a.name: a for a in _ACTIONS}

# species substitution: what a character does instead of an action it cannot perform
SUBSTITUTE: dict[tuple[str, str], str] = {
    ("robot", "GROOM"): "SYSTEM_CHECK", ("buddy", "GROOM"): "STRETCH", ("dog", "GROOM"): "SCRATCH",
    ("robot", "SNIFF"): "SCAN", ("buddy", "SNIFF"): "LOOK_AROUND",
    ("robot", "SCRATCH"): "RECALIBRATE", ("buddy", "SCRATCH"): "THINK",
    ("robot", "SCRATCH_EAR"): "RECALIBRATE", ("buddy", "SCRATCH_EAR"): "THINK",
    ("robot", "SHAKE"): "RECALIBRATE", ("buddy", "SHAKE"): "STRETCH",
    ("robot", "TAIL_WAG"): "HEAD_ROTATION", ("buddy", "TAIL_WAG"): "WAVE", ("rabbit", "TAIL_WAG"): "NOSE_TWITCH",
    ("robot", "TAIL_SWISH"): "PROCESSING", ("buddy", "TAIL_SWISH"): "NOD", ("rabbit", "TAIL_SWISH"): "NOSE_TWITCH",
    ("robot", "TAIL_SWING"): "HEAD_ROTATION", ("buddy", "TAIL_SWING"): "NOD", ("rabbit", "TAIL_SWING"): "EAR_TWITCH",
    ("robot", "ROLL"): "SPIN", ("buddy", "ROLL"): "SPIN",
    ("robot", "LICK_PAW"): "SYSTEM_CHECK", ("dog", "LICK_PAW"): "SCRATCH", ("fox", "LICK_PAW"): "GROOM",
    ("rabbit", "LICK_PAW"): "GROOM", ("buddy", "LICK_PAW"): "STRETCH",
    ("robot", "PANT"): "PROCESSING", ("buddy", "PANT"): "HAPPY", ("cat", "PANT"): "TAIL_SWISH",
    ("fox", "PANT"): "TAIL_SWING", ("rabbit", "PANT"): "NOSE_TWITCH",
    ("robot", "PLAY_BOW"): "SYSTEM_CHECK", ("buddy", "PLAY_BOW"): "STRETCH", ("cat", "PLAY_BOW"): "STRETCH",
    ("rabbit", "PLAY_BOW"): "BOUNCE",
    ("robot", "POUNCE"): "JUMP", ("buddy", "POUNCE"): "JUMP", ("dog", "POUNCE"): "PLAY_BOW",
    ("rabbit", "POUNCE"): "DOUBLE_HOP",
    ("robot", "ARCH_BACK"): "RECALIBRATE", ("buddy", "ARCH_BACK"): "STRETCH", ("dog", "ARCH_BACK"): "STRETCH",
    ("fox", "ARCH_BACK"): "STRETCH", ("rabbit", "ARCH_BACK"): "STRETCH",
    ("robot", "LOAF"): "CHARGE", ("buddy", "LOAF"): "SIT", ("dog", "LOAF"): "DOWN", ("fox", "LOAF"): "LIE_DOWN",
    ("robot", "DOWN"): "CHARGE", ("buddy", "DOWN"): "SIT", ("cat", "DOWN"): "LOAF", ("fox", "DOWN"): "LIE_DOWN",
    ("rabbit", "DOWN"): "LOAF",
    ("robot", "PAW"): "WAVE", ("buddy", "PAW"): "WAVE",
    ("robot", "LIE_DOWN"): "CHARGE", ("buddy", "LIE_DOWN"): "SIT",
    ("robot", "PAW_AT_CURSOR"): "LASER_LOOK", ("buddy", "PAW_AT_CURSOR"): "POINT", ("rabbit", "PAW_AT_CURSOR"): "SNIFF",
    ("robot", "DOUBLE_HOP"): "HOP", ("buddy", "DOUBLE_HOP"): "HOP", ("cat", "DOUBLE_HOP"): "HOP",
    ("dog", "DOUBLE_HOP"): "BOUNCE", ("fox", "DOUBLE_HOP"): "HOP",
    ("robot", "NOSE_TWITCH"): "PROCESSING", ("buddy", "NOSE_TWITCH"): "NOD",
    ("cat", "NOSE_TWITCH"): "EAR_TWITCH", ("dog", "NOSE_TWITCH"): "SNIFF", ("fox", "NOSE_TWITCH"): "EAR_TWITCH",
    ("robot", "FREEZE"): "PROCESSING", ("buddy", "FREEZE"): "SURPRISED", ("cat", "FREEZE"): "LOOK_UP",
    ("dog", "FREEZE"): "LOOK_UP", ("fox", "FREEZE"): "ALERT",
    ("robot", "SNEAK"): "MECHANICAL_WALK", ("buddy", "SNEAK"): "WALK", ("dog", "SNEAK"): "WALK",
    ("rabbit", "SNEAK"): "WALK", ("robot", "QUIET_WALK"): "MECHANICAL_WALK", ("buddy", "QUIET_WALK"): "WALK",
    ("dog", "QUIET_WALK"): "WALK", ("rabbit", "QUIET_WALK"): "WALK",
    ("robot", "THINK"): "PROCESSING", ("cat", "THINK"): "CURIOUS", ("dog", "THINK"): "CURIOUS",
    ("fox", "THINK"): "LOOK_AROUND", ("rabbit", "THINK"): "CURIOUS",
    ("cat", "READ"): "LOAF", ("dog", "READ"): "DOWN", ("fox", "READ"): "SIT", ("rabbit", "READ"): "LOAF",
    ("cat", "WRITE"): "SIT", ("dog", "WRITE"): "SIT", ("fox", "WRITE"): "SIT", ("rabbit", "WRITE"): "SIT",
    ("robot", "WRITE"): "PROCESSING",
    ("cat", "THUMBS_UP"): "NOD", ("dog", "THUMBS_UP"): "PAW", ("fox", "THUMBS_UP"): "NOD",
    ("rabbit", "THUMBS_UP"): "NOD",
    ("buddy", "SCAN"): "LOOK_AROUND", ("buddy", "PROCESSING"): "THINK", ("buddy", "HEAD_ROTATION"): "SHAKE_HEAD",
    ("buddy", "CHARGE"): "SIT", ("buddy", "LOW_BATTERY"): "SLEEPY", ("buddy", "ERROR"): "CONFUSED",
    ("buddy", "RECALIBRATE"): "STRETCH", ("buddy", "LASER_LOOK"): "POINT", ("buddy", "SYSTEM_CHECK"): "STRETCH",
    ("buddy", "SLEEP_MODE"): "SLEEP", ("buddy", "BOOT"): "WAKE", ("buddy", "MECHANICAL_WALK"): "WALK",
    ("buddy", "MECHANICAL_RUN"): "RUN",
}
_ANIMAL_FROM_ROBOT = {"SCAN": "SNIFF", "PROCESSING": "CURIOUS", "HEAD_ROTATION": "LOOK_AROUND", "CHARGE": "LOAF",
                      "LOW_BATTERY": "SLEEPY", "ERROR": "SURPRISED", "RECALIBRATE": "SHAKE", "LASER_LOOK": "LOOK_UP",
                      "SYSTEM_CHECK": "STRETCH", "SLEEP_MODE": "SLEEP", "BOOT": "WAKE", "MECHANICAL_WALK": "WALK",
                      "MECHANICAL_RUN": "RUN"}
for _sp in ("cat", "dog", "fox", "rabbit"):
    for _a, _b in _ANIMAL_FROM_ROBOT.items():
        SUBSTITUTE.setdefault((_sp, _a), _b)
SUBSTITUTE.setdefault(("dog", "LOAF"), "DOWN")
SUBSTITUTE.setdefault(("robot", "LOAF"), "CHARGE")
SUBSTITUTE[("dog", "GROOM")] = "SCRATCH"
SUBSTITUTE[("rabbit", "LOAF")] = "LOAF"
SUBSTITUTE.update({
    ("dog", "HEAD_BUMP"): "LEAN_TO_CURSOR", ("fox", "HEAD_BUMP"): "LEAN_TO_CURSOR",
    ("rabbit", "HEAD_BUMP"): "LEAN_TO_CURSOR", ("robot", "HEAD_BUMP"): "LEAN_TO_CURSOR",
    ("buddy", "HEAD_BUMP"): "LEAN_TO_CURSOR",
    ("dog", "PURR"): "TAIL_WAG", ("fox", "PURR"): "TAIL_SWING", ("rabbit", "PURR"): "NOSE_TWITCH",
    ("robot", "PURR"): "PROCESSING", ("buddy", "PURR"): "NOD",
    ("robot", "SNEEZE"): "ERROR", ("robot", "SHIVER"): "RECALIBRATE",
})

# alternative names (spec / artist vocabulary) -> library actions
ALIASES = {
    "QUICK_TURN": "TURN_HOP", "STOP_SKID": "SKID_STOP", "SHAKE_BODY": "SHAKE", "FALL": "FALL_OVER",
    "PROCESS": "PROCESSING", "HEAD_SCAN": "HEAD_ROTATION", "BATTERY_LOW": "LOW_BATTERY", "SYSTEM_ERROR": "ERROR",
    "STRETCH_CAT": "ARCH_BACK", "BAT_AT_CURSOR": "PAW_AT_CURSOR", "TAIL_FLICK": "TAIL_SWISH", "BIG_JUMP": "HIGH_JUMP",
    "SCRATCH_EAR": "SCRATCH_EAR", "STAND_UP": "STAND", "LIE": "LIE_DOWN", "WAKE_UP": "WAKE", "COOLDOWN": "COOL_DOWN",
}
SUBSTITUTE.update({
    ("cat", "LOOK_AT_WATCH"): "LOOK_UP", ("dog", "LOOK_AT_WATCH"): "LOOK_UP", ("fox", "LOOK_AT_WATCH"): "LOOK_UP",
    ("rabbit", "LOOK_AT_WATCH"): "LOOK_UP", ("robot", "LOOK_AT_WATCH"): "SYSTEM_CHECK",
    ("cat", "CHECK_PHONE"): "LOOK_AROUND", ("dog", "CHECK_PHONE"): "SNIFF", ("fox", "CHECK_PHONE"): "LISTEN",
    ("rabbit", "CHECK_PHONE"): "NOSE_TWITCH", ("robot", "CHECK_PHONE"): "PROCESSING",
    ("cat", "FIST_PUMP"): "BOUNCE", ("dog", "FIST_PUMP"): "EXCITED_JUMP", ("fox", "FIST_PUMP"): "BOUNCE",
    ("rabbit", "FIST_PUMP"): "BOUNCE",
    ("robot", "LISTEN"): "SCAN", ("buddy", "LISTEN"): "THINK",
    ("dog", "TAIL_WRAP"): "SIT", ("rabbit", "TAIL_WRAP"): "LOAF", ("robot", "TAIL_WRAP"): "SIT",
    ("buddy", "TAIL_WRAP"): "SIT",
    ("dog", "CURL_UP"): "DOWN", ("rabbit", "CURL_UP"): "LOAF", ("robot", "CURL_UP"): "CHARGE",
    ("buddy", "CURL_UP"): "SIT",
    ("cat", "EXCITED_JUMP"): "JUMP", ("fox", "EXCITED_JUMP"): "JUMP", ("rabbit", "EXCITED_JUMP"): "DOUBLE_HOP",
    ("robot", "EXCITED_JUMP"): "JUMP", ("buddy", "EXCITED_JUMP"): "JUMP_STAR",
})

# action variants: the same intent performed differently (chosen by personality / mood / energy)
VARIANTS: dict[str, tuple[str, ...]] = {
    "JUMP": ("JUMP", "JUMP_SPIN", "JUMP_STAR"),
    "TURN": ("TURN", "TURN_HOP", "TURN_LOOK"),
    "CELEBRATE": ("CELEBRATE", "CHEER", "DANCE"),
    "LAUGH": ("LAUGH", "LAUGH_HARD"),
    "SHOUT": ("SHOUT_SOFT", "SHOUT", "SHOUT_BIG"),
}


def pick_variant(name: str, rng: random.Random, *, mood: float = 0.0, energy: float = 50.0,
                 playfulness: float = 0.5, species: str = "cat", ready=None, now: float = 0.0) -> str:
    """Pick one variant of ``name``: playful / happy / energetic characters choose the flashier ones."""
    opts = VARIANTS.get(name.upper())
    if not opts:
        return name
    e = max(0.0, min(1.0, energy / 100.0))
    md = max(-1.0, min(1.0, mood))
    weights = []
    for i, v in enumerate(opts):
        if resolve(v, species) is None or (ready is not None and not ready(v, now)):
            continue
        if i == 0:
            w = 1.0 + max(0.0, -md) * 0.8 + (1 - e) * 0.6           # plain version: tired / grumpy
        else:
            w = (0.35 + playfulness * 0.8) * (0.5 + e) * (1 + 0.6 * max(0.0, md)) / i
        weights.append((v, w))
    if not weights:
        return opts[0]
    total = sum(w for _, w in weights)
    x = rng.uniform(0, total)
    for v, w in weights:
        x -= w
        if x <= 0:
            return v
    return weights[-1][0]


def species_of(profile_or_id) -> str:
    sid = getattr(profile_or_id, "id", profile_or_id) or "cat"
    return sid if sid in QUADS | BIPEDS else "cat"


def resolve(name: str, species: str) -> Optional[ActionDef]:
    """Action ``name`` as performed by ``species`` (with substitution), or None."""
    seen = set()
    n = name.upper().replace(" ", "_")
    n = ALIASES.get(n, n)
    while n not in seen:
        seen.add(n)
        a = ACTIONS.get(n)
        if a is None:
            return None
        if a.species is None or species in a.species:
            return a
        n = SUBSTITUTE.get((species, n), "")
        if not n:
            return None
    return None


def available(species: str) -> list[str]:
    """Every action name this species can perform natively."""
    return [a.name for a in _ACTIONS if a.species is None or species in a.species]


def species_actions(species: str) -> list[str]:
    return [a.name for a in _ACTIONS if a.species is not None and species in a.species and a.category == "species"]


# =========================================================================
@dataclass
class ActiveAction:
    adef: ActionDef
    requested: str
    start: float
    duration: float
    priority: Priority
    stop_at: Optional[float] = None
    fired: set = field(default_factory=set)
    weight: float = 0.0
    t: float = 0.0
    u: float = 0.0

    @property
    def name(self) -> str:
        return self.adef.name

    @property
    def track(self) -> str:
        return self.adef.track

    def stopping(self) -> bool:
        return self.stop_at is not None


class ActionPlayer:
    MAX_WAIT = 0.8

    def __init__(self, species: str = "cat", rng: random.Random | None = None):
        self.species = species
        self.rng = rng or random.Random()
        self.active: list[ActiveAction] = []
        self.pending: dict[str, tuple] = {}
        self.cooldowns: dict[str, float] = {}
        self.cooldown_scale = 1.0
        self.blend_scale = 1.0          # Playground "blend time" (x blend in / out of every action)
        self.events: list[tuple[str, str]] = []
        self.history: list[tuple[float, str]] = []

    # ---------------------------------------------------------- queries
    def current(self, track: str = "full") -> Optional[ActiveAction]:
        for a in reversed(self.active):
            if a.track == track and not a.stopping():
                return a
        return None

    def names(self) -> list[str]:
        return [a.name for a in self.active if not a.stopping()]

    def ready(self, name: str, now: float) -> bool:
        a = resolve(name, self.species)
        return a is not None and now >= self.cooldowns.get(a.name, -1e9)

    def at_exit(self, a: ActiveAction, now: float) -> bool:
        if not a.adef.exits:
            return True
        u = self._u(a, now)
        return any(abs(u - e) < 0.06 for e in a.adef.exits) or u >= 0.98

    def _u(self, a: ActiveAction, now: float) -> float:
        t = now - a.start
        if a.duration <= 0:
            return 1.0
        if a.adef.loop:
            return (t % a.duration) / a.duration
        return max(0.0, min(1.0, t / a.duration))

    # ---------------------------------------------------------- control
    def play(self, name: str, now: float, priority: Priority | None = None, duration: float | None = None,
             force: bool = False, ignore_cooldown: bool = False) -> Optional[ActiveAction]:
        """Start an action (resolved for this species). Returns the ActiveAction, or None if refused
        (unknown, on cooldown, or a higher-priority action owns the track). Equal / higher priority
        requests wait for the playing action's next exit point (``pending``)."""
        a = resolve(name, self.species)
        if a is None:
            return None
        pr = a.priority if priority is None else Priority(priority)
        if not force and not ignore_cooldown and now < self.cooldowns.get(a.name, -1e9):
            return None
        dur = a.duration if duration is None else duration
        if dur is None or dur <= 0:
            dur = max(0.0, dur or 0.0)
        new = ActiveAction(a, name.upper(), now, dur, pr)
        if a.track == "additive":
            # additive layers stack (ear twitch + tail swing + nod ...); only restart the same one
            for other in self.active:
                if other.name == a.name and not other.stopping():
                    self._stop(other, now)
            self._start(new, now)
            return new
        cur = self.current(a.track)
        if cur is not None:
            if cur.priority > pr and not force:
                return None
            if pr < Priority.CRITICAL_REACTION and not force and not self.at_exit(cur, now):
                self.pending[a.track] = (new, now)
                return new
            self._stop(cur, now)
        self._start(new, now)
        return new

    def _start(self, new: ActiveAction, now: float) -> None:
        new.start = now
        self.active.append(new)
        cd = new.adef.cooldown * self.cooldown_scale
        if cd > 0:
            self.cooldowns[new.adef.name] = now + cd
        self.events.append((ACTION_START, new.name))
        self.history.append((now, new.name))
        del self.history[:-50]

    def _stop(self, a: ActiveAction, now: float) -> None:
        if a.stop_at is None:
            a.stop_at = now

    def stop(self, name_or_track: str, now: float) -> None:
        for a in self.active:
            if a.track == name_or_track or a.name == name_or_track.upper() or a.requested == name_or_track.upper():
                self._stop(a, now)
        self.pending.pop(name_or_track, None)

    def stop_all(self, now: float, below: Priority = Priority.CRITICAL_REACTION) -> None:
        for a in self.active:
            if a.priority < below:
                self._stop(a, now)
        self.pending.clear()

    def blocking_priority(self) -> Priority:
        cur = self.current("full")
        return cur.priority if cur is not None else Priority.MICRO_MOVEMENT

    # ----------------------------------------------------------- update
    def update(self, now: float) -> list[tuple[str, str]]:
        for track, (new, asked) in list(self.pending.items()):
            cur = self.current(track)
            if cur is None or self.at_exit(cur, now) or now - asked >= self.MAX_WAIT:
                if cur is not None:
                    self._stop(cur, now)
                del self.pending[track]
                self._start(new, now)
        keep = []
        for a in self.active:
            ad = a.adef
            a.t = now - a.start
            a.u = self._u(a, now)
            if not ad.loop and a.duration > 0 and a.stop_at is None and \
                    a.t >= a.duration - ad.blend_out * self.blend_scale * 0.5:
                a.stop_at = now
            for (eu, ev) in ad.events:
                key = (eu, ev, int(a.t / a.duration) if ad.loop and a.duration > 0 else 0)
                if a.u >= eu and key not in a.fired:
                    a.fired.add(key)
                    self.events.append((ev, a.name))
            bi, bo = ad.blend_in * self.blend_scale, ad.blend_out * self.blend_scale
            w_in = _ss(a.t / bi) if bi > 0 else 1.0
            w_out = 1.0
            if a.stop_at is not None:
                w_out = 1.0 - _ss((now - a.stop_at) / max(1e-3, bo))
            if a.duration <= 0 and a.stop_at is None:
                a.stop_at = now
            a.weight = max(0.0, min(1.0, w_in * w_out))
            if a.stop_at is not None and now - a.stop_at >= bo:
                self.events.append((ACTION_END, a.name))
                continue
            keep.append(a)
        self.active = keep
        ev, self.events = self.events, []
        return ev

    def layers(self) -> list[ActiveAction]:
        order = {"full": 0, "upper": 1, "additive": 2}
        return sorted((a for a in self.active if a.weight > 0.001), key=lambda a: (order.get(a.track, 3), a.start))
