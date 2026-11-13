"""EmotionController: how the companion feels, and how that shows.

An emotion is a *style* (ears, eye size, brows, mouth, tail wag, bounce,
blink rate, movement speed) that is cross-faded onto whatever the body is
doing, so a happy walk and a worried walk use the same walk cycle.
Emotions decay back to the character's baseline after their hold time.

Living-character upgrade
------------------------
* Every style also carries **continuous face and posture parameters**
  (smile, mouth_open, mouth_width, lid_upper / lid_lower / lid_tilt, pupil,
  brow_raise, cheek, slump, chest) plus a valence / arousal pair.  Faces are
  composed from these numbers, so emotions blend instead of swapping drawings.
* **Intensity 0..1**: 0.2 angry = mildly annoyed, 1.0 = exaggerated cartoon
  anger (marks, trembling and flushing only appear at higher intensities).
* **Transitions**: big jumps in valence are routed through bridging
  emotions (ANGRY -> CALMING -> target, SCARED -> RELIEVED -> target,
  ANGRY -> RELIEVED -> HAPPY -> CELEBRATING) and fade more slowly.
* **Mixing**: ``set_mix({HAPPY: 0.8, ANGRY: 0.2})`` blends several emotions at
  once (the Playground's emotion mixer).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, fields, replace
from typing import Optional


class Emotion(str, enum.Enum):
    NEUTRAL = "NEUTRAL"
    HAPPY = "HAPPY"
    CURIOUS = "CURIOUS"
    ALERT = "ALERT"
    SUSPICIOUS = "SUSPICIOUS"
    EXCITED = "EXCITED"
    SLEEPY = "SLEEPY"
    WORRIED = "WORRIED"
    ANNOYED = "ANNOYED"
    CELEBRATING = "CELEBRATING"
    SAD = "SAD"
    # --- emotion pack ---
    ANGRY = "ANGRY"
    GRUMPY = "GRUMPY"
    SCARED = "SCARED"
    EMBARRASSED = "EMBARRASSED"
    LOVE = "LOVE"
    BORED = "BORED"
    SMUG = "SMUG"
    DISGUSTED = "DISGUSTED"
    SHOCKED = "SHOCKED"
    LAUGHING = "LAUGHING"
    DETERMINED = "DETERMINED"
    DIZZY = "DIZZY"
    RELIEVED = "RELIEVED"
    PROUD = "PROUD"
    # --- living-character upgrade ---
    SURPRISED = "SURPRISED"
    CONFUSED = "CONFUSED"
    CALMING = "CALMING"
    CONFIDENT = "CONFIDENT"
    LONELY = "LONELY"
    PLAYFUL = "PLAYFUL"
    JEALOUS = "JEALOUS"
    IMPATIENT = "IMPATIENT"
    RELAXED = "RELAXED"
    FOCUSED = "FOCUSED"
    NERVOUS = "NERVOUS"
    GRATEFUL = "GRATEFUL"


@dataclass(frozen=True)
class EmotionStyle:
    ear: float = 0.0            # -1 droop .. +1 perked
    eye_scale: float = 1.0
    eye_open: float = 1.0       # lid opening multiplier
    brow: float = 0.0           # -1 angry/determined .. +1 worried/sad
    tail_amp: float = 6.0       # wag amplitude (deg)
    tail_hz: float = 0.6
    bounce: float = 0.0         # idle bounce height (units)
    bounce_hz: float = 2.0
    blush: float = 0.5
    speed_mult: float = 1.0
    blink_mult: float = 1.0     # >1 blinks more often
    head_tilt: float = 0.0
    stillness: float = 0.0      # 1 = freeze secondary motion (alert)
    mouth: Optional[str] = None
    eye_style: Optional[str] = None
    tremble: float = 0.0        # body shiver amplitude (units) - scared, furious
    flush: float = 0.0          # red face tint 0..1 - angry, embarrassed
    mark: Optional[str] = None  # persistent symbol (anger vein, gloom, hearts ...)
    name: Optional[str] = None  # the emotion this style shows (set for every entry of STYLES)
    # continuous face rig (added to the pose; 0 = neutral)
    smile: float = 0.0          # -1 frown .. +1 big smile
    mouth_open: float = 0.0     # 0..1 (jaw follows)
    mouth_width: float = 0.0    # -0.5 small / pursed .. +0.5 wide
    lid_upper: float = 0.0      # 0 open .. 1 closed (upper eyelid)
    lid_lower: float = 0.0      # 0 .. 1 cheek-raised lower lid (happy squint)
    lid_tilt: float = 0.0       # -1 angry (inner corner down) .. +1 sad (inner corner up)
    pupil: float = 1.0          # pupil size: <1 tiny (fear / shock), >1 big (love / cute)
    brow_raise: float = 0.0     # -1 lowered .. +1 raised
    cheek: float = 0.0          # cheek raise 0..1
    # posture
    slump: float = 0.0          # 0..1 drooping body / head (sad, bored, lonely)
    chest: float = 0.0          # -1 shrinking .. +1 puffed up (proud, angry, confident)
    # affect (used for transitions and mood)
    valence: float = 0.0        # -1 unpleasant .. +1 pleasant
    arousal: float = 0.3        # 0 calm .. 1 agitated
    # full-body language (the pose library): readable with every icon / effect switched off
    lean: float = 0.0           # deg: + leans forward (eager, angry), - pulls back (scared, proud)
    crouch: float = 0.0         # knees bend / body sinks (scared, angry coil); negative = up on its toes
    head_drop: float = 0.0      # units: + head hangs low (sad, sleepy), - held high (alert, proud)
    head_fwd: float = 0.0       # units: + head pushed forward (angry, curious), - pulled back (scared)
    head_pitch: float = 0.0     # deg: + chin down (glare, sad), - chin up (confident)
    head_roll: float = 0.0      # deg: cocked head (curious, confused, love)
    shoulders: float = 0.0      # -1 drooped .. +1 raised / tense
    hunch: float = 0.0          # 0..1 curl in on itself (scared, nervous, embarrassed)
    fists: float = 0.0          # clenched hands / braced paws
    akimbo: float = 0.0         # hands on hips
    hug: float = 0.0            # arms pulled to the chest
    arms_out: float = 0.0       # arms open
    tail_pose: float = 0.0      # -1 tucked .. +1 high
    stance: float = 0.0         # -1 narrow .. +1 wide planted
    weight: float = 0.0         # -1 weight back .. +1 weight forward (body over the feet)
    sway: float = 0.0           # slow idle sway (bored, sleepy, happy humming)
    ear_asym: float = 0.0       # one ear up, one down (confused)
    breath_rate: float = 1.0    # x breathing rate (scared fast / sleep slow)
    breath_depth: float = 1.0   # x breathing depth (angry strong / scared shallow)
    blink_speed: float = 1.0    # x blink speed (nervous rapid, sleepy slow)


STYLES: dict[Emotion, EmotionStyle] = {
    Emotion.NEUTRAL: EmotionStyle(mouth="smile", blush=0.6),      # resting face: a soft, friendly smile
    Emotion.HAPPY: EmotionStyle(ear=0.3, tail_amp=18, tail_hz=1.6, bounce=1.2, bounce_hz=2.2, blush=0.85,
                                speed_mult=1.05, mouth="smile"),
    Emotion.CURIOUS: EmotionStyle(ear=0.7, eye_scale=1.08, tail_amp=8, tail_hz=0.9, head_tilt=9, mouth="o"),
    Emotion.ALERT: EmotionStyle(ear=1.0, eye_scale=1.18, eye_open=1.1, tail_amp=2, tail_hz=0.3, stillness=0.8,
                                blink_mult=0.4, mouth="o", eye_style="wide"),
    Emotion.SUSPICIOUS: EmotionStyle(ear=0.6, eye_open=0.62, brow=-0.5, tail_amp=6, tail_hz=2.4, mouth="flat",
                                     stillness=0.4),
    Emotion.EXCITED: EmotionStyle(ear=0.9, eye_scale=1.12, tail_amp=28, tail_hz=3.2, bounce=3.2, bounce_hz=3.4,
                                  blush=1.0, speed_mult=1.25, mouth="grin"),
    Emotion.SLEEPY: EmotionStyle(ear=-0.6, eye_open=0.45, tail_amp=3, tail_hz=0.3, speed_mult=0.7,
                                 blink_mult=2.2, head_tilt=-5, mouth="sleep"),
    Emotion.WORRIED: EmotionStyle(ear=-0.3, brow=0.8, eye_scale=1.05, tail_amp=4, tail_hz=1.2, mouth="w"),
    Emotion.ANNOYED: EmotionStyle(ear=0.2, eye_open=0.75, brow=-0.9, tail_amp=10, tail_hz=3.0, mouth="flat"),
    Emotion.CELEBRATING: EmotionStyle(ear=1.0, tail_amp=32, tail_hz=3.6, bounce=2.0, bounce_hz=3.0, blush=1.0,
                                      speed_mult=1.2, mouth="grin", eye_style="happy"),
    Emotion.SAD: EmotionStyle(ear=-0.9, brow=0.9, eye_open=0.8, tail_amp=2, tail_hz=0.3, speed_mult=0.8,
                              head_tilt=-6, mouth="frown"),
    # --- emotion pack (cartoon emotions; anger is playful grumpiness, never hostile) ---
    Emotion.ANGRY: EmotionStyle(ear=-0.2, eye_open=0.8, brow=-1.0, tail_amp=22, tail_hz=3.8, speed_mult=1.15,
                                stillness=0.2, mouth="teeth", eye_style="angry", tremble=0.5, flush=0.8,
                                mark="anger"),
    Emotion.GRUMPY: EmotionStyle(ear=-0.1, eye_open=0.7, brow=-0.7, tail_amp=10, tail_hz=2.2, speed_mult=0.9,
                                 mouth="pout", eye_style="half", flush=0.3),
    Emotion.SCARED: EmotionStyle(ear=-1.0, eye_scale=1.2, brow=0.9, tail_amp=2, tail_hz=0.5, speed_mult=1.3,
                                 blink_mult=0.5, mouth="wobble", eye_style="shock", tremble=0.8, mark="sweat"),
    Emotion.EMBARRASSED: EmotionStyle(ear=-0.5, brow=0.6, blush=1.0, tail_amp=4, tail_hz=1.0, head_tilt=-8,
                                      mouth="w", flush=0.9, mark="sweat"),
    Emotion.LOVE: EmotionStyle(ear=0.4, blush=1.0, tail_amp=14, tail_hz=1.0, bounce=0.8, bounce_hz=1.4,
                               head_tilt=7, mouth="smile", eye_style="heart", mark="heart"),
    Emotion.BORED: EmotionStyle(ear=-0.5, eye_open=0.5, tail_amp=3, tail_hz=0.25, speed_mult=0.75, blink_mult=1.6,
                                head_tilt=-4, mouth="flat", eye_style="half"),
    Emotion.SMUG: EmotionStyle(ear=0.5, eye_open=0.7, brow=-0.2, tail_amp=10, tail_hz=0.8, head_tilt=-5,
                               mouth="smirk", eye_style="half", blush=0.7),
    Emotion.DISGUSTED: EmotionStyle(ear=-0.6, eye_open=0.65, brow=-0.4, tail_amp=4, tail_hz=1.5, head_tilt=-6,
                                    mouth="tongue", eye_style="half"),
    Emotion.SHOCKED: EmotionStyle(ear=1.0, eye_scale=1.3, eye_open=1.15, brow=0.6, tail_amp=0, tail_hz=0.1,
                                  stillness=0.9, blink_mult=0.3, mouth="shout", eye_style="shock", mark="!!"),
    Emotion.LAUGHING: EmotionStyle(ear=0.6, blush=1.0, tail_amp=24, tail_hz=3.0, bounce=1.6, bounce_hz=5.0,
                                   head_tilt=-6, mouth="laugh", eye_style="happy", mark="haha"),
    Emotion.DETERMINED: EmotionStyle(ear=0.9, eye_open=0.9, brow=-0.8, tail_amp=6, tail_hz=1.2, speed_mult=1.2,
                                     mouth="flat", eye_style="focused", mark="sparkle"),
    Emotion.DIZZY: EmotionStyle(ear=-0.4, tail_amp=8, tail_hz=0.7, speed_mult=0.6, head_tilt=10, mouth="w",
                                eye_style="spiral", mark="stars"),
    Emotion.RELIEVED: EmotionStyle(ear=0.2, eye_open=0.6, brow=0.3, tail_amp=8, tail_hz=0.8, blush=0.7,
                                   mouth="smile", eye_style="closed", mark="sigh"),
    Emotion.PROUD: EmotionStyle(ear=1.0, eye_open=0.85, tail_amp=12, tail_hz=1.0, head_tilt=-6, blush=0.8,
                                mouth="smile", eye_style="happy", mark="sparkle"),
}

# continuous face / posture / affect parameters per emotion (merged into STYLES below)
FACE = {
    Emotion.NEUTRAL: dict(smile=0.32, cheek=0.14, lid_lower=0.05, valence=0.25, arousal=0.3),
    Emotion.HAPPY: dict(smile=0.7, cheek=0.4, lid_lower=0.15, brow_raise=0.2, valence=0.7, arousal=0.5),
    Emotion.CURIOUS: dict(mouth_open=0.2, mouth_width=-0.3, brow_raise=0.6, pupil=1.1, valence=0.3, arousal=0.5),
    Emotion.ALERT: dict(mouth_open=0.25, mouth_width=-0.35, brow_raise=0.8, pupil=0.85, valence=-0.1, arousal=0.85),
    Emotion.SUSPICIOUS: dict(smile=-0.1, lid_upper=0.35, lid_tilt=-0.4, brow_raise=-0.3, valence=-0.3, arousal=0.5),
    Emotion.EXCITED: dict(smile=0.9, mouth_open=0.5, mouth_width=0.3, cheek=0.6, brow_raise=0.6, pupil=1.15,
                          chest=0.3, valence=0.8, arousal=0.95),
    Emotion.SLEEPY: dict(lid_upper=0.55, brow_raise=-0.2, slump=0.4, valence=0.1, arousal=0.05),
    Emotion.WORRIED: dict(smile=-0.35, lid_tilt=0.6, brow_raise=0.4, pupil=0.95, valence=-0.4, arousal=0.6),
    Emotion.ANNOYED: dict(smile=-0.3, lid_upper=0.25, lid_tilt=-0.6, brow_raise=-0.5, valence=-0.5, arousal=0.6),
    Emotion.CELEBRATING: dict(smile=1.0, mouth_open=0.55, mouth_width=0.35, cheek=0.8, lid_lower=0.4, chest=0.4,
                              valence=1.0, arousal=1.0),
    Emotion.SAD: dict(smile=-0.7, lid_upper=0.25, lid_tilt=0.9, brow_raise=0.3, slump=0.6, chest=-0.3,
                      valence=-0.7, arousal=0.2),
    Emotion.ANGRY: dict(smile=-0.6, mouth_open=0.25, mouth_width=0.3, lid_upper=0.35, lid_tilt=-1.0,
                        brow_raise=-0.9, chest=0.6, valence=-0.8, arousal=0.95),
    Emotion.GRUMPY: dict(smile=-0.5, mouth_width=-0.3, lid_upper=0.45, lid_tilt=-0.5, brow_raise=-0.6, chest=0.2,
                         valence=-0.5, arousal=0.4),
    Emotion.SCARED: dict(smile=-0.4, mouth_open=0.35, mouth_width=-0.2, lid_tilt=0.7, pupil=0.55, brow_raise=0.9,
                         chest=-0.6, valence=-0.8, arousal=0.95),
    Emotion.EMBARRASSED: dict(smile=0.15, mouth_width=-0.3, lid_lower=0.3, lid_tilt=0.5, brow_raise=0.3, cheek=0.5,
                              slump=0.25, valence=-0.2, arousal=0.6),
    Emotion.LOVE: dict(smile=0.8, cheek=0.7, lid_lower=0.35, pupil=1.35, brow_raise=0.3, valence=0.9, arousal=0.55),
    Emotion.BORED: dict(smile=-0.2, mouth_width=-0.1, lid_upper=0.55, brow_raise=-0.2, slump=0.55,
                        valence=-0.3, arousal=0.05),
    Emotion.SMUG: dict(smile=0.45, lid_upper=0.45, lid_tilt=-0.2, brow_raise=0.1, chest=0.4, valence=0.5,
                       arousal=0.35),
    Emotion.DISGUSTED: dict(smile=-0.6, mouth_width=0.2, lid_upper=0.35, lid_lower=0.35, brow_raise=-0.5,
                            chest=-0.2, valence=-0.7, arousal=0.55),
    Emotion.SHOCKED: dict(mouth_open=0.9, mouth_width=-0.1, pupil=0.5, brow_raise=1.0, chest=-0.2, valence=-0.3,
                          arousal=1.0),
    Emotion.LAUGHING: dict(smile=1.0, mouth_open=0.7, mouth_width=0.4, cheek=1.0, lid_lower=0.7, brow_raise=0.3,
                           valence=0.9, arousal=0.9),
    Emotion.DETERMINED: dict(smile=0.1, mouth_width=0.1, lid_upper=0.2, lid_tilt=-0.5, brow_raise=-0.6, chest=0.6,
                             valence=0.4, arousal=0.75),
    Emotion.DIZZY: dict(smile=-0.1, mouth_open=0.15, brow_raise=0.3, slump=0.3, valence=-0.2, arousal=0.4),
    Emotion.RELIEVED: dict(smile=0.5, lid_upper=0.6, lid_tilt=0.3, brow_raise=0.4, slump=0.15, valence=0.5,
                           arousal=0.2),
    Emotion.PROUD: dict(smile=0.6, lid_upper=0.3, cheek=0.5, brow_raise=0.2, chest=0.9, valence=0.8, arousal=0.55),
    Emotion.SURPRISED: dict(mouth_open=0.55, mouth_width=-0.25, pupil=0.8, brow_raise=1.0, valence=0.1, arousal=0.9),
    Emotion.CONFUSED: dict(smile=-0.15, mouth_width=-0.2, lid_tilt=0.3, brow_raise=0.5, valence=-0.1, arousal=0.4),
    Emotion.CALMING: dict(smile=0.0, lid_upper=0.4, lid_tilt=0.2, brow_raise=0.1, valence=0.0, arousal=0.25),
    Emotion.CONFIDENT: dict(smile=0.5, lid_upper=0.15, cheek=0.3, brow_raise=0.1, chest=0.7, valence=0.6,
                            arousal=0.5),
    Emotion.LONELY: dict(smile=-0.4, lid_upper=0.3, lid_tilt=0.8, brow_raise=0.4, pupil=1.15, slump=0.5,
                         chest=-0.3, valence=-0.6, arousal=0.15),
    Emotion.PLAYFUL: dict(smile=0.8, mouth_open=0.3, cheek=0.5, lid_lower=0.2, pupil=1.15, brow_raise=0.4,
                          valence=0.8, arousal=0.8),
    Emotion.JEALOUS: dict(smile=-0.4, mouth_width=-0.3, lid_upper=0.45, lid_tilt=-0.3, brow_raise=-0.4,
                          valence=-0.5, arousal=0.5),
    Emotion.IMPATIENT: dict(smile=-0.3, mouth_width=-0.2, lid_upper=0.25, lid_tilt=-0.4, brow_raise=-0.3,
                            valence=-0.4, arousal=0.75),
    Emotion.RELAXED: dict(smile=0.4, lid_upper=0.45, cheek=0.2, slump=0.2, valence=0.6, arousal=0.1),
    Emotion.FOCUSED: dict(smile=0.0, lid_upper=0.2, lid_tilt=-0.3, brow_raise=-0.3, valence=0.2, arousal=0.55),
    Emotion.NERVOUS: dict(smile=-0.2, mouth_width=-0.2, lid_tilt=0.6, pupil=0.85, brow_raise=0.6, chest=-0.3,
                          valence=-0.4, arousal=0.75),
    Emotion.GRATEFUL: dict(smile=0.8, lid_upper=0.35, lid_lower=0.3, cheek=0.7, brow_raise=0.4, valence=0.9,
                           arousal=0.4),
}

# body-language styles for the new emotions (ears, tail, bounce, marks ...)
STYLES.update({
    Emotion.SURPRISED: EmotionStyle(ear=1.0, eye_scale=1.22, eye_open=1.12, brow=0.5, tail_amp=4, tail_hz=0.5,
                                    stillness=0.6, blink_mult=0.4, mouth="o", eye_style="wide", mark="!"),
    Emotion.CONFUSED: EmotionStyle(ear=-0.2, brow=0.5, tail_amp=6, tail_hz=0.8, head_tilt=14, mouth="w", mark="?"),
    Emotion.CALMING: EmotionStyle(ear=0.0, eye_open=0.85, tail_amp=4, tail_hz=0.5, speed_mult=0.9, mark="sigh"),
    Emotion.CONFIDENT: EmotionStyle(ear=0.8, tail_amp=12, tail_hz=1.0, head_tilt=-5, blush=0.6, speed_mult=1.1,
                                    mouth="smile"),
    Emotion.LONELY: EmotionStyle(ear=-0.8, brow=0.8, eye_open=0.85, tail_amp=2, tail_hz=0.3, speed_mult=0.75,
                                 head_tilt=-7, mouth="frown", mark="gloom"),
    Emotion.PLAYFUL: EmotionStyle(ear=0.9, eye_scale=1.08, tail_amp=26, tail_hz=2.6, bounce=2.2, bounce_hz=2.8,
                                  blush=0.9, speed_mult=1.2, mouth="grin"),
    Emotion.JEALOUS: EmotionStyle(ear=-0.3, eye_open=0.7, brow=-0.5, tail_amp=12, tail_hz=2.6, head_tilt=-6,
                                  mouth="pout", eye_style="half"),
    Emotion.IMPATIENT: EmotionStyle(ear=0.3, eye_open=0.8, brow=-0.5, tail_amp=16, tail_hz=3.4, bounce=0.6,
                                    bounce_hz=4.0, mouth="flat"),
    Emotion.RELAXED: EmotionStyle(ear=0.1, eye_open=0.7, tail_amp=6, tail_hz=0.5, speed_mult=0.85, blink_mult=1.3,
                                  blush=0.7, mouth="smile"),
    Emotion.FOCUSED: EmotionStyle(ear=0.6, eye_open=0.9, brow=-0.4, tail_amp=3, tail_hz=0.4, stillness=0.5,
                                  blink_mult=0.7, mouth="flat", eye_style="focused"),
    Emotion.NERVOUS: EmotionStyle(ear=-0.5, brow=0.7, tail_amp=8, tail_hz=3.0, tremble=0.3, blink_mult=1.8,
                                  mouth="wobble", mark="sweat"),
    Emotion.GRATEFUL: EmotionStyle(ear=0.5, eye_open=0.8, tail_amp=16, tail_hz=1.4, bounce=0.8, bounce_hz=1.6,
                                   blush=1.0, head_tilt=8, mouth="smile", eye_style="happy", mark="heart"),
})
for _e, _vals in FACE.items():
    STYLES[_e] = replace(STYLES[_e], **_vals)

# ----------------------------------------------------------------- pose library
# Full-body stand poses (skeleton-level, not faces). Each emotion uses one, scaled by intensity.
POSES: dict[str, dict] = {
    "neutral_stand": {},
    "relaxed_stand": dict(lean=-1.0, crouch=0.06, head_drop=1.5, head_pitch=3, shoulders=-0.3, tail_pose=0.2,
                          sway=0.6, breath_rate=0.85, breath_depth=1.1),
    "alert_stand": dict(lean=3.0, crouch=-0.12, head_drop=-4.5, head_fwd=2.0, head_pitch=-4, shoulders=0.5,
                        tail_pose=0.6, stance=0.3, weight=0.5, breath_rate=1.3, breath_depth=0.7),
    "happy_stand": dict(lean=-2.0, head_drop=-2.5, head_pitch=-3, head_roll=6, shoulders=0.1, arms_out=0.18,
                        tail_pose=0.8, sway=0.5, breath_rate=1.15),
    "angry_stand": dict(lean=7.0, crouch=0.28, head_drop=1.5, head_fwd=4.5, head_pitch=8, shoulders=0.7, fists=1.0,
                        stance=1.0, tail_pose=0.35, weight=0.8, breath_rate=1.45, breath_depth=1.9),
    "sad_stand": dict(lean=4.0, crouch=0.2, head_drop=8.0, head_fwd=1.0, head_pitch=16, shoulders=-0.9, hunch=0.3,
                      tail_pose=-0.85, stance=-0.3, sway=0.25, breath_rate=0.7, breath_depth=1.35, blink_speed=0.75),
    "scared_stand": dict(lean=-9.0, crouch=0.55, head_drop=3.0, head_fwd=-4.0, head_pitch=6, shoulders=0.8, hunch=0.75,
                         hug=0.85, tail_pose=-1.0, stance=-0.4, weight=-1.0, breath_rate=2.2, breath_depth=0.55,
                         blink_speed=1.6),
    "sleepy_stand": dict(lean=2.5, crouch=0.22, head_drop=6.5, head_pitch=13, shoulders=-0.7, tail_pose=-0.3,
                         sway=1.2, breath_rate=0.6, breath_depth=1.5, blink_speed=0.5),
    "confident_stand": dict(lean=-4.5, head_drop=-3.5, head_pitch=-7, head_roll=-3, shoulders=0.0, akimbo=1.0,
                            stance=0.7, tail_pose=0.95, weight=-0.2, breath_depth=1.2),
    "confused_stand": dict(head_roll=18, head_fwd=1.5, shoulders=0.5, arms_out=0.3, weight=-0.25, ear_asym=1.0,
                           tail_pose=0.1),
    "bored_stand": dict(lean=2.5, crouch=0.16, head_drop=4.5, head_pitch=7, shoulders=-0.55, tail_pose=-0.35,
                        sway=0.9, breath_rate=0.75, blink_speed=0.8),
    "excited_stand": dict(lean=4.0, crouch=-0.06, head_drop=-3.5, head_fwd=1.5, shoulders=0.4, arms_out=0.6,
                          tail_pose=1.0, stance=0.35, weight=0.4, breath_rate=1.7, breath_depth=0.9),
    "curious_stand": dict(lean=5.5, head_drop=-1.0, head_fwd=4.5, head_roll=13, tail_pose=0.55, weight=0.7,
                          breath_rate=1.1),
    "surprised_stand": dict(lean=-6.0, crouch=-0.15, head_drop=-3.5, head_fwd=-3.0, shoulders=0.9, arms_out=0.5,
                            tail_pose=0.85, weight=-0.6, breath_rate=0.4, blink_speed=2.0),
    "love_stand": dict(lean=3.0, head_pitch=4, head_roll=11, shoulders=0.25, hug=0.35, tail_pose=0.7, sway=0.8,
                       weight=0.4, breath_rate=0.95),
    "proud_stand": dict(lean=-5.0, head_drop=-4.0, head_pitch=-8, shoulders=-0.1, akimbo=0.6, stance=0.5,
                        tail_pose=1.0, breath_depth=1.3),
    "determined_stand": dict(lean=6.0, crouch=0.2, head_fwd=3.0, head_pitch=5, fists=0.7, stance=0.6, weight=0.6,
                             tail_pose=0.4, breath_depth=1.4),
    "nervous_stand": dict(lean=-3.0, crouch=0.2, head_drop=1.5, shoulders=0.6, hunch=0.45, hug=0.45, tail_pose=-0.5,
                          weight=-0.3, breath_rate=1.6, breath_depth=0.6, blink_speed=1.7, sway=0.3),
    "embarrassed_stand": dict(lean=2.0, crouch=0.15, head_drop=4.0, head_pitch=10, head_roll=-8, shoulders=0.5,
                              hunch=0.5, hug=0.3, tail_pose=-0.5),
    "disgusted_stand": dict(lean=-5.0, head_fwd=-3.5, head_pitch=4, head_roll=-9, shoulders=0.4, weight=-0.6,
                            tail_pose=-0.2),
    "grumpy_stand": dict(lean=-2.0, head_pitch=6, head_roll=-4, shoulders=0.2, tail_pose=0.0, weight=-0.2,
                         breath_depth=1.3),
    "suspicious_stand": dict(lean=-3.0, crouch=0.15, head_pitch=7, head_fwd=1.0, shoulders=0.3, stance=0.3,
                             tail_pose=0.1, weight=-0.3),
    "impatient_stand": dict(lean=1.0, head_pitch=3, akimbo=0.55, shoulders=0.3, stance=0.4, tail_pose=0.3,
                            breath_rate=1.3),
    "lonely_stand": dict(lean=3.0, crouch=0.15, head_drop=6.0, head_pitch=12, head_roll=6, shoulders=-0.7, hunch=0.2,
                         hug=0.25, tail_pose=-0.7, breath_rate=0.75),
    "playful_stand": dict(lean=7.0, crouch=0.3, head_drop=2.0, head_fwd=2.5, head_roll=8, tail_pose=1.0, stance=0.6,
                          weight=0.6, breath_rate=1.4),
    "laughing_stand": dict(lean=-4.0, head_drop=-1.0, head_pitch=-6, shoulders=0.5, tail_pose=0.7,
                           breath_rate=1.8),
    "relieved_stand": dict(lean=1.0, crouch=0.08, head_drop=1.0, head_pitch=-2, shoulders=-0.8, tail_pose=0.3,
                           breath_rate=0.8, breath_depth=1.6),
    "dizzy_stand": dict(crouch=0.15, head_roll=10, sway=1.8, stance=0.4, tail_pose=-0.2),
    "focused_stand": dict(lean=3.0, head_fwd=2.5, head_pitch=4, shoulders=0.15, tail_pose=0.2, breath_rate=0.9,
                          blink_speed=0.9),
}

EMOTION_POSE = {
    Emotion.NEUTRAL: "neutral_stand", Emotion.HAPPY: "happy_stand", Emotion.CURIOUS: "curious_stand",
    Emotion.ALERT: "alert_stand", Emotion.SUSPICIOUS: "suspicious_stand", Emotion.EXCITED: "excited_stand",
    Emotion.SLEEPY: "sleepy_stand", Emotion.WORRIED: "nervous_stand", Emotion.ANNOYED: "grumpy_stand",
    Emotion.CELEBRATING: "excited_stand", Emotion.SAD: "sad_stand", Emotion.ANGRY: "angry_stand",
    Emotion.GRUMPY: "grumpy_stand", Emotion.SCARED: "scared_stand", Emotion.EMBARRASSED: "embarrassed_stand",
    Emotion.LOVE: "love_stand", Emotion.BORED: "bored_stand", Emotion.SMUG: "confident_stand",
    Emotion.DISGUSTED: "disgusted_stand", Emotion.SHOCKED: "surprised_stand", Emotion.LAUGHING: "laughing_stand",
    Emotion.DETERMINED: "determined_stand", Emotion.DIZZY: "dizzy_stand", Emotion.RELIEVED: "relieved_stand",
    Emotion.PROUD: "proud_stand", Emotion.SURPRISED: "surprised_stand", Emotion.CONFUSED: "confused_stand",
    Emotion.CALMING: "relaxed_stand", Emotion.CONFIDENT: "confident_stand", Emotion.LONELY: "lonely_stand",
    Emotion.PLAYFUL: "playful_stand", Emotion.JEALOUS: "grumpy_stand", Emotion.IMPATIENT: "impatient_stand",
    Emotion.RELAXED: "relaxed_stand", Emotion.FOCUSED: "focused_stand", Emotion.NERVOUS: "nervous_stand",
    Emotion.GRATEFUL: "love_stand",
}
for _e, _pose in EMOTION_POSE.items():
    STYLES[_e] = replace(STYLES[_e], **POSES[_pose])
for _e in list(STYLES):                  # every style knows its emotion (sheet faces pick by it)
    STYLES[_e] = replace(STYLES[_e], name=_e.value)


def pose_style(name: str) -> EmotionStyle:
    """A pose-library entry as a style (neutral face) - for the Playground / tests."""
    return replace(STYLES[Emotion.NEUTRAL], **POSES[name])


# emotional families (used to route big jumps through bridging emotions)
ANGER = {Emotion.ANGRY, Emotion.GRUMPY, Emotion.ANNOYED, Emotion.SUSPICIOUS, Emotion.IMPATIENT, Emotion.JEALOUS}
FEAR = {Emotion.SCARED, Emotion.NERVOUS, Emotion.WORRIED, Emotion.SHOCKED}
SORROW = {Emotion.SAD, Emotion.LONELY, Emotion.BORED}
JOY = {Emotion.HAPPY, Emotion.EXCITED, Emotion.CELEBRATING, Emotion.LAUGHING, Emotion.LOVE, Emotion.PLAYFUL,
       Emotion.PROUD, Emotion.GRATEFUL, Emotion.CONFIDENT}

_NUM = [f.name for f in fields(EmotionStyle) if f.type in ("float", float)]


def blend_styles(a: EmotionStyle, b: EmotionStyle, t: float) -> EmotionStyle:
    t = max(0.0, min(1.0, t))
    vals = {n: getattr(a, n) + (getattr(b, n) - getattr(a, n)) * t for n in _NUM}
    dominant = b if t >= 0.5 else a
    return replace(dominant, **vals)


class EmotionController:
    FADE = 0.35   # seconds to cross-fade emotions (longer for big valence jumps)

    def __init__(self, baseline: str | Emotion = Emotion.NEUTRAL):
        self.baseline = Emotion(baseline) if not isinstance(baseline, Emotion) else baseline
        self.current = self.baseline
        self.intensity = 1.0
        self._previous = self.baseline
        self._prev_intensity = 1.0
        self._mix = 1.0
        self._fade = self.FADE
        self._until: Optional[float] = None
        self._now = 0.0
        self._queue: list[tuple[Emotion, float, float, Optional[float]]] = []   # (emotion, intensity, step s, hold)
        self.manual_mix: Optional[dict] = None      # Playground emotion mixer override
        self.override_style: Optional[EmotionStyle] = None   # Playground pose-library preview
        self.routing = True

    # ---------------------------------------------------------------- control
    def bridge(self, src: Emotion, dst: Emotion) -> list[tuple[Emotion, float]]:
        """Bridging emotions for a big change of heart (each with its duration)."""
        if not self.routing or src == dst:
            return []
        vs, vd = STYLES[src].valence, STYLES[dst].valence
        if vd - vs < 0.9:
            return []
        if src in ANGER:
            if dst in (Emotion.CELEBRATING, Emotion.EXCITED, Emotion.LAUGHING):
                return [(Emotion.RELIEVED, 0.5), (Emotion.HAPPY, 0.4)]
            return [(Emotion.CALMING, 0.55)]
        if src in FEAR:
            return [(Emotion.RELIEVED, 0.5)]
        if src in SORROW:
            return [(Emotion.NEUTRAL, 0.35)]
        return []

    def set(self, emotion: Emotion | str, intensity: float = 1.0, hold: float | None = 3.0,
            now: float | None = None) -> None:
        emotion = Emotion(emotion)
        now = self._now if now is None else now
        steps = self.bridge(self.current, emotion) if self.intensity > 0.35 else []
        self._queue = []
        if steps:
            first, dur = steps[0]
            self._switch(first, 0.9, now, slow=True)
            self._until = now + dur
            for e, d in steps[1:]:
                self._queue.append((e, 0.9, d, None))
            self._queue.append((emotion, max(0.0, min(1.0, intensity)), 0.0, hold))
            return
        self._switch(emotion, intensity, now)
        self._until = None if hold is None else now + hold

    def _switch(self, emotion: Emotion, intensity: float, now: float, slow: bool = False) -> None:
        if emotion != self.current:
            self._previous = self.current
            self._prev_intensity = self.intensity
            self._mix = 0.0
            self.current = emotion
            dv = abs(STYLES[emotion].valence - STYLES[self._previous].valence)
            self._fade = self.FADE * (1.0 + 0.6 * dv) * (1.3 if slow else 1.0)
        self.intensity = max(0.0, min(1.0, intensity))

    def set_mix(self, weights: Optional[dict]) -> None:
        """Blend several emotions at once ({Emotion: weight 0..1}); None returns to normal control."""
        if not weights:
            self.manual_mix = None
            return
        self.manual_mix = {Emotion(k): max(0.0, min(1.0, float(v))) for k, v in weights.items()}

    def reset(self) -> None:
        self._queue = []
        self.set(self.baseline, 1.0, None)

    def update(self, dt: float, now: float) -> None:
        self._now = now
        if self._until is not None and now >= self._until:
            if self._queue:
                e, inten, step, hold = self._queue.pop(0)
                self._switch(e, inten, now)
                if step > 0:
                    self._until = now + step
                else:
                    self._until = None if hold is None else now + hold
            elif self.current != self.baseline:
                self._switch(self.baseline, 1.0, now)
                self.intensity = 1.0
                self._until = None
            else:
                self._until = None
        self._mix = min(1.0, self._mix + dt / max(0.05, self._fade))

    # ---------------------------------------------------------------- output
    @staticmethod
    def styled(e: Emotion, intensity: float) -> EmotionStyle:
        """An emotion at an intensity: 0.2 = a hint, 1.0 = full cartoon exaggeration."""
        base = STYLES[Emotion.NEUTRAL]
        st = blend_styles(base, STYLES[e], intensity)
        if intensity < 0.5 and st.mark:          # symbols / shivers only for strong feelings
            st = replace(st, mark=None, tremble=st.tremble * intensity)
        return st

    def style(self) -> EmotionStyle:
        if self.override_style is not None:
            return self.override_style
        if self.manual_mix:
            return mix_styles(self.manual_mix)
        cur = self.styled(self.current, self.intensity)
        prev = self.styled(self._previous, self._prev_intensity)
        eased = self._mix * self._mix * (3 - 2 * self._mix)
        return blend_styles(prev, cur, eased)

    @property
    def valence(self) -> float:
        return self.style().valence

    def wag(self, now: float) -> float:
        s = self.style()
        return s.tail_amp * math.sin(now * 2 * math.pi * s.tail_hz)


def mix_styles(weights: dict) -> EmotionStyle:
    """Weighted blend of several emotions (weights 0..1 each, relative to neutral)."""
    base = STYLES[Emotion.NEUTRAL]
    total = sum(weights.values())
    if total <= 1e-6:
        return base
    vals = {}
    norm = max(1.0, total)
    for n in _NUM:
        b = getattr(base, n)
        vals[n] = b + sum((getattr(STYLES[e], n) - b) * w for e, w in weights.items()) / norm
    dom = max(weights.items(), key=lambda kv: kv[1])
    st = STYLES[dom[0]] if dom[1] >= 0.5 else base
    out = replace(st, **vals)
    if dom[1] < 0.5:
        out = replace(out, mark=None)
    return out
