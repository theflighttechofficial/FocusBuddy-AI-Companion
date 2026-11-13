"""Per-character behaviour & locomotion parameters (CharacterProfile).

Every number that makes a cat feel different from a robot lives here, so
personalities are data, not duplicated code.  Speeds/strides are in
logical pixels at companion size 1.0 and scale with the companion size.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class CharacterProfile:
    id: str = "default"
    gait: str = "trot"              # trot | hop | mech | biped
    walk_speed: float = 80.0        # px/s
    run_speed: float = 280.0
    acceleration: float = 420.0     # px/s^2
    deceleration: float = 700.0
    turn_time: float = 0.32         # seconds for a 180 degree turn
    stride_walk: float = 34.0       # px travelled per full gait cycle
    stride_run: float = 82.0
    body_bob: float = 1.0           # multipliers for secondary motion
    run_lean: float = 1.0
    tail_swing: float = 1.0
    ear_response: float = 1.0
    head_lag: float = 1.0
    blink_interval: tuple[float, float] = (2.2, 5.5)
    double_blink: float = 0.15
    idle_frequency: float = 1.0     # >1 = picks new idle actions sooner
    curiosity: float = 0.5
    energy: float = 0.5
    sleepiness: float = 0.3
    playfulness: float = 0.5
    overshoot: float = 0.6          # settle wobble when stopping
    celebration: str = "jump"       # jump | spin | wag | dance | flip | cheer
    baseline: str = "NEUTRAL"       # resting emotion
    weights: dict = field(default_factory=dict)   # idle behaviour -> weight
    run_chance: float = 0.15        # chance a roam becomes a sprint
    # --- second generation locomotion ---------------------------------
    jog_speed: float = 170.0        # px/s (trot / jog level between walk and run)
    sprint_speed: float = 420.0     # short burst, mostly for interventions
    stride_jog: float = 56.0
    stride_sprint: float = 112.0
    sprint_time: float = 1.4        # longest sprint burst (s)
    jump_height: float = 46.0       # px at size 1.0
    breath_rate: float = 0.32       # Hz at rest
    mechanical: bool = False        # robot: linear / stepped interpolation, stiff springs
    # --- living-character upgrade: species jump styles -------------------
    jump_crouch: float = 0.13       # anticipation crouch before takeoff (s)
    jump_style: str = "normal"      # agile (cat/fox) | high (rabbit) | bouncy (dog) | mech (robot) | normal
    land_rebound: float = 0.0       # small extra bounce after landing (0..1, dog / rabbit)
    # --- personality traits (behaviour probabilities, not looks) ---------
    sociability: float = 0.5        # seeks the user / cursor (dog high, cat low)
    independence: float = 0.5       # wanders off on its own business (cat high)
    nervousness: float = 0.3        # startles, freezes, ducks (rabbit high)
    chaos: float = 0.2              # random bursts: zoomies, pounces at nothing (cat)
    precision: float = 0.2          # tidy, systematic patrols and movements (robot)
    gaze_hold: float = 1.0          # how long it looks at something before glancing away

    def scaled(self, **kw) -> "CharacterProfile":
        return replace(self, **kw)


BASE_WEIGHTS = {"walk": 30, "look": 15, "sit": 15, "nap": 10, "stretch": 8, "play": 10, "run": 7,
                "yawn": 2, "scratch": 1, "groom": 1, "watch_user": 4, "peek": 2, "explore": 4,
                "read": 0, "other_monitor": 1}


def _w(**over) -> dict:
    d = dict(BASE_WEIGHTS)
    d.update(over)
    return d


PROFILES: dict[str, CharacterProfile] = {
    # curious, lazy, playful, independent
    "cat": CharacterProfile(
        id="cat", sociability=0.35, independence=0.85, nervousness=0.35, chaos=0.55, precision=0.1, gaze_hold=0.75, jump_crouch=0.09, jump_style="agile", jog_speed=118, sprint_speed=450, stride_jog=44, stride_sprint=118, jump_height=54, breath_rate=0.3,
        gait="trot", walk_speed=56, run_speed=300, acceleration=520, deceleration=900,
        turn_time=0.26, stride_walk=30, stride_run=90, tail_swing=1.3, ear_response=1.2, head_lag=1.0,
        blink_interval=(2.8, 7.0), double_blink=0.1, idle_frequency=0.85, curiosity=0.8, energy=0.45,
        sleepiness=0.65, playfulness=0.7, overshoot=0.5, celebration="spin", baseline="HAPPY",
        run_chance=0.12,
        weights=_w(walk=22, look=12, sit=16, nap=13, stretch=8, groom=10, play=9, run=6, yawn=4,
                   scratch=3, watch_user=6, peek=5, explore=7)),
    # energetic, friendly, loyal
    "dog": CharacterProfile(
        id="dog", sociability=0.95, independence=0.2, nervousness=0.2, chaos=0.35, precision=0.1, gaze_hold=1.6, jump_crouch=0.13, jump_style="bouncy", land_rebound=0.6, jog_speed=130, sprint_speed=440, stride_jog=48, stride_sprint=120, jump_height=48, breath_rate=0.42,
        gait="trot", walk_speed=66, run_speed=310, acceleration=620, deceleration=760,
        turn_time=0.3, stride_walk=34, stride_run=92, body_bob=1.25, tail_swing=1.6, ear_response=1.4,
        head_lag=1.2, blink_interval=(2.0, 4.5), double_blink=0.2, idle_frequency=1.35, curiosity=0.6,
        energy=0.95, sleepiness=0.2, playfulness=0.95, overshoot=0.9, celebration="wag",
        baseline="HAPPY", run_chance=0.3,
        weights=_w(walk=28, run=14, play=16, watch_user=12, sit=8, scratch=7, look=8, nap=4, stretch=4,
                   yawn=1, groom=0, peek=3, explore=5)),
    # calm, clever, alert
    "fox": CharacterProfile(
        id="fox", sociability=0.45, independence=0.7, nervousness=0.4, chaos=0.15, precision=0.4, gaze_hold=1.2, jump_crouch=0.1, jump_style="agile", jog_speed=126, sprint_speed=500, stride_jog=48, stride_sprint=126, jump_height=50, breath_rate=0.3,
        gait="trot", walk_speed=60, run_speed=330, acceleration=380, deceleration=650,
        turn_time=0.34, stride_walk=34, stride_run=96, body_bob=0.7, tail_swing=1.1, ear_response=1.3,
        head_lag=0.8, blink_interval=(3.0, 6.5), double_blink=0.08, idle_frequency=0.75, curiosity=0.7,
        energy=0.55, sleepiness=0.35, playfulness=0.4, overshoot=0.35, celebration="flip",
        baseline="HAPPY", run_chance=0.1,
        weights=_w(walk=26, look=20, sit=14, nap=8, stretch=6, groom=6, run=6, play=5, watch_user=9,
                   peek=7, explore=8, yawn=2)),
    # nervous, cute, energetic
    "rabbit": CharacterProfile(
        id="rabbit", sociability=0.5, independence=0.4, nervousness=0.9, chaos=0.3, precision=0.1, gaze_hold=0.7, jump_crouch=0.1, jump_style="high", land_rebound=0.3, jog_speed=150, sprint_speed=420, stride_jog=62, stride_sprint=124, jump_height=58, breath_rate=0.5,
        gait="hop", walk_speed=70, run_speed=280, acceleration=900, deceleration=1100,
        turn_time=0.16, stride_walk=46, stride_run=96, body_bob=1.0, tail_swing=0.6, ear_response=1.6,
        head_lag=1.3, blink_interval=(1.6, 4.0), double_blink=0.25, idle_frequency=1.4, curiosity=0.6,
        energy=0.8, sleepiness=0.3, playfulness=0.6, overshoot=0.8, celebration="jump",
        baseline="HAPPY", run_chance=0.25,
        weights=_w(walk=26, look=16, sit=12, nap=8, groom=10, run=12, play=8, scratch=4, stretch=4,
                   watch_user=5, peek=4, explore=4)),
    # logical, precise, efficient
    "robot": CharacterProfile(
        id="robot", sociability=0.55, independence=0.4, nervousness=0.1, chaos=0.05, precision=0.95, gaze_hold=1.3, jump_crouch=0.22, jump_style="mech", jog_speed=150, sprint_speed=330, stride_jog=46, stride_sprint=84, jump_height=34, breath_rate=0.25, mechanical=True,
        gait="mech", walk_speed=78, run_speed=250, acceleration=700, deceleration=1200,
        turn_time=0.42, stride_walk=30, stride_run=70, body_bob=0.45, run_lean=0.5, tail_swing=0.0,
        ear_response=0.0, head_lag=0.4, blink_interval=(3.5, 6.0), double_blink=0.3, idle_frequency=1.0,
        curiosity=0.6, energy=0.6, sleepiness=0.15, playfulness=0.35, overshoot=0.15,
        celebration="dance", baseline="HAPPY", run_chance=0.1,
        weights=_w(walk=28, look=18, sit=8, nap=5, stretch=5, watch_user=14, run=6, play=6, peek=6,
                   explore=6, yawn=0, scratch=2, groom=0)),
    # supportive, friendly, focused
    "buddy": CharacterProfile(
        id="buddy", sociability=0.85, independence=0.3, nervousness=0.25, chaos=0.1, precision=0.5, gaze_hold=1.1, jump_crouch=0.14, jog_speed=130, sprint_speed=320, stride_jog=44, stride_sprint=80, jump_height=40, breath_rate=0.28,
        gait="biped", walk_speed=70, run_speed=230, acceleration=420, deceleration=760,
        turn_time=0.3, stride_walk=30, stride_run=66, body_bob=0.9, run_lean=0.9, tail_swing=0.0,
        ear_response=0.0, head_lag=0.9, blink_interval=(2.5, 5.5), double_blink=0.12, idle_frequency=0.9,
        curiosity=0.5, energy=0.5, sleepiness=0.3, playfulness=0.4, overshoot=0.5, celebration="cheer",
        baseline="HAPPY", run_chance=0.08,
        weights=_w(walk=22, look=10, sit=10, read=22, stretch=10, nap=6, watch_user=10, yawn=4, run=4,
                   play=4, scratch=0, groom=0, peek=3, explore=4)),
}

PERSONALITY_IDS = list(PROFILES)


GAIT_LEVELS = ("WALK", "JOG", "RUN", "SPRINT")


def get_profile(profile_id: str | None) -> CharacterProfile:
    return PROFILES.get(profile_id or "", PROFILES["cat"])
