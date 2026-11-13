"""The living-character validation / showcase sequence (shared by ``--showcase``, the
Playground's "Validation run" button and the tests).

    SLEEP -> WAKE -> YAWN -> STRETCH -> STAND -> LOOK AROUND -> WALK -> JOG -> RUN -> SPRINT ->
    TURN -> JUMP -> LAND -> SHAKE -> SIT -> LOOK AT CURSOR -> CURSOR MOVES -> EYES TRACK ->
    HEAD FOLLOWS -> CURIOUS -> PLAY -> REST

The steps are ordinary Behavior plan steps, so they run on the desktop and on the
Playground stage alike.  ``mark(label)`` is called as each beat starts.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from .actions import Priority
from .attention import AttentionTarget as A
from .states import CompanionState as S

LIVING_SEQUENCE = ["SLEEP", "WAKE", "YAWN", "STRETCH", "STAND", "LOOK AROUND", "WALK", "JOG", "RUN", "SPRINT",
                   "TURN", "JUMP", "LAND", "SHAKE", "SIT", "LOOK AT CURSOR", "CURSOR MOVES", "EYES TRACK",
                   "HEAD FOLLOWS", "CURIOUS", "PLAY", "REST"]

DISTRACTION_SEQUENCE = ["DISTRACTION", "NOTICE", "HEAD TURN", "EYES TRACK", "ALERT", "STAND", "WALK", "JOG", "RUN",
                        "SPRINT", "ARRIVE", "STOP", "LOOK AT WINDOW", "LOOK AT USER", "SHOUT / WARNING", "COUNTDOWN",
                        "USER CLOSES", "RELIEF", "HAPPY", "JUMP", "SPIN", "CELEBRATE", "ROAMING"]


def living_steps(species: str, x0: float, span: float, ground: float, head_h: float,
                 mark: Optional[Callable[[str], None]] = None, cursor: Optional[Callable[[float, float], None]] = None,
                 sleep_s: float = 2.4) -> list[tuple]:
    """Plan steps for the living sequence.

    ``x0`` / ``span``  horizontal room to use (px), ``ground`` the floor y, ``head_h`` roughly the
    character height (to place the virtual cursor), ``cursor(x, y)`` optionally moves a drawn cursor.
    """
    P = Priority.SPECIAL_ACTION

    def m(label: str) -> tuple:
        return ("call", (lambda: mark(label)) if mark else (lambda: None))

    def x(f: float) -> float:
        return x0 + span * f

    def cur(px: float, py: float) -> tuple:
        return ("call", (lambda: cursor(px, py)) if cursor else (lambda: None))

    rest = {"cat": "LOAF", "rabbit": "LOAF", "dog": "DOWN", "fox": "LIE_DOWN", "robot": "CHARGE"}.get(species, "SIT")
    cy = ground - head_h * 1.25
    steps: list[tuple] = [
        m("SLEEP"), ("state", S.SLEEPING, sleep_s),
        m("WAKE"), ("state", S.WAKING, 1.3),
        m("YAWN"), ("action", "YAWN", None, P),
        m("STRETCH"), ("action", "STRETCH", None, P),
        m("STAND"), ("action", "STAND", 0.9, P),
        m("LOOK AROUND"), ("state", S.LOOKING_AROUND, 2.0),
        m("WALK"), ("goto_gait", x(0.24), ground, "walk"),
        m("JOG"), ("goto_gait", x(0.44), ground, "jog"),
        m("RUN"), ("goto_gait", x(0.66), ground, "run"),
        m("SPRINT"), ("goto_gait", x(0.92), ground, "sprint"),
        m("TURN"), ("turn_around",),
        m("JUMP"), ("jump", None, 0.0, 0.0, 1),
        m("LAND"), ("state", S.LANDING, 0.35),
        m("SHAKE"), ("action", "SHAKE", None, P),
        m("SIT"), ("state", S.SITTING, 0.9),
        m("LOOK AT CURSOR"), cur(x(0.7), cy), ("look", A.MOUSE, (x(0.7), cy), 0.9, None),
        m("CURSOR MOVES"),
    ]
    for i in range(8):                      # the cursor drifts across: eyes track, then the head follows
        px = x(0.7 - 0.07 * i)
        py = cy + math.sin(i * 0.9) * head_h * 0.25
        if i == 2:
            steps.append(m("EYES TRACK"))
        if i == 5:
            steps.append(m("HEAD FOLLOWS"))
        steps += [cur(px, py), ("look", A.MOUSE, (px, py), 0.28, None)]
    steps += [
        m("CURIOUS"), ("action", "CURIOUS", None, P), ("state", S.IDLE, 0.4),
        m("PLAY"), ("action", "PLAY", 2.2, P),
        m("REST"), ("action", rest, 3.0, P), ("state", S.SITTING, 1.0),
    ]
    return steps


# ----------------------------------------------------------------- character showcase
CHARACTER_SHOWCASE = ["WALK", "JOG", "RUN", "SPRINT", "TURN", "JUMP", "LAND", "SIT", "SLEEP", "WAKE", "STRETCH",
                      "WAVE", "PLAY", "LAUGH", "ANGRY", "SHOUT", "SCARED", "SAD", "HAPPY", "LOVE", "BORED",
                      "CELEBRATE"]


def character_showcase_steps(b, mark: Optional[Callable[[str], None]] = None) -> list[tuple]:
    """``python main.py --character-showcase``: one character performing the whole range - locomotion,
    physical actions and the full-body emotional performances (works with every effect switched off)."""
    from .emotion import Emotion
    P = Priority.SPECIAL_ACTION

    def m(label: str) -> tuple:
        return ("call", (lambda: mark(label)) if mark else (lambda: None))
    sc = b.mover.scale
    neutral = ("emotion", Emotion.NEUTRAL, 1.0, None)
    beats = {
        "WALK": [("goto_rel", 170 * sc, "walk")],
        "JOG": [("goto_rel", 240 * sc, "jog")],
        "RUN": [("goto_rel", 340 * sc, "run")],
        "SPRINT": [("goto_rel", 460 * sc, "sprint")],
        "TURN": [("turn_around",), ("state", S.IDLE, 0.4)],
        "JUMP": [("jump", None, 0.0, 0.0, 1)],
        "LAND": [("state", S.LANDING, 0.4), ("state", S.IDLE, 0.4)],
        "SIT": [("state", S.SITTING, 1.8)],
        "SLEEP": [("state", S.YAWNING, 1.7), ("state", S.SLEEPING, 2.6)],
        "WAKE": [("action", "EAR_TWITCH", None, P), ("state", S.SLEEPING, 0.5), ("state", S.WAKING, 1.2)],
        "STRETCH": [("action", "STRETCH", None, P), ("state", S.IDLE, 0.4)],
        "WAVE": [("look", A.USER, None, 0.3, S.IDLE), ("action", "WAVE", None, P)],
        "PLAY": [("action", "PLAY", 2.4, P)],
        "LAUGH": b.laugh_steps(1.0),
        "ANGRY": b.angry_steps(1.0),
        "SHOUT": b.shout_performance(2),
        "SCARED": b.scared_steps(),
        "SAD": [("emotion", Emotion.SAD, 1.0, 7.0), ("goto_rel", 70 * sc, "slow"), ("state", S.IDLE, 2.4)],
        "HAPPY": [("emotion", Emotion.HAPPY, 1.0, 5.0), ("action", "BOUNCE", None, P), ("goto_rel", 120 * sc, "walk"),
                  ("state", S.IDLE, 1.4)],
        "LOVE": [("look", A.USER, None, 0.3, S.IDLE)] + b.love_steps() + [("action", "LOVE", 1.8, P)],
        "BORED": [("action", "BORED", None, P), ("state", S.LOOKING_AROUND, 1.8), ("action", "SIGH", None, P),
                  ("state", S.IDLE, 1.0)],
        "CELEBRATE": [("emotion", Emotion.CELEBRATING, 1.0, 4.0), ("jump", None, 0.0, 0.0, 1),
                      ("action", "SPIN", None, P), ("state", S.CELEBRATING, 1.6), ("action", "CHEER", None, P),
                      ("state", S.IDLE, 0.6)],
    }
    steps: list[tuple] = []
    for label in CHARACTER_SHOWCASE:
        steps.append(m(label))
        steps += beats[label]
        if label in ("LAUGH", "ANGRY", "SHOUT", "SCARED", "SAD", "HAPPY", "LOVE", "BORED"):
            steps.append(neutral)
    return steps


# ----------------------------------------------------------- final acceptance sequence (cat)
ACCEPTANCE_SEQUENCE = ["SLEEPING", "EAR TWITCH", "WAKE", "YAWN", "STRETCH", "STAND", "WALK", "LOOK AT CURSOR",
                       "CURSOR MOVES", "EYES FOLLOW", "HEAD FOLLOWS", "CURIOUS", "POUNCE", "LAND", "SHAKE", "GROOM",
                       "SIT", "SLEEP"]


def acceptance_steps(species: str, x0: float, span: float, ground: float, head_h: float,
                     mark: Optional[Callable[[str], None]] = None,
                     cursor: Optional[Callable[[float, float], None]] = None) -> list[tuple]:
    """Sleeping -> ear twitch -> wake -> yawn -> stretch -> stand -> walk -> look at the cursor -> it
    moves: eyes follow, head follows -> curious -> pounce -> land -> shake -> groom -> sit -> sleep."""
    P = Priority.SPECIAL_ACTION

    def m(label: str) -> tuple:
        return ("call", (lambda: mark(label)) if mark else (lambda: None))

    def x(f: float) -> float:
        return x0 + span * f

    def cur(px: float, py: float) -> tuple:
        return ("call", (lambda: cursor(px, py)) if cursor else (lambda: None))
    cy = ground - head_h * 1.2
    steps: list[tuple] = [
        m("SLEEPING"), ("state", S.SLEEPING, 2.2),
        m("EAR TWITCH"), ("action", "EAR_TWITCH", None, P), ("state", S.SLEEPING, 0.7),
        m("WAKE"), ("state", S.WAKING, 1.2),
        m("YAWN"), ("action", "YAWN", None, P),
        m("STRETCH"), ("action", "STRETCH", None, P),
        m("STAND"), ("action", "STAND", 0.8, P),
        m("WALK"), ("goto_gait", x(0.35), ground, "walk"), ("state", S.IDLE, 0.3),
        m("LOOK AT CURSOR"), cur(x(0.62), cy), ("look", A.MOUSE, (x(0.62), cy), 0.9, None),
        m("CURSOR MOVES"),
    ]
    for i in range(8):
        px = x(0.62 - 0.035 * i)
        py = cy + math.sin(i * 0.8) * head_h * 0.3
        if i == 2:
            steps.append(m("EYES FOLLOW"))
        if i == 5:
            steps.append(m("HEAD FOLLOWS"))
        steps += [cur(px, py), ("look", A.MOUSE, (px, py), 0.26, None)]
    steps += [
        m("CURIOUS"), ("action", "CURIOUS", None, P), ("face", x(0.7)),
        m("POUNCE"), ("action", "POUNCE", None, P),
        m("LAND"), ("state", S.LANDING, 0.35),
        m("SHAKE"), ("action", "SHAKE", None, P),
        m("GROOM"), ("action", "GROOM", 2.4, P),
        m("SIT"), ("state", S.SITTING, 1.4),
        m("SLEEP"), ("state", S.YAWNING, 1.5), ("state", S.SLEEPING, 1.5),
    ]
    return steps


# ------------------------------------------------------------------ action / gait showcases (media)
ACTION_SHOWCASE = ["WALK", "RUN", "JUMP", "LAND", "SIT", "SLEEP", "WAKE", "STRETCH", "WAVE", "DANCE", "ANGRY",
                   "SHOUT", "SCARED", "CELEBRATE"]
GAIT_SHOWCASE = ["WALK", "JOG", "RUN", "SPRINT", "STOP"]


def action_showcase_steps(b, mark: Optional[Callable[[str], None]] = None) -> list[tuple]:
    """walk -> run -> jump -> land -> sit -> sleep -> wake -> stretch -> wave -> dance -> angry ->
    shout -> scared -> celebrate (media / tests; plain plan steps)."""
    from .emotion import Emotion
    P = Priority.SPECIAL_ACTION

    def m(label: str) -> tuple:
        return ("call", (lambda: mark(label)) if mark else (lambda: None))
    sc = b.mover.scale
    neutral = ("emotion", Emotion.NEUTRAL, 1.0, None)
    beats = {
        "WALK": [("goto_rel", 200 * sc, "walk")],
        "RUN": [("goto_rel", 360 * sc, "run"), ("state", S.IDLE, 0.3)],
        "JUMP": [("jump", None, 0.0, 0.0, 1)],
        "LAND": [("state", S.LANDING, 0.4), ("state", S.IDLE, 0.3)],
        "SIT": [("state", S.SITTING, 1.6)],
        "SLEEP": [("state", S.YAWNING, 1.4), ("state", S.SLEEPING, 2.2)],
        "WAKE": [("state", S.WAKING, 1.2)],
        "STRETCH": [("action", "STRETCH", None, P), ("state", S.IDLE, 0.3)],
        "WAVE": [("look", A.USER, None, 0.3, S.IDLE), ("action", "WAVE", None, P)],
        "DANCE": [("emotion", Emotion.HAPPY, 1.0, 4.0), ("action", "DANCE", 2.4, P), ("state", S.IDLE, 0.3)],
        "ANGRY": b.angry_steps(1.0),
        "SHOUT": b.shout_performance(2),
        "SCARED": b.scared_steps(),
        "CELEBRATE": [("emotion", Emotion.CELEBRATING, 1.0, 4.0), ("jump", None, 0.0, 0.0, 1),
                      ("action", "SPIN", None, P), ("state", S.CELEBRATING, 1.6), ("state", S.IDLE, 0.4)],
    }
    steps: list[tuple] = []
    for label in ACTION_SHOWCASE:
        steps.append(m(label))
        steps += beats[label]
        if label in ("ANGRY", "SHOUT", "SCARED", "DANCE"):
            steps.append(neutral)
    return steps


def gait_showcase_steps(b, mark: Optional[Callable[[str], None]] = None) -> list[tuple]:
    def m(label: str) -> tuple:
        return ("call", (lambda: mark(label)) if mark else (lambda: None))
    sc = b.mover.scale
    return [m("WALK"), ("goto_rel", 260 * sc, "walk"), m("JOG"), ("goto_rel", 360 * sc, "jog"),
            m("RUN"), ("goto_rel", 520 * sc, "run"), m("SPRINT"), ("goto_rel", 700 * sc, "sprint"),
            m("STOP"), ("state", S.IDLE, 1.0)]
