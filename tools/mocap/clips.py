"""The motion-capture clips FocusBuddy uses, cut from CMU trials (``SS_TT``) - see build_library.py.

Fields: ``src`` trial, ``t`` (start, end) seconds in the trial (after its T-pose frame), ``loop``,
``speed`` playback factor, ``face`` heading reference ("mean" | "start" | "end"), ``yaw`` extra turn
(deg), ``root`` horizontal root motion ("inplace" | "start" | "end" | "feet"), ``center`` centre a
lying body on its length (0..1), ``lock_y`` cap the hips at this many leg lengths (jumps: the flight
comes from the physics), ``mirror`` swap left / right, ``reverse`` play backwards, ``search`` loop
end search window (s).
"""

CLIPS = {
    # ---- standing idles (natural weight shifts, breathing, small looks)
    "idle": {"src": "82_08", "t": (0.3, 6.2), "loop": True, "search": 0.6},
    "idle_01": {"src": "113_21", "t": (0.2, 5.6), "loop": True, "search": 0.6},
    "idle_02": {"src": "139_02", "t": (0.2, 7.6), "loop": True, "search": 0.6},
    # a calm "normal wait": small head turns, hands loose (2.0)
    "idle_03": {"src": "137_28", "t": (0.5, 8.0), "loop": True, "search": 0.8, "straighten": True},
    # ---- locomotion
    "walk": {"src": "16_15", "t": (0.9, 2.0), "loop": True, "search": 0.2},
    # a graceful, narrow-stepping walk (long skirts / a saree)
    "walk_lady": {"src": "137_24", "t": (2.3, 3.5), "loop": True, "search": 0.25, "straighten": True},
    "jog": {"src": "35_17", "t": (0.3, 1.0), "loop": True, "search": 0.12},
    "run": {"src": "09_01", "t": (0.2, 0.9), "loop": True, "search": 0.12},
    # ---- gestures
    # one-armed gestures come in both hands ("_m" = mirrored): the companion waves with the arm
    # nearer the viewer, whichever way it faces
    "wave": {"src": "141_16", "t": (0.0, 2.4), "root": "feet", "mirror": True},
    "wave_m": {"src": "141_16", "t": (0.0, 2.4), "root": "feet"},
    "wave_big": {"src": "143_25", "t": (0.4, 5.0), "root": "feet"},
    "wave_big_m": {"src": "143_25", "t": (0.4, 5.0), "root": "feet", "mirror": True},
    "stretch": {"src": "143_30", "t": (0.0, 3.1), "root": "feet"},
    # 2.0: a stretch with a big yawn, and impatient waiting (weight shifts, hands, a sigh)
    "yawn": {"src": "141_13", "t": (0.0, 3.1), "root": "feet"},
    "fidget": {"src": "141_20", "t": (0.3, 3.2), "root": "feet"},
    "shrug": {"src": "141_21", "t": (0.0, 2.3), "root": "feet"},
    "bow": {"src": "113_02", "t": (0.0, 3.4), "root": "feet"},
    "celebrate": {"src": "79_69", "t": (0.3, 6.6), "root": "feet"},
    "laugh": {"src": "79_70", "t": (0.3, 6.2), "root": "feet"},
    "cry": {"src": "79_72", "t": (0.5, 7.6), "root": "feet"},
    "scared": {"src": "79_73", "t": (0.5, 6.0), "root": "feet"},
    "angry": {"src": "79_74", "t": (0.5, 7.0), "root": "feet"},
    "dance": {"src": "141_12", "t": (0.0, 3.4), "root": "feet"},
    # ---- jump (in place; the flight itself comes from the physics)
    "jump": {"src": "16_01", "t": (0.4, 2.0), "root": "feet", "lock_y": 1.0, "events": "jump"},
    # ---- sitting on the floor, legs out, leaning back on the hands (one frame of reference: ref_t)
    "sit_down": {"src": "114_16", "t": (2.3, 8.2), "speed": 2.0, "ref_t": 1.0, "root": "ref"},
    "sit": {"src": "114_16", "t": (9.6, 14.3), "loop": True, "search": 0.8, "ref_t": 1.0, "root": "ref"},
    "stand_up": {"src": "114_16", "t": (2.3, 8.2), "speed": 2.4, "ref_t": 1.0, "root": "ref", "reverse": True},
    # ---- lying down to sleep (on the back, legs stretched out) and getting up
    "lie_down": {"src": "113_08", "t": (0.9, 5.0), "speed": 1.3, "ref_t": 0.5, "root": "ref", "center": 1.0,
                 "straight_legs": 1.0},
    "sleep": {"src": "113_08", "t": (5.2, 9.8), "loop": True, "search": 0.8, "ref_t": 0.5, "root": "ref",
              "center": 1.0, "straight_legs": 1.0},
    "wake": {"src": "113_08", "t": (10.1, 14.3), "speed": 1.4, "ref_t": 0.5, "root": "ref", "center": 1.0,
             "straight_legs": 1.0},
}
