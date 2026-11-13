"""Paths, constants and default settings."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "FocusBuddy"
IS_WINDOWS = sys.platform.startswith("win")
FROZEN = bool(getattr(sys, "frozen", False))

# Project root when running from source; bundle dir when frozen (PyInstaller).
ROOT_DIR = Path(__file__).resolve().parent.parent
RES_DIR = Path(getattr(sys, "_MEIPASS", ROOT_DIR))
ASSETS_DIR = RES_DIR / "assets"
EXTENSION_DIR = RES_DIR / "browser_extension"
DEFAULT_BRIDGE_PORT = 47621


def data_dir() -> Path:
    """Directory for the database, logs and caches.

    * ``FOCUSBUDDY_DATA_DIR`` env var wins (used by tests / portable installs).
    * Frozen .exe  -> %LOCALAPPDATA%\\FocusBuddy
    * From source  -> <project>/data
    """
    env = os.environ.get("FOCUSBUDDY_DATA_DIR")
    if env:
        path = Path(env)
    elif FROZEN:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        path = Path(base) / APP_NAME
    else:
        path = ROOT_DIR / "data"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        path = Path.home() / ".focusbuddy"
        path.mkdir(parents=True, exist_ok=True)
    return path


# Sensitivity presets: tau = seconds for dwell score to reach ~63% of its max.
# companion size presets (factor of the 124 px base box)
SIZE_PRESETS = {"small": 1.0, "medium": 1.35, "large": 1.75, "xl": 2.3,
                "quarter": None}            # "Giant": about a quarter of the screen's height (per screen)
SIZE_RANGE = (0.6, 5.0)
SIZE_LABELS = (("small", "Small"), ("medium", "Medium"), ("large", "Large"), ("xl", "Extra large"),
               ("quarter", "Giant (¼ screen)"))
QUARTER = 0.25                             # "Giant": the character's height as a share of the screen's

SENSITIVITY_PRESETS = {
    "relaxed": {"tau": 180.0, "min_seconds": 10, "spacing": 30},
    "balanced": {"tau": 100.0, "min_seconds": 5, "spacing": 20},
    "strict": {"tau": 50.0, "min_seconds": 3, "spacing": 10},
}

DEFAULT_SETTINGS: dict = {
    # ---- 2.0 intelligence
    "coach_nudges": True,          # offer a focus session at the start of your usual risky hour
    "context_reactions": True,     # the companion reacts to flow, restlessness, long stretches, late nights
    "ai_chat_llm": True,           # use a local Ollama model for free conversation when one is running
    "ai_model": "",                # "" = pick automatically
    "ollama_url": "http://127.0.0.1:11434",
    "daily_report": True,          # an end-of-day summary from the companion (once, after 6 pm)
    "last_report_day": "",
    "lifelike_level": "high",      # low | normal | high: micro-expressions, gaze, breathing, fidgets
    # ---- 2.1 routines, care and focus extras
    "care": {},                    # companion care state (needs, treats, bond XP, badges) - intel/care.py
    "morning_checkin": True,       # a hello and "what's the one thing today?" the first time you're here
    "last_checkin_day": "",
    "wind_down": True,             # a gentle "let's wrap up" at the wind-down time
    "wind_down_time": "21:30",
    "last_wind_down_day": "",
    "nap_when_away": True,         # the companion naps off-screen while you're away, back when you are
    "nap_away_minutes": 10,
    "nap_schedule": "",            # e.g. "13:30, 16:00": scheduled naps
    "nap_schedule_minutes": 20,
    "ambient_sound": "off",        # off | rain | cafe | brown | pink | forest | fireplace
    "ambient_volume": 0.4,
    "ambient_mode": "focus",       # focus (during sessions) | always
    "deep_focus": False,           # sessions: quiet companion, no chatter, distractions closed
    "deep_restore": {},            # settings Deep focus changed for the running session (restored after)
    # --- general ---
    "user_name": "",
    "first_run_done": False,
    "extension_prompted": False,   # the browser-extension setup was offered (wizard / once afterwards)
    "start_with_windows": False,
    "start_minimized": True,
    "show_companion": True,
    "companion_opacity": 1.0,
    "always_on_top": True,
    "click_through": False,
    "notifications_enabled": True,
    # --- companion ---
    "character": "cat",
    "companion_name": "",          # empty -> character default name
    "theme": "system",             # system | light | dark  (system follows the Windows app theme)
    "accent": "brand",             # brand (lavender) | peach | mint | sky | rose | system | "#RRGGBB"
    "motion": "full",              # full | reduced | off  (Windows "animation effects" off -> reduced)
    "window_geometry": "",         # dashboard size / position (restored on open)
    "companion_size": 1.35,        # 0.6 .. 3.0 (1.0 = 124 px box); Ctrl + mouse wheel on the companion
    "movement_speed": 1.0,         # 0.5 .. 2.0
    "idle_activity": 0.5,          # 0 (calm) .. 1 (playful)
    "walking_frequency": 0.5,      # 0 .. 1
    "speech_bubbles": True,
    "sound_enabled": True,
    "sound_volume": 0.5,
    "footstep_sounds": False,
    "sound_level": "medium",       # off | low | medium | high
    "muted": False,
    # --- companion animation & behaviour (animation upgrade) ---
    "size_preset": "medium",       # small | medium | large | xl | custom
    "size_version": 1,             # 2 = presets after the "bigger companion" update
    "animation_intensity": 1.0,    # 0.3 .. 1.5 secondary motion strength
    "animation_fps": 60,           # 30 | 60 | 120
    "personality": "",             # "" = the character's own personality
    "cursor_interaction": "look",  # ignore | look | follow | chase | run_away
    "click_reaction": "random",    # random | wave | jump | spin | look
    "roaming": "desktop",          # floor | desktop | all  (all = other monitors too)
    "edge_peeking": True,
    "petting": True,
    "stay_near_during_focus": True,
    "celebration_style": "",       # "" = character default | jump | spin | wag | dance | flip | cheer
    "debug_overlay": False,
    "debug_skeleton": False,          # draw bones / IK targets / foot contacts on the companion
    # second-generation animation (Low / Normal / High)
    "animation_smoothness": "normal",
    "movement_smoothness": "normal",
    "secondary_motion": "normal",
    "physics_strength": "normal",
    "idle_activity_level": "normal",
    "character_energy": "normal",
    "particles": True,
    "time_of_day_behavior": True,
    "surprise_events": True,
    # emotion pack
    "angry_reactions": True,          # playful grumpiness: stomping, steam, a cartoon "HEY!" (never harsh)
    "emotion_intensity": "normal",    # low / normal / high
    "renderer": "production",         # production (3D models from the character sheets) / illustrated
                                      # (character-sheet art) / 3d (procedural 3D) / 2d (flat vector art)
    "renderer_version": 1,            # 3 = the 3D character models were offered as the default
    "render_quality": "normal",       # low / medium (= normal) / high / ultra
    "auto_quality": True,             # step 3D quality down if frames keep missing their budget
    "shadows": True,                  # contact shadow under the character
    "cursor_reactions": "subtle",     # off | subtle | playful  (eyes / head follow, curiosity, pounce)
    "stumbles": True,                 # very rare trip / fall-over moments (never during focus)
    "window_interactions": True,      # lean on / peek around windows, react when a window is dragged at it
    # --- the companion's life (companion/life.py): routine, personality, the close scene ---
    "playfulness": 0.8,               # calm 0 .. playful .. chaotic 1 (antics, play)
    "roaming_frequency": 0.6,         # rare 0 .. regular .. frequent 1 (walks, patrols, leaving for a while)
    "chatter": 0.7,                   # unsolicited dialogue: quiet 0 .. some .. chatty 1 (cheerful by default)
    "focus_behavior": "balanced",     # strict (only essential warnings) | balanced (silent study partner) | playful
    "roam_area": "mostly_edges",      # edges | mostly_edges | desktop  (where it spends its time)
    "quiet_mode": False,              # "be quiet": no chatter, no unsolicited interactions
    "leave_and_return": True,         # wanders off-screen now and then and comes back with an entrance
    "close_button": "ask",            # dashboard X: ask (the companion asks: nap / stay / quit) | stay | nap
    "wake_scene": True,               # wake-up scene + entrance when FocusBuddy starts
    "hide_in_fullscreen": True,       # step out of the way while a full-screen app / presentation runs
    "companion_life": {},             # the companion's memory (local only): mood, energy, sessions, affinity ...
    # --- monitoring ---
    "monitoring_enabled": True,
    "sensitivity": "balanced",     # relaxed | balanced | strict
    "min_distraction_seconds": 5,
    "poll_interval_ms": 1000,
    "daily_distraction_allowance_min": 30,
    "idle_threshold_seconds": 300,
    "context_classifier": True,
    "use_title_fallback": True,
    "detect_games": True,
    "focus_speedup": 2.0,          # distractions are noticed this much faster during a focus session          # exes running from game / launcher folders count as distracting
    "rules_version": 1,            # default-rule set already merged into the user's rules (see app._migrate_rules)
    # --- thresholds ---
    "threshold_mild": 40,
    "threshold_warning": 70,
    "threshold_strong": 90,
    # --- blocking ---
    "blocking_mode": "ask",        # warn | ask | auto   (auto must be opted into) - outside focus sessions
    # during a focus session: same (= blocking_mode) | auto_close (countdown, then close any distracting
    # site / app) | instant (3 s countdown, no walk, and the extension blocks distracting sites on load)
    "focus_enforcement": "auto_close",
    "keyboard_tab_close": True,    # without the extension, close the tab with the browser's Ctrl+W
                                   # (only the verified foreground browser window / tab)
    "grace_period_seconds": 10,
    "escalation_spacing_seconds": 20,
    "website_action": "close_tab", # close_tab | redirect | minimize
    "app_action": "close",         # close (polite WM_CLOSE, minimized if ignored) | minimize
    "allow_terminal_close": False,
    "snooze_short_min": 2,
    "snooze_long_min": 5,
    # extra time (snooze / ignore / keep open): once_per_site | once_per_session | unlimited | never
    "override_policy": "once_per_site",
    # --- focus ---
    "focus_minutes": 25,
    "break_minutes": 5,
    "long_break_minutes": 15,
    "long_break_every": 4,
    "auto_start_next": False,
    "auto_start_break": True,
    "relax_blocking_on_break": True,
    "daily_focus_goal_min": 240,
    # --- browser bridge ---
    "bridge_port": DEFAULT_BRIDGE_PORT,
}

BROWSER_EXES = {
    "chrome.exe": "chrome",
    "msedge.exe": "edge",
    "firefox.exe": "firefox",
    "brave.exe": "brave",
    "opera.exe": "opera",
    "opera_gx.exe": "opera",
    "vivaldi.exe": "vivaldi",
    "arc.exe": "arc",
    "chromium.exe": "chromium",
    "librewolf.exe": "firefox",
    "waterfox.exe": "firefox",
    "zen.exe": "firefox",
    # non-Windows names (future platforms / demo mode)
    "google-chrome": "chrome",
    "chromium": "chromium",
    "firefox": "firefox",
    "microsoft-edge": "edge",
}

# Browsers whose extension reports look like plain Chrome.
CHROMIUM_FAMILY = {"chrome", "brave", "opera", "vivaldi", "arc", "chromium"}
