"""FocusBuddy application controller - wires every subsystem together.

    WindowManager --> ActivityMonitor --> RuleBasedDetector --> InterventionEngine
                         ^   (BrowserBridge <- extension)            |   ^
                         |                                           v   | user choices
                     StatsTracker <------------------------ Companion view (overlay + bubble)
                                                                     |
                                                       Blocker (SafetyGuard) -> close / minimize

Each subsystem is isolated: if one fails (no database, no browser
extension, missing asset, permission denied) the rest keeps working.
"""

from __future__ import annotations

import datetime
import os
import random
import time
from typing import Optional

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QInputDialog, QMenu

from . import __version__
from .blocking.application import ApplicationBlocker
from .blocking.blocker import Blocker
from .blocking.browser import BrowserBlocker
from .blocking.safety import SafetyGuard
from .browser.bridge import BrowserBridge
from .companion.character import Companion
from .companion.characters import selectable_characters, companion_name, get_character
from .companion.sprites import discover_sprite_packs
from .companion.states import CompanionState as S
from .config import ASSETS_DIR, SIZE_LABELS, SIZE_PRESETS, SIZE_RANGE, data_dir
from .core.classifier import CompositeClassifier, FeedbackLearner, KeywordContextClassifier
from .core.detector import DetectorContext, RuleBasedDetector, Thresholds
from .core.focus_manager import FocusTimer, Phase
from .core.intervention import Episode, InterventionEngine
from .core.messages import MessageManager
from .core.monitor import ActivityMonitor
from .core.rules import DEFAULT_RULES, RULES_VERSION, Category, RuleSet
from .core.stats import StatsTracker
from .core.whitelist import TemporaryWhitelist
from .platform import create_autostart, create_window_manager
from .platform.geometry import to_logical
from .storage.database import Database
from .storage.settings import Settings
from .ui.icons import app_icon
from .ui.tray import Tray
from .utils.logging_setup import get_logger, kv, setup_logging
from .utils.sound import SoundManager

log = get_logger("app")


class FocusBuddyApp(QObject):
    def __init__(self, qapp, *, demo: bool = False, data_path=None, window_manager=None, debug: bool = False,
                 force_wizard: bool = False, start_minimized: bool | None = None, profile=None, profiles=None,
                 start_locked: bool = False):
        super().__init__()
        self.qapp = qapp
        self.profile = profile                  # core.profiles.Profile (None in demo / tests)
        self.profiles = profiles                # core.profiles.ProfileStore
        self.locked = bool(start_locked and profile is not None and profile.has_pin)
        self.restart_args = None                # set by sign out / switch profile
        self.demo = demo
        self.data_dir = data_path or data_dir()
        setup_logging(self.data_dir / "logs", debug)
        log.info("FocusBuddy starting %s", kv(version=__version__, demo=demo or None))

        # ---- storage -------------------------------------------------------
        self.db = Database(self.data_dir / "focusbuddy.db")
        fresh = self.db.rule_count() == 0
        if fresh:
            self.db.replace_rules([{"kind": r.kind, "pattern": r.pattern, "category": r.category.value,
                                    "auto_close": r.auto_close} for r in DEFAULT_RULES])
        self.settings = Settings(self.db)
        if fresh:
            self.settings.set("rules_version", RULES_VERSION)
        self._migrate_rules()
        if int(self.settings.get("renderer_version", 1) or 1) < 3:
            # the 3D models reconstructed from the character sheets become the default look
            # (the illustrated sheet art stays available in Settings -> Appearance)
            if self.settings["renderer"] in ("illustrated", "production"):
                self.settings.set("renderer", "production")
            self.settings.set("renderer_version", 3)
        if not self.settings["auto_quality"]:
            # 2.4: "Adaptive 3D quality" became the Performance mode; switched off = keep my quality
            self.settings.set("performance_mode", "quality")
            self.settings.set("auto_quality", True)
        if int(self.settings.get("size_version", 1) or 1) < 2:
            # the presets grew (the new characters are long animals): keep custom sizes, move presets
            if SIZE_PRESETS.get(self.settings["size_preset"]) is not None:
                self.settings.set("companion_size", SIZE_PRESETS[self.settings["size_preset"]])
            self.settings.set("size_version", 2)
        if self.profile is not None and not self.settings["user_name"]:
            self.settings.set("user_name", self.profile.name)
        self._apply_motion()
        self._apply_theme()
        from .ui.theme import watch_system_theme
        watch_system_theme(lambda: self._apply_theme() if self.settings["theme"] == "system" else None)
        self.db.close_dangling_sessions(time.time())
        discover_sprite_packs(ASSETS_DIR / "characters", self.data_dir / "characters")

        # ---- platform ------------------------------------------------------
        self.wm = window_manager or create_window_manager()
        self.autostart = create_autostart()
        self.bridge: Optional[BrowserBridge] = BrowserBridge(int(self.settings["bridge_port"]))
        if not self.bridge.start():
            log.warning("Continuing without browser extension support")

        # ---- core ----------------------------------------------------------
        self.whitelist = TemporaryWhitelist()
        self.learner = FeedbackLearner()
        self.rules = RuleSet.from_rows(self.db.rules())
        self.detector = RuleBasedDetector(self.rules, self.whitelist,
                                          CompositeClassifier(KeywordContextClassifier(), self.learner),
                                          lambda: self.settings["context_classifier"])
        self.monitor = ActivityMonitor(self.wm, self.bridge, brands_provider=lambda: self.rules.known_site_names(),
                                       use_title_fallback=lambda: self.settings["use_title_fallback"],
                                       own_pid=os.getpid())
        self.safety = SafetyGuard(self.settings, os.getpid())
        self.app_blocker = ApplicationBlocker(self.wm)
        self.blocker = Blocker(self.app_blocker, BrowserBlocker(self.bridge), self.safety, self.settings)
        self._policy: dict = {"active": False}
        if self.bridge:
            self.bridge.policy_provider = self._browser_policy
        self.stats = StatsTracker(self.db, idle_threshold=self.settings["idle_threshold_seconds"])
        self.focus = FocusTimer()
        self.tau_override: Optional[float] = None
        self.monitoring = bool(self.settings["monitoring_enabled"])
        self.current_activity = None
        self.current_assessment = None
        self.last_productive_handle: Optional[int] = None
        self._last_poll = time.time()

        # ---- companion -----------------------------------------------------
        spec = get_character(self.settings["character"])
        self.messages = MessageManager(self.settings["personality"] or spec.personality)
        self._update_message_defaults()
        self.sound = SoundManager(self.data_dir / "cache", self.settings["sound_enabled"], self.settings["sound_volume"])
        self.sound.muted = self.settings["muted"]
        self.companion = Companion(spec, size_factor=self.settings["companion_size"],
                                   always_on_top=self.settings["always_on_top"],
                                   idle_provider=self._idle_seconds, personality=self.settings["personality"])
        self.companion.life.load(self.settings["companion_life"])
        self.companion.life.on_quit_ready = self.quit
        self._apply_companion_settings()
        if self.settings["click_through"]:
            self.companion.set_click_through(True)
        self.companion.double_clicked.connect(lambda: self.show_dashboard())
        self.companion.context_menu.connect(self._show_context_menu)
        self.companion.bubble_action.connect(self._on_bubble_action)
        self.companion.clicked.connect(self._on_companion_clicked)
        self.companion.grabbed.connect(lambda: self._say("DRAG", chance=0.5))
        self.companion.dropped.connect(lambda: self._say("FALL", chance=0.6))
        self.companion.petted.connect(lambda: self._say("PETTED", chance=0.8))
        self.companion.sound.connect(self.sound.play)
        self.companion.emote.connect(self._on_emote)
        self.companion.resize_requested.connect(self._resize_step)

        # ---- intelligence (all local) -------------------------------------
        from .intel.api import AppApi
        from .intel.assistant import Assistant, LocalLLM
        from .intel.coach import FocusCoach
        from .intel.context import ContextEngine
        self.coach = FocusCoach(self.db)
        self.context = ContextEngine()
        self.llm = LocalLLM(self.settings["ollama_url"], self.settings["ai_model"])
        self.api = AppApi(self)
        self.assistant = Assistant(self.api, self.llm if self.settings["ai_chat_llm"] else None)
        self.focus_task_id: Optional[int] = None
        self._focus_today_cache = (0.0, 0.0)          # (monotonic time, seconds)
        self._last_nudge = 0.0
        self.tasks_listeners: list = []
        from .intel.routines import Routines
        self.routines = Routines(self)
        self.care_listeners: list = []

        self.engine = InterventionEngine(self.settings, self.whitelist, self.blocker, self, recorder=self._record_episode,
                                         return_to_focus=self._return_to_focus, learner=self.learner)

        # ---- shell UI ------------------------------------------------------
        self.tray = Tray(lambda color: app_icon(get_character(self.settings["character"]), color))
        self.tray.open_dashboard.connect(lambda: self.show_dashboard())
        self.tray.open_settings.connect(lambda: self.show_dashboard("settings"))
        self.tray.toggle_monitoring.connect(lambda: self.set_monitoring(not self.monitoring))
        self.tray.start_focus.connect(self.start_focus)
        self.tray.end_focus.connect(self.end_focus)
        self.tray.toggle_mute.connect(lambda: self.settings.set("muted", not self.settings["muted"]))
        self.tray.toggle_companion.connect(lambda: self.show_companion(not self.settings["show_companion"]))
        self.tray.quit_requested.connect(self.request_quit)
        self.tray.toggle_nap.connect(self.toggle_nap)
        self.tray.toggle_quiet.connect(lambda: self.settings.set("quiet_mode", not self.settings["quiet_mode"]))
        self.tray.lock_requested.connect(self.lock)
        self.tray.switch_requested.connect(lambda: self.switch_profile())
        self.tray.act_lock.setVisible(bool(self.profile is not None and self.profile.has_pin))
        self.dashboard = None
        self.wizard = None
        self.quitting = False
        self._icon = app_icon(spec)
        qapp.setWindowIcon(self._icon)

        self.settings.on_change(self._on_setting)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll)
        self.chatter_timer = QTimer(self)
        self.chatter_timer.timeout.connect(self._chatter)
        self.world_timer = QTimer(self)
        self.world_timer.timeout.connect(self._refresh_world)
        self.world_timer.timeout.connect(self._idle_lock)
        self.world_timer.timeout.connect(self._watch_presence)
        self.intel_timer = QTimer(self)
        self.intel_timer.timeout.connect(self._intel_tick)
        self._last_glance_key = None
        self._force_wizard = force_wizard
        self._start_minimized = self.settings["start_minimized"] if start_minimized is None else start_minimized
        if not self.wm.available and not demo:
            log.warning("Window monitoring unavailable on this platform")

    # ================================================================== start
    def start(self) -> None:
        if self.settings["size_preset"] == "quarter":          # Giant: a quarter of *this* screen
            self.settings.set("companion_size", self.preset_size("quarter"))
        wizard = self._force_wizard or not self.settings["first_run_done"]
        # opened by the user: the app comes first (full screen), the companion strolls in a moment
        # later. Started quietly with Windows: straight to the companion.
        dashboard_first = not wizard and not self._start_minimized and not self.demo
        self._companion_started = False
        if dashboard_first:
            self.show_dashboard()
            QTimer.singleShot(1100, self._start_companion)
        else:
            self._start_companion()
        self._life_saved = time.perf_counter()
        self.tray.show()
        self._refresh_tray()
        self.poll_timer.start(max(250, int(self.settings["poll_interval_ms"])))
        self.chatter_timer.start(self._chatter_interval())
        self._active_s = 0.0                 # how long the user has been at the computer without a break
        self._last_reminder = {"STRETCH": 0.0, "WATER": 0.0}
        self._away = False
        if self.settings["wake_scene"]:
            QTimer.singleShot(9000, self._greet_time_of_day)   # after the wake-up scene (skipped while busy)
        self.world_timer.start(1500)
        self.intel_timer.start(60_000)
        if self.settings["debug_overlay"]:
            self.companion.set_debug(True)
        if self._force_wizard or not self.settings["first_run_done"]:
            QTimer.singleShot(400, self.show_wizard)
        else:
            if not self.settings["wake_scene"] or self.demo:
                QTimer.singleShot(1500, lambda: self._say("GREETING") if self.demo or random.random() < 0.4
                                  else self._greet_time_of_day())
            if not self.settings["extension_prompted"] and not self.demo and self.bridge is not None:
                # set up before the extension step existed: offer it once (if it isn't connected already)
                QTimer.singleShot(20_000, self._offer_extension)
        QTimer.singleShot(3000, self.sound.prepare)

    def _start_companion(self) -> None:
        if self._companion_started or self.quitting:
            return
        self._companion_started = True
        self.companion.set_visible(self.settings["show_companion"])
        self.companion.start()
        # the companion's day starts: curled up at the screen edge -> wakes -> enters the desktop
        if self.settings["wake_scene"] and self.settings["show_companion"] and not self.demo:
            self.companion.life.wake_up()
        else:
            self.companion.life.skip_intro()

    def request_quit(self) -> None:
        """Quit FocusBuddy completely - the companion says goodbye and walks off first (a few
        seconds at most), then the process exits."""
        if self.quitting or getattr(self, "_quit_pending", False):
            return
        life = self.companion.life
        if not (self.companion.visible_wanted and not self.companion.life_hidden and life.started
                and self.settings["wake_scene"]):
            self.quit()
            return
        self._quit_pending = True
        life.goodbye(self.quit)
        QTimer.singleShot(6500, self.quit)            # never hang on the farewell

    def show_companion(self, on: bool) -> None:
        """Show / hide the desktop companion (the user's choice, remembered). Hiding says how to get it
        back: the dashboard's Home page, Settings, the tray icon or Ctrl+H in the dashboard."""
        self.settings.set("show_companion", bool(on))
        spec = get_character(self.settings["character"])
        name = companion_name(spec, self.settings["companion_name"])
        if not on:
            msg = (f"{name} is hidden. Open FocusBuddy and click “Show {name}” on Home "
                   f"(or use the tray icon) to bring {name} back.")
            self.tray.notify("FocusBuddy", msg, key="companion-hidden", min_interval=5)
            if self.dashboard is not None and self.dashboard.isVisible():
                self.dashboard.toast(f"{name} is hidden - click “Show {name}” on Home to bring them back")
        elif self.dashboard is not None and self.dashboard.isVisible():
            self.dashboard.toast(f"{name} is back on your desktop!", "ok")
        if self.dashboard is not None:
            try:
                self.dashboard.pages[0].tick()
            except Exception:  # noqa: BLE001
                pass

    def toggle_nap(self) -> None:
        """Put the companion to sleep (FocusBuddy keeps running quietly) / wake it up."""
        life = self.companion.life
        if life.napping:
            life.wake_from_nap()
        else:
            life.nap(announce=not self.settings["quiet_mode"])
        self._refresh_tray()

    def on_dashboard_closed(self) -> None:
        """The dashboard's X: it hides (FocusBuddy keeps running); the companion reacts."""
        if self.quitting or not self.companion.visible_wanted:
            return
        mode = self.settings["close_button"]
        if mode == "ask":
            self.companion.life.close_scene()
        elif mode == "nap":
            self.companion.life.nap(announce=not self.settings["quiet_mode"])

    def save_life(self) -> None:
        try:
            self.settings.set("companion_life", self.companion.life.snapshot())
        except Exception:  # noqa: BLE001 - memory is best effort
            log.exception("saving the companion's memory failed")

    def quit(self) -> None:
        if self.quitting:
            return
        log.info("FocusBuddy exiting")
        self.save_life()
        try:
            self.routines.shutdown()
        except Exception:  # noqa: BLE001
            log.exception("routines shutdown failed")
        self.quitting = True
        try:
            if self.focus.active:
                self.stats.session_finished(self.focus.elapsed(), "cancelled")
            self.engine.reset()
            self.stats.flush()
        finally:
            self.poll_timer.stop()
            self.world_timer.stop()
            self.companion.stop()
            self.tray.hide()
            if self.bridge:
                self.bridge.stop()
            self.db.close()
            if self.dashboard is not None:
                self.dashboard.hide()
            # exit() (not quit()): Qt 6's quit() is cancelled when a window ignores its close event
            self.qapp.exit(0)

    # ------------------------------------------------------- browser focus policy
    def _browser_policy(self) -> dict:
        """What the extension blocks on its own during a focus session (read from the bridge thread)."""
        return dict(self._policy)

    def _compute_policy(self) -> dict:
        mode = self.settings["focus_enforcement"]
        if not self.monitoring or self.focus.phase != Phase.FOCUS or mode not in ("auto_close", "instant"):
            return {"active": False}
        cats = ("distracting", "blocked") if mode == "instant" else ("blocked",)
        block = sorted(r.pattern for r in self.rules.sites if r.category.value in cats)
        allow = sorted(r.pattern for r in self.rules.sites if r.category.value == "productive")
        allow += sorted(self.whitelist.active(time.time()))
        return {"active": bool(block), "block": block, "allow": allow, "mode": mode}

    def _sync_policy(self) -> None:
        p = self._compute_policy()
        sig = (p.get("active"), tuple(p.get("block", ())), tuple(p.get("allow", ())))
        now = time.time()
        if p.get("active"):
            p["until"] = int((now + 90) * 1000)       # expires on its own if FocusBuddy stops
        self._policy = p
        if sig != getattr(self, "_policy_sig", None):
            self._policy_sig = sig
            if self.bridge:
                self.bridge.notify_policy_changed()
            log.info("Browser focus policy %s", kv(active=p.get("active"), sites=len(p.get("block", ()))))

    def _on_browser_event(self, ev: dict) -> None:
        if ev.get("type") != "blocked":
            return
        host = ev.get("host") or "a distracting site"
        log.info("Extension blocked a site %s", kv(website=host))
        try:
            from .storage.database import day_key
            self.db.bump_stats(day_key(time.time()), blocked=1)
        except Exception:  # noqa: BLE001
            log.exception("could not record the block")
        self.companion.say(self._msg("FOCUS_BLOCKED", site=host)
                           or f"Not now - {host} can wait until the session is over.", duration=4)

    def _migrate_rules(self) -> None:
        """New default rules reach existing installs; the user's own rules / edits always win."""
        have = int(self.settings.get("rules_version", 1) or 1)
        if have >= RULES_VERSION:
            return
        existing = {(r["kind"], r["pattern"]) for r in self.db.rules()}
        added = 0
        for r in DEFAULT_RULES:
            if (r.kind, r.pattern) not in existing:
                self.db.upsert_rule(r.kind, r.pattern, r.category.value, r.auto_close)
                added += 1
        self.settings.set("rules_version", RULES_VERSION)
        log.info("Default rules updated %s", kv(version=RULES_VERSION, added=added))

    # ============================================================== main loop
    def _idle_seconds(self) -> float:
        try:
            return float(self.wm.get_idle_seconds())
        except Exception:  # noqa: BLE001
            return 0.0

    def _detector_context(self) -> DetectorContext:
        s = self.settings
        return DetectorContext(
            focus_active=self.focus.phase == Phase.FOCUS,
            tau=self.tau_override or s.sensitivity["tau"],
            thresholds=Thresholds(s["threshold_mild"], s["threshold_warning"], s["threshold_strong"]),
            distraction_seconds_today=self.stats.distraction_seconds_today(),
            allowance_seconds=s["daily_distraction_allowance_min"] * 60,
            detect_games=bool(s["detect_games"]),
            focus_speedup=float(s["focus_speedup"]))

    def _poll(self) -> None:
        now = time.time()
        dt = now - self._last_poll
        self._last_poll = now
        try:
            for ev in self.focus.tick():
                self._on_focus_event(ev)
        except Exception:  # noqa: BLE001
            log.exception("focus tick failed")
        acks = self.bridge.drain_acks() if self.bridge else []
        for ack in acks:
            if not ack["ok"]:
                log.warning("Browser command failed %s", kv(id=ack["id"], err=ack["error"]))
        try:
            # confirm tab / app closes; fall back (Ctrl+W on the verified tab, then minimize) if one didn't happen
            for r in self.blocker.check_pending(acks, now):
                if not r.ok:
                    self._say("CLOSE_FAILED")
        except Exception:  # noqa: BLE001
            log.exception("block verification failed")
        if self.bridge:
            for ev in self.bridge.drain_events():
                self._on_browser_event(ev)
        self._sync_policy()
        if not self.monitoring:
            return
        try:
            act = self.monitor.poll()
            if act is None or act.is_self:
                self.stats.add_tick(None, None, dt)
                self.engine.update(act, None)
                return
            asm = self.detector.analyze_activity(act, self._detector_context())
            self.current_activity, self.current_assessment = act, asm
            if asm.category == Category.PRODUCTIVE and act.handle:
                self.last_productive_handle = act.handle
            if act.key != self._last_glance_key:
                self._last_glance_key = act.key
                rect = to_logical(act.window)
                if rect is not None and self.engine.episode is None:
                    self.companion.glance(rect.cx, rect.cy, 2.0)
                    self.companion.behavior.active_window = (rect.x, rect.y, rect.w, rect.h)
            self.stats.add_tick(act, asm, dt)
            self._observe_context(act, asm)
            on_break = self.focus.phase == Phase.BREAK
            self.engine.update(act, asm, focus_active=self.focus.phase == Phase.FOCUS, on_break=on_break)
            if asm.score >= self.settings["threshold_mild"]:
                log.debug("Distraction score %s", kv(target=asm.target_key, score=asm.score, band=asm.band))
        except Exception:  # noqa: BLE001 - one bad poll must never kill monitoring
            log.exception("monitoring poll failed")

    # =========================================================== view (engine)
    def _rect(self, ep: Episode):
        try:
            fresh = self.wm.get_window(ep.activity.handle) if ep.activity.handle else None
            return to_logical(fresh or ep.activity.window)
        except Exception:  # noqa: BLE001
            return None

    def _msg(self, event: str, severity: int | None = None, ep: Episode | None = None, **kw) -> str:
        if ep is not None:
            kw.setdefault("site", ep.display)
            kw.setdefault("minutes", max(1, round(ep.duration(time.time()) / 60)))
            kw.setdefault("score", ep.assessment.score)
        return self.messages.get_message(event, severity, **kw)

    def _on_emote(self, key: str) -> None:
        """Dialogue for emotion-pack moments (a short cartoon shout, being poked, woken, dizzy)."""
        if key == "SHOUT":
            text = self._msg("SHOUT")
            if text:
                self.companion.say(text, duration=1.8)
            return
        self._say(key, chance={"POKED": 1.0, "WOKEN": 0.8, "DIZZY": 0.7}.get(key, 0.6))

    def _say(self, event: str, chance: float = 1.0, **kw) -> None:
        if random.random() > chance:
            return
        text = self._msg(event, **kw)
        if text:
            self.companion.say(text)

    def _buttons(self, ep: Episode, stage: str) -> tuple[list[tuple[str, str, bool]], str]:
        """Bubble buttons. 'Extra time' choices only while the allowance lasts."""
        extra = self.engine.can_override(ep.key)
        n_short, n_long = self.settings["snooze_short_min"], self.settings["snooze_long_min"]
        if stage == "warning":
            btns = [("return", "Return to Focus", True)]
            btns += [("snooze_short", f"Give Me {n_short} Minutes", False), ("ignore", "Ignore Once", False)] if extra \
                else [("close", "Close It", False)]
        elif stage == "intervention":
            btns = [("close", "Close It", True)]
            btns += [("keep_open", "Keep It Open", False), ("snooze_short", f"Give Me {n_short} Minutes", False)] \
                if extra else [("return", "Return to Focus", False)]
        else:  # countdown
            btns = ([("keep_open", "Keep Open", False)] if extra else [("return", "Return to Focus", False)])
            btns.append(("close_now", "Close Now", True))
            if extra:
                btns.append(("snooze_long", f"Give Me {n_long} Minutes", False))
        note = "" if extra else " · no extra time left"
        return btns, note

    def _ask_stage(self, ep: Episode, stage: str) -> None:
        btns, note = self._buttons(ep, stage)
        if stage == "warning":
            sev = 3 if ep.duration(time.time()) > 120 else 2
            self.companion.ask(self._msg("DISTRACTION_WARNING", sev, ep=ep), btns, title=f"Distraction · {ep.display}{note}")
        elif stage == "intervention":
            self.companion.ask(self._msg("DISTRACTION_INTERVENTION", ep=ep), btns,
                               title=f"Score {ep.assessment.score} · {ep.display}{note}")
        else:
            secs = max(0, int(round((ep.countdown_until or time.time()) - time.time())))
            self.companion.ask(self._msg("COUNTDOWN", ep=ep, seconds=secs), btns,
                               title=f"Auto-block · {ep.display}{note}", accent="#E5484D")

    def _ask_when_ready(self, ep: Episode, stage: str) -> callable:
        """Bubble appears after the companion has arrived and looked at you (max 5 s wait)."""
        shown = {"done": False}

        def show():
            if shown["done"] or self.engine.episode is not ep or ep.ignored:
                return
            shown["done"] = True
            self._ask_stage(ep, stage)
            self.sound.play("alert")
        QTimer.singleShot(5000, show)
        return show

    def show_notice(self, ep: Episode) -> None:
        self.companion.life.distraction_started(ep.key)
        self.companion.notice(self._rect(ep))
        text = self._msg("SNOOZE_OVER" if ep.again else "DISTRACTION_NOTICE", ep=ep)
        self.companion.say(text, title="Hmm…")
        self.sound.play("pop")

    def show_warning(self, ep: Episode) -> None:
        self.companion.life.distraction_started(ep.key)
        ready = self._ask_when_ready(ep, "warning")
        # notice -> head turn -> alert -> walk -> jog -> run -> short sprint -> natural stop
        # repeat offences get the grumpy look; otherwise (grumpy reactions on) it arrives annoyed, stomps and
        # shouts a cartoon "HEY!" before the warning - with them off it just points
        angry_ok = bool(self.settings["angry_reactions"])
        mood = "grumpy" if (ep.again and angry_ok) else ("angry" if angry_ok else "neutral")
        self.companion.investigate(self._rect(ep), run=False, arrive_state=S.WARNING, on_ready=ready,
                                   gait="escalate", mood=mood)

    def show_intervention(self, ep: Episode) -> None:
        self.companion.life.distraction_started(ep.key)
        ready = self._ask_when_ready(ep, "intervention")
        self.companion.investigate(self._rect(ep), run=True, arrive_state=S.WARNING, on_ready=ready)
        self.notify(f"You've been on {ep.display} for {max(1, round(ep.duration(time.time()) / 60))} minutes.",
                    key=f"dist:{ep.key}", min_interval=300)

    def show_countdown(self, ep: Episode, remaining: float, total: float, first: bool) -> None:
        secs = int(round(remaining))
        if first:
            # stand ready next to the X button (a playful "HEY!" first, if angry reactions are on)
            self.companion.prepare_press(self._rect(ep), shout=bool(self.settings["angry_reactions"]))
            self._ask_stage(ep, "countdown")
            self.sound.play("alert")
        else:
            self.companion.countdown(remaining, total, f"Closing {ep.display} in {secs}s…")
            if secs <= 3:
                self.sound.play("tick")

    def request_block(self, ep: Episode, do_block) -> None:
        """Walk to the close button and physically press it; the safe close runs at paw contact."""
        self.companion.hide_bubble()
        self.companion.press_close(self._rect(ep), do_block)

    def show_override_denied(self, ep: Episode) -> None:
        self.companion.say(self._msg("OVERRIDE_DENIED", ep=ep), duration=4)
        self.companion.play_emotion("GRUMPY" if self.settings["angry_reactions"] else "SHAKE_HEAD", hold=S.WARNING,
                                    seconds=1.8)
        stage = {4: "countdown", 3: "intervention"}.get(int(ep.level), "warning")
        QTimer.singleShot(1800, lambda: self.engine.episode is ep and not ep.ignored and self._ask_stage(ep, stage))

    def show_blocked(self, ep: Episode, result) -> None:
        self.companion.hide_bubble()
        event = "AUTO_CLOSED" if ep.level >= 4 else result.message_event
        if result.message_event == "NO_EXTENSION":
            event = "NO_EXTENSION"
        text = self._msg(event, ep=ep)

        def after():
            # happy -> jump -> celebrate -> dance -> back to roaming
            self.companion.life.distraction_ended(returned=True)
            self.companion.celebrate_return()
            self.companion.say(text)
            self.sound.play("celebrate")
        QTimer.singleShot(450, after)   # let the paw finish retracting first
        log.info("Distraction blocked %s", kv(target=ep.key, action=result.action))

    def show_block_failed(self, ep: Episode, result) -> None:
        self.companion.perform(S.CONFUSED, 2.5)
        self.companion.say(self._msg(result.message_event, ep=ep))

    def show_returned(self, ep: Episode, celebrate: bool) -> None:
        self.companion.life.distraction_ended(returned=True)
        self.companion.hide_bubble()
        if celebrate:
            self.companion.celebrate_return()
            self.companion.say(self._msg("RETURNED", ep=ep))
            self.sound.play("celebrate")
        else:
            self.companion.release()

    def show_snoozed(self, ep: Episode, minutes: int) -> None:
        self.companion.life.distraction_ended(returned=False)
        self.companion.release()
        self.companion.say(self._msg("SNOOZED", ep=ep, minutes=minutes))

    def show_ignored(self, ep: Episode, kind: str) -> None:
        # mood dips a little; after repeated ignored warnings for the same thing: a short comic sulk
        self.companion.life.distraction_ended(returned=False)
        self.companion.warning_ignored(ep.key)
        self.companion.say(self._msg("IGNORED" if kind == "ignore" else "KEEP_OPEN", ep=ep))

    def show_return_to_focus(self, ep: Episode, ok: bool) -> None:
        self.companion.life.distraction_ended(returned=ok)
        self.companion.release()
        self.companion.say(self._msg("RETURN_TO_FOCUS", ep=ep))

    def clear(self) -> None:
        self.companion.life.distraction_ended(returned=False)
        if self.companion.bubble.has_buttons and not self.companion.life.asking:
            self.companion.hide_bubble()
        if self.companion.state in (S.WATCHING, S.WARNING, S.ALERT, S.CHASING, S.CLOSING_DISTRACTION):
            self.companion.release()

    # ------------------------------------------------------------------ hooks
    def _record_episode(self, ep: Episode, now: float) -> None:
        self.stats.record_episode(ep, now)

    def _return_to_focus(self, ep: Episode) -> bool:
        h = self.last_productive_handle
        if h and self.wm.is_window(h) and h != ep.activity.handle:
            if self.wm.activate_window(h):
                return True
        # No productive window known: tuck the distraction away (user asked for it).
        if not self.safety.is_protected(ep.activity.exe):
            return self.app_blocker.minimize(ep.activity)
        return False

    def _on_bubble_action(self, action: str) -> None:
        if self.routines.on_bubble(action):
            return
        if action in ("coach_focus", "streak_focus"):
            self.companion.bubble.hide_now()
            self.start_focus(int(self.settings["focus_minutes"]))
            return
        if action in ("coach_later", "report_ok"):
            self.companion.bubble.hide_now()
            return
        if action == "report_open":
            self.companion.bubble.hide_now()
            self.show_dashboard("reports")
            return
        if self.companion.life.answer(action):
            return
        self.engine.handle_action(action)

    def _on_companion_clicked(self) -> None:
        if self.engine.episode is None and not self.settings["quiet_mode"] and not self.companion.life.busy():
            self._say("CLICKED", chance=0.7)

    def _chatter_interval(self) -> int:
        """Small talk every few minutes - more often the chattier the companion is set."""
        ch = max(0.05, float(self.settings["chatter"]))
        lo, hi = 2.0 + 4.0 * (1 - ch), 4.0 + 8.0 * (1 - ch)          # chatter 1: 2-4 min .. 0.3: ~5-10 min
        return int(random.uniform(lo, hi) * 60_000)

    def _can_chat(self, chance: float = 0.9) -> bool:
        life = self.companion.life
        return (self.engine.episode is None and self.settings["speech_bubbles"] and not self.settings["muted"]
                and self.companion.visible_wanted and not self.companion.life_hidden
                and not life.busy() and life.chat_ok(chance))

    def _happy_gesture(self) -> None:
        """A little bounce / wave / wag with what it says (only when it's free and not in a focus session)."""
        b = self.companion.behavior
        if self.focus.phase == Phase.FOCUS or b.state not in (S.IDLE, S.SITTING, S.LOOKING_AROUND):
            return
        from .companion.actions import Priority
        biped = self.companion.spec.skeleton == "biped"
        pool = ("WAVE", "HAPPY", "THUMBS_UP", "NOD", "BOUNCE") if biped else ("HAPPY", "TAIL_WAG", "BOUNCE", "PLAY_BOW")
        try:
            b.play_action(random.choice(pool), priority=Priority.IDLE)
        except Exception:  # noqa: BLE001 - a gesture is decoration
            pass

    def _chatter(self) -> None:
        """Unprompted, cheerful small talk: chatter, cheers, compliments, fun facts and caring reminders
        (stretch after a long stretch of work, water now and then). It stays quiet during distraction
        handling, in quiet mode, while you're away and (by the focus behaviour) in focus sessions."""
        interval = self.chatter_timer.interval() / 1000.0
        self.chatter_timer.setInterval(self._chatter_interval())
        idle = self._idle_seconds()
        self._active_s = self._active_s + interval if idle < 180 else 0.0
        if idle > 240 or not self._can_chat(0.95):
            return
        now = time.perf_counter()
        if self._active_s > 50 * 60 and now - self._last_reminder["STRETCH"] > 45 * 60:
            event = "STRETCH"
            self._last_reminder["STRETCH"] = now
        elif self._active_s > 25 * 60 and now - self._last_reminder["WATER"] > 70 * 60 and random.random() < 0.5:
            event = "WATER"
            self._last_reminder["WATER"] = now
        elif self.focus.phase == Phase.FOCUS:
            event = "ENCOURAGE"
        else:
            event = random.choices(("IDLE_CHATTER", "CHEER", "COMPLIMENT", "FUN"), (45, 25, 12, 18))[0]
        self._say(event)
        self._happy_gesture()

    def _greet_time_of_day(self) -> None:
        if not self._can_chat(1.0) or not self.settings["first_run_done"]:
            return
        h = datetime.datetime.now().hour
        part = "MORNING" if 5 <= h < 12 else "AFTERNOON" if h < 17 else "EVENING" if h < 22 else "NIGHT"
        self._say("GREET_" + part)
        self._happy_gesture()

    def _watch_presence(self) -> None:
        """Back after a while away: a happy welcome (every 1.5 s with the world refresh)."""
        idle = self._idle_seconds()
        try:
            self.routines.presence(idle)
        except Exception:  # noqa: BLE001
            log.exception("presence routine failed")
        if idle > 600:
            self._away = True
        elif self._away and idle < 3:
            self._away = False
            self._active_s = 0.0
            if self._can_chat(1.0):
                self._say("WELCOME_BACK")
                self._happy_gesture()

    # ================================================================== focus
    def start_focus(self, minutes: int, task_id: Optional[int] = None) -> None:
        self.focus_task_id = task_id
        if self.focus.phase == Phase.BREAK:
            self.focus.cancel()
            for ev in self.focus.tick():
                self._on_focus_event(ev)
        if self.focus.start(minutes):
            for ev in self.focus.tick():
                self._on_focus_event(ev)

    def pause_focus(self) -> None:
        self.focus.pause()
        for ev in self.focus.tick():
            self._on_focus_event(ev)

    def resume_focus(self) -> None:
        self.focus.resume()
        for ev in self.focus.tick():
            self._on_focus_event(ev)

    def end_focus(self) -> None:
        self.focus.cancel()
        for ev in self.focus.tick():
            self._on_focus_event(ev)

    def skip_break(self) -> None:
        if self.focus.phase == Phase.BREAK:
            self.end_focus()

    def _on_focus_event(self, ev) -> None:
        k = ev.kind
        m = int(round(ev.minutes))
        log.info("Focus event %s", kv(kind=k, minutes=m))
        try:
            self.companion.life.on_focus(k, **(ev.data or {}))
        except Exception:  # noqa: BLE001
            log.exception("life focus hook failed")
        try:
            self.routines.on_focus_event(k, ev.data or {})
        except Exception:  # noqa: BLE001
            log.exception("routines focus hook failed")
        if k == "focus_started":
            self.stats.session_started(self.focus.started_at, int(self.focus.planned))
            if self.focus_task_id is not None:
                self.db.set_session_task(self.stats.session_id, self.focus_task_id)
            self.engine.set_scope(f"focus:{self.focus.started_at:.0f}")
            self.companion.set_focus_mode(True)
            self.companion.set_break_mode(False)
            self.companion.focus_start()
            self._say("FOCUS_START", minutes=m)
        elif k == "focus_paused":
            self._say("FOCUS_PAUSED")
        elif k == "focus_resumed":
            self._say("FOCUS_RESUMED")
        elif k == "focus_halfway":
            self._say("FOCUS_HALFWAY")
        elif k == "focus_completed":
            self._credit_task(ev.data.get("focused_seconds", 0), True)
            self.stats.session_finished(ev.data.get("focused_seconds", 0), "completed")
            self.engine.set_scope(None)
            self.companion.set_focus_mode(False)
            self.companion.focus_complete()
            self.companion.say(self._msg("FOCUS_COMPLETE", minutes=m), duration=7)
            self.sound.play("celebrate")
            self.notify(f"Focus session completed! You focused for {m} minutes.", key="focus_done", min_interval=5)
            if self.settings["auto_start_break"]:
                every = max(2, self.settings["long_break_every"])
                long = ev.data.get("count", 0) % every == 0
                self.focus.start_break(self.settings["long_break_minutes" if long else "break_minutes"])
        elif k == "focus_cancelled":
            self._credit_task(ev.data.get("focused_seconds", 0), False)
            self.stats.session_finished(ev.data.get("focused_seconds", 0), "cancelled")
            self.engine.set_scope(None)
            self.companion.set_focus_mode(False)
            self.companion.release()
            self._say("FOCUS_CANCELLED")
        elif k == "break_started":
            self.companion.set_break_mode(True)
            QTimer.singleShot(3500, lambda: self.companion.say(self._msg("BREAK_START", minutes=m), duration=7))
        elif k == "break_completed":
            self.companion.set_break_mode(False)
            if not ev.data.get("skipped"):
                self._say("BREAK_OVER")
                self.notify(f"Your {m}-minute break is over.", key="break_done", min_interval=5)
                self.sound.play("notify")
                if self.settings["auto_start_next"]:
                    self.start_focus(self.settings["focus_minutes"])
        self._refresh_tray()

    # ============================================================== settings
    def _update_message_defaults(self) -> None:
        spec = get_character(self.settings["character"])
        self.messages.set_defaults(name=self.settings["user_name"] or "friend",
                                   buddy=companion_name(spec, self.settings["companion_name"]))

    def _apply_companion_settings(self) -> None:
        s, c = self.settings, self.companion
        c.speech_enabled = s["speech_bubbles"] and not s["muted"]
        c.set_speed(s["movement_speed"])
        c.set_opacity(s["companion_opacity"])
        c.set_intensity(s["animation_intensity"])
        c.set_fps(s["animation_fps"])
        c.click_reaction = s["click_reaction"]
        c.petting_enabled = s["petting"]
        c.footsteps = s["footstep_sounds"]
        c.animator.celebration = s["celebration_style"] or None
        b = c.behavior
        b.walking_frequency = s["walking_frequency"]
        b.idle_activity = s["idle_activity"]
        b.idle_sleep_after = s["idle_threshold_seconds"]
        b.cursor_mode = s["cursor_interaction"]
        b.roam_level = s["roaming"]
        b.edge_peeking = s["edge_peeking"]
        b.stay_near_focus = s["stay_near_during_focus"]
        c.animator.idle_activity = s["idle_activity"]
        c.set_motion_settings(animation_smoothness=s["animation_smoothness"],
                              movement_smoothness=s["movement_smoothness"], secondary_motion=s["secondary_motion"],
                              physics_strength=s["physics_strength"], character_energy=s["character_energy"],
                              particles=s["particles"], time_of_day=s["time_of_day_behavior"],
                              surprises=s["surprise_events"])
        c.debug_draw = {"skeleton": True, "ik": True, "contacts": True} if s["debug_skeleton"] else None
        b.angry_reactions = bool(s["angry_reactions"])
        c.animator.emotion_amount = {"low": 0.5, "normal": 1.0, "high": 1.4}.get(s["emotion_intensity"], 1.0)
        from .companion.expression import LEVELS as _LIFE
        c.animator.face.amount = _LIFE.get(s["lifelike_level"], 1.0)
        from .companion.renderer import set_default_mode
        set_default_mode(s["renderer"], s["render_quality"])
        c.auto_quality = bool(s["auto_quality"])
        c.set_performance(s["performance_mode"])
        c.set_render_mode(s["renderer"], s["render_quality"])
        c.set_shadows(bool(s["shadows"]))
        b.cursor_interaction = s["cursor_reactions"]
        b.stumbles = bool(s["stumbles"])
        b.window_awareness = bool(s["window_interactions"])
        c.life.apply_settings(playfulness=float(s["playfulness"]), roaming_frequency=float(s["roaming_frequency"]),
                              chatter=float(s["chatter"]) if s["speech_bubbles"] else 0.0,
                              focus_behavior=s["focus_behavior"], roam_area=s["roam_area"],
                              quiet=bool(s["quiet_mode"]), leave_and_return=bool(s["leave_and_return"]),
                              user_name=s["user_name"] or "friend")
        self._apply_sound_level(s["sound_level"])

    def _apply_sound_level(self, level: str) -> None:
        vol = {"off": 0.0, "low": 0.25, "medium": 0.5, "high": 0.9}.get(level)
        if vol is None:
            return
        self.sound.enabled = level != "off" and self.settings["sound_enabled"]
        self.sound.volume = vol

    def _fullscreen_app(self) -> bool:
        """Is a full-screen app / presentation / game in front (covering a whole monitor)?"""
        try:
            act = self.wm.get_active_window()
        except Exception:  # noqa: BLE001
            return False
        if act is None or getattr(act, "pid", 0) == os.getpid():
            return False
        if (act.exe or "").lower() in ("explorer.exe", "searchhost.exe", "shellexperiencehost.exe",
                                       "startmenuexperiencehost.exe", "lockapp.exe"):
            return False
        r = to_logical(act)
        if r is None:
            return False
        from PySide6.QtGui import QGuiApplication
        for scr in QGuiApplication.screens():
            g = scr.geometry()
            if r.x <= g.x() + 1 and r.y <= g.y() + 1 and r.right >= g.x() + g.width() - 1 and \
                    r.bottom >= g.y() + g.height() - 1:
                return True
        return False

    def _refresh_world(self) -> None:
        """Give the companion its world: visible windows (logical coords) it can stand on / avoid."""
        if time.perf_counter() - getattr(self, "_life_saved", time.perf_counter()) > 60.0:
            self._life_saved = time.perf_counter()
            self.save_life()
        if self.settings["hide_in_fullscreen"] and self.wm.available:
            self.companion.life.set_fullscreen(self._fullscreen_app())
        self.tray.set_napping(self.companion.life.napping)
        if not self.companion.visible_wanted:
            return
        need = self.settings["roaming"] != "floor" or self.engine.episode is not None
        if not need:
            self.companion.set_windows(None)
            return
        try:
            infos = self.wm.list_windows()
        except Exception:  # noqa: BLE001
            infos = []
        env = self.companion.env
        char_h = self.companion.px
        out: list[dict] = []
        higher: list = []
        for info in infos:
            r = to_logical(info)
            if r is None:
                continue
            l, t, rr, b = env.screen_bounds(r.cx, r.cy)
            mid = (max(r.x, l) + min(r.right, rr)) / 2
            covered = any(h.x <= mid <= h.right and h.y <= r.y - 3 <= h.bottom for h in higher)
            perch = (not covered and r.y - char_h * 0.95 >= t and r.w >= char_h * 1.6
                     and r.y < b - char_h * 0.5)
            out.append({"handle": r.handle, "x": r.x, "y": r.y, "w": r.w, "h": r.h, "perchable": perch,
                        "minimized": info.minimized})
            higher.append(r)
        self.companion.set_windows(out)

    def _apply_theme(self) -> None:
        """Light / dark / system theme + accent, switched live: the app style sheet is replaced and
        every open window repaints in place (nothing is closed or rebuilt)."""
        from .ui.theme import STATE, apply_theme, retheme_all
        before = (STATE.get("dark"), STATE.get("accent_hex"))
        apply_theme(self.qapp, str(self.settings["theme"]), str(self.settings["accent"]))
        from .ui.theme import C
        STATE["accent_hex"] = C["primary"]
        if before == (STATE.get("dark"), STATE.get("accent_hex")):
            return
        retheme_all(self.qapp)

    def _apply_motion(self) -> None:
        from .ui import motion
        from .ui.theme import set_motion
        level = str(self.settings["motion"])
        if level == "full" and not motion.windows_animations_enabled():
            level = "reduced"                    # Windows "Animation effects" is off
        set_motion(level)
        motion.install(self.qapp)

    # ------------------------------------------------------------ profiles / lock
    def lock(self) -> None:
        """Lock the window (profiles with a PIN). The companion keeps running."""
        if self.profile is None or not self.profile.has_pin or self.locked:
            return
        self.locked = True
        if self.dashboard is not None:
            self.dashboard.hide()
        log.info("Locked %s", kv(profile=self.profile.id))

    def _unlock(self) -> bool:
        from .ui.signin import SignInWindow
        w = SignInWindow(self.profiles, "unlock", self.profile.id, self._icon)
        w.exec()
        if w.switch_requested:
            self.switch_profile()
            return False
        if w.result_id == self.profile.id:
            self.locked = False
            return True
        return False

    def switch_profile(self, extra: list | None = None) -> None:
        """Sign out: FocusBuddy restarts into the profile chooser."""
        if self.profiles is None:
            return
        try:
            self.profiles.forget_session()          # signing out: the next start asks to log in
        except Exception:  # noqa: BLE001
            log.exception("could not forget the session")
        self.restart_args = ["--choose-profile"] + list(extra or [])
        self.quit()

    def delete_current_profile(self) -> None:
        if self.profiles is None or self.profile is None or len(self.profiles.profiles) <= 1:
            return
        self.switch_profile(["--delete-profile", self.profile.id])

    def _idle_lock(self) -> None:
        p = self.profile
        if p is not None and p.has_pin and p.lock_after_min and not self.locked:
            if self._idle_seconds() >= p.lock_after_min * 60:
                self.lock()

    # ============================================================ intelligence
    def focus_today_seconds(self) -> float:
        """Focused seconds today (cached for a minute - the database read flushes the stats)."""
        t, v = self._focus_today_cache
        if time.perf_counter() - t > 60:
            try:
                v = float(self.stats.today()["focus_seconds"])
            except Exception:  # noqa: BLE001
                v = 0.0
            self._focus_today_cache = (time.perf_counter(), v)
        return v

    def _observe_context(self, act, asm) -> None:
        try:
            phase = {Phase.FOCUS: "focus", Phase.PAUSED: "focus", Phase.BREAK: "break"}.get(self.focus.phase, "idle")
            cat = asm.category.value if not asm.whitelisted else "neutral"
            if self.engine.episode is not None and not asm.whitelisted:
                cat = asm.category.value
            moments = self.context.observe(act.key, cat, self._idle_seconds(), phase,
                                           focus_today=self.focus_today_seconds(),
                                           streak=self._streak_cached(), sessions_today=self._sessions_today_cached())
        except Exception:  # noqa: BLE001 - context is a bonus, never break monitoring
            log.exception("context observe failed")
            return
        for m in moments:
            self._on_moment(m)

    def _streak_cached(self) -> int:
        c = getattr(self, "_streak_c", None)
        if c is None or time.perf_counter() - c[0] > 600:
            c = (time.perf_counter(), self.db.streak_days())
            self._streak_c = c
        return c[1]

    def _sessions_today_cached(self) -> int:
        c = getattr(self, "_sess_c", None)
        if c is None or time.perf_counter() - c[0] > 120:
            try:
                n = int(self.db.day_summary(datetime.date.today().strftime("%Y-%m-%d"))["sessions"])
            except Exception:  # noqa: BLE001
                n = 0
            c = (time.perf_counter(), n)
            self._sess_c = c
        return c[1]

    MOMENT_REACTIONS = {
        # moment: (dialogue key, gesture for people, gesture for pets, emotion, chance)
        "FLOW": ("FLOW", "THUMBS_UP", "TAIL_WAG", "PROUD", 0.8),
        "RESTLESS": ("RESTLESS", "THINK", "CURIOUS", "WORRIED", 0.9),
        "MARATHON": ("MARATHON", "STRETCH", "STRETCH", "SLEEPY", 1.0),
        "LATE_NIGHT": ("LATE_NIGHT", "YAWN", "YAWN", "SLEEPY", 1.0),
        "QUICK_RETURN": ("QUICK_RETURN", "CLAP", "HAPPY", "EXCITED", 0.9),
        "MILESTONE": ("MILESTONE", "CELEBRATE", "BOUNCE", "PROUD", 1.0),
        "SLUMP": ("SLUMP", "SIGH", "SIGH", "WORRIED", 0.8),
    }

    def _on_moment(self, m) -> None:
        from .intel.context import memory_text
        text = memory_text(m)
        if text:
            self.db.remember(m.kind.lower(), text, m.data)
        if not self.settings["context_reactions"] or self.settings["quiet_mode"]:
            return
        if m.kind == "STREAK_RISK":
            if self._can_chat(1.0) and self.focus.phase not in (Phase.FOCUS, Phase.PAUSED):
                self.companion.ask(self._msg("STREAK_RISK", streak=m.data.get("streak", 2)),
                                   [("streak_focus", "Start a session", True), ("coach_later", "Later", False)],
                                   title=f"🔥 {m.data.get('streak', 2)}-day streak")
            return
        r = self.MOMENT_REACTIONS.get(m.kind)
        if r is None:
            return
        key, g_people, g_pets, emo, chance = r
        in_focus = self.focus.phase == Phase.FOCUS
        if in_focus and m.kind not in ("MILESTONE", "LATE_NIGHT", "MARATHON"):
            return                                  # sessions stay quiet
        try:
            from .companion.emotion import Emotion
            self.companion.behavior.set_emotion(Emotion(emo), 0.8, 8.0)
        except Exception:  # noqa: BLE001
            pass
        if not self._can_chat(chance):
            return
        self._say(key, hours=m.data.get("hours", 1), minutes=int(m.data.get("minutes", 0) or 0))
        b = self.companion.behavior
        if b.state in (S.IDLE, S.SITTING, S.LOOKING_AROUND, S.WALKING) and not in_focus:
            from .companion.actions import Priority
            biped = self.companion.spec.skeleton == "biped"
            try:
                b.play_action(g_people if biped else g_pets, priority=Priority.IDLE)
            except Exception:  # noqa: BLE001 - a gesture is decoration
                pass

    def _intel_tick(self) -> None:
        """Once a minute: the coach's pre-emptive nudge, the end-of-day report."""
        now = time.perf_counter()
        try:
            if (self.settings["coach_nudges"] and self.monitoring and self.focus.phase == Phase.IDLE
                    and self._idle_seconds() < 90 and now - self._last_nudge > 45 * 60
                    and datetime.datetime.now().minute < 20 and self._can_chat(1.0)):
                line = self.coach.nudge()
                if line:
                    self._last_nudge = now
                    self.companion.ask(line, [("coach_focus", f"Start {self.settings['focus_minutes']} min", True),
                                              ("coach_later", "Not now", False)], title="Focus coach")
            today = datetime.date.today().strftime("%Y-%m-%d")
            if (self.settings["daily_report"] and datetime.datetime.now().hour >= 18
                    and self.settings["last_report_day"] != today and self._idle_seconds() < 120
                    and self.focus.phase == Phase.IDLE and self._can_chat(1.0)):
                from .intel.reports import day_report
                self.stats.flush()
                rep = day_report(self.db, coach=self.coach)
                if rep["summary"]["focus_seconds"] + rep["summary"]["distraction_seconds"] > 1800:
                    self.settings.set("last_report_day", today)
                    self.companion.ask(" ".join(rep["lines"][:2]), [("report_open", "See my day", True),
                                                                   ("report_ok", "Nice!", False)],
                                       title="Your day so far")
        except Exception:  # noqa: BLE001
            log.exception("intel tick failed")
        try:
            self.routines.tick()
            for fn in list(self.care_listeners):
                fn()
        except Exception:  # noqa: BLE001
            log.exception("routines tick failed")

    def _credit_task(self, focused_seconds: float, completed: bool) -> None:
        tid = self.focus_task_id
        if tid is None:
            return
        try:
            self.db.add_task_time(tid, float(focused_seconds or 0), session=True)
            t = self.db.task(tid)
            if t is not None:
                self.db.remember("session", f"{int(round(float(focused_seconds or 0) / 60))} min on “{t['title']}”"
                                 + ("" if completed else " (stopped early)"))
        except Exception:  # noqa: BLE001
            log.exception("task credit failed")
        self.focus_task_id = None
        self.tasks_changed()

    def complete_task(self, task_id: int) -> None:
        t = self.db.task(task_id)
        self.db.update_task(task_id, done_at=time.time())
        if t is not None:
            self.db.remember("task_done", f"Finished “{t['title']}”.")
            if self._can_chat(1.0):
                self.companion.say(self._msg("TASK_DONE", task=t["title"]), duration=5)
                try:
                    from .companion.actions import Priority
                    b = self.companion.behavior
                    biped = self.companion.spec.skeleton == "biped"
                    b.play_action("CLAP" if biped else "HAPPY", priority=Priority.IDLE)
                except Exception:  # noqa: BLE001
                    pass
        self.tasks_changed()

    def tasks_changed(self) -> None:
        for fn in list(self.tasks_listeners):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass

    def start_break(self, minutes: int) -> None:
        if self.focus.phase in (Phase.FOCUS, Phase.PAUSED):
            self.focus.cancel()
            for ev in self.focus.tick():
                self._on_focus_event(ev)
        self.focus.start_break(int(minutes))
        for ev in self.focus.tick():
            self._on_focus_event(ev)

    def chat(self, text: str, allow_llm: bool = True):
        """One chat turn (call from a worker thread when the local model may be used)."""
        r = self.assistant.reply(text, allow_llm=allow_llm and bool(self.settings["ai_chat_llm"]))
        return r

    def _on_setting(self, key: str, value) -> None:
        if key == "motion":
            self._apply_motion()
            return
        if key in ("theme", "accent"):
            QTimer.singleShot(0, self._apply_theme)
            return
        c = self.companion
        if key in ("animation_intensity", "animation_fps", "click_reaction", "petting", "footstep_sounds",
                   "celebration_style", "cursor_interaction", "roaming", "edge_peeking", "stay_near_during_focus",
                   "walking_frequency", "idle_activity", "sound_level", "animation_smoothness",
                   "movement_smoothness", "secondary_motion", "physics_strength", "character_energy", "particles",
                   "time_of_day_behavior", "surprise_events", "debug_skeleton", "angry_reactions",
                   "emotion_intensity", "renderer", "render_quality", "performance_mode", "auto_quality",
                   "shadows", "playfulness", "roaming_frequency",
                   "chatter", "focus_behavior", "roam_area", "quiet_mode", "leave_and_return", "lifelike_level"):
            self._apply_companion_settings()
            if key == "quiet_mode":
                self.tray.set_quiet(bool(value))
            return
        if key in ("ambient_sound", "ambient_volume", "ambient_mode"):
            self.routines.apply_ambient()
            return
        if key in ("ollama_url", "ai_model"):
            self.llm.configure(self.settings["ollama_url"], self.settings["ai_model"])
            return
        if key == "ai_chat_llm":
            self.assistant.llm = self.llm if value else None
            return
        if key == "personality":
            c.set_personality(value)
            spec = get_character(self.settings["character"])
            self.messages.set_personality(value or spec.personality)
            return
        if key == "debug_overlay":
            c.set_debug(bool(value))
            return
        if key == "idle_activity_level":
            lvl = {"low": 0.2, "normal": 0.5, "high": 0.85}.get(str(value))
            if lvl is not None:
                self.settings.set("idle_activity", lvl)
            return
        if key == "size_preset":
            preset = self.preset_size(value)
            if preset is not None:
                self.settings.set("companion_size", preset)
            return
        if key == "companion_size":
            c.set_size(value)
        elif key == "companion_opacity":
            c.set_opacity(value)
        elif key == "movement_speed":
            c.set_speed(value)
        elif key == "show_companion":
            c.set_visible(value)
            self.tray.set_companion_visible(value)
        elif key == "always_on_top":
            c.set_always_on_top(value)
        elif key == "click_through":
            c.set_click_through(value)
        elif key in ("speech_bubbles", "muted"):
            c.speech_enabled = self.settings["speech_bubbles"] and not self.settings["muted"]
            self.sound.muted = self.settings["muted"]
            self.tray.set_muted(self.settings["muted"])
            self._apply_companion_settings()
        elif key == "sound_enabled":
            self.sound.enabled = value and self.settings["sound_level"] != "off"
        elif key == "sound_volume":
            self.sound.volume = value
        elif key in ("user_name", "companion_name"):
            self._update_message_defaults()
            c.life.settings.user_name = self.settings["user_name"] or "friend"
        elif key == "idle_threshold_seconds":
            self.stats.idle_threshold = value
            c.behavior.idle_sleep_after = value
        elif key == "poll_interval_ms":
            self.poll_timer.setInterval(max(250, int(value)))
        elif key == "monitoring_enabled" and bool(value) != self.monitoring:
            self.set_monitoring(bool(value))

    def set_monitoring(self, on: bool) -> None:
        if on == self.monitoring:
            return
        self.monitoring = on
        if self.settings["monitoring_enabled"] != on:
            self.settings.set("monitoring_enabled", on)
        if not on:
            self.engine.reset()
            self.stats.flush()
        self._say("MONITORING_RESUMED" if on else "MONITORING_PAUSED")
        log.info("Monitoring %s", "started" if on else "paused")
        self._refresh_tray()

    def set_autostart(self, on: bool) -> None:
        ok = self.autostart.set_enabled(on)
        if not ok and on:
            log.warning("Autostart is not supported on this platform")

    def set_character(self, cid: str) -> None:
        spec = get_character(cid)
        self.settings.set("character", spec.id)
        self.companion.set_character(spec)
        if self.settings["size_preset"] == "quarter":          # Giant follows the new body's shape
            self.settings.set("companion_size", self.preset_size("quarter"))
        self.messages.set_personality(self.settings["personality"] or spec.personality)
        self._update_message_defaults()
        self._icon = app_icon(spec)
        self.qapp.setWindowIcon(self._icon)
        self._refresh_tray()
        if self.dashboard:
            self.dashboard.refresh_character()
        # a quick hello, then back to its own life (then_home=False left it waving forever)
        from .companion.actions import Priority
        b = self.companion.behavior
        if b.mode != "held":
            b._start_plan([("state", S.WAVING, 1.1)] + b.followup_steps(), mode="auto",
                          priority=Priority.INTERACTION)
        self._say("GREETING")

    def reload_rules(self) -> None:
        self.rules = RuleSet.from_rows(self.db.rules())
        self.detector.set_rules(self.rules)
        log.info("Rules reloaded count=%s", len(self.rules.sites) + len(self.rules.apps))

    # ================================================================== shell
    def status(self) -> tuple[str, str]:
        f = self.focus
        if not self.monitoring:
            return "paused", "Monitoring paused"
        if f.phase == Phase.FOCUS:
            m, s = divmod(int(f.remaining()), 60)
            return "focus", f"Focusing · {m}:{s:02d} left"
        if f.phase == Phase.PAUSED:
            return "paused", "Focus paused"
        if f.phase == Phase.BREAK:
            return "break", "On a break"
        return "monitoring", "Monitoring"

    def _refresh_tray(self) -> None:
        kind, text = self.status()
        self.tray.set_status(kind, text)
        self.tray.set_monitoring(self.monitoring)
        self.tray.set_focus_running(self.focus.active)
        self.tray.set_muted(self.settings["muted"])
        self.tray.set_companion_visible(self.settings["show_companion"])
        self.tray.set_quiet(bool(self.settings["quiet_mode"]))
        self.tray.set_napping(self.companion.life.napping)

    def notify(self, message: str, key: str | None = None, min_interval: float = 60.0) -> None:
        if self.settings["notifications_enabled"]:
            self.tray.notify("FocusBuddy", message, key, min_interval)

    def show_dashboard(self, page: str | None = None) -> None:
        if self.locked and not self._unlock():
            return
        try:
            if self.dashboard is None:
                from .ui.dashboard import Dashboard
                self.dashboard = Dashboard(self, self._icon)
            self.dashboard.show_page(page or "home")
            # FocusBuddy always opens full screen (maximized: the taskbar stays reachable)
            self.dashboard.setWindowState((self.dashboard.windowState() & ~Qt.WindowMinimized) | Qt.WindowMaximized)
            self.dashboard.showMaximized()
            self.dashboard.raise_()
            self.dashboard.activateWindow()
        except Exception:  # noqa: BLE001
            log.exception("could not open dashboard")

    def _offer_extension(self) -> None:
        if self.quitting or self.settings["extension_prompted"] or self.bridge is None:
            return
        self.settings.set("extension_prompted", True)
        if self.bridge.connected(within=600):
            return
        from .ui.wizard import ExtensionDialog
        self._ext_dialog = ExtensionDialog(self, self._icon)
        self._ext_dialog.show()

    def show_wizard(self) -> None:
        from .ui.wizard import SetupWizard
        self.wizard = SetupWizard(self, self._icon)
        self.wizard.exec()
        self.wizard = None
        self.settings.set("first_run_done", True)
        QTimer.singleShot(300, lambda: self._say("GREETING"))
        self.show_dashboard()

    def _show_context_menu(self, pos) -> None:
        m = QMenu()
        title = m.addAction("FocusBuddy")
        title.setEnabled(False)
        st = m.addAction(f"Status: {self.status()[1]}")
        st.setEnabled(False)
        m.addSeparator()
        if self.focus.active:
            if self.focus.phase == Phase.FOCUS:
                m.addAction("Pause Focus Session", self.pause_focus)
            else:
                m.addAction("Resume Focus Session", self.resume_focus)
            m.addAction("End Focus Session", self.end_focus)
        else:
            fm = m.addMenu("Start Focus Session")
            for mins in (25, 45, 60, 90):
                fm.addAction(f"{mins} minutes", lambda mins=mins: self.start_focus(mins))
            fm.addAction("Custom…", self._custom_focus)
        m.addAction("Resume Monitoring" if not self.monitoring else "Pause Monitoring",
                    lambda: self.set_monitoring(not self.monitoring))
        m.addAction("Settings", lambda: self.show_dashboard("settings"))
        m.addAction("Distraction History", lambda: self.show_dashboard("history"))
        cm = m.addMenu("Change Character")
        for spec in selectable_characters():
            a = QAction(f"{spec.emoji}  {spec.display_name}", cm)
            a.setCheckable(True)
            a.setChecked(spec.id == self.settings["character"])
            a.triggered.connect(lambda _=False, cid=spec.id: self.set_character(cid))
            cm.addAction(a)
        m.addAction("Statistics", lambda: self.show_dashboard("stats"))
        sm = m.addMenu("Size")
        for key, text in SIZE_LABELS:
            a = QAction(text, sm)
            a.setCheckable(True)
            a.setChecked(self.settings["size_preset"] == key)
            a.triggered.connect(lambda _=False, k=key: self.settings.set("size_preset", k))
            sm.addAction(a)
        sm.addSeparator()
        sm.addAction("Bigger  (Ctrl + wheel up)", lambda: self._resize_step(1))
        sm.addAction("Smaller  (Ctrl + wheel down)", lambda: self._resize_step(-1))
        m.addSeparator()
        care = self.routines.care
        km = m.addMenu(f"Care  ·  bond level {care.level}")
        fa = km.addAction(f"Give a treat 🍪  ({care.s.treats} left)", lambda: self.care_action("feed"))
        fa.setEnabled(care.s.treats > 0)
        km.addAction("Play together 🎾", lambda: self.care_action("play"))
        km.addAction("Pet 💛", lambda: self.care_action("pet"))
        km.addSeparator()
        km.addAction("Open the Care page", lambda: self.show_dashboard("care"))
        life = self.companion.life
        m.addAction("Wake up" if life.napping else "Take a nap 💤", self.toggle_nap)
        q = QAction("Quiet mode", m)
        q.setCheckable(True)
        q.setChecked(bool(self.settings["quiet_mode"]))
        q.triggered.connect(lambda on: self.settings.set("quiet_mode", bool(on)))
        m.addAction(q)
        m.addAction("Hide Companion", lambda: self.show_companion(False))
        m.addSeparator()
        m.addAction("Quit FocusBuddy", self.request_quit)
        m.exec(pos)

    def care_action(self, action: str) -> dict:
        res = self.routines.care_action(action)
        for fn in list(self.care_listeners):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
        return res

    def preset_size(self, key: str) -> float | None:
        """The size factor of a preset; "quarter" (Giant) makes the character about a quarter of the
        current screen's height tall."""
        if key == "quarter":
            from PySide6.QtGui import QGuiApplication
            from .companion.character import BASE_SIZE
            from .config import QUARTER
            scr = None
            try:
                scr = QGuiApplication.screenAt(self.companion.window.geometry().center())
            except Exception:  # noqa: BLE001
                pass
            scr = scr or QGuiApplication.primaryScreen()
            h = scr.availableGeometry().height() if scr is not None else 1040
            fill = 0.6 if self.companion.spec.skeleton == "quadruped" else 0.67   # character height / its box
            lo, hi = SIZE_RANGE
            return round(max(lo, min(hi, QUARTER * h / (BASE_SIZE * fill))), 2)
        return SIZE_PRESETS.get(key)

    def _resize_step(self, step: int) -> None:
        """Ctrl + wheel on the companion: 10 % bigger / smaller per notch."""
        lo, hi = SIZE_RANGE
        cur = float(self.settings["companion_size"])
        new = round(max(lo, min(hi, cur * (1.1 if step > 0 else 1 / 1.1))), 2)
        if abs(new - cur) > 1e-3:
            self.settings.set("size_preset", "custom")
            self.settings.set("companion_size", new)

    def _custom_focus(self) -> None:
        mins, ok = QInputDialog.getInt(None, "Custom focus session", "Minutes:", self.settings["focus_minutes"], 1, 600)
        if ok:
            self.start_focus(mins)
