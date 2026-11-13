"""FocusBuddy — AI Distraction Helper.

    python main.py                 normal start
    python main.py --minimized     start in the tray (used by "Start with Windows")
    python main.py --demo          safe demo with simulated windows (any OS)
    python main.py --wizard        show the first-run wizard again
    python main.py --debug         verbose logging
    python main.py --studio        animation studio (developer preview)
    python main.py --playground    companion playground (actions, sequences, skeleton view)
    python main.py --character-showcase   one character performing the whole range (cycles through all)
    python main.py --body-language-test   emotions with every icon / particle off (acting only)
    python main.py --emotion-showcase     22 emotions, body and face, camera cuts
    python main.py --showcase      living-character showcase (sleep ... rest) then the distraction flow
                                   (runs with simulated windows - nothing on your desktop is touched)
"""

from __future__ import annotations

import argparse
import sys
import traceback


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="FocusBuddy", description="Desktop companion that helps you stay focused.")
    p.add_argument("--minimized", action="store_true", help="start hidden in the system tray")
    p.add_argument("--demo", action="store_true", help="run the scripted demo with simulated windows")
    p.add_argument("--demo-capture", metavar="DIR", help="(demo) save frames of the demo into DIR")
    p.add_argument("--demo-fast", action="store_true", help="(demo) quit automatically when the demo finishes")
    p.add_argument("--wizard", action="store_true", help="show the setup wizard")
    p.add_argument("--choose-profile", action="store_true", help="show the profile chooser (sign in / switch)")
    p.add_argument("--profile", default="", help="sign in to this profile id")
    p.add_argument("--delete-profile", default="", help=argparse.SUPPRESS)
    p.add_argument("--debug", action="store_true", help="verbose logs")
    p.add_argument("--studio", action="store_true", help="open the animation studio (developer preview)")
    p.add_argument("--playground", action="store_true", help="open the companion playground")
    p.add_argument("--character-showcase", action="store_true",
                   help="character showcase: walk / jog / run / sprint / jump ... and full-body emotions")
    p.add_argument("--body-language-test", action="store_true",
                   help="emotions with all icons, particles and text switched off")
    p.add_argument("--emotion-showcase", action="store_true",
                   help="22 emotions in turn, body and face (effects off)")
    p.add_argument("--character", default="cat", help="(showcase / playground) character id")
    p.add_argument("--self-test", action="store_true",
                   help="load, validate and render every character offscreen, print a report, exit")
    p.add_argument("--self-test-report", default=None, help="(with --self-test) also write the JSON report here")
    p.add_argument("--showcase", action="store_true",
                   help="standalone showcase: the full living-character performance, then the demo "
                        "distraction flow (simulated windows)")
    args = p.parse_args(argv)
    if args.showcase:
        args.demo = True             # the showcase always runs on the safe simulated desktop
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if sys.platform.startswith("win"):
        # Crisp rendering on high-DPI displays and a proper taskbar identity.
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("TheFlightTechLabs.FocusBuddy")
        except Exception:  # noqa: BLE001
            pass

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication, QMessageBox

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    qapp = QApplication(sys.argv[:1])
    qapp.setApplicationName("FocusBuddy")
    qapp.setOrganizationName("The Flight Tech Labs")
    qapp.setQuitOnLastWindowClosed(False)

    from focusbuddy.ui.theme import apply_theme
    apply_theme(qapp, "system")          # the app re-applies the user's theme choice once settings load

    if args.self_test:
        from focusbuddy.selftest import run as self_test
        return self_test(args.self_test_report)

    if args.character_showcase or args.body_language_test or args.emotion_showcase:
        from focusbuddy.ui.showcase import CharacterShowcase
        win = CharacterShowcase(args.character, body_language=args.body_language_test,
                                emotions=args.emotion_showcase)
        win.show()
        qapp.setQuitOnLastWindowClosed(True)
        return qapp.exec()
    if args.playground:
        from focusbuddy.ui.playground import CompanionPlayground
        pg = CompanionPlayground(standalone=True)
        pg.show()
        return qapp.exec()
    if args.studio:
        from focusbuddy.ui.studio import AnimationStudio
        studio = AnimationStudio()
        studio.show()
        return qapp.exec()

    from focusbuddy.utils.single_instance import SingleInstance
    instance = SingleInstance("FocusBuddy-demo" if args.demo else "FocusBuddy")
    if not instance.acquire():
        instance.notify_running()
        print("FocusBuddy is already running - opened its dashboard.")
        return 0

    _shown = {"n": 0}

    def excepthook(exc_type, exc, tb):
        import logging
        logging.getLogger("focusbuddy").error("Unhandled error:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))
        if _shown["n"] < 1 and QApplication.instance() is not None:
            _shown["n"] += 1
            QMessageBox.warning(None, "FocusBuddy", "Something went wrong, but FocusBuddy is still running.\n"
                                                    "Details were written to the log file.")
    sys.excepthook = excepthook

    if args.demo:
        from focusbuddy.demo import DemoDirector
        director = DemoDirector(qapp, capture_dir=args.demo_capture, auto_quit=args.demo_fast,
                                showcase=args.showcase)
        app = director.app
        instance.on_activate(lambda: app.show_dashboard())
        director.start()
    else:
        from focusbuddy.app import FocusBuddyApp
        picked = choose_profile(args)
        if picked is None:
            instance.release() if hasattr(instance, "release") else None
            return 0
        store, profile, verified = picked
        app = FocusBuddyApp(qapp, debug=args.debug, force_wizard=args.wizard,
                            # started by clicking it: the dashboard opens. Started with Windows
                            # (--minimized): the "start quietly" setting decides
                            start_minimized=None if args.minimized else False,
                            data_path=store.data_dir(profile), profile=profile, profiles=store,
                            start_locked=profile.has_pin and profile.require_pin_on_start and not verified)
        instance.on_activate(lambda: app.show_dashboard())
        app.start()
    rc = qapp.exec()
    restart = getattr(locals().get("app"), "restart_args", None)
    if restart is not None:
        try:
            instance.release() if hasattr(instance, "release") else None
        except Exception:  # noqa: BLE001
            pass
        relaunch(restart)
    return rc


def choose_profile(args):
    """Sign in before the app starts. Returns (store, profile, verified) or None (closed).

    A device that is signed in (the account's "Keep me signed in") opens straight away - the log-in
    appears once. Otherwise: no profile yet -> sign up; accounts on this device -> log in; a profile
    from before accounts -> it gets an e-mail and password once (its data stays put). Several old
    profiles without accounts, or "Use a profile without an account", use the profile chooser."""
    from focusbuddy.config import data_dir
    from focusbuddy.core.profiles import ProfileStore
    store = ProfileStore(data_dir())
    if args.delete_profile and args.delete_profile in store.profiles and len(store.profiles) > 1:
        store.delete(args.delete_profile)
    if args.profile and store.get(args.profile):
        return store, store.get(args.profile), False
    if not args.choose_profile:
        p = store.remembered()
        if p is not None:
            store.touch(p.id)
            return store, p, False
    from focusbuddy.ui.login import LoginWindow
    accounts = [p for p in store.profiles.values() if p.has_account]
    legacy = [p for p in store.profiles.values() if not p.has_account]
    if not store.profiles:
        w = LoginWindow(store, "signup")
    elif accounts:
        w = LoginWindow(store, "login")
    elif len(legacy) == 1:
        w = LoginWindow(store, "upgrade", legacy[0].id)
    else:
        w = None
    if w is not None:
        w.exec()
        if w.result_id:
            return store, store.get(w.result_id), True
        if not w.want_profiles:
            return None
    # the profile chooser (profiles without accounts, PINs)
    from focusbuddy.ui.signin import SignInWindow
    c = SignInWindow(store, "choose")
    c.exec()
    if not c.result_id:
        return None
    p = store.get(c.result_id)
    if not p.has_account and not args.choose_profile:
        # once: the chosen profile gets an account so the next start is a normal log-in
        u = LoginWindow(store, "upgrade", p.id)
        u.exec()
        if u.result_id:
            return store, store.get(u.result_id), True
    return store, p, p.has_pin


def relaunch(extra: list[str]) -> None:
    """Start a fresh FocusBuddy (sign out / switch profile)."""
    import subprocess
    if getattr(sys, "frozen", False):
        cmd = [sys.executable] + list(extra)
    else:
        cmd = [sys.executable, str(__import__("pathlib").Path(__file__).resolve())] + list(extra)
    kw = {}
    if sys.platform.startswith("win"):
        kw["creationflags"] = 0x00000008 | 0x00000200          # DETACHED_PROCESS | NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    subprocess.Popen(cmd, close_fds=True, **kw)  # noqa: S603 - our own executable


if __name__ == "__main__":
    sys.exit(main())
