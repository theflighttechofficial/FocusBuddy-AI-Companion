"""Render every dashboard page + wizard step to PNG with sample data (visual QA / README).

    python tools/screenshots.py OUT_DIR
"""

import datetime as dt
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["FOCUSBUDDY_DATA_DIR"] = tempfile.mkdtemp(prefix="fb-shots-")

from PySide6.QtWidgets import QApplication  # noqa: E402

from focusbuddy.ui.theme import apply_theme  # noqa: E402


def seed(db):
    rng = random.Random(7)
    today = dt.date.today()
    for i in range(7):
        d = today - dt.timedelta(days=6 - i)
        day = d.strftime("%Y-%m-%d")
        focus = rng.randint(2 * 3600, 5 * 3600)
        db.add_activity_time({(day, "productive", "code.exe"): focus * 0.7,
                              (day, "productive", "github.com"): focus * 0.3,
                              (day, "distracting", "youtube.com"): rng.randint(600, 2400),
                              (day, "distracting", "instagram.com"): rng.randint(200, 1200),
                              (day, "distracting", "reddit.com"): rng.randint(100, 900),
                              (day, "distracting", "steam.exe"): rng.randint(0, 600)})
        base = dt.datetime.combine(d, dt.time(9, 0)).timestamp()
        for k in range(rng.randint(2, 4)):
            sid = db.start_session(base + k * 4000, 1500 if k % 2 else 2700)
            db.finish_session(sid, base + k * 4000 + 1500, rng.randint(1300, 2700), rng.randint(0, 3), "completed")
        for k in range(rng.randint(5, 14)):
            site = rng.choice(["youtube.com", "instagram.com", "reddit.com", None])
            db.add_distraction_event(timestamp=base + rng.randint(0, 30000), application="Steam" if site is None else "Chrome",
                                     website=site, duration=rng.randint(20, 900), score=rng.randint(42, 98),
                                     level=rng.randint(1, 4), action=rng.choice(["closed", "returned", "snoozed", "ignored", "minimized", "left"]),
                                     focus_session=rng.random() < 0.6)
        db.bump_stats(day, recovered_seconds=rng.randint(300, 1800), prevented=rng.randint(3, 9))


def main(out, theme="light"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    qapp = QApplication.instance() or QApplication([])
    apply_theme(qapp, theme)
    from focusbuddy.app import FocusBuddyApp
    from focusbuddy.platform.simulated import SimulatedWindowManager
    from focusbuddy.config import data_dir
    from focusbuddy.core.profiles import ProfileStore
    store = ProfileStore(data_dir())
    me = store.create("Varun", "cat", "2468")
    store.create("Aisha", "saree_girl")
    store.create("Study group", "initials")
    app = FocusBuddyApp(qapp, window_manager=SimulatedWindowManager(), start_minimized=True,
                        data_path=store.data_dir(me), profile=me, profiles=store)
    app.settings.update({"first_run_done": True, "user_name": "Varun", "theme": theme, "motion": "off"})
    app._apply_theme()
    seed(app.db)
    app.start_focus(45)
    app.focus._run_since -= 17 * 60  # pretend we're 17 minutes in
    app.show_dashboard()
    d = app.dashboard
    d.resize(1240, 860)
    for name in ["home", "focus", "stats", "history", "rules", "companion", "settings", "privacy", "help", "faq",
                 "about"]:
        d.show_page(name)
        qapp.processEvents()
        d.grab().save(str(out / f"page_{name}.png"))
    from focusbuddy.ui.about import AboutDialog
    from focusbuddy.ui.signin import SignInWindow
    for mode, name in (("choose", "signin_choose"), ("unlock", "signin_unlock"), ("create", "signin_create")):
        w = SignInWindow(store, mode, me.id)
        w.resize(900, 680)
        w.show()
        qapp.processEvents()
        w.grab().save(str(out / f"{name}.png"))
        w.close()
    ab = AboutDialog(d)
    ab.show()
    qapp.processEvents()
    ab.grab().save(str(out / "about.png"))
    ab.close()
    from focusbuddy.ui.wizard import SetupWizard
    w = SetupWizard(app)
    w.resize(820, 640)
    w.show()
    for i in range(w.stack.count()):
        w._go(i)
        qapp.processEvents()
        w.grab().save(str(out / f"wizard_{i + 1}.png"))
    print("saved to", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "screenshots", sys.argv[2] if len(sys.argv) > 2 else "light")
