<div align="center">

<img src="assets/icons/focusbuddy.png" width="96" alt="FocusBuddy icon">

# FocusBuddy

**A 3D desktop pet that walks over and closes YouTube when you drift during a focus session.**

<a href="https://github.com/theflighttechofficial/FocusBuddy-AI-Companion/releases/latest"><img src="https://img.shields.io/badge/Download_for_Windows-v2.3.0-8b5cf6?style=for-the-badge&logo=windows&logoColor=white" alt="Download for Windows"></a>

[![Windows build & test](https://github.com/theflighttechofficial/FocusBuddy-AI-Companion/actions/workflows/windows.yml/badge.svg)](https://github.com/theflighttechofficial/FocusBuddy-AI-Companion/actions/workflows/windows.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)
![Platform](https://img.shields.io/badge/Windows-10%20%7C%2011-0078D4?logo=windows)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white)
![Privacy](https://img.shields.io/badge/data-stays%20on%20your%20PC-ec4899)

<img src="docs/demo.gif" width="560" alt="Demo: the cat notices a YouTube tab, walks over, offers Return to Focus, closes the tab and celebrates">
<br><sub>Recorded from the built-in <code>--demo</code> mode: notice → walk over → ask → close the tab → celebrate.</sub>

</div>

## Quick start

1. **Download** [`FocusBuddy-Setup-2.3.0.exe`](https://github.com/theflighttechofficial/FocusBuddy-AI-Companion/releases/latest) and run it. No administrator rights needed. If SmartScreen appears, click **More info → Run anyway** (the installer is not code-signed).
2. **Pick a companion** in the setup wizard. It also walks you through loading the browser extension.
3. **Start a 25-minute focus session** from the Home page, then open youtube.com. Within a minute or two your companion walks over and asks you to get back to work.

No installer? Run from source in [three commands](#run-from-source), or try the safe demo with `python main.py --demo`.

## What's new: smoother on every PC

This update makes the companion and the app fluid, especially on Windows laptops and low-end PCs.

| Area | Before | Now |
|---|---|---|
| Walking and running | Judder: frame steps jumped between 0, 15 and 31 ms on Windows | Even steps, frames synced to your display's refresh |
| Blinking and talking | A 25–30 ms freeze on every blink frame | Only the changed eye and mouth pixels go to the GPU |
| State changes | Ears, tails and hands could jump in a single frame | Every change blends; worst jump reduced by about 3× |
| Opening the dashboard | Over 20 seconds on the first open | About 1.5 seconds |
| Slow or older PCs | Quality only went down, never back up | Detects your hardware and adjusts detail live, both ways |
| Laptops on battery | Same work as when plugged in | Follows Windows battery saver automatically |

See the [changelog](CHANGELOG.md) for the full list.

---

<div align="center">
<img src="docs/lineup.png" alt="The eight FocusBuddy companions: cat, dog, fox, rabbit, robot, Study Buddy, Study Girl and Saree Girl" width="100%">

[What's new](#whats-new-smoother-on-every-pc) · [Features](#features) · [Screenshots](#screenshots) · [How it works](#how-it-works) · [Install](#installation) · [Browser extension](#browser-extension) · [Privacy](#privacy--safety) · [Changelog](CHANGELOG.md) · [Technical reference](docs/TECHNICAL.md)
</div>

FocusBuddy is a desktop pet with a job. Your companion wanders your screen, naps on the taskbar and watches your cursor. When you slip onto a distracting site, it notices, walks over to that window and asks if you want to get back to work. If you allowed it, it presses the window's close button with its paw.

It asks before it acts. It says *"Hey… I think we wandered off task. Want to get back to it?"*, not *"STOP WASTING TIME"*.

## Features

**A companion that feels alive**
- Eight rigged, textured **3D characters**: cat, dog, fox, rabbit, robot, Study Buddy, Study Girl and Saree Girl. Each has its own personality, gait and dialogue.
- Walks, jogs, runs, sprints, sits, sleeps, stretches and plays, with planted feet, IK and spring physics on tails and ears.
- **No cuts between moves.** Any sudden change in a pose eases in over about 0.2 seconds instead of popping. Planted feet stay where they are.
- 25 emotions, read from body language and faces that blink, squint and smile.
- A daily routine: it wakes up, wanders, gets into mischief, leaves and comes back, and remembers you between launches.
- Click, drag, pet or poke it. Resize it with **Ctrl + mouse wheel**, from small up to a quarter of your screen.

**Smooth on any PC**
- **Performance mode** in *Settings → Companion → Appearance*: **Auto** (default), Best quality, Balanced or Battery saver.
- Auto checks your graphics card, CPU and memory, picks a starting quality, then raises or lowers the 3D detail live from real frame times. It waits before trying again, so it never flickers between levels.
- Frames follow your display's refresh rate. Battery saver drops to low detail at 30 FPS on its own.
- Heavy work stays off the screen's critical path: dashboard pages build when you first open them, and character previews load in the background.

**A focus engine that is fair**
- Knows the active app and, through the browser extension, the **exact website**.
- Scores each distraction from 0 to 100 using dwell time, revisits, app switching and context. A YouTube *Python tutorial* scores lower than a *funny compilation*.
- Escalates gently: **notice → warning → offer → optional countdown**. Extra time is allowed once per site per session.
- Focus sessions (25 / 45 / 60 / 90 min or custom) with breaks, a daily goal and a celebration at the end.

**An app you want to open**
- Dashboard with Home, Focus, Tasks, Care, Coach, Chat, Statistics, Reports, History, Sites & Apps, Companion and Playground pages.
- A local **Focus Coach** that learns your best and riskiest hours and plans your day.
- Light and dark themes, five accent colours, smooth motion, **Ctrl+K** command search and keyboard shortcuts.
- Local profiles with an optional PIN, a setup wizard and a tray icon.

## Screenshots

<table>
  <tr>
    <td width="50%"><img src="docs/screenshots/home.png" alt="Home page, light theme"><br><sub><b>Home</b> — today at a glance, with your companion in the banner</sub></td>
    <td width="50%"><img src="docs/screenshots/dark_home.png" alt="Home page, dark theme"><br><sub><b>Dark theme</b> — follows Windows or set it yourself</sub></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/focus.png" alt="Focus mode page"><br><sub><b>Focus mode</b> — sessions, goals, breaks and deep focus</sub></td>
    <td><img src="docs/screenshots/dark_stats.png" alt="Statistics page"><br><sub><b>Statistics</b> — focus vs distraction, streaks and recovered time</sub></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/coach.png" alt="Focus Coach page"><br><sub><b>Focus Coach</b> — learns your rhythm, on this PC only</sub></td>
    <td><img src="docs/screenshots/reports.png" alt="Reports page"><br><sub><b>Reports</b> — your day and week in plain words</sub></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/companion.png" alt="Companion picker page"><br><sub><b>Companion</b> — pick a character, size and personality</sub></td>
    <td><img src="docs/screenshots/care.png" alt="Companion Care page"><br><sub><b>Care</b> — focus earns treats; feed, play and grow your bond</sub></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/rules.png" alt="Sites and Apps rules page"><br><sub><b>Sites &amp; Apps</b> — decide what is focus, distracting or blocked</sub></td>
    <td><img src="docs/screenshots/history.png" alt="Distraction history page"><br><sub><b>History</b> — every episode, searchable and exportable to CSV</sub></td>
  </tr>
  <tr>
    <td><img src="docs/screenshots/playground.png" alt="Companion Playground page"><br><sub><b>Playground</b> — trigger actions, mix emotions, orbit the camera</sub></td>
    <td><img src="docs/screenshots/settings.png" alt="Settings page"><br><sub><b>Settings</b> — account, PIN, appearance, monitoring and blocking</sub></td>
  </tr>
</table>

<details>
<summary><b>First-run setup wizard and profiles</b></summary>
<br>

| Welcome | Choose your companion |
|---|---|
| ![Wizard welcome step](docs/screenshots/wizard_1.png) | ![Wizard companion step](docs/screenshots/wizard_2.png) |

| What to do about distractions | Ready to go |
|---|---|
| ![Wizard blocking step](docs/screenshots/wizard_4.png) | ![Wizard final step](docs/screenshots/wizard_7.png) |

| Profile chooser |
|---|
| ![Who's focusing today? profile chooser](docs/screenshots/signin_choose.png) |

</details>

<details>
<summary><b>All eight companions in motion</b></summary>
<br>

Rendered with the app's own 3D renderer: idle, walk, run, sit, wave, celebrate and sleep.

![Every companion in its main states](docs/companions_3d.png)

Face rig close-ups: neutral, blink, happy, smile, open mouth, sad and angry.

![Companion faces](docs/companion_faces_3d.png)

</details>

## How it works

```
Active window ─► Activity monitor ─► Distraction score (0–100) ─► Intervention engine ─► Companion walks over
      ▲                ▲   (browser extension ─► exact URL)              │      ▲ your choice
      │                │                                                 ▼      │
      └──── Safe blocker (protected apps never touched) ◄──── close tab / minimize / redirect
                                Stats ──► local SQLite ──► Dashboard
```

The monitor polls about once a second. Each level asks before it acts:

| Level | What the companion does |
|---|---|
| 1 · Notice | Turns its head and tracks the window with its eyes |
| 2 · Warning | Walks, then runs over, leaps onto the window and asks: *Return to Focus / Give Me 2 Minutes / Ignore Once* |
| 3 · Offer | *Close It / Keep It Open / Give Me 2 Minutes* |
| 4 · Countdown | Stands by the **✕** with a visible countdown and *Keep Open* first, then presses it |

Level 4 runs only during a focus session you started, or when you turn on *Automatic blocking*. See [Distraction detection](docs/TECHNICAL.md#distraction-detection) for the exact scoring.

## Installation

### Installer (recommended)

Download `FocusBuddy-Setup-2.3.0.exe` from the [latest release](https://github.com/theflighttechofficial/FocusBuddy-AI-Companion/releases/latest). It needs Windows 10 or 11 and nothing else: Python is bundled.

### Run from source

**Requirements:** Windows 10 or 11, and Python 3.10–3.13 from [python.org](https://www.python.org/downloads/) with *Add python.exe to PATH* ticked.

Double-click **`run.bat`**. The first run creates a virtual environment and installs the dependencies. Or from a terminal:

```bat
git clone https://github.com/theflighttechofficial/FocusBuddy-AI-Companion.git
cd FocusBuddy-AI-Companion
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The only runtime dependencies are **PySide6-Essentials** and **NumPy**.

### Build the installer yourself

```bat
build_installer.bat
```

This builds `FocusBuddy.exe` with PyInstaller, runs its self-test and compiles `installer\Output\FocusBuddy-Setup-2.3.0.exe` with Inno Setup 6. The installer:

- installs for your user only, with no administrator rights
- adds Start menu and optional desktop shortcuts, and *start when I sign in*
- has a normal uninstaller in *Settings → Apps*

> [!NOTE]
> The setup file is not code-signed. If Windows SmartScreen shows *"Windows protected your PC"*, click **More info → Run anyway**.

For a portable build without an installer, run `build_windows.bat`. The result is `dist\FocusBuddy\FocusBuddy.exe`, which runs without Python.

## Browser extension

A desktop app cannot reliably read the URL of a browser tab or close one tab safely. The **FocusBuddy Connector** extension in [`browser_extension/`](browser_extension/) does that part.

| Browser | Steps |
|---|---|
| Chrome, Edge, Brave, Opera, Vivaldi | Open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and choose the `browser_extension` folder |
| Firefox 121+ | Open `about:debugging#/runtime/this-firefox`, click **Load Temporary Add-on…** and pick `browser_extension/manifest.json` |

The setup wizard walks you through this and turns green when the extension connects.

The extension sends only the active tab's address, and only to `127.0.0.1:47621` on your own PC. It acts on a tab only if the tab still shows the same site. FocusBuddy works without the extension, too: it then reads the site from the window title and closes tabs with the browser's own **Ctrl+W**.

## Usage

| Command | What it does |
|---|---|
| `python main.py` | Normal start. The setup wizard runs on first launch |
| `python main.py --demo` | Safe scripted demo with fake editor and browser windows |
| `python main.py --minimized` | Start in the system tray |
| `python main.py --wizard` | Run the setup wizard again |
| `python main.py --playground` | Open the Companion Playground |
| `python main.py --character-showcase --character fox` | Watch one character perform every action |
| `python main.py --self-test` | Load and render every character offscreen, then exit 0 or 1 |
| `FocusBuddy.exe --self-test --self-test-report selftest.json` | The same check for a built app, with a JSON report |

**Try it on your real browser:**

1. Install the browser extension.
2. Right-click the companion and choose *Start Focus Session → 25 minutes*.
3. Open youtube.com and stay there. Within a minute or two, the companion walks over and asks.
4. Go back to your editor. The companion celebrates and the Statistics page updates.

**If the companion feels slow:** open *Settings → Companion → Appearance* and check that *Performance* is set to **Auto**. Pick **Battery saver** on very old PCs. The log (`%LOCALAPPDATA%\FocusBuddy\logs\focusbuddy.log`) records every quality change and the device class it detected.

**Shortcuts:** `Ctrl+K` search · `Ctrl+1`–`9` pages · `Ctrl+H` hide or show the companion · `Ctrl+L` lock · `Ctrl+,` settings · `F1` help.

## Privacy & safety

> [!IMPORTANT]
> **Everything stays on your PC.** There is no server, no account in the cloud and no upload.

| Stored locally | Never stored or collected |
|---|---|
| App names, website domains, durations, scores, actions | Window titles, full URLs, keystrokes, screenshots, passwords, message contents, files |

- PINs and passwords are kept only as salted PBKDF2-SHA256 hashes.
- The coach, reports and chat run on this PC. An optional Ollama model is reached only at a localhost address.
- Outside focus sessions, automatic blocking is **off by default**. It needs *Automatic blocking* mode **and** a rule marked *Auto-close*.
- Explorer, Task Manager, Settings, security tools, password managers and system processes are **never** touched. Terminals are protected unless you allow them.
- Apps that may hold unsaved work are minimized, not closed. `TerminateProcess` is never used.
- You can export or delete your history from the History page at any time.

## Tech stack

| Layer | Technology |
|---|---|
| App and UI | Python 3.10+, PySide6 (Qt 6), OpenGL |
| 3D renderer | NumPy CPU rasteriser with a GPU skinning path; GLB / glTF characters with PBR textures |
| Smoothness | Display-synced frame loop, pose inertialization, adaptive quality governor, background asset loading |
| Windows integration | Win32 through `ctypes` only: no pywin32, no hooks, no keylogger |
| Storage | SQLite (standard library) |
| Browser | Manifest V3 extension in plain JavaScript |
| Packaging | PyInstaller and Inno Setup, built and self-tested in GitHub Actions |

## Project structure

```
FocusBuddy/
├── main.py               entry point (CLI flags, single instance)
├── focusbuddy/
│   ├── core/             monitor, rules, scoring, interventions, focus timer, stats
│   ├── companion/        rig, gaits, IK, actions, brain, emotions, 3D renderers, overlay,
│   │                     pacing (display sync), performance (device profile + governor), inertia
│   ├── blocking/         safety guard, tab and app actions
│   ├── browser/          localhost bridge to the extension, title parser
│   ├── intel/            coach, reports, routines, chat assistant
│   ├── platform/         Win32 backend + simulated backend for the demo
│   ├── storage/          SQLite database and typed settings
│   └── ui/               dashboard, wizard, sign-in, tray, theme, playground
├── browser_extension/    FocusBuddy Connector (Chrome / Edge / Firefox)
├── assets/               characters (GLB + art), dialogue, motion capture, icons
├── installer/            Inno Setup script
├── tests/                pytest suite (timing, performance governor, inertialization, faces, runtime)
└── tools/                asset pipeline, rigging, screenshots, validators
```

The platform code sits behind `platform/base.py`. A macOS or Linux backend only needs to implement that interface.

## Development

```bat
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q                      # run the test suite
python -m tools.validate_asset --all     # check every bundled character
python main.py --self-test               # render every character offscreen
python tools/screenshots.py out/         # render every dashboard page with sample data
```

Timing rule for contributors: use `time.perf_counter()` for anything frame-related. `time.monotonic()` only updates every 15.6 ms on Windows and makes motion judder; a test fails if it appears in `focusbuddy/`.

The [technical reference](docs/TECHNICAL.md) covers the character system, how to add your own GLB character, animations and emotions, the database schema, configuration options and troubleshooting.

## Roadmap

- [ ] Chrome Web Store and signed Firefox add-on
- [ ] macOS and Linux window backends
- [ ] ML activity classifier behind the `DistractionDetector` interface
- [ ] Custom character creator
- [ ] Calendar integration and shared focus rooms
- [ ] Optional, opt-in sync

## License

The source code is under the [MIT License](LICENSE). The 3D character models and motion-capture clips in `assets/` keep their own third-party terms, listed below and in the LICENSE file.

## Credits

- 3D companions generated from the project's own character art with **Hunyuan3D** (Tencent) and **TripoSR**, then rigged by [`tools/rig`](tools/rig/). The Hunyuan3D models fall under the [Tencent Hunyuan 3D Community License](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE). Read it before any commercial use. Each character folder has its own `LICENSE.md`.
- Motion-capture clips from the [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu). See [`assets/mocap/LICENSE.md`](assets/mocap/LICENSE.md).

<div align="center">
<br>
<sub>Made with care for everyone who opened YouTube "just for a second".</sub>
</div>
