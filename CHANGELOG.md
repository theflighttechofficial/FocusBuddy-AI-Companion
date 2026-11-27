# Changelog

All notable changes to FocusBuddy. Full engineering notes for each release are in
[docs/TECHNICAL.md](docs/TECHNICAL.md).

## [Unreleased]

### Added
- **Performance mode** (Settings → Companion → Appearance): *Auto* (default), *Best quality*,
  *Balanced* and *Battery saver*. Auto profiles the PC once (graphics adapter, CPU cores, memory,
  battery / battery saver) to choose a starting quality and a ceiling. It then raises or lowers the
  3D detail live from measured frame times, with hysteresis so it never flips back and forth.
  Windows battery saver switches to the lowest quality at 30 FPS automatically.
- Frames are synchronised to the display refresh on Windows (`companion/pacing.py`). The old 16 ms
  timer ran at 62.5 Hz against a 60 Hz screen, so one frame in about 25 was doubled or dropped.
- Pose inertialization (`companion/inertia.py`): when the rig makes a discrete decision (which paw
  waves, the tail lifted off the floor, a clip starting), the jump is absorbed and fades over about
  0.2 s instead of appearing as a one-frame cut. Continuous motion passes through unchanged.
  Planted legs are excluded, so feet never slide.
- A `tests/` suite (the CI job ran `pytest` with no tests).
- **Bigger heads for the people**: Study Buddy, Study Girl and Saree Girl have heads 1.5× larger.
  The growth is baked into the model at load (`"head_grow": 1.5` in their `metadata.json`). It fades
  out smoothly down the neck, so shoulders, collars and long hair keep their size and nothing
  stretches. Their face rig, motion capture and every level of detail follow the new head.
- **Natural hand gestures for the people**: the hello wave lifts the elbow first, then swings the
  forearm from the elbow beside the head, with a loose wrist. Clapping hands visibly meet and part
  in front of the chest, pointing reaches out across the screen, and thumbs-up is held with a bent
  elbow. Gestures are placed for the 3/4 view, so they no longer point at the camera or cover the
  face. The robot waves out wide at shoulder height, which suits its short arms.
- **A real rabbit hop**: the front paws land, the hind feet swing up under the body, the hind legs
  push off and the rabbit arcs through the air, nose up and then nose down. Before, its legs
  stayed nearly locked straight and it shuffled in place.

### Fixed
- Switching companions left the new one **waving forever**. Every hello wave now lasts about
  1 second and flows straight into another activity. People no longer wave at nobody when they
  pick a "happy wag" idle.
- **Walk judder on Windows**: animation used `time.monotonic()`, which only ticks every 15.6 ms on
  Windows. Frame deltas jumped between 0, 15 and 31 ms. All timing now uses `time.perf_counter()`.
- **Blink and talk hitches**: every blink or mouth frame re-uploaded the whole 2048–4096 px face
  texture (25–30 ms). Only the changed eye and mouth tiles are sent now, and the redrawn lids are
  cached. Output is pixel-identical.
- **Dashboard took over 20 s to open**: the Companion page built eight full 3D previews up front.
  Pages are now built on first use. Previews load in the background (portrait first, then a
  cross-fade to the live model) and share one frame budget. Typical open time is about 1.5 s.
- Changing any companion setting rebuilt the 3D renderer (a visible hitch) and reset the adapted
  quality.
- After startup or a character switch, the first slow frame throttled the companion to 8 FPS for
  about 1.5 s.
- Root-caused pops: the jump take-off and landing pitch, the sitting tail flipping sides, and the
  waving hand switching arms mid-wave.
- Changing *Shadow* now applies immediately. *Adaptive 3D quality* became the Performance mode;
  if it was switched off, the mode starts as *Best quality*.
- The focus timer no longer counts negative time when the system clock is set back.

## [2.3.0] - 2026-11-13

### Changed
- Study Buddy, Study Girl and Saree Girl have rebuilt faces: a shorter, rounded chin and jaw, one
  seamless face texture without patch seams, and soft skin shading on face, ears, neck and arms.
- The jaw now follows the head bone, so head turns no longer shear it flat.
- Blinks close with skin from around the eye. The open mouth follows the lip line and the smile.

### Notes
- Turn the new face off per character with `"refined_face": false` in its `metadata.json`.

## [2.2.0]

### Added
- Command palette (Ctrl+K): every action, page, setting, companion and help topic, with fuzzy search.
- Quick actions on Home: Focus, Deep work 50, Break 5, Pause monitoring, Add a task.
- Drag to reorder Home sections.
- A live companion in the Home banner that follows your cursor and reacts when poked.
- Poke reflexes on the desktop: head, body and tail react differently to a click.

### Changed
- Baked ambient occlusion, softer studio lighting, colour-matched outlines and eye catchlights for
  all eight companions. Shapes and colours are unchanged.
- Natural blink timing and varied breathing.

## [2.1.0]

### Added
- Companion care (Care page): food, fun and love, treats earned by focusing, bond levels and badges.
- Naps off-screen: the companion walks off the screen edge to sleep and comes back the same way.
- Routines: morning check-in, evening wind-down, session goals and Deep focus mode.
- Ambient focus sounds generated on the PC: rain, café, brown and pink noise, forest, fireplace.

### Fixed
- Paws no longer skate: each frame is rendered off-screen, then the window moves and shows it in one step.
- Hind feet no longer trail behind (the "seal" walk). Feet follow the rig's planted feet.
- Natural walk cadence, smoother limbs, tails that no longer pass through the body.

## [2.0.0]

### Added
- Focus Coach: your best and riskiest hours, learned from your own history on this PC.
- Tasks with estimates and priorities; start a focus session on a task.
- Reports for today and this week.
- Chat: ask the companion to start a session, add a task or summarise your day.
- Talking faces, micro-expressions and new reactions (sneeze, hiccups, shiver, daydream, humming).

### Changed
- Faster CPU renderer.

[2.3.0]: https://github.com/theflighttechofficial/FocusBuddy-AI-Companion/releases/tag/v2.3.0
