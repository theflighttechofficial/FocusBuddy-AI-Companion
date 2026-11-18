# Changelog

All notable changes to FocusBuddy. Full engineering notes for each release are in
[docs/TECHNICAL.md](docs/TECHNICAL.md).

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
