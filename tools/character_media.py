"""Media generated from the actual runtime renderer (deterministic, headless). Nothing is drawn by
hand: every image is the production renderer painting the real companion stack (behaviour,
animation, rig, skinning, shading).

    python tools/character_media.py character-sheet docs/character_sheet.png [ids...]
    python tools/character_media.py body            docs/body_language.png    # 12 emotions, effects OFF
    python tools/character_media.py emotions        docs/emotion_sheet.png [ids...]   # all 22 emotions
    python tools/character_media.py poses           docs/pose_library.png
    python tools/character_media.py lineup          docs/characters_3d.png    # views and LODs
    python tools/character_media.py showcase        OUT_DIR [ids...]  # --character-showcase frames
    python tools/character_media.py emotion-showcase OUT_DIR [ids...] # --emotion-showcase frames
    python tools/character_media.py actions         OUT_DIR [ids...]  # walk ... celebrate frames
    python tools/character_media.py gaits           OUT_DIR [ids...]  # walk / jog / run / sprint grid
    python tools/character_media.py encode          OUT_DIR out.gif|out.mp4 [fps]

The eight built-in companions are 3D reconstructions of their character sheets (TripoSR +
tools/build_3d_companions.py). ``fox_sample`` and ``robot_expressive`` are the reference GLBs.
Captions say which.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FOCUSBUDDY_FAKE_HOUR", "14")

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

DEMO = ["cat", "dog", "fox", "rabbit", "robot", "buddy"]
ARTIST = ["fox_sample", "robot_expressive"]
CHARS = DEMO
BG = "#EEF1F6"
INK = "#3A4150"
BODY12 = ["NEUTRAL", "HAPPY", "SAD", "ANGRY", "SCARED", "SURPRISED", "EXCITED", "SLEEPY", "BORED", "CONFIDENT",
          "CURIOUS", "LOVE"]


def _font(g: QPainter, px: int, bold: bool = False) -> None:
    f = QFont()
    f.setPixelSize(px)
    f.setBold(bold)
    g.setFont(f)


def _label(cid: str) -> str:
    from focusbuddy.companion.characters import asset_label, get_character
    return asset_label(get_character(cid))


def _sim(cid: str, px: float, quality: str = "high"):
    from focusbuddy.ui.playground import PlaygroundSim
    sim = PlaygroundSim(cid, char_px=px, mode="production")
    sim.set_quality(quality)
    sim.set_effects(False)
    sim.renderer.reset_orbit()
    return sim


def _settle(sim, secs: float, paint_every: int = 0, g=None, rect=None) -> None:
    for i in range(int(secs * 60)):
        sim.step(1 / 60)
        if paint_every and g is not None and i % paint_every == 0:
            sim.renderer.paint(g, rect, sim.pose, 1)        # clips advance with painted frames


def _scratch():
    img = QImage(64, 64, QImage.Format_ARGB32)
    return img, QPainter(img)


def _sheet(cols: list[str], cells, out: str, title: str, rows: list[str], cell: int = 190,
           subtitle: str = "Production renderer (skinned GLB), no icons / particles / text.") -> None:
    head, left = 64, 150
    img = QImage(left + cell * len(cols), head + cell * len(rows), QImage.Format_ARGB32)
    img.fill(QColor(BG))
    g = QPainter(img)
    g.setRenderHint(QPainter.Antialiasing)
    g.setPen(QColor(INK))
    _font(g, 17, True)
    g.drawText(QRectF(12, 6, img.width() - 24, 26), Qt.AlignLeft | Qt.AlignVCenter, title)
    _font(g, 12)
    g.drawText(QRectF(12, 30, img.width() - 24, 18), Qt.AlignLeft | Qt.AlignVCenter, subtitle)
    _font(g, 13, True)
    for i, c in enumerate(cols):
        g.drawText(QRectF(left + i * cell, head - 18, cell, 18), Qt.AlignCenter, c.replace("_", " ").lower())
    for r, cid in enumerate(rows):
        g.setPen(QColor(INK))
        _font(g, 13, True)
        g.drawText(QRectF(6, head + r * cell + cell / 2 - 20, left - 12, 20), Qt.AlignRight | Qt.AlignVCenter, cid)
        _font(g, 10)
        g.setPen(QColor("#6A7080"))
        g.drawText(QRectF(6, head + r * cell + cell / 2, left - 12, 34), Qt.AlignRight | Qt.TextWordWrap,
                   _label(cid))
        for i, c in enumerate(cols):
            cells(g, cid, c, QRectF(left + i * cell, head + r * cell, cell, cell))
        print(cid, flush=True)
    g.end()
    img.save(out)


def _emotion_cells(names):
    sims = {}

    def cell(g, cid, emo, rect):
        sim = sims.get(cid) or sims.setdefault(cid, _sim(cid, rect.width() * 0.8))
        sim.set_emotion(emo)
        simg, sg = _scratch()
        _settle(sim, 2.4, 6, sg, QRectF(0, 0, 64, 64))
        sg.end()
        sim.renderer.paint(g, rect, sim.pose, 1)
    return cell


def body(out: str, ids=None) -> None:
    _sheet(BODY12, _emotion_cells(BODY12), out, "Body-language test: 12 emotions with every effect switched off",
           ids or ARTIST + DEMO, cell=150)


def emotions(out: str, ids=None) -> None:
    from focusbuddy.ui.showcase import EMOTION_SHOWCASE
    _sheet(EMOTION_SHOWCASE, _emotion_cells(EMOTION_SHOWCASE), out, "Emotion showcase: 22 emotions (body + face)",
           ids or ARTIST + ["cat", "buddy"], cell=130)


def poses(out: str, ids=None) -> None:
    names = ["neutral_stand", "happy_stand", "angry_stand", "sad_stand", "scared_stand", "sleepy_stand",
             "confident_stand", "curious_stand", "bored_stand", "excited_stand", "surprised_stand", "proud_stand"]
    sims = {}

    def cell(g, cid, name, rect):
        sim = sims.get(cid) or sims.setdefault(cid, _sim(cid, rect.width() * 0.8))
        sim.set_emotion("NEUTRAL")
        sim.set_pose(name)
        simg, sg = _scratch()
        _settle(sim, 1.8, 6, sg, QRectF(0, 0, 64, 64))
        sg.end()
        sim.renderer.paint(g, rect, sim.pose, 1)
    _sheet(names, cell, out, "Pose library: 12 named full-body stands (skeleton, neutral face)", ids or ARTIST + DEMO,
           cell=150)


def character_sheet(out: str, ids=None) -> None:
    """Per character: front / 3/4 / side / back / close face / full body walking + 5 emotions."""
    cols = ["front", "3/4", "side", "back", "close face", "walking", "neutral", "happy", "angry", "sad", "scared"]

    def cell(g, cid, col, rect):
        sim = _sim(cid, rect.width() * 0.8)
        emo = col.upper() if col in ("neutral", "happy", "angry", "sad", "scared") else "NEUTRAL"
        sim.set_emotion(emo)
        if col == "walking":
            sim.b._start_plan([("goto_rel", 900, "walk")], mode="directed")
        simg, sg = _scratch()
        _settle(sim, 1.6 if col != "walking" else 1.2, 6, sg, QRectF(0, 0, 64, 64))
        sg.end()
        preset = {"front": "Front", "3/4": "3/4", "side": "Side", "back": "Back", "close face": "3/4"}.get(col)
        if preset:
            sim.orbit_preset(preset)
        if col in ("neutral", "happy", "angry", "sad", "scared"):
            sim.orbit_preset("3/4")
        g.save()
        g.setClipRect(rect)
        if col == "close face":
            # a bigger render placed so the head sits in the middle of the cell
            from focusbuddy.companion.renderer import GROUND_MARGIN, UNITS
            big = rect.width() * 2.4
            simg2, sg2 = _scratch()
            sim.renderer.paint(sg2, QRectF(0, 0, big, big), sim.pose, 1)
            sg2.end()
            u, v = getattr(sim.renderer, "head_pos", (0.0, -60.0))
            s2 = big / UNITS
            x = rect.center().x() - big / 2 - u * s2
            y = rect.center().y() + rect.height() * 0.06 - big + GROUND_MARGIN * s2 - v * s2
            sim.renderer.paint(g, QRectF(x, y, big, big), sim.pose, 1)
        else:
            sim.renderer.paint(g, rect, sim.pose, 1)
        g.restore()
    _sheet(cols, cell, out, "Character sheet: views, close-up, walking and core emotions (runtime renderer)",
           ids or ARTIST + DEMO, cell=150)


def lineup(out: str, ids=None) -> None:
    cols = ["side", "3/4", "front", "back", "low LOD", "high LOD"]

    def cell(g, cid, col, rect):
        q = {"low LOD": "low"}.get(col, "high")
        sim = _sim(cid, rect.width() * 0.8, q)
        if col in ("side", "3/4", "front", "back"):
            sim.orbit_preset({"side": "Side", "3/4": "3/4", "front": "Front", "back": "Back"}[col])
        _settle(sim, 1.0)
        sim.renderer.paint(g, rect, sim.pose, 1)
        st = sim.renderer.stats
        g.setPen(QColor("#7A8090"))
        _font(g, 10)
        g.drawText(QRectF(rect.x(), rect.y() + 4, rect.width(), 14), Qt.AlignCenter,
                   f"{st.get('triangles', 0)} tris  {st.get('total_ms', 0):.0f} ms  {st.get('backend', '')}")
        g.setPen(QColor(INK))
        _font(g, 13, True)
    _sheet(cols, cell, out, "GLB characters - views and LODs", ids or ARTIST + DEMO)


# ------------------------------------------------------------------ animated media
def _record(sim, steps, out_dir: str, n0: int, W=720, H=440, every=5, limit=150.0, caption_prefix="",
            done=lambda: False):
    from focusbuddy.companion.actions import Priority
    from focusbuddy.ui.playground import PlaygroundCanvas
    canvas = PlaygroundCanvas(sim)
    canvas.show_hint = False
    canvas.resize(W, H)
    state = {"label": ""}
    finished = {"v": False}

    def mark(lbl):
        state["label"] = lbl
        canvas.caption = f"{caption_prefix}{lbl.lower()}"
    plan = steps(sim.b, mark) + [("call", lambda: finished.__setitem__("v", True))]
    sim.b._start_plan(plan, mode="directed", priority=Priority.SPECIAL_ACTION)
    n, k = n0, 0
    while not finished["v"] and k < int(limit * 60):
        sim.step(1 / 60)
        canvas.cam_x += (sim.b.x - canvas.cam_x) * min(1.0, 4 / 60)
        if k % every == 0:
            canvas.grab().toImage().save(os.path.join(out_dir, f"f_{n:04d}.png"))
            n += 1
        k += 1
    return n


def showcase(out_dir: str, chars: list[str]) -> None:
    from focusbuddy.ui.showcase import CharacterShowcase
    os.makedirs(out_dir, exist_ok=True)
    W, H, n = 720, 560, 0
    for cid in chars:
        w = CharacterShowcase(cid, standalone=False)
        w.cycle.setChecked(False)
        w.canvas.setParent(None)                 # record the stage alone at a fixed size
        w.canvas.resize(W, H)
        k = 0
        while not w.done and k < 60 * 120:
            w.step(1 / 60)
            if k % 5 == 0:                       # 12 fps
                w.canvas.grab().toImage().save(os.path.join(out_dir, f"f_{n:04d}.png"))
                n += 1
            k += 1
        print(cid, n, flush=True)


def emotion_showcase(out_dir: str, chars: list[str]) -> None:
    from focusbuddy.ui.showcase import CharacterShowcase
    os.makedirs(out_dir, exist_ok=True)
    W, H, n = 720, 560, 0
    for cid in chars:
        w = CharacterShowcase(cid, standalone=False, emotions=True)
        w.cycle.setChecked(False)
        w.canvas.setParent(None)
        w.canvas.resize(W, H)
        k = 0
        while not w.done and k < 60 * 90:
            w.step(1 / 60)
            if k % 5 == 0:
                w.canvas.grab().toImage().save(os.path.join(out_dir, f"f_{n:04d}.png"))
                n += 1
            k += 1
        print(cid, n, flush=True)


def actions(out_dir: str, chars: list[str]) -> None:
    from focusbuddy.companion.showcase import action_showcase_steps
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for cid in chars:
        sim = _sim(cid, 260.0, "medium")
        n = _record(sim, action_showcase_steps, out_dir, n, caption_prefix=f"{cid}: ")
        print(cid, n, flush=True)


def gaits(out_dir: str, chars: list[str]) -> None:
    """2 x N grid of characters doing walk -> jog -> run -> sprint -> stop at the same time."""
    from focusbuddy.companion.actions import Priority
    from focusbuddy.companion.showcase import gait_showcase_steps
    os.makedirs(out_dir, exist_ok=True)
    cols = 2
    rows = (len(chars) + 1) // 2
    TW, TH = 420, 220
    sims, labels = [], []
    for cid in chars:
        sim = _sim(cid, 150.0, "medium")
        lab = {"v": ""}
        labels.append(lab)
        sim.b._start_plan(gait_showcase_steps(sim.b, lambda l, lab=lab: lab.__setitem__("v", l)), mode="directed",
                          priority=Priority.SPECIAL_ACTION)
        sims.append(sim)
    cam = [s.b.x for s in sims]
    for k in range(60 * 14):
        for i, s in enumerate(sims):
            s.step(1 / 60)
            cam[i] += (s.b.x - cam[i]) * (4 / 60)
        if k % 5:
            continue
        img = QImage(TW * cols, TH * rows, QImage.Format_ARGB32)
        img.fill(QColor(BG))
        g = QPainter(img)
        g.setRenderHint(QPainter.Antialiasing)
        for i, s in enumerate(sims):
            ox, oy = (i % cols) * TW, (i // cols) * TH
            g.setPen(QColor("#C9D0E0"))
            g.drawLine(ox, oy + TH - 30, ox + TW, oy + TH - 30)
            x = ox + TW / 2 + (s.b.x - cam[i])
            r = QRectF(x - 75, oy + TH - 30 - 150 + 150 * 18 / 140, 150, 150)
            s.renderer.paint(g, r, s.pose, 1)
            g.setPen(QColor(INK))
            _font(g, 12, True)
            g.drawText(QRectF(ox + 8, oy + 6, TW - 16, 16), Qt.AlignLeft, f"{chars[i]}: {labels[i]['v'].lower()}")
        g.end()
        img.save(os.path.join(out_dir, f"f_{k // 5:04d}.png"))


def encode(frames: str, out: str, fps: float = 12.0) -> None:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise SystemExit("ffmpeg not found")
    src = os.path.join(frames, "f_%04d.png")
    if out.endswith(".gif"):
        subprocess.run([ff, "-loglevel", "error", "-y", "-framerate", str(fps), "-i", src, "-vf",
                        "fps=10,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];"
                        "[b][p]paletteuse=dither=bayer:bayer_scale=4", out], check=True)
    else:
        subprocess.run([ff, "-loglevel", "error", "-y", "-framerate", str(fps), "-i", src, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "24", out], check=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    what, rest = sys.argv[1], sys.argv[2:]
    if what == "encode":
        encode(rest[0], rest[1], float(rest[2]) if len(rest) > 2 else 12.0)
    elif what in ("showcase", "emotion-showcase", "actions", "gaits"):
        fn = {"showcase": showcase, "emotion-showcase": emotion_showcase, "actions": actions, "gaits": gaits}[what]
        fn(rest[0], rest[1:] or (["fox_sample"] if what != "gaits" else ARTIST + ["cat", "dog"]))
    else:
        fn = {"body": body, "emotions": emotions, "poses": poses, "lineup": lineup,
              "character-sheet": character_sheet}[what]
        fn(rest[0], rest[1:] or None)
