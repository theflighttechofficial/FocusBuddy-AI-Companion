"""``FocusBuddy --self-test``: a packaging / installation check that needs no desktop session.

For every character in the selector it loads the GLB asset (or notes the procedural fallback),
validates it, and renders a few frames with the production renderer (GPU skinning when an
OpenGL context exists, CPU otherwise) and with the 2D fallback. The report is printed and, with
``--self-test-report FILE``, written as JSON. Exit code 0 = everything loaded and rendered.

It is what the Windows CI job runs against the built ``FocusBuddy.exe`` (fresh runner, no
development environment on PATH for the exe itself). It does not prove anything about real GPU
hardware - CI machines render with software OpenGL or the CPU path.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback


def run(report_path: str | None = None) -> int:
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter

    from .companion.character_asset import asset_dir, check
    from .companion.characters import all_characters
    from .companion.renderer import make_renderer
    from .config import ASSETS_DIR
    from .ui.playground import PlaygroundSim

    out = {"python": sys.version.split()[0], "platform": platform.platform(), "frozen": bool(getattr(sys, "frozen", False)),
           "assets_dir": str(ASSETS_DIR), "characters": [], "ok": True}
    try:
        from .companion.render3d_gl import GLSkinRaster
        gl = GLSkinRaster.create()
        out["opengl"] = getattr(gl, "renderer_name", "") if gl is not None else "unavailable (CPU path)"
        out["gpu_skinning_max_bones"] = getattr(gl, "max_bones", 0) if gl is not None else 0
    except Exception as e:  # noqa: BLE001
        out["opengl"] = f"error: {e}"
    for spec in all_characters():
        rec = {"id": spec.id, "name": spec.display_name}
        try:
            d = asset_dir(spec.id)
            rec["asset"] = str(d) if d is not None else None
            if d is not None:
                checks, issues = check(d)
                rec["valid"] = not any(c.status == "fail" for c in checks)
                rec["warnings"] = [str(c) for c in checks if c.status == "warn"]
            sim = PlaygroundSim(spec.id, char_px=160, mode="production")
            sim.set_effects(False)
            img = QImage(200, 200, QImage.Format_ARGB32)
            t0 = time.perf_counter()
            for _ in range(6):
                sim.step(1 / 30)
                img.fill(QColor(0, 0, 0, 0))
                g = QPainter(img)
                sim.renderer.paint(g, QRectF(0, 0, 200, 200), sim.pose, 1)
                g.end()
            rec["render_ms"] = round((time.perf_counter() - t0) * 1000 / 6, 1)
            rec["backend"] = getattr(sim.renderer, "backend", "2d")
            rec["mode"] = getattr(sim.renderer, "mode", "2d")
            bits = img.constBits()
            alpha = bytes(bits)[3::4]
            rec["pixels"] = sum(1 for a in alpha if a > 0)
            r2 = make_renderer(spec, "2d")
            g = QPainter(img)
            r2.paint(g, QRectF(0, 0, 200, 200), sim.pose, 1)
            g.end()
            from .companion.illustrated import has_pack
            if has_pack(spec.id):
                ir = make_renderer(spec, "illustrated")
                img.fill(QColor(0, 0, 0, 0))
                g = QPainter(img)
                ir.paint(g, QRectF(0, 0, 200, 200), sim.pose, 1, "IDLE")
                g.end()
                rec["illustrated_frames"] = len(ir.pack.frames)
                rec["illustrated_pixels"] = sum(1 for a in bytes(img.constBits())[3::4] if a > 0)
            rec["ok"] = rec["pixels"] > 200 and rec.get("valid", True) and \
                rec.get("illustrated_pixels", 1000) > 200
        except Exception:  # noqa: BLE001
            rec["ok"] = False
            rec["error"] = traceback.format_exc(limit=4)
        out["ok"] = out["ok"] and rec["ok"]
        out["characters"].append(rec)
    try:
        from .companion.render3d_gl import GLSkinRaster
        if GLSkinRaster._shared is not None:
            GLSkinRaster._shared.release()
    except Exception:  # noqa: BLE001
        pass
    text = json.dumps(out, indent=2)
    print(text)
    if report_path:
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0 if out["ok"] else 1
