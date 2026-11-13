"""Validate a FocusBuddy character asset (GLB / GLTF + metadata.json).

    python -m tools.validate_asset assets/characters/cat
    python -m tools.validate_asset path/to/model.glb --kind biped
    python -m tools.validate_asset --all            # every installed character

Reports mesh, skeleton, required bones, skin weights, materials, textures (and missing ones),
animations, morph targets, scale, ground alignment and orientation (see README -> "Character
assets"). Exit code 1 when any check fails, 0 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")        # Qt only decodes textures, no window

from focusbuddy.companion.character_asset import check, search_dirs  # noqa: E402


def _targets(args) -> list[Path]:
    if args.all:
        out = []
        for d in search_dirs():
            if Path(d).is_dir():
                out += sorted(p for p in Path(d).iterdir() if p.is_dir() and
                              ((p / "model.glb").is_file() or (p / "model.gltf").is_file()))
        return out
    return [Path(p) for p in args.paths]


def _print(s: str) -> None:
    try:
        print(s)
    except UnicodeEncodeError:                                   # legacy Windows consoles
        print(s.replace("✓", "OK").replace("✗", "X"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", help="character folder or model file")
    p.add_argument("--kind", choices=("biped", "quadruped"), help="override metadata.json 'kind'")
    p.add_argument("--all", action="store_true", help="validate every installed character")
    p.add_argument("--quiet", action="store_true", help="only the verdict and problems")
    p.add_argument("--issues", action="store_true", help="also print the raw issue list")
    args = p.parse_args(argv)
    targets = _targets(args)
    if not targets:
        p.error("give a character folder / model file, or --all")
    try:
        from PySide6.QtGui import QGuiApplication
        _app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])   # noqa: F841 (texture decoding)
    except Exception:  # noqa: BLE001
        pass
    failed = 0
    for t in targets:
        checks, issues = check(t, args.kind)
        bad = [c for c in checks if c.status == "fail"] or [i for i in issues if i.level == "error"]
        failed += bool(bad)
        _print(f"{t}")
        for c in checks:
            if args.quiet and c.status in ("ok", "info"):
                continue
            _print(f"  {c}")
        if args.issues:
            for i in issues:
                _print(f"    {i}")
        warn = [c for c in checks if c.status == "warn"]
        _print("  Character is NOT valid." if bad else
               ("  Character is valid." + (f" ({len(warn)} warning(s))" if warn else "")))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
