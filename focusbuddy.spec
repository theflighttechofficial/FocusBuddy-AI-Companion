# PyInstaller spec - build with:  pyinstaller --noconfirm focusbuddy.spec
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ["main.py"],
    pathex=[],
    datas=[("assets", "assets"), ("browser_extension", "browser_extension")],
    hiddenimports=["PySide6.QtNetwork", "PySide6.QtOpenGL", "numpy"],
    excludes=["tkinter", "unittest", "pydoc", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtWebEngineCore"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FocusBuddy",
    console=False,
    icon="assets/icons/focusbuddy.ico",
    version=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="FocusBuddy")
