@echo off
REM Builds dist\FocusBuddy\FocusBuddy.exe (no Python needed on the target PC).
REM   build_windows.bat         build only (a few minutes)
REM   build_windows.bat test    run the test suite first (adds ~5-10 minutes)
setlocal
cd /d "%~dp0"

echo [1/3] Preparing Python environment (first time takes a few minutes)...
if not exist .venv ( py -3 -m venv .venv || python -m venv .venv || goto :nopython )
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
pip install -r requirements.txt -r requirements-dev.txt || goto :failed

if /i "%~1"=="test" (
    echo [2/3] Running tests - this takes several minutes, please wait...
    python -m pytest -q || goto :failed
) else (
    echo [2/3] Skipping tests ^(run "build_windows.bat test" to include them^)
)

echo [3/3] Building FocusBuddy.exe with PyInstaller...
pyinstaller --noconfirm focusbuddy.spec || goto :failed

echo.
echo ============================================================
echo  Done. Your app is here:
echo  %CD%\dist\FocusBuddy\FocusBuddy.exe
echo  (keep the whole dist\FocusBuddy folder together)
echo ============================================================
start "" explorer "%CD%\dist\FocusBuddy"
pause
exit /b 0

:nopython
echo Python 3 was not found. Install Python 3.11-3.13 from python.org (tick "Add python.exe to PATH").
pause
exit /b 1

:failed
echo.
echo Build FAILED - scroll up to see the error, or send a screenshot of this window.
pause
exit /b 1
