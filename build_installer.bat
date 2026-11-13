@echo off
REM Builds the Windows installer: installer\Output\FocusBuddy-Setup-<version>.exe
REM Needs Python 3.11+ (python.org or Microsoft Store). Inno Setup 6 is installed with winget if missing.
setlocal
cd /d "%~dp0"

echo [1/4] Python environment...
if not exist .venv ( py -3 -m venv .venv || python -m venv .venv || (echo Python 3 not found - install it from python.org & pause & exit /b 1) )
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
pip install -r requirements.txt -r requirements-dev.txt || (echo pip install failed & pause & exit /b 1)

if /i "%~1"=="test" (
    echo [2/4] Running tests - this takes several minutes...
    python -m pytest -q || (echo Tests failed & pause & exit /b 1)
) else (
    echo [2/4] Skipping tests ^(run "build_installer.bat test" to include them^)
)

echo [3/4] Building FocusBuddy.exe...
pyinstaller --noconfirm focusbuddy.spec || (echo PyInstaller failed & pause & exit /b 1)
start "" /wait dist\FocusBuddy\FocusBuddy.exe --self-test --self-test-report selftest.json
if errorlevel 1 (echo Self-test of the built app failed - see selftest.json & pause & exit /b 1)

echo [4/4] Building the installer...
for /f "tokens=2 delims==" %%v in ('findstr /b "__version__" focusbuddy\__init__.py') do set VER=%%~v
set VER=%VER: =%
set VER=%VER:"=%
set ISCC=
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set ISCC="%ProgramFiles%\Inno Setup 6\ISCC.exe"
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set ISCC="%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not defined ISCC (
    echo Inno Setup not found - installing it with winget...
    winget install --id JRSoftware.InnoSetup -e --accept-source-agreements --accept-package-agreements
    if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
    if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set ISCC="%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
)
if not defined ISCC (echo Install Inno Setup 6 from jrsoftware.org, then run this again. & pause & exit /b 1)
%ISCC% /DAppVersion=%VER% installer\FocusBuddy.iss || (echo Installer build failed & pause & exit /b 1)
echo.
echo Done: %CD%\installer\Output\FocusBuddy-Setup-%VER%.exe
start "" explorer "%CD%\installer\Output"
pause
endlocal
