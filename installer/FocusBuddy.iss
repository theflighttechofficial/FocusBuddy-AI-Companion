; Inno Setup script - builds FocusBuddy-Setup-<version>.exe from dist\FocusBuddy (PyInstaller output).
; Build with build_installer.bat (or: ISCC.exe /DAppVersion=2.3.0 installer\FocusBuddy.iss).
;
; * per-user install, no administrator rights needed: %LOCALAPPDATA%\Programs\FocusBuddy
; * Start menu entry, optional desktop icon, optional "start with Windows"
; * listed in Settings > Apps > Installed apps with a normal uninstaller
; * uninstalling keeps your profiles and history (%LOCALAPPDATA%\FocusBuddy) unless you choose to remove them

#ifndef AppVersion
  #define AppVersion "2.3.0"
#endif

[Setup]
AppId={{7F3C2A61-4B8E-4D52-9C1A-6E0F2B9D8A41}
AppName=FocusBuddy
AppVersion={#AppVersion}
AppVerName=FocusBuddy {#AppVersion}
AppPublisher=The Flight Tech Labs
DefaultDirName={autopf}\FocusBuddy
DefaultGroupName=FocusBuddy
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=Output
OutputBaseFilename=FocusBuddy-Setup-{#AppVersion}
SetupIconFile=..\assets\icons\focusbuddy.ico
UninstallDisplayIcon={app}\FocusBuddy.exe
UninstallDisplayName=FocusBuddy
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
CloseApplications=force
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "startup"; Description: "Start FocusBuddy when I sign in to Windows"; GroupDescription: "Start-up:"

[Files]
Source: "..\dist\FocusBuddy\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\FocusBuddy"; Filename: "{app}\FocusBuddy.exe"
Name: "{group}\Uninstall FocusBuddy"; Filename: "{uninstallexe}"
Name: "{autodesktop}\FocusBuddy"; Filename: "{app}\FocusBuddy.exe"; Tasks: desktopicon

[Registry]
; same value the app's own Settings > General > "Start with Windows" switch uses
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "FocusBuddy"; \
  ValueData: """{app}\FocusBuddy.exe"" --minimized"; Tasks: startup; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueName: "FocusBuddy"; \
  Flags: uninsdeletevalue dontcreatekey

[Run]
Filename: "{app}\FocusBuddy.exe"; Description: "Launch FocusBuddy"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{cmd}"; Parameters: "/C taskkill /IM FocusBuddy.exe /F"; Flags: runhidden; RunOnceId: "StopFocusBuddy"

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\FocusBuddy');
    if DirExists(DataDir) then
      if MsgBox('Also delete your FocusBuddy profiles, focus history and settings?' + #13#10 + #13#10 +
                'Choose No to keep them for a later reinstall.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
