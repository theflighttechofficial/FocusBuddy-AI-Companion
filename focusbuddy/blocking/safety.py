"""Safety rules for every blocking action.

Principles
  * Nothing is closed silently: automatic actions need automatic blocking
    enabled *and* the specific rule marked auto-close, after a countdown.
  * System / security software is never touched (not even minimized).
  * Terminals are protected unless the user explicitly allows it.
  * Apps that may hold unsaved work are minimized instead of closed when
    acting automatically.  Closing is always a polite WM_CLOSE so the app
    can still ask to save.  FocusBuddy never force-kills a process.
  * The target window is re-verified right before acting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.activity import Activity
from ..core.rules import Rule

PROTECTED_EXES = {
    # shell & system UI
    "explorer.exe", "dwm.exe", "csrss.exe", "winlogon.exe", "wininit.exe", "services.exe", "lsass.exe",
    "smss.exe", "svchost.exe", "sihost.exe", "taskhostw.exe", "ctfmon.exe", "fontdrvhost.exe",
    "shellexperiencehost.exe", "startmenuexperiencehost.exe", "searchhost.exe", "searchapp.exe",
    "searchui.exe", "lockapp.exe", "logonui.exe", "textinputhost.exe", "applicationframehost.exe",
    "systemsettings.exe", "systemsettingsbroker.exe", "control.exe", "mmc.exe", "regedit.exe",
    "taskmgr.exe", "resmon.exe", "perfmon.exe", "msconfig.exe", "consent.exe", "useraccountcontrolsettings.exe",
    "msiexec.exe", "trustedinstaller.exe", "tiworker.exe", "wuauclt.exe", "usoclient.exe",
    "rundll32.exe", "dllhost.exe", "conhost.exe", "runtimebroker.exe", "securityhealthsystray.exe",
    "securityhealthservice.exe", "securityhealthhost.exe", "sechealthui.exe", "smartscreen.exe",
    # antivirus / security
    "msmpeng.exe", "mpcmdrun.exe", "nissrv.exe", "mbam.exe", "mbamtray.exe", "avastui.exe", "avgui.exe",
    "bdagent.exe", "nortonsecurity.exe", "ns.exe", "mcuicnt.exe", "mcshield.exe", "kavtray.exe", "avp.exe",
    "ekrn.exe", "egui.exe", "sophosui.exe", "csfalconservice.exe", "sentinelagent.exe",
    # password managers / auth
    "keepass.exe", "keepassxc.exe", "1password.exe", "bitwarden.exe",
    # ourselves
    "focusbuddy.exe",
}

TERMINAL_EXES = {"windowsterminal.exe", "wt.exe", "cmd.exe", "powershell.exe", "pwsh.exe", "bash.exe",
                 "wsl.exe", "mintty.exe", "alacritty.exe", "wezterm-gui.exe", "putty.exe"}

# Apps where closing could lose unsaved work: auto-actions minimize instead.
UNSAVED_RISK_EXES = {
    "winword.exe", "excel.exe", "powerpnt.exe", "onenote.exe", "notepad.exe", "notepad++.exe",
    "code.exe", "cursor.exe", "sublime_text.exe", "photoshop.exe", "illustrator.exe", "afterfx.exe",
    "premiere pro.exe", "blender.exe", "obs64.exe", "devenv.exe", "pycharm64.exe", "idea64.exe",
    "acad.exe", "matlab.exe", "rstudio.exe", "gimp-2.10.exe", "krita.exe", "audacity.exe",
}

ACTIONS = ("close_tab", "redirect_tab", "close_tab_keys", "minimize", "close_window")


@dataclass
class SafetyDecision:
    allowed: bool
    action: str
    reason: str = ""


class SafetyGuard:
    def __init__(self, settings, own_pid: int):
        self.settings = settings
        self.own_pid = own_pid

    def is_protected(self, exe: str) -> bool:
        exe = (exe or "").lower()
        if exe in PROTECTED_EXES:
            return True
        if exe in TERMINAL_EXES and not self.settings.get("allow_terminal_close", False):
            return True
        return False

    def check(self, activity: Activity, action: str, rule: Optional[Rule], automatic: bool,
              enforced: bool = False) -> SafetyDecision:
        """``enforced``: a focus session with "close distractions automatically" is running - the user
        consented when starting the session, so any distracting / blocked target may be closed after
        the countdown (still never protected apps, never whole browser windows, never unsaved work)."""
        if action not in ACTIONS:
            return SafetyDecision(False, action, "unknown action")
        if activity is None or activity.window is None or not activity.exe:
            return SafetyDecision(False, action, "target unknown")
        if activity.is_self or activity.window.pid == self.own_pid:
            return SafetyDecision(False, action, "cannot act on FocusBuddy itself")
        if self.is_protected(activity.exe):
            return SafetyDecision(False, action, "protected application")
        if rule is None:
            # Only things the user (or defaults) listed as distracting can be acted on.
            return SafetyDecision(False, action, "no rule for this target")
        if rule.category.value not in ("distracting", "blocked"):
            return SafetyDecision(False, action, "target is not marked distracting")
        if automatic and not enforced:
            if self.settings.get("blocking_mode") != "auto":
                return SafetyDecision(False, action, "automatic blocking disabled")
            if not rule.auto_close:
                return SafetyDecision(False, action, "rule not marked auto-close")
        if action in ("close_tab", "redirect_tab", "close_tab_keys") and not activity.is_browser:
            return SafetyDecision(False, action, "not a browser")
        if action == "close_window":
            if activity.is_browser:
                # Closing a whole browser window could close many unrelated tabs.
                return SafetyDecision(True, "minimize", "browser windows are minimized, not closed")
            if automatic and activity.exe in UNSAVED_RISK_EXES:
                return SafetyDecision(True, "minimize", "may contain unsaved work")
        return SafetyDecision(True, action)
