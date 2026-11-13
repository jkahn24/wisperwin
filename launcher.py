"""
WisperWin Launcher
------------------
Registers the configured hotkey (config.yaml `hotkey:`, default Ctrl+Space)
system-wide and signals the main app via a named Windows event each time it
is pressed.

Run this at login (Windows Startup folder). It will launch wisperwin.py
automatically if it is not already running — both at startup and each
time the hotkey is pressed while the main app is closed.

Why a separate launcher?
  RegisterHotKey claims the hotkey at the OS level before any app
  (including IDEs) can intercept it. Running this as a lightweight
  always-on daemon keeps the hotkey registered even when the main
  overlay is busy processing, and re-launches it on demand if the
  user has closed WisperWin.
"""

import ctypes
import ctypes.wintypes
import subprocess
import sys
import threading
from pathlib import Path

from hotkey_util import parse_hotkey_safe

# ── Windows API constants ─────────────────────────────────────────────────────
MOD_NOREPEAT = 0x4000   # prevent repeated WM_HOTKEY while held
HOTKEY_ID    = 1
WM_HOTKEY    = 0x0312
SYNCHRONIZE  = 0x00100000


def _configured_hotkey() -> dict:
    """Read `hotkey:` from config.yaml — falls back to Ctrl+Space on any error
    (missing file, missing PyYAML, bad spec) so the launcher can never die
    just because the config is broken."""
    spec = None
    try:
        import yaml
        cfg_path = Path(__file__).parent / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                spec = (yaml.safe_load(f) or {}).get("hotkey")
    except Exception:
        pass
    return parse_hotkey_safe(spec)

EVENT_NAME        = "WisperWin_HotkeyEvent"
MUTEX_NAME        = "WisperWin_Launcher"
APP_MUTEX_NAME    = "WisperWin_SingleInstance"

# Seconds to wait before auto-launching at login — lets Windows settle and
# other tools start first before WisperWin loads its Whisper model.
STARTUP_DELAY_S = 60

user32   = ctypes.windll.user32
# use_last_error=True atomically captures GetLastError() into a ctypes-private
# slot before any intervening ctypes call can clobber it, so get_last_error()
# reliably returns 183 (ERROR_ALREADY_EXISTS) from CreateMutexW.
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)


def _get_pythonw() -> Path:
    """Prefer the tool's isolated venv Python; fall back to system Python."""
    venv = Path(__file__).parent / "venv" / "Scripts" / "pythonw.exe"
    if venv.exists():
        return venv
    system = Path(sys.executable).parent / "pythonw.exe"
    if system.exists():
        return system
    return Path(sys.executable)


def _app_is_running() -> bool:
    """Return True if wisperwin.py is currently running (checks its mutex)."""
    h = kernel32.OpenMutexW(SYNCHRONIZE, False, APP_MUTEX_NAME)
    if h:
        kernel32.CloseHandle(h)
        return True
    return False


_last_launch_t: float = 0.0          # monotonic time of the most recent launch
_LAUNCH_DEBOUNCE_S: float = 20.0     # ignore re-launch requests within this window


def _launch_main_app(extra_args: list[str] | None = None):
    """Start wisperwin.py in background.

    Guarded by a 20-second debounce: wisperwin.py takes ~10 s to import its
    heavy deps (ctranslate2, CUDA) before creating the singleton mutex.
    Without debounce, pressing Ctrl+Space during that window spawns a second
    instance before the first has registered itself as running.
    """
    global _last_launch_t
    import time as _time
    now = _time.monotonic()
    if now - _last_launch_t < _LAUNCH_DEBOUNCE_S:
        return
    _last_launch_t = now
    app_path = Path(__file__).parent / "wisperwin.py"
    cmd = [str(_get_pythonw()), str(app_path)] + (extra_args or [])
    subprocess.Popen(
        cmd,
        creationflags=0x00000008,        # DETACHED_PROCESS
        close_fds=True,
    )


def main():
    # Give this process a distinct identity so Windows doesn't group it with
    # other pythonw.exe processes (e.g. other tray tools) in taskbar / Start Menu routing.
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("WisperWin.App")

    # ── Single-instance guard ─────────────────────────────────────────────────
    # Clear first: a stale 183 from an earlier API call survives a SUCCESSFUL
    # CreateMutexW (success doesn't reset last-error), making a fresh launcher
    # exit as if a twin were running (2026-06-10, hit across all tools).
    ctypes.set_last_error(0)
    _mutex = kernel32.CreateMutexW(None, True, MUTEX_NAME)
    if ctypes.get_last_error() == 183:   # ERROR_ALREADY_EXISTS
        sys.exit(0)

    # ── Named event — wisperwin.py can watch this (optional IPC) ─────────────
    h_event = kernel32.CreateEventW(None, False, False, EVENT_NAME)
    if not h_event:
        sys.exit(1)

    # ── Launch main app at startup (staggered) ───────────────────────────────
    _startup_timer: threading.Timer | None = None
    if not _app_is_running():
        if STARTUP_DELAY_S > 0:
            _startup_timer = threading.Timer(STARTUP_DELAY_S, lambda: _app_is_running() or _launch_main_app())
            _startup_timer.daemon = True
            _startup_timer.start()
        else:
            _launch_main_app()

    # ── Register hotkey ───────────────────────────────────────────────────────
    hk = _configured_hotkey()
    if not user32.RegisterHotKey(None, HOTKEY_ID, hk["mod_flags"] | MOD_NOREPEAT, hk["vk"]):
        import time; time.sleep(1)
        if not user32.RegisterHotKey(None, HOTKEY_ID, hk["mod_flags"] | MOD_NOREPEAT, hk["vk"]):
            import os
            log = Path(os.environ["TEMP"]) / "wisperwin_launcher.log"
            log.write_text(f"RegisterHotKey failed — {hk['label']} may be claimed by another app.\n")
            sys.exit(1)

    # ── Message loop ──────────────────────────────────────────────────────────
    msg = ctypes.wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
        if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
            if _app_is_running():
                # App is running — signal it via the named event.
                # wisperwin.py handles the keypress itself via pynput,
                # so this is a belt-and-suspenders signal only.
                kernel32.SetEvent(h_event)
            else:
                # Hotkey pressed before delayed startup fired — cancel timer,
                # launch immediately with --hotkey so recording starts at once.
                if _startup_timer and _startup_timer.is_alive():
                    _startup_timer.cancel()
                _launch_main_app(extra_args=["--hotkey"])
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))

    user32.UnregisterHotKey(None, HOTKEY_ID)


if __name__ == "__main__":
    main()
