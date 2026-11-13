"""
WisperWin Uninstaller
---------------------
Run directly:  python uninstall.py
Or double-click if .py files are associated with Python.

Removes:
  - Windows Startup shortcut
  - (Optional) Hugging Face model cache for the configured Whisper model
  - This script prints what it is doing at each step.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


APP_DIR    = Path(__file__).parent
CONFIG_YML = APP_DIR / "config.yaml"
STARTUP    = (
    Path(os.environ["APPDATA"])
    / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
)
LNK        = STARTUP / "WisperWin.lnk"
HF_CACHE   = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
FAST_CACHE = HF_CACHE / "hub"


def _kill_running():
    """Terminate any running WisperWin or launcher processes."""
    try:
        import psutil
        targets = {"wisperwin.py", "launcher.py"}
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmd = " ".join(proc.info["cmdline"] or []).lower()
                if any(t in cmd for t in targets):
                    proc.terminate()
                    print(f"  Terminated PID {proc.pid}")
            except Exception:
                pass
    except ImportError:
        # psutil not available — try taskkill by window title
        subprocess.run(
            ["taskkill", "/F", "/IM", "pythonw.exe"],
            capture_output=True,
        )


def _remove_startup_shortcut():
    if LNK.exists():
        LNK.unlink()
        print(f"  Removed startup shortcut: {LNK}")
    else:
        print(f"  Startup shortcut not found (already removed?): {LNK}")


def _remove_model_cache(model_name: str):
    """Delete the faster-whisper / Hugging Face cache for the given model."""
    # faster-whisper stores models under:
    #   ~/.cache/huggingface/hub/models--Systran--faster-whisper-<name>/
    #   ~/.cache/huggingface/hub/models--Systran--faster-distil-whisper-<name>/
    removed = False
    if FAST_CACHE.exists():
        for candidate in FAST_CACHE.iterdir():
            slug = candidate.name.lower()
            if model_name.lower().replace("-", "--") in slug or model_name.lower() in slug:
                shutil.rmtree(candidate, ignore_errors=True)
                print(f"  Removed model cache: {candidate}")
                removed = True
    if not removed:
        print(f"  Model cache not found for '{model_name}' (may have been cleaned already)")


def _confirm(msg: str) -> bool:
    try:
        ans = input(f"{msg} [y/N]: ").strip().lower()
        return ans in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def main():
    print("=" * 54)
    print("  WisperWin Uninstaller")
    print("=" * 54)
    print()

    if not _confirm("This will remove WisperWin from your system. Continue?"):
        print("Aborted.")
        sys.exit(0)

    print()
    print("Stopping running processes…")
    _kill_running()

    print()
    print("Removing startup shortcut…")
    _remove_startup_shortcut()

    # Model cache (optional — can be several GB)
    print()
    model_name = "distil-large-v3"
    if CONFIG_YML.exists():
        try:
            import yaml
            with open(CONFIG_YML, "r") as f:
                cfg = yaml.safe_load(f) or {}
            model_name = cfg.get("model", model_name)
        except Exception:
            pass

    if _confirm(f"Delete Whisper model cache (~1–3 GB) for '{model_name}'?"):
        print("Removing model cache…")
        _remove_model_cache(model_name)
    else:
        print("  Skipped model cache — files remain in ~/.cache/huggingface/")

    # App files (optional — user may want to keep config/logs)
    print()
    if _confirm("Delete the WisperWin app folder and all files?"):
        print(f"Removing app directory: {APP_DIR}")
        # Self-delete: schedule via a detached cmd that waits 2 s then deletes
        bat = Path(os.environ["TEMP"]) / "wisperwin_selfdelete.bat"
        bat.write_text(
            f'@echo off\n'
            f'ping -n 3 127.0.0.1 >nul\n'
            f'rmdir /s /q "{APP_DIR}"\n'
            f'del "%~f0"\n',
            encoding="utf-8",
        )
        subprocess.Popen(
            ["cmd", "/c", str(bat)],
            creationflags=0x00000008,  # DETACHED_PROCESS
        )
        print("  Scheduled — folder will be deleted in ~2 seconds.")
    else:
        print("  Skipped app folder deletion.")

    print()
    print("WisperWin uninstalled. Goodbye!")
    print()
    input("Press Enter to close…")


if __name__ == "__main__":
    main()
