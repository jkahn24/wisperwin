"""
Shared hotkey-spec parser for WisperWin.

Used by both launcher.py (RegisterHotKey — needs Win32 MOD_* flags + VK code)
and wisperwin.py (pynput watcher — needs modifier set + VK code).
Pure stdlib so the lightweight launcher can import it without the venv's
heavy dependencies.

Spec format: modifiers joined with '+', e.g. "ctrl+space", "ctrl+shift+space",
"alt+f9", "win+d". Case-insensitive. Supported keys: space, a–z, 0–9, f1–f24.
"""

# Win32 RegisterHotKey modifier flags
MOD_ALT     = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT   = 0x0004
MOD_WIN     = 0x0008

_MOD_ALIASES = {
    "ctrl": "ctrl", "control": "ctrl",
    "alt": "alt",
    "shift": "shift",
    "win": "win", "windows": "win", "super": "win", "meta": "win",
}
_MOD_FLAGS = {"ctrl": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT, "win": MOD_WIN}

# Virtual-key codes for the modifiers (GetAsyncKeyState checks in wisperwin.py)
MOD_VKS = {"ctrl": 0x11, "alt": 0x12, "shift": 0x10, "win": 0x5B}

DEFAULT_SPEC = "ctrl+space"


def _key_to_vk(key: str) -> int:
    key = key.lower()
    if key == "space":
        return 0x20
    if len(key) == 1 and (key.isascii() and (key.isalpha() or key.isdigit())):
        return ord(key.upper())
    if key.startswith("f") and key[1:].isdigit():
        n = int(key[1:])
        if 1 <= n <= 24:
            return 0x70 + n - 1   # VK_F1 = 0x70
    raise ValueError(f"unsupported hotkey key: {key!r}")


def parse_hotkey(spec: str) -> dict:
    """Parse a spec like 'ctrl+space' into everything both processes need.

    Returns: {"mods": set[str], "vk": int, "mod_flags": int, "label": str}
    Raises ValueError on anything unparseable — callers fall back to DEFAULT_SPEC.
    """
    parts = [p.strip().lower() for p in (spec or "").split("+") if p.strip()]
    if len(parts) < 1:
        raise ValueError(f"empty hotkey spec: {spec!r}")

    *mod_parts, key_part = parts
    mods = set()
    for m in mod_parts:
        if m not in _MOD_ALIASES:
            raise ValueError(f"unsupported hotkey modifier: {m!r}")
        mods.add(_MOD_ALIASES[m])
    if key_part in _MOD_ALIASES:
        raise ValueError(f"hotkey needs a non-modifier key, got: {spec!r}")

    vk        = _key_to_vk(key_part)
    mod_flags = sum(_MOD_FLAGS[m] for m in mods)
    label     = "+".join(
        [m.capitalize() for m in ("ctrl", "alt", "shift", "win") if m in mods]
        + [key_part.capitalize() if len(key_part) > 1 else key_part.upper()]
    )
    return {"mods": mods, "vk": vk, "mod_flags": mod_flags, "label": label}


def parse_hotkey_safe(spec: str) -> dict:
    """parse_hotkey with silent fallback to the default binding."""
    try:
        return parse_hotkey(spec)
    except (ValueError, TypeError):
        return parse_hotkey(DEFAULT_SPEC)
