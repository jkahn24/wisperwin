"""WisperWin audio helper — owns ALL WASAPI/COM endpoint work, disposably.

Why this process exists: wisperwin.py crashed three times (2026-08-03, -06, -10)
with an ACCESS VIOLATION in _ctypes.pyd — WASAPI/COM endpoint interfaces being
finalised across threads/COM apartments while the WH-1000XM6 reinitialised its
endpoints. The 2026-08-06 in-process lock demonstrably did not close the hole.
This process does: it is SINGLE-THREADED, so the cross-apartment collision
cannot occur — and if the COM stack dies anyway, it takes a ~15 MB helper with
it, not a dictation. The parent detects the death and respawns.

Protocol: JSON per line on stdin -> one JSON reply per line on stdout.
  {"id": 1, "op": "ping"}                      -> {"id": 1, "ok": true}
  {"id": 2, "op": "prewarm"}                   -> {"id": 2, "ok": true, "names": 3}
  {"id": 3, "op": "mute", "value": true}       -> {"id": 3, "ok": true, "changed": [[name, before, after], ...]}
  {"id": 4, "op": "input_volume", "value": 0.8} -> {"id": 4, "ok": true, "before": 0.79, "after": 0.8}
  {"id": 5, "op": "_test_sleep", "secs": 15}   -> (test hook for the parent's timeout-kill path)
Op failures reply {"ok": false, "error": ...} and the helper stays alive; only
EOF on stdin (parent gone) or a native COM crash ends it.

The mute sweep and volume code are ports of the 2026-08-05 lean-path versions
from wisperwin.py (IMMDeviceEnumerator directly, pycaw's GetAllDevices only for
one-time FriendlyName resolution — 104x measured difference). muted.flag stays
in the PARENT: it must survive this process's death, which is the whole point.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

LOG = Path(os.environ.get("TEMP", ".")) / "wisperwin_audio_helper.log"


def _log(msg: str):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except OSError:
        pass


def main():
    try:
        LOG.write_text(f"=== audio_helper start pid={os.getpid()} ===\n",
                       encoding="utf-8")
    except OSError:
        pass

    import comtypes
    try:
        comtypes.CoInitialize()
    except Exception:
        pass
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL, CoCreateInstance
    from pycaw.constants import CLSID_MMDeviceEnumerator, EDataFlow, DEVICE_STATE
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, IMMDeviceEnumerator

    # Endpoint id -> FriendlyName. Property-store reads are the expensive part
    # of pycaw's GetAllDevices, so resolve at most once per endpoint.
    names: dict[str, str] = {}

    def endpoint_name(dev_id: str) -> str:
        if dev_id in names:
            return names[dev_id]
        devices = d = None
        try:
            devices = AudioUtilities.GetAllDevices(
                data_flow=EDataFlow.eRender.value,
                device_state=DEVICE_STATE.ACTIVE.value)
            for d in devices:
                try:
                    names[d.id] = d.FriendlyName
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            del d
            del devices
            gc.collect()
        return names.get(dev_id, dev_id)

    def op_mute(value: bool) -> dict:
        """Flip mute on ALL active render endpoints (lean IMMDeviceEnumerator
        sweep). Names resolved AFTER the loop, with every endpoint interface
        released — same ordering discipline as the parent used, kept because it
        is also what makes a single sweep fast."""
        changed: list[tuple[str, int, int]] = []
        errors: list[str] = []
        enumerator = collection = None
        try:
            enumerator = CoCreateInstance(CLSID_MMDeviceEnumerator,
                                          IMMDeviceEnumerator, CLSCTX_ALL)
            collection = enumerator.EnumAudioEndpoints(EDataFlow.eRender.value,
                                                       DEVICE_STATE.ACTIVE.value)
            count = collection.GetCount()
        except Exception as e:
            return {"ok": False, "error": f"enum: {e}"}
        for i in range(count):
            dev = iface = vol = None
            try:
                dev = collection.Item(i)
                iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                vol = cast(iface, POINTER(IAudioEndpointVolume))
                before = vol.GetMute()
                if before != value:
                    vol.SetMute(value, None)
                    changed.append((dev.GetId(), before, vol.GetMute()))
            except Exception as e:
                errors.append(f"endpoint {i}: {e}")
            finally:
                del vol
                del iface
                del dev
        del collection
        del enumerator
        if count:
            gc.collect()
        out = {"ok": True,
               "changed": [[endpoint_name(i), b, a] for i, b, a in changed]}
        if errors:
            out["error"] = "; ".join(errors)
        return out

    def op_input_volume(value: float) -> dict:
        """Raise the default mic capture volume to at least `value`. Never lowers."""
        mic = interface = vol = None
        try:
            mic = AudioUtilities.GetMicrophone()
            interface = mic.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            vol = cast(interface, POINTER(IAudioEndpointVolume))
            before = vol.GetMasterVolumeLevelScalar()
            if before < value:
                vol.SetMasterVolumeLevelScalar(value, None)
            after = vol.GetMasterVolumeLevelScalar()
            return {"ok": True, "before": round(before, 4), "after": round(after, 4)}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            del vol
            del interface
            del mic
            gc.collect()

    _log("ready")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            op = req.get("op")
            if op == "ping":
                resp = {"ok": True}
            elif op == "prewarm":
                endpoint_name("")     # one slow sweep fills the whole cache
                resp = {"ok": True, "names": len(names)}
            elif op == "mute":
                resp = op_mute(bool(req["value"]))
            elif op == "input_volume":
                resp = op_input_volume(float(req["value"]))
            elif op == "_test_sleep":
                time.sleep(float(req["secs"]))
                resp = {"ok": True}
            else:
                resp = {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as e:
            resp = {"ok": False, "error": f"dispatch: {e}"}
        resp["id"] = rid
        try:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except OSError:
            break     # parent gone
    _log("stdin closed — exiting")
    try:
        comtypes.CoUninitialize()
    except Exception:
        pass


if __name__ == "__main__":
    main()
