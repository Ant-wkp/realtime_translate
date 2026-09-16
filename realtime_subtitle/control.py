"""Frontend command router: JSON commands to device/config actions.

Commands arrive from the WebSocket cmd_queue and are executed by the main
loop (phase 5). Heavy work stays in the callers' own threads; this module
only translates and dispatches. Unknown languages and failed device
operations return explicit errors instead of silent fallbacks.
"""

from __future__ import annotations

from .usb_device import PARAMETERS


class ControlHandler:
    def __init__(self, device, locator, translator):
        self.device = device          # ReSpeaker or None
        self.locator = locator        # SpeakerLocator
        self.translator = translator  # NllbCTranslate2Provider

    def handle(self, msg: dict) -> dict:
        try:
            cmd = msg.get("cmd")
            payload = msg.get("payload", {})
            handler = getattr(self, f"cmd_{cmd}", None)
            if handler is None:
                return {"ok": False, "error": f"unknown command: {cmd}"}
            result = handler(payload)
            return {"ok": True, "data": result} if result is not None else {"ok": True}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    # device parameters ---------------------------------------------------
    def cmd_list_commands(self, _payload) -> dict:
        groups = {}
        for name, (code, ptype, ro, rw) in PARAMETERS.items():
            resid = name.split("_")[0].lower()
            groups.setdefault(resid, []).append(
                {"name": name, "code": code, "type": ptype,
                 "access": "rw" if rw else "ro"}
            )
        return {"groups": groups}

    def cmd_read(self, payload) -> dict:
        if self.device is None:
            raise RuntimeError("device not connected")
        name = payload.get("name", "")
        if name not in PARAMETERS:
            raise ValueError(f"unknown parameter: {name}")
        return {"name": name, "value": self.device.read(name)}

    def cmd_write(self, payload) -> dict:
        if self.device is None:
            raise RuntimeError("device not connected")
        name = payload.get("name", "")
        value = payload.get("value")
        if name not in PARAMETERS:
            raise ValueError(f"unknown parameter: {name}")
        _, _, _, rw = PARAMETERS[name]
        if not rw:
            raise ValueError(f"{name} is read-only")
        if value is None:
            raise ValueError("value required")
        self.device.write(name, value)
        return {"name": name, "value": value}

    # translation direction -------------------------------------------------
    def cmd_get_target_langs(self, _payload) -> dict:
        return {
            "zh_to": self.translator.zh_to,
            "other_to": self.translator.other_to,
            "options": self.translator.target_langs,
        }

    def cmd_set_target_langs(self, payload) -> dict:
        zh_to = payload.get("zh_to")
        other_to = payload.get("other_to")
        options = self.translator.target_langs
        if zh_to not in options or other_to not in options:
            raise ValueError("unsupported target language")
        self.translator.set_target_langs(zh_to, other_to)
        return {"zh_to": zh_to, "other_to": other_to}

    # DOA zones -----------------------------------------------------------------
    def cmd_get_zones(self, _payload) -> dict:
        return {"zones": self.locator.get_zones()}

    def cmd_set_zones(self, payload) -> dict:
        zones = payload.get("zones")
        if not isinstance(zones, list) or not zones:
            raise ValueError("zones must be a non-empty list")
        self.locator.set_zones([str(z) for z in zones])
        return {"zones": self.locator.get_zones()}
