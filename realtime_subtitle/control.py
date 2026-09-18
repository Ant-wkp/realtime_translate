"""Frontend command router: JSON commands to device/config actions.

Commands arrive from the WebSocket cmd_queue and are executed by the main
loop (phase 5). Heavy work stays in the callers' own threads; this module
only translates and dispatches. Unknown languages and failed device
operations return explicit errors instead of silent fallbacks.

Configuration is volatile by policy: the commands that would persist it to
flash, or reset the device outright, are listed in usb_device and refused
here with an explanation rather than being passed down.
"""

from __future__ import annotations

from .usb_device import (DISRUPTIVE_CMDS, FLASH_PERSIST_CMDS, PARAMETERS,
                         RESID_GROUPS)


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
        for name, (resid, cmdid, count, access, ptype, desc) in PARAMETERS.items():
            gname = RESID_GROUPS.get(resid, f"resid_{resid}")
            entry = {"name": name, "resid": resid, "cmdid": cmdid,
                     "count": count, "type": ptype, "access": access,
                     "description": desc}
            # Tell the UI which rows are deliberately not writable, so it can
            # disable them instead of offering a control that would fail.
            if name in FLASH_PERSIST_CMDS:
                entry["blocked"] = True
                entry["block_reason"] = (
                    "会写入 flash 持久化配置，本项目禁止（持久化的波束开关可能"
                    "导致设备重启后挂起）"
                )
            elif name in DISRUPTIVE_CMDS:
                entry["blocked"] = True
                entry["block_reason"] = (
                    "会重启设备或重置全部参数，会丢弃本次运行建立的配置，本项目禁止"
                )
            groups.setdefault(gname, []).append(entry)
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
        if PARAMETERS[name][3] == "ro":
            raise ValueError(f"{name} is read-only")
        # The menu offers these, so a stray console call could reach them.
        # Rejecting here with an explanation is friendlier than letting
        # ReSpeaker.write() raise a PermissionError deeper down.
        if name in FLASH_PERSIST_CMDS:
            raise ValueError(
                f"{name} would save the configuration to flash and is blocked: "
                f"a persisted beam setting can hang the device on reboot. "
                f"Settings are applied per run instead."
            )
        if name in DISRUPTIVE_CMDS:
            raise ValueError(
                f"{name} reboots or resets the device and is blocked, because "
                f"it would drop the configuration this run applied."
            )
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
        """Accept either labelled ranges or bare labels.

        The range form is what the settings drawer sends; the bare-label form
        stays supported so an even split can still be requested by hand.
        """
        zones = payload.get("zones")
        if not isinstance(zones, list) or not zones:
            raise ValueError("zones must be a non-empty list")
        if all(isinstance(z, dict) for z in zones):
            self.locator.set_zones_from_dicts(zones)
        else:
            from .speaker_locator import zones_from_labels

            labels = [str(z) for z in zones if str(z).strip()]
            self.locator.set_zones(zones_from_labels(labels))
        return {"zones": self.locator.get_zones()}
