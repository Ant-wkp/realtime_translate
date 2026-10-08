"""Frontend command router: JSON commands to config actions.

Commands arrive from the WebSocket cmd_queue and are executed by the main
loop (phase 5). Heavy work stays in the callers' own threads; this module
only translates and dispatches. Unknown languages and failed operations
return explicit errors instead of silent fallbacks.

The device-parameter commands (list/read/write) were removed along with the
settings drawer's "USB command" panel: nothing in this project writes
device parameters any more, which is also why the configuration stays
volatile. Only the two pieces of runtime configuration the UI actually
exposes remain - translation direction and DOA zone labels.
"""

from __future__ import annotations


class ControlHandler:
    def __init__(self, locator, translator):
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
