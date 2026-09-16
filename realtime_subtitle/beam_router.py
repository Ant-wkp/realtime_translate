"""Beam routing for XVF3800: read-only recon, safe apply, rollback restore.

The OP_ALL sequence describes the 6-channel output routing as index pairs.
Because pair order is not always inferable, recon() only observes, and
apply() refuses to write whenever the mapping cannot be determined
unambiguously - refusing to write is always safe, writing the wrong pair
is not. The original route is saved before any write for restore().
"""

from __future__ import annotations

from .usb_device import ReSpeaker

AUDIO_CHANNELS = 6


class BeamRouter:
    def __init__(self, device: ReSpeaker):
        self.device = device
        self._original: list[int] | None = None

    # ------------------------------------------------------------------
    def recon(self) -> dict:
        """Read-only probe of the current output routing."""
        op_all = self.device.read("OP_ALL")
        pairs = [tuple(op_all[i:i + 2]) for i in range(0, len(op_all) - 1, 2)]
        return {"op_all": op_all, "pairs": pairs, "channels": AUDIO_CHANNELS}

    def describe(self, op_all: list[int]) -> dict:
        """Best-effort channel mapping description; detects ambiguity."""
        if len(op_all) < 12:
            return {"ok": False, "reason": "OP_ALL too short"}
        l = tuple(op_all[0:2])
        r = tuple(op_all[6:8])
        pair_set = [tuple(op_all[i:i + 2]) for i in range(0, 12, 2)]
        if l == r or op_all.count(op_all[0]) > 2:
            return {"ok": False, "reason": "pair order ambiguous", "pairs": pair_set}
        return {
            "ok": True,
            "L": l,
            "R": r,
            "pairs": pair_set,
            "mic_candidates": pair_set[2:6],
        }

    # ------------------------------------------------------------------
    def apply(self, candidate_pair: tuple[int, int], slot: str = "R") -> dict:
        """Route one output slot to a candidate beam/mic pair, with rollback."""
        info = self.describe(self.device.read("OP_ALL"))
        if not info.get("ok"):
            return {"ok": False, "reason": info.get("reason"), "changed": False}

        op_all = self.device.read("OP_ALL")
        self._original = list(op_all)

        if slot.upper() == "L":
            op_all[0:2] = list(candidate_pair)
        else:
            op_all[6:8] = list(candidate_pair)

        try:
            self.device.write("OP_ALL", op_all)
            verify = self.device.read("OP_ALL")
            ok = list(verify) == list(op_all)
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            self.restore()
            return {"ok": False, "reason": "verify failed, restored", "changed": False}
        return {"ok": True, "changed": True, "slot": slot, "pair": candidate_pair}

    def restore(self) -> bool:
        """Roll back to the route saved before the last apply()."""
        if self._original is None:
            return False
        try:
            self.device.write("OP_ALL", self._original)
            return True
        except Exception:  # noqa: BLE001
            return False
