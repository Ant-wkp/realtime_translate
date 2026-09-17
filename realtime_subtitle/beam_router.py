"""Beam routing for XVF3800: read-only recon, safe apply, rollback restore.

The AUDIO_MGR_OP_ALL sequence describes the 6-channel output routing as index pairs.
Because pair order is not always inferable, recon() only observes, and
apply() refuses to write whenever the mapping cannot be determined
unambiguously - refusing to write is always safe, writing the wrong pair
is not. The original route is saved before any write for restore().

The device also exposes exactly two *programmable* fixed beams. Pointing them
at two known seat positions and letting FIXEDBEAMSGATING pick whichever
carries more speech energy is the hardware's own answer to crosstalk between
two talkers: while one beam is active the other is silenced, so the audio that
reaches the recognizer contains one voice instead of the room mix. That is
FixedBeamController below, which follows the same save/verify/rollback rule.
"""

from __future__ import annotations

import math

from .usb_device import ReSpeaker

AUDIO_CHANNELS = 6

RAD_PER_DEG = math.pi / 180.0
DEG_PER_RAD = 180.0 / math.pi

# everything beam-related, in the order a human reads it. Read-only: the probe
# never writes, so it is safe on a live device.
PROBE_PARAMS = [
    "VERSION",
    "AEC_NUM_MICS",
    "AEC_MIC_ARRAY_TYPE",
    "AEC_MIC_ARRAY_GEO",
    "AEC_AZIMUTH_VALUES",
    "AEC_SPENERGY_VALUES",
    "DOA_VALUE",
    "AUDIO_MGR_SELECTED_AZIMUTHS",
    "AUDIO_MGR_SELECTED_CHANNELS",
    "AEC_FIXEDBEAMSONOFF",
    "AEC_FIXEDBEAMSGATING",
    "AEC_FIXEDBEAMSAZIMUTH_VALUES",
    "AEC_FIXEDBEAMSELEVATION_VALUES",
    "AEC_FIXEDBEAMNOISETHR",
    "AEC_ASROUTONOFF",
    "AEC_ASROUTGAIN",
    "AUDIO_MGR_OP_ALL",
    "AUDIO_MGR_OP_PACKED",
    "AUDIO_MGR_OP_L",
    "AUDIO_MGR_OP_R",
    "AUDIO_MGR_OP_UPSAMPLE",
    "I2S_INPUT_PACKED",
]

# parameters the firmware reports in radians; shown in degrees instead
_RADIANS = {
    "AEC_AZIMUTH_VALUES",
    "AEC_FIXEDBEAMSAZIMUTH_VALUES",
    "AEC_FIXEDBEAMSELEVATION_VALUES",
    "AUDIO_MGR_SELECTED_AZIMUTHS",
}

# the two fixed beams are the only steerable ones; free-running and
# auto-select are derived by the firmware
FIXED_BEAM_COUNT = 2


class FixedBeamController:
    """Steer the two fixed beams and enable energy gating, with rollback.

    Every parameter is read before it is written, and restore() puts all of
    them back, so an experiment cannot leave the array in a state the operator
    did not choose. Nothing is written to flash: the change lasts until the
    device is reset, which is what makes it safe to try.
    """

    def __init__(self, device: ReSpeaker):
        self.device = device
        self._saved: dict[str, list] = {}

    # ------------------------------------------------------------------
    def probe(self) -> dict:
        """Read-only survey. Radians are converted to degrees for reading."""
        out: dict[str, object] = {}
        for name in PROBE_PARAMS:
            try:
                vals = list(self.device.read(name))
            except Exception as exc:  # noqa: BLE001 - a dead param must not
                out[name] = f"unavailable ({exc})"      # hide the rest
                continue
            if name in _RADIANS:
                out[name] = {"deg": [round(v * DEG_PER_RAD, 1) for v in vals],
                             "rad": [round(v, 4) for v in vals]}
            else:
                out[name] = vals
        return out

    # ------------------------------------------------------------------
    def _remember(self, name: str) -> None:
        """Snapshot a parameter the first time we are about to overwrite it."""
        if name not in self._saved:
            self._saved[name] = list(self.device.read(name))

    def steer(self, az1: float, az2: float,
              elev1: float = 0.0, elev2: float = 0.0,
              gating: bool = True) -> dict:
        """Point fixed beam 1 at `az1` and beam 2 at `az2` (degrees, 0 = front).

        Gating lets speech energy decide which beam is audible; without it both
        beams stay live and the two voices are summed, which is the problem we
        are trying to avoid, so gating defaults on.
        """
        for name in ("AEC_FIXEDBEAMSAZIMUTH_VALUES",
                     "AEC_FIXEDBEAMSELEVATION_VALUES",
                     "AEC_FIXEDBEAMSGATING",
                     "AEC_FIXEDBEAMSONOFF"):
            self._remember(name)

        want_az = [az1 * RAD_PER_DEG, az2 * RAD_PER_DEG]
        try:
            self.device.write("AEC_FIXEDBEAMSAZIMUTH_VALUES", want_az)
            self.device.write("AEC_FIXEDBEAMSELEVATION_VALUES",
                              [elev1 * RAD_PER_DEG, elev2 * RAD_PER_DEG])
            self.device.write("AEC_FIXEDBEAMSGATING", [1 if gating else 0])
            self.device.write("AEC_FIXEDBEAMSONOFF", [1])
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": str(exc), "restored": self.restore()}

        # verify: the servicer can accept a write and still not apply it
        try:
            back = list(self.device.read("AEC_FIXEDBEAMSAZIMUTH_VALUES"))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"verify read failed: {exc}",
                    "restored": self.restore()}
        if len(back) != FIXED_BEAM_COUNT or any(
                abs(a - b) > 1e-3 for a, b in zip(back, want_az)):
            return {"ok": False,
                    "reason": f"azimuth not applied: read {back}, wanted {want_az}",
                    "restored": self.restore()}
        return {"ok": True, "azimuth_deg": [az1, az2],
                "elevation_deg": [elev1, elev2], "gating": gating}

    def restore(self) -> bool:
        """Put back every parameter this controller overwrote."""
        ok = True
        for name, vals in self._saved.items():
            try:
                self.device.write(name, vals)
            except Exception:  # noqa: BLE001 - keep restoring the rest
                ok = False
        self._saved.clear()
        return ok


class BeamRouter:
    def __init__(self, device: ReSpeaker):
        self.device = device
        self._original: list[int] | None = None

    # ------------------------------------------------------------------
    def recon(self) -> dict:
        """Read-only probe of the current output routing."""
        op_all = self.device.read("AUDIO_MGR_OP_ALL")
        pairs = [tuple(op_all[i:i + 2]) for i in range(0, len(op_all) - 1, 2)]
        return {"op_all": op_all, "pairs": pairs, "channels": AUDIO_CHANNELS}

    def describe(self, op_all: list[int]) -> dict:
        """Best-effort channel mapping description; detects ambiguity."""
        if len(op_all) < 12:
            return {"ok": False, "reason": "AUDIO_MGR_OP_ALL too short"}
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
        info = self.describe(self.device.read("AUDIO_MGR_OP_ALL"))
        if not info.get("ok"):
            return {"ok": False, "reason": info.get("reason"), "changed": False}

        op_all = self.device.read("AUDIO_MGR_OP_ALL")
        self._original = list(op_all)

        if slot.upper() == "L":
            op_all[0:2] = list(candidate_pair)
        else:
            op_all[6:8] = list(candidate_pair)

        try:
            self.device.write("AUDIO_MGR_OP_ALL", op_all)
            verify = self.device.read("AUDIO_MGR_OP_ALL")
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
            self.device.write("AUDIO_MGR_OP_ALL", self._original)
            return True
        except Exception:  # noqa: BLE001
            return False
