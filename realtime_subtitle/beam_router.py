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
import time

from .usb_device import GUARDED_CMDS, ReSpeaker

AUDIO_CHANNELS = 6

# Parameters steer() touches. Kept as a named tuple-ish list so the assertion
# below can check the whole write set at once: if someone later adds a
# flash-persisting or disruptive command to this sequence, the assert fires
# before anything reaches the device.
STEER_WRITES = (
    "AEC_FIXEDBEAMSAZIMUTH_VALUES",
    "AEC_FIXEDBEAMSELEVATION_VALUES",
    "AEC_FIXEDBEAMSGATING",
    "AEC_FIXEDBEAMSONOFF",          # must stay last: see steer()
)
assert not (set(STEER_WRITES) & GUARDED_CMDS), (
    "steer() must never write a flash-persisting or disruptive command"
)

RAD_PER_DEG = math.pi / 180.0
DEG_PER_RAD = 180.0 / math.pi

# Category numbers seen in this device's AUDIO_MGR_OP_ALL. They are the
# <category> half of each <category>, <source> pair. Read them as "which
# stage of the pipeline feeds this output". These are transcribed from what
# the device reports, not from a documented enumeration - XMOS publishes the
# pair format but not the category numbers - so TREAT THEM AS AN OBSERVATION,
# not as a specification:
#   8  the processed ("Conference") output - the device ships L = (8, 0)
#   7  the ASR beam path - the device ships R = (7, 3)
#   1  per-mic signals - the device puts (1, 0), (1, 2), (1, 1), (1, 3) in
#      the four secondary slots, and source 0..3 is exactly the four mics
#   4  far-end reference, raw          (from the XVF3800 documentation)
#   12 far-end reference, with gain/delay (from the XVF3800 documentation)
# The 4 and 12 values come from published documentation while 1, 7 and 8 come
# from this unit, which is why the probe reports rather than asserts.
CAT_CONFERENCE = 8
CAT_ASR_OUTPUT = 7
CAT_MIC_RAW = 1
CAT_FAR_RAW = 4
CAT_FAR_GAINED = 12

# For the ASR-output category the source is a beam index. The ordering is the
# one AEC_AZIMUTH_VALUES and AEC_SPENERGY_VALUES both use, and the device
# ships source 3, which is the auto-select beam in that same ordering - so the
# reading is corroborated rather than assumed.
BEAM_SOURCES = {0: "beam1", 1: "beam2", 2: "free-running", 3: "auto-select"}

# Where OP_L and OP_R land in the 6-channel capture stream. The firmware
# documentation states the two programmable outputs feed the first two
# channels and the remaining four are fixed per-mic channels.
SLOT_CHANNEL = {"L": 0, "R": 1}

# The stream is 6 channels regardless of routing: OP_L/OP_R choose what the
# first two carry, they do not change the channel count.

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

        AEC_FIXEDBEAMSONOFF is written last and deliberately *not* followed by
        a save: the firmware only supports enabling fixed beams at runtime, and
        persisting the flag has been reported to hang the device. The setting
        therefore lasts until the device is reset, and restore() reverses it.
        """
        for name in STEER_WRITES:
            self._remember(name)

        want_az = [az1 * RAD_PER_DEG, az2 * RAD_PER_DEG]
        try:
            self.device.write("AEC_FIXEDBEAMSAZIMUTH_VALUES", want_az)
            self.device.write("AEC_FIXEDBEAMSELEVATION_VALUES",
                              [elev1 * RAD_PER_DEG, elev2 * RAD_PER_DEG])
            self.device.write("AEC_FIXEDBEAMSGATING", [1 if gating else 0])
            # enabling the beams is the last step, after the geometry they
            # should use is already in place
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

    # ------------------------------------------------------------------
    def scan(self, step_deg: float = 30.0, settle: float = 0.35) -> dict:
        """Sweep the beams to find where the speech actually is.

        The array has only two steerable beams, so the sweep puts beam 1 at
        theta and beam 2 at theta+180 and reads both energies per step: six
        steps of 30 degrees already cover the full circle twice over. Gating is
        turned off for the sweep, because gating silences the beam it judges
        inactive and would zero the very reading we are measuring.

        Ambient noise moves, so the peak is only a hint - the caller decides
        what to do with it. The device is left exactly as found (the beams and
        gating are restored) unless the caller keeps the steering via steer().
        """
        if step_deg <= 0 or step_deg > 180:
            return {"ok": False, "reason": "step must be within (0, 180)"}

        for name in STEER_WRITES:
            self._remember(name)
        try:
            self.device.write("AEC_FIXEDBEAMSGATING", [0])
            self.device.write("AEC_FIXEDBEAMSONOFF", [1])
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": str(exc), "restored": self.restore()}

        steps: list[dict] = []
        try:
            angle = 0.0
            while angle < 360.0:
                self.device.write("AEC_FIXEDBEAMSAZIMUTH_VALUES",
                                  [angle * RAD_PER_DEG,
                                   ((angle + 180.0) % 360.0) * RAD_PER_DEG])
                time.sleep(settle)
                # re-assert: a device that dropped the flag mid-sweep would
                # silently return free-running readings instead
                if list(self.device.read("AEC_FIXEDBEAMSONOFF")) != [1]:
                    self.device.write("AEC_FIXEDBEAMSONOFF", [1])
                    time.sleep(settle)
                e1, e2 = [float(v) for v in
                          self.device.read("AEC_SPENERGY_VALUES")[:2]]
                steps.append({"beam1_deg": angle,
                              "beam2_deg": (angle + 180.0) % 360.0,
                              "e1": e1, "e2": e2})
                angle += step_deg
        except Exception as exc:  # noqa: BLE001
            self.restore()
            return {"ok": False, "reason": str(exc), "steps": steps}

        self.restore()

        readings = step_deg / 2.0      # each step represents half a step either side
        peaks = sorted(
            [{"deg": s["beam1_deg"], "energy": s["e1"]} for s in steps] +
            [{"deg": s["beam2_deg"], "energy": s["e2"]} for s in steps],
            key=lambda p: p["energy"], reverse=True)
        loudest = peaks[0] if peaks else {"deg": 0.0, "energy": 0.0}
        silent = loudest["energy"] <= 0.0
        # Both beams are swept, so one physical source shows up twice: once as
        # beam1 at its bearing and once as beam2 at the opposite step. Counting
        # the second sighting as a second person would recommend steering both
        # beams onto the same talker, so the runner-up must be far enough away
        # to be someone else - a full step away at minimum, which at 30 degrees
        # of steps means the two sightings of one source (180 apart) collapse.
        min_gap = max(90.0, 2.0 * readings)
        second = next((p for p in peaks[1:]
                       if _angle_gap(p["deg"], loudest["deg"]) >= min_gap
                       and p["energy"] > 0), None)
        return {"ok": True, "steps": steps, "step_deg": step_deg,
                "peak": loudest, "second": second,
                "silent": silent,
                "hint": ("未检测到语音：扫描期间需要有人持续说话" if silent else
                         f"能量峰在 {loudest['deg']:.0f}° (E={loudest['energy']:.0f})")}


def _angle_gap(a: float, b: float) -> float:
    """Shortest angular distance between two bearings, in degrees."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def pair_name(category: int, source: int) -> str:
    """Human name for one <category>, <source> pair, best effort.

    Returns a name carrying a "?" when the category was never observed on
    this unit, so an unverified reading cannot be mistaken for a known one.
    """
    if category == CAT_ASR_OUTPUT:
        return BEAM_SOURCES.get(source, f"beam?{source}")
    if category == CAT_CONFERENCE:
        return "conference"
    if category == CAT_MIC_RAW:
        return f"mic{source}"
    if category == CAT_FAR_RAW:
        return "far-end raw"
    if category == CAT_FAR_GAINED:
        return "far-end gained"
    return f"cat?{category}/src{source}"


def describe_pair(category: int, source: int) -> str:
    """A pair as it would be printed alongside the channel it lands on."""
    return f"({category},{source})={pair_name(category, source)}"


class BeamRouter:
    def __init__(self, device: ReSpeaker):
        self.device = device
        self._original: list[int] | None = None

    def _remember(self) -> None:
        """Snapshot the routing the first time, and never again.

        Recording it on every write looks harmless but is not: setting two
        slots in a row (which is exactly what the dual-lane setup does) would
        leave `_original` holding the state after the *first* change, so
        restore() would put back a half-applied route and quietly leave the
        other slot pointing at the experiment. Only the first snapshot is the
        state the operator actually had.
        """
        if self._original is None:
            self._original = list(self.device.read("AUDIO_MGR_OP_ALL"))

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
        self._remember()

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
        """Roll back to the route saved before the first apply()."""
        if self._original is None:
            return False
        try:
            self.device.write("AUDIO_MGR_OP_ALL", self._original)
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    def current_pair(self, slot: str) -> tuple[int, int]:
        """The <category>, <source> pair the given slot currently outputs."""
        op_all = self.device.read("AUDIO_MGR_OP_ALL")
        base = 0 if slot.upper() == "L" else 6
        if len(op_all) < base + 2:
            raise ValueError(f"AUDIO_MGR_OP_ALL too short for slot {slot}")
        return tuple(op_all[base:base + 2])

    def route_slot(self, slot: str, category: int, source: int) -> dict:
        """Point one output slot at a category/source, with rollback.

        This is the whole of the channel-routing capability: a lot of callers
        only need "put beam 1 on L and beam 2 on R", and for that the
        generic apply(), which needs recon() to disambiguate pair order, is
        more machinery than necessary.
        """
        # snapshot the pre-experiment route, then start from the *current*
        # route, not from that snapshot: setting two slots in a row must
        # accumulate, and starting from the snapshot each time would silently
        # discard the previous slot's change.
        self._remember()
        op_all = list(self.device.read("AUDIO_MGR_OP_ALL"))
        base = 0 if slot.upper() == "L" else 6
        if len(op_all) < base + 2:
            return {"ok": False, "reason": "AUDIO_MGR_OP_ALL too short"}
        op_all[base:base + 2] = [category, source]
        try:
            self.device.write("AUDIO_MGR_OP_ALL", op_all)
            ok = list(self.device.read("AUDIO_MGR_OP_ALL")) == op_all
        except Exception as exc:  # noqa: BLE001
            self.restore()
            return {"ok": False, "reason": str(exc)}
        if not ok:
            self.restore()
            return {"ok": False, "reason": "verify failed, restored"}
        return {"ok": True, "slot": slot.upper(),
                "pair": (category, source),
                "channel": SLOT_CHANNEL.get(slot.upper()),
                "describe": describe_pair(category, source),
                "restore_to": list(self._original)}
