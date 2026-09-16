"""XVF3800 USB control plane: vendor transfers for VAD, DOA and parameters.

Only the control plane lives here; audio data flows over USB Audio into
PortAudio. Scalar parameters come back as single-element tuples and array
parameters as lists - read() normalizes both.
"""

from __future__ import annotations

import struct
import threading

import usb.core
import usb.util

VID = 0x20F1
PID = 0x0030

# wIndex command codes (subset of the official xvf_host command table)
CMD = {
    "DOA_VALUE": 0x00D1,
    "VAD_VALUE": 0x00D2,
    "AEC_FIXEDBEAMSONOFF": 0x00A9,
    "AEC_AZIMUTH_VALUES": 0x00B2,
    "AEC_SPENERGY_VALUES": 0x00B3,
    "AUDIO_MGR_OP_L": 0x0052,
    "AUDIO_MGR_OP_R": 0x0053,
    "OP_ALL": 0x0056,
    "AEC_MIC_ARRAY_GEO": 0x00A0,
}

# name -> (code, type, readable, writable)
PARAMETERS = {
    "DOA_VALUE": (CMD["DOA_VALUE"], "float", True, False),
    "VAD_VALUE": (CMD["VAD_VALUE"], "uint", True, False),
    "AEC_FIXEDBEAMSONOFF": (CMD["AEC_FIXEDBEAMSONOFF"], "uint", True, True),
    "AEC_AZIMUTH_VALUES": (CMD["AEC_AZIMUTH_VALUES"], "float_array", True, True),
    "AEC_SPENERGY_VALUES": (CMD["AEC_SPENERGY_VALUES"], "float_array", True, False),
    "AUDIO_MGR_OP_L": (CMD["AUDIO_MGR_OP_L"], "uint", True, True),
    "AUDIO_MGR_OP_R": (CMD["AUDIO_MGR_OP_R"], "uint", True, True),
    "OP_ALL": (CMD["OP_ALL"], "uint_array", True, True),
    "AEC_MIC_ARRAY_GEO": (CMD["AEC_MIC_ARRAY_GEO"], "float_array", True, False),
}


class ReSpeaker:
    """Thin wrapper over one XVF3800 control interface."""

    RETRY = 100

    def __init__(self, timeout: int = 1000):
        self.timeout = timeout
        self._lock = threading.Lock()
        self.dev = find()
        if self.dev is None:
            raise RuntimeError("reSpeaker XVF3800 not found")

    def close(self) -> None:
        if self.dev is not None:
            usb.util.dispose_resources(self.dev)
            self.dev = None

    # ------------------------------------------------------------------
    def _ctrl_transfer(self, cmd: int, data=None, length: int = 64):
        with self._lock:
            return self.dev.ctrl_transfer(
                0x40,  # vendor, host-to-device
                cmd,
                0,
                0,
                data if data is not None else length,
                self.timeout,
            )

    def read(self, name: str):
        """Read a parameter, normalizing scalar tuples and arrays."""
        code, ptype, _, _ = PARAMETERS[name]
        raw = self._ctrl_transfer(code, length=64)
        if ptype == "uint":
            return list(struct.unpack("<" + "H" * (len(raw) // 2), raw))
        if ptype == "float":
            return list(struct.unpack("<" + "f" * (len(raw) // 4), raw))
        if ptype == "uint_array":
            return list(struct.unpack("<" + "H" * (len(raw) // 2), raw))
        return list(struct.unpack("<" + "f" * (len(raw) // 4), raw))

    def write(self, name: str, value) -> None:
        code, ptype, _, writable = PARAMETERS[name]
        if not writable:
            raise ValueError(f"{name} is read-only")
        if ptype == "uint":
            data = struct.pack("<H", int(value))
        elif ptype == "uint_array":
            data = struct.pack("<" + "H" * len(value), *[int(v) for v in value])
        else:
            data = struct.pack("<" + "f" * len(value), *[float(v) for v in value])
        self._ctrl_transfer(code, data=data)

    # convenience -------------------------------------------------------
    def read_vad(self) -> int:
        vals = self.read("VAD_VALUE")
        return int(vals[0]) if vals else 0

    def read_doa(self) -> float | None:
        """DOA_VALUE returns (azimuth_deg, speech_activity) in one read."""
        vals = self.read("DOA_VALUE")
        if not vals:
            return None
        azimuth = vals[0]
        self._last_vad = int(vals[1]) if len(vals) > 1 else getattr(self, "_last_vad", 0)
        return azimuth


def find():
    """Locate the XVF3800 control device, detach kernel driver if needed."""
    dev = usb.core.find(idVendor=VID, idProduct=PID)
    if dev is None:
        return None
    if dev.is_kernel_driver_active(0):
        dev.detach_kernel_driver(0)
    try:
        dev.set_configuration()
    except usb.core.USBError:
        pass  # already configured
    return dev
