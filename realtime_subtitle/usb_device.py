"""XVF3800 USB control plane: vendor transfers for VAD, DOA and parameters.

Device: Seeed reSpeaker XVF3800 4-Mic Array, VID 0x2886 / PID 0x001A.
Control uses vendor-specific control transfers on endpoint 0 (the XVF3800
firmware Servicer protocol):

    bmRequestType = CTRL_IN/OUT | TYPE_VENDOR | RECIPIENT_DEVICE
    bRequest      = 0
    wValue        = 0x80 | cmdid   (read sets the high bit)
    wIndex        = resid          (routes to the on-chip servicer)
    wLength       = read: payload bytes + 1 status byte
                    write: payload bytes only (no status prefix)

Status code 64 (SERVICER_COMMAND_RETRY) means the DSP servicer is busy -
retry up to 100 times with 10 ms sleeps. The first byte of a read response
is always the status; payload bytes follow it. A busy servicer rejects
writes with a USB stall, which surfaces as USBError.

PARAMETERS mirrors the official Seeed xvf_host.py command table:
name -> (resid, cmdid, count, access, type, description), where count is
the number of values, not bytes: uint8/char are 1 byte each, uint16 is
2 bytes, float/radians/int32/uint32 are 4 bytes each.
"""

from __future__ import annotations

import struct
import threading
import time

import usb.core
import usb.util

VID = 0x2886
PID = 0x001A

CTRL_IN = 0xC0
CTRL_OUT = 0x40
STATUS_RETRY = 64
RETRY_LIMIT = 100
RETRY_DELAY = 0.01

# resid -> servicer group, for the control API listing
RESID_GROUPS = {
    48: "application",
    33: "aec",
    35: "audio_mgr",
    20: "gpo",
    17: "post_processing",
}

_TYPE_SIZE = {"uint8": 1, "char": 1, "uint16": 2,
              "uint32": 4, "int32": 4, "float": 4, "radians": 4}
_FMT = {"uint8": "B", "uint16": "H", "uint32": "I",
        "int32": "i", "float": "f", "radians": "f"}

# name -> (resid, cmdid, count, access, type, description)
# Kept in sync with Seeed python_control/xvf_host.py (sw_xvf3800).
PARAMETERS = {
    # APPLICATION_SERVICER_RESID commands
    "VERSION": (48, 0, 3, "ro", "uint8", "Firmware version: major, minor, patch"),
    "BLD_MSG": (48, 1, 50, "ro", "char", "Build message, normally the build configuration name"),
    "BLD_HOST": (48, 2, 30, "ro", "char", "CI build host used to build the firmware"),
    "BLD_REPO_HASH": (48, 3, 40, "ro", "char", "GIT hash of the sw_xvf3800 repo used for the firmware"),
    "BLD_MODIFIED": (48, 4, 6, "ro", "char", "Whether the firmware repo was modified from the official release"),
    "BOOT_STATUS": (48, 5, 3, "ro", "char", "Booted via SPI or JTAG/FLASH"),
    "TEST_CORE_BURN": (48, 6, 1, "rw", "uint8", "Enable core burn; reboots the chip and resets all parameters"),
    "REBOOT": (48, 7, 1, "wo", "uint8", "Set to any value to reboot the chip and reset parameters"),
    "USB_BIT_DEPTH": (48, 8, 2, "rw", "uint8", "USB bit depth IN/OUT (16/24/32); setting reboots the chip"),
    "SAVE_CONFIGURATION": (48, 9, 1, "wo", "uint8", "Set to any value to save the configuration to flash"),
    "CLEAR_CONFIGURATION": (48, 10, 1, "wo", "uint8", "Set to any value to revert to the default configuration"),
    "AIC3104_HP_LEVEL": (48, 11, 1, "rw", "uint8", "AIC3104 headphone level [0..9]"),
    "AIC3104_LINEOUT_LEVEL": (48, 12, 1, "rw", "uint8", "AIC3104 lineout level [0..9]"),

    # AEC_RESID commands
    "SHF_BYPASS": (33, 70, 1, "rw", "uint8", "AEC bypass"),
    "AEC_NUM_MICS": (33, 71, 1, "ro", "int32", "Number of microphone inputs into the AEC"),
    "AEC_NUM_FARENDS": (33, 72, 1, "ro", "int32", "Number of farend inputs into the AEC"),
    "AEC_MIC_ARRAY_TYPE": (33, 73, 1, "ro", "int32", "Microphone array type (1 linear, 2 squarecular)"),
    "AEC_MIC_ARRAY_GEO": (33, 74, 12, "ro", "float", "Array geometry: 3 XYZ coordinates in m per mic"),
    "AEC_AZIMUTH_VALUES": (33, 75, 4, "ro", "radians", "Azimuths: beam1, beam2, free-running beam, auto-select beam"),
    "TEST_AEC_DISABLE_CONTROL": (33, 76, 1, "wo", "uint32", "Disable control processing in the AEC (testing only)"),
    "AEC_CURRENT_IDLE_TIME": (33, 77, 1, "ro", "uint32", "AEC processing current idle time in 10ns ticks"),
    "AEC_MIN_IDLE_TIME": (33, 78, 1, "ro", "uint32", "AEC processing minimum idle time in 10ns ticks"),
    "AEC_RESET_MIN_IDLE_TIME": (33, 79, 1, "wo", "uint32", "Reset the AEC minimum idle time"),
    "AEC_SPENERGY_VALUES": (33, 80, 4, "ro", "float", "Speech energy per beam; >0 indicates speech"),
    "AEC_FIXEDBEAMSAZIMUTH_VALUES": (33, 81, 2, "rw", "radians", "Azimuths for fixed beams 1 and 2"),
    "AEC_FIXEDBEAMSELEVATION_VALUES": (33, 82, 2, "rw", "radians", "Elevation angles for fixed beams 1 and 2"),
    "AEC_FIXEDBEAMSGATING": (33, 83, 1, "rw", "uint8", "Enable gating for fixed beams; energy picks the active beam"),
    "SPECIAL_CMD_AEC_FAR_MIC_INDEX": (33, 90, 2, "wo", "int32", "Set far/mic index for AEC filter read; triggers the read"),
    "SPECIAL_CMD_AEC_FILTER_COEFF_START_OFFSET": (33, 91, 1, "rw", "int32", "Start offset for filter coefficient get/set"),
    "SPECIAL_CMD_AEC_FILTER_COEFFS": (33, 92, 15, "rw", "float", "Get or set AEC filter coefficients"),
    "SPECIAL_CMD_AEC_FILTER_LENGTH": (33, 93, 1, "ro", "int32", "AEC filter length"),
    "AEC_FILTER_CMD_ABORT": (33, 94, 1, "wo", "int32", "Reset the special command state machine"),
    "AEC_AECPATHCHANGE": (33, 0, 1, "ro", "int32", "AEC path change detection (0/1)"),
    "AEC_HPFONOFF": (33, 1, 1, "rw", "int32", "High-pass filter (0 off, 1 on70, 2 on125, 3 on150, 4 on180)"),
    "AEC_AECSILENCELEVEL": (33, 2, 2, "rw", "float", "Power threshold for signal detection in adaptive filter (set,cur)"),
    "AEC_AECCONVERGED": (33, 3, 1, "ro", "int32", "Whether the AEC has converged (0/1)"),
    "AEC_AECEMPHASISONOFF": (33, 4, 1, "rw", "int32", "Pre/de-emphasis filtering (0 off, 1 on, 2 on_eq)"),
    "AEC_FAR_EXTGAIN": (33, 5, 1, "rw", "float", "External gain in dB applied to far-end reference signals"),
    "AEC_PCD_COUPLINGI": (33, 6, 1, "rw", "float", "Sensitivity parameter for PCD [0.0 .. 1.0]"),
    "AEC_PCD_MINTHR": (33, 7, 1, "rw", "float", "Minimum threshold used in PCD [0.0 .. 0.02]"),
    "AEC_PCD_MAXTHR": (33, 8, 1, "rw", "float", "Maximum threshold used in PCD [0.025 .. 0.2]"),
    "AEC_RT60": (33, 9, 1, "ro", "float", "Current RT60 estimate in seconds; negative if invalid"),
    "AEC_ASROUTONOFF": (33, 35, 1, "rw", "int32", "ASR output (1) vs AEC residuals per mic (0)"),
    "AEC_ASROUTGAIN": (33, 36, 1, "rw", "float", "Fixed gain applied to ASR output [0.0 .. 1000.0]"),
    "AEC_FIXEDBEAMSONOFF": (33, 37, 1, "rw", "int32", "Fixed focused beam mode (0 off, 1 on)"),
    "AEC_FIXEDBEAMNOISETHR": (33, 38, 2, "rw", "float", "Noise canceller update threshold for fixed beam mode [0.0 .. 1.0]"),

    # AUDIO_MGR_RESID commands
    "AUDIO_MGR_MIC_GAIN": (35, 0, 1, "rw", "float", "Pre-SHF microphone gain"),
    "AUDIO_MGR_REF_GAIN": (35, 1, 1, "rw", "float", "Pre-SHF reference gain"),
    "AUDIO_MGR_CURRENT_IDLE_TIME": (35, 2, 1, "ro", "int32", "Audio manager current idle time in 10ns ticks"),
    "AUDIO_MGR_MIN_IDLE_TIME": (35, 3, 1, "ro", "int32", "Audio manager minimum idle time in 10ns ticks"),
    "AUDIO_MGR_RESET_MIN_IDLE_TIME": (35, 4, 1, "wo", "int32", "Reset audio manager minimum idle time"),
    "MAX_CONTROL_TIME": (35, 5, 1, "ro", "int32", "Audio manager max control time"),
    "RESET_MAX_CONTROL_TIME": (35, 6, 1, "wo", "int32", "Reset audio manager max control time"),
    "I2S_CURRENT_IDLE_TIME": (35, 7, 1, "ro", "int32", "I2S current idle time in 10ns ticks"),
    "I2S_MIN_IDLE_TIME": (35, 8, 1, "ro", "int32", "I2S minimum idle time in 10ns ticks"),
    "I2S_RESET_MIN_IDLE_TIME": (35, 9, 1, "wo", "int32", "Reset I2S minimum idle time"),
    "I2S_INPUT_PACKED": (35, 10, 1, "rw", "uint8", "Expect packed input on I2S/USB channels when nonzero"),
    "AUDIO_MGR_SELECTED_AZIMUTHS": (35, 11, 2, "ro", "radians", "Processed DoA and auto-select beam DoA; NaN if no speech"),
    "AUDIO_MGR_SELECTED_CHANNELS": (35, 12, 2, "rw", "uint8", "Channels output to MUX_USER_CHOSEN_CHANNELS"),
    "AUDIO_MGR_OP_PACKED": (35, 13, 2, "rw", "uint8", "Packing status for L and R output channels"),
    "AUDIO_MGR_OP_UPSAMPLE": (35, 14, 2, "rw", "uint8", "Upsample status for L and R output channels"),
    "AUDIO_MGR_OP_L": (35, 15, 2, "rw", "uint8", "Category and source for the L output channel"),
    "AUDIO_MGR_OP_L_PK0": (35, 16, 2, "rw", "uint8", "Category/source, first L source in packed mode"),
    "AUDIO_MGR_OP_L_PK1": (35, 17, 2, "rw", "uint8", "Category/source, second L source in packed mode"),
    "AUDIO_MGR_OP_L_PK2": (35, 18, 2, "rw", "uint8", "Category/source, third L source in packed mode"),
    "AUDIO_MGR_OP_R": (35, 19, 2, "rw", "uint8", "Category and source for the R output channel"),
    "AUDIO_MGR_OP_R_PK0": (35, 20, 2, "rw", "uint8", "Category/source, first R source in packed mode"),
    "AUDIO_MGR_OP_R_PK1": (35, 21, 2, "rw", "uint8", "Category/source, second R source in packed mode"),
    "AUDIO_MGR_OP_R_PK2": (35, 22, 2, "rw", "uint8", "Category/source, third R source in packed mode"),
    "AUDIO_MGR_OP_ALL": (35, 23, 12, "rw", "uint8", "Category/source for all 3 sources on L and R (6 pairs)"),
    "I2S_INACTIVE": (35, 24, 1, "ro", "uint8", "0 when the main audio loop exchanges I2S samples, else 1"),
    "AUDIO_MGR_FAR_END_DSP_ENABLE": (35, 25, 1, "rw", "uint8", "Enable/disable XVF3800 far-end DSP"),
    "AUDIO_MGR_SYS_DELAY": (35, 26, 1, "rw", "int32", "Reference signal delay in samples before the SHF algorithm"),
    "I2S_DAC_DSP_ENABLE": (35, 27, 1, "rw", "uint8", "DAC performs DSP on the far-end reference signal"),

    # GPO_SERVICER_RESID commands
    "GPO_READ_VALUES": (20, 0, 5, "ro", "uint8", "Logic levels of GPO pins X0D11, X0D30, X0D31, X0D33, X0D39"),
    "GPO_WRITE_VALUE": (20, 1, 2, "wo", "uint8", "Set logic level of a selected GPO pin"),
    "GPO_PORT_PIN_INDEX": (20, 2, 2, "rw", "uint32", "GPO port/pin index the following commands target"),
    "GPO_PIN_VAL": (20, 3, 3, "wo", "uint8", "Write one pin: port_index, pin_index, value"),
    "GPO_PIN_ACTIVE_LEVEL": (20, 4, 1, "rw", "uint32", "Active level of the GPO port/pin (1 high, 0 low)"),
    "LED_EFFECT": (20, 12, 1, "rw", "uint8", "LED mode: 0 off, 1 breath, 2 rainbow, 3 single, 4 doa, 5 ring"),
    "LED_BRIGHTNESS": (20, 13, 1, "rw", "uint8", "LED brightness for breath and rainbow modes"),
    "LED_GAMMIFY": (20, 14, 1, "rw", "uint8", "Enable gamma correction (0/1)"),
    "LED_SPEED": (20, 15, 1, "rw", "uint8", "Effect speed of breath and rainbow modes"),
    "LED_COLOR": (20, 16, 1, "rw", "uint32", "LED color for breath and single color modes"),
    "LED_DOA_COLOR": (20, 17, 2, "rw", "uint32", "DOA mode colors: base color, doa color"),
    "DOA_VALUE": (20, 18, 2, "ro", "uint16", "payload[0] = azimuth 0-359, payload[1] = 1 if speech detected"),
    "LED_RING_COLOR": (20, 19, 12, "rw", "uint32", "Ring mode color, one per LED"),

    # PP_RESID commands
    "PP_CURRENT_IDLE_TIME": (17, 70, 1, "ro", "uint32", "PP processing current idle time in 10ns ticks"),
    "PP_MIN_IDLE_TIME": (17, 71, 1, "ro", "uint32", "PP processing minimum idle time in 10ns ticks"),
    "PP_RESET_MIN_IDLE_TIME": (17, 72, 1, "wo", "uint32", "Reset the PP minimum idle time"),
    "PP_NL_MODEL_CMD_ABORT": (17, 94, 1, "wo", "int32", "Reset the NL model special command state machine"),
    "PP_EQUALIZATION_CMD_ABORT": (17, 100, 1, "wo", "int32", "Reset the equalization special command state machine"),
    "PP_AGCONOFF": (17, 10, 1, "rw", "int32", "Automatic gain control (0 off, 1 on)"),
    "PP_AGCMAXGAIN": (17, 11, 1, "rw", "float", "Maximum AGC gain factor [1.0 .. 1000.0]"),
    "PP_AGCDESIREDLEVEL": (17, 12, 1, "rw", "float", "Target power level of the output signal [1e-8 .. 1.0]"),
    "PP_AGCGAIN": (17, 13, 1, "rw", "float", "Current AGC gain factor [1.0 .. 1000.0]"),
    "PP_AGCTIME": (17, 14, 1, "rw", "float", "Ramp-up/down time-constant [0.5 .. 4.0] s"),
    "PP_AGCFASTTIME": (17, 15, 1, "rw", "float", "Ramp-down time-constant for peaks [0.05 .. 4.0] s"),
    "PP_AGCALPHAFASTGAIN": (17, 16, 1, "rw", "float", "Gain threshold enabling fast alpha mode [0.0 .. 1000.0]"),
    "PP_AGCALPHASLOW": (17, 17, 1, "rw", "float", "Slow memory parameter for speech power [0.0 .. 1.0]"),
    "PP_AGCALPHAFAST": (17, 18, 1, "rw", "float", "Fast memory parameter for speech power [0.0 .. 1.0]"),
    "PP_LIMITONOFF": (17, 19, 1, "rw", "int32", "Limiter on communication output (0 off, 1 on)"),
    "PP_LIMITPLIMIT": (17, 20, 1, "rw", "float", "Maximum limiter power [1e-8 .. 1.0]"),
    "PP_MIN_NS": (17, 21, 1, "rw", "float", "Gain-floor for stationary noise suppression [0.0 .. 1.0]"),
    "PP_MIN_NN": (17, 22, 1, "rw", "float", "Gain-floor for non-stationary noise suppression [0.0 .. 1.0]"),
    "PP_ECHOONOFF": (17, 23, 1, "rw", "int32", "Echo suppression (0 off, 1 on)"),
    "PP_GAMMA_E": (17, 24, 1, "rw", "float", "Over-subtraction factor of echo, direct/early [0.0 .. 2.0]"),
    "PP_GAMMA_ETAIL": (17, 25, 1, "rw", "float", "Over-subtraction factor of echo, tail [0.0 .. 2.0]"),
    "PP_GAMMA_ENL": (17, 26, 1, "rw", "float", "Over-subtraction factor of non-linear echo [0.0 .. 5.0]"),
    "PP_NLATTENONOFF": (17, 27, 1, "rw", "int32", "Non-linear echo attenuation (0 off, 1 on)"),
    "PP_NLAEC_MODE": (17, 28, 1, "rw", "int32", "Non-linear AEC training mode (0 normal, 1 train, 2 train2)"),
    "PP_MGSCALE": (17, 29, 3, "rw", "float", "Minimum gain scale for acoustic echo suppression (max, min, cur)"),
    "PP_FMIN_SPEINDEX": (17, 30, 1, "rw", "float", "Frequency below SPEINDEX is more suppressed in double talk"),
    "PP_DTSENSITIVE": (17, 31, 1, "rw", "int32", "Tradeoff between echo suppression and doubletalk performance"),
    "PP_ATTNS_MODE": (17, 32, 1, "rw", "int32", "Additional AGC gain reduction during non-speech (0/1)"),
    "PP_ATTNS_NOMINAL": (17, 33, 1, "rw", "float", "Reduction during non-speech at nominal speech level [0.0 .. 1.0]"),
    "PP_ATTNS_SLOPE": (17, 34, 1, "rw", "float", "Extra suppression during non-speech vs AGC level [0.0 .. 5.0]"),
}


def find():
    """Locate the XVF3800 device.

    Vendor control transfers are device-recipient EP0 requests: no
    interface claim, kernel-driver detach or configuration change is
    needed, so snd-usb-audio keeps streaming while VAD/DOA are polled.
    """
    return usb.core.find(idVendor=VID, idProduct=PID)


class ReSpeaker:
    """Thin wrapper over the XVF3800 servicer protocol."""

    def __init__(self, timeout: int = 1000):
        self.timeout = timeout
        self._lock = threading.Lock()
        self.dev = find()
        if self.dev is None:
            raise RuntimeError(
                "reSpeaker XVF3800 not found (VID 0x2886 / PID 0x001A)"
            )

    def close(self) -> None:
        if self.dev is not None:
            try:
                usb.util.dispose_resources(self.dev)
            finally:
                self.dev = None

    # ------------------------------------------------------------------
    def _transfer(self, resid: int, cmdid: int, out: bool,
                  payload: bytes | None = None, w_length: int = 0):
        bm = CTRL_OUT if out else CTRL_IN
        w_value = cmdid if out else (0x80 | cmdid)
        data = payload if out else w_length
        return self.dev.ctrl_transfer(bm, 0, w_value, resid, data,
                                      self.timeout)

    def _read_raw(self, name: str) -> bytes:
        with self._lock:
            resid, cmdid, count, access, ptype, _ = PARAMETERS[name]
            if access == "wo":
                raise ValueError(f"{name} is write-only")
            w_length = count * _TYPE_SIZE[ptype] + 1  # +1 status byte
            for _ in range(RETRY_LIMIT):
                resp = self._transfer(resid, cmdid, out=False,
                                      w_length=w_length)
                if not resp:
                    raise RuntimeError(f"{name}: empty response")
                status = int(resp[0])
                if status == 0:
                    return bytes(resp[1:])
                if status != STATUS_RETRY:
                    raise RuntimeError(f"{name}: servicer status {status}")
                time.sleep(RETRY_DELAY)
            raise RuntimeError(f"{name}: servicer busy after {RETRY_LIMIT} retries")

    def _write_raw(self, name: str, payload: bytes) -> None:
        with self._lock:
            resid, cmdid, _, _, _, _ = PARAMETERS[name]
            for attempt in range(RETRY_LIMIT):
                try:
                    self._transfer(resid, cmdid, out=True, payload=payload)
                    return
                except usb.core.USBError:
                    # a busy servicer stalls the OUT transfer; retry
                    if attempt == RETRY_LIMIT - 1:
                        raise
                    time.sleep(RETRY_DELAY)

    # ------------------------------------------------------------------
    def read(self, name: str):
        """Read a parameter: list of decoded values (str for char)."""
        _, _, count, _, ptype, _ = PARAMETERS[name]
        raw = self._read_raw(name)
        if ptype == "char":
            return raw.rstrip(b"\x00").decode("utf-8", errors="ignore")
        size = count * _TYPE_SIZE[ptype]
        return list(struct.unpack("<" + _FMT[ptype] * count, raw[:size]))

    def write(self, name: str, value) -> None:
        _, _, count, access, ptype, _ = PARAMETERS[name]
        if access == "ro":
            raise ValueError(f"{name} is read-only")
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            value = [value]
        if len(value) != count:
            raise ValueError(f"{name} expects {count} value(s), got {len(value)}")
        if ptype == "char":
            payload = str(value[0]).encode("utf-8")
        elif ptype in ("float", "radians"):
            payload = struct.pack("<" + "f" * count, *[float(v) for v in value])
        else:
            payload = struct.pack("<" + _FMT[ptype] * count,
                                  *[int(v) for v in value])
        self._write_raw(name, payload)

    # convenience ---------------------------------------------------------
    def read_vad(self) -> int:
        """Hardware speech activity flag: DOA_VALUE payload[1]."""
        vals = self.read("DOA_VALUE")
        return int(vals[1]) if len(vals) > 1 else 0

    def read_doa(self) -> tuple[float, int] | None:
        """DOA_VALUE: (azimuth_deg 0-359, speech_activity 0/1)."""
        vals = self.read("DOA_VALUE")
        if not vals:
            return None
        return float(vals[0]), (int(vals[1]) if len(vals) > 1 else 0)
