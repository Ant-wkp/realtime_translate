"""P2a: record the raw 6-channel capture and report what each channel holds.

Read-only: this never touches device parameters, it only records. The point is
to establish, from the hardware itself, which channel carries what before any
routing experiment builds on the assumption.

It uses one duplex PortAudio stream rather than separate record and playback
streams. Two streams on the same USB device both crash and, separately, cannot
guarantee that the far-end reference and the capture share a clock - a duplex
stream does, which is what makes the far-end test meaningful.

    python tools/p2a_capture.py --baseline 3 --live 6
    python tools/p2a_capture.py --stimulus far-end   # play into the array
    python tools/p2a_capture.py --report /tmp/cap.npy

Channel expectations, for reference while reading the output:
  ch0  processed "Conference" output   (AUDIO_MGR_OP_L = (8,0))
  ch1  ASR beam output, auto-select    (AUDIO_MGR_OP_R = (7,3))
  ch2-5 four raw mics; arecord reports a 6-channel device
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_RATE = 16000
CAPTURE_CHANNELS = 6
STIMULUS = (
    "/home/ant/.cache/modelscope/models/"
    "iic--speech_campplus_sv_zh-cn_16k-common/snapshots/master/examples/"
    "speaker2_a_cn_16k.wav"
)


def find_array() -> int:
    import sounddevice as sd

    for d in sd.query_devices():
        name = d["name"].lower()
        if ("xvf3800" in name or "respeaker" in name) and d["max_input_channels"] >= CAPTURE_CHANNELS:
            return d["index"]
    raise SystemExit("XVF3800 not found among sounddevice inputs")


def load_stimulus(path: str) -> np.ndarray:
    with wave.open(path) as w:
        rate = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = raw.astype(np.float32) / 32768.0
    if rate != SAMPLE_RATE:
        raise SystemExit(f"stimulus is {rate} Hz, expected {SAMPLE_RATE}")
    return audio


def record_duplex(device: int, seconds: float,
                  stimulus: np.ndarray | None,
                  play_from: float, play_seconds: float) -> np.ndarray:
    """Record `seconds` of 6-channel audio, optionally playing into the array.

    Playback goes to the same USB device, i.e. into the XVF3800's own far-end
    reference path. That is deliberate: it is the only way to inject a known
    signal without depending on the operator, and the far-end reference is the
    one routing (4,0)/(12,0) the documentation gives numeric values for.
    """
    import sounddevice as sd

    frames = int(seconds * SAMPLE_RATE)
    out = np.zeros((frames, 2), dtype=np.float32)
    if stimulus is not None:
        start = int(play_from * SAMPLE_RATE)
        n = min(int(play_seconds * SAMPLE_RATE), stimulus.size)
        end = min(start + n, frames)
        out[start:end, 0] = stimulus[: end - start]
        out[start:end, 1] = stimulus[: end - start]

    captured = np.zeros((frames, CAPTURE_CHANNELS), dtype=np.float32)

    def callback(indata, outdata, nframes, time_info, status):
        captured[:] = 0  # keep the array valid; real data is collected below
        outdata[:] = out[:nframes]

    # A blocking duplex stream is simplest and keeps both directions on one
    # clock. Read the input in a loop rather than via a callback so the
    # recording buffer is filled in order.
    with sd.Stream(device=(device, device), samplerate=SAMPLE_RATE,
                   channels=(CAPTURE_CHANNELS, 2), dtype="float32",
                   blocksize=1024) as stream:
        pos = 0
        play_pos = 0
        while pos < frames:
            block = min(1024, frames - pos)
            chunk, _ = stream.read(block)
            captured[pos:pos + block] = chunk[:block]
            pos += block
            # feed playback from the pre-built buffer
            remaining = out[play_pos:play_pos + block]
            if remaining.size:
                ob = np.zeros((block, 2), dtype=np.float32)
                ob[: remaining.shape[0]] = remaining
                stream.write(ob)
                play_pos += block
    return captured


def rms(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(a.astype(np.float64) ** 2, axis=0))


def peaks(a: np.ndarray) -> np.ndarray:
    return np.max(np.abs(a), axis=0)


def report(name: str, base: np.ndarray, live: np.ndarray) -> None:
    b, l, pk = rms(base), rms(live), peaks(live)
    print(f"\n=== {name} ===")
    print("%3s %12s %12s %9s %11s  %s"
          % ("ch", "base_rms", "live_rms", "delta_dB", "live_peak", "moved"))
    for c in range(CAPTURE_CHANNELS):
        db = (20 * np.log10(max(l[c], 1e-12) / b[c]) if b[c] > 1e-9
              else float("inf"))
        moved = "yes" if l[c] > max(b[c] * 2.0, 1e-4) else "no"
        print("%3d %12.6f %12.6f %+9.1f %11.5f  %s"
              % (c, b[c], l[c], db, pk[c], moved))


def per_second(a: np.ndarray) -> None:
    """Level over time, one row per second.

    Printed because the array's outputs need a couple of seconds to settle
    after the stream opens, and that transient is loud enough to swamp any
    baseline-versus-live comparison taken naively. Seeing it is the difference
    between measuring the device and measuring the transient.
    """
    print("\n--- per-second RMS (dBFS) ---")
    print("%5s" % "t" + "".join("%9s" % f"ch{c}"
                                  for c in range(CAPTURE_CHANNELS)))
    for s in range(a.shape[0] // SAMPLE_RATE):
        seg = a[s * SAMPLE_RATE:(s + 1) * SAMPLE_RATE]
        row = rms(seg)
        print("%5d" % s + "".join("%9.1f" % (20 * np.log10(max(v, 1e-12)))
                                     for v in row))


def correlation(a: np.ndarray) -> None:
    """Which channels carry the same content (high correlation = same source).

    This is what tells the processed outputs apart from the raw mic array: the
    four mics of one array all hear the same talker, so they move together,
    while a beamformed or noise-suppressed output does not.
    """
    live = [c for c in range(CAPTURE_CHANNELS) if rms(a)[c] > 1e-4]
    if len(live) < 2:
        print("\n(too few active channels for a correlation matrix)")
        return
    x = a[:, live]
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    corr = (x.T @ x) / len(x)
    print("\n--- channel correlation (1.00 = identical content) ---")
    print("      " + "".join("%7d" % c for c in live))
    for i, c in enumerate(live):
        print("ch%-4d" % c + "".join("%7.2f" % corr[i, j]
                                     for j in range(len(live))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=float, default=3.0)
    ap.add_argument("--live", type=float, default=6.0)
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds discarded after the stream opens; the array's "
                         "outputs need ~2s to settle and that transient would "
                         "otherwise be mistaken for the baseline (default 3)")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--stimulus", choices=["none", "far-end"], default="none",
                    help="far-end plays a wav into the array; none expects the "
                         "operator to make noise during the live window")
    ap.add_argument("--save", default=None, help="write the capture to .npy")
    ap.add_argument("--report", default=None,
                    help="analyse a previously saved .npy instead of recording")
    args = ap.parse_args(argv)

    if args.report:
        x = np.load(args.report)
        per_second(x)
        half = x.shape[0] // 3
        report(f"re-analysed {args.report}", x[:half], x[half:])
        correlation(x)
        return 0

    device = args.device if args.device is not None else find_array()
    total = args.settle + args.baseline + args.live
    stim = load_stimulus(STIMULUS) if args.stimulus == "far-end" else None
    if stim is None:
        print(f"[2a] recording {total:.0f}s on device {device}: "
              f"{args.settle:.0f}s settle (discarded), {args.baseline:.0f}s quiet, "
              f"then {args.live:.0f}s - PLEASE MAKE NOISE during the last window")
    else:
        print(f"[2a] recording {total:.0f}s on device {device}: "
              f"{args.settle:.0f}s settle (discarded), playing "
              f"{min(stim.size/SAMPLE_RATE, args.live):.1f}s of audio into the "
              f"array's own far-end reference")

    x = record_duplex(device, total, stim,
                      play_from=args.settle + args.baseline,
                      play_seconds=args.live)
    per_second(x)
    s0 = int(args.settle * SAMPLE_RATE)
    b = int((args.settle + args.baseline) * SAMPLE_RATE)
    report("capture (after settle)", x[s0:b], x[b:])
    correlation(x[b:])

    if args.save:
        np.save(args.save, x)
        print(f"\n[2a] saved {args.save}  ({total:.0f}s x "
              f"{CAPTURE_CHANNELS}ch @ {SAMPLE_RATE} Hz; note the first "
              f"{args.settle:.0f}s is settle and should be ignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())