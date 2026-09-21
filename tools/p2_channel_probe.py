"""P2 probe: which USB capture channel does each output selection land on?

The dual-lane design needs one fixed beam per lane, so it needs two capture
channels carrying two different beams. The two programmable outputs are OP_L
and OP_R, and the firmware documentation says they feed the first two channels
of the capture stream - but "which channel is which beam" is exactly the kind
of thing that is documented in one place, contradicted in another, and matters
enough to measure. So this probe measures it and prints only observations.

Three questions, in the order they must be answered:

  1. Is the documented slot-to-channel claim true? Set L to the far-end raw
     reference and R to the far-end with gain, play audio, and see which
     channels move. The far-end pair is the right choice because it is the one
     mapping the documentation states numerically, so a wrong result
     invalidates the *claim*, not just our guess.

  2. Do channels 2-5 really ignore the output selection? The same run answers
     it: if OP_L/OP_R only ever move channels 0 and 1, that is the claim the
     pipeline's channel indices rest on.

  3. Does the ASR-output category expose beams by index? Channel 1 ships as
     (7, 3), and source 3 is the auto-select beam in the same ordering
     AEC_AZIMUTH_VALUES uses. If category 7 with source 0/1 really are beam 1
     and beam 2, then routing becomes (7, 0) on L and (7, 1) on R - no beam
     scan, no ambiguity. This is the load-bearing guess and it is labelled as
     such everywhere it appears.

Nothing here writes flash: OP_L/OP_R/AUDIO_MGR_OP_ALL are runtime parameters,
and the unit's original routing is restored on every exit path.

    python tools/p2_channel_probe.py                 # all stages
    python tools/p2_channel_probe.py --stage farend  # just the far-end check
    python tools/p2_channel_probe.py --dry-run       # capture only, no writes
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realtime_subtitle.beam_router import (     # noqa: E402
    CAT_ASR_OUTPUT, CAT_FAR_GAINED, CAT_FAR_RAW, SLOT_CHANNEL, describe_pair,
)

SAMPLE_RATE = 16000
CAPTURE_CHANNELS = 6


def find_capture_device() -> str:
    """The ALSA name for the array, as arecord spells it."""
    out = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if "xvf3800" in line.lower() or "respeaker" in line.lower():
            # "card 1: XVF3800 [reSpeaker XVF3800 4-Mic Array], device 0: ..."
            card = line.split(":")[0].split()[-1]
            return f"hw:{card},0"
    raise SystemExit(
        "no XVF3800 capture device found by arecord - is the array plugged in?\n"
        + out.stdout
    )


def capture(device: str, seconds: float) -> np.ndarray:
    """Record the full 6-channel stream, returning (samples, channels)."""
    frames = int(seconds * SAMPLE_RATE)
    cmd = ["arecord", "-D", device, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
           "-c", str(CAPTURE_CHANNELS), "-t", "raw", "-q",
           "--samples", str(frames)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(f"arecord failed: {proc.stderr.decode(errors='replace')}")
    raw = np.frombuffer(proc.stdout, dtype=np.int16)
    if raw.size == 0:
        raise SystemExit("arecord returned no samples")
    usable = (raw.size // CAPTURE_CHANNELS) * CAPTURE_CHANNELS
    return raw[:usable].reshape(-1, CAPTURE_CHANNELS).astype(np.float32) / 32768.0


def channel_rms(audio: np.ndarray) -> list[float]:
    return [float(np.sqrt(np.mean(audio[:, c] ** 2))) for c in range(audio.shape[1])]


def channel_peak(audio: np.ndarray) -> list[int]:
    return [int(np.max(np.abs(audio[:, c])) * 32767) for c in range(audio.shape[1])]


def compare(before: np.ndarray, during: np.ndarray) -> list[dict]:
    """Per-channel change, in dB, between a quiet baseline and a signal run.

    dB rather than raw RMS: the channels sit at wildly different levels (the
    processed output is normalised, the raw mics are not), so an absolute
    threshold would call a quiet-but-moving channel silent. A ratio against
    the same channel's own baseline does not care about that offset.
    """
    base = channel_rms(before)
    live = channel_rms(during)
    rows = []
    for c, (b, l) in enumerate(zip(base, live)):
        if b > 1e-9:
            db = 20.0 * np.log10(max(l, 1e-12) / b)
        else:
            db = float("inf") if l > 1e-6 else 0.0
        rows.append({"ch": c, "base": b, "live": l, "db": float(db),
                     "moved": l > max(b * 2.0, 1e-4)})
    return rows


def print_rows(rows: list[dict], label: str) -> None:
    print(f"\n  {label}")
    print(f"  {'ch':>3} {'base_rms':>10} {'live_rms':>10} {'delta':>8}  moved")
    for r in rows:
        db = "   inf" if r["db"] == float("inf") else f"{r['db']:+7.1f}"
        print(f"  {r['ch']:>3} {r['base']:>10.6f} {r['live']:>10.6f} "
              f"{db:>8}  {'yes' if r['moved'] else 'no'}")


def make_stimulus() -> str:
    """A stimulus with obvious broadband content, written to /tmp."""
    # reuse a known-good speech wav if present, else synthesise a sweep
    candidates = [
        "/home/ant/.cache/modelscope/models/"
        "iic--speech_campplus_sv_zh-cn_16k-common/snapshots/master/examples/"
        "speaker2_a_cn_16k.wav",
    ]
    path = next((p for p in candidates if Path(p).exists()), None)
    if path is None:
        raise SystemExit("no stimulus wav found")
    return path


def play(path: str) -> None:
    subprocess.run(["aplay", "-D", "default", "-q", path], check=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["farend", "beam", "all"], default="all")
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="capture length per measurement (default 3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="capture channels without writing any routing")
    ap.add_argument("--stimulus", default=None, help="wav to play as the signal")
    args = ap.parse_args(argv)

    from realtime_subtitle.usb_device import ReSpeaker
    try:
        device = ReSpeaker()
    except Exception as exc:  # noqa: BLE001
        print(f"[p2] XVF3800 control plane unavailable: {exc}")
        return 1

    from realtime_subtitle.beam_router import BeamRouter
    router = BeamRouter(device)

    capture_dev = find_capture_device()
    print(f"[p2] control plane ok; capture device {capture_dev}")
    recon = router.recon()
    print(f"[p2] AUDIO_MGR_OP_ALL = {recon['op_all']}")
    pairs = [tuple(recon['op_all'][i:i + 2]) for i in range(0, 12, 2)]
    for i, p in enumerate(pairs):
        slot = "L" if i < 3 else "R"
        ch = "ch0/ch1" if i in (0, 3) else "  (secondary slot)"
        print(f"       {slot} slot {i % 3}: {describe_pair(*p)}  {ch}")
    print(f"[p2] device reports OP_L={router.current_pair('L')} "
          f"OP_R={router.current_pair('R')}")

    stimulus = args.stimulus or make_stimulus()
    print(f"[p2] stimulus: {stimulus}")

    # baseline with the unit quiet; the operator must stay silent for this
    print(f"[p2] capturing {args.seconds:.0f}s baseline - keep quiet ...")
    baseline = capture(capture_dev, args.seconds)
    print_rows([{"ch": c, "base": 0.0, "live": v, "db": 0.0, "moved": False}
                for c, v in enumerate(channel_rms(baseline))],
               "baseline levels (no stimulus)")

    if args.dry_run:
        print("\n[p2] --dry-run: not writing any routing. Capturing with the "
              "current route so the capture path itself is exercised.")
        during = capture(capture_dev, args.seconds)
        print_rows(compare(baseline, during), "current route (device internal)")
        device.close()
        return 0

    results = {}
    try:
        if args.stage in ("farend", "all"):
            print("\n=== stage 1: far-end routing (documented mapping) ===")
            print("[p2] OP_L <- (4,0) far-end raw, OP_R <- (12,0) far-end gained")
            print(router.route_slot("L", CAT_FAR_RAW, 0))
            print(router.route_slot("R", CAT_FAR_GAINED, 0))
            print(f"[p2] playing stimulus, capturing {args.seconds:.0f}s ...")
            t = threading.Thread(target=play, args=(stimulus,), daemon=True)
            t.start()
            during = capture(capture_dev, args.seconds)
            t.join(timeout=1)
            rows = compare(baseline, during)
            print_rows(rows, "with far-end routed to the two outputs")
            moved = [r["ch"] for r in rows if r["moved"]]
            print(f"[p2] channels that moved: {moved}")
            if moved == [0, 1]:
                print("[p2] CONFIRMED: the two outputs land on ch0 and ch1, "
                      "and ch2-5 ignore the output selection")
            elif set(moved) <= {0, 1} and moved:
                print(f"[p2] PARTIAL: only {moved} moved - the two outputs do "
                      f"land in ch0/ch1, but one side may not have carried the "
                      f"stimulus (check far-end routing / playback)")
            else:
                print("[p2] CONTRADICTED: expected ch0/ch1 to move; "
                      "the slot-to-channel claim does not hold as stated")
            results["farend"] = rows
            print(f"[p2] restoring: {router.restore()}")
            time.sleep(0.3)

        if args.stage in ("beam", "all"):
            print("\n=== stage 2: does category 7 expose beams by index? ===")
            print("[p2] UNVERIFIED GUESS being tested: (7,0)=beam1, (7,1)=beam2. "
                  "If this holds, the lanes need no beam scan at all.")
            print(router.route_slot("L", CAT_ASR_OUTPUT, 0))
            print(router.route_slot("R", CAT_ASR_OUTPUT, 1))
            # With both beams live (no gating) a single talker reaches both, so
            # this stage can only show whether the two outputs differ from each
            # other and from the recorded baseline - it cannot by itself prove
            # which beam is which. Steer the beams to different bearings if you
            # want the beams themselves distinguished.
            print(f"[p2] recording {args.seconds:.0f}s on both beams ...")
            t = threading.Thread(target=play, args=(stimulus,), daemon=True)
            t.start()
            during = capture(capture_dev, args.seconds)
            t.join(timeout=1)
            rows = compare(baseline, during)
            print_rows(rows, "with (7,0) on L and (7,1) on R")
            print("[p2] NOTE: identical ch0/ch1 here is expected when both beams "
                  "hear the same talker; it does not falsify the guess. To test "
                  "the mapping itself, steer beam1 and beam2 apart and have a "
                  "talker move between them.")
            results["beam"] = rows
            print(f"[p2] restoring: {router.restore()}")
    finally:
        restored = router.restore()
        after = router.recon()["op_all"]
        print(f"\n[p2] final restore={restored}, AUDIO_MGR_OP_ALL={after}")
        device.close()

    print("\n[p2] done - nothing persisted to flash (no SAVE_CONFIGURATION was "
          "issued, and the original routing was written back)")
    return 0


if __name__ == "__main__":
    sys.exit(main())