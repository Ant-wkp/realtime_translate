"""P7 automated: verify each lane really is bound to its own beam.

The manual P7 run needs two people standing at known bearings. That is the
only way to test the case the feature exists for - two people talking at once -
but it is not needed to test the *plumbing* underneath it, and the plumbing is
where a mistake would be silent.

The trick is to hold the sound source still and move the beams instead. A
laptop speaker sits somewhere out in the room at some bearing. Steer beam 1 at
that bearing and beam 2 away from it, and lane 0 should be the loud one. Swap
the steering, and lane 1 should be the loud one. Nothing moves except a
register we control, so the test is repeatable and needs nobody to stand
anywhere.

That establishes exactly two things, both of which the manual test depends on:

  1. The routing is what we think: OP_L carries beam 1 and OP_R carries beam 2,
     verified per lane rather than assumed from a pair table.
  2. The two lanes track the beams independently: swapping the beams swaps
     which lane is loud, so a lane is not simply mirroring the other.

It cannot establish that two simultaneous voices separate. One source is one
source; that remains the manual run.

The speaker has no relation to the array's coordinate system, so the source
bearing is not assumed - it is found with a beam scan while playing a tone,
which is the device's own per-beam energy and needs no bearing convention. The
scan is read-only and restores itself.

    python tools/p7_autoroute.py --selftest        # no device needed
    python tools/p7_autoroute.py                   # scan, then verify
    python tools/p7_autoroute.py --bearing 180     # skip the scan
"""

from __future__ import annotations

import argparse
import math
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
    CAT_ASR_OUTPUT, BeamRouter, FixedBeamController,
)

SAMPLE_RATE = 16000
CAPTURE_CHANNELS = 6
BLOCK = 1024
METRIC_WINDOW = int(0.025 * SAMPLE_RATE)

# Separation bands, in dB. Half a beamwidth of a 6.7 cm aperture is worth a lot
# at a few hundred hertz, but the source is broad and the room reflects, so the
# realistic figure is well short of the ideal.
BAND_EXCELLENT = 12.0
BAND_USABLE = 6.0
BAND_MARGINAL = 3.0

# A test tone is better than speech here. It is stationary, so the level
# statistic does not depend on where in a sentence the capture window landed,
# and it is narrow-band enough that reflections show up as level rather than
# as noise. 440 Hz rides where the array's noise floor is low.
TONE_HZ = 440.0
TONE_AMP = 0.25


def find_array() -> int:
    import sounddevice as sd

    for d in sd.query_devices():
        name = d["name"].lower()
        if ("xvf3800" in name or "respeaker" in name) and \
                d["max_input_channels"] >= CAPTURE_CHANNELS:
            return d["index"]
    raise SystemExit("XVF3800 not found among sounddevice inputs")


def find_speaker(exclude: int) -> int:
    """A host speaker, for making sound away from the array.

    The array's own USB output is deliberately not used: playing there injects
    the signal as the far-end reference, which the echo canceller then removes
    from exactly the two channels under test. A separate speaker is acoustically
    coupled and arrives through the microphones like a person would.
    """
    import sounddevice as sd

    for d in sd.query_devices():
        name = d["name"].lower()
        if d["max_output_channels"] < 1 or d["index"] == exclude:
            continue
        if "xvf3800" in name or "respeaker" in name or "usb audio" in name:
            continue
        # prefer a plain hardware device: the plugin names are resamplers and
        # routes, not speakers
        if "(" in d["name"] and "hw:" in d["name"]:
            return d["index"]
    return sd.default.device[1]


def tone(seconds: float, rate: int) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    # a slow amplitude ramp on and off, so the capture never catches a click
    env = np.minimum(1.0, np.minimum(t / 0.15, (seconds - t) / 0.15))
    return (TONE_AMP * env * np.sin(2 * math.pi * TONE_HZ * t)).astype(np.float32)


def output_rate(device: int) -> int:
    """The rate to open the speaker at.

    Host devices on this machine accept 48 kHz and refuse 16 kHz outright
    (`Invalid sample rate`, PortAudio -9997), so the tone is generated at
    whatever the device wants rather than at the capture rate. Playing 16 kHz
    straight into a 48 kHz device is not a resampling convenience - the stream
    simply will not open, and a silent failure there would leave the test
    measuring the room's own noise instead of the source, which is exactly the
    kind of false pass this file exists to avoid.
    """
    import sounddevice as sd

    want = int(sd.query_devices(device)["default_samplerate"] or 48000)
    for sr in (want, 48000, 44100, 16000):
        try:
            sd.check_output_settings(device=device, samplerate=sr, channels=2)
            return sr
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
    raise SystemExit(f"speaker device {device} accepts none of 48k/44.1k/16k")


class Player:
    """Plays the tone on repeat in a background thread.

    A failure to open is raised, not swallowed: the whole test is meaningless
    without a known source, and a background thread that dies quietly would let
    it pass on room noise.
    """

    def __init__(self, device: int, block: float = 2.0):
        import sounddevice as sd

        self.sd = sd
        self.device = device
        self.rate = output_rate(device)
        self.buf = tone(block, self.rate)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self.playing = False

    def start(self) -> None:
        if self.playing:
            return
        self.playing = True
        self._stop.clear()
        self._error = None

        def loop():
            try:
                with self.sd.OutputStream(device=self.device,
                                          samplerate=self.rate,
                                          channels=2, dtype="float32",
                                          blocksize=BLOCK) as out:
                    pos = 0
                    while not self._stop.is_set():
                        n = BLOCK
                        chunk = self.buf[pos:pos + n]
                        if chunk.size < n:
                            chunk = np.pad(chunk, (0, n - chunk.size))
                            pos = 0
                        else:
                            pos += n
                        out.write(np.repeat(chunk[:, None], 2, axis=1))
            except Exception as exc:  # noqa: BLE001 - surfaced in start()
                self._error = exc

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        time.sleep(0.6)          # let the stream open before measuring
        if self._error is not None:
            raise SystemExit(f"扬声器 {self.device} 打开失败（{self.rate} Hz）："
                             f"{self._error}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.playing = False


def peak_level(a: np.ndarray) -> np.ndarray:
    x = a.astype(np.float64)
    n = x.shape[0] // METRIC_WINDOW
    if n == 0:
        return np.sqrt(np.mean(x ** 2, axis=0))
    frames = x[: n * METRIC_WINDOW].reshape(n, METRIC_WINDOW, x.shape[1])
    return np.sqrt(np.mean(frames ** 2, axis=1)).max(axis=0)


def db(a: float, b: float) -> float:
    return 20.0 * math.log10(max(a, 1e-12) / max(b, 1e-12))


def band(value: float) -> str:
    if value >= BAND_EXCELLENT:
        return f"优秀(≥{BAND_EXCELLENT:.0f}dB)"
    if value >= BAND_USABLE:
        return f"可用({BAND_USABLE:.0f}~{BAND_EXCELLENT:.0f}dB)"
    if value >= BAND_MARGINAL:
        return f"勉强({BAND_MARGINAL:.0f}~{BAND_USABLE:.0f}dB)"
    return f"几乎无分离(<{BAND_MARGINAL:.0f}dB)"


def capture(device: int, seconds: float, control,
            chans: list[int]) -> tuple[np.ndarray, np.ndarray]:
    import sounddevice as sd

    frames = int(seconds * SAMPLE_RATE)
    got = np.zeros((frames, CAPTURE_CHANNELS), dtype=np.float32)
    energies: list[list[float]] = []
    with sd.InputStream(device=device, samplerate=SAMPLE_RATE,
                        channels=CAPTURE_CHANNELS, dtype="float32",
                        blocksize=BLOCK) as stream:
        pos = 0
        while pos < frames:
            want = min(BLOCK, frames - pos)
            chunk, _ = stream.read(want)
            got[pos:pos + want] = chunk[:want]
            pos += want
            if control is not None:
                try:
                    energies.append([float(v) for v in
                                     control.read("AEC_SPENERGY_VALUES")])
                except Exception:  # noqa: BLE001
                    pass
    e = np.array(energies) if energies else np.zeros((0, 4))
    return got, e


def measure(device: int, seconds: float, control,
            chans: list[int]) -> dict:
    got, energies = capture(device, seconds, control, chans)
    levels = peak_level(got)
    out = {"ch0": float(levels[chans[0]]), "ch1": float(levels[chans[1]]),
           "ch_db": db(float(levels[chans[0]]), float(levels[chans[1]]))}
    if energies.size:
        out["e0"] = float(energies[:, 0].max())
        out["e1"] = float(energies[:, 1].max())
        out["energy_db"] = db(out["e0"], out["e1"])
    else:
        out.update({"e0": 0.0, "e1": 0.0, "energy_db": 0.0})
    return out


def route_verdict(on_a: dict, on_b: dict) -> tuple[bool, str]:
    """Steering swapped: the whole picture must invert.

    Two things are required at once, and the second is the one that catches a
    real failure. First, the energy must follow the beam - that only checks
    SPENERGY, which the device computes itself and which would be right even if
    the routing were wrong. Second, the *capture channels* must swap with it,
    which is the part that proves OP_L and OP_R really carry the two different
    beams rather than the same one.

    On top of both, the swap is only meaningful if it is an inversion rather
    than two unrelated numbers: the sign has to flip.
    """
    e_db, c_db = on_a["energy_db"], on_b["energy_db"]
    ch_a, ch_b = on_a["ch_db"], on_b["ch_db"]
    lines = [
        f"beam1 对准声源: SPENERGY {on_a['e0']:.0f} vs {on_a['e1']:.0f} "
        f"→ {e_db:+.1f} dB",
        f"                通道 ch0 vs ch1 → {ch_a:+.1f} dB",
        f"beam2 对准声源: SPENERGY {on_b['e0']:.0f} vs {on_b['e1']:.0f} "
        f"→ {e_db:+.1f} dB",
        f"                通道 ch0 vs ch1 → {ch_b:+.1f} dB",
    ]
    e_ok = (e_db >= BAND_MARGINAL) and (on_b["energy_db"] <= -BAND_MARGINAL)
    c_ok = (ch_a >= BAND_MARGINAL) and (ch_b <= -BAND_MARGINAL)
    flipped = (ch_a > 0) != (ch_b > 0)
    sep = min(abs(e_db), abs(on_b["energy_db"]), abs(ch_a), abs(ch_b))
    lines.append(f"最小分离度 {abs(sep):.1f} dB（{band(abs(sep))}）")

    # A source we cannot hear is not a source. Without this the test can pass
    # on whatever the room happens to contain - a swing in the device's own
    # energy readings looks exactly like a real beam response, and the earlier
    # run of this file passed that way while its tone was failing to play.
    quiet_a = max(on_a["ch0"], on_a["ch1"])
    quiet_b = max(on_b["ch0"], on_b["ch1"])
    if min(quiet_a, quiet_b) < 4e-3:
        lines.append(f"❌ 采集电平过低（{min(quiet_a, quiet_b):.5f}）—— "
                     f"测试音很可能没有真正发声，本结果不成立。"
                     f"请确认扬声器音量，或用 --speaker 指定设备")
        return False, "\n     ".join(lines)

    if e_ok and c_ok and flipped:
        lines.append("✅ 两个独立读数都随波束互换而翻转 —— "
                     "lane 与波束的绑定成立，OP_L/OP_R 确实各带一路波束")
        return True, "\n     ".join(lines)
    if e_ok and not c_ok:
        lines.append("❌ SPENERGY 跟随了波束，但**采集通道没有** —— "
                     "路由没有把两条波束分到两个通道，须查 OP_L/OP_R 落点")
        return False, "\n     ".join(lines)
    if c_ok and not e_ok:
        lines.append("❌ 采集通道互换了，但 SPENERGY 没有 —— "
                     "波束指向可能未生效（SONOFF/GATING 状态？）")
        return False, "\n     ".join(lines)
    if not flipped:
        lines.append("❌ 通道没有互换 —— 两个通道可能装的是同一路信号")
        return False, "\n     ".join(lines)
    lines.append(f"❌ 分离度过低（{abs(sep):.1f} dB）—— "
                 f"声源可能离两个波束等距，或音量太小")
    return False, "\n     ".join(lines)


def _selftest() -> int:
    """Exercise the verdict against synthetic numbers."""
    print("判定逻辑自检（合成数据，不碰设备）")

    def s(e0, e1, c0, c1):
        return {"e0": e0, "e1": e1, "ch0": c0, "ch1": c1,
                "energy_db": db(e0, e1), "ch_db": db(c0, c1)}

    cases = [
        ("正常互换", s(3e6, 2e5, .06, .003), s(2e5, 3e6, .003, .06), True),
        ("能量对但通道不换", s(3e6, 2e5, .06, .06), s(2e5, 3e6, .06, .06), False),
        ("通道换但能量不动", s(3e6, 3e6, .06, .003), s(3e6, 3e6, .003, .06), False),
        ("分离度不足", s(3e6, 2.5e6, .06, .05), s(2.5e6, 3e6, .05, .06), False),
        ("通道装同一路", s(3e6, 2e5, .05, .05), s(2e5, 3e6, .05, .05), False),
    ]
    ok = True
    for name, a, b, want in cases:
        got, _ = route_verdict(a, b)
        flag = "ok " if got == want else "BAD"
        if got != want:
            ok = False
        print(f"  {flag} {name} → {got}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bearing", type=float, default=None,
                    help="source bearing in degrees; skip the scan if known")
    ap.add_argument("--off", type=float, default=None,
                    help="the bearing to steer the other beam to "
                         "(default: source bearing + 180)")
    ap.add_argument("--window", type=float, default=4.0)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--speaker", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    from realtime_subtitle.usb_device import ReSpeaker

    try:
        control = ReSpeaker()
    except Exception as exc:  # noqa: BLE001
        print(f"[autoroute] 控制面不可用：{exc}")
        return 1

    array = args.device if args.device is not None else find_array()
    speaker = args.speaker if args.speaker is not None else find_speaker(array)
    router = BeamRouter(device=control)
    steer = FixedBeamController(control)
    orig_route = list(router.recon()["op_all"])

    print(f"[autoroute] 阵列采集设备 {array}，扬声器设备 {speaker}")
    print(f"[autoroute] 原路由 {orig_route}")
    player = Player(speaker)

    try:
        # --- find the source's bearing with the device's own energy reading ---
        bearing = args.bearing
        if bearing is None:
            print(f"[autoroute] 播放 {TONE_HZ:.0f}Hz 测试音，扫描声源方位 ...")
            player.start()
            scan = steer.scan(step_deg=45.0, settle=0.5)
            player.stop()
            if not scan.get("ok"):
                print(f"[autoroute] 扫描失败：{scan.get('reason')}")
                return 1
            bearing = scan["peak"]["deg"]
            print(f"[autoroute] 声源方位 {bearing:.0f}° "
                  f"(E={scan['peak']['energy']:.0f})")
            if scan.get("second"):
                print(f"[autoroute] 次峰 {scan['second']['deg']:.0f}° "
                      f"(E={scan['second']['energy']:.0f}) —— 可能是反射")
        else:
            print(f"[autoroute] 使用给定方位 {bearing:.0f}°（跳过扫描）")

        off = args.off if args.off is not None else (bearing + 180.0) % 360.0
        print(f"[autoroute] 对比: beam1@{bearing:.0f}° vs beam2@{off:.0f}°")
        print(f"[autoroute] 路由 OP_L -> (7,0), OP_R -> (7,1)（双轨实际配置）")
        for slot, beam in (("L", 0), ("R", 1)):
            r = router.route_slot(slot, CAT_ASR_OUTPUT, beam)
            if not r.get("ok"):
                print(f"[autoroute] 路由写入失败：{r}")
                return 1
        print(f"[autoroute] 路由现已 {router.recon()['op_all']}")

        results = {}
        player.start()
        for tag, a1, a2 in (("A", bearing, off), ("B", off, bearing)):
            r = steer.steer(a1, a2, gating=False)
            if not r.get("ok"):
                print(f"[autoroute] steer 失败：{r}")
                return 1
            time.sleep(0.6)                    # let the beamformer settle
            m = measure(array, args.window, control, [0, 1])
            results[tag] = m
            print(f"  {tag}: beam1={a1:.0f}° e0={m['e0']:.0f} e1={m['e1']:.0f} "
                  f"ch0={m['ch0']:.5f} ch1={m['ch1']:.5f}")
        player.stop()

        print()
        print("=" * 68)
        ok, why = route_verdict(results["A"], results["B"])
        print(("✅ " if ok else "❌ ") + why)
    finally:
        player.stop()
        print(f"\n[autoroute] 恢复波束: {steer.restore()}")
        print(f"[autoroute] 恢复路由: {router.restore()}")
        after = router.recon()["op_all"]
        print(f"[autoroute] 路由现已 {after}  "
              f"{'✅ 与原值一致' if after == orig_route else '❌ 未恢复!'}")
        control.close()

    print("""
本测试证明了什么、没证明什么

  证明了：lane 与波束的绑定、OP_L/OP_R 各带一路波束、两条 lane 互相独立
          （互换波束会互换哪个通道更响，且两个独立读数一致）

  没证明：两个人**同时**说话能否分离。只有一个声源，无法构成那个场景。
          那需要真人版：
            python -u -m realtime_subtitle.main --beam-lanes AZ0 AZ1 --debug-gate
            python -u tools/p7_verify.py --lanes AZ0 AZ1

  另注：这里的分离度是**窄带单音**在扬声器方位上的结果，会比真实语音
        （宽带、有起伏、房间反射）乐观。它验证的是管道，不是可用的分离度。
""")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())