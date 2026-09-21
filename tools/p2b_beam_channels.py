"""P2b: does the ASR-output category really expose beams by index?

The dual-lane design routes OP_L to (7,0) and OP_R to (7,1), on the reading
that category 7 is the beamformer output and its source is a beam index. The
device ships channel 1 as (7,3), and 3 is auto-select in the same ordering
AEC_AZIMUTH_VALUES uses, which is suggestive but not proof: if (7,0) and (7,1)
turned out to be the same beam, both lanes would receive one voice and the
whole design would be built on nothing.

Telling two beams apart needs the beams to be aimed differently, so the test
steers them and then swaps them:

    steer(beam1 -> theta, beam2 -> theta+180)   clap   measure ch0 vs ch1
    steer(beam1 -> theta+180, beam2 -> theta)   clap   measure ch0 vs ch1

With gating off both beams stay live, so a talker reaches both and absolute
levels prove nothing. What does prove something is that the *comparison*
reverses: whichever channel is louder must swap when the beams swap. If it does
not, the two channels are not tracking the two beams, and the finding is
negative.

The talker's bearing comes from a beam scan, which reads the device's own
SPENERGY - the one bearing source with no mirror ambiguity, because it is a
single scalar per beam rather than a set of relative delays.

Read-only with respect to flash. Both the routing and the steering are
snapshotted before any write and restored on every exit path.

    python tools/p2b_beam_channels.py --theta 180
    python tools/p2b_beam_channels.py --selftest
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realtime_subtitle.beam_router import (      # noqa: E402
    CAT_ASR_OUTPUT, SLOT_CHANNEL, BeamRouter, FixedBeamController,
)

SAMPLE_RATE = 16000
CAPTURE_CHANNELS = 6
OUT_CHANNELS = [0, 1]           # the two programmable outputs
BEAM_SLOTS = ["L", "R"]         # which slot carries which beam index
METRIC_WINDOW = int(0.025 * SAMPLE_RATE)
BLOCK = 1024


def find_array() -> int:
    import sounddevice as sd

    for d in sd.query_devices():
        name = d["name"].lower()
        if ("xvf3800" in name or "respeaker" in name) and \
                d["max_input_channels"] >= CAPTURE_CHANNELS:
            return d["index"]
    raise SystemExit("XVF3800 not found among sounddevice inputs")


def peak_level(a: np.ndarray) -> np.ndarray:
    """Per-channel maximum of short-window RMS (transient-proof, see p2a)."""
    x = a.astype(np.float64)
    n = x.shape[0] // METRIC_WINDOW
    if n == 0:
        return np.sqrt(np.mean(x ** 2, axis=0))
    frames = x[: n * METRIC_WINDOW].reshape(n, METRIC_WINDOW, x.shape[1])
    return np.sqrt(np.mean(frames ** 2, axis=1)).max(axis=0)


def record_with_energy(device_obj, dev_idx: int, seconds: float
                       ) -> tuple[np.ndarray, list[list[float]]]:
    """Record while polling SPENERGY, so the cross-check has data during sound.

    Reading SPENERGY before the stimulus measures silence: the per-beam speech
    energies are zero whenever nobody is talking, which is exactly what the
    first run reported and why its cross-check said nothing. Polling during the
    capture gives the one reading that is independent of the audio path.
    """
    import sounddevice as sd

    frames = int(seconds * SAMPLE_RATE)
    got = np.zeros((frames, CAPTURE_CHANNELS), dtype=np.float32)
    energies: list[list[float]] = []
    with sd.InputStream(device=dev_idx, samplerate=SAMPLE_RATE,
                        channels=CAPTURE_CHANNELS, dtype="float32",
                        blocksize=BLOCK) as stream:
        pos = 0
        while pos < frames:
            want = min(BLOCK, frames - pos)
            chunk, _ = stream.read(want)
            got[pos:pos + want] = chunk[:want]
            pos += want
            try:
                energies.append([float(v) for v in
                                 device_obj.read("AEC_SPENERGY_VALUES")])
            except Exception:  # noqa: BLE001 - polling is best effort
                pass
    return got, energies


def record(device: int, seconds: float) -> np.ndarray:
    import sounddevice as sd

    frames = int(seconds * SAMPLE_RATE)
    got = np.zeros((frames, CAPTURE_CHANNELS), dtype=np.float32)
    with sd.InputStream(device=device, samplerate=SAMPLE_RATE,
                        channels=CAPTURE_CHANNELS, dtype="float32",
                        blocksize=BLOCK) as stream:
        pos = 0
        while pos < frames:
            want = min(BLOCK, frames - pos)
            chunk, _ = stream.read(want)
            got[pos:pos + want] = chunk[:want]
            pos += want
    return got


def pattern_verdict(winners: list[int]) -> tuple[bool, str]:
    """The winner must alternate every time the beams are swapped.

    A single swap is weak evidence - with two channels and no effect at all,
    the louder one flips half the time by chance. Alternation is what makes
    chance implausible, and the more segments the stronger it gets: three
    alternations happen by chance one time in four, four in eight.

    Note that "returns to the starting channel" is *not* the condition. That
    only holds for an odd number of segments; an even-length A-B-A-B run
    legitimately ends on the other channel. Testing it would reject a good
    four-segment result, which is exactly what the first version of this
    function did.
    """
    n = len(winners)
    if n < 3:
        return False, "至少需要三段（A-B-A）才能把偶然压到 1/4 以下"
    alternates = all(winners[i] != winners[i + 1] for i in range(n - 1))
    by_chance = 0.5 ** (n - 1)
    seq = "→".join(f"ch{OUT_CHANNELS[w]}" for w in winners)
    if alternates:
        return True, (f"序列 {seq} 每次交换都**交替** → 通道跟随波束，"
                      f"cat 7 的 source 是波束索引（这种规律偶然出现的概率 "
                      f"{by_chance:.0%}）")
    if len(set(winners)) == 1:
        return False, (f"序列 {seq}：三次都是 ch{OUT_CHANNELS[winners[0]]} 更响 → "
                       f"ch0/ch1 未分别跟随两个波束，假设不成立")
    return False, (f"序列 {seq} 不呈交替 → 结果不稳定，噪声或旁瓣可能在主导；"
                   f"建议把两波束指向差减小到 90° 重试")


def modulate(x: np.ndarray, seg_len: int) -> np.ndarray:
    """Per-segment short-window peak level for every channel.

    Splitting one continuous capture into segments is what removes the
    operator from the experiment. Comparing one clapping session against
    another assumes the talker held still, and three hand-made sessions showed
    exactly how false that is: the two identical configurations disagreed with
    each other and the levels fell by a factor of ten across the run. Inside a
    single capture any drift is slow and shared, while a beam swap is a step.
    """
    level = peak_level(x)
    n = x.shape[0] // seg_len
    if n == 0:
        return np.zeros((0, x.shape[1]))
    rows = []
    for i in range(n):
        rows.append(peak_level(x[i * seg_len:(i + 1) * seg_len]))
    return np.array(rows)


def modulation_verdict(per_seg: np.ndarray, pattern: list[int]
                       ) -> tuple[bool, str]:
    """Does the ch0-vs-ch1 balance follow the beam pattern?

    `pattern` is which beam index faces the talker in each segment (0 or 1).
    The audio balance ch0/ch1 in dB is compared against it: a channel that
    carries beam b should be louder in exactly the segments where b faces the
    talker. Correlation is used rather than the individual winners, because a
    monotone drift common to both channels cancels in the difference instead
    of deciding a segment.
    """
    if per_seg.shape[0] < 4:
        return False, "段数太少，无法把调制作成证据"
    bal = 20 * np.log10((per_seg[:, 0] + 1e-12) / (per_seg[:, 1] + 1e-12))
    pat = np.array(pattern[:len(bal)], dtype=float)
    det = "  ".join(f"{p:.0f}:{v:+.1f}dB" for p, v in zip(pattern, bal))

    # Effect size, not correlation alone. A modulation that follows the beam
    # switch is evidence, but only if it is bigger than the scatter between
    # segments that had the *same* setting - otherwise the "term" is just
    # noise that happened to land in phase. An absolute dB threshold would
    # not distinguish the two: a small but perfectly repeating 1 dB swing is
    # real, while a large 6 dB swing that also varies by 6 dB within a group
    # is not.
    g0 = bal[pat == 0]
    g1 = bal[pat == 1]
    if g0.size == 0 or g1.size == 0:
        return False, "波束模式没有两个取值，无法比较"
    effect = abs(float(g0.mean() - g1.mean()))
    spread = float(np.sqrt((g0.var() + g1.var()) / 2)) if g0.size > 1 else 0.0
    if effect < max(1.0, 1.2 * spread):
        return False, (f"ch0/ch1 平衡不跟随波束切换：两组均值差 {effect:.1f} dB，"
                       f"而组内散布 {spread:.1f} dB\n"
                       f"     逐段(波束:平衡) {det}\n"
                       f"     → 两个 slot 未分到不同波束，cat 7 source 假设不成立")
    which = "ch0" if g0.mean() > g1.mean() else "ch1"
    beam = "beam1(7,0)" if which == "ch0" else "beam2(7,1)"
    return True, (f"ch0/ch1 平衡跟随波束切换：两组均值差 {effect:.1f} dB，"
                  f"组内散布仅 {spread:.1f} dB\n"
                  f"     逐段(波束:平衡) {det}\n"
                  f"     → cat 7 的 source **是**波束索引；{beam} 落在 {which}")


def verdict(louder_first: int, louder_second: int) -> tuple[bool, str]:
    """Did the louder channel swap when the beams swapped?

    `louder_N` is the index into OUT_CHANNELS that won in run N. When the beams
    are exchanged, the channel carrying beam 1 in the first run should lose in
    the second - i.e. the winner has to change.
    """
    if louder_first == louder_second:
        return False, (f"两个通道的相对响度**没有随波束交换而翻转**"
                       f"（两次都是 ch{OUT_CHANNELS[louder_first]} 更响）。"
                       f"说明 ch0/ch1 并未分别跟随两个波束——"
                       f"cat 7 的 source **不是**波束索引，"
                       f"或两个 slot 拿到的是同一路信号。")
    return True, (f"相对响度随波束交换而翻转"
                  f"（先 ch{OUT_CHANNELS[louder_first]}，后 "
                  f"ch{OUT_CHANNELS[louder_second]}）。"
                  f"说明 cat 7 的 source **是**波束索引，"
                  f"两个 slot 确实拿到不同波束。")


def _selftest() -> int:
    """Exercise the verdict rules before trusting them on hardware."""
    ok = True

    # modulation: ch0 louder when beam1 faces the talker, alternating
    def seg(bal_db: float) -> np.ndarray:
        row = np.zeros(CAPTURE_CHANNELS)
        row[0] = 10 ** (bal_db / 40)      # amplitude, squared below
        row[1] = 10 ** (-bal_db / 40)
        return row ** 2

    cases = [
        ([(0, +6), (1, -6), (0, +6), (1, -6)], True,
         "平衡随波束交替 → 成立"),
        ([(0, -6), (1, +6), (0, -6), (1, +6)], True,
         "反向交替也成立（只是归属相反）"),
        ([(0, +1), (1, -1), (0, +1), (1, -1)], True,
         "幅度小但完美重复，组内散布为 0 → 成立（噪声不会规律同相）"),
        ([(0, +6), (1, +6), (0, +6), (1, +6)], False,
         "不跟随切换 → 否决"),
    ]
    for data, want, why in cases:
        per_seg = np.array([seg(b) for _, b in data])
        pattern = [p for p, _ in data]
        got, _ = modulation_verdict(per_seg, pattern)
        flag = "ok " if got == want else "BAD"
        if got != want:
            ok = False
        print(f"  {flag} 调制 {pattern} → {got}  ({why})")

    for ws, want, why in [([0, 1, 0], True, "交替"),
                          ([0, 1, 1], False, "未交替"),
                          ([0, 1, 0, 1], True, "四段交替")]:
        got, _ = pattern_verdict(ws)
        flag = "ok " if got == want else "BAD"
        if got != want:
            ok = False
        print(f"  {flag} 段序列 {ws} → {got}  ({why})")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theta", type=float, required=False,
                    help="the talker's bearing in degrees, from --beam-scan")
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--segment", type=float, default=2.0,
                    help="seconds per beam setting inside the single capture "
                         "(default 2; shorter means more swaps and less chance "
                         "to drift)")
    ap.add_argument("--segments", type=int, default=8,
                    help="number of beam settings in the capture (default 8, "
                         "alternating)")
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.theta is None:
        print("需要 --theta（先用 --beam-scan 扫出说话人方位）")
        return 2

    from realtime_subtitle.usb_device import ReSpeaker

    device = ReSpeaker()
    router = BeamRouter(device)
    beams = FixedBeamController(device)
    dev_idx = args.device if args.device is not None else find_array()

    orig_route = list(router.recon()["op_all"])
    theta = args.theta % 360.0
    print(f"[p2b] 说话人方位 θ = {theta:.0f}°（原路由 {orig_route}）")
    print("[p2b] 方法：**一次连续录音**内反复切换波束（调制），比较段内 ch0/ch1")
    print("      平衡。段内比较不需要你站着不动——两次独立拍手曾因此结论矛盾。")

    print("\n[p2b] 配置路由 OP_L=(7,0), OP_R=(7,1) ...")
    print("     ", router.route_slot("L", CAT_ASR_OUTPUT, 0))
    print("     ", router.route_slot("R", CAT_ASR_OUTPUT, 1))

    seg_len = int(args.segment * SAMPLE_RATE)
    n_seg = args.segments
    total = n_seg * args.segment
    pattern = [i % 2 for i in range(n_seg)]      # 0 = beam1 faces talker

    print(f"[p2b] 将切换 {n_seg} 段，每段 {args.segment:.0f}s，共 {total:.0f}s")
    print("[p2b] 请**全程持续拍手/说话**，我会在段边界切换波束")
    print(f"[p2b] 3 秒后开始 ...", flush=True)
    time.sleep(3.0)

    energies: list[list[float]] = []
    try:
        import sounddevice as sd

        frames = int(total * SAMPLE_RATE)
        got = np.zeros((frames, CAPTURE_CHANNELS), dtype=np.float32)
        with sd.InputStream(device=dev_idx, samplerate=SAMPLE_RATE,
                            channels=CAPTURE_CHANNELS, dtype="float32",
                            blocksize=BLOCK) as stream:
            pos = 0
            seg_i = -1
            while pos < frames:
                if pos // seg_len != seg_i:
                    seg_i = pos // seg_len
                    target_beam = args.theta if pattern[seg_i] == 0 else args.theta + 180
                    other = args.theta + 180 if pattern[seg_i] == 0 else args.theta
                    r = beams.steer(target_beam, other, gating=False)
                    print(f"    段{seg_i + 1}: beam1={target_beam:.0f}° "
                          f"beam2={other:.0f}° {'ok' if r.get('ok') else 'FAIL'}"
                          f"  {'(beam1 对你)' if pattern[seg_i] == 0 else '(beam2 对你)'}",
                          flush=True)
                want = min(BLOCK, frames - pos)
                chunk, _ = stream.read(want)
                got[pos:pos + want] = chunk[:want]
                pos += want
                try:
                    energies.append([float(v) for v in
                                     device.read("AEC_SPENERGY_VALUES")])
                except Exception:  # noqa: BLE001
                    pass
        print("    录音结束 ■", flush=True)
    finally:
        print(f"\n[p2b] 恢复波束: {beams.restore()}")
        print(f"[p2b] 恢复路由: {router.restore()}")
        after = router.recon()["op_all"]
        print(f"[p2b] 路由现已: {after}   "
              f"{'✅ 与原值一致' if after == orig_route else '❌ 未恢复!'}")
        device.close()

    per_seg = modulate(got, seg_len)
    # trim the first segment: the beams are still settling through it
    if per_seg.shape[0] > 2:
        per_seg = per_seg[1:]
        pattern_used = pattern[1:]
    else:
        pattern_used = pattern
    print(f"\n各段 ch0/ch1 短窗峰值（首段已丢弃，波束尚在稳定）:")
    print(f"  {'段':>3} {'beam 对你':>10} {'ch0':>10} {'ch1':>10}  {'平衡dB':>8}")
    for i, row in enumerate(per_seg):
        bal = 20 * np.log10((row[0] + 1e-12) / (row[1] + 1e-12))
        who = "beam1" if pattern_used[i] == 0 else "beam2"
        print(f"  {i + 1:>3} {who:>10} {row[0]:>10.5f} {row[1]:>10.5f} "
              f"{bal:>+8.1f}")

    if energies:
        e = np.array(energies)
        mid = len(e) // 2
        print(f"\nSPENERGY 前半段均值 beam1={e[:mid, 0].mean():.0f} "
              f"beam2={e[:mid, 1].mean():.0f}  "
              f"后半段 beam1={e[mid:, 0].mean():.0f} "
              f"beam2={e[mid:, 1].mean():.0f}")

    print("\n" + "=" * 72)
    ok, why = modulation_verdict(per_seg, pattern_used)
    print(("✅ " if ok else "❌ ") + why)
    print("\n局限:")
    print("  - 只证明两 slot 分到**不同**波束；具体归属由上式直接给出")
    print("  - 若你对位置不敏感（两个波束都收到你等量），平衡不跟随切换，")
    print("    该判据会否定——此时需把两波束指向差调到 90° 再试")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())