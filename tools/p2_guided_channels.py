"""P2: guided identification of which capture channel is which microphone.

The six-channel layout is known at the block level - two processed outputs then
four raw mics - but the four raw channels are only *assumed* to follow the
device's source order, which is what `--audio-source mic0..mic3` relies on. A
channel index that is silently off by one would look like a working system that
puts the wrong microphone's audio into a lane, so it is worth ten minutes of
somebody walking around the array.

The arrays's own geometry makes this measurable: the four microphones sit at
the corners of a square (45 degrees apart in bearing), so a sound source placed
near one corner is measurably loudest in that corner's channel. Move the source
a quarter turn at a time and the loudest raw channel should advance by one each
step; the permutation that shows up *is* the source-order mapping.

Deliberately orientation-free. Nothing here needs to know which way the array
is facing or which corner is "front", because that is unknowable without a
calibrated source. The operator is asked only to move a quarter turn in the
same direction each time, and the answer is read off the *change* in the
pattern rather than any absolute bearing. The device's own DOA reading is
printed alongside as a cross-check, but it is not trusted: it is known to drift
in a quiet room, so it never decides the result.

Recording runs on one stream for the whole session. Reopening a stream per step
would re-introduce the two-second settling transient each time, and the array's
levels right after a stream opens are loud enough to be mistaken for a sound
source.

Read-only with respect to the device: the control plane is used only to print
DOA and geometry. No parameters are written.

    python tools/p2_guided_channels.py                 # 4 positions
    python tools/p2_guided_channels.py --nosep 6       # 6 turns for confirmation
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

SAMPLE_RATE = 16000
CAPTURE_CHANNELS = 6
RAW_CHANNELS = [2, 3, 4, 5]        # the four raw mic channels
BLOCK = 1024

# Metric window for the level statistic. A clap lasts a few milliseconds, so
# averaging over the whole capture window dilutes it into the noise: a 5 ms
# transient spread over 5 s loses ~30 dB and the channels then differ by only a
# decibel or two, which is exactly what the first attempt measured. Taking the
# maximum of short-window RMS keeps the transient's amplitude, which is the
# quantity that actually varies with which microphone is nearest.
METRIC_WINDOW = int(0.025 * SAMPLE_RATE)      # 25 ms

# mic positions as the device itself reports them, used only to describe the
# expected pattern in the output
MIC_BEARINGS = {0: 315.0, 1: 45.0, 2: 135.0, 3: 225.0}


def find_array() -> int:
    import sounddevice as sd

    for d in sd.query_devices():
        name = d["name"].lower()
        if ("xvf3800" in name or "respeaker" in name) and \
                d["max_input_channels"] >= CAPTURE_CHANNELS:
            return d["index"]
    raise SystemExit("XVF3800 not found among sounddevice inputs")


def rms(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return np.zeros(CAPTURE_CHANNELS)
    return np.sqrt(np.mean(a.astype(np.float64) ** 2, axis=0))


# ---------------------------------------------------------------------------
# time-difference-of-arrival analysis
#
# Amplitude turned out to be the wrong instrument. In a far field every mic
# hears a source at nearly the same level - at 1 m the four mics differ by
# under a millimetre of path length - so "which channel is loudest" mostly
# ranks the room. What does not blur with distance is *when* each mic hears the
# source: 47 mm of array radius means up to 275 us of spread, which survives
# perfectly at any distance.
# ---------------------------------------------------------------------------
SOUND_C_PER_US = 343.0 / 1e6      # metres travelled per microsecond


def onsets(a: np.ndarray, threshold: float = 0.25,
           refract: int = 400) -> list[int]:
    """Sample indices where a transient starts, from the broadband envelope."""
    env = np.sqrt(np.mean(a.astype(np.float64) ** 2, axis=1))
    env = np.convolve(env, np.ones(16) / 16, mode="same")
    thr = env.max() * threshold
    out: list[int] = []
    i = 0
    while i < len(env):
        if env[i] > thr:
            j = i
            while j < len(env) and env[j] > thr:
                j += 1
            out.append(i + int(np.argmax(env[i:j])))
            i = j + refract
        else:
            i += 1
    return out


def _lag(ref: np.ndarray, sig: np.ndarray, max_lag: int = 40) -> float:
    """Sub-sample lag of sig behind ref, by cross-correlation + parabola fit."""
    n = min(len(ref), len(sig))
    r = ref[:n] - ref[:n].mean()
    s = sig[:n] - sig[:n].mean()

    def cc(lag: int) -> float:
        if lag < 0:
            return float(np.dot(r[-lag:], s[:n + lag]))
        if lag > 0:
            return float(np.dot(r[:n - lag], s[lag:]))
        return float(np.dot(r, s))

    best = max(range(-max_lag, max_lag + 1), key=cc)
    if -max_lag < best < max_lag:
        y0, y1, y2 = cc(best - 1), cc(best), cc(best + 1)
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            best += 0.5 * (y0 - y2) / den
    return best / SAMPLE_RATE * 1e6


def channel_delays(x: np.ndarray, ref_slot: int = 0) -> tuple[np.ndarray, int]:
    """Mean delay of each raw channel relative to `ref_slot`, in microseconds.

    The reference is fixed across every transient on purpose. Measuring each
    event against its own loudest channel gives every event a different origin,
    and averaging those differences is meaningless - it was the first thing
    this analysis got wrong.
    """
    events = onsets(x)
    rows = []
    for s in events:
        lo, hi = s - 32, s + 128
        if lo < 0 or hi > x.shape[0]:
            continue
        seg = x[lo:hi][:, RAW_CHANNELS].astype(np.float64)
        if np.max(np.abs(seg)) < 5e-3:      # too quiet to time
            continue
        rows.append([_lag(seg[:, ref_slot], seg[:, c]) for c in range(4)])
    if not rows:
        return np.zeros(4), 0
    return np.mean(rows, axis=0), len(rows)


def mic_angles_rad() -> np.ndarray:
    return np.array([MIC_BEARINGS[i] for i in sorted(MIC_BEARINGS)]) \
        * np.pi / 180.0


def fit_bearing(delays: np.ndarray, angles: np.ndarray,
                step: float = 0.5) -> tuple[float, float]:
    """Source bearing and residual for one delay vector under `angles`.

    Model: a plane wave from bearing theta reaches mic i at
    t_i = -(R/c) cos(theta - beta_i), so the delay relative to mic 0 depends
    only on the differences of those cosines.
    """
    radius = 0.0333
    best = (0.0, float("inf"))
    for deg in np.arange(0.0, 360.0, step):
        th = deg * np.pi / 180.0
        pred = -(radius / SOUND_C_PER_US) * (
            np.cos(th - angles) - np.cos(th - angles[0]))
        err = float(np.sum((pred - delays) ** 2))
        if err < best[1]:
            best = (deg, err)
    return best


def rank_permutations(delays: np.ndarray) -> list[tuple[float, tuple]]:
    """Every channel-to-microphone assignment, ordered by fit residual.

    A square array is symmetric under rotation, so the fit cannot tell one
    corner from another: cyclic shifts of an assignment are indistinguishable
    from rotating the assumed array, and the fitted bearing absorbs them. What
    it *can* tell apart is ordering - whether consecutive channels are
    neighbouring mics. A good fit therefore arrives as a tie-set of the eight
    order-preserving assignments, not as a single winner.
    """
    import itertools

    angles = mic_angles_rad()
    out = []
    for perm in itertools.permutations(range(4)):
        total = 0.0
        for d in delays:
            total += fit_bearing(d, angles[list(perm)])[1]
        out.append((float(np.sqrt(total / (len(delays) * 4))), perm))
    return sorted(out, key=lambda t: t[0])


def ring_order_perms() -> set[tuple]:
    """The assignments where consecutive channels are neighbouring mics."""
    import itertools

    keep = set()
    for p in itertools.permutations(range(4)):
        diffs = {(p[(i + 1) % 4] - p[i]) % 4 for i in range(4)}
        if diffs in ({1}, {3}):
            keep.add(p)
    return keep


def peak_level(a: np.ndarray, window: int = METRIC_WINDOW) -> np.ndarray:
    """Per-channel maximum of short-window RMS over the capture.

    This is the statistic that survives a transient source. Plain RMS over the
    whole window answers "how much energy in total", which for a clap is
    dominated by the length of the window rather than by proximity; the maximum
    of short-window RMS answers "how loud did it get", which is what proximity
    actually controls.
    """
    if a.size < window:
        return rms(a)
    x = a.astype(np.float64)
    n = x.shape[0] // window
    if n == 0:
        return rms(a)
    frames = x[: n * window].reshape(n, window, x.shape[1])
    per_window = np.sqrt(np.mean(frames ** 2, axis=1))     # (n, channels)
    return per_window.max(axis=0)


def drain(stream, seconds: float, keep: list | None = None) -> None:
    """Read `seconds` of audio, optionally appending it to `keep`."""
    n = int(seconds * SAMPLE_RATE)
    got = 0
    while got < n:
        want = min(BLOCK, n - got)
        chunk, _ = stream.read(want)
        if keep is not None:
            keep.append(chunk[:want].copy())
        got += want


def say(msg: str) -> None:
    print(msg, flush=True)


def analyse(steps: list[np.ndarray], base_rms: np.ndarray) -> dict:
    """Turn per-position captures into a verdict, as data.

    Separated from printing so the decision logic can be exercised against
    synthetic patterns - the alternative is debugging it by walking around the
    array, which is both slow and unrepeatable.

    The signal is the *rotation*: with the four mics at the corners of a square,
    moving a source a quarter turn must advance the loudest raw channel by
    exactly one, cyclically. A constant step of +1 means the channels follow
    bearing order in the source order; -1 means the same but reversed; anything
    else means the channels are not simply consecutive mics.
    """
    gains, loudest = [], []
    for x in steps:
        live = peak_level(x)
        g = [live[c] / max(base_rms[c], 1e-9) for c in RAW_CHANNELS]
        gains.append(g)
        loudest.append(RAW_CHANNELS[int(np.argmax(g))])
    positions = [RAW_CHANNELS.index(c) for c in loudest]
    n = len(positions)
    diffs = ([(positions[(i + 1) % n] - positions[i]) % len(RAW_CHANNELS)
              for i in range(n)] if n > 1 else [])
    verdict = "forward" if diffs and all(d == 1 for d in diffs) else \
              "reverse" if diffs and all(d == len(RAW_CHANNELS) - 1 for d in diffs) \
              else "repeat" if len(set(loudest)) < len(loudest) else "irregular"
    return {"gains": gains, "loudest": loudest, "positions": positions,
            "diffs": diffs, "verdict": verdict}


def _selftest() -> int:
    """Check analyse() against synthetic patterns before trusting it."""
    base = np.ones(CAPTURE_CHANNELS)

    def step(active: int, level: float = 30.0) -> np.ndarray:
        a = np.zeros((SAMPLE_RATE, CAPTURE_CHANNELS), dtype=np.float32)
        a[:, active] = level      # a constant is fine: only the level matters
        return a

    cases = [
        ("forward", [step(c) for c in [2, 3, 4, 5]]),
        # a different starting corner, same rotation
        ("forward", [step(c) for c in [4, 5, 2, 3]]),
        # a quarter turn the other way
        ("reverse", [step(c) for c in [5, 4, 3, 2]]),
        # one channel loudest twice: the operator likely did not really move,
        # so this is reported as "repeat" rather than as a broken mapping
        ("repeat", [step(c) for c in [2, 3, 2, 5]]),
        # no repeats but no consistent step either: the channels are not
        # consecutive mics
        ("irregular", [step(c) for c in [2, 3, 5, 4]]),
    ]
    ok = True
    for want, steps in cases:
        got = analyse(steps, base)["verdict"]
        flag = "ok " if got == want else "BAD"
        if got != want:
            ok = False
        print(f"  {flag} expected {want:9s} got {got}")
    return 0 if ok else 1


def read_doa(device) -> str:
    """The device's own bearing estimate, as a cross-check only."""
    try:
        vals = [float(v) for v in device.read("AEC_AZIMUTH_VALUES")]
    except Exception as exc:  # noqa: BLE001
        return f"(unavailable: {exc})"
    parts = [f"{v * 180 / np.pi:6.1f}°" if np.isfinite(v) else "   nan"
             for v in vals]
    return "beam1/beam2/free/auto = " + " ".join(parts)


def tdoa_report(steps: list[np.ndarray]) -> int:
    """Decide the channel order from arrival times, not levels."""
    print("=" * 72)
    print("TDOA 分析（用到达时差，不用电平）")
    print("=" * 72)
    delays, counts = [], []
    for i, x in enumerate(steps):
        d, n = channel_delays(x)
        delays.append(d)
        counts.append(n)
        print(f"位置{i+1}: {n:2d} 个事件  "
              + " ".join(f"ch{RAW_CHANNELS[c]}={d[c]:+7.1f}"
                          for c in range(4)) + "  µs")
    delays = np.array(delays)
    total_events = sum(counts)
    if total_events < 8:
        print(f"\n尚不能结论: 只找到 {total_events} 个可用于定时的事件；"
              f"请重跑并持续拍手/敲击，或靠近阵列一些。")
        return 1

    ranked = rank_permutations(delays)
    best = ranked[0][0]
    tie = [p for err, p in ranked if err <= best * 1.10 + 1.0]
    ring = ring_order_perms()
    print(f"\n最佳拟合 RMS = {best:.1f} µs（阵列理论上限 275 µs）")
    print(f"并列最优的排列有 {len(tie)} 种（方形阵列旋转对称，循环移位不可区分）")

    ident = (0, 1, 2, 3)
    ident_err = next(err for err, p in ranked if p == ident)
    scrambles = [(err, p) for err, p in ranked if p not in ring]
    print(f"\n  ch2..ch5 = mic0..mic3（恒等）      RMS = {ident_err:5.1f} µs")
    if scrambles:
        print(f"  最优的**乱序**排列                RMS = {scrambles[0][0]:5.1f} µs")

    # The decision that actually matters: are consecutive channels adjacent
    # mics? Rotation is unobservable here, so that - not which corner is
    # "first" - is the claim worth testing.
    if ident in tie:
        print("\n结论: 恒等排列在可接受范围内 → **即使不假设任何旋转**，")
        print("      ch2..ch5 也确实是**相邻麦克风**。因此: ")
        print("        - `--audio-source mic0..mic3` 的通道号映射可用")
        print("        - 具体哪个通道对应哪个物理角，本质上不可由本次测量区分")
        print("          （需一个已知方位的声源才能定；对字幕用途无影响）")
        if scrambles and scrambles[0][0] > best * 1.5:
            print(f"      而乱序假设被强烈否定（RMS {scrambles[0][0]:.0f} vs "
                  f"{best:.0f} µs）→ 四个通道不是任意顺序，是按环绕序排列")
        return 0

    print("\n结论: 恒等排列的拟合明显差于最优排列 →")
    print("      ch2..ch5 **未必**就是 mic0..mic3 的环绕序。")
    print("      最优几个候选（RMS 越小越好）:")
    for err, p in ranked[:4]:
        print(f"        RMS={err:5.1f} µs  "
              + "".join(f"ch{RAW_CHANNELS[i]}=mic{p[i]} " for i in range(4)))
    print("      `--audio-source micN` 的映射需按上表修正")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nosep", type=int, default=4,
                    help="number of source positions around the array "
                         "(default 4; fewer than 4 cannot establish the "
                         "four-way rotation)")
    ap.add_argument("--window", type=float, default=5.0,
                    help="seconds of noise per position (default 5)")
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--save", default="/tmp/p2_guided_channels.npy",
                    help="write all captures to .npy (default "
                         "/tmp/p2_guided_channels.npy, so a weak run can be "
                         "re-analysed offline instead of repeated)")
    ap.add_argument("--save-only", action="store_true",
                    help="print no verdict; just record and save for offline "
                         "analysis")
    ap.add_argument("--analyse", default=None,
                    help="analyse a previously saved .npy and exit (no device "
                         "needed): reports the TDOA verdict")
    ap.add_argument("--selftest", action="store_true",
                    help="check the verdict logic against synthetic patterns, "
                         "then exit (does not touch the device)")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if args.analyse:
        data = np.load(args.analyse)
        print(f"[p2] 离线分析 {args.analyse}  {data.shape}")
        return tdoa_report(list(data))

    import sounddevice as sd

    device_idx = args.device if args.device is not None else find_array()

    control = None
    try:
        from realtime_subtitle.usb_device import ReSpeaker

        control = ReSpeaker()
    except Exception as exc:  # noqa: BLE001
        say(f"[p2] control plane unavailable ({exc}); DOA cross-check disabled")

    say("=" * 72)
    say("引导式通道识别")
    say("=" * 72)
    say("")
    say("阵列是方形四麦，四个麦克风分别在四个角上：")
    say("    mic0 = 315°   mic1 = 45°   mic2 = 135°   mic3 = 225°")
    say("")
    say("声源靠近哪个角，那个角的麦克风就最响。")
    say(f"接下来会做 {args.nosep} 次录制，每次你把声源移动约 90°")
    say("（比如每次都往你的左手边挪一格），旋转方向要**保持一致**。")
    say("")
    say("每次的流程：")
    say("  1. 看到『就位』→ 移动声源到下一个位置，站好别出声")
    say("  2. 看到『开始发声』→ 持续拍手或说话，直到提示停止")
    say("  3. 看到『停』→ 停手，准备下一次")
    say("")
    say("注意事项：")
    say("  - 声源尽量离阵列近（20~40cm）、保持同一距离，只改方位")
    say("  - 每次都要发出**足够响**的声音，否则会提示重录")
    say("  - 不需要知道哪边是正前方；脚本靠方位的变化量来判断")
    say("")
    input("准备好后按 Enter 开始 ...")
    say("")
    say(f"[p2] 打开 6 通道输入流（设备 {device_idx}），稳定期 {args.settle:.0f}s ...")

    baseline = []
    steps: list[np.ndarray] = []
    doa_lines: list[str] = []

    with sd.InputStream(device=device_idx, samplerate=SAMPLE_RATE,
                        channels=CAPTURE_CHANNELS, dtype="float32",
                        blocksize=BLOCK) as stream:
        drain(stream, args.settle)                    # drop the transient
        say("[p2] 采集稳定。先录 2s 本底（请安静）...")
        drain(stream, 2.0, baseline)
        base = np.concatenate(baseline, axis=0)
        # the baseline uses the same statistic as the measurement, so the ratio
        # compares like with like - a noise floor's own peaks against a clap's
        base_rms = peak_level(base)
        say("[p2] 本底(短窗峰值): " + " ".join(f"ch{c}={base_rms[c]:.5f}"
                                                for c in range(CAPTURE_CHANNELS)))

        for i in range(args.nosep):
            say("")
            say(f"--- 第 {i + 1}/{args.nosep} 次 ---")
            if i == 0:
                say("就位：站在你想作为**起点**的位置，正对或侧对阵列都可以。")
            else:
                say("就位：把声源移动约 90°，方向与上一次相同。")
            drain(stream, 2.0)                        # time to reposition
            if control is not None:
                doa_lines.append(read_doa(control))
                say(f"    设备 DOA 参考: {doa_lines[-1]}")

            for c in (3, 2, 1):
                say(f"  {c} ...")
                time.sleep(1.0)
            say("开始发声 ▶▶▶  （持续拍手/说话）")
            buf: list[np.ndarray] = []
            drain(stream, args.window, buf)
            say("停 ■")

            x = np.concatenate(buf, axis=0)
            steps.append(x)
            live = peak_level(x)
            # report the raw channels against the baseline so a too-quiet run is
            # caught now rather than producing a confident wrong answer later
            gain = [live[c] / max(base_rms[c], 1e-9) for c in RAW_CHANNELS]
            loud = max(gain)
            second = sorted(gain)[-2] if len(gain) > 1 else 0.0
            say("    " + " ".join(f"ch{c} x{gain[k]:5.1f}"
                                  for k, c in enumerate(RAW_CHANNELS)))
            # contrast matters as much as absolute level: if the best channel is
            # only slightly above the runner-up, ranking them is a coin toss and
            # the verdict would be noise dressed up as a mapping.
            if loud < 3.0:
                say(f"    ⚠ 最响的原始通道只有本底的 {loud:.1f} 倍，太轻了；"
                    f"请离近一点、拍得更用力，下次会更好")
            elif loud < 1.3 * second:
                say(f"    ⚠ 最响 {loud:.1f} 倍但次高 {second:.1f} 倍，区分度不足；"
                    f"请更靠近某一颗麦克风")
            else:
                say(f"    ✓ 最响 {loud:.1f} 倍，次高 {second:.1f} 倍（区分度 "
                    f"{loud / max(second, 1e-9):.2f}x）")

    if control is not None:
        control.close()

    # ---- analysis -----------------------------------------------------
    say("")
    say("=" * 72)
    say("判定")
    say("=" * 72)
    say("")
    result = analyse(steps, base_rms)
    if args.save_only:
        np.save(args.save, np.stack(steps))
        say(f"[p2] 已保存 {args.save}；未给判定（--save-only）")
        np.save(args.save.replace(".npy", "_base.npy"), base_rms)
        say(f"[p2] 本底已保存 {args.save.replace('.npy', '_base.npy')}")
        return 0
    say("%6s %9s %9s %9s %9s   %s"
        % ("位置", *[f"ch{c}" for c in RAW_CHANNELS], "最响"))
    for i, g in enumerate(result["gains"]):
        say("%6d %9.1f %9.1f %9.1f %9.1f   ch%d"
            % (i + 1, *g, result["loudest"][i]))

    say("")
    # The mapping shows up as a rotation: turn the source a quarter turn and
    # the loudest raw channel should advance by one, cyclically. Which
    # direction it advances tells us the source-order-to-bearing orientation,
    # and any break in the rotation means the four channels are not simply
    # consecutive mics.
    order = RAW_CHANNELS
    say(f"最响通道序列(按 source 槽序 {order} 折算): {result['positions']}")
    say(f"相邻步的位移(模 4): {result['diffs']}")
    say("")

    # Contrast first: a clean rotation on weak data is still weak data. If the
    # winning channel barely beats the runner-up, the "loudest" column is
    # ranking noise and any verdict below would be an artefact of the window
    # rather than a property of the array.
    strong = 0
    for gains in result["gains"]:
        ordered = sorted(gains)
        if ordered[-1] >= 3.0 and ordered[-1] >= 1.3 * ordered[-2]:
            strong += 1
    say(f"区分度合格的位置: {strong}/{len(result['gains'])} "
        f"(需 3 倍以上且比次高高 30%)")
    if strong < len(result["gains"]):
        say("")
        say(f"尚不能结论: 只有 {strong}/{len(result['gains'])} 个位置的信号足够强。")
        say("      弱信号下『最响』可能只是在给噪声排名。请重跑：")
        say("        - 离阵列更近（15~25cm）")
        say("        - 每次持续拍手/敲击 5s 以上，要响")
        say("        - 声源放在某个**角**上，而不是边中")
        result["verdict"] = "weak"
    # A rotation can only be established if the source actually went all the
    # way round: with fewer positions than channels the pattern is necessarily
    # consistent, so a clean "forward" there would be an artefact of having
    # asked too few questions rather than a finding.
    elif args.nosep < len(RAW_CHANNELS):
        say(f"尚不能结论: 只测了 {args.nosep} 个位置，而原始通道有 "
            f"{len(RAW_CHANNELS)} 个。要定下四点循环必须至少走满一圈，"
            f"请用 --nosep {len(RAW_CHANNELS)}（或 8 以交叉验证）重跑。")
    elif result["verdict"] == "forward":
        say("结论: 每转 90° 最响通道前进一格 → 四个原始通道**按方位顺序排列**，")
        say("      且与所测通道序列同向。对应关系为:")
        for k, c in enumerate(order):
            say(f"        source {k} -> capture ch{c}")
        say("      即 ch2..ch5 紧随 source 0..3，`--audio-source mic0..mic3` 的")
        say("      通道映射成立")
    elif result["verdict"] == "reverse":
        say("结论: 每转 90° 最响通道**后退**一格 → 四个通道仍是连续的，")
        say("      但 source 序与方位序**反向**。`--audio-source mic0..mic3`")
        say("      的通道号本身正确，只是 mic0 对应的是反方向的角")
    elif result["verdict"] == "repeat":
        say(f"结论: 有位置重复命中同一通道 → {args.nosep} 个位置可能未真正相隔 90°，")
        say("      或声源太偏（一颗麦克风始终最响）。建议重跑，")
        say("      确保每次都真正绕阵列转过一格")
    else:
        say("结论: 位移不是恒定的一格 → 四个原始通道**不与方位顺序一致**，")
        say("      或位置间隔不均匀。再跑一次（可加 --nosep 8）以确认；")
        say("      若仍不规则，说明不能用 source 序推断通道方位")

    say("")
    say("局限(如实):")
    say("  - 本脚本只定**原始麦克风的方位对应**，不能定哪些通道是波束输出")
    say("    （那需要改路由，见 tools/p2_channel_probe.py）")
    say("      ch0/ch1 是否分别为 beam1/beam2 仍未定")
    say("  - 判定依赖操作者每次都真正绕阵列转过约 90°；非线性说明不了更多")
    say("  - 设备 DOA 仅作参考，本脚本不用它下结论（空房间会漂移）")
    say("  - 全程只读录制，未写入任何设备参数")
    say("")
    say("若四项判定为 forward/reverse，还需人工确认一步：")
    say("    source 0 对应哪个物理角。把声源放在某个角，看最响的是哪个")
    say("    source 槽，即可把『角的位置』与『source 编号』对上。")

    if args.save:
        np.save(args.save, np.stack(steps))
        say(f"\n[p2] 已保存 {args.save} ({len(steps)} x "
            f"{steps[0].shape[0]} x {CAPTURE_CHANNELS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())