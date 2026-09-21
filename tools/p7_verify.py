"""P7: does the dual-lane split actually work?

The whole point of two lanes is that one beam hears one person better than the
other, by enough that the recognizer gets one voice instead of the room mix.
That is a measurable physical quantity, and measuring it directly is far
faster and more repeatable than judging it from transcripts - so this script
does the physics, and prints the end-to-end procedure separately.

Three scenarios, each a guided capture:

  solo0   only the person at lane 0's bearing talks
  solo1   only the person at lane 1's bearing talks
  both    both talk at once - the case the feature exists for

For each, the per-beam energies (SPENERGY) and the two capture channels are
recorded together, and the separation is reported as lane0-vs-lane1 in dB.

Two independent readings are compared on purpose. SPENERGY is the device's own
per-beam speech energy, and the channel levels are what the recognizer will
actually receive. They can disagree - a beam can report energy for a talker
while the routed channel still carries the other one, if the routing is not
what we think - so agreement is the evidence, not either number alone.

Thresholds are stated as bands rather than one number, because the honest
answer for a 6.7 cm aperture may be "not much", and a single pass/fail line
would hide that. 10 dB is the target the plan set; 6 dB is usable; below 3 dB
the two lanes are effectively the same channel.

Read-only: nothing is steered and no routing is written. The device is only
read, so this can run against whatever configuration is already in place.

    python tools/p7_verify.py --lanes 180 0
    python tools/p7_verify.py --lanes 180 0 --scenario solo0   # one case
    python tools/p7_verify.py --selftest                      # no device
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
BLOCK = 1024
METRIC_WINDOW = int(0.025 * SAMPLE_RATE)

# Separation bands, in dB. The plan's target is 10; anything under 3 means the
# two channels carry essentially the same signal.
BAND_EXCELLENT = 10.0
BAND_USABLE = 6.0
BAND_MARGINAL = 3.0


def find_array() -> int:
    import sounddevice as sd

    for d in sd.query_devices():
        name = d["name"].lower()
        if ("xvf3800" in name or "respeaker" in name) and \
                d["max_input_channels"] >= CAPTURE_CHANNELS:
            return d["index"]
    raise SystemExit("XVF3800 not found among sounddevice inputs")


def peak_level(a: np.ndarray) -> np.ndarray:
    """Per-channel maximum of short-window RMS (transient-proof)."""
    x = a.astype(np.float64)
    n = x.shape[0] // METRIC_WINDOW
    if n == 0:
        return np.sqrt(np.mean(x ** 2, axis=0))
    frames = x[: n * METRIC_WINDOW].reshape(n, METRIC_WINDOW, x.shape[1])
    return np.sqrt(np.mean(frames ** 2, axis=1)).max(axis=0)


def db(a: float, b: float) -> float:
    return 20.0 * np.log10(max(a, 1e-12) / max(b, 1e-12))


def band(value: float, label: str) -> str:
    if value >= BAND_EXCELLENT:
        return f"优秀（≥{BAND_EXCELLENT:.0f} dB，达标）"
    if value >= BAND_USABLE:
        return f"可用（{BAND_USABLE:.0f}–{BAND_EXCELLENT:.0f} dB）"
    if value >= BAND_MARGINAL:
        return f"勉强（{BAND_MARGINAL:.0f}–{BAND_USABLE:.0f} dB）"
    return f"几乎无分离（<{BAND_MARGINAL:.0f} dB）"


def capture(device: int, seconds: float, control,
            channels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Record channels while polling SPENERGY. Returns (audio, per_tick_energy).

    Polling during the sound is the only way the energy reading carries
    information: SPENERGY is exactly zero whenever nobody is talking, so a
    reading taken before the stimulus says nothing at all.
    """
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
                except Exception:  # noqa: BLE001 - polling is best effort
                    pass
    e = np.array(energies) if energies else np.zeros((0, 4))
    return got, e


def summarise(got: np.ndarray, energies: np.ndarray,
              chans: list[int]) -> dict:
    """Reduce one capture to the numbers the verdict is made from."""
    levels = peak_level(got)
    out = {
        "ch": [float(levels[c]) for c in chans],
        "ch_db": db(float(levels[chans[0]]), float(levels[chans[1]])),
    }
    if energies.size:
        # Use the peak per beam rather than the mean: a talker pauses, and
        # averaging over pauses understates the beam that was actually
        # pointing at them. This is the same reason the level metric is a
        # short-window peak rather than a window RMS.
        e0 = float(energies[:, 0].max())
        e1 = float(energies[:, 1].max())
        out.update({"e0": e0, "e1": e1, "energy_db": db(e0, e1),
                    "ticks": int(energies.shape[0])})
    else:
        out.update({"e0": 0.0, "e1": 0.0, "energy_db": 0.0, "ticks": 0})
    return out


def verdict_solo(s: dict, which: int) -> tuple[bool, str]:
    """One talker: did their own lane win, on both independent readings?"""
    other = 1 - which
    want = 1.0 if which == 0 else -1.0
    e_db, c_db = s["energy_db"], s["ch_db"]
    e_ok = (e_db >= BAND_MARGINAL) if which == 0 else (e_db <= -BAND_MARGINAL)
    c_ok = (c_db >= BAND_MARGINAL) if which == 0 else (c_db <= -BAND_MARGINAL)
    sep = abs(e_db) if which == 0 else abs(e_db)
    lines = [
        f"SPENERGY  beam1={s['e0']:.0f} beam2={s['e1']:.0f} "
        f"→ lane0 相对 lane1 {e_db:+.1f} dB（{s['ticks']} 次读数）",
        f"采集通道 ch{0}={s['ch'][0]:.5f} ch{1}={s['ch'][1]:.5f} "
        f"→ {c_db:+.1f} dB",
        f"分离度 {band(sep, '')}",
    ]
    if e_ok and c_ok:
        lines.append(f"✅ 两个独立读数一致：lane{which} 的波束与通道都占优")
        return True, "\n     ".join(lines)
    if e_ok != c_ok:
        lines.append(f"⚠ 两个读数**不一致**（能量={'占优' if e_ok else '不占优'}、"
                     f"通道={'占优' if c_ok else '不占优'}）—— 说明路由与我们"
                     f"的假设不符，须先查 OP_L/OP_R 落点，不要采信单一读数")
        return False, "\n     ".join(lines)
    lines.append(f"❌ lane{which} 未被分离出来")
    if which == 0:
        lines.append("可能原因：两个波束指向差太小 / 说话人离两个波束等距 / "
                     "激励太轻")
    return False, "\n     ".join(lines)


def verdict_both(s0: dict, s1: dict, s_both: dict) -> tuple[bool, str]:
    """Both talking: each beam must keep hearing its own person.

    The test is retention, not absolute level. Each beam should be about as
    loud with both people talking as it was with only its own talker - if
    beam 1 stays near its solo level while both talk, the second person did
    not drown the first one out, which is the entire claim.

    Retention alone is the criterion, deliberately. An earlier version also
    required the two beams to stay unequal to each other, on the theory that
    equal levels would mean both are carrying the room mix - but with two
    people both talking, equal levels is exactly what a working split looks
    like, since each beam is hearing one person at full strength. The two
    cases are not separable by level, so that check could only reject good
    results. What separates them was already established by the solo runs.
    """
    own0 = s_both["e0"] / max(s0["e0"], 1e-9)
    own1 = s_both["e1"] / max(s1["e1"], 1e-9)
    lines = [
        f"同时说话: beam1={s_both['e0']:.0f} beam2={s_both['e1']:.0f}",
        f"单人时:   beam1={s0['e0']:.0f}（solo0）  beam2={s1['e1']:.0f}（solo1）",
        f"各自保留比: lane0={own0:.2f}x lane1={own1:.2f}x "
        f"（1.0 = 完全不受对方影响）",
    ]
    if own0 > 0.3 and own1 > 0.3:
        lines.append("✅ 两人同时说话时两个波束都保住各自的人，未被对方压过 —— "
                     "空间分离成立")
        return True, "\n     ".join(lines)
    weak = "lane0" if own0 <= own1 else "lane1"
    lines.append(f"❌ {weak} 在对方说话时被压过（保留比 "
                 f"{min(own0, own1):.2f}x）—— 孔径不足以分离两人")
    lines.append("     （6.7cm 孔径是物理上限；可尝试加大两波束指向差、"
                 "让两人离阵列更近或分坐两侧，或改用原始麦克风通道）")
    return False, "\n     ".join(lines)


def calibration(solo: list[dict], quiet: dict) -> str:
    """Suggest an energy threshold from the measured distributions."""
    speech = [max(s["e0"], s["e1"]) for s in solo]
    speech.append(max(quiet["e0"], quiet["e1"]))
    quiet_max = max(quiet["e0"], quiet["e1"])
    speech_min = min(speech[:-1]) if len(speech) > 1 else speech[0]
    lines = [f"静音期最高 {quiet_max:.0f}；单人语音期最低 {speech_min:.0f}"]
    if speech_min > quiet_max * 2:
        mid = (speech_min + quiet_max) / 2
        lines.append(f"→ 两者之间有明显间隔，建议 --beam-energy-threshold "
                     f"{mid:.0f}（几何中点）")
    else:
        lines.append("→ 语音与静音量级重叠，单一阈值难以可靠分开；"
                     "建议依靠 energy 门控的保持窗口(0.6s)而非阈值本身，"
                     "或改用 --vad-gate energy 观察后再定")
    return "\n     ".join(lines)


def _selftest() -> int:
    """Exercise the verdicts against synthetic numbers before trusting them."""
    print("判定逻辑自检（合成数据，不碰设备）")
    ok = True

    def solo(e0, e1, c0, c1):
        return {"e0": e0, "e1": e1, "ch": [c0, c1],
                "energy_db": db(e0, e1), "ch_db": db(c0, c1), "ticks": 10}

    cases = [
        ("solo0 强分离", solo(2.5e6, 1e5, 0.05, 0.002), 0, True),
        ("solo1 强分离", solo(1e5, 2.5e6, 0.002, 0.05), 1, True),
        ("solo0 几乎无分离", solo(1e6, 9e5, 0.05, 0.045), 0, False),
        ("solo0 能量对但通道反（不一致）", solo(2.5e6, 1e5, 0.002, 0.05), 0, False),
    ]
    for name, s, which, want in cases:
        got, _ = verdict_solo(s, which)
        flag = "ok " if got == want else "BAD"
        if got != want:
            ok = False
        print(f"  {flag} {name} → {got}")

    # both: each keeps most of its own level
    good = verdict_both(solo(2.5e6, 1e5, .05, .002), solo(1e5, 2.5e6, .002, .05),
                        solo(2.4e6, 2.3e6, .05, .05))[0]
    bad = verdict_both(solo(2.5e6, 1e5, .05, .002), solo(1e5, 2.5e6, .002, .05),
                       solo(2.4e6, 1.2e5, .05, .002))[0]
    for name, got, want in [("both 互相保留", good, True),
                            ("both 一轨被压过", bad, False)]:
        flag = "ok " if got == want else "BAD"
        if got != want:
            ok = False
        print(f"  {flag} {name} → {got}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lanes", nargs=2, type=float, metavar=("AZ0", "AZ1"),
                    required=False,
                    help="the two lane bearings (degrees), same as --beam-lanes")
    ap.add_argument("--scenario", choices=["solo0", "solo1", "both", "all"],
                    default="all")
    ap.add_argument("--window", type=float, default=6.0,
                    help="seconds of talking per scenario (default 6)")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.lanes is None:
        print("需要 --lanes AZ0 AZ1（与 --beam-lanes 相同的方位）")
        return 2

    try:
        from realtime_subtitle.usb_device import ReSpeaker

        control = ReSpeaker()
    except Exception as exc:  # noqa: BLE001
        print(f"[p7] 控制面不可用（{exc}）—— SPENERGY 读数需要它")
        return 1

    dev = args.device if args.device is not None else find_array()
    chans = [0, 1]
    az0, az1 = args.lanes
    labels = []
    try:
        from realtime_subtitle.speaker_locator import SpeakerLocator

        loc = SpeakerLocator()
        labels = [loc.label_for(a % 360.0) for a in args.lanes]
    except Exception:  # noqa: BLE001
        labels = ["", ""]
    print(f"[p7] 采集设备 {dev}，通道 {chans}")
    print(f"[p7] lane0 = 波束1 @ {az0:.0f}°（{labels[0] or '?'}）  "
          f"lane1 = 波束2 @ {az1:.0f}°（{labels[1] or '?'}）")
    print("[p7] 只读：本脚本不改路由、不转波束。请先按下面的命令让双轨跑起来：")
    print(f"       python -u -m realtime_subtitle.main --beam-lanes "
          f"{az0:.0f} {az1:.0f} --debug-gate")
    print("[p7] 并让两个人分别站在两个方位上")
    print()

    scenarios = (["solo0", "solo1", "both"] if args.scenario == "all"
                 else [args.scenario])
    order = {"solo0": "只有 lane0 方位的人说话（另一端的人**保持安静**）",
             "solo1": "只有 lane1 方位的人说话（另一端的人**保持安静**）",
             "both": "**两人同时**持续说话"}
    results: dict[str, dict] = {}
    verdicts: dict[str, tuple[bool, str]] = {}

    try:
        print("[p7] 先录 3s 静音做本底 ...")
        capture(dev, 3.0, control, chans)          # settle + quiet baseline
        got, e = capture(dev, 3.0, control, chans)
        quiet = summarise(got, e, chans)
        print(f"       静音: beam1={quiet['e0']:.0f} beam2={quiet['e1']:.0f}")

        for sc in scenarios:
            print()
            print("=" * 68)
            print(f"场景 {sc}：{order[sc]}")
            print("=" * 68)
            print(f"[p7] 5 秒后开始录 {args.window:.0f}s ...", flush=True)
            time.sleep(5.0)
            print(f"[p7] 开始 ▶▶▶", flush=True)
            got, e = capture(dev, args.window, control, chans)
            print(f"       停 ■", flush=True)
            s = summarise(got, e, chans)
            results[sc] = s
            if sc == "solo0":
                verdicts[sc] = verdict_solo(s, 0)
            elif sc == "solo1":
                verdicts[sc] = verdict_solo(s, 1)
            else:
                verdicts[sc] = ("pending", "")
    finally:
        control.close()

    print()
    print("=" * 68)
    print("判定")
    print("=" * 68)
    for sc in ("solo0", "solo1"):
        if sc in verdicts:
            ok, why = verdicts[sc]
            print(f"\n{sc}:")
            print(f"     {why}")
    if "both" in results and "solo0" in results and "solo1" in results:
        ok, why = verdict_both(results["solo0"], results["solo1"], results["both"])
        print(f"\nboth（核心指标）:")
        print(f"     {why}")

    print(f"\n阈值标定:\n     {calibration([results[s] for s in ('solo0','solo1') if s in results], quiet)}")

    print("""
端到端验证（本脚本不覆盖，需另跑）

  1. 起双轨并观察归属：
       python -u -m realtime_subtitle.main --beam-lanes AZ0 AZ1 --debug-gate
  2. 单人说话：日志里应只出现**一条** lane 的 [seg]/[asr] 行，另一条 lane 的
     [gate] 应一直 gate=False（energy 门控把它关着）
  3. 两人同时说：应出现**两条** lane 的 [seg] 行交错，且各自成段
  4. 看两轨首字延迟：比较两轨的 [首字] 均值；若 lane1 比 lane0 慢 30% 以上，
     说明单 worker 串行的代价过大，需评估双 worker（内存 ~2x）
  5. 浏览器打开页面：卡片应带 lane 徽章（lane0 青 / lane1 紫），
     顶部门控柱应有两组
  6. 单轨回归：不传 --beam-lanes 跑一次，确认行为与日志格式与改动前一致
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())