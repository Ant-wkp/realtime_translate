"""P1.5 probe: can a single streaming AutoModel serve two interleaved caches?

The dual-lane design rests on one assumption that is cheap to check and
expensive to get wrong: that the streaming Paraformer keeps *all* of its state
in the `cache` dict the caller hands it, so two utterances can be advanced on
the same model instance without seeing each other. Reading the FunASR 1.4.12
source says that is nearly true - `cache` holds the audio remainder
(`prev_samples`), the encoder k/v caches, the CIF accumulators, the decoder
FSMN state and the frontend's fbank remainder - but "nearly" is not good
enough for something the whole architecture leans on, and there are a few
instance-level attributes a mistake could route through
(`overlap_chunk.get_chunk_size` writes `chunk_size_pad_shift_cur` on the
object; `AutoModel.inference` deep-updates `self.kwargs` with the call's
`cache`). So this probe measures it instead of trusting the reading.

Two different Chinese utterances are streamed through ONE provider under
several interleaving schedules, and each stream's per-call output is compared
against that same stream run alone. If the model is cache-scoped, every
comparison is byte-identical. A single mismatch means the shared-instance
design is unsound and the fallback applies: share the offline and punc models
but give each lane its own streaming model.

    python tools/p15_interleave_probe.py
    python tools/p15_interleave_probe.py --two-instance   # extra control

Read-only with respect to hardware: no device access, no pipeline changes, no
writes outside the terminal.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realtime_subtitle.funasr_provider import FunasrProvider   # noqa: E402

SAMPLE_RATE = 16000

# Two utterances with disjoint content, so a leak is unmistakable: if A's text
# grows characters that only B ever said, the caches are entangled. Both are
# real 16 kHz mono Chinese speech shipped with the speaker-verification model.
FIXTURE_DIR = (
    "/home/ant/.cache/modelscope/models/"
    "iic--speech_campplus_sv_zh-cn_16k-common/snapshots/master/examples"
)
FIXTURE_A = "speaker1_a_cn_16k.wav"
FIXTURE_B = "speaker2_a_cn_16k.wav"


def load_fixture(name: str) -> np.ndarray:
    """Load a 16 kHz mono wav as float32 in [-1, 1]."""
    path = os.path.join(FIXTURE_DIR, name)
    with wave.open(path) as w:
        if w.getnchannels() != 1 or w.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                f"{name}: expected 16 kHz mono, got "
                f"{w.getframerate()} Hz / {w.getnchannels()} ch"
            )
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def to_chunks(audio: np.ndarray, chunk_samples: int) -> list[np.ndarray]:
    """Split into fixed-size blocks, zero-padding the last one."""
    n = math.ceil(audio.size / chunk_samples)
    padded = np.zeros(n * chunk_samples, dtype=np.float32)
    padded[: audio.size] = audio
    return [padded[i * chunk_samples:(i + 1) * chunk_samples] for i in range(n)]


# ---------------------------------------------------------------------------
# scenario plans
# ---------------------------------------------------------------------------
def plan_alone(sid: str, chunks: list[np.ndarray]) -> list[tuple]:
    return [("feed", sid, c) for c in chunks] + [("finish", sid, None)]


def plan_alternate(a_chunks, b_chunks) -> list[tuple]:
    """Strict A,B,A,B interleave - the scheme the pipeline would actually use."""
    steps = []
    for x, y in zip(a_chunks, b_chunks):
        steps += [("feed", "A", x), ("feed", "B", y)]
    tail_a = a_chunks[len(b_chunks):]
    tail_b = b_chunks[len(a_chunks):]
    steps += [("feed", "A", c) for c in tail_a]
    steps += [("feed", "B", c) for c in tail_b]
    return steps + [("finish", "A", None), ("finish", "B", None)]


def plan_burst(a_chunks, b_chunks, group: int = 5) -> list[tuple]:
    """Alternate in bursts rather than every tick, so a stream accumulates a
    longer uninterleaved run and any 'last call wins' state has a chance to
    show up as a discontinuity at the burst boundary."""
    steps = []
    for i in range(0, max(len(a_chunks), len(b_chunks)), group):
        steps += [("feed", "A", c) for c in a_chunks[i:i + group]]
        steps += [("feed", "B", c) for c in b_chunks[i:i + group]]
    return steps + [("finish", "A", None), ("finish", "B", None)]


def plan_final_while_streaming(a_chunks, b_chunks) -> list[tuple]:
    """Close A in the middle of B. The tail pass runs `is_final=True`, which
    resets the cache via init_cache - if that reset touched shared state, B's
    remaining output is where it would break."""
    half = len(b_chunks) // 2
    steps = [("feed", "A", c) for c in a_chunks]
    steps += [("feed", "B", c) for c in b_chunks[:half]]
    steps += [("finish", "A", None)]
    steps += [("feed", "B", c) for c in b_chunks[half:]]
    steps += [("finish", "B", None)]
    return steps


def plan_reopen(a_chunks) -> list[tuple]:
    """Finish a stream, then start a fresh one with the same audio. Proves the
    is_final cache reset leaves nothing behind for the next stream."""
    steps = [("feed", "A", c) for c in a_chunks] + [("finish", "A", None)]
    steps += [("feed", "A2", c) for c in a_chunks] + [("finish", "A2", None)]
    return steps


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------
def play(provider, steps: list[tuple], shared_cache: dict | None = None,
         on_start=None) -> tuple[dict, dict]:
    """Run a step plan. Returns (per-stream call log, per-stream final text).

    The call log records exactly what `feed()` returned - "" when the decoder
    produced nothing new, else the cumulative text. That is the same value the
    pipeline broadcasts, so it captures both *content* and *timing*: a run
    that emits the right words on the wrong ticks fails the comparison.

    `shared_cache` is for the negative control only: pointing every stream at
    one cache dict must corrupt the streams, which is how the probe's ability
    to detect corruption gets demonstrated rather than assumed.
    """
    log: dict[str, list[str]] = {}
    finals: dict[str, str] = {}
    started: set[str] = set()
    for op, sid, payload in steps:
        if op == "feed":
            if sid not in started:
                provider.start_utt(sid)
                started.add(sid)
                log[sid] = []
                if shared_cache is not None:
                    provider._utts[sid].cache = shared_cache
            log[sid].append(provider.feed(sid, payload))
        elif op == "finish":
            text, _audio = provider.finish_online(sid)
            finals[sid] = text
        else:
            raise ValueError(f"unknown step {op!r}")
    if on_start is not None:
        on_start(provider)
    return log, finals


def summarise(calls: list[str]) -> str:
    """The cumulative text at the end of a call log."""
    return next((c for c in reversed(calls) if c), "")


def char_trigrams(text: str) -> set[str]:
    """Character 3-grams of the text with whitespace removed, for the
    cross-contamination check (ASR output is space-separated characters)."""
    s = "".join(text.split())
    return {s[i:i + 3] for i in range(max(0, len(s) - 2))}


def build_provider(chunk_ms: float):
    print("[p15] loading FunASR Paraformer (offline + punc + streaming) ...")
    provider = FunasrProvider()
    if provider._stream is None:
        raise SystemExit("streaming model unavailable - nothing to probe")
    print(f"[p15] chunk stride preset: {provider.chunk_size} "
          f"({provider.chunk_size[1] * 60} ms)")
    return provider


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-ms", type=float, default=100.0,
                    help="feed block size, matching the main-loop tick (default 100)")
    ap.add_argument("--two-instance", action="store_true",
                    help="also run B on a second provider as an uncontaminated "
                         "control (roughly doubles memory)")
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(4)   # keep CPU ops reproducible

    chunk = int(args.chunk_ms * SAMPLE_RATE / 1000)
    audio_a = load_fixture(FIXTURE_A)
    audio_b = load_fixture(FIXTURE_B)
    chunks_a = to_chunks(audio_a, chunk)
    chunks_b = to_chunks(audio_b, chunk)
    print(f"[p15] A {FIXTURE_A} {audio_a.size / SAMPLE_RATE:.2f}s "
          f"-> {len(chunks_a)} blocks")
    print(f"[p15] B {FIXTURE_B} {audio_b.size / SAMPLE_RATE:.2f}s "
          f"-> {len(chunks_b)} blocks")

    provider = build_provider(args.chunk_ms)

    # offline decode of each fixture: the yardstick for "did these two ever
    # bleed into each other", independent of the streaming path
    truth = {}
    for tag, audio in (("A", audio_a), ("B", audio_b)):
        res = provider._offline.generate(input=audio, batch_size_s=300,
                                         disable_pbar=True)
        truth[tag] = "".join((res[0].get("text", "") if res else "").split())
    print(f"[p15] offline truth A = {truth['A']!r}")
    print(f"[p15] offline truth B = {truth['B']!r}")

    scenarios: list[tuple[str, list[tuple]]] = [
        ("alone_A", plan_alone("A", chunks_a)),
        ("alone_B", plan_alone("B", chunks_b)),
        ("alternate", plan_alternate(chunks_a, chunks_b)),
        ("burst", plan_burst(chunks_a, chunks_b)),
        ("final_mid_stream", plan_final_while_streaming(chunks_a, chunks_b)),
        ("reopen", plan_reopen(chunks_a)),
        ("alone_A_repeat", plan_alone("A", chunks_a)),
    ]

    runs: dict[str, tuple[dict, dict]] = {}
    for name, steps in scenarios:
        print(f"[p15] running {name} ({len(steps)} steps) ...")
        runs[name] = play(provider, steps)
        # informational: what the instance-level kwargs currently hold. It is
        # expected to point at the most recent call's cache; the question the
        # probe answers is whether the *next* call's reset makes that harmless.
        held = provider._stream.kwargs.get("cache")
        print(f"        self.kwargs['cache'] id={id(held) if held is not None else None}")

    checks: list[tuple[str, str, str, str, str]] = [
        ("alternate leaves A untouched", "alternate", "A", "alone_A", "A"),
        ("alternate leaves B untouched", "alternate", "B", "alone_B", "B"),
        ("burst leaves A untouched", "burst", "A", "alone_A", "A"),
        ("burst leaves B untouched", "burst", "B", "alone_B", "B"),
        ("A's final mid-stream leaves B untouched",
         "final_mid_stream", "B", "alone_B", "B"),
        ("A's final mid-stream leaves A untouched",
         "final_mid_stream", "A", "alone_A", "A"),
        ("reopen reproduces alone exactly", "reopen", "A2", "alone_A", "A"),
        ("alone is reproducible after traffic",
         "alone_A_repeat", "A", "alone_A", "A"),
    ]

    failures = 0
    print("\n=== per-call output comparisons ===")
    for label, r1, s1, r2, s2 in checks:
        calls_1 = runs[r1][0].get(s1, [])
        calls_2 = runs[r2][0].get(s2, [])
        if calls_1 == calls_2:
            print(f"PASS  {label} ({len(calls_1)} calls identical)")
            continue
        failures += 1
        print(f"FAIL  {label}")
        n = max(len(calls_1), len(calls_2))
        for i in range(n):
            a = calls_1[i] if i < len(calls_1) else "<missing>"
            b = calls_2[i] if i < len(calls_2) else "<missing>"
            if a != b:
                print(f"        call {i}: {r1}/{s1}={a!r}")
                print(f"        call {i}: {r2}/{s2}={b!r}")

    print("\n=== final text comparisons ===")
    for label, r1, s1, r2, s2 in checks:
        f1 = runs[r1][1].get(s1)
        f2 = runs[r2][1].get(s2)
        if f1 is None or f2 is None:
            continue
        if f1 == f2:
            print(f"PASS  {label} (final {f1!r})")
            continue
        failures += 1
        print(f"FAIL  {label}")
        print(f"        {r1}/{s1} final = {f1!r}")
        print(f"        {r2}/{s2} final = {f2!r}")

    print("\n=== final texts ===")
    for name, (_log, finals) in runs.items():
        for sid, text in sorted(finals.items()):
            print(f"{name:18s} {sid:3s} {text!r}")

    # cross-contamination: does either stream's output contain content that
    # could only have come from the other utterance?
    print("\n=== cross-contamination ===")
    inter = runs["alternate"]
    for sid, other in (("A", "B"), ("B", "A")):
        got = "".join(summarise(inter[0].get(sid, [])).split())
        shared = char_trigrams(got) & char_trigrams(truth[other])
        if not shared:
            print(f"PASS  {sid} shares no 3-gram with offline {other}")
        else:
            print(f"WARN  {sid} shares {len(shared)} 3-gram(s) with offline "
                  f"{other}: {sorted(shared)[:5]}")

    # Negative control. A probe that cannot fail proves nothing, so point both
    # streams at ONE cache dict and check the corruption is actually seen. If
    # this control passes (i.e. looks clean) the probe is broken and every
    # PASS above is worthless.
    print("\n=== negative control: forced shared cache (must FAIL) ===")
    forced = FunasrProvider()
    try:
        shared = {}
        log_n, finals_n = play(forced, plan_alternate(chunks_a, chunks_b),
                               shared_cache=shared)
        a_ok = log_n.get("A") == runs["alone_A"][0]["A"]
        b_ok = log_n.get("B") == runs["alone_B"][0]["B"]
        if a_ok and b_ok:
            failures += 1
            print("FAIL  forced shared cache still reproduced both streams - "
                  "the probe cannot detect what it was written to detect")
        else:
            print(f"OK    forced shared cache corrupted the streams as expected "
                  f"(A clean={a_ok}, B clean={b_ok})")
            print(f"        A final = {finals_n.get('A')!r}")
            print(f"        B final = {finals_n.get('B')!r}")
    finally:
        forced.close()

    if args.two_instance:
        print("\n=== control: B on a second provider ===")
        second = FunasrProvider()
        log_b, finals_b = play(second, plan_alone("B", chunks_b))
        calls_ctrl = log_b["B"]
        if calls_ctrl == runs["alone_B"][0]["B"]:
            print("PASS  control B matches alone_B (second instance agrees)")
        else:
            failures += 1
            print("FAIL  control B differs from alone_B - the baseline itself "
                  "is not reproducible, so the interleave result means less")
        second.close()

    provider.close()
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{failures} CHECK(S) FAILED'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())