"""Real-time bilingual subtitle entry: assembly, gating, segmentation.

Single process, six concurrency units (PortAudio callback thread,
VadDoaReader, this main loop, TranscriberWorker, TranslationThread,
BroadcastServer) connected by queues. The main loop is a polling
convergence loop: each 100 ms tick is an atomic transaction of drain
inputs -> gate -> route audio -> dispatch results, and every heavy
operation inside it is just a queue put.

Gating is three layers: hardware VAD (energy, control plane) AND software
VAD (Silero, inlined each tick) AND segment-level duration/ratio/energy
filters at flush time. Discards always abort the streaming session, clear
buffers, reset counters and return to idle so no partial is left behind.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time

import numpy as np

if __package__ is None or __package__ == "":
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_pkg_dir)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    __package__ = "realtime_subtitle"

from .asr import AsrResult, detect_language
from .compute import DEVICE_CHOICES, describe_platform
from .segmenter import SegmentCfg, Segmenter, trim_tail

# avoid XET download failures with huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

SAMPLE_RATE = 16000

VAD_INTERVAL = 0.1          # main loop tick, seconds
SILENCE_HOLD = 0.6          # trailing silence before flush
MIN_SECONDS = 0.3
MAX_SECONDS = 20.0          # force-cut overlong segments
PRE_ROLL_DEFAULT = 1.0      # ring-buffer window retained before segment start
SW_FRAME_THRESHOLD = 0.5    # Silero per-frame speech threshold
SHORT_EXEMPT_SECONDS = 1.0  # short-segment exemption window
SHORT_EXEMPT_MAX_PROB = 0.5 # peak probability required for exemption
RATIO_THRESHOLD = 0.35      # segment speech-frame ratio floor
TRIM_TAIL_ENERGY = 1e-4     # RMS below which tail frames are trimmed

ARRAY_DEVICE_NAMES = ("respeaker xvf3800", "xvf3800", "respeaker")
class VadDoaReader(threading.Thread):
    """Poll DOA_VALUE (angle + speech activity) off the control plane.

    Isolates USB control transfers (which can block 1 s+ on RETRY storms)
    from the strictly-paced main loop.
    """

    def __init__(self, device, interval: float = 0.1):
        super().__init__(daemon=True, name="VadDoaReader")
        self.device = device
        self.interval = interval
        self.vad_queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                vals = self.device.read("DOA_VALUE")
                azimuth = vals[0] if vals else None
                hw_vad = int(vals[1]) if len(vals) > 1 else 0
                self.vad_queue.put({"hw_vad": hw_vad, "doa": azimuth,
                                    "error": False})
            except Exception:  # noqa: BLE001
                self.vad_queue.put({"hw_vad": 0, "doa": None, "error": True})
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop.set()


class TranslationThread(threading.Thread):
    """Serial NLLB worker; translation never blocks source display."""

    def __init__(self, translator, server, speech_end_at: dict | None = None):
        super().__init__(daemon=True, name="TranslationThread")
        self.translator = translator
        self.server = server
        self.speech_end_at = speech_end_at if speech_end_at is not None else {}
        self.job_queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        # rolling end-to-end stats, printed on the terminal only (the
        # frontend stays a pure renderer and gets no latency fields)
        self._e2e_n = 0
        self._e2e_sum = 0.0

    def summary(self) -> str:
        if not self._e2e_n:
            return ""
        return f"{self._e2e_n} 句, 均值 {self._e2e_sum / self._e2e_n:.2f}s"

    def submit(self, task_id: str, text: str, lang: str) -> None:
        self.job_queue.put({"task_id": task_id, "text": text, "lang": lang})

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            start = time.monotonic()
            try:
                text, target = self.translator.translate(job["text"], job["lang"])
                trans_ms = int((time.monotonic() - start) * 1000)
                print(f"[tr] {job['task_id']} -> {target} in {trans_ms}ms: {text[:60]!r}")
                self.server.broadcast({
                    "type": "translation",
                    "id": job["task_id"],
                    "ts": time.time(),
                    "text": text,
                    "target_lang": target,
                })
                # end-to-end stats: speech end -> translation printed here.
                # Terminal-only (old-version semantics): latency measurement
                # lives on the backend, the frontend never sees it.
                speech_end = self.speech_end_at.pop(job["task_id"], None)
                if speech_end is not None:
                    e2e = time.monotonic() - speech_end
                    self._e2e_n += 1
                    self._e2e_sum += e2e
                    print(f"[耗时] {job['task_id'][:8]} 说完→译文上屏: {e2e:.2f}s "
                          f"(均值 {self._e2e_sum / self._e2e_n:.2f}s / "
                          f"{self._e2e_n} 句)")
            except Exception as exc:  # noqa: BLE001
                print(f"[NLLB] translate failed: {exc}")

    def stop(self) -> None:
        self._stop.set()


class FinalDecodeThread(threading.Thread):
    """Offline second-pass re-decode, decoupled from the streaming worker.

    In 2-pass mode the offline model re-decodes the whole utterance (~1s for
    a long sentence). Running that on the ASR worker would stall the next
    utterance's feed queue and delay its first partial by the decode time.
    This thread keeps the streaming worker free: online partials keep
    flowing while the final text is being reconstructed.
    """

    def __init__(self, provider, on_result):
        super().__init__(daemon=True, name="FinalDecodeThread")
        self.provider = provider
        self.on_result = on_result
        self.job_queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()

    def submit(self, task_id: str, audio, doa: str = "", partial_hint: str = "") -> None:
        self.job_queue.put({"task_id": task_id, "audio": audio,
                            "doa": doa, "partial_hint": partial_hint})

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            start = time.monotonic()
            mode = "2pass-final"
            language = ""
            try:
                res = self.provider.transcribe(job["audio"])
                text = (res.text or "").strip()
                language = res.language
            except Exception as exc:  # noqa: BLE001
                print(f"[ASR] final decode {job['task_id']} failed: {exc}")
                text = ""
                mode = "2pass-final-failed"

            # The offline pass can legitimately return nothing (very short or
            # noisy audio) even though the online pass already recognized
            # something. The card on screen was created by that partial, so
            # the final result has to close it out one way or another - a
            # final that never arrives leaves the text blinking forever.
            # Promote the streaming partial rather than emitting an empty
            # result for the caller to drop.
            if not text:
                text = (job["partial_hint"] or "").strip()
                if text:
                    mode = "2pass-final-partial"

            result = AsrResult(
                text=text,
                language=language,
                latency=time.monotonic() - start,
                meta={
                    "engine": self.provider.name,
                    "mode": mode,
                    "task_id": job["task_id"],
                    "doa": job["doa"],
                    "final_replaces_partial": True,
                    "partial_hint": job["partial_hint"],
                },
            )
            self.on_result(result)

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# CLI / dependency resolution
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="reSpeaker real-time bilingual subtitles")
    p.add_argument("--asr-chunk-size", choices=["480", "600"], default="480",
                   help="streaming decode stride: 480ms (default, lower latency) "
                        "or 600ms (upstream default, longer lookahead)")
    p.add_argument("--audio-source", default="conference",
                   choices=["conference", "asr"],
                   help="which processed output to recognize: 'conference' is "
                        "the full AEC+beam+post-processing chain, 'asr' is the "
                        "shorter beamformer output")
    p.add_argument("--ch", type=int, default=None,
                   help="explicit capture channel index (0-based), overriding "
                        "--audio-source. 0/1 are the two processed outputs, "
                        "2-5 the raw microphones")
    p.add_argument("--in-channels", type=int, default=None)
    p.add_argument("--device", default=None, help="input device name substring")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--vad-interval", type=float, default=VAD_INTERVAL)
    p.add_argument("--silence-hold", type=float, default=SILENCE_HOLD)
    p.add_argument("--min-seconds", type=float, default=MIN_SECONDS)
    p.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    p.add_argument("--pre-roll", type=float, default=PRE_ROLL_DEFAULT,
                   help="seconds of audio retained before segment start; the "
                        "first streaming chunk is filled from this ring buffer "
                        "so the model decodes the opening words immediately "
                        "instead of waiting 600ms for a full chunk")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--device-type", choices=list(DEVICE_CHOICES),
                   default="auto",
                   help="inference device: 'auto' (default) uses CUDA when it "
                        "is actually available, otherwise CPU")
    p.add_argument("--nllb-device", choices=list(DEVICE_CHOICES),
                   default=None,
                   help="override --device-type for the translator only. "
                        "CTranslate2 on aarch64 is a CPU-only build on PyPI, "
                        "so its own 'auto' degrades to CPU when CUDA is "
                        "missing instead of aborting")
    p.add_argument("--nllb-compute-type", default=None,
                   help="pin the CTranslate2 quantisation (e.g. int8, "
                        "int8_float16, float16); default follows the device")
    p.add_argument("--nllb-beam", type=int, default=None,
                   help="NLLB beam width. Default is greedy (1), which is "
                        "about 20%% faster than 4 on both x86 and Orin; raise "
                        "it to 4+ if a translation reads badly")
    p.add_argument("--vad-device", choices=list(DEVICE_CHOICES),
                   default="cpu",
                   help="Silero VAD device; CPU by default because the model "
                        "runs 30 times per second and GPU launch overhead "
                        "outweighs the gain")
    p.add_argument("--no-hw-vad", action="store_true",
                   help="gate on the software VAD alone. Needed when the "
                        "XVF3800 control plane is unavailable, otherwise the "
                        "hardware speech flag stays 0 and no segment ever "
                        "opens (silent, no error)")
    p.add_argument("--host", default="0.0.0.0",
                   help="web server bind address (default 0.0.0.0 so the "
                        "page is reachable from the LAN)")
    p.add_argument("--debug-gate", action="store_true",
                   help="print per-tick gate inputs and every discard reason")
    return p.parse_args(argv)





def resolve_input_device(name: str | None) -> int:
    """Pick the input device: explicit name match, else the XVF3800 array
    (this program's expected hardware), else the system default."""
    import sounddevice as sd

    devs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
    if name is not None:
        for d in devs:
            if name.lower() in d["name"].lower():
                return d["index"]
        listing = "\n".join(f"  [{d['index']}] {d['name']} "
                            f"({d['max_input_channels']} ch)" for d in devs)
        raise RuntimeError(
            f"input device not found: {name}\n"
            f"available input devices:\n{listing or '  (none)'}"
        )
    for d in devs:
        if any(frag in d["name"].lower() for frag in ARRAY_DEVICE_NAMES):
            return d["index"]
    default = sd.default.device[0]
    fallback_name = "default"
    for d in devs:
        if d["index"] == default:
            fallback_name = d["name"]
            break
    print(f"[audio] WARNING: reSpeaker XVF3800 not detected, using default "
          f"input '{fallback_name}' (pass --device <name> to override)")
    return default


def resolve_audio_channel(args, in_channels: int) -> tuple[int, str]:
    if args.ch is not None:
        return args.ch, f"ch{args.ch + 1} (explicit)"
    mapping = {"conference": 0, "asr": 1}
    idx = mapping[args.audio_source]
    if idx >= in_channels:
        # a device with a single input channel still has to yield a channel 0
        print(f"[audio] WARNING: --audio-source {args.audio_source} needs "
              f"channel {idx + 1} but the device exposes {in_channels}; "
              f"falling back to ch1")
        idx = 0
    desc = {
        "conference": "Conference processed output (AEC+beam+post)",
        "asr": "ASR beam output (auto-select, no post)",
    }[args.audio_source]
    return idx, desc


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return 0
    device = None
    try:
        from .usb_device import ReSpeaker

        device = ReSpeaker()
        print("[USB] XVF3800 control plane connected")
    except Exception as exc:
        print(f"[USB] control plane unavailable: {exc}")
    if device is None and not args.no_hw_vad:
        print("[USB] WARNING: no hardware VAD - the gate is hardware AND "
              "software, so nothing will be recognized. Re-run with "
              "--no-hw-vad to gate on the software VAD alone.")

    # ---- audio input ---------------------------------------------------
    input_device = resolve_input_device(args.device)
    dev_info = sd.query_devices(input_device, "input")
    in_channels = args.in_channels or dev_info["max_input_channels"]
    ch_index, ch_desc = resolve_audio_channel(args, in_channels)
    if not 0 <= ch_index < in_channels:
        raise SystemExit(
            f"--ch {ch_index} is out of range for '{dev_info['name']}' "
            f"({in_channels} input channels)"
        )
    print(f"[audio] device {input_device}, {in_channels} ch, "
          f"extract ch{ch_index + 1}: {ch_desc}")
    print(f"[compute] {describe_platform()}")

    from .speaker_locator import SpeakerLocator
    from .sw_vad import SileroVad

    locator = SpeakerLocator()

    from .funasr_provider import FunasrProvider

    chunk = (FunasrProvider.CHUNK_SIZE_480 if args.asr_chunk_size == "480"
             else FunasrProvider.CHUNK_SIZE_600)
    provider = FunasrProvider(chunk_size=chunk, device=args.device_type)
    print(f"[ASR] FunASR Paraformer (zh, streaming 2pass, "
          f"stride {chunk[1] * 60}ms, device {provider.describe_device()})")

    from .translator import DEFAULT_BEAM_SIZE, NllbCTranslate2Provider

    translator = NllbCTranslate2Provider(
        device=args.nllb_device or args.device_type,
        compute_type=args.nllb_compute_type,
        beam_size=args.nllb_beam or DEFAULT_BEAM_SIZE,
    )
    print(f"[NLLB] model: {translator.model_path} "
          f"({translator.device}, {translator.compute_type}, "
          f"beam {translator.beam_size}, "
          f"{translator.zh_to}<-zh / {translator.other_to}<-other)")

    sw_vad = SileroVad(threshold=SW_FRAME_THRESHOLD, device=args.vad_device)
    print(f"[VAD] Silero software VAD ready (device {sw_vad.device})")

    # ---- servers ---------------------------------------------------------
    from .control import ControlHandler
    from .ws_server import BroadcastServer

    server = BroadcastServer(host=args.host, port=args.port)
    server.start()
    control = ControlHandler(locator, translator)
    print(f"[WS] subtitle service: http://{args.host}:{args.port}/ "
          f"(from another machine use this host's IP)")

    speech_end_at: dict[str, float] = {}
    seg_durations: dict[str, float] = {}
    seg_open_at: dict[str, float] = {}
    seg_seed_secs: dict[str, float] = {}
    first_char_logged: set[str] = set()
    first_char_stats = {"n": 0, "sum": 0.0, "miss": 0}
    first_char_lock = threading.Lock()

    translation = TranslationThread(translator, server,
                                     speech_end_at=speech_end_at)
    translation.start()

    result_queue: queue.Queue = queue.Queue()

    def on_asr_result(result):
        result_queue.put(result)

    def on_partial(stream_id, text):
        if text:
            with first_char_lock:
                if stream_id not in first_char_logged:
                    first_char_logged.add(stream_id)
                    opened = seg_open_at.get(stream_id)
                    if opened is not None:
                        dt = time.monotonic() - opened
                        first_char_stats["n"] += 1
                        first_char_stats["sum"] += dt
                        n = first_char_stats["n"]
                        avg = first_char_stats["sum"] / n
                        seed = seg_seed_secs.get(stream_id, 0.0)
                        print(f"[首字] {stream_id[:8]} 张嘴→首字: {dt:.2f}s "
                              f"(均值 {avg:.2f}s / {n} 句, seed {seed:.2f}s)")
        server.broadcast({
            "type": "partial", "id": stream_id, "ts": time.time(), "text": text,
        })

    from .transcriber import TranscriberWorker

    final_decode = FinalDecodeThread(provider, on_result=on_asr_result)
    final_decode.start()

    def on_final_request(req):
        audio = (trim_tail(req.audio, TRIM_TAIL_ENERGY)
                 if req.audio is not None and req.audio.size else req.audio)
        final_decode.submit(req.task_id, audio, doa=req.doa,
                            partial_hint=req.partial_hint)

    transcriber = TranscriberWorker(provider, on_result=on_asr_result,
                                    on_partial=on_partial,
                                    on_final_request=on_final_request)
    transcriber.start()

    vad_reader = None
    if device is not None:
        vad_reader = VadDoaReader(device, interval=args.vad_interval)
        vad_reader.start()
    seg_cfg = SegmentCfg(min_seconds=args.min_seconds,
                         max_seconds=args.max_seconds,
                         silence_hold=args.silence_hold,
                         pre_roll=args.pre_roll,
                         short_exempt_seconds=SHORT_EXEMPT_SECONDS,
                         short_exempt_max_prob=SHORT_EXEMPT_MAX_PROB,
                         ratio_threshold=RATIO_THRESHOLD,
                         trim_tail_energy=TRIM_TAIL_ENERGY,
                         debug=args.debug_gate)

    segmenter = Segmenter(seg_cfg, sw_vad)

    audio_queue: queue.Queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass
        audio_queue.put(indata[:, ch_index].copy())

    def gate_verdict(sw_prob: float) -> bool:
        if args.no_hw_vad:
            return sw_prob >= SW_FRAME_THRESHOLD
        return hw_vad == 1 and sw_prob >= SW_FRAME_THRESHOLD

    def apply_actions(actions):
        for act in actions:
            if act.flush_line:
                print(act.flush_line)
            if act.debug_line:
                print(act.debug_line)
            if act.kind == "open":
                seg_open_at[act.seg_id] = time.monotonic()
                locator.reset()
                seg_seed_secs[act.seg_id] = act.seed_secs
                transcriber.stream_start(act.seg_id)
                transcriber.stream_feed(
                    act.seg_id,
                    act.audio if act.audio is not None and act.audio.size
                    else np.zeros(0, np.float32),
                    act.seg_id)
                segmenter.mark_stream_open()
                print(f"[seg] open {act.seg_id} (seed {act.seed_secs:.2f}s)")
            elif act.kind == "feed":
                transcriber.stream_feed(act.seg_id, act.audio, act.seg_id)
            elif act.kind == "discard":
                if act.stream_opened and act.seg_id:
                    transcriber.stream_abort(act.seg_id)
                if act.seg_id:
                    server.broadcast({"type": "discard", "id": act.seg_id,
                                      "ts": time.time()})
                    seg_open_at.pop(act.seg_id, None)
                    seg_seed_secs.pop(act.seg_id, None)
                    first_char_logged.discard(act.seg_id)
            elif act.kind == "flush":
                seg_id = act.seg_id
                if act.stream_opened:
                    transcriber.stream_finish(seg_id, task_id=seg_id,
                                              doa=locator.mode())
                else:
                    transcriber.submit(act.audio, task_id=seg_id,
                                       doa=locator.mode())
                speech_end_at[seg_id] = time.monotonic()
                seg_durations[seg_id] = act.seg_seconds
                seg_open_at.pop(seg_id, None)
                seg_seed_secs.pop(seg_id, None)
                if seg_id not in first_char_logged:
                    first_char_stats["miss"] += 1
                first_char_logged.discard(seg_id)

    try:
        with sd.InputStream(
            device=input_device, channels=in_channels,
            samplerate=SAMPLE_RATE, dtype="float32", callback=audio_callback,
        ):
            import signal

            def _on_term(_signum, _frame):
                raise KeyboardInterrupt

            for _sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    signal.signal(_sig, _on_term)
                except (ValueError, OSError):  # not the main thread / unavail.
                    pass

            print("[main] running - Ctrl+C to stop")
            while True:
                now = time.time()


                hw_vad = 0
                while True:
                    try:
                        snap = vad_reader.vad_queue.get_nowait() if vad_reader else None
                    except queue.Empty:
                        break
                    if snap is None:
                        continue
                    hw_vad = snap["hw_vad"]
                    if snap["doa"] is not None:
                        doa_last = snap["doa"]
                        if segmenter.active:
                            locator.observe(doa_last)
                chunks = []
                while True:
                    try:
                        chunks.append(audio_queue.get_nowait())
                    except queue.Empty:
                        break
                speaking = segmenter.active
                actions = segmenter.feed(chunks, gate_verdict, hw_vad, now)
                sw_prob = segmenter.last_prob
                gate = segmenter.last_gate

                if args.debug_gate and (hw_vad or sw_prob >= 0.05 or speaking):
                    print(f"[gate] hw={hw_vad} sw={sw_prob:.3f} gate={gate} "
                          f"state={'SPK' if speaking else 'IDLE'} "
                          f"doa={doa_last}")

                apply_actions(actions)

                while True:
                    try:
                        msg = server.cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                    if msg.get("cmd"):
                        resp = control.handle(msg)
                        server.broadcast({
                            "type": "cmd_result",
                            "ts": time.time(), **resp,
                        })

                while True:
                    try:
                        result = result_queue.get_nowait()
                    except queue.Empty:
                        break
                    task_id = result.meta.get("task_id", "")
                    is_final = bool(result.meta.get("final_replaces_partial"))
                    speech_end = speech_end_at.pop(task_id, None) if is_final else None
                    seg_len = seg_durations.pop(task_id, None) if is_final else None
                    if not result.text:
                        if is_final:
                            server.broadcast({"type": "discard", "id": task_id,
                                              "ts": time.time()})
                        continue
                    print(f"[asr] {task_id} final={is_final} "
                          f"mode={result.meta.get('mode', '')} "
                          f"decode={result.latency * 1000:.0f}ms "
                          f"text={result.text[:40]!r}")
                    if is_final and speech_end is not None:
                        src_e2e = time.monotonic() - speech_end
                        seg_txt = f" (段时长 {seg_len:.2f}s)" if seg_len else ""
                        print(f"[耗时] {task_id[:8]} 说完→原文上屏: "
                              f"{src_e2e:.2f}s{seg_txt}")
                    lang = detect_language(result.text)
                    server.broadcast({
                        "type": "final" if is_final else "source",
                        "id": task_id,
                        "ts": time.time(),
                        "doa": result.meta.get("doa", ""),
                        "text": result.text,
                        "lang": lang,
                    })
                    translation.submit(task_id, result.text, lang)

                server.broadcast({
                    "type": "vad.state",
                    "hw_vad": hw_vad,
                    "sw_prob": round(sw_prob, 3),
                    "gate_mode": "and",
                    "segment_state": "SPEAKING" if speaking else "IDLE",
                })

                elapsed = time.time() - now
                time.sleep(max(0.0, args.vad_interval - elapsed))
    except KeyboardInterrupt:
        print("\n[main] stopping")
    finally:
        summary = translation.summary()
        if summary:
            print(f"[耗时] 译文端到端汇总: {summary}")
        with first_char_lock:
            fn = first_char_stats["n"]
            fmiss = first_char_stats["miss"]
            fsum = first_char_stats["sum"]
        if fn:
            print(f"[首字] 汇总: {fn} 句有 partial, 均值 {fsum / fn:.2f}s"
                  + (f", 另 {fmiss} 句太短无 partial" if fmiss else ""))
        with first_char_lock:
            if segmenter.counters.flushed or segmenter.counters.discarded:
                print(f"[seg] {segmenter.summary()}")
        if vad_reader:
            vad_reader.stop()
        transcriber.stop()
        final_decode.stop()
        translation.stop()
        server.stop()
        provider.close()
        if device:
            device.close()
        print("[main] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
