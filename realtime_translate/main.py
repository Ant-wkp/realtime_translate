from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

if __package__ is None or __package__ == "":
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_pkg_dir)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    __package__ = "realtime_translate"

from .asr import AsrResult, detect_language
from .compute import DEVICE_CHOICES, describe_platform
from .segmenter import SegmentCfg, Segmenter, trim_tail

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

SAMPLE_RATE = 16000
VAD_INTERVAL = 0.1
SILENCE_HOLD = 0.6
MIN_SECONDS = 0.3
MAX_SECONDS = 20.0
PRE_ROLL = 1.0
SW_FRAME_THRESHOLD = 0.5
SHORT_EXEMPT_SECONDS = 1.0
SHORT_EXEMPT_MAX_PROB = 0.5
RATIO_THRESHOLD = 0.35
TRIM_TAIL_ENERGY = 1e-4
USB_PROBE_TIMEOUT = 2.0
SHUTDOWN_TIMEOUT = 5.0

ARRAY_DEVICE_NAMES = ("respeaker xvf3800", "xvf3800", "respeaker")
CAPTURE_CHANNEL = 0

@dataclass(frozen=True)
class SubtitleEvent:
    kind: str
    task_id: str
    text: str = ""
    target_lang: str = ""
    error: str = ""
    result: AsrResult | None = None

@dataclass
class SegmentRecord:
    opened_at: float
    seed_secs: float
    state: str = "OPEN"
    speech_end_at: float | None = None
    seg_seconds: float = 0.0
    first_char_seen: bool = False

class SubtitleDispatcher:
    def __init__(self, server, translation, clock=time.monotonic,
                 wall_clock=time.time, log=print):
        self.server = server
        self.translation = translation
        self.clock = clock
        self.wall_clock = wall_clock
        self.log = log
        self.segments: dict[str, SegmentRecord] = {}
        self.first_char_stats = {"n": 0, "sum": 0.0, "miss": 0}
        self.source_stats = {"n": 0, "sum": 0.0}
        self.translation_stats = {"n": 0, "sum": 0.0, "failed": 0}

    def open(self, task_id: str, seed_secs: float = 0.0,
             opened_at: float | None = None) -> None:
        self.segments[task_id] = SegmentRecord(
            self.clock() if opened_at is None else opened_at, seed_secs)

    def finish(self, task_id: str, speech_end_at: float | None,
               seg_seconds: float) -> None:
        record = self.segments.get(task_id)
        if record is not None and record.state == "OPEN":
            record.state = "FINISHING"
            record.speech_end_at = speech_end_at
            record.seg_seconds = seg_seconds

    def discard(self, task_id: str) -> None:
        if self.segments.pop(task_id, None) is not None:
            self.server.broadcast({"type": "discard", "id": task_id,
                                   "ts": self.wall_clock()})

    def handle(self, event: SubtitleEvent) -> None:
        record = self.segments.get(event.task_id)
        if record is None:
            return
        if event.kind == "partial":
            if record.state not in ("OPEN", "FINISHING") or not event.text:
                return
            self.server.broadcast({"type": "partial", "id": event.task_id,
                                   "ts": self.wall_clock(), "text": event.text})
            if not record.first_char_seen:
                record.first_char_seen = True
                dt = self.clock() - record.opened_at
                stats = self.first_char_stats
                stats["n"] += 1
                stats["sum"] += dt
                self.log(f"[首字] {event.task_id[:8]} 开段→首个 partial 广播入队: "
                         f"{dt:.2f}s (均值 {stats['sum'] / stats['n']:.2f}s / "
                         f"{stats['n']} 句, seed {record.seed_secs:.2f}s)")
        elif event.kind == "asr_error":
            if record.state in ("OPEN", "FINISHING"):
                self.discard(event.task_id)
        elif event.kind == "asr_result":
            if record.state != "FINISHING" or event.result is None:
                return
            result = event.result
            if not result.text:
                self.discard(event.task_id)
                return
            record.state = "FINALIZED"
            if not record.first_char_seen:
                self.first_char_stats["miss"] += 1
            lang = detect_language(result.text)
            is_final = bool(result.meta.get("final_replaces_partial"))
            self.server.broadcast({
                "type": "final" if is_final else "source", "id": event.task_id,
                "ts": self.wall_clock(), "doa": result.meta.get("doa", ""),
                "text": result.text, "lang": lang,
            })
            if record.speech_end_at is not None:
                dt = self.clock() - record.speech_end_at
                self.source_stats["n"] += 1
                self.source_stats["sum"] += dt
                self.log(f"[耗时] {event.task_id[:8]} 末次门开→原文广播入队: "
                         f"{dt:.2f}s (段时长 {record.seg_seconds:.2f}s)")
            self.log(f"[asr] {event.task_id} final={is_final} "
                     f"mode={result.meta.get('mode', '')} "
                     f"decode={result.latency * 1000:.0f}ms "
                     f"text={result.text[:40]!r}")
            self.translation.submit(event.task_id, result.text, lang)
        elif event.kind == "translation":
            if record.state != "FINALIZED":
                return
            msg = {"type": "translation", "id": event.task_id,
                   "ts": self.wall_clock(), "text": event.text,
                   "target_lang": event.target_lang}
            if event.error:
                msg["error"] = event.error
            self.server.broadcast(msg)
            stats = self.translation_stats
            if event.error:
                stats["failed"] += 1
                self.log(f"[NLLB] {event.task_id} translate failed: {event.error}")
            elif record.speech_end_at is not None:
                dt = self.clock() - record.speech_end_at
                stats["n"] += 1
                stats["sum"] += dt
                self.log(f"[耗时] {event.task_id[:8]} 末次门开→译文广播入队: "
                         f"{dt:.2f}s (均值 {stats['sum'] / stats['n']:.2f}s / "
                         f"{stats['n']} 句)")
            record.state = "COMPLETE"
            del self.segments[event.task_id]

    def summary(self) -> None:
        stats = self.first_char_stats
        if stats["n"] or stats["miss"]:
            avg = stats["sum"] / stats["n"] if stats["n"] else 0.0
            self.log(f"[首字] 汇总: {stats['n']} 句有 partial, 均值 {avg:.2f}s, "
                     f"另 {stats['miss']} 句无 partial")
        stats = self.translation_stats
        if stats["n"]:
            self.log(f"[耗时] 译文广播入队汇总: {stats['n']} 句, "
                     f"均值 {stats['sum'] / stats['n']:.2f}s")
        if stats["failed"]:
            self.log(f"[NLLB] 翻译失败 {stats['failed']} 句（未计入成功耗时均值）")

class VadDoaReader(threading.Thread):
    def __init__(self, device, interval: float = 0.1):
        super().__init__(daemon=True, name="VadDoaReader")
        self.device = device
        self.interval = interval
        self.vad_queue: queue.Queue = queue.Queue()
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                vals = self.device.read("DOA_VALUE", stop_event=self._stop_evt)
                snap = {"hw_vad": int(vals[1]) if len(vals) > 1 else 0,
                        "doa": vals[0] if vals else None, "error": False}
            except Exception:
                snap = {"hw_vad": 0, "doa": None, "error": True}
            if self._stop_evt.is_set():
                break
            self.vad_queue.put(snap)
            self._stop_evt.wait(self.interval)

    def stop(self) -> None:
        self._stop_evt.set()

class TranslationThread(threading.Thread):
    def __init__(self, translator, on_result):
        super().__init__(daemon=True, name="TranslationThread")
        self.translator = translator
        self.on_result = on_result
        self.job_queue: queue.Queue = queue.Queue()
        self._stop_evt = threading.Event()

    def submit(self, task_id: str, text: str, lang: str) -> None:
        if not self._stop_evt.is_set():
            self.job_queue.put({"task_id": task_id, "text": text, "lang": lang})

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                job = self.job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._stop_evt.is_set():
                break
            start = time.monotonic()
            try:
                text, target = self.translator.translate(job["text"], job["lang"])
                trans_ms = int((time.monotonic() - start) * 1000)
                print(f"[tr] {job['task_id']} -> {target} in {trans_ms}ms: {text[:60]!r}")
                event = SubtitleEvent("translation", job["task_id"], text=text,
                                      target_lang=target)
            except Exception as exc:
                event = SubtitleEvent("translation", job["task_id"],
                                      error=str(exc) or type(exc).__name__)
            if not self._stop_evt.is_set():
                self.on_result(event)

    def stop(self) -> None:
        self._stop_evt.set()

class FinalDecodeThread(threading.Thread):
    def __init__(self, provider, on_result):
        super().__init__(daemon=True, name="FinalDecodeThread")
        self.provider = provider
        self.on_result = on_result
        self.job_queue: queue.Queue = queue.Queue()
        self._stop_evt = threading.Event()

    def submit(self, task_id: str, audio, doa: str = "", partial_hint: str = "") -> None:
        if not self._stop_evt.is_set():
            self.job_queue.put({"task_id": task_id, "audio": audio,
                                "doa": doa, "partial_hint": partial_hint})

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                job = self.job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._stop_evt.is_set():
                break
            start = time.monotonic()
            mode = "2pass-final"
            language = ""
            try:
                res = self.provider.transcribe(job["audio"])
                text = (res.text or "").strip()
                language = res.language
            except Exception as exc:
                print(f"[ASR] final decode {job['task_id']} failed: {exc}")
                text = ""
                mode = "2pass-final-failed"
            if not text:
                text = (job["partial_hint"] or "").strip()
                if text:
                    mode = "2pass-final-partial"
            result = AsrResult(
                text=text, language=language, latency=time.monotonic() - start,
                meta={"engine": self.provider.name, "mode": mode,
                      "task_id": job["task_id"], "doa": job["doa"],
                      "final_replaces_partial": True,
                      "partial_hint": job["partial_hint"]},
            )
            if not self._stop_evt.is_set():
                self.on_result(result)

    def stop(self) -> None:
        self._stop_evt.set()

def connect_control(no_hw_vad: bool, timeout: float = USB_PROBE_TIMEOUT):
    device = None
    try:
        from .usb_device import ReSpeaker

        deadline = time.monotonic() + timeout
        device = ReSpeaker()
        vals = device.read("DOA_VALUE", deadline=deadline)
        if len(vals) < 2:
            raise RuntimeError("DOA_VALUE has no hardware speech flag")
        print("[USB] XVF3800 control plane connected (DOA_VALUE verified)")
        return device
    except BaseException as exc:
        if device is not None:
            try:
                device.close()
            except Exception as close_exc:
                print(f"[USB] cleanup failed: {close_exc}")
        if not isinstance(exc, Exception):
            raise
        if not no_hw_vad:
            raise RuntimeError(
                f"USB control plane unavailable: {exc}. Re-run with "
                "--no-hw-vad to explicitly use software VAD alone.") from exc
        print(f"[USB] control plane unavailable: {exc}; using software VAD alone")
        return None

def shutdown_resources(workers, resources, timeout: float = SHUTDOWN_TIMEOUT,
                       clock=time.monotonic) -> bool:
    deadline = clock() + timeout
    clean = True
    for worker in workers:
        try:
            worker.stop()
        except Exception as exc:
            print(f"[shutdown] cannot stop {worker.name}: {exc}")
            clean = False
    for worker in workers:
        try:
            if worker.is_alive():
                worker.join(max(0.0, deadline - clock()))
            if worker.is_alive():
                print(f"[shutdown] timeout waiting for {worker.name}")
                clean = False
        except Exception as exc:
            print(f"[shutdown] cannot join {worker.name}: {exc}")
            clean = False
    for name, resource, owners in resources:
        if resource is None:
            continue
        if any(owner is not None and owner.is_alive() for owner in owners):
            print(f"[shutdown] keeping {name} open: worker still using it")
            clean = False
            continue
        try:
            resource.close()
        except Exception as exc:
            print(f"[shutdown] cannot close {name}: {exc}")
            clean = False
    return clean

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="reSpeaker real-time translation")
    p.add_argument("--asr-chunk-size", choices=["480", "600"], default="480",
                   help="streaming decode stride: 480ms (default, lower latency) "
                        "or 600ms (upstream default, longer lookahead)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--vad-interval", type=float, default=VAD_INTERVAL,
                   help="main-loop tick and DOA poll interval in seconds; also "
                        "bounds how soon a segment can be flushed")
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
    p.add_argument("--no-hw-vad", action="store_true",
                   help="gate on the software VAD alone. Needed when the "
                        "XVF3800 control plane is unavailable; the default "
                        "hardware mode fails startup instead of silently "
                        "running without recognition")
    p.add_argument("--host", default="0.0.0.0",
                   help="web server bind address (default 0.0.0.0 so the "
                        "page is reachable from the LAN)")
    p.add_argument("--debug-gate", action="store_true",
                   help="print per-tick gate inputs and every discard reason")
    return p.parse_args(argv)

def resolve_input_device() -> int:
    import sounddevice as sd

    devs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
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
          f"input '{fallback_name}'")
    return default

def main(argv=None) -> int:
    args = parse_args(argv)
    import signal

    device = provider = translator = sw_vad = None
    transcriber = final_decode = translation = vad_reader = None
    dispatcher = segmenter = server = None
    workers = []
    closing = threading.Event()
    events: queue.Queue = queue.Queue()
    signal_handlers = {}
    exit_code = 0

    def on_term(_signum, _frame):
        if not closing.is_set():
            closing.set()
            raise KeyboardInterrupt

    def post(event):
        if not closing.is_set():
            events.put(event)

    def on_asr_result(result):
        post(SubtitleEvent("asr_result", result.meta.get("task_id", ""), result=result))

    def on_partial(stream_id, text):
        post(SubtitleEvent("partial", stream_id, text=text))

    def on_asr_error(task_id, error):
        post(SubtitleEvent("asr_error", task_id, error=error))

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                previous = signal.getsignal(sig)
                signal.signal(sig, on_term)
                signal_handlers[sig] = previous
            except (ValueError, OSError):
                pass

        device = connect_control(args.no_hw_vad)
        import sounddevice as sd

        input_device = resolve_input_device()
        dev_info = sd.query_devices(input_device, "input")
        in_channels = dev_info["max_input_channels"]
        ch_index = CAPTURE_CHANNEL
        if not 0 <= ch_index < in_channels:
            raise RuntimeError(
                f"capture channel {ch_index} is out of range for "
                f"'{dev_info['name']}' ({in_channels} input channels)")
        print(f"[audio] device {input_device}, {in_channels} ch, "
              f"extract ch{ch_index + 1} (conference processed output)")
        print(f"[compute] {describe_platform()}")

        from .speaker_locator import SpeakerLocator
        from .sw_vad import SileroVad
        from .funasr_provider import FunasrProvider
        from .translator import DEFAULT_BEAM_SIZE, NllbCTranslate2Provider
        from .control import ControlHandler
        from .ws_server import BroadcastServer
        from .transcriber import TranscriberWorker

        locator = SpeakerLocator()
        chunk = (FunasrProvider.CHUNK_SIZE_480 if args.asr_chunk_size == "480"
                 else FunasrProvider.CHUNK_SIZE_600)
        provider = FunasrProvider(chunk_size=chunk, device=args.device_type)
        print(f"[ASR] FunASR Paraformer (zh, streaming 2pass, "
              f"stride {chunk[1] * 60}ms, device {provider.describe_device()})")

        translator = NllbCTranslate2Provider(
            device=args.nllb_device or args.device_type,
            compute_type=args.nllb_compute_type,
            beam_size=args.nllb_beam or DEFAULT_BEAM_SIZE)
        print(f"[NLLB] model: {translator.model_path} "
              f"({translator.device}, {translator.compute_type}, "
              f"beam {translator.beam_size}, "
              f"{translator.zh_to}<-zh / {translator.other_to}<-other)")
        sw_vad = SileroVad(threshold=SW_FRAME_THRESHOLD)
        print(f"[VAD] Silero software VAD ready (device {sw_vad.device})")

        server = BroadcastServer(host=args.host, port=args.port)
        workers.append(server)
        server.start()
        control = ControlHandler(locator, translator)
        print(f"[WS] translation service: http://{args.host}:{server.port}/ "
              "(from another machine use this host's IP)")

        translation = TranslationThread(translator, on_result=post)
        dispatcher = SubtitleDispatcher(server, translation,
                                        clock=time.monotonic, wall_clock=time.time)
        final_decode = FinalDecodeThread(provider, on_result=on_asr_result)

        def on_final_request(req):
            if closing.is_set():
                return
            audio = (trim_tail(req.audio, TRIM_TAIL_ENERGY)
                     if req.audio is not None and req.audio.size else req.audio)
            final_decode.submit(req.task_id, audio, doa=req.doa,
                                partial_hint=req.partial_hint)

        transcriber = TranscriberWorker(
            provider, on_result=on_asr_result, on_partial=on_partial,
            on_final_request=on_final_request, on_error=on_asr_error)
        for worker in (translation, final_decode, transcriber):
            workers.append(worker)
            worker.start()
        if device is not None:
            vad_reader = VadDoaReader(device, interval=args.vad_interval)
            workers.append(vad_reader)
            vad_reader.start()

        seg_cfg = SegmentCfg(
            min_seconds=MIN_SECONDS, max_seconds=MAX_SECONDS,
            silence_hold=SILENCE_HOLD, pre_roll=PRE_ROLL,
            short_exempt_seconds=SHORT_EXEMPT_SECONDS,
            short_exempt_max_prob=SHORT_EXEMPT_MAX_PROB,
            ratio_threshold=RATIO_THRESHOLD, trim_tail_energy=TRIM_TAIL_ENERGY,
            debug=args.debug_gate)
        segmenter = Segmenter(seg_cfg, sw_vad)
        audio_queue: queue.Queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if not closing.is_set():
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
                    dispatcher.open(act.seg_id, act.seed_secs, opened_at=now)
                    locator.reset()
                    transcriber.stream_start(act.seg_id)
                    transcriber.stream_feed(
                        act.seg_id, act.audio if act.audio is not None
                        else np.zeros(0, np.float32), act.seg_id)
                    segmenter.mark_stream_open()
                    print(f"[seg] open {act.seg_id} (seed {act.seed_secs:.2f}s)")
                elif act.kind == "feed":
                    transcriber.stream_feed(act.seg_id, act.audio, act.seg_id)
                elif act.kind == "discard":
                    dispatcher.discard(act.seg_id)
                    if act.stream_opened and act.seg_id:
                        transcriber.stream_abort(act.seg_id)
                elif act.kind == "flush":
                    dispatcher.finish(act.seg_id, act.speech_end_at, act.seg_seconds)
                    if act.stream_opened:
                        transcriber.stream_finish(act.seg_id, task_id=act.seg_id,
                                                  doa=locator.mode())
                    else:
                        transcriber.submit(act.audio, task_id=act.seg_id,
                                           doa=locator.mode())

        with sd.InputStream(
            device=input_device, channels=in_channels, samplerate=SAMPLE_RATE,
            dtype="float32", callback=audio_callback,
        ):
            print("[main] running - Ctrl+C to stop")
            while not closing.is_set():
                now = time.monotonic()
                hw_vad = 0
                doa_last = None
                while vad_reader is not None:
                    try:
                        snap = vad_reader.vad_queue.get_nowait()
                    except queue.Empty:
                        break
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
                actions = segmenter.feed(chunks, gate_verdict, hw_vad, now)
                sw_prob = segmenter.last_prob
                gate = segmenter.last_gate
                apply_actions(actions)

                if args.debug_gate and (hw_vad or sw_prob >= 0.05 or segmenter.active):
                    doa_txt = doa_last if doa_last is not None else "-"
                    print(f"[gate] hw={hw_vad} sw={sw_prob:.3f} gate={gate} "
                          f"state={'SPK' if segmenter.active else 'IDLE'} doa={doa_txt}")

                while True:
                    try:
                        msg = server.cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                    if isinstance(msg, dict) and msg.get("cmd"):
                        server.broadcast({"type": "cmd_result", "ts": time.time(),
                                          **control.handle(msg)})
                while True:
                    try:
                        event = events.get_nowait()
                    except queue.Empty:
                        break
                    if (event.kind == "asr_error" and segmenter.active
                            and segmenter.current_id == event.task_id):
                        apply_actions([segmenter.discard(f"asr {event.error}")])
                    else:
                        dispatcher.handle(event)

                server.broadcast({
                    "type": "vad.state", "hw_vad": hw_vad,
                    "sw_prob": round(sw_prob, 3),
                    "gate_mode": "software" if args.no_hw_vad else "and",
                    "segment_state": "SPEAKING" if segmenter.active else "IDLE",
                })
                elapsed = time.monotonic() - now
                time.sleep(max(0.0, args.vad_interval - elapsed))
    except KeyboardInterrupt:
        print("\n[main] stopping")
    except Exception as exc:
        print(f"[main] error: {exc}")
        exit_code = 1
    finally:
        closing.set()
        resources = [
            ("software VAD", sw_vad, ()),
            ("translator", translator, (translation,)),
            ("ASR provider", provider, (transcriber, final_decode)),
            ("USB device", device, (vad_reader,)),
        ]
        if not shutdown_resources(workers, resources):
            exit_code = 1
        if dispatcher is not None:
            dispatcher.summary()
            dispatcher.segments.clear()
        if segmenter is not None and (segmenter.counters.flushed
                                      or segmenter.counters.discarded):
            print(f"[seg] {segmenter.summary()}")
        for sig, previous in signal_handlers.items():
            signal.signal(sig, previous)
        print("[main] stopped")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
