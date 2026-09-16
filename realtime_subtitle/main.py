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
import uuid

import numpy as np

# Support both "python -m realtime_subtitle.main" and "python main.py":
# when executed as a script there is no package context, so inject the
# repo root into sys.path and register the package name before any
# relative import below runs.
if __package__ is None or __package__ == "":
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_pkg_dir)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    __package__ = "realtime_subtitle"

from .asr import AsrResult

# avoid XET download failures with huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

SAMPLE_RATE = 16000

# default segmentation parameters
VAD_INTERVAL = 0.1          # main loop tick, seconds
SILENCE_HOLD = 0.6          # trailing silence before flush
MIN_SECONDS = 0.5           # shortest valid segment
MAX_SECONDS = 20.0          # force-cut overlong segments
PRE_ROLL_SECONDS = 0.5      # buffer before segment start, merged on open
SW_FRAME_THRESHOLD = 0.5    # Silero per-frame speech threshold
SHORT_EXEMPT_SECONDS = 1.0  # short-segment exemption window
SHORT_EXEMPT_MAX_PROB = 0.5 # peak probability required for exemption
RATIO_THRESHOLD = 0.35      # segment speech-frame ratio floor
TRIM_TAIL_ENERGY = 1e-4     # RMS below which tail frames are trimmed


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

    def __init__(self, translator, server, speech_end_at: dict | None = None,
                 asr_done_at: dict | None = None):
        super().__init__(daemon=True, name="TranslationThread")
        self.translator = translator
        self.server = server
        self.speech_end_at = speech_end_at if speech_end_at is not None else {}
        self.asr_done_at = asr_done_at if asr_done_at is not None else {}
        self.job_queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()

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
                    "latency_ms": trans_ms,
                })
                # end-to-end stats: speech end -> translation on screen
                speech_end = self.speech_end_at.pop(job["task_id"], None)
                asr_done = self.asr_done_at.pop(job["task_id"], None)
                if speech_end is not None and asr_done is not None:
                    self.server.broadcast({
                        "type": "latency.update",
                        "id": job["task_id"],
                        "ts": time.time(),
                        "asr_ms": int((asr_done - speech_end) * 1000),
                        "trans_ms": trans_ms,
                        "e2e_ms": int((time.monotonic() - speech_end) * 1000),
                    })
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
            try:
                res = self.provider.transcribe(job["audio"])
                result = AsrResult(
                    text=res.text,
                    language=res.language,
                    latency=time.monotonic() - start,
                    meta={
                        "engine": self.provider.name,
                        "mode": "2pass-final",
                        "task_id": job["task_id"],
                        "doa": job["doa"],
                        "final_replaces_partial": True,
                        "partial_hint": job["partial_hint"],
                    },
                )
                self.on_result(result)
            except Exception as exc:  # noqa: BLE001
                print(f"[ASR] final decode {job['task_id']} failed: {exc}")

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# CLI / dependency resolution
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="reSpeaker real-time bilingual subtitles")
    p.add_argument("--asr", choices=["funasr", "whisper"], default="funasr")
    p.add_argument("--model", default="tiny", help="whisper model size")
    p.add_argument("--streaming", action="store_true",
                   help="FunASR streaming 2-pass with partial results")
    p.add_argument("--vad-gate", choices=["hw", "and", "confirm"], default="and")
    p.add_argument("--audio-source", default="conference",
                   choices=["conference", "asr", "mic0", "mic1", "mic2", "mic3"])
    p.add_argument("--ch", type=int, default=None, help="explicit channel index (0-based)")
    p.add_argument("--in-channels", type=int, default=None)
    p.add_argument("--device", default=None, help="input device name substring")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--vad-interval", type=float, default=VAD_INTERVAL)
    p.add_argument("--silence-hold", type=float, default=SILENCE_HOLD)
    p.add_argument("--min-seconds", type=float, default=MIN_SECONDS)
    p.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    p.add_argument("--beam-recon", action="store_true", help="read-only beam probe then exit")
    p.add_argument("--beam-select", action="store_true", help="apply beam route then exit")
    p.add_argument("--list-devices", action="store_true")
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
        raise RuntimeError(f"input device not found: {name}")
    # auto-detect the reSpeaker XVF3800 audio interface
    for d in devs:
        if "xvf3800" in d["name"].lower() or "reSpeaker XVF3800" in d["name"]:
            return d["index"]
    # hardware not found: fall back to the default device but say so
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
    mapping = {"conference": 0, "asr": 1, "mic0": 2, "mic1": 3, "mic2": 4, "mic3": 5}
    idx = mapping.get(args.audio_source, 0)
    if idx >= in_channels:
        idx = 0
    desc = {
        "conference": "Conference processed output (AEC+beam+post)",
        "asr": "ASR beam output (auto-select, no post)",
        "mic0": "raw mic 0", "mic1": "raw mic 1",
        "mic2": "raw mic 2", "mic3": "raw mic 3",
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

    # ---- device control plane (optional for audio-only runs) ----------
    device = None
    try:
        from .usb_device import ReSpeaker

        device = ReSpeaker()
        print("[USB] XVF3800 control plane connected")
    except Exception as exc:  # noqa: BLE001
        print(f"[USB] control plane unavailable: {exc}")

    if args.beam_recon or args.beam_select:
        if device is None:
            print("beam tools require the USB control plane")
            return 1
        from .beam_router import BeamRouter

        router = BeamRouter(device)
        recon = router.recon()
        info = router.describe(recon["op_all"])
        print(f"[beam] AUDIO_MGR_OP_ALL={recon['op_all']}")
        print(f"[beam] describe={info}")
        if args.beam_select:
            if not info.get("ok"):
                print("[beam] mapping ambiguous, refusing to write")
                return 1
            candidate = info["pairs"][2]
            print(f"[beam] applying candidate {candidate} -> slot R")
            print(router.apply(candidate, "R"))
        return 0

    # ---- audio input ---------------------------------------------------
    input_device = resolve_input_device(args.device)
    dev_info = sd.query_devices(input_device, "input")
    in_channels = args.in_channels or dev_info["max_input_channels"]
    ch_index, ch_desc = resolve_audio_channel(args, in_channels)
    print(f"[audio] device {input_device}, {in_channels} ch, extract ch{ch_index + 1}: {ch_desc}")

    from .speaker_locator import SpeakerLocator

    locator = SpeakerLocator()

    # ---- inference providers --------------------------------------------
    if args.asr == "funasr":
        from .funasr_provider import FunasrProvider

        provider = FunasrProvider()
        print("[ASR] FunASR Paraformer (zh, streaming 2pass)" if args.streaming
              else "[ASR] FunASR Paraformer (zh, offline)")
    else:
        from .whisper_provider import WhisperProvider

        provider = WhisperProvider(args.model)
        print(f"[ASR] Whisper {args.model}")

    from .translator import NllbCTranslate2Provider

    translator = NllbCTranslate2Provider()
    print(f"[NLLB] model: {translator.model_path}")

    from .sw_vad import SileroVad

    sw_vad = SileroVad(threshold=SW_FRAME_THRESHOLD)
    print("[VAD] Silero software VAD ready")

    # ---- servers ---------------------------------------------------------
    from .control import ControlHandler
    from .ws_server import BroadcastServer

    server = BroadcastServer(port=args.port)
    server.start()
    control = ControlHandler(device, locator, translator)
    print(f"[WS] subtitle service: http://0.0.0.0:{args.port}/")

    # per-segment timing anchors for end-to-end latency stats
    speech_end_at: dict[str, float] = {}
    asr_done_at: dict[str, float] = {}

    translation = TranslationThread(translator, server,
                                     speech_end_at=speech_end_at,
                                     asr_done_at=asr_done_at)
    translation.start()

    result_queue: queue.Queue = queue.Queue()

    def on_asr_result(result):
        result_queue.put(result)

    def on_partial(stream_id, text):
        server.broadcast({
            "type": "partial", "id": stream_id, "ts": time.time(), "text": text,
        })

    from .transcriber import TranscriberWorker

    final_decode = FinalDecodeThread(provider, on_result=on_asr_result)
    final_decode.start()

    def on_final_request(req):
        final_decode.submit(req.task_id, req.audio, doa=req.doa,
                            partial_hint=req.partial_hint)

    transcriber = TranscriberWorker(provider, on_result=on_asr_result,
                                    on_partial=on_partial,
                                    on_final_request=on_final_request)
    transcriber.start()

    vad_reader = None
    if device is not None:
        vad_reader = VadDoaReader(device, interval=args.vad_interval)
        vad_reader.start()

    # ---- segmentation state machine ---------------------------------
    audio_queue: queue.Queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass  # overflow flags are tolerated; queue drops are the backpressure
        audio_queue.put(indata[:, ch_index].copy())

    speaking = False
    utterance: list[np.ndarray] = []
    pre_roll: list[np.ndarray] = []
    seg_start = 0.0
    last_speech_at = 0.0
    hw_speech_ticks = 0
    sw_speech_frames = 0
    total_frames = 0            # counted only up to last_speech_at
    prob_sum = 0.0
    max_prob = 0.0
    current_id = ""
    stream_opened = False
    doa_last = None
    # confirm-mode state
    confirm_pending = False
    confirm_sw_reject = 0

    def gate_open(hw_vad: int, sw_prob: float) -> bool:
        nonlocal confirm_pending, confirm_sw_reject
        if args.vad_gate == "hw":
            return hw_vad == 1
        if args.vad_gate == "and":
            return hw_vad == 1 and sw_prob >= SW_FRAME_THRESHOLD
        # confirm: hw opens the door, software decides entry/exit
        if confirm_pending:
            if sw_prob >= SW_FRAME_THRESHOLD:
                confirm_sw_reject = 0
                return True
            confirm_sw_reject += 1
            if confirm_sw_reject >= 5:
                confirm_pending = False
                confirm_sw_reject = 0
                return False
            return True
        if hw_vad == 1:
            confirm_pending = True
            confirm_sw_reject = 0
            return sw_prob >= SW_FRAME_THRESHOLD
        return False

    def discard_segment(reason: str = "") -> None:
        """Single discard path: abort stream, broadcast, reset, go idle."""
        nonlocal speaking, utterance, hw_speech_ticks, sw_speech_frames
        nonlocal total_frames, prob_sum, max_prob, current_id, stream_opened
        if stream_opened and current_id:
            transcriber.stream_abort(current_id)
        if current_id:
            server.broadcast({"type": "discard", "id": current_id, "ts": time.time()})
        if args.debug_gate and current_id:
            ratio = sw_speech_frames / total_frames if total_frames else 0.0
            avg = prob_sum / total_frames if total_frames else 0.0
            print(f"[VAD] discard {current_id} ({reason}): dur={audio_dur():.2f}s "
                  f"ratio={ratio:.2f} avg_prob={avg:.2f} max_prob={max_prob:.2f} "
                  f"hw_ticks={hw_speech_ticks}/{total_frames * SAMPLE_RATE // 512 or 1}")
        utterance = []
        hw_speech_ticks = 0
        sw_speech_frames = 0
        total_frames = 0
        prob_sum = 0.0
        max_prob = 0.0
        current_id = ""
        stream_opened = False
        speaking = False
        sw_vad.reset()  # legal reset point: utterance truly ended

    def audio_dur() -> float:
        return sum(c.size for c in utterance) / SAMPLE_RATE if utterance else 0.0

    def flush_segment() -> None:
        nonlocal speaking, utterance, hw_speech_ticks, sw_speech_frames
        nonlocal total_frames, prob_sum, max_prob, current_id, stream_opened
        audio = np.concatenate(utterance) if utterance else np.zeros(0, np.float32)
        utterance = []
        duration = audio.size / SAMPLE_RATE
        seg_id = current_id
        current_id = ""
        was_stream = stream_opened
        stream_opened = False

        # level 1: duration
        if duration < args.min_seconds:
            discard_segment(f"duration {duration:.2f}s < min {args.min_seconds}s")
            return
        # level 2: speech ratio (denominator excludes trailing silence)
        ratio = sw_speech_frames / total_frames if total_frames else 0.0
        avg_prob = prob_sum / total_frames if total_frames else 0.0
        exempt = (duration < SHORT_EXEMPT_SECONDS
                  and max_prob >= SHORT_EXEMPT_MAX_PROB)
        if args.debug_gate:
            print(f"[flush] dur={duration:.2f}s ratio={ratio:.2f} avg_prob={avg_prob:.2f} "
                  f"max_prob={max_prob:.2f} exempt={exempt} hw_ticks={hw_speech_ticks}")
        if ratio < RATIO_THRESHOLD and not exempt:
            discard_segment(f"ratio {ratio:.2f} < {RATIO_THRESHOLD} (avg_prob={avg_prob:.2f})")
            return
        # level 3: energy (trim trailing silence, then floor RMS)
        audio = _trim_tail(audio)
        if audio.size and float(np.sqrt(np.mean(audio ** 2))) < np.sqrt(TRIM_TAIL_ENERGY):
            discard_segment(f"energy {float(np.sqrt(np.mean(audio ** 2)))**2:.2e} < {TRIM_TAIL_ENERGY:.2e}")
            return

        if was_stream and seg_id:
            transcriber.stream_finish(seg_id, task_id=seg_id, doa=locator.mode())
        else:
            transcriber.submit(audio, task_id=seg_id, doa=locator.mode())
        # reset for next segment (without discarding this one)
        hw_speech_ticks = 0
        sw_speech_frames = 0
        total_frames = 0
        prob_sum = 0.0
        max_prob = 0.0
        speaking = False
        sw_vad.reset()

    def _trim_tail(audio: np.ndarray) -> np.ndarray:
        n = audio.size
        step = 512
        while n > step:
            if float(np.sqrt(np.mean(audio[n - step:n] ** 2))) > np.sqrt(TRIM_TAIL_ENERGY):
                break
            n -= step
        return audio[:n]

    try:
        with sd.InputStream(
            device=input_device, channels=in_channels,
            samplerate=SAMPLE_RATE, dtype="float32", callback=audio_callback,
        ):
            print("[main] running - Ctrl+C to stop")
            while True:
                now = time.time()

                # 2) drain audio
                chunks = []
                while True:
                    try:
                        chunks.append(audio_queue.get_nowait())
                    except queue.Empty:
                        break

                # 3) drain hw VAD / DOA
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
                        if speaking:
                            locator.observe(doa_last)

                # 3.5) software VAD on this tick's audio
                sw_prob = sw_vad.feed(np.concatenate(chunks)) if chunks else sw_vad.last_probability
                for c in chunks:
                    pass  # stats computed below on the tick level

                gate = gate_open(hw_vad, sw_prob)
                frame_now = int(now * 1000)

                if args.debug_gate and (hw_vad or sw_prob >= 0.05 or speaking):
                    print(f"[gate] hw={hw_vad} sw={sw_prob:.3f} gate={gate} "
                          f"state={'SPK' if speaking else 'IDLE'} doa={doa_last}")

                # 4) route audio (three destinations)
                for c in chunks:
                    if speaking:
                        utterance.append(c)
                    elif not speaking and hw_vad == 1:
                        pre_roll.append(c)
                    else:
                        pass
                # streaming: one merged feed per tick (100ms) instead of
                # one task per callback block - the worker queue drains
                # 3-4x fewer tasks and partials surface earlier
                if speaking and stream_opened and chunks:
                    transcriber.stream_feed(
                        current_id,
                        np.concatenate(chunks) if len(chunks) > 1 else chunks[0],
                        current_id)
                if not speaking and pre_roll and gate:
                    pass  # merged at segment open below

                # state transitions
                if not speaking:
                    if gate:
                        speaking = True
                        seg_start = now
                        last_speech_at = now
                        current_id = uuid.uuid4().hex[:12]
                        if pre_roll:
                            utterance = list(pre_roll)
                            pre_roll = []
                        if args.streaming and provider.supports_streaming:
                            transcriber.stream_start(current_id)
                            transcriber.stream_feed(current_id, np.concatenate(utterance) if utterance else np.zeros(0, np.float32), current_id)
                            stream_opened = True
                        print(f"[seg] open {current_id}")
                else:
                    # accumulate statistics only while speech may be active
                    if chunks:
                        n = sum(c.size for c in chunks)
                        # count frames only up to last speech for ratio denominator
                        if gate:
                            sw_speech_frames += n
                            total_frames += n
                            last_speech_at = now
                        else:
                            total_frames += n
                    prob_sum += sw_prob * (sum(c.size for c in chunks))
                    max_prob = max(max_prob, sw_prob)
                    hw_speech_ticks += 1 if hw_vad else 0

                    silence = now - last_speech_at
                    overlong = now - seg_start >= args.max_seconds
                    # fast-cut: Silero confirms the silence earlier than the
                    # fixed hold - once the trailing silence exceeds 0.25s
                    # and the current frame is clearly non-speech, the
                    # utterance is over; no need to wait the full 0.6s hold
                    fast_cut = (
                        speaking
                        and silence > 0.25
                        and sw_prob < 0.2
                    )
                    if silence >= args.silence_hold or overlong or fast_cut:
                        seg_id_before_flush = current_id
                        print(f"[seg] flush after {now - seg_start:.2f}s")
                        flush_segment()
                        if seg_id_before_flush:
                            speech_end_at[seg_id_before_flush] = time.monotonic()
                        pre_roll = []

                # 5) frontend commands
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

                # 6) drain ASR results
                while True:
                    try:
                        result = result_queue.get_nowait()
                    except queue.Empty:
                        break
                    task_id = result.meta.get("task_id", "")
                    if not result.text:
                        continue
                    is_final = bool(result.meta.get("final_replaces_partial"))
                    if args.debug_gate:
                        print(f"[asr] {task_id} final={is_final} lang={result.language} "
                              f"lat={result.latency:.2f}s text={result.text[:60]!r}")
                    server.broadcast({
                        "type": "final" if is_final else "source",
                        "id": task_id,
                        "ts": time.time(),
                        "doa": result.meta.get("doa", ""),
                        "text": result.text,
                        "lang": result.language,
                        "latency_ms": int(result.latency * 1000),
                    })
                    asr_done_at[task_id] = time.monotonic()
                    translation.submit(task_id, result.text, result.language)

                # heartbeat broadcast
                server.broadcast({
                    "type": "vad.state",
                    "hw_vad": hw_vad,
                    "sw_prob": round(sw_prob, 3),
                    "gate_mode": args.vad_gate,
                    "segment_state": "SPEAKING" if speaking else "IDLE",
                })

                elapsed = time.time() - now
                time.sleep(max(0.0, args.vad_interval - elapsed))
    except KeyboardInterrupt:
        print("\n[main] stopping")
    finally:
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
