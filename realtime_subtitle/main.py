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
from collections import deque

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

from .asr import AsrResult, detect_language

# avoid XET download failures with huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

SAMPLE_RATE = 16000

# default segmentation parameters
VAD_INTERVAL = 0.1          # main loop tick, seconds
SILENCE_HOLD = 0.6          # trailing silence before flush
# Shortest valid segment, measured as *voiced* time. This number has to be
# interpreted against real speech only: the trailing-silence hold runs for a
# full silence_hold and used to be counted into duration, which inflated every
# segment by ~0.6s and let the filter pass near-silence. With duration now
# honest, 0.5 would reject ordinary two-character replies, so it is 0.3.
MIN_SECONDS = 0.3
MAX_SECONDS = 20.0          # force-cut overlong segments
PRE_ROLL_DEFAULT = 1.0      # ring-buffer window retained before segment start
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
    p.add_argument("--asr", choices=["funasr", "whisper"], default="funasr")
    p.add_argument("--model", default="tiny", help="whisper model size")
    # Streaming 2-pass is the default: partials are what make subtitles feel
    # live, so it takes --no-streaming to opt out rather than a flag to opt in.
    p.add_argument("--streaming", dest="streaming", action="store_true", default=True,
                   help="FunASR streaming 2-pass with partial results (default)")
    p.add_argument("--no-streaming", dest="streaming", action="store_false",
                   help="disable streaming: decode whole segments offline only")
    p.add_argument("--asr-chunk-size", choices=["480", "600"], default="480",
                   help="streaming decode stride: 480ms (default, lower latency) "
                        "or 600ms (upstream default, longer lookahead)")
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
    p.add_argument("--pre-roll", type=float, default=PRE_ROLL_DEFAULT,
                   help="seconds of audio retained before segment start; the "
                        "first streaming chunk is filled from this ring buffer "
                        "so the model decodes the opening words immediately "
                        "instead of waiting 600ms for a full chunk")
    p.add_argument("--beam-recon", action="store_true", help="read-only beam probe then exit")
    p.add_argument("--beam-probe", action="store_true",
                   help="read-only survey of every beam parameter then exit")
    p.add_argument("--beam-fixed", nargs=2, type=float, metavar=("AZ1", "AZ2"),
                   default=None,
                   help="steer the two fixed beams to these azimuths (degrees, "
                        "0 = array front) and enable energy gating")
    p.add_argument("--beam-fixed-no-gating", action="store_true",
                   help="with --beam-fixed, leave both beams live instead of "
                        "letting speech energy silence the inactive one")
    p.add_argument("--beam-apply-in-run", action="store_true",
                   help="keep the fixed-beam steering applied while subtitles "
                        "run (default is to steer, report and restore)")
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

    if args.beam_probe:
        if device is None:
            print("[beam] probe requires the USB control plane")
            return 1
        from .beam_router import FixedBeamController

        survey = FixedBeamController(device).probe()
        for name, val in survey.items():
            print(f"[beam] {name} = {val}")
        return 0

    beam_controller = None
    if args.beam_fixed is not None:
        if device is None:
            print("[beam] fixed-beam steering requires the USB control plane")
            return 1
        from .beam_router import FixedBeamController

        beam_controller = FixedBeamController(device)
        result = beam_controller.steer(
            args.beam_fixed[0], args.beam_fixed[1],
            gating=not args.beam_fixed_no_gating)
        print(f"[beam] steer {args.beam_fixed} gating="
              f"{not args.beam_fixed_no_gating} -> {result}")
        if not result.get("ok"):
            return 1
        if not args.beam_apply_in_run:
            # report, then hand the array back exactly as it was found: a
            # measurement run must not leave the beams pointing somewhere the
            # operator never chose
            print(f"[beam] restoring original beam config: "
                  f"{beam_controller.restore()}")
            beam_controller = None

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

        chunk = (FunasrProvider.CHUNK_SIZE_480 if args.asr_chunk_size == "480"
                 else FunasrProvider.CHUNK_SIZE_600)
        provider = FunasrProvider(chunk_size=chunk)
        if args.streaming:
            stride_ms = chunk[1] * 60
            print(f"[ASR] FunASR Paraformer (zh, streaming 2pass, "
                  f"stride {stride_ms}ms)")
        else:
            print("[ASR] FunASR Paraformer (zh, offline)")
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
    seg_durations: dict[str, float] = {}
    # first-character latency: segment open -> first non-empty partial
    seg_open_at: dict[str, float] = {}
    seg_seed_secs: dict[str, float] = {}
    first_char_logged: set[str] = set()
    # rolling stats for the [首字] line; guarded because on_partial runs on
    # the transcriber worker while flush runs on the main loop
    first_char_stats = {"n": 0, "sum": 0.0, "miss": 0}
    first_char_lock = threading.Lock()

    translation = TranslationThread(translator, server,
                                     speech_end_at=speech_end_at)
    translation.start()

    result_queue: queue.Queue = queue.Queue()

    def on_asr_result(result):
        result_queue.put(result)

    def on_partial(stream_id, text):
        # first-character latency: first non-empty partial after the segment
        # opened. This is the number the pre-roll seed is meant to shrink -
        # the [耗时] line measures "after speech ended", which is a different
        # (and unchanged) quantity.
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
        # trim trailing silence before the offline re-decode: hold-period
        # ticks feed quiet audio into the stream, and silence tails both
        # slow the decode and seed phantom punctuation
        audio = _trim_tail(req.audio) if req.audio is not None and req.audio.size else req.audio
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

    # ---- segmentation state machine ---------------------------------
    audio_queue: queue.Queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            pass  # overflow flags are tolerated; queue drops are the backpressure
        audio_queue.put(indata[:, ch_index].copy())

    speaking = False
    utterance: list[np.ndarray] = []
    # Ring buffer of the most recent audio, kept unconditionally (silence
    # included) so that the instant a segment opens the whole window can be
    # fed as the first streaming chunk. The Paraformer online model advances
    # one chunk_stride (600ms) at a time, so without this the opening words
    # wait for a full stride to accumulate - the dominant term in the ~0.7s
    # first-character latency. 1s of history fills the stride at once.
    pre_roll: deque[np.ndarray] = deque()
    pre_roll_samples = 0
    PRE_ROLL_SAMPLES_MAX = int(args.pre_roll * SAMPLE_RATE)
    seg_start = 0.0
    last_speech_at = 0.0
    silence_low_ticks = 0     # consecutive ticks with sw_prob < 0.2
    hw_speech_ticks = 0
    sw_speech_frames = 0
    total_frames = 0            # every sample seen while speaking
    silent_tail_frames = 0      # ...of which these came after the last voiced tick
    silent_tail_prob = 0.0      # probability mass of that silent tail
    speech_samples = 0          # voiced samples only (pre-roll seed excluded)
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
        nonlocal silence_low_ticks, speech_samples, pre_roll_samples
        nonlocal silent_tail_frames, silent_tail_prob
        if stream_opened and current_id:
            transcriber.stream_abort(current_id)
        if current_id:
            server.broadcast({"type": "discard", "id": current_id, "ts": time.time()})
            seg_open_at.pop(current_id, None)
            seg_seed_secs.pop(current_id, None)
            first_char_logged.discard(current_id)
        if args.debug_gate and current_id:
            # same denominator the flush filter uses: frames up to the last
            # voiced tick, so discarded and kept segments are comparable
            denom = total_frames - silent_tail_frames
            ratio = sw_speech_frames / denom if denom else 0.0
            avg = (prob_sum - silent_tail_prob) / denom if denom else 0.0
            print(f"[VAD] discard {current_id} ({reason}): dur={audio_dur():.2f}s "
                  f"ratio={ratio:.2f} avg_prob={avg:.2f} max_prob={max_prob:.2f} "
                  f"hw_ticks={hw_speech_ticks}/{denom // 512 or 1}")
        utterance = []
        hw_speech_ticks = 0
        sw_speech_frames = 0
        total_frames = 0
        silent_tail_frames = 0
        silent_tail_prob = 0.0
        speech_samples = 0
        prob_sum = 0.0
        max_prob = 0.0
        current_id = ""
        stream_opened = False
        speaking = False
        silence_low_ticks = 0
        # the discarded audio must not seed the next segment either
        pre_roll.clear()
        pre_roll_samples = 0
        sw_vad.reset()  # legal reset point: utterance truly ended

    def audio_dur() -> float:
        """Voiced length of the current segment (seed and trailing hold
        excluded) - the same quantity the min-length filter judges."""
        return speech_samples / SAMPLE_RATE

    def pre_roll_push(chunk: np.ndarray) -> None:
        """Append one audio block, evicting the oldest beyond the window."""
        nonlocal pre_roll_samples
        pre_roll.append(chunk)
        pre_roll_samples += chunk.size
        while pre_roll_samples > PRE_ROLL_SAMPLES_MAX and len(pre_roll) > 1:
            pre_roll_samples -= pre_roll.popleft().size

    def pre_roll_take() -> np.ndarray:
        """Drain the window into one contiguous block (oldest first)."""
        nonlocal pre_roll_samples
        if not pre_roll:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(pre_roll) if len(pre_roll) > 1 else pre_roll[0]
        pre_roll.clear()
        pre_roll_samples = 0
        return audio

    def flush_segment() -> bool:
        """End the current segment. Returns True if it was handed to ASR,
        False if a filter rejected it (in which case it has already been
        aborted and the frontend told to drop the card).
        """
        nonlocal speaking, utterance, hw_speech_ticks, sw_speech_frames
        nonlocal total_frames, prob_sum, max_prob, current_id, stream_opened
        nonlocal silence_low_ticks, speech_samples
        nonlocal silent_tail_frames, silent_tail_prob
        audio = np.concatenate(utterance) if utterance else np.zeros(0, np.float32)
        utterance = []
        # duration drives the min-length filter, so it must reflect voiced
        # audio only: the pre-roll seed is history rather than this utterance,
        # and the hold period we just waited out is silence, not speech
        duration = speech_samples / SAMPLE_RATE
        # ratio denominator stops at the last voiced tick: a sentence followed
        # by a long quiet tail must not be diluted into a discard
        voiced_span_frames = total_frames - silent_tail_frames
        voiced_prob = prob_sum - silent_tail_prob

        # current_id and stream_opened deliberately stay set while the filters
        # run: discard_segment() needs both to abort the streaming session and
        # to tell the frontend to drop the card, and it clears them itself.
        # Clearing them here first (as this used to) made the abort and the
        # discard broadcast silent no-ops, so a rejected segment left its
        # partial text on screen with no translation ever arriving.
        # level 1: duration
        if duration < args.min_seconds:
            discard_segment(f"duration {duration:.2f}s < min {args.min_seconds}s")
            return False
        # level 2: speech ratio (denominator excludes trailing silence)
        ratio = sw_speech_frames / voiced_span_frames if voiced_span_frames else 0.0
        avg_prob = voiced_prob / voiced_span_frames if voiced_span_frames else 0.0
        exempt = (duration < SHORT_EXEMPT_SECONDS
                  and max_prob >= SHORT_EXEMPT_MAX_PROB)
        if args.debug_gate:
            print(f"[flush] dur={duration:.2f}s ratio={ratio:.2f} avg_prob={avg_prob:.2f} "
                  f"max_prob={max_prob:.2f} exempt={exempt} hw_ticks={hw_speech_ticks}")
        if ratio < RATIO_THRESHOLD and not exempt:
            discard_segment(f"ratio {ratio:.2f} < {RATIO_THRESHOLD} (avg_prob={avg_prob:.2f})")
            return False
        # level 3: energy (trim trailing silence, then floor RMS)
        audio = _trim_tail(audio)
        if audio.size and float(np.sqrt(np.mean(audio ** 2))) < np.sqrt(TRIM_TAIL_ENERGY):
            discard_segment(f"energy {float(np.sqrt(np.mean(audio ** 2)))**2:.2e} < {TRIM_TAIL_ENERGY:.2e}")
            return False

        # every filter passed: take ownership of the segment and hand it on
        seg_id = current_id
        current_id = ""
        was_stream = stream_opened
        stream_opened = False

        if was_stream and seg_id:
            transcriber.stream_finish(seg_id, task_id=seg_id, doa=locator.mode())
        else:
            transcriber.submit(audio, task_id=seg_id, doa=locator.mode())
        # reset for next segment (without discarding this one)
        hw_speech_ticks = 0
        sw_speech_frames = 0
        total_frames = 0
        silent_tail_frames = 0
        silent_tail_prob = 0.0
        speech_samples = 0
        prob_sum = 0.0
        max_prob = 0.0
        speaking = False
        silence_low_ticks = 0
        sw_vad.reset()
        return True

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
                    # the ring buffer is fed unconditionally - silence and
                    # speech alike - so it is always at full depth when the
                    # gate opens (see pre_roll declaration). Keeping it warm
                    # while speaking also means the next segment opens with
                    # the previous tail already buffered.
                    pre_roll_push(c)
                    if speaking:
                        utterance.append(c)
                # streaming: one merged feed per tick (100ms) instead of
                # one task per callback block - the worker queue drains
                # 3-4x fewer tasks and partials surface earlier
                if speaking and stream_opened and chunks:
                    transcriber.stream_feed(
                        current_id,
                        np.concatenate(chunks) if len(chunks) > 1 else chunks[0],
                        current_id)

                # state transitions
                if not speaking:
                    if gate:
                        speaking = True
                        seg_start = now
                        last_speech_at = now
                        current_id = uuid.uuid4().hex[:12]
                        seg_open_at[current_id] = time.monotonic()
                        # start a fresh DOA tally: the reported direction must
                        # describe this utterance, and the counter is what
                        # flush_segment() reads back via locator.mode()
                        locator.reset()
                        # take the whole pre-roll window as the seed audio:
                        # the online model's own prev_samples buffer is then
                        # already past one 600ms stride, so the opening words
                        # decode on the very first generate() call
                        seed = pre_roll_take()
                        utterance = [seed] if seed.size else []
                        seg_seed_secs[current_id] = seed.size / SAMPLE_RATE
                        if args.streaming and provider.supports_streaming:
                            transcriber.stream_start(current_id)
                            transcriber.stream_feed(current_id, seed if seed.size else np.zeros(0, np.float32), current_id)
                            stream_opened = True
                        print(f"[seg] open {current_id} "
                              f"(seed {seed.size / SAMPLE_RATE:.2f}s)")
                else:
                    # accumulate statistics only while speech may be active
                    if chunks:
                        n = sum(c.size for c in chunks)
                        # Voiced time is counted only while the gate is open.
                        # This branch also runs through the whole tail wait
                        # (silence_hold), so counting every sample here would
                        # inflate duration by ~0.6s and leave the min-length
                        # filter and the short-speech exemption both judging
                        # the wrong quantity.
                        if gate:
                            speech_samples += n
                            sw_speech_frames += n
                            total_frames += n
                            silent_tail_frames = 0
                            silent_tail_prob = 0.0
                            last_speech_at = now
                        else:
                            # held aside rather than dropped: silence between
                            # words still belongs to the segment, silence
                            # after the last word does not
                            total_frames += n
                            silent_tail_frames += n
                    tick_prob = sw_prob * (sum(c.size for c in chunks))
                    prob_sum += tick_prob
                    if not gate:
                        silent_tail_prob += tick_prob
                    max_prob = max(max_prob, sw_prob)
                    hw_speech_ticks += 1 if hw_vad else 0

                    # end-of-utterance confirmation: count CONSECUTIVE ticks
                    # with clearly non-speech Silero probability. Word gaps
                    # inside a sentence hold probabilities around 0.3-0.5, so
                    # three straight ticks below 0.2 (~0.3s) is a strong
                    # silence signal. A single-tick cut (tried before)
                    # fragmented utterances mid-sentence and starved the
                    # 600ms streaming chunks; consecutive confirmation is
                    # immune to momentary dips because any speech-like tick
                    # resets the counter.
                    if gate or sw_prob >= 0.2:
                        silence_low_ticks = 0
                    else:
                        silence_low_ticks += 1

                    silence = now - last_speech_at
                    overlong = now - seg_start >= args.max_seconds
                    early_end = silence_low_ticks >= 3 and silence >= 0.3
                    if silence >= args.silence_hold or overlong or early_end:
                        seg_id_before_flush = current_id
                        print(f"[seg] flush after {now - seg_start:.2f}s "
                              f"(tail wait {silence:.2f}s)")
                        delivered = flush_segment()
                        # Timing anchors are only meaningful for segments that
                        # actually reach ASR. A rejected segment already aborted
                        # its stream and told the frontend to drop the card, so
                        # recording anchors here would leak entries that no
                        # result will ever pop.
                        if delivered and seg_id_before_flush:
                            speech_end_at[seg_id_before_flush] = time.monotonic()
                            seg_durations[seg_id_before_flush] = now - seg_start
                            seg_open_at.pop(seg_id_before_flush, None)
                            seg_seed_secs.pop(seg_id_before_flush, None)
                            # a flushed segment that never produced a partial
                            # is the short-utterance case: the 600ms stride
                            # means one-chunk sentences decode only at finish.
                            # Counted so the summary can report coverage
                            # instead of silently omitting them.
                            if seg_id_before_flush not in first_char_logged:
                                first_char_stats["miss"] += 1
                            first_char_logged.discard(seg_id_before_flush)
                        # Must clear: keeping the tail would seed the next
                        # stream with this sentence's speech, and the fresh
                        # stream would re-decode those words into its own
                        # partial (visible duplicate text). The hold period
                        # then refills the window with real silence, so a
                        # normal silence_hold flush still opens the next
                        # segment with a ~0.6s seed.
                        pre_roll.clear()
                        pre_roll_samples = 0

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
                    is_final = bool(result.meta.get("final_replaces_partial"))
                    # Timing anchors are single-use: pop them as soon as the
                    # segment's result shows up, whichever branch handles it,
                    # so nothing accumulates for discarded or empty segments.
                    speech_end = speech_end_at.pop(task_id, None) if is_final else None
                    seg_len = seg_durations.pop(task_id, None) if is_final else None
                    if not result.text:
                        # Nothing usable came out of this utterance. A card
                        # only exists if a partial was already broadcast for
                        # this id, and silently skipping would leave that card
                        # on screen with its cursor blinking and no translation
                        # ever arriving - so tell the frontend to drop it.
                        if is_final:
                            server.broadcast({"type": "discard", "id": task_id,
                                              "ts": time.time()})
                        continue
                    print(f"[asr] {task_id} final={is_final} "
                          f"mode={result.meta.get('mode', '')} "
                          f"decode={result.latency * 1000:.0f}ms "
                          f"text={result.text[:40]!r}")
                    # terminal-only latency stat (old-version format): the
                    # final text replacing the partial marks "source on
                    # screen" for this utterance
                    if is_final and speech_end is not None:
                        src_e2e = time.monotonic() - speech_end
                        seg_txt = f" (段时长 {seg_len:.2f}s)" if seg_len else ""
                        print(f"[耗时] {task_id[:8]} 说完→原文上屏: "
                              f"{src_e2e:.2f}s{seg_txt}")
                    # The ASR engine's own `language` field describes what it
                    # emits (FunASR always reports "zh"), not what it heard, so
                    # the card label and the translation direction are both
                    # derived from the recognized text instead.
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
        if vad_reader:
            vad_reader.stop()
        # put the beam configuration back if we were holding a steering change
        if beam_controller is not None:
            print(f"[beam] restoring original beam config: "
                  f"{beam_controller.restore()}")
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
