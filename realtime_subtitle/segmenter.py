from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .sw_vad import SoftwareVad

SAMPLE_RATE = 16000

@dataclass(frozen=True)
class SegmentCfg:

    min_seconds: float = 0.3
    max_seconds: float = 20.0
    silence_hold: float = 0.6
    pre_roll: float = 1.0
    short_exempt_seconds: float = 1.0
    short_exempt_max_prob: float = 0.5
    ratio_threshold: float = 0.35
    trim_tail_energy: float = 1e-4
    debug: bool = False

@dataclass
class Action:

    kind: str
    seg_id: str = ""
    audio: np.ndarray | None = None
    seed_secs: float = 0.0
    seg_seconds: float = 0.0
    tail_wait: float = 0.0
    delivered: bool = False
    reason: str = ""
    stream_opened: bool = False
    flush_line: str = ""
    debug_line: str = ""

@dataclass
class Counters:

    flushed: int = 0
    discarded: int = 0
    discard_reasons: list[str] = field(default_factory=list)

    def note_discard(self, reason: str) -> None:
        self.discarded += 1
        head = reason.split(" ", 1)[0]
        self.discard_reasons.append(head)

def trim_tail(audio: np.ndarray, floor_energy: float) -> np.ndarray:
    n = audio.size
    step = 512
    floor = np.sqrt(floor_energy)
    while n > step:
        if float(np.sqrt(np.mean(audio[n - step:n] ** 2))) > floor:
            break
        n -= step
    return audio[:n]

class Segmenter:

    def __init__(self, cfg: SegmentCfg, vad: SoftwareVad):
        self.cfg = cfg
        self.vad = vad
        self.counters = Counters()
        self.last_prob = 0.0
        self.last_gate = False

        self.utterance: list[np.ndarray] = []
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_samples = 0
        self._pre_roll_max = int(cfg.pre_roll * SAMPLE_RATE)

        self.speaking = False
        self.current_id = ""
        self.stream_opened = False
        self.seg_start = 0.0
        self.last_speech_at = 0.0

        self.hw_speech_ticks = 0
        self.sw_speech_frames = 0
        self.total_frames = 0
        self.silent_tail_frames = 0
        self.silent_tail_prob = 0.0
        self.speech_samples = 0
        self.prob_sum = 0.0
        self.max_prob = 0.0
        self.silence_low_ticks = 0

    @property
    def active(self) -> bool:
        return self.speaking

    def voiced_seconds(self) -> float:
        return self.speech_samples / SAMPLE_RATE

    def _pre_roll_push(self, chunk: np.ndarray) -> None:
        self._pre_roll.append(chunk)
        self._pre_roll_samples += chunk.size
        while (self._pre_roll_samples > self._pre_roll_max
               and len(self._pre_roll) > 1):
            self._pre_roll_samples -= self._pre_roll.popleft().size

    def _pre_roll_take(self) -> np.ndarray:
        if not self._pre_roll:
            return np.zeros(0, dtype=np.float32)
        audio = (np.concatenate(self._pre_roll) if len(self._pre_roll) > 1
                 else self._pre_roll[0])
        self._pre_roll.clear()
        self._pre_roll_samples = 0
        return audio

    def _drop_pre_roll(self) -> None:
        self._pre_roll.clear()
        self._pre_roll_samples = 0

    def feed(self, chunks: list[np.ndarray], gate_fn, hw_vad: int,
             now: float) -> list[Action]:

        actions: list[Action] = []
        n_samples = sum(c.size for c in chunks) if chunks else 0

        if chunks:
            sw_prob = self.vad.feed(np.concatenate(chunks) if len(chunks) > 1
                                    else chunks[0])
        else:
            sw_prob = self.vad.last_probability
        self.last_prob = sw_prob
        gate = gate_fn(sw_prob)
        self.last_gate = gate

        for c in chunks:
            self._pre_roll_push(c)
            if self.speaking:
                self.utterance.append(c)

        if self.speaking and self.stream_opened and chunks:
            merged = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            actions.append(Action(kind="feed", seg_id=self.current_id,
                                  audio=merged))

        if not self.speaking:
            if gate:
                actions.append(self._open(now))
            return actions

        if chunks:
            if gate:
                self.speech_samples += n_samples
                self.sw_speech_frames += n_samples
                self.total_frames += n_samples
                self.silent_tail_frames = 0
                self.silent_tail_prob = 0.0
                self.last_speech_at = now
            else:
                self.total_frames += n_samples
                self.silent_tail_frames += n_samples
        tick_prob = sw_prob * n_samples
        self.prob_sum += tick_prob
        if not gate:
            self.silent_tail_prob += tick_prob
        self.max_prob = max(self.max_prob, sw_prob)
        self.hw_speech_ticks += 1 if hw_vad else 0

        if gate or sw_prob >= 0.2:
            self.silence_low_ticks = 0
        else:
            self.silence_low_ticks += 1

        silence = now - self.last_speech_at
        overlong = now - self.seg_start >= self.cfg.max_seconds
        early_end = self.silence_low_ticks >= 3 and silence >= 0.3
        if silence >= self.cfg.silence_hold or overlong or early_end:
            actions.append(self._flush(silence, now))
        return actions

    def _open(self, now: float) -> Action:
        self.speaking = True
        self.seg_start = now
        self.last_speech_at = now
        self.current_id = uuid.uuid4().hex[:12]
        seed = self._pre_roll_take()
        self.utterance = [seed] if seed.size else []
        self.stream_opened = False
        return Action(kind="open", seg_id=self.current_id, audio=seed,
                      seed_secs=seed.size / SAMPLE_RATE)

    def _flush(self, silence: float, now: float) -> Action:
        audio = (np.concatenate(self.utterance) if self.utterance
                 else np.zeros(0, np.float32))
        self.utterance = []
        duration = self.voiced_seconds()
        voiced_span = self.total_frames - self.silent_tail_frames
        voiced_prob = self.prob_sum - self.silent_tail_prob

        debug = ""
        if self.cfg.debug:
            ratio = self.sw_speech_frames / voiced_span if voiced_span else 0.0
            avg = voiced_prob / voiced_span if voiced_span else 0.0
            exempt = (duration < self.cfg.short_exempt_seconds
                      and self.max_prob >= self.cfg.short_exempt_max_prob)
            debug = (f"[flush] dur={duration:.2f}s "
                     f"ratio={ratio:.2f} avg_prob={avg:.2f} "
                     f"max_prob={self.max_prob:.2f} exempt={exempt} "
                     f"hw_ticks={self.hw_speech_ticks}")
        flush_line = (f"[seg] flush after {now - self.seg_start:.2f}s "
                      f"(tail wait {silence:.2f}s)")

        if duration < self.cfg.min_seconds:
            return self.discard(
                f"duration {duration:.2f}s < min {self.cfg.min_seconds}s",
                debug_line=debug, flush_line=flush_line)
        ratio = self.sw_speech_frames / voiced_span if voiced_span else 0.0
        avg_prob = voiced_prob / voiced_span if voiced_span else 0.0
        exempt = (duration < self.cfg.short_exempt_seconds
                  and self.max_prob >= self.cfg.short_exempt_max_prob)
        if ratio < self.cfg.ratio_threshold and not exempt:
            return self.discard(
                f"ratio {ratio:.2f} < {self.cfg.ratio_threshold} "
                f"(avg_prob={avg_prob:.2f})", debug_line=debug,
                flush_line=flush_line)
        audio = trim_tail(audio, self.cfg.trim_tail_energy)
        if audio.size:
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < np.sqrt(self.cfg.trim_tail_energy):
                return self.discard(
                    f"energy {rms ** 2:.2e} < {self.cfg.trim_tail_energy:.2e}",
                    debug_line=debug, flush_line=flush_line)

        action = Action(kind="flush", seg_id=self.current_id, audio=audio,
                        delivered=True, flush_line=flush_line, debug_line=debug,
                        stream_opened=self.stream_opened,
                        seg_seconds=now - self.seg_start, tail_wait=silence)
        self.counters.flushed += 1
        self._reset(keep_speaking=False)
        return action

    def discard(self, reason: str, debug_line: str = "",
                flush_line: str = "") -> Action:
        action = Action(kind="discard", seg_id=self.current_id,
                        delivered=False, reason=reason,
                        flush_line=flush_line)
        action.stream_opened = self.stream_opened
        self.counters.note_discard(reason)
        if self.cfg.debug:
            denom = self.total_frames - self.silent_tail_frames
            ratio = self.sw_speech_frames / denom if denom else 0.0
            avg = ((self.prob_sum - self.silent_tail_prob) / denom
                   if denom else 0.0)
            extra = (f"[VAD] discard {action.seg_id} ({reason}): "
                     f"dur={self.voiced_seconds():.2f}s ratio={ratio:.2f} "
                     f"avg_prob={avg:.2f} max_prob={self.max_prob:.2f} "
                     f"hw_ticks={self.hw_speech_ticks}/{denom // 512 or 1}")
            action.debug_line = f"{debug_line}\n{extra}" if debug_line else extra
        self._reset(keep_speaking=False)
        return action

    def _reset(self, keep_speaking: bool) -> None:
        self.hw_speech_ticks = 0
        self.sw_speech_frames = 0
        self.total_frames = 0
        self.silent_tail_frames = 0
        self.silent_tail_prob = 0.0
        self.speech_samples = 0
        self.prob_sum = 0.0
        self.max_prob = 0.0
        self.current_id = ""
        self.stream_opened = False
        self.speaking = keep_speaking
        self.silence_low_ticks = 0
        self.utterance = []
        self._drop_pre_roll()
        self.vad.reset()

    def mark_stream_open(self) -> None:
        self.stream_opened = True

    def summary(self) -> str:
        if not self.counters.discarded and not self.counters.flushed:
            return ""
        reasons: dict[str, int] = {}
        for r in self.counters.discard_reasons:
            reasons[r] = reasons.get(r, 0) + 1
        tail = ", ".join(f"{k}×{v}" for k, v in sorted(reasons.items()))
        return (f"交付 {self.counters.flushed}, "
                f"丢弃 {self.counters.discarded}"
                + (f" [{tail}]" if tail else ""))
