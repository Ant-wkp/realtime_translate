"""Segmenter: audio-to-segment state machine, as pure logic.

This is the segmentation half of what used to be inline in main(): the
per-tick gating decision, the pre-roll ring buffer, the trailing-silence hold
and the three flush-time filters. It is deliberately free of I/O - the
caller feeds it audio and a gate verdict, and gets back a list of Actions to
perform. That keeps the transcription queue, the broadcast server and the
device control plane out of here, which is what makes the logic testable
without hardware.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .sw_vad import SoftwareVad

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SegmentCfg:
    """Segment-level filters, mirroring the CLI defaults in main()."""

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
    """What the caller should do with the audio this tick produced."""

    kind: str                 # open | feed | flush | discard
    seg_id: str = ""
    audio: np.ndarray | None = None
    seed_secs: float = 0.0
    seg_seconds: float = 0.0
    tail_wait: float = 0.0
    delivered: bool = False
    reason: str = ""
    #: for "flush"/"discard": whether a streaming session was open, so the
    #: caller knows whether to finish or abort it
    stream_opened: bool = False
    #: pre-formatted diagnostics to print, so the segmenter keeps the wording
    #: and the relative order of every line exactly as it was inline
    flush_line: str = ""
    debug_line: str = ""


@dataclass
class Counters:
    """Rolling statistics, printed on exit."""

    flushed: int = 0
    discarded: int = 0
    discard_reasons: list[str] = field(default_factory=list)

    def note_discard(self, reason: str) -> None:
        self.discarded += 1
        head = reason.split(" ", 1)[0]
        self.discard_reasons.append(head)


def trim_tail(audio: np.ndarray, floor_energy: float) -> np.ndarray:
    """Drop trailing near-silent frames.

    Used both before the offline re-decode and as the third flush filter, so
    it lives beside the segmenter rather than inside main().
    """
    n = audio.size
    step = 512
    floor = np.sqrt(floor_energy)
    while n > step:
        if float(np.sqrt(np.mean(audio[n - step:n] ** 2))) > floor:
            break
        n -= step
    return audio[:n]


class Segmenter:
    """The segmentation state machine.

    Call `feed()` once per main-loop tick with that tick's audio and the gate
    verdict, then act on the returned Actions. The sequence inside `feed()`
    mirrors the original inline loop exactly, so behaviour is unchanged from
    before the extraction.
    """

    def __init__(self, cfg: SegmentCfg, vad: SoftwareVad):
        self.cfg = cfg
        self.vad = vad
        self.counters = Counters()
        # most recent VAD readings, for the caller's gate debug and heartbeat
        self.last_prob = 0.0
        self.last_gate = False

        # audio buffering
        self.utterance: list[np.ndarray] = []
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_samples = 0
        self._pre_roll_max = int(cfg.pre_roll * SAMPLE_RATE)

        # segment state
        self.speaking = False
        self.current_id = ""
        self.stream_opened = False
        self.seg_start = 0.0
        self.last_speech_at = 0.0

        # statistics feeding the flush-time filters
        self.hw_speech_ticks = 0
        self.sw_speech_frames = 0
        self.total_frames = 0
        self.silent_tail_frames = 0
        self.silent_tail_prob = 0.0
        self.speech_samples = 0
        self.prob_sum = 0.0
        self.max_prob = 0.0
        self.silence_low_ticks = 0

    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.speaking

    def voiced_seconds(self) -> float:
        """Voiced length of the current segment (seed and hold excluded) -
        the same quantity the min-length filter judges."""
        return self.speech_samples / SAMPLE_RATE

    # -- ring buffer ----------------------------------------------------
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
        """Discard the buffered history.

        Called on flush and discard alike: keeping the tail would seed the
        next stream with this sentence's speech and re-decode those words as
        visibly duplicated text.
        """
        self._pre_roll.clear()
        self._pre_roll_samples = 0

    # ------------------------------------------------------------------
    def feed(self, chunks: list[np.ndarray], gate_fn, hw_vad: int,
             now: float) -> list[Action]:
        """Consume one tick. Returns the Actions the caller must perform.

        `chunks` is this tick's audio for this lane and `gate_fn` is the
        caller's gate verdict callback, `gate_fn(sw_prob) -> bool`. The
        software VAD probability is computed here, once, so that it drives the
        gate verdict, the statistics and the end-of-utterance confirmation from
        a single consistent reading. `hw_vad` only feeds the diagnostic
        hardware-tick counter.
        """
        actions: list[Action] = []
        n_samples = sum(c.size for c in chunks) if chunks else 0

        # software VAD first: the gate verdict for this tick depends on it
        if chunks:
            sw_prob = self.vad.feed(np.concatenate(chunks) if len(chunks) > 1
                                    else chunks[0])
        else:
            sw_prob = self.vad.last_probability
        self.last_prob = sw_prob
        gate = gate_fn(sw_prob)
        self.last_gate = gate

        # 1) ring buffer is fed unconditionally - silence and speech alike -
        # so it is always at full depth the instant the gate opens
        for c in chunks:
            self._pre_roll_push(c)
            if self.speaking:
                self.utterance.append(c)

        # 2) one merged feed per tick rather than per callback block: the
        # worker queue drains 3-4x fewer tasks and partials surface earlier
        if self.speaking and self.stream_opened and chunks:
            merged = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            actions.append(Action(kind="feed", seg_id=self.current_id,
                                  audio=merged))

        if not self.speaking:
            if gate:
                actions.append(self._open(now))
            return actions

        # 3) statistics. Voiced time is counted only while the gate is open:
        # this branch also runs through the whole tail wait, so counting every
        # sample here would inflate duration by roughly silence_hold and leave
        # the min-length filter judging the wrong quantity.
        if chunks:
            if gate:
                self.speech_samples += n_samples
                self.sw_speech_frames += n_samples
                self.total_frames += n_samples
                self.silent_tail_frames = 0
                self.silent_tail_prob = 0.0
                self.last_speech_at = now
            else:
                # held aside rather than dropped: silence between words still
                # belongs to the segment, silence after the last word does not
                self.total_frames += n_samples
                self.silent_tail_frames += n_samples
        tick_prob = sw_prob * n_samples
        self.prob_sum += tick_prob
        if not gate:
            self.silent_tail_prob += tick_prob
        self.max_prob = max(self.max_prob, sw_prob)
        self.hw_speech_ticks += 1 if hw_vad else 0

        # 4) end-of-utterance confirmation counts CONSECUTIVE quiet ticks.
        # Word gaps inside a sentence sit around 0.3-0.5, so three straight
        # ticks below 0.2 (~0.3s) is a strong silence signal; any speech-like
        # tick resets the counter, which makes it immune to momentary dips.
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

    # ------------------------------------------------------------------
    def _open(self, now: float) -> Action:
        self.speaking = True
        self.seg_start = now
        self.last_speech_at = now
        self.current_id = uuid.uuid4().hex[:12]
        # take the whole pre-roll window as the seed: the online model's own
        # prev_samples buffer is then already past one decode stride, so the
        # opening words decode on the very first generate() call
        seed = self._pre_roll_take()
        self.utterance = [seed] if seed.size else []
        self.stream_opened = False
        return Action(kind="open", seg_id=self.current_id, audio=seed,
                      seed_secs=seed.size / SAMPLE_RATE)

    def _flush(self, silence: float, now: float) -> Action:
        """Apply the three filters, then either discard or hand the segment on.

        `current_id` and `stream_opened` deliberately stay set while the
        filters run: the discard path needs both to abort the streaming
        session and to tell the frontend to drop the card, and it clears them
        itself. Clearing them first made both silent no-ops.
        """
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
        # printed unconditionally, and printed before the [flush] stats line:
        # both ordering and wording match the original inline loop
        flush_line = (f"[seg] flush after {now - self.seg_start:.2f}s "
                      f"(tail wait {silence:.2f}s)")

        # level 1: duration
        if duration < self.cfg.min_seconds:
            return self.discard(
                f"duration {duration:.2f}s < min {self.cfg.min_seconds}s",
                debug_line=debug, flush_line=flush_line)
        # level 2: speech ratio (denominator stops at the last voiced tick)
        ratio = self.sw_speech_frames / voiced_span if voiced_span else 0.0
        avg_prob = voiced_prob / voiced_span if voiced_span else 0.0
        exempt = (duration < self.cfg.short_exempt_seconds
                  and self.max_prob >= self.cfg.short_exempt_max_prob)
        if ratio < self.cfg.ratio_threshold and not exempt:
            return self.discard(
                f"ratio {ratio:.2f} < {self.cfg.ratio_threshold} "
                f"(avg_prob={avg_prob:.2f})", debug_line=debug,
                flush_line=flush_line)
        # level 3: energy (trim trailing silence, then floor RMS)
        audio = trim_tail(audio, self.cfg.trim_tail_energy)
        if audio.size:
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < np.sqrt(self.cfg.trim_tail_energy):
                return self.discard(
                    f"energy {rms ** 2:.2e} < {self.cfg.trim_tail_energy:.2e}",
                    debug_line=debug, flush_line=flush_line)

        # every filter passed: hand it on and reset for the next segment
        action = Action(kind="flush", seg_id=self.current_id, audio=audio,
                        delivered=True, flush_line=flush_line, debug_line=debug,
                        stream_opened=self.stream_opened,
                        seg_seconds=now - self.seg_start, tail_wait=silence)
        self.counters.flushed += 1
        self._reset(keep_speaking=False)
        return action

    def discard(self, reason: str, debug_line: str = "",
                flush_line: str = "") -> Action:
        """The single discard path: one Action, caller aborts and broadcasts.

        Whether a streaming session was open rides along in the Action,
        because the caller needs it to decide whether an abort is owed to the
        worker - the segmenter clears its own flag during the reset.
        """
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
        # the audio just consumed must not seed the next segment either
        self._drop_pre_roll()
        # a legal reset point: the utterance truly ended
        self.vad.reset()

    # ------------------------------------------------------------------
    def mark_stream_open(self) -> None:
        """Record that the caller opened a streaming session for this segment."""
        self.stream_opened = True

    def note_delivered(self, seg_id: str) -> None:
        """Compatibility hook for callers tracking handoff themselves."""

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