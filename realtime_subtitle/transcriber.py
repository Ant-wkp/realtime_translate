"""Transcription worker: serial ASR queue decoupled from the main loop.

submit() from the 100 ms main loop is always a cheap queue put. A daemon
worker consumes segments one at a time (naturally order-preserving), a
failed segment never blocks the rest of the queue, and results carry the
same task id the flush state machine stamped so downstream translation and
the frontend can merge by id.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

from .asr import AsrProvider, AsrResult


@dataclass
class Task:
    audio: object
    task_id: str
    doa: str = ""
    stream_id: str | None = None
    kind: str = "transcribe"  # transcribe | finish | abort
    lane: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class FinalRequest:
    """Closed utterance awaiting the offline second-pass re-decode.

    Produced by the streaming worker (online tail is already broadcast);
    consumed by FinalDecodeThread so the multi-second offline decode never
    blocks the next utterance's streaming feed queue.
    """
    task_id: str
    stream_id: str
    audio: object
    partial_hint: str = ""
    doa: str = ""
    lane: int = 0


class TranscriberWorker(threading.Thread):
    def __init__(self, provider: AsrProvider, on_result=None, on_partial=None,
                 on_final_request=None):
        super().__init__(daemon=True, name="TranscriberWorker")
        self.provider = provider
        self.on_result = on_result
        self.on_partial = on_partial
        self.on_final_request = on_final_request
        self._task_queue: queue.Queue = queue.Queue()
        self._stop_evt = threading.Event()

    # -- called from the main loop (cheap) -----------------------------
    def submit(self, audio, task_id: str, doa: str = "", stream_id: str | None = None,
               lane: int = 0, meta: dict | None = None) -> None:
        self._task_queue.put(
            Task(audio=audio, task_id=task_id, doa=doa, stream_id=stream_id,
                 kind="transcribe", lane=lane, meta=meta or {})
        )

    def stream_start(self, stream_id: str, lane: int = 0) -> None:
        """Open a streaming utterance session on the worker thread."""
        self._task_queue.put(
            Task(audio=None, task_id=stream_id, kind="start",
                 stream_id=stream_id, lane=lane)
        )

    def stream_feed(self, stream_id: str, audio, task_id: str,
                    lane: int = 0) -> None:
        self._task_queue.put(
            Task(audio=audio, task_id=task_id, kind="feed",
                 stream_id=stream_id, lane=lane)
        )

    def stream_finish(self, stream_id: str, task_id: str, doa: str = "",
                      lane: int = 0) -> None:
        self._task_queue.put(
            Task(audio=None, task_id=task_id, doa=doa, kind="finish",
                 stream_id=stream_id, lane=lane)
        )

    def stream_abort(self, stream_id: str) -> None:
        self._task_queue.put(
            Task(audio=None, task_id="", kind="abort", stream_id=stream_id)
        )

    def stop(self) -> None:
        self._stop_evt.set()

    # -- worker thread ---------------------------------------------------
    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                task = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._handle(task)

    def _handle(self, task: Task) -> None:
        try:
            if task.kind == "start":
                if hasattr(self.provider, "start_utt"):
                    self.provider.start_utt(task.stream_id)
                return
            if task.kind == "abort":
                self.provider.abort(task.stream_id)
                return
            if task.kind == "feed":
                partial = self.provider.feed(task.stream_id, task.audio)
                if partial and self.on_partial:
                    self.on_partial(task.stream_id, partial, task.lane)
                return
            if task.kind == "finish":
                # online tail only: broadcast the last partial immediately,
                # then hand the audio to the offline re-decode thread so the
                # next utterance's feeds are never blocked by a 1s+ decode
                if hasattr(self.provider, "finish_online"):
                    partial, audio = self.provider.finish_online(task.stream_id)
                    if partial and self.on_partial:
                        self.on_partial(task.stream_id, partial, task.lane)
                    if self.on_final_request is not None:
                        self.on_final_request(FinalRequest(
                            task_id=task.task_id,
                            stream_id=task.stream_id,
                            audio=audio,
                            partial_hint=partial,
                            doa=task.doa,
                            lane=task.lane,
                        ))
                    return
                result = self.provider.finish(task.stream_id)
                result.meta["final_replaces_partial"] = True
                self._emit(result, task)
                return
            result = self.provider.transcribe(task.audio)
            self._emit(result, task)
        except Exception as exc:  # noqa: BLE001 - one bad segment never blocks
            print(f"[ASR] segment {task.task_id} failed: {exc}")

    def _emit(self, result: AsrResult, task: Task) -> None:
        result.stream_id = task.stream_id
        result.meta["task_id"] = task.task_id
        result.meta["doa"] = task.doa
        result.meta["lane"] = task.lane
        result.meta.update(task.meta)
        if self.on_result:
            self.on_result(result)
