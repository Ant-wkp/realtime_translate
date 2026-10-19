
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

from .asr import AsrProvider, AsrResult

@dataclass
class Task:
    audio: object
    task_id: str
    doa: str = ""
    stream_id: str | None = None
    kind: str = "transcribe"
    meta: dict = field(default_factory=dict)

@dataclass
class FinalRequest:
    task_id: str
    stream_id: str
    audio: object
    partial_hint: str = ""
    doa: str = ""

class TranscriberWorker(threading.Thread):
    def __init__(self, provider: AsrProvider, on_result=None, on_partial=None,
                 on_final_request=None, on_error=None):
        super().__init__(daemon=True, name="TranscriberWorker")
        self.provider = provider
        self.on_result = on_result
        self.on_partial = on_partial
        self.on_final_request = on_final_request
        self.on_error = on_error
        self._task_queue: queue.Queue = queue.Queue()
        self._stop_evt = threading.Event()

    def submit(self, audio, task_id: str, doa: str = "", stream_id: str | None = None,
               meta: dict | None = None) -> None:
        if self._stop_evt.is_set():
            return
        self._task_queue.put(
            Task(audio=audio, task_id=task_id, doa=doa, stream_id=stream_id,
                 kind="transcribe", meta=meta or {})
        )

    def stream_start(self, stream_id: str) -> None:
        if self._stop_evt.is_set():
            return
        self._task_queue.put(
            Task(audio=None, task_id=stream_id, kind="start", stream_id=stream_id)
        )

    def stream_feed(self, stream_id: str, audio, task_id: str) -> None:
        if self._stop_evt.is_set():
            return
        self._task_queue.put(
            Task(audio=audio, task_id=task_id, kind="feed", stream_id=stream_id)
        )

    def stream_finish(self, stream_id: str, task_id: str, doa: str = "") -> None:
        if self._stop_evt.is_set():
            return
        self._task_queue.put(
            Task(audio=None, task_id=task_id, doa=doa, kind="finish",
                 stream_id=stream_id)
        )

    def stream_abort(self, stream_id: str) -> None:
        if self._stop_evt.is_set():
            return
        self._task_queue.put(
            Task(audio=None, task_id="", kind="abort", stream_id=stream_id)
        )

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                task = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._stop_evt.is_set():
                break
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
                if partial and self.on_partial and not self._stop_evt.is_set():
                    self.on_partial(task.stream_id, partial)
                return
            if task.kind == "finish":
                if hasattr(self.provider, "finish_online"):
                    partial, audio = self.provider.finish_online(task.stream_id)
                    if partial and self.on_partial and not self._stop_evt.is_set():
                        self.on_partial(task.stream_id, partial)
                    if self.on_final_request is not None and not self._stop_evt.is_set():
                        self.on_final_request(FinalRequest(
                            task_id=task.task_id,
                            stream_id=task.stream_id,
                            audio=audio,
                            partial_hint=partial,
                            doa=task.doa,
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
            if task.stream_id and task.kind != "abort":
                try:
                    self.provider.abort(task.stream_id)
                except Exception as abort_exc:
                    print(f"[ASR] abort {task.stream_id} failed: {abort_exc}")
            if self.on_error and task.kind != "abort" and not self._stop_evt.is_set():
                self.on_error(task.task_id, str(exc))

    def _emit(self, result: AsrResult, task: Task) -> None:
        result.stream_id = task.stream_id
        result.meta["task_id"] = task.task_id
        result.meta["doa"] = task.doa
        result.meta.update(task.meta)
        if self.on_result and not self._stop_evt.is_set():
            self.on_result(result)
