"""FunASR Paraformer backend: offline whole-segment and streaming 2-pass.

Key contract with FunASR 1.4.x streaming generate(): each call returns the
*incremental* delta text, not the cumulative transcript, so the provider
keeps per-stream accumulated audio plus concatenated partial text. finish()
re-decodes the whole utterance with the offline model for the final result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .asr import AsrProvider, AsrResult


@dataclass
class _Utterance:
    stream_id: str
    chunks: list = field(default_factory=list)          # full audio (for final re-decode)
    cache: dict = field(default_factory=dict)           # streaming state, persists across chunks
    partial_parts: list = field(default_factory=list)   # accumulated partial text
    aborted: bool = False


class FunasrProvider(AsrProvider):
    name = "funasr"
    supports_streaming = True

    # FunASR chunk_size units are 60ms frames: [5, 10, 5] =
    # 300ms left context, 600ms center, 300ms right context
    CHUNK_SIZE = [5, 10, 5]

    def __init__(self, sample_rate: int = 16000):
        from funasr import AutoModel

        self._offline = AutoModel(model="paraformer-zh", disable_update=True)
        self._punc = AutoModel(model="ct-punc", disable_update=True)
        try:
            self._stream = AutoModel(model="paraformer-zh-streaming", disable_update=True)
        except Exception:  # noqa: BLE001 - streaming optional
            self._stream = None
        self._utts: dict[str, _Utterance] = {}

    # ------------------------------------------------------------------
    # offline
    # ------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray) -> AsrResult:
        start = time.monotonic()
        res = self._offline.generate(input=audio, batch_size_s=300, disable_pbar=True)
        text = res[0].get("text", "") if res else ""
        text = self._add_punc(text)
        return AsrResult(
            text=text.strip(),
            language="zh",
            latency=time.monotonic() - start,
            meta={"engine": self.name, "mode": "offline"},
        )

    def _add_punc(self, text: str) -> str:
        if not text:
            return text
        res = self._punc.generate(input=text, disable_pbar=True)
        return res[0].get("text", text) if res else text

    # ------------------------------------------------------------------
    # streaming 2-pass
    # ------------------------------------------------------------------
    def start_utt(self, stream_id: str) -> None:
        self._utts[stream_id] = _Utterance(stream_id=stream_id)

    def feed(self, stream_id: str, audio: np.ndarray) -> str:
        utt = self._utts.get(stream_id)
        if utt is None or utt.aborted:
            return ""
        audio = np.asarray(audio, dtype=np.float32)
        utt.chunks.append(audio)
        if self._stream is None:
            return "".join(utt.partial_parts)

        # Pass every tick's audio straight through: the model keeps its own
        # leftover buffer (cache["prev_samples"]) and advances one 600ms
        # chunk whenever enough audio has accumulated. Small frequent feeds
        # keep the 300ms right-context window supplied so tokens keep
        # flowing; batching at 600ms starves that context and stalls all
        # partials until is_final (verified against funasr 1.4.12
        # paraformer_streaming/model.py inference()).
        res = self._stream.generate(
            input=audio,
            cache=utt.cache,            # state persists across chunks
            is_final=False,
            chunk_size=self.CHUNK_SIZE,
            disable_pbar=True,
        )
        delta = res[0].get("text", "") if res else ""
        if delta:
            utt.partial_parts.append(delta)
        return "".join(utt.partial_parts)

    def finish_online(self, stream_id: str) -> tuple[str, np.ndarray]:
        """Close the online pass only: flush the decoder tail with
        is_final=True, return (full partial text, whole utterance audio).

        Split out from the offline re-decode so the streaming worker is
        never blocked by a multi-second second-pass decode: the caller can
        broadcast the tail partial immediately and re-decode on another
        thread via transcribe().
        """
        utt = self._utts.pop(stream_id, None)
        if utt is None or utt.aborted:
            return "", np.zeros(0, dtype=np.float32)
        if self._stream is not None:
            # is_final flushes the tail chunk: the model decodes whatever
            # sits in cache["prev_samples"] plus the right-context tokens it
            # had been holding back, releasing the last words immediately.
            try:
                res = self._stream.generate(
                    input=np.zeros(0, dtype=np.float32),
                    cache=utt.cache,
                    is_final=True,
                    chunk_size=self.CHUNK_SIZE,
                    disable_pbar=True,
                )
                delta = res[0].get("text", "") if res else ""
                if delta:
                    utt.partial_parts.append(delta)
            except Exception:  # noqa: BLE001 - tail pass is best-effort
                pass
        audio = (
            np.concatenate(utt.chunks) if utt.chunks
            else np.zeros(0, dtype=np.float32)
        )
        return "".join(utt.partial_parts), audio

    def finish(self, stream_id: str) -> AsrResult:
        """Single-thread finish: online tail + offline re-decode inline.

        Kept for compatibility (tests, non-streaming callers); the live
        pipeline uses finish_online + transcribe on a separate thread.
        """
        partial, audio = self.finish_online(stream_id)
        start = time.monotonic()
        res = self._offline.generate(input=audio, batch_size_s=300, disable_pbar=True)
        text = res[0].get("text", "") if res else ""
        text = self._add_punc(text)
        return AsrResult(
            text=text.strip(),
            language="zh",
            stream_id=stream_id,
            latency=time.monotonic() - start,
            meta={
                "engine": self.name,
                "mode": "2pass-final",
                "partial_hint": partial,
            },
        )

    def abort(self, stream_id: str) -> None:
        utt = self._utts.get(stream_id)
        if utt is not None:
            utt.aborted = True
        self._utts.pop(stream_id, None)

    def close(self) -> None:  # pragma: no cover
        self._utts.clear()
