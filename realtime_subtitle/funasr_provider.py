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
    chunks: list = field(default_factory=list)
    partial_parts: list = field(default_factory=list)
    aborted: bool = False


class FunasrProvider(AsrProvider):
    name = "funasr"
    supports_streaming = True

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
        res = self._offline.generate(input=audio, batch_size_s=300)
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
        res = self._punc.generate(input=text)
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
        utt.chunks.append(np.asarray(audio, dtype=np.float32))
        if self._stream is None:
            return "".join(utt.partial_parts)

        chunk_ms = max(int(audio.size / 16), 600)  # >=600ms per streaming chunk
        res = self._stream.generate(
            input=audio,
            cache={},
            is_final=False,
            chunk_size=[0, chunk_ms, 600],
        )
        delta = res[0].get("text", "") if res else ""
        if delta:
            utt.partial_parts.append(delta)
        return "".join(utt.partial_parts)

    def finish(self, stream_id: str) -> AsrResult:
        utt = self._utts.pop(stream_id, None)
        if utt is None:
            return AsrResult(text="", language="zh", stream_id=stream_id)
        start = time.monotonic()
        audio = (
            np.concatenate(utt.chunks) if utt.chunks else np.zeros(0, dtype=np.float32)
        )
        res = self._offline.generate(input=audio, batch_size_s=300)
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
                "partial_hint": "".join(utt.partial_parts),
            },
        )

    def abort(self, stream_id: str) -> None:
        utt = self._utts.get(stream_id)
        if utt is not None:
            utt.aborted = True
        self._utts.pop(stream_id, None)

    def close(self) -> None:  # pragma: no cover
        self._utts.clear()
