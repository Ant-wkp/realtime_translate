"""ASR provider abstraction shared by FunASR and Whisper backends."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def detect_language(text: str) -> str:
    """High-recall language routing: any CJK char means Chinese source.

    Mixed sentences like 'the API response time is too slow' would be
    misclassified by a ratio threshold, so presence (not proportion) decides.
    """
    if not text:
        return "en"
    return "zh" if _CJK_RE.search(text) else "en"


@dataclass
class AsrResult:
    text: str = ""
    language: str = "zh"
    stream_id: str | None = None
    is_partial: bool = False
    latency: float = 0.0
    meta: dict = field(default_factory=dict)


class AsrProvider:
    """Unified ASR interface: offline transcribe plus streaming utterances."""

    name = "abstract"
    supports_streaming = False

    def transcribe(self, audio: np.ndarray) -> AsrResult:
        raise NotImplementedError

    # streaming API ---------------------------------------------------
    def start_utt(self, stream_id: str) -> None:  # pragma: no cover
        pass

    def feed(self, stream_id: str, audio: np.ndarray) -> str:
        """Feed one chunk, return the cumulative partial text."""
        raise NotImplementedError

    def finish(self, stream_id: str) -> AsrResult:
        raise NotImplementedError

    def abort(self, stream_id: str) -> None:  # pragma: no cover
        pass

    def close(self) -> None:  # pragma: no cover
        pass
