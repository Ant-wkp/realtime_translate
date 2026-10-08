
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")

def detect_language(text: str) -> str:
    if not text:
        return "en"
    if _KANA_RE.search(text):
        return "ja"
    if _HANGUL_RE.search(text):
        return "ko"
    if _CJK_RE.search(text):
        return "zh"
    return "en"

@dataclass
class AsrResult:
    text: str = ""
    language: str = "zh"
    stream_id: str | None = None
    latency: float = 0.0
    meta: dict = field(default_factory=dict)

class AsrProvider:

    name = "abstract"

    def transcribe(self, audio: np.ndarray) -> AsrResult:
        raise NotImplementedError

    def start_utt(self, stream_id: str) -> None:  # pragma: no cover
        pass

    def feed(self, stream_id: str, audio: np.ndarray) -> str:
        raise NotImplementedError

    def finish(self, stream_id: str) -> AsrResult:
        raise NotImplementedError

    def abort(self, stream_id: str) -> None:  # pragma: no cover
        pass

    def close(self) -> None:  # pragma: no cover
        pass
