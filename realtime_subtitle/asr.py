"""ASR provider abstraction shared by FunASR and Whisper backends."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")


def detect_language(text: str) -> str:
    """Script-based language routing for the translation direction.

    Only scripts that a character class can decide on its own are named.
    Kana and hangul are checked before CJK ideographs because Japanese mixes
    both: a kana character is proof of Japanese, whereas kanji alone is not
    proof of Chinese. Latin, Cyrillic and the rest fall back to "en", which
    is the direction used for Latin-script input - telling French from German
    needs a real language-ID model, not a regex.

    Presence, not proportion, decides: a mixed sentence like "the API 响应时间
    太慢" stays Chinese because it contains ideographs.
    """
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
