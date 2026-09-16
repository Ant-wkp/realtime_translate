"""Whisper ASR backend via faster-whisper (int8, multi-language fallback)."""

from __future__ import annotations

import time

import numpy as np

from .asr import AsrProvider, AsrResult, detect_language


class WhisperProvider(AsrProvider):
    name = "whisper"
    supports_streaming = False

    def __init__(self, model_size: str = "tiny", device: str = "cpu"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_size, device=device, compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> AsrResult:
        start = time.monotonic()
        segments, info = self.model.transcribe(audio, beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        lang = info.language if info.language else detect_language(text)
        return AsrResult(
            text=text,
            language=lang,
            latency=time.monotonic() - start,
            meta={"engine": self.name, "probability": info.language_probability},
        )

    def close(self) -> None:  # pragma: no cover
        pass
