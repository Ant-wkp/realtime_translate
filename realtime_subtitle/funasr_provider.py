from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .asr import AsrProvider, AsrResult
from .compute import resolve_torch_device

_HUB_ALIASES = {
    "paraformer-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "paraformer-zh-streaming": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "ct-punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
}

_REQUIRED_MODEL_FILES = ("configuration.json", "config.yaml")

def _local_model_path(alias: str) -> str | None:
    model_id = _HUB_ALIASES.get(alias)
    if not model_id:
        return None
    roots = []
    env = os.environ.get("MODELSCOPE_CACHE")
    if env:
        roots.append(os.path.join(env, "models"))
    roots.append(os.path.expanduser("~/.cache/modelscope/models"))
    dir_name = model_id.replace("/", "--").replace(".", "-")
    for root in roots:
        pattern = os.path.join(root, dir_name, "snapshots", "*")
        for snap in sorted(glob.glob(pattern), reverse=True):
            if all(os.path.isfile(os.path.join(snap, f))
                   for f in _REQUIRED_MODEL_FILES):
                return snap
    return None

@dataclass
class _Utterance:
    stream_id: str
    chunks: list = field(default_factory=list)
    cache: dict = field(default_factory=dict)
    partial_parts: list = field(default_factory=list)
    broadcast_len: int = 0
    aborted: bool = False

class FunasrProvider(AsrProvider):
    name = "funasr"

    CHUNK_SIZE_600 = [5, 10, 5]
    CHUNK_SIZE_480 = [5, 8, 4]

    ENCODER_LOOK_BACK = 4
    DECODER_LOOK_BACK = 1

    def __init__(self, sample_rate: int = 16000, chunk_size: list | None = None,
                 device: str = "auto"):
        self._ensure_audio_backend()

        from funasr import AutoModel

        self.chunk_size = list(chunk_size) if chunk_size else list(self.CHUNK_SIZE_480)
        self.device = resolve_torch_device(device)

        def resolve(name: str) -> str:
            local = _local_model_path(name)
            if local:
                return local
            print(f"[ASR] {name}: no complete local copy, letting FunASR "
                  f"resolve it (needs network)")
            return name

        self._offline = AutoModel(model=resolve("paraformer-zh"),
                                  disable_update=True, device=self.device)
        self._punc = AutoModel(model=resolve("ct-punc"),
                               disable_update=True, device=self.device)
        try:
            self._stream = AutoModel(model=resolve("paraformer-zh-streaming"),
                                     disable_update=True,
                                     device=self.device)
        except Exception:  # noqa: BLE001 - streaming optional
            self._stream = None
        self._utts: dict[str, _Utterance] = {}

    @staticmethod
    def _ensure_audio_backend() -> None:
        try:
            import torchaudio
        except Exception:
            try:
                import kaldi_native_fbank
            except ImportError:
                import subprocess
                import sys as _sys

                print("[ASR] torchaudio unavailable; installing "
                      "kaldi-native-fbank fallback frontend")
                subprocess.run(
                    [_sys.executable, "-m", "pip", "install", "-q",
                     "kaldi-native-fbank"],
                    check=False,
                )

    def describe_device(self) -> str:
        for mdl in (self._offline, self._stream):
            inner = getattr(mdl, "model", None)
            found = getattr(inner, "device", None)
            if found is not None:
                return str(found)
        return self.device

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

    def start_utt(self, stream_id: str) -> None:
        self._utts[stream_id] = _Utterance(stream_id=stream_id)

    def feed(self, stream_id: str, audio: np.ndarray) -> str:
        utt = self._utts.get(stream_id)
        if utt is None or utt.aborted:
            return ""
        audio = np.asarray(audio, dtype=np.float32)
        utt.chunks.append(audio)
        if self._stream is None:
            return ""

        res = self._stream.generate(
            input=audio,
            cache=utt.cache,
            is_final=False,
            chunk_size=self.chunk_size,
            encoder_chunk_look_back=self.ENCODER_LOOK_BACK,
            decoder_chunk_look_back=self.DECODER_LOOK_BACK,
            disable_pbar=True,
        )
        delta = res[0].get("text", "") if res else ""
        if delta:
            utt.partial_parts.append(delta)

        text = "".join(utt.partial_parts)
        if len(text) == utt.broadcast_len:
            return ""
        utt.broadcast_len = len(text)
        return text

    def finish_online(self, stream_id: str) -> tuple[str, np.ndarray]:
        utt = self._utts.pop(stream_id, None)
        if utt is None or utt.aborted:
            return "", np.zeros(0, dtype=np.float32)
        if self._stream is not None:
            try:
                res = self._stream.generate(
                    input=np.zeros(0, dtype=np.float32),
                    cache=utt.cache,
                    is_final=True,
                    chunk_size=self.chunk_size,
                    encoder_chunk_look_back=self.ENCODER_LOOK_BACK,
                    decoder_chunk_look_back=self.DECODER_LOOK_BACK,
                    disable_pbar=True,
                )
                delta = res[0].get("text", "") if res else ""
                if delta:
                    utt.partial_parts.append(delta)
            except Exception:
                pass
        audio = (
            np.concatenate(utt.chunks) if utt.chunks
            else np.zeros(0, dtype=np.float32)
        )
        return "".join(utt.partial_parts), audio

    def finish(self, stream_id: str) -> AsrResult:
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

    def close(self) -> None:
        self._utts.clear()
