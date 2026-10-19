
from __future__ import annotations

import logging
import json
import os

import ctranslate2
import sentencepiece as spm

from .asr import detect_language
from .compute import (ct2_compute_type, ct2_cuda_library_status,
                      resolve_ct2_device)

log = logging.getLogger(__name__)

LANG_TO_NLLB = {
    "zh": "zho_Hans",
    "en": "eng_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "ar": "arb_Arab",
    "th": "tha_Thai",
    "vi": "vie_Latn",
}

TARGET_LANGS = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
}

MIN_TRANSLATE_CHARS = 2

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "runtime_settings.json")

DEFAULT_BEAM_SIZE = 1

def load_settings() -> dict:
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}

def save_settings(settings: dict) -> None:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        log.warning("cannot persist settings to %s: %s", SETTINGS_FILE, exc)

def _find_model_dir() -> str:
    override = os.environ.get("NLLB_MODEL_DIR")
    if override:
        return override
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "models", "nllb-200-distilled-1.3B-ct2-int8"),
        "models/nllb-200-distilled-1.3B-ct2-int8",
        os.path.join(here, "models", "nllb-200-distilled-600M-ct2-int8"),
        "models/nllb-200-distilled-600M-ct2-int8",
    ]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "model.bin")):
            return path
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.isdir(hub):
        for repo in ("models--JustFrederik--nllb-200-distilled-1.3B-ct2-int8",
                     "models--JustFrederik--nllb-200-distilled-600M-ct2-int8"):
            snaps = os.path.join(hub, repo, "snapshots")
            if os.path.isdir(snaps):
                for rev in sorted(os.listdir(snaps)):
                    d = os.path.join(snaps, rev)
                    if os.path.isfile(os.path.join(d, "model.bin")):
                        return d
    return candidates[1]

def _find_spm_file(model_path: str) -> str:
    for name in ("sentencepiece.model", "sentencepiece.bpe.model"):
        p = os.path.join(model_path, name)
        if os.path.isfile(p):
            return p
    return os.path.join(model_path, "sentencepiece.model")

class NllbCTranslate2Provider:
    def __init__(
        self,
        model_path: str | None = None,
        target_langs: dict[str, str] | None = None,
        device: str = "auto",
        compute_type: str | None = None,
        beam_size: int = DEFAULT_BEAM_SIZE,
    ):
        self.model_path = model_path or _find_model_dir()
        self.device = resolve_ct2_device(device)
        if device == "cuda" and self.device != "cuda":
            log.warning(
                "CUDA requested but this ctranslate2 build has no usable "
                "CUDA backend; translating on CPU with int8"
            )
        if self.device == "cuda":
            ok, detail = ct2_cuda_library_status()
            if not ok:
                log.warning("ct2 CUDA libraries %s; translation may fail at "
                            "the first call", detail)
        self.compute_type = ct2_compute_type(self.device, compute_type)
        self.translator = ctranslate2.Translator(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._fallback = None
        self._fallback_ready = (self.device == "cuda"
                                and self.compute_type != "float32")
        self._retry_count = 0
        self.sp = spm.SentencePieceProcessor(
            model_file=_find_spm_file(self.model_path)
        )
        self.beam_size = beam_size
        self.length_penalty = 1.0
        self.repetition_penalty = 1.1
        self.target_langs = target_langs or dict(TARGET_LANGS)
        saved = load_settings()
        self.zh_to = saved.get("zh_to") if saved.get("zh_to") in self.target_langs else "en"
        self.other_to = (saved.get("other_to")
                         if saved.get("other_to") in self.target_langs else "zh")

    def set_target_langs(self, zh_to: str, other_to: str) -> None:
        self.zh_to = zh_to
        self.other_to = other_to
        save_settings({"zh_to": zh_to, "other_to": other_to})

    def resolve_direction(self, text: str, detected: str = "") -> tuple[str, str]:
        source = detect_language(text) or detected or "zh"
        target = self.zh_to if source == "zh" else self.other_to
        if source == target:
            return "", ""
        src_token = LANG_TO_NLLB.get(source)
        tgt_token = LANG_TO_NLLB.get(target)
        if not src_token or not tgt_token:
            log.warning("unsupported language pair src=%s tgt=%s, skip", source, target)
            return "", ""
        return src_token, tgt_token

    def _translate_batch(self, source_tokens: list, target_prefix: list,
                         max_decoding: int):
        try:
            return self.translator.translate_batch(
                [source_tokens],
                target_prefix=[target_prefix],
                beam_size=self.beam_size,
                max_decoding_length=max_decoding,
                length_penalty=self.length_penalty,
                repetition_penalty=self.repetition_penalty,
                return_scores=False,
            )
        except RuntimeError as exc:
            if self.device != "cuda" or not self._fallback_ready:
                raise
            if self._fallback is None:
                self._fallback = ctranslate2.Translator(
                    self.model_path, device="cuda", compute_type="float32",
                )
            self._retry_count += 1
            if self._retry_count <= 3:
                log.warning("GPU translate failed (%s); retrying in float32 "
                            "[%d]", exc, self._retry_count)
            return self._fallback.translate_batch(
                [source_tokens],
                target_prefix=[target_prefix],
                beam_size=self.beam_size,
                max_decoding_length=max_decoding,
                length_penalty=self.length_penalty,
                repetition_penalty=self.repetition_penalty,
                return_scores=False,
            )

    def translate(self, text: str, detected: str = "") -> tuple[str, str]:
        text = (text or "").strip()
        if len(text) < MIN_TRANSLATE_CHARS:
            return text, detected or detect_language(text)

        src_token, tgt_token = self.resolve_direction(text, detected)
        if not src_token:
            return text, detected

        tokens = self.sp.encode(text, out_type=str)
        source_tokens = [src_token] + tokens + ["</s>"]
        target_prefix = [tgt_token]
        max_decoding = max(32, min(128, len(tokens) * 3))

        results = self._translate_batch(source_tokens, target_prefix,
                                        max_decoding)
        out_tokens = results[0].hypotheses[0]
        if out_tokens and out_tokens[0] == tgt_token:
            out_tokens = out_tokens[1:]
        translated = self.sp.decode(out_tokens)
        target_lang = next((code for code, tok in LANG_TO_NLLB.items()
                            if tok == tgt_token), tgt_token.split("_")[0])
        return translated, target_lang

    def close(self) -> None:  # pragma: no cover
        pass
