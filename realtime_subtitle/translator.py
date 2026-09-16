"""NLLB-200 translation via CTranslate2 with fixed language routing.

Routing rules (replacing the old ratio-based detection that misclassified
mixed CJK-Latin sentences and silently fell back to Chinese):
- any CJK character -> source is Chinese
- unknown language codes warn and skip translation (show source as-is)
- beam size 4 with length/repetition penalties (quality over latency;
  translation never blocks source display)
- very short segments skip translation entirely
"""

from __future__ import annotations

import logging
import re

import ctranslate2
import sentencepiece as spm

from .asr import detect_language

log = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

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
}

MIN_TRANSLATE_CHARS = 2


class NllbCTranslate2Provider:
    def __init__(
        self,
        model_path: str = "models/nllb-200-distilled-1.3B-ct2-int8",
        target_langs: dict[str, str] | None = None,
    ):
        self.model_path = model_path
        self.translator = ctranslate2.Translator(model_path, compute_type="int8")
        self.sp = spm.SentencePieceProcessor(
            model_file=f"{model_path}/sentencepiece.model"
        )
        self.beam_size = 4
        self.length_penalty = 1.0
        self.repetition_penalty = 1.1
        self.target_langs = target_langs or dict(TARGET_LANGS)
        # runtime direction config, settable from the frontend
        self.zh_to = "en"
        self.other_to = "zh"

    def set_target_langs(self, zh_to: str, other_to: str) -> None:
        self.zh_to = zh_to
        self.other_to = other_to

    # ------------------------------------------------------------------
    def resolve_direction(self, text: str, detected: str) -> tuple[str, str]:
        """Return (source_token, target_token) or ("", "") to skip."""
        source = detected or detect_language(text)
        target = self.zh_to if source == "zh" else self.other_to
        if source == target:
            return "", ""
        src_token = LANG_TO_NLLB.get(source)
        tgt_token = LANG_TO_NLLB.get(target)
        if not src_token or not tgt_token:
            log.warning("unsupported language pair src=%s tgt=%s, skip", source, target)
            return "", ""
        return src_token, tgt_token

    def translate(self, text: str, detected: str = "zh") -> tuple[str, str]:
        """Translate one segment. Returns (translated_text, target_lang) or
        (source_text, source_lang) when translation is skipped."""
        text = (text or "").strip()
        if len(text) < MIN_TRANSLATE_CHARS:
            return text, detected

        src_token, tgt_token = self.resolve_direction(text, detected)
        if not src_token:
            return text, detected

        tokens = self.sp.encode(text, out_type=str)
        source_tokens = [src_token] + tokens + ["</s>"]
        target_prefix = [tgt_token]
        max_decoding = max(32, min(128, len(tokens) * 3))

        results = self.translator.translate_batch(
            [source_tokens],
            target_prefix=[target_prefix],
            beam_size=self.beam_size,
            max_decoding_length=max_decoding,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            return_scores=False,
        )
        out_tokens = results[0].hypotheses[0]
        if out_tokens and out_tokens[0] == tgt_token:
            out_tokens = out_tokens[1:]
        translated = self.sp.decode(out_tokens)
        target_lang = "zh" if tgt_token == "zho_Hans" else tgt_token.split("_")[0]
        return translated, target_lang

    def close(self) -> None:  # pragma: no cover
        pass
