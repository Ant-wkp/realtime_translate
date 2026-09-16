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
import os
import re

import ctranslate2
import sentencepiece as spm

from .asr import detect_language

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


def _find_model_dir() -> str:
    """Locate an NLLB CT2 model: local models/ dir first, then HF cache.

    Search paths are resolved relative to the package directory so the
    CWD does not matter (python -m ... vs python main.py)."""
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
    return candidates[1]  # not found; ctranslate2 will raise a clear error


def _find_spm_file(model_path: str) -> str:
    """CT2 NLLB repos ship either sentencepiece.model or .bpe.model."""
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
    ):
        self.model_path = model_path or _find_model_dir()
        self.translator = ctranslate2.Translator(self.model_path, compute_type="int8")
        self.sp = spm.SentencePieceProcessor(
            model_file=_find_spm_file(self.model_path)
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
    def resolve_direction(self, text: str, detected: str = "") -> tuple[str, str]:
        """Return (source_token, target_token) or ("", "") to skip.

        The source language comes from the recognized *text*, not from the ASR
        engine. A Chinese-only engine (FunASR) reports language="zh" for every
        utterance because that is the language it emits, not a judgement about
        what it heard - trusting it routed English speech through the
        Chinese-source branch, which is why "other language -> Chinese" had no
        effect on English input.

        Anything that is not Chinese is routed to `other_to`, so an
        unidentifiable or unexpected source still translates instead of being
        skipped.
        """
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

    def translate(self, text: str, detected: str = "") -> tuple[str, str]:
        """Translate one segment. Returns (translated_text, target_lang) or
        (source_text, source_lang) when translation is skipped."""
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
        # report the same short code the frontend configures with ("en", not
        # "eng"), so the card label and the settings dropdown agree
        target_lang = next((code for code, tok in LANG_TO_NLLB.items()
                            if tok == tgt_token), tgt_token.split("_")[0])
        return translated, target_lang

    def close(self) -> None:  # pragma: no cover
        pass
