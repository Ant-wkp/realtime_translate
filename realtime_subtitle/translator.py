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
import json
import os
import re

import ctranslate2
import sentencepiece as spm

from .asr import detect_language
from .compute import (ct2_compute_type, ct2_cuda_library_status,
                      resolve_ct2_device)

log = logging.getLogger(__name__)

# NLLB-200 knows 200 languages, not 13: the full set is the
# `additional_special_tokens` array in the model's special_tokens_map.json, and
# every code below is a member of it. This table is the *commonly used* subset
# for two reasons -
#
#   1. it is what the settings dropdown offers, and a 200-entry list is not a
#      usable control;
#   2. only these four have a source side. detect_language() classifies by
#      script and can only return ja / ko / zh / en, so the source tokens for
#      fr, de, es and the rest are currently unreachable - a sentence in those
#      languages is read as "en". They are listed anyway because the target
#      side is reachable and because swapping in a real language-ID model (or a
#      multilingual ASR) only requires the source side to start producing these
#      codes.
#
# Adding a language is a one-line change here: it flows to the dropdown via
# control.cmd_get_target_langs, which publishes this dict verbatim.
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

# Runtime settings that survive a restart. The translation direction is a
# user preference, not a per-run experiment - losing it on every restart made
# the setting look broken ("I picked Japanese and it still translates to
# English") because the hard-coded default is zh_to="en". Unlike the device
# parameters, nothing here is written to the XVF3800's flash: this is a plain
# JSON file next to the package.
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "runtime_settings.json")

# Default beam width. NLLB-600M decoded one segment at a time is dominated by
# encoder/decoder passes, and the beam multiplies the decoder pass: measured
# on this repo's samples, 4 -> 1 is 246ms -> 198ms on x86 and 1026ms -> 828ms
# on an AGX Orin (CPU int8). Subtitles favour latency over the last increment
# of fluency, so the default is greedy; the quality difference was only
# spot-checked on three sentences, not evaluated as a corpus, so raise it via
# --nllb-beam if a translation ever reads badly rather than trusting this
# number to mean "as good as beam 4".
DEFAULT_BEAM_SIZE = 1


def load_settings() -> dict:
    """Read the persisted runtime settings; empty dict when absent/broken.

    A malformed or unreadable file must never stop the program: the settings
    are a convenience, and the defaults below are valid."""
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict) -> None:
    """Persist runtime settings, ignoring failures.

    Best-effort on purpose: a read-only install directory should not turn
    "remember my language" into a crash."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        log.warning("cannot persist settings to %s: %s", SETTINGS_FILE, exc)


def _find_model_dir() -> str:
    """Locate an NLLB CT2 model: env override, local models/ dir, HF cache.

    Search paths are resolved relative to the package directory so the
    CWD does not matter (python -m ... vs python main.py).

    NLLB_MODEL_DIR short-circuits the whole search. It exists because the
    deployment copies the model somewhere convenient, and the
    ~/.cache/huggingface/hub/models--*/snapshots/<rev> layout is an
    implementation detail of huggingface_hub rather than something worth
    reproducing on the target host.

    The model files themselves are platform independent: a CT2 int8
    checkpoint carries no CPU/GPU flag, only the engine that loads it does.
    """
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
        device: str = "auto",
        compute_type: str | None = None,
        beam_size: int = DEFAULT_BEAM_SIZE,
    ):
        self.model_path = model_path or _find_model_dir()
        # Both "auto" and "cuda" resolve to CPU when the installed
        # ctranslate2 has no usable CUDA. That is not defensive coding for
        # its own sake: PyPI ships a CPU-only wheel for linux_aarch64, and the
        # failure there is `This CTranslate2 package was not compiled with
        # CUDA support` raised from inside the constructor - losing a little
        # speed beats failing to start.
        self.device = resolve_ct2_device(device)
        if device == "cuda" and self.device != "cuda":
            log.warning(
                "CUDA requested but this ctranslate2 build has no usable "
                "CUDA backend; translating on CPU with int8"
            )
        if self.device == "cuda":
            # Report - do not "fix". A CUDA_DYNAMIC_LOADING build resolves
            # cuBLAS/cuDNN by soname through the normal linker search, so the
            # only useful thing to do here is say so when that will fail, and
            # let the failure be a clear startup warning instead of a
            # confusing RuntimeError on the first subtitle. Preloading was
            # tried and measured to be harmful; see ct2_cuda_library_status.
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
        # Safety net for the GPU path: a second translator in plain float32,
        # used only when the first one raises mid-session (see
        # _translate_batch). Built lazily - its extra CUDA memory is not worth
        # paying for unless a failure actually happens. Not created when the
        # main translator already runs float32, since the retry would be
        # identical.
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
        # runtime direction config: restored from disk so a restart does not
        # silently revert to "Chinese -> English". Only codes still present in
        # the table are accepted, so a stale file cannot inject a pair the
        # model cannot express.
        saved = load_settings()
        self.zh_to = saved.get("zh_to") if saved.get("zh_to") in self.target_langs else "en"
        self.other_to = (saved.get("other_to")
                         if saved.get("other_to") in self.target_langs else "zh")

    def set_target_langs(self, zh_to: str, other_to: str) -> None:
        self.zh_to = zh_to
        self.other_to = other_to
        save_settings({"zh_to": zh_to, "other_to": other_to})

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

    def _translate_batch(self, source_tokens: list, target_prefix: list,
                         max_decoding: int):
        """Run one translation, retrying once without quantisation on GPU.

        The retry exists because a GPU cuBLAS failure must not cost the user a
        subtitle. The default compute type is already the reliable one (see
        CT2_TYPE_CUDA), but a user-supplied `--nllb-compute-type int8_float16`
        can reintroduce the failing path, and a transient driver-level error is
        always possible; in either case the fallback recomputes in float32,
        which is slower but exercises no int8 kernel.

        CPU failures are not retried - there is nothing to degrade to, and a
        silent retry would only hide a real problem.
        """
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

        results = self._translate_batch(source_tokens, target_prefix,
                                        max_decoding)
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
