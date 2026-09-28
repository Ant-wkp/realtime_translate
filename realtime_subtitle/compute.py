"""Compute-backend resolution shared by every inference provider.

The same source tree has to run on an x86 host (CPU only) and on a Jetson
(aarch64 + CUDA), so no module may hard-code a backend. Everything that
depends on the platform goes through here:

  * `resolve_torch_device()`  - ASR / VAD device for torch-backed models
  * `resolve_ct2_device()`    - CTranslate2 device, probing real support
  * `ct2_compute_type()`      - the quantisation that goes with that device
  * `describe_platform()`     - one-line banner so the active backend is
                                visible instead of inferred

Two facts drive the design:

1. Model *weights* are platform independent - a CT2 int8 model or a Paraformer
   state_dict carries no CPU/GPU flag, the engine decides where it runs. The
   split that matters is in the engines: torch must be a CUDA build
   (`2.x+cu130`, not the CPU-only wheel PyPI serves for aarch64), and
   ctranslate2 must have been compiled with CUDA. So devices are *probed*,
   never assumed.
2. FunASR's `AutoModel` defaults to `device="cuda"` and silently falls back to
   CPU when torch cannot see a GPU. A silent fallback is exactly what makes
   "is it actually using the GPU?" unanswerable, so the providers are handed
   an explicit device and print what they ended up with.
"""

from __future__ import annotations

import logging
import os
import platform

log = logging.getLogger(__name__)

# Device preferences accepted on the command line. "auto" means "use CUDA
# when it demonstrably works, otherwise CPU" - it never guesses.
DEVICE_CHOICES = ("auto", "cuda", "cpu")

# int8 weights are the right default on CPU. On GPU the quantisation is
# chosen for *reliability*, not raw speed, and the measurement is the reason:
#
#   compute_type   concurrent with ASR (i.e. how the program really runs)
#   int8_float16    6/24 succeeded, median 320 ms
#   float16        24/24 succeeded, median 406 ms
#
# The int8 GPU path calls cuBLAS int8 GEMM kernels that fail with
# `CUBLAS_STATUS_NOT_SUPPORTED` once another CUDA workload (FunASR on the same
# GPU) is active; the error is per-call and looks random until you notice it
# only happens under concurrency. float16 costs ~27% more per segment and
# never fails. Subtitles that are missing entirely are worse than subtitles
# that take 90 ms longer, so float16 wins. The same 6/24 vs 24/24 split was
# reproduced twice.
#
# `int8_float16` remains reachable via --nllb-compute-type for anyone who
# wants to compare on their own workload.
CT2_TYPE_CPU = "int8"
CT2_TYPE_CUDA = "float16"


def cuda_available() -> bool:
    """True when torch was built with CUDA and can see a device.

    Imported lazily: this module is also used by unit tests that must run
    without torch installed."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - a broken driver must not be fatal
        return False


def ctranslate2_cuda_available() -> bool:
    """True when the *installed ctranslate2 build* can execute on CUDA.

    Distinct from `cuda_available()`: torch and ctranslate2 are separate
    packages with separate builds, and the aarch64 wheel of the latter is
    CPU-only on PyPI. Probing here is what turns "GPU requested but
    unavailable" into a warning instead of a startup crash.

    The device count is checked first on purpose. `get_supported_compute_types
    ('cuda')` on a machine whose driver is too old raises, and - worse - it
    populates an internal CUDA context on the way, so a second call returns a
    non-empty set. Calling it in the wrong order makes the same machine report
    both "no CUDA" and "CUDA ready" within one process.
    """
    try:
        import ctranslate2
    except ImportError:
        return False
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return True
    except Exception:  # noqa: BLE001 - build without CUDA / driver mismatch
        return False
    # A CUDA-capable device is invisible, so the only remaining question is
    # whether the build supports it at all. Report False rather than probing
    # further: a build that has CUDA but sees no device cannot run on it.
    return False


def resolve_torch_device(pref: str = "auto") -> str:
    """Resolve the device string for torch-backed models (ASR).

    "auto" -> "cuda" only when torch can see a GPU, else "cpu". An explicit
    "cuda" is returned as-is so that a misconfiguration fails loudly in the
    provider rather than being silently downgraded (except for ctranslate2,
    see `resolve_ct2_device`).
    """
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda"
    return "cuda" if cuda_available() else "cpu"


def resolve_ct2_device(pref: str = "auto") -> str:
    """Resolve the device for CTranslate2, degrading instead of crashing.

    Unlike torch, ctranslate2 cannot be trusted to fail usefully: a build
    without CUDA raises `ValueError: This CTranslate2 package was not
    compiled with CUDA support` from deep inside the constructor. So both
    "auto" and an explicit "cuda" resolve to "cpu" when the installed build
    has no usable CUDA - the caller logs a warning and carries on with
    int8/CPU.
    """
    if pref == "cpu":
        return "cpu"
    if not ctranslate2_cuda_available():
        return "cpu"
    return "cuda"


def ct2_compute_type(device: str, pref: str | None = None) -> str:
    """Quantisation to request for `device`, unless the user pinned one."""
    if pref:
        return pref
    return CT2_TYPE_CUDA if device == "cuda" else CT2_TYPE_CPU


# Libraries a CUDA-enabled ctranslate2 dlopens by soname at runtime. Listed
# only so a failure can be explained; nothing here is preloaded, and that is a
# deliberate reversal of an earlier attempt - see `ct2_cuda_library_status`.
_CT2_CUDA_LIBS = ("libcublasLt.so.13", "libcublas.so.13", "libcudnn.so.9")


def ct2_cuda_library_status() -> tuple[bool, str]:
    """Report whether a dynamically-loading ct2 can resolve its CUDA libs.

    A CUDA_DYNAMIC_LOADING=ON ct2 build dlopens cuBLAS/cuDNN by soname on
    first use, so it needs those sonames resolvable through the normal linker
    search (ldconfig / rpath / LD_LIBRARY_PATH). This function only *checks*
    that, in a throwaway dlopen with RTLD_NOLOAD semantics implied by taking
    no global handle.

    It deliberately does not load anything itself. An earlier version
    preloaded the libraries with RTLD_GLOBAL to "help" ct2 find them, and that
    made things strictly worse: measured on an Orin (L4T 39.2 / CUDA 13.2)
    with a mix of short and long inputs,

        no preload                -> 20 ok / 0 failed
        preload torch's cuBLAS    -> 10 ok / 10 failed

    The failures were deterministic per input text (some sentences always
    failed, others never did) with `cuBLAS failed with status
    CUBLAS_STATUS_NOT_SUPPORTED` - i.e. a genuine kernel-selection problem,
    not the intermittent library-clash that the earlier investigation
    assumed from a measurement that happened to use only short sentences.
    The toolkit's sonames are already registered by ldconfig, so ct2 resolves
    them on its own; forcing a second copy into the process only creates a
    second candidate for the same soname.

    Returns (ok, detail) for the banner/log, never raises.
    """
    import ctypes

    missing = []
    for lib in _CT2_CUDA_LIBS:
        try:
            ctypes.CDLL(lib)
        except OSError:
            missing.append(lib)
    if missing:
        return False, "unresolved: " + ", ".join(missing)
    return True, "resolvable via linker search"


def describe_platform() -> str:
    """One-line summary of arch / torch / CUDA, for the startup banner.

    Deliberately verbose about the negatives: a torch without CUDA and a
    CPU-only ctranslate2 build are the two states that make the GPU question
    answerable from a log rather than by guesswork."""
    parts = [f"{platform.machine()}/{platform.system().lower()}"]
    try:
        import torch

        build = torch.version.cuda or "cpu-build"
        parts.append(f"torch {torch.__version__} ({build})")
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:  # noqa: BLE001
                name = "unknown"
            parts.append(f"GPU {name}")
        else:
            parts.append("GPU not visible to torch")
    except ImportError:
        parts.append("torch missing")
    try:
        import ctranslate2

        ct2 = f"ctranslate2 {ctranslate2.__version__}"
        ct2 += " CUDA" if ctranslate2_cuda_available() else " CPU-only"
        parts.append(ct2)
    except ImportError:
        parts.append("ctranslate2 missing")
    return " | ".join(parts)