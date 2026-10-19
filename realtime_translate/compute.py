from __future__ import annotations

import logging
import platform

log = logging.getLogger(__name__)

DEVICE_CHOICES = ("auto", "cuda", "cpu")

CT2_TYPE_CPU = "int8"
CT2_TYPE_CUDA = "float16"

def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False

def ctranslate2_cuda_available() -> bool:
    try:
        import ctranslate2
    except ImportError:
        return False
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return True
    except Exception:
        return False
    return False

def resolve_torch_device(pref: str = "auto") -> str:
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda"
    return "cuda" if cuda_available() else "cpu"

def resolve_ct2_device(pref: str = "auto") -> str:
    if pref == "cpu":
        return "cpu"
    if not ctranslate2_cuda_available():
        return "cpu"
    return "cuda"

def ct2_compute_type(device: str, pref: str | None = None) -> str:
    if pref:
        return pref
    return CT2_TYPE_CUDA if device == "cuda" else CT2_TYPE_CPU

_CT2_CUDA_LIBS = ("libcublasLt.so.13", "libcublas.so.13", "libcudnn.so.9")

def ct2_cuda_library_status() -> tuple[bool, str]:
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
