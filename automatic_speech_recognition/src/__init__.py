"""Top level package for the speech pipeline.

Submodules are resolved lazily so that lightweight helpers such as
``src.utils`` can be imported without loading torch, qwen_tts and transformers.
"""
from typing import Any

__all__ = ["Qwen3TTSService", "Qwen3TranslatorService", "Qwen3ASRService", "ASRPipeline"]

_LAZY = {
    "Qwen3TTSService": (__name__ + ".tts", "Qwen3TTSService"),
    "Qwen3TranslatorService": (__name__ + ".translator", "Qwen3TranslatorService"),
    "Qwen3ASRService": (__name__ + ".asr", "Qwen3ASRService"),
    "ASRPipeline": (__name__ + ".pipeline", "ASRPipeline"),
}

def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib
        module_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'src' has no attribute {name!r}")
