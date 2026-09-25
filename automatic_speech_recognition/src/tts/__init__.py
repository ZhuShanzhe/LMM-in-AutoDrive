from typing import Any

__all__ = ["Qwen3TTSService"]

def __getattr__(name: str) -> Any:
    if name == "Qwen3TTSService":
        from .qwen3_tts_service import Qwen3TTSService
        globals()[name] = Qwen3TTSService
        return Qwen3TTSService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")