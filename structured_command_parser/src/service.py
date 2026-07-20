from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from .pipeline import ChineseEnglishCommandPipeline


@dataclass(frozen=True)
class CommandParserConfig:
    translator_model_path: str
    parser_model_path: str
    default_modality: str = "VOICE"
    max_input_chars: int = 512

    @classmethod
    def shared_model(
        cls,
        model_path: str,
        *,
        default_modality: str = "VOICE",
        max_input_chars: int = 512,
    ) -> "CommandParserConfig":
        return cls(
            translator_model_path=model_path,
            parser_model_path=model_path,
            default_modality=default_modality,
            max_input_chars=max_input_chars,
        )


class DrivingCommandService:
    """Long-lived, thread-safe integration boundary for ASR command messages."""

    def __init__(
        self,
        config: CommandParserConfig,
        *,
        pipeline: ChineseEnglishCommandPipeline | None = None,
    ) -> None:
        if config.default_modality not in {"VOICE", "TEXT"}:
            raise ValueError("default_modality must be VOICE or TEXT")
        if config.max_input_chars <= 0:
            raise ValueError("max_input_chars must be positive")
        self.config = config
        self.pipeline = pipeline or ChineseEnglishCommandPipeline(
            config.translator_model_path,
            config.parser_model_path,
        )
        self._inference_lock = Lock()

    @classmethod
    def from_shared_model(
        cls,
        model_path: str,
        *,
        default_modality: str = "VOICE",
        max_input_chars: int = 512,
    ) -> "DrivingCommandService":
        return cls(
            CommandParserConfig.shared_model(
                model_path,
                default_modality=default_modality,
                max_input_chars=max_input_chars,
            )
        )

    def warmup(self) -> None:
        """Load model weights before the first ASR request."""
        self.pipeline.translator.runtime.load()

    def parse_asr_text(
        self,
        text: str,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return self._parse(
            text,
            modality=self.config.default_modality,
            request_id=request_id,
        )

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Consume the minimal JSON-like message emitted by an ASR adapter."""
        if not isinstance(message, dict):
            raise TypeError("message must be a dictionary")
        text = message.get("text")
        if not isinstance(text, str):
            raise ValueError("message.text must be a string")
        request_id = message.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise ValueError("message.request_id must be a string when provided")
        modality = message.get("modality", self.config.default_modality)
        if modality not in {"VOICE", "TEXT"}:
            raise ValueError("message.modality must be VOICE or TEXT")
        return self._parse(text, modality=modality, request_id=request_id)

    def _parse(
        self,
        text: str,
        *,
        modality: str,
        request_id: str | None,
    ) -> dict[str, Any]:
        normalized_input = text.strip()
        if not normalized_input:
            raise ValueError("ASR text cannot be empty")
        if len(normalized_input) > self.config.max_input_chars:
            raise ValueError(
                f"ASR text exceeds max_input_chars={self.config.max_input_chars}"
            )
        with self._inference_lock:
            return self.pipeline.parse(
                normalized_input,
                modality=modality,
                request_id=request_id,
            )
