from __future__ import annotations

from threading import Lock
from typing import Any

from .modernbert_parser import ModernBertEnglishIntentParser
from .factory import make_document
from .normalizer import normalize_text
from .rule_parser import RuleIntentParser


class ModernBertCommandService:
    """Thread-safe English command parsing boundary for the translation module."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        device: str = "cuda",
        max_input_chars: int = 512,
        parser: Any | None = None,
    ) -> None:
        if max_input_chars <= 0:
            raise ValueError("max_input_chars must be positive")
        if parser is None and not model_path:
            raise ValueError("model_path is required when parser is not provided")
        self.max_input_chars = max_input_chars
        self.parser = parser or ModernBertEnglishIntentParser(
            str(model_path),
            device=device,
        )
        self._inference_lock = Lock()

    def warmup(self) -> None:
        """Load model weights before the service accepts requests."""
        with self._inference_lock:
            self.parser.warmup()

    def parse_text(
        self,
        text: str,
        *,
        request_id: str | None = None,
        modality: str = "TEXT",
        source_text: str | None = None,
        source_language: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = " ".join(text.strip().split())
        if not normalized:
            raise ValueError("English command cannot be empty")
        if len(normalized) > self.max_input_chars:
            raise ValueError(
                f"English command exceeds max_input_chars={self.max_input_chars}"
            )
        if modality not in {"VOICE", "TEXT"}:
            raise ValueError("modality must be VOICE or TEXT")
        if source_text is not None and (
            not isinstance(source_text, str) or not source_text.strip()
        ):
            raise ValueError("source_text must be a non-empty string when provided")
        if source_language is not None and (
            not isinstance(source_language, str) or len(source_language) < 2
        ):
            raise ValueError("source_language must be a language tag when provided")
        with self._inference_lock:
            return self.parser.parse(
                normalized,
                modality=modality,
                request_id=request_id,
                **(
                    {
                        "source_text": source_text,
                        "source_language": source_language,
                    }
                    if source_text is not None
                    else {}
                ),
            )

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Consume the JSON-like message emitted by an upstream translator."""
        if not isinstance(message, dict):
            raise TypeError("message must be a dictionary")
        language = message.get("language", "en-US")
        if language not in {"en", "en-US", "en-GB"}:
            raise ValueError("message.language must identify English text")
        request_id = message.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise ValueError("message.request_id must be a string when provided")
        return self.parse_text(
            message.get("text"),
            request_id=request_id,
            modality=message.get("modality", "TEXT"),
            source_text=message.get("source_text"),
            source_language=message.get("source_language"),
        )

    def parse_source_text(self, text: str, *, request_id: str | None = None,
                          modality: str = 'TEXT') -> dict[str, Any]:
        """Parse supported Chinese source rules, never feed Chinese to the English model."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Source command must be a non-empty string')
        if len(text) > self.max_input_chars:
            raise ValueError('Source command exceeds max_input_chars')
        if modality not in {'TEXT', 'VOICE'}:
            raise ValueError('modality must be VOICE or TEXT')
        document = RuleIntentParser().parse(text, modality=modality, request_id=request_id)
        if document is None:
            document = make_document(raw_text=text, normalized_text=normalize_text(text),
                modality=modality, request_id=request_id, category='META_CONTROL', urgency='NORMAL',
                steps=[], status='NEEDS_CLARIFICATION', method='RULE', model=None, confidence=0.,
                latency_ms=0., missing_slots=['intent.steps'],
                warnings=['Chinese source expression requires an upstream translation or clarification.'])
        document['parse_result'].update(source='chinese_source_rules',
            source_kind='SOURCE_LANGUAGE_RULE', model_prediction=False)
        return document
