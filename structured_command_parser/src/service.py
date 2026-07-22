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
    allow_llm_fallback: bool = False
    semantic_model_path: str | None = None
    semantic_similarity_threshold: float = 0.58
    semantic_top_k: int = 7
    semantic_device: str = "cpu"
    semantic_cpu_threads: int = 1
    parser_backend: str = "modernbert"
    parser_device: str = "cuda"

    @classmethod
    def shared_model(
        cls,
        model_path: str,
        *,
        default_modality: str = "VOICE",
        max_input_chars: int = 512,
        allow_llm_fallback: bool = False,
        semantic_model_path: str | None = None,
        semantic_similarity_threshold: float = 0.58,
        semantic_top_k: int = 7,
        semantic_device: str = "cpu",
        semantic_cpu_threads: int = 1,
        parser_backend: str = "qwen",
        parser_device: str = "cuda",
    ) -> "CommandParserConfig":
        return cls(
            translator_model_path=model_path,
            parser_model_path=model_path,
            default_modality=default_modality,
            max_input_chars=max_input_chars,
            allow_llm_fallback=allow_llm_fallback,
            semantic_model_path=semantic_model_path,
            semantic_similarity_threshold=semantic_similarity_threshold,
            semantic_top_k=semantic_top_k,
            semantic_device=semantic_device,
            semantic_cpu_threads=semantic_cpu_threads,
            parser_backend=parser_backend,
            parser_device=parser_device,
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
            allow_llm_fallback=config.allow_llm_fallback,
            semantic_model_path=config.semantic_model_path,
            semantic_similarity_threshold=config.semantic_similarity_threshold,
            semantic_top_k=config.semantic_top_k,
            semantic_device=config.semantic_device,
            semantic_cpu_threads=config.semantic_cpu_threads,
            parser_backend=config.parser_backend,
            parser_device=config.parser_device,
        )
        self._inference_lock = Lock()

    @classmethod
    def realtime(
        cls,
        *,
        default_modality: str = "VOICE",
        max_input_chars: int = 512,
        semantic_model_path: str | None = None,
        semantic_similarity_threshold: float = 0.58,
        semantic_top_k: int = 7,
        semantic_device: str = "cpu",
        semantic_cpu_threads: int = 1,
    ) -> "DrivingCommandService":
        """Create the hard real-time service without requiring model weights."""
        return cls(
            CommandParserConfig.shared_model(
                "__realtime_no_model__",
                default_modality=default_modality,
                max_input_chars=max_input_chars,
                allow_llm_fallback=False,
                semantic_model_path=semantic_model_path,
                semantic_similarity_threshold=semantic_similarity_threshold,
                semantic_top_k=semantic_top_k,
                semantic_device=semantic_device,
                semantic_cpu_threads=semantic_cpu_threads,
            )
        )

    @classmethod
    def from_shared_model(
        cls,
        model_path: str,
        *,
        default_modality: str = "VOICE",
        max_input_chars: int = 512,
        allow_llm_fallback: bool = False,
        semantic_model_path: str | None = None,
        semantic_similarity_threshold: float = 0.58,
        semantic_top_k: int = 7,
        semantic_device: str = "cpu",
        semantic_cpu_threads: int = 1,
    ) -> "DrivingCommandService":
        return cls(
            CommandParserConfig.shared_model(
                model_path,
                default_modality=default_modality,
                max_input_chars=max_input_chars,
                allow_llm_fallback=allow_llm_fallback,
                semantic_model_path=semantic_model_path,
                semantic_similarity_threshold=semantic_similarity_threshold,
                semantic_top_k=semantic_top_k,
                semantic_device=semantic_device,
                semantic_cpu_threads=semantic_cpu_threads,
            )
        )

    @classmethod
    def production(
        cls,
        translator_model_path: str,
        modernbert_model_path: str,
        *,
        default_modality: str = "VOICE",
        max_input_chars: int = 512,
        semantic_model_path: str | None = None,
        semantic_similarity_threshold: float = 0.58,
        semantic_top_k: int = 7,
        semantic_device: str = "cpu",
        semantic_cpu_threads: int = 1,
        parser_device: str = "cuda",
    ) -> "DrivingCommandService":
        """Create the rule-first production chain with ModernBERT English parsing."""
        return cls(
            CommandParserConfig(
                translator_model_path=translator_model_path,
                parser_model_path=modernbert_model_path,
                default_modality=default_modality,
                max_input_chars=max_input_chars,
                allow_llm_fallback=True,
                semantic_model_path=semantic_model_path,
                semantic_similarity_threshold=semantic_similarity_threshold,
                semantic_top_k=semantic_top_k,
                semantic_device=semantic_device,
                semantic_cpu_threads=semantic_cpu_threads,
                parser_backend="modernbert",
                parser_device=parser_device,
            )
        )

    @classmethod
    def qwen_comparison(
        cls,
        model_path: str,
        **kwargs: Any,
    ) -> "DrivingCommandService":
        """Create an explicit Qwen parser comparison chain, never the default."""
        return cls.from_shared_model(
            model_path,
            allow_llm_fallback=True,
            **kwargs,
        )

    def warmup(self) -> None:
        """Load weights when optional LLM fallback is enabled."""
        self.pipeline.warmup()

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
