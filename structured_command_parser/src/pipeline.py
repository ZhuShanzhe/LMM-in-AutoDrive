from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import uuid4

from .english_parser import QwenEnglishIntentParser
from .translator import ConstrainedQwenTranslator


class ChineseEnglishCommandPipeline:
    def __init__(self, translator_model_path: str, parser_model_path: str) -> None:
        self.translator = ConstrainedQwenTranslator(translator_model_path)
        self.parser = QwenEnglishIntentParser(parser_model_path)
        if translator_model_path == parser_model_path:
            self.parser.runtime = self.translator.runtime

    def parse(
        self,
        text: str,
        *,
        modality: str = "VOICE",
        request_id: str | None = None,
    ) -> dict[str, Any]:
        if modality not in {"TEXT", "VOICE"}:
            raise ValueError("modality must be TEXT or VOICE")
        pipeline_request_id = request_id or f"cmd-{uuid4().hex[:16]}"
        started = perf_counter()
        translation = self.translator.translate(text)
        intent = self.parser.parse(
            translation.translated_text,
            modality=modality,
            request_id=pipeline_request_id,
        )
        return {
            "pipeline_version": "1.0.0",
            "request_id": pipeline_request_id,
            "source": {
                "modality": modality,
                "language": "zh-CN",
                "raw_text": text,
                "normalized_text": translation.normalized_source_text,
            },
            "translation": translation.to_dict(),
            "driving_intent": intent,
            "total_latency_ms": round((perf_counter() - started) * 1000, 3),
        }
