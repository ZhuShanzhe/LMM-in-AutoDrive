from __future__ import annotations

from time import perf_counter
from typing import Any

from .factory import make_document
from .llm_parser import QwenIntentParser
from .normalizer import normalize_text
from .rule_parser import RuleIntentParser


class HybridCommandParser:
    def __init__(self, model_path: str | None = None) -> None:
        self.rule_parser = RuleIntentParser()
        self.llm_parser = QwenIntentParser(model_path) if model_path else None

    def parse(
        self,
        text: str,
        *,
        modality: str = "TEXT",
        request_id: str | None = None,
    ) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("Command text cannot be empty")
        if modality not in {"TEXT", "VOICE"}:
            raise ValueError("modality must be TEXT or VOICE")

        rule_result = self.rule_parser.parse(
            text, modality=modality, request_id=request_id
        )
        if rule_result is not None:
            return rule_result
        if self.llm_parser is not None:
            return self.llm_parser.parse(
                text, modality=modality, request_id=request_id
            )

        started = perf_counter()
        normalized_text = normalize_text(text)
        return make_document(
            raw_text=text,
            normalized_text=normalized_text,
            modality=modality,
            category="META_CONTROL",
            urgency="NORMAL",
            steps=[],
            status="NEEDS_CLARIFICATION",
            method="HYBRID",
            model=None,
            confidence=0.0,
            latency_ms=(perf_counter() - started) * 1000,
            request_id=request_id,
            missing_slots=["intent.steps"],
            warnings=["规则无法完整解析，且当前未启用 Qwen 模型。"],
            clarification_question="请换一种更明确的方式说明驾驶动作、目标和方向。",
        )

