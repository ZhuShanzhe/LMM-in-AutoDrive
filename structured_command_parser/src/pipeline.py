from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import uuid4

from .english_parser import QwenEnglishIntentParser
from .factory import make_document
from .modernbert_parser import ModernBertEnglishIntentParser
from .normalizer import normalize_text
from .rule_parser import RuleIntentParser
from .semantic_parser import SemanticIntentParser
from .translator import ConstrainedQwenTranslator


class ChineseEnglishCommandPipeline:
    def __init__(
        self,
        translator_model_path: str,
        parser_model_path: str,
        *,
        allow_llm_fallback: bool = True,
        semantic_model_path: str | None = None,
        semantic_similarity_threshold: float = 0.58,
        semantic_top_k: int = 7,
        semantic_device: str = "cpu",
        semantic_cpu_threads: int = 1,
        parser_backend: str = "modernbert",
        parser_device: str = "cuda",
    ) -> None:
        if parser_backend not in {"modernbert", "qwen"}:
            raise ValueError("parser_backend must be modernbert or qwen")
        self.allow_llm_fallback = allow_llm_fallback
        self.parser_backend = parser_backend
        self.rule_parser = RuleIntentParser()
        self.semantic_parser = (
            SemanticIntentParser(
                semantic_model_path,
                similarity_threshold=semantic_similarity_threshold,
                top_k=semantic_top_k,
                device=semantic_device,
                cpu_threads=semantic_cpu_threads,
            )
            if semantic_model_path
            else None
        )
        self.translator = ConstrainedQwenTranslator(translator_model_path)
        self.parser = (
            ModernBertEnglishIntentParser(parser_model_path, device=parser_device)
            if parser_backend == "modernbert"
            else QwenEnglishIntentParser(parser_model_path)
        )
        if (
            parser_backend == "qwen"
            and translator_model_path == parser_model_path
        ):
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
        fast_intent = self.rule_parser.parse(
            text,
            modality=modality,
            request_id=pipeline_request_id,
        )
        if self._rule_can_short_circuit(fast_intent):
            return self._result(
                text,
                modality,
                pipeline_request_id,
                fast_intent,
                started,
                execution_path="REALTIME_RULE",
            )
        semantic_intent = (
            self.semantic_parser.parse(
                text,
                modality=modality,
                request_id=pipeline_request_id,
            )
            if self.semantic_parser is not None
            else None
        )
        if self._prefer_semantic(fast_intent, semantic_intent):
            return self._result(
                text,
                modality,
                pipeline_request_id,
                semantic_intent,
                started,
                execution_path="SEMANTIC_RETRIEVAL",
            )
        if fast_intent is not None:
            return self._result(
                text,
                modality,
                pipeline_request_id,
                fast_intent,
                started,
                execution_path="REALTIME_RULE",
            )

        if not self.allow_llm_fallback:
            normalized = normalize_text(text)
            intent = make_document(
                raw_text=text,
                normalized_text=normalized,
                modality=modality,
                category="META_CONTROL",
                urgency="NORMAL",
                steps=[],
                status="NEEDS_CLARIFICATION",
                method="HYBRID",
                model=None,
                confidence=0.0,
                latency_ms=(perf_counter() - started) * 1000,
                request_id=pipeline_request_id,
                missing_slots=["intent.steps"],
                warnings=["实时规则未覆盖该表达，LLM 回退已关闭。"],
                clarification_question="请用更明确的驾驶动作、目标和方向重新表述。",
            )
            return self._result(
                text,
                modality,
                pipeline_request_id,
                intent,
                started,
                execution_path="REALTIME_SAFE_FALLBACK",
            )

        translation = self.translator.translate(text)
        intent = self.parser.parse(
            translation.translated_text,
            modality=modality,
            request_id=pipeline_request_id,
        )
        return {
            "pipeline_version": "1.1.0",
            "execution_path": (
                "TRANSLATION_MODERNBERT_FALLBACK"
                if self.parser_backend == "modernbert"
                else "TRANSLATION_QWEN_COMPARISON"
            ),
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

    def warmup(self) -> None:
        if self.semantic_parser is not None:
            self.semantic_parser.load()
        if self.allow_llm_fallback:
            self.translator.runtime.load()
            if self.parser_backend == "modernbert":
                self.parser.warmup()

    @staticmethod
    def _rule_can_short_circuit(rule_intent: dict[str, Any] | None) -> bool:
        return rule_intent is not None

    @staticmethod
    def _prefer_semantic(
        rule_intent: dict[str, Any] | None,
        semantic_intent: dict[str, Any] | None,
    ) -> bool:
        if semantic_intent is None:
            return False
        if rule_intent is None:
            return True
        rule_status = rule_intent["parse_result"]["status"]
        semantic_status = semantic_intent["parse_result"]["status"]
        if rule_status in {"UNSUPPORTED", "NEEDS_CLARIFICATION", "INVALID"}:
            return False
        if semantic_status != "VALID":
            return False
        rule_actions = [step["action"] for step in rule_intent["intent"]["steps"]]
        semantic_actions = [
            step["action"] for step in semantic_intent["intent"]["steps"]
        ]
        return len(semantic_actions) > len(rule_actions)

    def _result(
        self,
        text: str,
        modality: str,
        request_id: str,
        intent: dict[str, Any],
        started: float,
        *,
        execution_path: str,
    ) -> dict[str, Any]:
        normalized = normalize_text(text)
        return {
            "pipeline_version": "1.1.0",
            "execution_path": execution_path,
            "request_id": request_id,
            "source": {
                "modality": modality,
                "language": "zh-CN",
                "raw_text": text,
                "normalized_text": normalized,
            },
            "translation": {
                "source_text": text,
                "normalized_source_text": normalized,
                "translated_text": "",
                "source_language": "zh-CN",
                "target_language": "en-US",
                "model": "SKIPPED",
                "glossary_version": self.translator.glossary_version,
                "matched_terms": [],
                "term_constraints_passed": True,
                "warnings": ["实时路径已跳过生成式翻译。"],
                "latency_ms": 0.0,
            },
            "driving_intent": intent,
            "total_latency_ms": round((perf_counter() - started) * 1000, 3),
        }
