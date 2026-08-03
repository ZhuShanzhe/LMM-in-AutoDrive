from __future__ import annotations

import unittest

from structured_command_parser.src.service import (
    CommandParserConfig,
    DrivingCommandService,
)


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def parse(self, text: str, *, modality: str, request_id: str | None):
        call = {"text": text, "modality": modality, "request_id": request_id}
        self.calls.append(call)
        return {"request_id": request_id, "source": call, "driving_intent": {}}


class DrivingCommandServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = FakePipeline()
        self.service = DrivingCommandService(
            CommandParserConfig.shared_model("unused", max_input_chars=20),
            pipeline=self.pipeline,  # type: ignore[arg-type]
        )

    def test_asr_text_is_trimmed_and_forwarded(self) -> None:
        result = self.service.parse_asr_text("  向左变道  ", request_id="asr-1")
        self.assertEqual(result["request_id"], "asr-1")
        self.assertEqual(
            self.pipeline.calls,
            [{"text": "向左变道", "modality": "VOICE", "request_id": "asr-1"}],
        )

    def test_message_contract_allows_text_modality(self) -> None:
        self.service.handle_message(
            {"request_id": "ui-1", "text": "停车", "modality": "TEXT"}
        )
        self.assertEqual(self.pipeline.calls[0]["modality"], "TEXT")

    def test_invalid_input_is_rejected_before_inference(self) -> None:
        for text in ("", " " * 3, "过长" * 11):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.service.parse_asr_text(text)
        self.assertEqual(self.pipeline.calls, [])

    def test_realtime_factory_does_not_enable_llm_fallback(self) -> None:
        service = DrivingCommandService.realtime()
        self.assertFalse(service.config.allow_llm_fallback)
        result = service.parse_asr_text("向左变道", request_id="fast-service")
        self.assertEqual(result["execution_path"], "REALTIME_RULE")
        self.assertLess(result["total_latency_ms"], 50.0)

    def test_realtime_factory_accepts_semantic_model(self) -> None:
        service = DrivingCommandService.realtime(semantic_model_path="semantic-model")
        self.assertEqual(service.config.semantic_model_path, "semantic-model")
        self.assertFalse(service.config.allow_llm_fallback)

    def test_production_factory_uses_modernbert(self) -> None:
        service = DrivingCommandService.production("translator", "modernbert")
        self.assertTrue(service.config.allow_llm_fallback)
        self.assertEqual(service.config.parser_backend, "modernbert")
        self.assertEqual(service.config.parser_model_path, "modernbert")

    def test_qwen_comparison_is_explicit(self) -> None:
        service = DrivingCommandService.qwen_comparison("qwen")
        self.assertTrue(service.config.allow_llm_fallback)
        self.assertEqual(service.config.parser_backend, "qwen")


if __name__ == "__main__":
    unittest.main()
