from __future__ import annotations

import unittest

from structured_command_parser.src.pipeline import ChineseEnglishCommandPipeline
from structured_command_parser.src.modernbert_parser import ModernBertEnglishIntentParser


class RealtimePipelineTests(unittest.TestCase):
    def test_modernbert_is_default_model_backend(self) -> None:
        pipeline = ChineseEnglishCommandPipeline(
            "unused-translator",
            "unused-modernbert",
            allow_llm_fallback=False,
        )
        self.assertEqual(pipeline.parser_backend, "modernbert")
        self.assertIsInstance(pipeline.parser, ModernBertEnglishIntentParser)

    def test_known_command_skips_both_model_generations(self) -> None:
        pipeline = ChineseEnglishCommandPipeline(
            "unused-model",
            "unused-model",
            allow_llm_fallback=False,
        )
        result = pipeline.parse("突发车辆加塞，紧急避让", request_id="fast-1")
        self.assertEqual(result["execution_path"], "REALTIME_RULE")
        self.assertEqual(result["translation"]["model"], "SKIPPED")
        self.assertEqual(
            result["driving_intent"]["intent"]["steps"][0]["action"], "AVOID"
        )
        self.assertLess(result["total_latency_ms"], 50.0)

    def test_unknown_command_fails_safe_without_llm_delay(self) -> None:
        pipeline = ChineseEnglishCommandPipeline(
            "unused-model",
            "unused-model",
            allow_llm_fallback=False,
        )
        result = pipeline.parse("就照我刚才想的那样开", request_id="fast-2")
        self.assertEqual(result["execution_path"], "REALTIME_SAFE_FALLBACK")
        self.assertEqual(
            result["driving_intent"]["parse_result"]["status"],
            "NEEDS_CLARIFICATION",
        )
        self.assertLess(result["total_latency_ms"], 50.0)

    def test_high_confidence_rule_short_circuits_semantic_model(self) -> None:
        pipeline = ChineseEnglishCommandPipeline(
            "unused-model",
            "unused-model",
            allow_llm_fallback=False,
            semantic_model_path="missing-model",
        )
        result = pipeline.parse("向左变道", request_id="fast-semantic-skip")
        self.assertEqual(result["execution_path"], "REALTIME_RULE")


if __name__ == "__main__":
    unittest.main()
