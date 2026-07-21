from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from structured_command_parser.src.english_parser import QwenEnglishIntentParser
from structured_command_parser.src.translator import ConstrainedQwenTranslator


class TranslationGlossaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.translator = ConstrainedQwenTranslator("unused-model")

    def test_longest_glossary_term_wins(self) -> None:
        protected, matches = self.translator._protect_terms("请向左变道后减速")
        self.assertIn("__DRIVE_TERM_000__", protected)
        self.assertIn("__DRIVE_TERM_001__", protected)
        self.assertEqual(
            [match.source for match in matches], ["向左变道", "减速"]
        )

    def test_placeholders_restore_exact_terms(self) -> None:
        protected, matches = self.translator._protect_terms("避让行人")
        self.assertEqual(
            self.translator._restore_placeholders(protected, matches),
            "yield to pedestrian",
        )

    def test_known_alias_is_replaced_by_canonical_term(self) -> None:
        _, matches = self.translator._protect_terms("保持当前车道")
        self.assertEqual(
            self.translator._enforce_terms("Go straight.", matches),
            "keep the current lane.",
        )

    def test_missing_canonical_term_is_detected(self) -> None:
        _, matches = self.translator._protect_terms("避让行人")
        missing = self.translator._missing_targets("Yield safely.", matches)
        self.assertEqual({item.target for item in missing}, {"yield to", "pedestrian"})

    def test_ambiguous_direction_is_not_invented(self) -> None:
        lane = self.translator._preserve_source_semantics(
            "往旁边并线", "change lane to the left"
        )
        turn = self.translator._preserve_source_semantics(
            "到前面拐弯", "turn right ahead"
        )
        self.assertEqual(lane, "change lane")
        self.assertNotRegex(turn, r"left|right")
        self.assertEqual(
            self.translator._preserve_source_semantics(
                "换个道", "change lane to the left"
            ),
            "change lane",
        )

    def test_clean_preserves_empty_output_for_retry_detection(self) -> None:
        self.assertEqual(self.translator._clean("   "), "")

    def test_empty_generation_uses_clarification_fallback(self) -> None:
        self.translator.runtime.generate = lambda *args, **kwargs: ""
        result = self.translator.translate("执行一个驾驶动作")
        self.assertEqual(result.translated_text, "ambiguous driving command")
        self.assertTrue(any("empty text" in item for item in result.warnings))


class EnglishPayloadTests(unittest.TestCase):
    def test_trailing_model_text_is_ignored(self) -> None:
        decoded = QwenEnglishIntentParser._decode_payload(
            '{"commands": [], "status": "NEEDS_CLARIFICATION"} trailing text'
        )
        self.assertEqual(decoded["status"], "NEEDS_CLARIFICATION")

    def test_collision_request_is_unsupported(self) -> None:
        payload = QwenEnglishIntentParser._normalize_payload(
            {"commands": [{"action": "RESUME"}]},
            "Collide with the vehicle ahead.",
        )
        self.assertEqual(payload["status"], "UNSUPPORTED")
        self.assertEqual(payload["commands"], [])

    def test_speed_and_lane_are_canonicalized(self) -> None:
        payload = QwenEnglishIntentParser._normalize_payload(
            {
                "commands": [
                    {"action": "SET_SPEED", "target_speed_mps": 60},
                    {"action": "CHANGE_LANE", "direction": "RIGHT"},
                ]
            },
            "Set the speed to 60 km/h and change lane to the left.",
        )
        self.assertAlmostEqual(payload["commands"][0]["target_speed_mps"], 16.667)
        self.assertEqual(payload["commands"][1]["direction"], "LEFT")

    def test_explicit_turn_and_meta_actions_are_restored(self) -> None:
        turn = QwenEnglishIntentParser._normalize_payload(
            {"commands": [], "status": "NEEDS_CLARIFICATION"},
            "Turn left at the upcoming junction.",
        )
        self.assertEqual(turn["commands"], [{"action": "TURN", "direction": "LEFT"}])
        cancel = QwenEnglishIntentParser._normalize_payload(
            {"commands": [], "status": "NEEDS_CLARIFICATION"},
            "Cancel the previous command.",
        )
        self.assertEqual(cancel["commands"], [{"action": "CANCEL"}])

    def test_negative_speed_delta_becomes_directional_change(self) -> None:
        payload = QwenEnglishIntentParser._normalize_payload(
            {"commands": [{"action": "ADJUST_SPEED", "speed_delta_mps": -5}]},
            "Slow down.",
        )
        self.assertEqual(
            payload["commands"], [{"action": "ADJUST_SPEED", "change": "DECREASE"}]
        )

    def test_general_turn_is_not_a_lane_change(self) -> None:
        payload = QwenEnglishIntentParser._normalize_payload(
            {"commands": [{"action": "CHANGE_LANE", "direction": "LEFT"}]},
            "Turn left in 200 meters.",
        )
        self.assertEqual(payload["commands"], [{"action": "TURN", "direction": "LEFT"}])

    def test_paraphrased_speed_units_are_supported(self) -> None:
        payload = QwenEnglishIntentParser._normalize_payload(
            {"commands": [{"action": "ADJUST_SPEED"}]},
            "Drive at 15 meters per second.",
        )
        self.assertEqual(
            payload["commands"],
            [{"action": "SET_SPEED", "target_speed_mps": 15.0}],
        )

    def test_road_user_actions_are_not_substituted(self) -> None:
        avoid = QwenEnglishIntentParser._normalize_payload(
            {"commands": [{"action": "YIELD"}]},
            "Avoid the cyclist on the right.",
        )
        overtake = QwenEnglishIntentParser._normalize_payload(
            {"commands": [{"action": "AVOID"}]},
            "Overtake the slow vehicle ahead.",
        )
        self.assertEqual([item["action"] for item in avoid["commands"]], ["AVOID"])
        self.assertEqual(
            [item["action"] for item in overtake["commands"]], ["OVERTAKE"]
        )

    def test_decelerate_and_yield_is_one_speed_action(self) -> None:
        payload = QwenEnglishIntentParser._normalize_payload(
            {"commands": [{"action": "YIELD", "target_type": "PEDESTRIAN"}]},
            "Decelerate and yield to the pedestrian.",
        )
        self.assertEqual(
            [command["action"] for command in payload["commands"]],
            ["ADJUST_SPEED"],
        )

    def test_traffic_rule_violations_are_unsupported(self) -> None:
        for text in (
            "Ignore the red light and continue.",
            "Drive against traffic to bypass congestion.",
            "Set the speed to 180 km/h.",
        ):
            with self.subTest(text=text):
                payload = QwenEnglishIntentParser._normalize_payload(
                    {"commands": [{"action": "RESUME"}]}, text
                )
                self.assertEqual(payload["status"], "UNSUPPORTED")
                self.assertEqual(payload["commands"], [])

    def test_reference_stop_requires_clarification(self) -> None:
        payload = QwenEnglishIntentParser._normalize_payload(
            {"commands": [{"action": "STOP"}]}, "Stop at that side."
        )
        self.assertEqual(payload["status"], "NEEDS_CLARIFICATION")
        self.assertEqual(payload["commands"], [])

    def test_vague_avoid_target_requires_clarification(self) -> None:
        payload = QwenEnglishIntentParser._normalize_payload(
            {"commands": [{"action": "AVOID"}]}, "Avoid that thing."
        )
        self.assertEqual(payload["status"], "NEEDS_CLARIFICATION")
        self.assertEqual(payload["commands"], [])

    def test_reduce_the_speed_is_a_decrease(self) -> None:
        payload = QwenEnglishIntentParser._normalize_payload(
            {"commands": []}, "Reduce the speed."
        )
        self.assertEqual(
            payload["commands"], [{"action": "ADJUST_SPEED", "change": "DECREASE"}]
        )

    def test_overtake_does_not_carry_lane_direction(self) -> None:
        payload = QwenEnglishIntentParser._normalize_payload(
            {
                "commands": [
                    {"action": "CHANGE_LANE", "direction": "LEFT"},
                    {"action": "OVERTAKE", "direction": "LEFT"},
                ]
            },
            "Change lane to the left and overtake the slow vehicle.",
        )
        self.assertEqual(payload["commands"][0]["direction"], "LEFT")
        self.assertNotIn("direction", payload["commands"][1])

    def test_stop_before_continuing_preserves_order(self) -> None:
        payload = QwenEnglishIntentParser._normalize_payload(
            {
                "commands": [
                    {"action": "RESUME"},
                    {"action": "STOP"},
                ]
            },
            "Continue driving, stop at the red light before continuing driving.",
        )
        self.assertEqual(
            [command["action"] for command in payload["commands"]],
            ["STOP", "RESUME"],
        )

    def test_pull_over_speed_and_resume_are_not_collapsed_to_stop(self) -> None:
        payload = QwenEnglishIntentParser._normalize_payload(
            {
                "commands": [
                    {"action": "PULL_OVER"},
                    {"action": "STOP"},
                    {"action": "RESUME"},
                ]
            },
            "Pull over, decelerate to 30 km/h, then continue driving.",
        )
        self.assertEqual(
            [command["action"] for command in payload["commands"]],
            ["PULL_OVER", "SET_SPEED", "RESUME"],
        )


if __name__ == "__main__":
    unittest.main()
