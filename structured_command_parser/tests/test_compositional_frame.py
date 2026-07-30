import unittest

from structured_command_parser.src.compositional_frame import (
    enrich_commands_with_frame,
    extract_semantic_frame,
)
from structured_command_parser.src.semantic_decomposer import (
    decompose_atomic_actions,
)
from structured_command_parser.src.semantic_normalizer import (
    filter_suppressed_actions,
    normalize_semantics,
)
from structured_command_parser.src.speed_slots import (
    restore_source_target_speeds,
)


class SemanticNormalizerTests(unittest.TestCase):
    def test_unseen_lane_change_paraphrase_is_canonicalized(self):
        result = normalize_semantics(
            "Find an opportunity to merge into the left lane."
        )
        self.assertEqual(
            result["normalized_text"],
            "change to the left lane when safe.",
        )
        self.assertEqual(
            result["edits"][0]["type"], "SYNONYM_CANONICALIZATION"
        )

    def test_explicit_negation_suppresses_only_the_negated_action(self):
        result = normalize_semantics(
            "Do not turn left, instead continue straight."
        )
        self.assertEqual(result["suppressed_intents"][0]["action"], "TURN")
        self.assertEqual(
            result["suppressed_intents"][0]["parameters"]["direction"],
            "LEFT",
        )
        commands = filter_suppressed_actions(
            [
                {"action": "TURN", "direction": "LEFT"},
                {"action": "PROCEED", "direction": "STRAIGHT"},
            ],
            result["suppressed_intents"],
        )
        self.assertEqual(commands, [{"action": "PROCEED", "direction": "STRAIGHT"}])

    def test_ambiguous_chinese_asr_homophone_requires_confirmation(self):
        result = normalize_semantics("前方路口又转")
        self.assertTrue(result["requires_confirmation"])
        self.assertEqual(result["edits"][0]["replacement"], "右转")
        self.assertIn("又转", result["normalized_text"])


class CompositionalFrameTests(unittest.TestCase):
    def test_compound_target_speed_survives_normalization_and_grounding(self):
        cases = (
            (
                "Stay in the current lane and drive at a speed of "
                "50 kilometers per hour.",
                ["KEEP_LANE", "SET_SPEED"],
                13.889,
            ),
            (
                "Maintain current lane, speed up to 60 kilometers per hour.",
                ["KEEP_LANE", "SET_SPEED"],
                16.667,
            ),
            (
                "Reduce to 40 kilometers per hour and keep in the current lane.",
                ["SET_SPEED", "KEEP_LANE"],
                11.111,
            ),
            (
                "Road clear, speed up to 50 kilometers per hour.",
                ["SET_SPEED"],
                13.889,
            ),
        )
        for text, expected_actions, expected_speed in cases:
            with self.subTest(text=text):
                commands = decompose_atomic_actions(text)
                self.assertEqual(
                    [command["action"] for command in commands],
                    expected_actions,
                )
                speed_commands = [
                    command
                    for command in commands
                    if command["action"] == "SET_SPEED"
                ]
                self.assertEqual(len(speed_commands), 1)
                self.assertAlmostEqual(
                    speed_commands[0]["target_speed_mps"],
                    expected_speed,
                    places=3,
                )

    def test_english_asr_speed_unit_variants_are_deterministic(self):
        cases = (
            "Cruise at 40 kph.",
            "Keep 40 kmph.",
            "Maintain a speed of 40 kilometres per hour.",
        )
        for text in cases:
            with self.subTest(text=text):
                commands = decompose_atomic_actions(text)
                self.assertEqual([item["action"] for item in commands], ["SET_SPEED"])
                self.assertAlmostEqual(
                    commands[0]["target_speed_mps"],
                    11.111,
                    places=3,
                )

    def test_source_language_speed_overrides_translation_slot(self):
        restored, changed = restore_source_target_speeds(
            [{"action": "SET_SPEED", "target_speed_mps": 13.889}],
            "保持40公里速度行驶",
        )
        self.assertTrue(changed)
        self.assertEqual(restored[0]["action"], "SET_SPEED")
        self.assertAlmostEqual(
            restored[0]["target_speed_mps"],
            11.111,
            places=3,
        )
        self.assertEqual(restored[0]["source_unit"], "km/h")

    def test_source_language_speed_upgrades_relative_speed_action(self):
        restored, changed = restore_source_target_speeds(
            [{"action": "ADJUST_SPEED", "change": "INCREASE"}],
            "保持40公里速度行驶",
        )
        self.assertTrue(changed)
        self.assertEqual(
            restored,
            [
                {
                    "action": "SET_SPEED",
                    "target_speed_mps": 11.111,
                    "source_value": 40.0,
                    "source_unit": "km/h",
                }
            ],
        )

    def test_unit_and_function_token_predictions_are_not_entities(self):
        text = "Maintain the current lane, speed up to 50 km/h."
        surfaces = ("the current", "to", "50 km", "h")
        predicted_spans = []
        for surface in surfaces:
            start = text.index(surface)
            predicted_spans.append(
                {
                    "role": "ENTITY",
                    "text": surface,
                    "start": start,
                    "end": start + len(surface),
                    "confidence": 0.99,
                }
            )
        frame = extract_semantic_frame(text, predicted_spans=predicted_spans)
        self.assertEqual(frame["entities"], [])

    def test_next_modifier_does_not_shadow_junction_entity(self):
        text = "Turn left at the next intersection."
        start = text.index("the next")
        frame = extract_semantic_frame(
            text,
            predicted_spans=[
                {
                    "role": "ENTITY",
                    "text": "the next",
                    "start": start,
                    "end": start + len("the next"),
                    "confidence": 0.99,
                }
            ],
        )
        self.assertEqual(len(frame["entities"]), 1)
        self.assertEqual(frame["entities"][0]["type"], "JUNCTION")

    def test_visible_relation_may_follow_its_entity(self):
        frame = extract_semantic_frame(
            "When the traffic light comes into view, pull over on the right."
        )
        self.assertEqual(len(frame["entities"]), 1)
        self.assertEqual(frame["entities"][0]["type"], "TRAFFIC_LIGHT")
        self.assertEqual(
            frame["goal_conditions"],
            [
                {
                    "predicate": "VISIBLE",
                    "subject": "EGO",
                    "object": "target_1",
                    "source_span": "comes into view",
                }
            ],
        )

    def test_stop_before_red_truck_keeps_entity_and_goal(self):
        frame = extract_semantic_frame(
            "Slow down and stop before the red truck."
        )
        self.assertEqual(len(frame["entities"]), 1)
        entity = frame["entities"][0]
        self.assertEqual(entity["type"], "VEHICLE")
        self.assertEqual(entity["canonical_attributes"]["color"], "RED")
        self.assertEqual(
            entity["canonical_attributes"]["vehicle_subtype"], "TRUCK"
        )
        self.assertEqual(frame["goal_conditions"][0]["predicate"], "BEFORE")

        commands = enrich_commands_with_frame(
            [
                {"action": "ADJUST_SPEED", "change": "DECREASE"},
                {"action": "STOP"},
            ],
            frame,
        )
        self.assertEqual(commands[0]["target_ref"], "target_1")
        self.assertEqual(
            commands[1]["goal_conditions"][0]["predicate"], "BEFORE"
        )

    def test_ordinal_junction_and_visibility_condition_are_composable(self):
        ordinal = extract_semantic_frame(
            "Turn right after the second junction."
        )
        self.assertEqual(
            ordinal["entities"][0]["canonical_attributes"]["ordinal"], 2
        )
        self.assertEqual(ordinal["goal_conditions"][0]["predicate"], "AFTER")

        visible = extract_semantic_frame(
            "When you see the bus stop, pull over and stop on the right."
        )
        self.assertEqual(visible["entities"][0]["type"], "LANDMARK")
        self.assertEqual(visible["goal_conditions"][0]["predicate"], "VISIBLE")

    def test_safe_distance_is_a_constraint_on_the_referred_vehicle(self):
        normalized = normalize_semantics(
            "Follow the vehicle ahead, but don't get too close."
        )
        frame = extract_semantic_frame(normalized["normalized_text"])
        self.assertEqual(frame["entities"][0]["type"], "VEHICLE")
        self.assertIn(
            "SAFE_DISTANCE",
            {item["predicate"] for item in frame["goal_conditions"]},
        )


if __name__ == "__main__":
    unittest.main()
