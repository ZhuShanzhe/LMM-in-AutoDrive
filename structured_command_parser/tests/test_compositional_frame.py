import unittest

from structured_command_parser.src.compositional_frame import (
    enrich_commands_with_frame,
    extract_semantic_frame,
)
from structured_command_parser.src.semantic_normalizer import (
    filter_suppressed_actions,
    normalize_semantics,
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
