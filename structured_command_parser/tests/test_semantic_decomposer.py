import unittest

from structured_command_parser.src.semantic_decomposer import (
    decompose_atomic_actions,
)
from structured_command_parser.src.semantic_normalizer import normalize_semantics


def _actions(text: str) -> list[str]:
    return [item["action"] for item in decompose_atomic_actions(text)]


class SemanticDecomposerTests(unittest.TestCase):
    def test_overtake_then_return_is_two_ordered_actions(self) -> None:
        self.assertEqual(
            _actions(
                "Get past the red truck on the right and "
                "move back into the original lane."
            ),
            ["OVERTAKE", "RESUME"],
        )

    def test_entity_position_is_not_action_direction(self) -> None:
        commands = decompose_atomic_actions(
            "Get past the red truck on the right."
        )
        self.assertEqual(commands, [{"action": "OVERTAKE"}])

    def test_local_direction_binding_ignores_bus_stop_noun(self) -> None:
        commands = decompose_atomic_actions(
            "Once you see the bus stop, pull over to the right and stop."
        )
        self.assertEqual(
            commands,
            [
                {"action": "PULL_OVER", "direction": "RIGHT"},
                {"action": "STOP"},
            ],
        )

    def test_counterfactual_remainder(self) -> None:
        self.assertEqual(
            decompose_atomic_actions("continue straight"),
            [{"action": "PROCEED", "direction": "STRAIGHT"}],
        )

    def test_dummy_it_does_not_trigger_anaphora_clarification(self) -> None:
        result = normalize_semantics("Take the right lane when it is safe.")
        self.assertEqual(result["unresolved_references"], [])

    def test_referential_it_still_requires_clarification(self) -> None:
        result = normalize_semantics("Follow it, but keep a safe distance.")
        self.assertEqual(result["unresolved_references"], ["anaphoric_target"])


if __name__ == "__main__":
    unittest.main()
