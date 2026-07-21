import copy
import json
import unittest
from pathlib import Path

from scene_understanding.core.risk_assessment import assess_world_state
from scene_understanding.core.semantic_alignment import (
    align_instruction_reference,
    validate_semantic_alignment,
)


ROOT = Path(__file__).resolve().parents[2]
WORLD_STATE_EXAMPLE = ROOT / "schemas" / "examples" / "world_state.example.json"
ALIGNMENT_EXAMPLE = ROOT / "schemas" / "examples" / "semantic_alignment.example.json"


class SemanticAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world_state = json.loads(WORLD_STATE_EXAMPLE.read_text(encoding="utf-8"))

    def test_aligns_front_vehicle_with_risk(self):
        risk = assess_world_state(self.world_state)
        result = align_instruction_reference(
            "前车",
            self.world_state,
            risk_assessment=risk,
        )
        self.assertEqual(validate_semantic_alignment(result), [])
        self.assertTrue(result["alignment_success"])
        self.assertEqual(result["matched_entity"]["entity_id"], "carla_actor_42")
        self.assertEqual(result["matched_entity"]["risk_level"], "medium")

    def test_aligns_left_lane(self):
        result = align_instruction_reference("向左变道", self.world_state)
        self.assertTrue(result["alignment_success"])
        self.assertEqual(result["matched_entity"]["entity_id"], "lane:12:-2")
        self.assertEqual(result["matched_entity"]["lane_relation"], "left_adjacent_lane")

    def test_reports_unavailable_right_lane(self):
        result = align_instruction_reference("右车道", self.world_state)
        self.assertFalse(result["alignment_success"])
        self.assertEqual(result["reason_code"], "right_lane_unavailable")

    def test_aligns_current_junction(self):
        state = copy.deepcopy(self.world_state)
        state["ego"]["is_junction"] = True
        state["environment"]["is_intersection"] = True
        result = align_instruction_reference("路口", state)
        self.assertTrue(result["alignment_success"])
        self.assertEqual(result["matched_entity"]["entity_type"], "junction")

    def test_unknown_reference_fails_safely(self):
        result = align_instruction_reference("播放音乐", self.world_state)
        self.assertFalse(result["alignment_success"])
        self.assertIsNone(result["matched_entity"])
        self.assertEqual(result["reason_code"], "unknown_reference")

    def test_checked_in_example_is_valid(self):
        example = json.loads(ALIGNMENT_EXAMPLE.read_text(encoding="utf-8"))
        self.assertEqual(validate_semantic_alignment(example), [])


if __name__ == "__main__":
    unittest.main()
