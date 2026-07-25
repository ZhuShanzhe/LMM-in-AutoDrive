import copy
import json
import unittest
from pathlib import Path

from scene_understanding.src.control_decision import (
    build_control_decision,
    validate_control_decision,
)


ROOT = Path(__file__).resolve().parents[2]


def load_example(name: str) -> dict:
    return json.loads(
        (ROOT / "scene_understanding" / "schemas" / "examples" / name).read_text(encoding="utf-8")
    )


def driving_intent(
    action: str = "ADJUST_SPEED",
    parameters: dict | None = None,
    *,
    target: dict | None = None,
    on_blocked: str = "SAFE_STOP",
    status: str = "VALID",
) -> dict:
    steps = []
    if status == "VALID":
        step = {
            "step_id": "step_1",
            "action": action,
            "target": target,
            "parameters": parameters or {},
            "depends_on": [],
            "preconditions": [],
            "on_blocked": on_blocked,
        }
        steps.append(step)
    return {
        "schema_version": "1.0.0",
        "request_id": "decision-test-001",
        "input": {
            "modality": "TEXT",
            "language": "zh-CN",
            "raw_text": "test",
            "normalized_text": "test",
        },
        "intent": {
            "category": "BASIC_CONTROL",
            "urgency": "NORMAL",
            "steps": steps,
            "constraints": {
                "safety_first": True,
                "obey_traffic_rules": True,
                "driving_style": "CONSERVATIVE",
            },
        },
        "parse_result": {
            "status": status,
            "method": "RULE",
            "model": None,
            "confidence": 0.95,
            "missing_slots": [],
            "warnings": [],
            "latency_ms": 1.0,
        },
    }


def semantic_alignment(
    intent: dict,
    frame_id: str,
    *,
    required: bool = False,
    success: bool | None = None,
    reason: str = "target_not_required",
) -> dict:
    steps = []
    if intent["intent"]["steps"]:
        step = intent["intent"]["steps"][0]
        entity = None
        if success is True:
            entity = {
                "entity_type": "actor",
                "entity_id": "carla_actor_42",
                "category": "vehicle",
                "distance_m": 20.0,
                "relative_position": "front",
                "lane_relation": "ego_lane",
                "risk_level": "low",
            }
        steps.append(
            {
                "step_id": step["step_id"],
                "action": step["action"],
                "target": step.get("target"),
                "alignment_required": required,
                "alignment_success": success,
                "candidate_count": 1 if success else 0,
                "matched_entity": entity,
                "reason_code": reason,
            }
        )
    return {
        "schema_version": "1.0.0",
        "request_id": intent["request_id"],
        "world_state_frame_id": frame_id,
        "parse_status": intent["parse_result"]["status"],
        "alignment_status": "COMPLETE" if success else "NOT_REQUIRED",
        "target_count": int(required),
        "matched_target_count": int(success is True),
        "step_alignments": steps,
    }


class ControlDecisionTests(unittest.TestCase):
    def setUp(self):
        self.world_state = load_example("world_state.example.json")
        self.risk = load_example("risk_assessment.example.json")
        self.risk["risk_level"] = "low"
        self.risk["reason_codes"] = ["distance_above_25m"]
        self.risk["recommended_action"] = "monitor"

    def build(self, intent: dict, alignment: dict | None = None) -> dict:
        if alignment is None:
            alignment = semantic_alignment(intent, self.world_state["frame_id"])
        return build_control_decision(
            intent, self.world_state, alignment, self.risk
        )

    def test_emits_controller_compatible_deceleration(self):
        intent = driving_intent(
            "ADJUST_SPEED", {"change": "DECREASE"}, target=None
        )
        result = self.build(intent)
        self.assertEqual(validate_control_decision(result), [])
        self.assertEqual(result["decision_status"], "READY")
        self.assertEqual(result["action"], "decelerate")
        self.assertEqual(result["target_speed_kmh"], 36.0)
        self.assertEqual(result["source_step_id"], "step_1")

    def test_risk_deceleration_overrides_lane_change(self):
        intent = driving_intent(
            "CHANGE_LANE", {"direction": "LEFT"}, target=None
        )
        self.risk["risk_level"] = "medium"
        self.risk["reason_codes"] = ["distance_10_to_25m"]
        self.risk["recommended_action"] = "decelerate"
        result = self.build(intent)
        self.assertEqual(result["decision_status"], "BLOCKED")
        self.assertEqual(result["action"], "decelerate")
        self.assertIn("risk_requires_deceleration", result["blocked_reason_codes"])

    def test_emergency_risk_has_highest_priority(self):
        intent = driving_intent("KEEP_LANE")
        self.risk["risk_level"] = "high"
        self.risk["reason_codes"] = ["collision_event"]
        self.risk["recommended_action"] = "emergency_brake"
        result = self.build(intent)
        self.assertEqual(result["action"], "emergency_brake")
        self.assertTrue(result["emergency"])
        self.assertEqual(result["target_speed_kmh"], 0.0)

    def test_non_valid_parse_status_stops(self):
        intent = driving_intent(status="NEEDS_CLARIFICATION")
        alignment = semantic_alignment(intent, self.world_state["frame_id"])
        result = self.build(intent, alignment)
        self.assertEqual(result["decision_status"], "SAFE_FALLBACK")
        self.assertEqual(result["action"], "stop")
        self.assertIsNone(result["source_step_id"])

    def test_unmatched_required_target_obeys_wait_for_safe(self):
        target = {"type": "PEDESTRIAN", "relation": "AHEAD_CROSSING"}
        intent = driving_intent(
            "ADJUST_SPEED",
            {"change": "DECREASE"},
            target=target,
            on_blocked="WAIT_FOR_SAFE",
        )
        alignment = semantic_alignment(
            intent,
            self.world_state["frame_id"],
            required=True,
            success=False,
            reason="no_matching_entity",
        )
        result = self.build(intent, alignment)
        self.assertEqual(result["decision_status"], "BLOCKED")
        self.assertEqual(result["action"], "decelerate")
        self.assertEqual(
            result["blocked_reason_codes"],
            ["no_matching_entity", "wait_for_safe"],
        )

    def test_unsafe_lane_change_uses_safe_stop_policy(self):
        intent = driving_intent(
            "CHANGE_LANE", {"direction": "LEFT"}, target=None
        )
        self.risk["lane_change"]["left"]["is_safe"] = False
        self.risk["lane_change"]["left"]["reason_codes"] = [
            "target_lane_front_gap_too_small"
        ]
        result = self.build(intent)
        self.assertEqual(result["decision_status"], "BLOCKED")
        self.assertEqual(result["action"], "stop")
        self.assertEqual(result["target_lane"], None)
        self.assertEqual(
            result["blocked_reason_codes"], ["target_lane_front_gap_too_small"]
        )

    def test_turn_without_planner_location_does_not_steer_blindly(self):
        intent = driving_intent(
            "TURN", {"direction": "LEFT"}, on_blocked="WAIT_FOR_SAFE"
        )
        result = self.build(intent)
        self.assertEqual(result["decision_status"], "BLOCKED")
        self.assertEqual(result["action"], "decelerate")
        self.assertIn("turn_target_location_missing", result["blocked_reason_codes"])

    def test_set_speed_converts_mps_to_kmh(self):
        intent = driving_intent("SET_SPEED", {"target_speed_mps": 16.667})
        result = self.build(intent)
        self.assertEqual(result["action"], "keep_lane")
        self.assertAlmostEqual(result["target_speed_kmh"], 60.0012)

    def test_grounded_overtake_accelerates_after_lane_change(self):
        intent = driving_intent(
            "OVERTAKE",
            target={"type": "SLOW_VEHICLE", "relation": "AHEAD"},
        )
        alignment = semantic_alignment(
            intent,
            self.world_state["frame_id"],
            required=True,
            success=True,
            reason="matched_world_object",
        )
        result = self.build(intent, alignment)
        self.assertEqual(result["decision_status"], "READY")
        self.assertEqual(result["action"], "accelerate")
        self.assertEqual(result["matched_entity_id"], "carla_actor_42")

    def test_unmatched_overtake_never_accelerates(self):
        intent = driving_intent(
            "OVERTAKE",
            target={"type": "SLOW_VEHICLE", "relation": "AHEAD"},
            on_blocked="SAFE_STOP",
        )
        alignment = semantic_alignment(
            intent,
            self.world_state["frame_id"],
            required=True,
            success=False,
            reason="no_matching_entity",
        )
        result = self.build(intent, alignment)
        self.assertEqual(result["decision_status"], "BLOCKED")
        self.assertEqual(result["action"], "stop")
        self.assertEqual(result["target_speed_kmh"], 0.0)
        self.assertEqual(
            result["blocked_reason_codes"],
            ["no_matching_entity", "safe_stop"],
        )

    def test_overtake_acceleration_remains_below_risk_override(self):
        intent = driving_intent(
            "OVERTAKE",
            target={"type": "SLOW_VEHICLE", "relation": "AHEAD"},
        )
        alignment = semantic_alignment(
            intent,
            self.world_state["frame_id"],
            required=True,
            success=True,
            reason="matched_world_object",
        )
        self.risk["risk_level"] = "medium"
        self.risk["reason_codes"] = ["distance_10_to_25m"]
        self.risk["recommended_action"] = "decelerate"
        result = self.build(intent, alignment)
        self.assertEqual(result["decision_status"], "BLOCKED")
        self.assertEqual(result["action"], "decelerate")
        self.assertIn("risk_requires_deceleration", result["blocked_reason_codes"])

    def test_stateful_executor_can_select_later_step(self):
        intent = driving_intent("KEEP_LANE")
        intent["intent"]["steps"].append(
            {
                "step_id": "step_2",
                "action": "SET_SPEED",
                "target": None,
                "parameters": {"target_speed_mps": 5.0},
                "depends_on": ["step_1"],
                "preconditions": [],
                "on_blocked": "SAFE_STOP",
            }
        )
        alignment = semantic_alignment(intent, self.world_state["frame_id"])
        alignment["step_alignments"].append(
            {
                "step_id": "step_2",
                "action": "SET_SPEED",
                "target": None,
                "alignment_required": False,
                "alignment_success": None,
                "candidate_count": 0,
                "matched_entity": None,
                "reason_code": "target_not_required",
            }
        )
        result = build_control_decision(
            intent,
            self.world_state,
            alignment,
            self.risk,
            source_step_id="step_2",
        )
        self.assertEqual(result["source_step_id"], "step_2")
        self.assertEqual(result["source_step_count"], 2)
        self.assertEqual(result["target_speed_kmh"], 18.0)

    def test_rejects_stale_risk_frame(self):
        intent = driving_intent("KEEP_LANE")
        alignment = semantic_alignment(intent, self.world_state["frame_id"])
        stale = copy.deepcopy(self.risk)
        stale["frame_id"] = "carla_stale"
        with self.assertRaisesRegex(ValueError, "frame_id mismatch"):
            build_control_decision(
                intent, self.world_state, alignment, stale
            )


if __name__ == "__main__":
    unittest.main()
