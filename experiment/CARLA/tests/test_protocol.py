import sys
import unittest
from pathlib import Path

from control.protocol import normalize_intent


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from structured_command_parser.src.rule_parser import RuleIntentParser


def driving_intent(action, parameters=None, status="VALID", urgency="NORMAL"):
    return {
        "schema_version": "1.1.0",
        "request_id": "test-001",
        "input": {
            "modality": "TEXT",
            "language": "zh-CN",
            "raw_text": "test",
            "normalized_text": "test",
        },
        "intent": {
            "category": "BASIC_CONTROL",
            "urgency": urgency,
            "steps": [
                {
                    "step_id": "step_1",
                    "action": action,
                    "parameters": parameters or {},
                    "trigger": {"type": "IMMEDIATE"},
                    "depends_on": [],
                    "preconditions": [],
                    "on_blocked": "SAFE_STOP",
                }
            ],
            "constraints": {
                "safety_first": True,
                "obey_traffic_rules": True,
                "driving_style": "NORMAL",
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


def control_decision(action, target_speed_kmh, emergency=False):
    """Representative scene_understanding ControlDecision contract."""
    return {
        "schema_version": "1.0.0",
        "request_id": "decision-001",
        "frame_id": "carla_000123",
        "decision_status": "READY",
        "action": action,
        "target_speed_kmh": target_speed_kmh,
        "target_lane": "left" if action == "lane_change_left" else None,
        "target_location": None,
        "emergency": emergency,
        "reason": "scene_understanding_test",
        "parse_status": "VALID",
        "parse_confidence": 0.96,
        "source_step_id": "step_1",
        "source_step_action": "CHANGE_LANE",
        "source_step_count": 3,
        "matched_entity_id": "carla_actor_42",
        "risk_level": "medium",
        "risk_reason_codes": ["distance_10_to_25m"],
        "blocked_reason_codes": [],
    }


class ProtocolDrivingIntentTest(unittest.TestCase):
    def test_colloquial_chinese_speed_reaches_control_protocol(self):
        parsed = RuleIntentParser().parse(
            "保持40公里速度行驶",
            request_id="spoken-speed",
        )
        self.assertIsNotNone(parsed)
        intent = normalize_intent(parsed)
        self.assertEqual(intent["action"], "keep_lane")
        self.assertAlmostEqual(
            intent["target_speed_kmh"],
            40.0,
            places=2,
        )

    def test_set_speed_maps_to_keep_lane_with_kmh(self):
        intent = normalize_intent(
            driving_intent("SET_SPEED", {"target_speed_mps": 16.667})
        )
        self.assertEqual(intent["action"], "keep_lane")
        self.assertAlmostEqual(intent["target_speed_kmh"], 60.0012)
        self.assertEqual(intent["request_id"], "test-001")

    def test_change_lane_direction_maps_to_flat_action(self):
        intent = normalize_intent(
            driving_intent("CHANGE_LANE", {"direction": "LEFT"})
        )
        self.assertEqual(intent["action"], "lane_change_left")
        self.assertEqual(intent["source_step_action"], "CHANGE_LANE")

    def test_emergency_brake_sets_emergency_and_zero_speed(self):
        intent = normalize_intent(
            driving_intent("EMERGENCY_BRAKE", urgency="EMERGENCY")
        )
        self.assertEqual(intent["action"], "emergency_brake")
        self.assertTrue(intent["emergency"])
        self.assertEqual(intent["target_speed_kmh"], 0.0)

    def test_non_valid_parse_status_falls_back_to_stop(self):
        intent = normalize_intent(driving_intent("KEEP_LANE", status="INVALID"))
        self.assertEqual(intent["action"], "stop")
        self.assertEqual(intent["target_speed_kmh"], 0.0)
        self.assertEqual(intent["parse_status"], "INVALID")

    def test_unimplemented_v11_action_stops_with_a_clear_reason(self):
        intent = normalize_intent(driving_intent("FOLLOW"))
        self.assertEqual(intent["action"], "stop")
        self.assertEqual(intent["reason"], "unsupported_parser_action_follow")

    def test_scene_understanding_control_decision_is_accepted(self):
        intent = normalize_intent(control_decision("lane_change_left", 36.0))
        self.assertEqual(intent["action"], "lane_change_left")
        self.assertEqual(intent["target_speed_kmh"], 36.0)
        self.assertEqual(intent["request_id"], "decision-001")
        self.assertEqual(intent["source_step_id"], "step_1")

    def test_scene_understanding_emergency_decision_is_preserved(self):
        intent = normalize_intent(
            control_decision("emergency_brake", 0.0, emergency=True)
        )
        self.assertEqual(intent["action"], "emergency_brake")
        self.assertTrue(intent["emergency"])
        self.assertEqual(intent["target_speed_kmh"], 0.0)


if __name__ == "__main__":
    unittest.main()
