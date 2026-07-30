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
        "schema_version": "1.0.0",
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


if __name__ == "__main__":
    unittest.main()
