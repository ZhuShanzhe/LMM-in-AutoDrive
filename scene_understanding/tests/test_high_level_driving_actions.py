from __future__ import annotations

import unittest

from scene_understanding.src.high_level_driving_actions import (
    evaluate_step_decision,
    map_step_action,
    validate_risk_assessment,
)


def risk(*, recommended_action: str = "monitor") -> dict:
    return {
        "risk_level": "low",
        "recommended_action": recommended_action,
        "reason_codes": ["test_risk"],
        "lane_change": {
            "left": {"is_safe": True, "reason_codes": ["target_lane_clear"]},
            "right": {"is_safe": True, "reason_codes": ["target_lane_clear"]},
        },
    }


class HighLevelDrivingActionsTests(unittest.TestCase):
    def test_set_speed_uses_controller_compatible_setpoint(self):
        action, target_speed, lane, location = map_step_action(
            {"step_id": "step_1", "action": "SET_SPEED", "parameters": {"target_speed_mps": 16.667}},
            36.0,
        )
        self.assertEqual((action, lane, location), ("keep_lane", None, None))
        self.assertAlmostEqual(target_speed, 60.0012)

    def test_emergency_recommendation_overrides_stop(self):
        result = evaluate_step_decision(
            {"step_id": "step_1", "action": "STOP", "parameters": {}},
            {"alignment_required": False, "alignment_success": None},
            risk(recommended_action="emergency_brake"),
            36.0,
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["action"], "emergency_brake")
        self.assertEqual(result["target_speed_kmh"], 0.0)

    def test_unsafe_lane_change_obeys_safe_stop_policy(self):
        assessment = risk()
        assessment["lane_change"]["left"] = {
            "is_safe": False,
            "reason_codes": ["target_lane_rear_gap_too_small"],
        }
        result = evaluate_step_decision(
            {
                "step_id": "step_1",
                "action": "CHANGE_LANE",
                "parameters": {"direction": "LEFT"},
                "on_blocked": "SAFE_STOP",
            },
            {"alignment_required": False, "alignment_success": None},
            assessment,
            36.0,
        )
        self.assertEqual(result["action"], "stop")
        self.assertEqual(result["blocked_reason_codes"], ["target_lane_rear_gap_too_small"])

    def test_rejects_incomplete_risk_assessment(self):
        with self.assertRaisesRegex(ValueError, "lane_change"):
            validate_risk_assessment({"risk_level": "low", "recommended_action": "monitor", "reason_codes": []})
