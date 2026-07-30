import unittest

from control.structured_vla_scene_bridge_policy import StructuredVlaSceneBridgePolicy


class _RulePolicy:
    def __init__(self, scheduled):
        self.active_scheduled_intent = scheduled


def _decision(*, speed=5.0, reason="plan_completed"):
    return {
        "decision_status": "READY",
        "action": "keep_lane",
        "target_speed_kmh": speed,
        "emergency": False,
        "reason": reason,
    }


def _world(speed_kmh):
    return {"ego": {"speed_mps": speed_kmh / 3.6}}


class StructuredVlaCruiseEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.policy = StructuredVlaSceneBridgePolicy.__new__(
            StructuredVlaSceneBridgePolicy
        )
        self.policy.rule_policy = _RulePolicy({"target_speed_kmh": 35.0})

    def test_completed_speed_plan_recovers_the_scheduled_cruise_speed(self):
        result = self.policy._cruise_envelope(
            _decision(), _world(5.0), {"recommended_action": "keep_lane"}
        )
        self.assertEqual(result["action"], "accelerate")
        self.assertEqual(result["target_speed_kmh"], 35.0)
        self.assertEqual(result["reason"], "vla_cruise_below_speed_setpoint")

    def test_speed_deadband_holds_lane_without_action_chatter(self):
        result = self.policy._cruise_envelope(
            _decision(speed=35.0, reason="normal_cruise"),
            _world(34.0),
            {"recommended_action": "keep_lane"},
        )
        self.assertEqual(result["action"], "keep_lane")
        self.assertEqual(result["target_speed_kmh"], 35.0)
        self.assertEqual(result["reason"], "vla_cruise_speed_setpoint_held")

    def test_rule_risk_deceleration_remains_untouched(self):
        original = _decision()
        result = self.policy._cruise_envelope(
            original, _world(5.0), {"recommended_action": "decelerate"}
        )
        self.assertIs(result, original)


if __name__ == "__main__":
    unittest.main()
