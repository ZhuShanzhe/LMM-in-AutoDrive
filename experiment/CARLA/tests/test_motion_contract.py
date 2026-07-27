import unittest

from control.motion_contract import apply_motion_constraints


class MotionContractTests(unittest.TestCase):
    def test_geometric_action_uses_scenario_cruise_from_rest(self):
        result = apply_motion_constraints(
            {"action": "lane_change_left", "target_speed_kmh": 0.0},
            {"default_speed_kmh": 35.0},
        )
        self.assertEqual(result["target_speed_kmh"], 35.0)

    def test_safety_actions_never_receive_cruise_speed(self):
        for action in ("stop", "emergency_brake", "decelerate"):
            result = apply_motion_constraints(
                {"action": action, "target_speed_kmh": 0.0},
                {"default_speed_kmh": 35.0},
            )
            self.assertEqual(result["target_speed_kmh"], 0.0)

    def test_explicit_speed_is_preserved(self):
        result = apply_motion_constraints(
            {"action": "turn_right", "target_speed_kmh": 20.0},
            {"default_speed_kmh": 35.0},
        )
        self.assertEqual(result["target_speed_kmh"], 20.0)


if __name__ == "__main__":
    unittest.main()
