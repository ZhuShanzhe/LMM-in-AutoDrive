from __future__ import annotations

import unittest

from control.safety_supervisor import (
    apply_adaptive_cruise_guard,
    apply_kinematic_conflict_guard,
    preserve_safe_lateral_maneuver,
)


def _decision() -> dict:
    return {
        "action": "keep_lane",
        "target_speed_kmh": 45.0,
        "target_lane": None,
        "target_location": None,
        "emergency": False,
        "reason": "fsm_target",
        "blocked_reason_codes": [],
    }


def _state(*objects: dict) -> dict:
    return {"objects": list(objects)}


class SafetySupervisorTests(unittest.TestCase):
    def test_safe_requested_lane_change_progresses_despite_medium_gap(self) -> None:
        decision = _decision()
        decision.update(
            {
                "action": "decelerate",
                "target_speed_kmh": 18.0,
                "reason": "risk_requires_deceleration",
            }
        )
        step = {
            "action": "CHANGE_LANE",
            "parameters": {"direction": "LEFT"},
        }
        risk = {
            "recommended_action": "decelerate",
            "lane_change": {"left": {"is_safe": True}},
            "object_assessments": [
                {
                    "relevant_to_ego_path": True,
                    "distance_is_safe": True,
                    "ttc_risk_level": "none",
                }
            ],
        }

        guarded, audit = preserve_safe_lateral_maneuver(
            decision,
            step,
            risk,
            speed_setpoint_kmh=40.0,
        )

        self.assertEqual(guarded["action"], "lane_change_left")
        self.assertEqual(guarded["target_speed_kmh"], 40.0)
        self.assertTrue(audit["override_applied"])

    def test_unsafe_gap_still_blocks_requested_lane_change(self) -> None:
        decision = _decision()
        decision.update(
            {
                "action": "decelerate",
                "reason": "risk_requires_deceleration",
            }
        )
        step = {
            "action": "CHANGE_LANE",
            "parameters": {"direction": "LEFT"},
        }
        risk = {
            "recommended_action": "decelerate",
            "lane_change": {"left": {"is_safe": True}},
            "object_assessments": [
                {
                    "relevant_to_ego_path": True,
                    "distance_is_safe": False,
                    "ttc_risk_level": "high",
                }
            ],
        }

        guarded, audit = preserve_safe_lateral_maneuver(
            decision,
            step,
            risk,
            speed_setpoint_kmh=40.0,
        )

        self.assertEqual(guarded["action"], "decelerate")
        self.assertFalse(audit["override_applied"])

    def test_safe_gap_medium_risk_uses_acc_instead_of_cumulative_stop(self) -> None:
        decision = _decision()
        decision["action"] = "decelerate"
        decision["target_speed_kmh"] = 0.0
        world_state = _state(
            {
                "object_id": "lead",
                "category": "vehicle",
                "speed_mps": 6.0,
            }
        )
        risk = {
            "object_assessments": [
                {
                    "object_id": "lead",
                    "relevant_to_ego_path": True,
                    "distance_is_safe": True,
                    "ttc_risk_level": "none",
                    "distance_m": 20.0,
                    "safe_distance_m": 10.0,
                }
            ]
        }

        guarded, audit = apply_adaptive_cruise_guard(
            decision,
            world_state,
            risk,
            speed_limit_kmh=45.0,
        )

        self.assertEqual(guarded["action"], "keep_lane")
        self.assertGreater(guarded["target_speed_kmh"], 21.6)
        self.assertTrue(audit["override_applied"])

    def test_cross_traffic_predicted_to_intersect_stops(self) -> None:
        risk = {
            "object_assessments": [
                {
                    "object_id": "cross-car",
                    "relevant_to_ego_path": True,
                }
            ]
        }
        guarded, audit = apply_kinematic_conflict_guard(
            _decision(),
            _state(
                {
                    "object_id": "cross-car",
                    "category": "vehicle",
                    "relative_position_ego_m": {
                        "longitudinal": 18.0,
                        "lateral": 12.0,
                    },
                    "relative_velocity_ego_mps": {
                        "longitudinal": -6.0,
                        "lateral": -4.0,
                    },
                }
            ),
            risk,
        )

        self.assertEqual(guarded["action"], "stop")
        self.assertTrue(audit["override_applied"])

    def test_medium_ttc_with_safe_distance_uses_controlled_deceleration(self) -> None:
        risk = {
            "object_assessments": [
                {
                    "object_id": "lead",
                    "relevant_to_ego_path": True,
                    "distance_is_safe": True,
                    "ttc_risk_level": "medium",
                }
            ]
        }
        guarded, audit = apply_kinematic_conflict_guard(
            _decision(),
            _state(
                {
                    "object_id": "lead",
                    "category": "vehicle",
                    "relative_position_ego_m": {
                        "longitudinal": 12.0,
                        "lateral": 0.0,
                    },
                    "relative_velocity_ego_mps": {
                        "longitudinal": -14.0,
                        "lateral": 0.0,
                    },
                }
            ),
            risk,
        )

        self.assertEqual(guarded["action"], "decelerate")
        self.assertEqual(guarded["target_speed_kmh"], 15.0)
        self.assertFalse(guarded["emergency"])
        self.assertEqual(
            audit["severity"],
            "CONTROLLED_DECELERATION",
        )

    def test_parallel_adjacent_vehicle_does_not_stop(self) -> None:
        guarded, audit = apply_kinematic_conflict_guard(
            _decision(),
            _state(
                {
                    "object_id": "adjacent-car",
                    "category": "vehicle",
                    "relative_position_ego_m": {
                        "longitudinal": 10.0,
                        "lateral": 3.6,
                    },
                    "relative_velocity_ego_mps": {
                        "longitudinal": -1.0,
                        "lateral": 0.0,
                    },
                }
            ),
        )

        self.assertEqual(guarded["action"], "keep_lane")
        self.assertFalse(audit["override_applied"])

    def test_rear_vehicle_does_not_trigger_braking(self) -> None:
        guarded, audit = apply_kinematic_conflict_guard(
            _decision(),
            _state(
                {
                    "object_id": "rear-car",
                    "category": "vehicle",
                    "relative_position_ego_m": {
                        "longitudinal": -2.0,
                        "lateral": 0.0,
                    },
                    "relative_velocity_ego_mps": {
                        "longitudinal": 8.0,
                        "lateral": 0.0,
                    },
                }
            ),
            {
                "object_assessments": [
                    {
                        "object_id": "rear-car",
                        "relevant_to_ego_path": True,
                    }
                ]
            },
        )

        self.assertEqual(guarded["action"], "keep_lane")
        self.assertFalse(audit["override_applied"])

    def test_irrelevant_predicted_intersection_does_not_stop(self) -> None:
        risk = {
            "object_assessments": [
                {
                    "object_id": "opposing-car",
                    "relevant_to_ego_path": False,
                }
            ]
        }
        guarded, audit = apply_kinematic_conflict_guard(
            _decision(),
            _state(
                {
                    "object_id": "opposing-car",
                    "category": "vehicle",
                    "relative_position_ego_m": {
                        "longitudinal": 18.0,
                        "lateral": 12.0,
                    },
                    "relative_velocity_ego_mps": {
                        "longitudinal": -6.0,
                        "lateral": -4.0,
                    },
                }
            ),
            risk,
        )

        self.assertEqual(guarded["action"], "keep_lane")
        self.assertFalse(audit["override_applied"])


if __name__ == "__main__":
    unittest.main()
