import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from control.scene_bridge_policy import SceneBridgeDecisionPolicy
from control.scheduled_scene_bridge_policy import ScheduledSceneBridgePolicy


ROOT = Path(__file__).resolve().parents[3]


class SceneBridgeDecisionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.intent = json.loads(
            (ROOT / "structured_command_parser" / "examples" / "basic_set_speed.json").read_text(
                encoding="utf-8"
            )
        )
        self.world_state = json.loads(
            (ROOT / "scene_understanding" / "schemas" / "examples" / "world_state.example.json").read_text(
                encoding="utf-8"
            )
        )
        self.world_state["objects"] = []

    def test_compiles_speed_step_and_applies_speed_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            intent_path = Path(directory) / "intent.json"
            intent_path.write_text(json.dumps(self.intent), encoding="utf-8")
            policy = SceneBridgeDecisionPolicy(str(intent_path), directory)
            policy.set_context({"route_target": {"x": 30.0, "y": 5.0, "z": 0.0}})
            policy.set_scene_world_state(self.world_state)

            decision = policy.decide({})
            self.assertEqual(decision["action"], "keep_lane")
            self.assertEqual(decision["target_speed_kmh"], 60.0012)
            self.assertIsNone(decision["target_location"])
            self.assertEqual(policy.telemetry()["active_step_id"], "step_2")

            completed_world = deepcopy(self.world_state)
            completed_world["frame_id"] = "carla_000124"
            completed_world["simulation_frame"] = 124
            completed_world["ego"]["speed_mps"] = 16.667
            feedback = policy.report_execution(completed_world, decision)
            self.assertEqual(feedback["outcome"], "COMPLETED")

            policy.set_scene_world_state(completed_world)
            final_decision = policy.decide({})
            self.assertEqual(final_decision["reason"], "plan_completed")

    def test_missing_scene_snapshot_stops_safely(self):
        policy = SceneBridgeDecisionPolicy("does-not-exist.json")
        decision = policy.decide({})
        self.assertEqual(decision["action"], "stop")
        self.assertEqual(decision["reason"], "scene_world_state_unavailable")

    def test_execution_guard_escalates_closing_same_lane_vehicle(self):
        policy = SceneBridgeDecisionPolicy("unused.json")
        policy.set_scene_world_state({
            "ego": {"speed_mps": 13.0},
            "objects": [{
                "object_id": "carla_actor_9",
                "lane_relation": "ego_lane",
                "distance_m": 18.0,
                "closing_speed_mps": 13.0,
                "relative_position_ego_m": {"longitudinal": 18.0},
            }],
        })
        decision = policy._apply_execution_safety_guard({
            "schema_version": "1.0.0",
            "request_id": "test-1",
            "frame_id": "carla_00000001",
            "decision_status": "BLOCKED",
            "action": "decelerate",
            "target_speed_kmh": 46.8,
            "target_lane": None,
            "target_location": None,
            "emergency": False,
            "reason": "risk_requires_deceleration",
            "parse_status": "VALID",
            "parse_confidence": 0.9,
            "source_step_id": "step_1",
            "source_step_action": "SET_SPEED",
            "source_step_count": 1,
            "matched_entity_id": None,
            "risk_level": "high",
            "risk_reason_codes": ["distance_10_to_25m"],
            "blocked_reason_codes": ["risk_requires_deceleration"],
        })
        self.assertEqual(decision["action"], "emergency_brake")
        self.assertTrue(decision["emergency"])
        self.assertIn("carla_stopping_distance_guard", decision["blocked_reason_codes"])

    def test_object_present_trigger_cruises_until_target_is_visible(self):
        policy = SceneBridgeDecisionPolicy("unused.json")
        policy._driving_intent = {
            "intent": {"steps": [{
                "step_id": "step_1",
                "action": "ADJUST_SPEED",
                "trigger": {"type": "OBJECT_PRESENT"},
            }]},
        }
        policy._plan_state = {"active_step_id": "step_1"}
        policy.set_context({"default_speed_kmh": 45.0})
        decision = policy._await_object_trigger({
            "schema_version": "1.0.0",
            "request_id": "test-2",
            "frame_id": "carla_00000001",
            "decision_status": "BLOCKED",
            "action": "stop",
            "target_speed_kmh": 0.0,
            "target_lane": None,
            "target_location": None,
            "emergency": False,
            "reason": "no_matching_entity_safe_stop",
            "parse_status": "VALID",
            "parse_confidence": 0.9,
            "source_step_id": "step_1",
            "source_step_action": "ADJUST_SPEED",
            "source_step_count": 1,
            "matched_entity_id": None,
            "risk_level": "none",
            "risk_reason_codes": [],
            "blocked_reason_codes": ["no_matching_entity", "safe_stop"],
        }, policy._driving_intent)
        self.assertEqual(decision["action"], "keep_lane")
        self.assertEqual(decision["target_speed_kmh"], 45.0)
        self.assertEqual(decision["reason"], "awaiting_object_trigger")

    def test_target_cleared_uses_completed_pedestrian_event(self):
        policy = SceneBridgeDecisionPolicy("unused.json")
        policy.set_context({"completed_event_scenarios": ["pedestrian_crossing"]})
        completed, reason = policy._scenario_target_cleared({
            "target": {"type": "PEDESTRIAN"},
        })
        self.assertTrue(completed)
        self.assertEqual(reason, "scenario_target_cleared")

    def test_speed_limit_keeps_an_active_lane_change_lateral_target(self):
        scheduled = {
            "action": "lane_change_right",
            "target_speed_kmh": 45.0,
            "target_lane": "right",
        }
        decision = {
            "decision_status": "BLOCKED",
            "action": "decelerate",
            "target_speed_kmh": 30.0,
            "target_lane": None,
            "target_location": None,
            "emergency": False,
            "reason": "risk_requires_deceleration",
            "blocked_reason_codes": ["risk_requires_deceleration"],
        }

        result = ScheduledSceneBridgePolicy._preserve_lane_change_for_speed_limit(
            scheduled, decision
        )

        self.assertEqual(result["decision_status"], "READY")
        self.assertEqual(result["action"], "lane_change_right")
        self.assertEqual(result["target_lane"], "right")
        self.assertEqual(result["target_speed_kmh"], 30.0)

    def test_emergency_does_not_preserve_lane_change(self):
        scheduled = {"action": "lane_change_left", "target_speed_kmh": 45.0}
        decision = {
            "action": "decelerate",
            "target_speed_kmh": 0.0,
            "emergency": True,
            "reason": "risk_requires_deceleration",
        }
        self.assertIs(
            ScheduledSceneBridgePolicy._preserve_lane_change_for_speed_limit(
                scheduled, decision
            ),
            decision,
        )


if __name__ == "__main__":
    unittest.main()
