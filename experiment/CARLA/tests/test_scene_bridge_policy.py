import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from control.scene_bridge_policy import SceneBridgeDecisionPolicy
from control.scene_understanding_json_policy import SceneUnderstandingJsonPolicy
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

    def test_compiles_speed_plan_and_applies_speed_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            intent_path = Path(directory) / "intent.json"
            intent_path.write_text(json.dumps(self.intent), encoding="utf-8")
            policy = SceneBridgeDecisionPolicy(str(intent_path), directory)
            policy.set_context({"route_target": {"x": 30.0, "y": 5.0, "z": 0.0}})
            policy.set_scene_world_state(self.world_state)

            decision = policy.decide({})
            self.assertEqual(decision["action"], "keep_lane")
            self.assertEqual(decision["target_speed_kmh"], 36.0)
            self.assertEqual(
                decision["target_location"], {"x": 30.0, "y": 5.0, "z": 0.0}
            )
            self.assertEqual(policy.telemetry()["active_step_id"], "step_1")

    def test_missing_scene_snapshot_stops_safely(self):
        policy = SceneBridgeDecisionPolicy("does-not-exist.json")
        decision = policy.decide({})
        self.assertEqual(decision["action"], "stop")
        self.assertEqual(decision["reason"], "scene_world_state_unavailable")

    def test_carla_boundary_reads_persisted_scene_control_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            intent_path = Path(directory) / "intent.json"
            intent_path.write_text(json.dumps(self.intent), encoding="utf-8")
            policy = SceneUnderstandingJsonPolicy(
                driving_intent_path=str(intent_path),
                output_dir=directory,
                max_age_frames=0,
            )
            policy.set_scene_world_state(self.world_state)
            decision = policy.decide({
                "frame_id": self.world_state["frame_id"],
                "simulation_frame": self.world_state["simulation_frame"],
            })
            self.assertTrue((Path(directory) / "control_decision.json").is_file())
            self.assertEqual(decision["action"], "keep_lane")
            self.assertEqual(policy.telemetry()["consumer"]["status"], "accepted")

    def test_final_rule_adjustment_is_persisted_before_carla_consumes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            intent_path = Path(directory) / "intent.json"
            intent_path.write_text(json.dumps(self.intent), encoding="utf-8")
            policy = SceneUnderstandingJsonPolicy(
                driving_intent_path=str(intent_path),
                output_dir=directory,
                max_age_frames=0,
            )
            policy.set_scene_world_state(self.world_state)
            frame = {
                "frame_id": self.world_state["frame_id"],
                "simulation_frame": self.world_state["simulation_frame"],
            }
            decision = policy.decide(frame)
            decision["target_speed_kmh"] = 42.0
            decision["reason"] = "rule_stabilized_speed"

            final = policy.persist_final_decision(decision, frame)
            persisted = json.loads((Path(directory) / "control_decision.json").read_text())

            self.assertEqual(final["target_speed_kmh"], 42.0)
            self.assertEqual(final["reason"], "rule_stabilized_speed")
            self.assertEqual(persisted["target_speed_kmh"], 42.0)
            self.assertEqual(persisted["reason"], "rule_stabilized_speed")

    def test_policy_has_no_direct_world_state_execution_guard(self):
        policy = SceneBridgeDecisionPolicy("unused.json")
        self.assertFalse(hasattr(policy, "_apply_execution_safety_guard"))

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

    def test_scheduled_turn_binds_route_planner_target_before_alignment(self):
        driving_intent = {
            "intent": {
                "steps": [{
                    "step_id": "step_1",
                    "action": "TURN",
                    "parameters": {"direction": "RIGHT"},
                    "target": {"text": "the intersection", "type": "LOCATION"},
                }]
            }
        }

        result = ScheduledSceneBridgePolicy._bind_planned_turn_target(
            driving_intent, {"x": 12.0, "y": -4.0, "z": 0.5}
        )

        step = result["intent"]["steps"][0]
        self.assertNotIn("target", step)
        self.assertNotIn("target_location", step["parameters"])
        self.assertIn("target", driving_intent["intent"]["steps"][0])

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

    def test_risk_recovery_holds_normal_deceleration_for_short_clear_interval(self):
        policy = ScheduledSceneBridgePolicy(None)
        policy._active_command_id = "c02"
        scheduled = {"command_id": "c02", "target_speed_kmh": 50.0}
        risk = {
            "action": "decelerate",
            "reason": "risk_requires_deceleration",
            "target_speed_kmh": 35.0,
            "risk_level": "medium",
            "blocked_reason_codes": [],
        }
        clear = {
            "decision_status": "READY",
            "action": "keep_lane",
            "target_speed_kmh": 50.0,
            "risk_level": "low",
            "blocked_reason_codes": [],
        }

        policy._stabilize_risk_recovery(scheduled, risk)
        held = policy._stabilize_risk_recovery(scheduled, clear)

        self.assertEqual(held["action"], "decelerate")
        self.assertEqual(held["target_speed_kmh"], 35.0)
        self.assertIn("risk_recovery_hold", held["blocked_reason_codes"])

    def test_scheduled_bridge_preserves_trusted_route_target_provenance(self):
        class Schedule:
            def decide(self, _world_state):
                return {
                    "command_id": "c02",
                    "action": "keep_lane",
                    "target_speed_kmh": 50.0,
                    "target_location": {"x": 20.0, "y": 5.0, "z": 0.0},
                    "route_target_trusted": True,
                    "driving_intent": {"intent": {"steps": []}},
                }

        class Bridge:
            def set_context(self, _context):
                pass

            def set_scene_world_state(self, _world_state):
                pass

            def decide(self, _world_state):
                return {
                    "action": "keep_lane",
                    "target_speed_kmh": 50.0,
                    "target_location": {"x": 20.0, "y": 5.0, "z": 0.0},
                    "reason": "step_ready",
                }

            def persist_final_decision(self, decision, _world_state):
                return dict(decision)

        policy = ScheduledSceneBridgePolicy(Schedule())
        policy._active_command_id = "c02"
        policy._bridge = Bridge()

        result = policy.decide({})

        self.assertTrue(result["route_target_trusted"])

    def test_route_managed_turn_marks_persisted_target_as_trusted(self):
        class Schedule:
            def decide(self, _world_state):
                return {
                    "command_id": "turn",
                    "action": "turn_left",
                    "target_speed_kmh": 30.0,
                    "target_location": {"x": 20.0, "y": 5.0, "z": 0.0},
                    "route_target_trusted": False,
                    "driving_intent": {"intent": {"steps": []}},
                }

        class Bridge:
            def set_context(self, _context):
                pass

            def set_scene_world_state(self, _world_state):
                pass

            def decide(self, _world_state):
                return {
                    "action": "turn_left",
                    "target_speed_kmh": 30.0,
                    "target_location": {"x": 20.0, "y": 5.0, "z": 0.0},
                    "reason": "driving_intent_turn",
                }

            def persist_final_decision(self, decision, _world_state):
                return dict(decision)

        policy = ScheduledSceneBridgePolicy(Schedule())
        policy._active_command_id = "turn"
        policy._bridge = Bridge()
        policy._context = {"turn_uses_local_branch": False}

        result = policy.decide({})

        self.assertTrue(result["route_target_trusted"])

    def test_completed_route_turn_keeps_guidance_during_success_hold(self):
        class Schedule:
            def decide(self, _world_state):
                return {
                    "command_id": "turn",
                    "action": "turn_left",
                    "target_speed_kmh": 30.0,
                    "target_location": {"x": 30.0, "y": 10.0, "z": 0.0},
                    "route_target_trusted": False,
                    "driving_intent": {"intent": {"steps": []}},
                }

        class Bridge:
            def set_context(self, _context):
                pass

            def set_scene_world_state(self, _world_state):
                pass

            def decide(self, _world_state):
                return {
                    "action": "turn_left",
                    "target_speed_kmh": 24.0,
                    "target_location": {"x": 30.0, "y": 10.0, "z": 0.0},
                    "reason": "plan_completed",
                    "emergency": False,
                }

            def persist_final_decision(self, decision, _world_state):
                return dict(decision)

        policy = ScheduledSceneBridgePolicy(Schedule())
        policy._active_command_id = "turn"
        policy._bridge = Bridge()
        policy._context = {"turn_uses_local_branch": False}

        result = policy.decide({})

        self.assertEqual(result["action"], "keep_lane")
        self.assertEqual(
            result["target_location"], {"x": 30.0, "y": 10.0, "z": 0.0}
        )
        self.assertTrue(result["route_target_trusted"])

    def test_completed_speed_plan_restores_cruise_after_risk_clears(self):
        class Schedule:
            def decide(self, _world_state):
                return {
                    "command_id": "speed",
                    "action": "keep_lane",
                    "target_speed_kmh": 60.0,
                    "target_location": {"x": 30.0, "y": 0.0, "z": 0.0},
                    "route_target_trusted": True,
                    "driving_intent": {"intent": {"steps": []}},
                }

        class Bridge:
            def set_context(self, _context):
                pass

            def set_scene_world_state(self, _world_state):
                pass

            def decide(self, _world_state):
                return {
                    "decision_status": "READY",
                    "action": "keep_lane",
                    "target_speed_kmh": 2.0,
                    "target_location": None,
                    "reason": "plan_completed",
                    "risk_level": "none",
                    "emergency": False,
                }

            def persist_final_decision(self, decision, _world_state):
                return dict(decision)

        policy = ScheduledSceneBridgePolicy(Schedule())
        policy._active_command_id = "speed"
        policy._bridge = Bridge()

        result = policy.decide({})

        self.assertEqual(result["action"], "keep_lane")
        self.assertEqual(result["target_speed_kmh"], 60.0)
        self.assertEqual(
            result["target_location"], {"x": 30.0, "y": 0.0, "z": 0.0}
        )

    def test_emergency_clears_risk_recovery_hold_immediately(self):
        policy = ScheduledSceneBridgePolicy(None)
        policy._active_command_id = "c02"
        scheduled = {"command_id": "c02", "target_speed_kmh": 50.0}
        policy._stabilize_risk_recovery(scheduled, {
            "action": "decelerate",
            "reason": "risk_requires_deceleration",
            "target_speed_kmh": 35.0,
            "risk_level": "medium",
            "blocked_reason_codes": [],
        })
        emergency = {"action": "emergency_brake", "emergency": True}

        self.assertIs(policy._stabilize_risk_recovery(scheduled, emergency), emergency)
        self.assertIsNone(policy._risk_recovery_command_id)

    def test_continuous_safety_monitor_is_not_exposed_or_marked_complete(self):
        class Schedule:
            def __init__(self):
                self.completed = []

            def decide(self, _world_state):
                return {
                    "command_id": "speed__continuous_cruise",
                    "action": "keep_lane",
                    "target_speed_kmh": 45.0,
                    "driving_intent": {"intent": {"steps": []}},
                    "continuous_safety_monitor": True,
                    "command_phase": "WAITING",
                }

            def set_context(self, _context):
                pass

            def mark_completed(self, command_id):
                self.completed.append(command_id)

        class Bridge:
            def set_context(self, _context):
                pass

            def set_scene_world_state(self, _world_state):
                pass

            def decide(self, _world_state):
                return {
                    "decision_status": "READY",
                    "action": "decelerate",
                    "target_speed_kmh": 30.0,
                    "reason": "risk_requires_deceleration",
                    "risk_level": "medium",
                }

            def persist_final_decision(self, decision, _world_state):
                return dict(decision)

            def report_execution(self, _world_state, _intent, _controller):
                return {"outcome": "COMPLETED"}

        schedule = Schedule()
        policy = ScheduledSceneBridgePolicy(schedule)
        policy._active_command_id = "speed__continuous_cruise"
        policy._bridge = Bridge()

        decision = policy.decide({})
        policy.report_execution({}, decision)

        self.assertIsNone(decision["command_id"])
        self.assertEqual(decision["command_phase"], "WAITING")
        self.assertEqual(decision["action"], "decelerate")
        self.assertEqual(schedule.completed, [])

    def test_settling_lane_change_skips_only_the_second_permission_gate(self):
        policy = ScheduledSceneBridgePolicy.__new__(ScheduledSceneBridgePolicy)
        policy._active_command_id = "c04"
        policy._lane_change_settling_command_id = "c04"
        scheduled = {"action": "lane_change_right", "target_speed_kmh": 50.0}
        decision = {
            "decision_status": "BLOCKED",
            "action": "decelerate",
            "target_speed_kmh": 24.0,
            "target_lane": None,
            "target_location": None,
            "emergency": False,
            "reason": "lane_change_right_blocked_wait_for_safe",
            "blocked_reason_codes": ["lane_change_not_permitted"],
        }

        result = policy._preserve_lane_change_during_settle(scheduled, decision)

        self.assertEqual(result["decision_status"], "READY")
        self.assertEqual(result["action"], "lane_change_right")
        self.assertEqual(result["target_speed_kmh"], 50.0)
        self.assertIn("lane_change_settling", result["blocked_reason_codes"])

    def test_lane_change_block_is_not_misreported_as_settling_before_execution(self):
        policy = ScheduledSceneBridgePolicy.__new__(ScheduledSceneBridgePolicy)
        policy._active_command_id = "c04"
        policy._lane_change_settling_command_id = None
        scheduled = {
            "command_id": "c04",
            "action": "lane_change_right",
            "target_speed_kmh": 50.0,
        }
        decision = {
            "decision_status": "BLOCKED",
            "action": "decelerate",
            "target_speed_kmh": 24.0,
            "target_lane": None,
            "target_location": None,
            "emergency": False,
            "risk_level": "none",
            "reason": "lane_change_right_blocked_wait_for_safe",
            "blocked_reason_codes": ["lane_change_not_permitted"],
        }

        result = policy._preserve_lane_change_during_settle(scheduled, decision)

        self.assertIs(result, decision)


if __name__ == "__main__":
    unittest.main()
