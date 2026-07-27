from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scene_understanding.scripts.run_execution_feedback import process_current_frame
from scene_understanding.scripts.run_carla_decision_bridge import build_decision
from scene_understanding.src.execution_feedback import evaluate_execution_feedback
from scene_understanding.tests.test_control_decision import driving_intent, load_example


def active_plan(intent: dict, step_id: str) -> dict:
    states = []
    for step in intent["intent"]["steps"]:
        states.append(
            {
                "step_id": step["step_id"],
                "action": step["action"],
                "status": "ACTIVE" if step["step_id"] == step_id else "PENDING",
                "activation_frame_id": "carla_000123" if step["step_id"] == step_id else None,
                "terminal_frame_id": None,
                "reason_codes": [],
            }
        )
    return {
        "request_id": intent["request_id"],
        "plan_status": "ACTIVE",
        "active_step_id": step_id,
        "step_states": states,
    }


def decision(intent: dict, world: dict, step_id: str, **overrides) -> dict:
    value = {
        "request_id": intent["request_id"],
        "frame_id": world["frame_id"],
        "decision_status": "READY",
        "source_step_id": step_id,
    }
    value.update(overrides)
    return value


class ExecutionFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.world = load_example("world_state.example.json")
        self.world["objects"] = []
        self.world["timestamp_s"] = 6.15

    def target_object(self, *, category="vehicle", longitudinal=20.0, lateral=0.0, lane_relation="ego_lane"):
        source = load_example("world_state.example.json")["objects"][0]
        source["category"] = category
        source["relative_position_ego_m"] = {
            "longitudinal": longitudinal,
            "lateral": lateral,
            "vertical": 0.0,
        }
        source["lane_relation"] = lane_relation
        return source

    @staticmethod
    def next_frame(world: dict, frame: int) -> dict:
        value = deepcopy(world)
        value["frame_id"] = f"carla_{frame:06d}"
        value["simulation_frame"] = frame
        value["timestamp_s"] = 6.15 + (frame - 123) * 0.05
        return value

    def test_completes_target_speed_only_inside_tolerance(self):
        intent = driving_intent("SET_SPEED", {"target_speed_mps": 10.0})
        intent["intent"]["steps"][0]["completion"] = {"type": "TARGET_SPEED_REACHED"}
        self.world["ego"]["speed_mps"] = 10.3
        tracker, feedback = evaluate_execution_feedback(
            intent,
            active_plan(intent, "step_1"),
            decision(intent, self.world, "step_1"),
            self.world,
        )
        self.assertEqual(tracker["active_step_id"], "step_1")
        self.assertEqual(feedback["outcome"], "COMPLETED")
        self.assertEqual(feedback["reason_codes"], ["target_speed_reached"])

    def test_does_not_complete_speed_outside_tolerance(self):
        intent = driving_intent("SET_SPEED", {"target_speed_mps": 10.0})
        intent["intent"]["steps"][0]["completion"] = {"type": "TARGET_SPEED_REACHED"}
        self.world["ego"]["speed_mps"] = 9.0
        _, feedback = evaluate_execution_feedback(
            intent,
            active_plan(intent, "step_1"),
            decision(intent, self.world, "step_1"),
            self.world,
        )
        self.assertIsNone(feedback)

    def test_collision_fails_active_step_even_without_completion_type(self):
        intent = driving_intent("KEEP_LANE", {})
        world = deepcopy(self.world)
        world["sensor_events"]["collisions"] = [{"other_actor_id": "2"}]
        _, feedback = evaluate_execution_feedback(
            intent,
            active_plan(intent, "step_1"),
            decision(intent, world, "step_1"),
            world,
        )
        self.assertEqual(feedback["outcome"], "FAILED")
        self.assertEqual(feedback["reason_codes"], ["collision_detected"])

    def test_junction_exit_requires_a_prior_junction_observation(self):
        intent = driving_intent("TURN", {"direction": "LEFT", "target_location": {"x": 1.0, "y": 2.0}})
        intent["intent"]["steps"][0]["completion"] = {"type": "JUNCTION_EXITED"}
        self.world["ego"]["is_junction"] = True
        tracker, feedback = evaluate_execution_feedback(
            intent,
            active_plan(intent, "step_1"),
            decision(intent, self.world, "step_1"),
            self.world,
        )
        self.assertTrue(tracker["junction_observed"])
        self.assertIsNone(feedback)
        next_world = deepcopy(self.world)
        next_world["frame_id"] = "carla_000124"
        next_world["simulation_frame"] = 124
        next_world["ego"]["is_junction"] = False
        _, feedback = evaluate_execution_feedback(
            intent,
            active_plan(intent, "step_1"),
            decision(intent, next_world, "step_1"),
            next_world,
            tracker=tracker,
        )
        self.assertEqual(feedback["reason_codes"], ["junction_exited"])

    def test_lane_change_requires_target_lane_to_remain_stable(self):
        intent = driving_intent("CHANGE_LANE", {"direction": "LEFT", "lane_count": 1})
        intent["intent"]["steps"][0]["completion"] = {"type": "LANE_CHANGE_COMPLETED"}
        tracker, feedback = evaluate_execution_feedback(
            intent,
            active_plan(intent, "step_1"),
            decision(intent, self.world, "step_1", target_lane="left"),
            self.world,
            required_stable_frames=3,
        )
        self.assertIsNone(feedback)
        for frame in (124, 125, 126):
            world = self.next_frame(self.world, frame)
            world["ego"]["lane_id"] = -2
            tracker, feedback = evaluate_execution_feedback(
                intent,
                active_plan(intent, "step_1"),
                decision(intent, world, "step_1", target_lane="left"),
                world,
                tracker=tracker,
                required_stable_frames=3,
            )
        self.assertEqual(feedback["reason_codes"], ["target_lane_reached", "target_lane_stable", "collision_free"])

    def test_targeted_lane_change_does_not_complete_after_target_disappears(self):
        intent = driving_intent(
            "CHANGE_LANE",
            {"direction": "LEFT", "lane_count": 1},
            target={"type": "SLOW_VEHICLE", "relation": "AHEAD"},
        )
        intent["intent"]["steps"][0]["completion"] = {"type": "LANE_CHANGE_COMPLETED"}
        observed = deepcopy(self.world)
        observed["objects"] = [self.target_object()]
        tracker, _ = evaluate_execution_feedback(
            intent,
            active_plan(intent, "step_1"),
            decision(intent, observed, "step_1", target_lane="left", matched_entity_id="carla_actor_42"),
            observed,
            required_stable_frames=1,
        )
        disappeared = self.next_frame(observed, 124)
        disappeared["ego"]["lane_id"] = -2
        disappeared["objects"] = []
        _, feedback = evaluate_execution_feedback(
            intent,
            active_plan(intent, "step_1"),
            decision(intent, disappeared, "step_1", target_lane="left", matched_entity_id="carla_actor_42"),
            disappeared,
            tracker=tracker,
            required_stable_frames=1,
        )
        self.assertIsNone(feedback)

    def test_pedestrian_target_cleared_requires_prior_crossing_and_deceleration(self):
        intent = driving_intent(
            "ADJUST_SPEED",
            {"change": "DECREASE"},
            target={"type": "PEDESTRIAN", "relation": "AHEAD_CROSSING"},
        )
        intent["intent"]["steps"][0]["completion"] = {"type": "TARGET_CLEARED"}
        crossing = deepcopy(self.world)
        crossing["ego"]["speed_mps"] = 8.0
        crossing["objects"] = [self.target_object(category="pedestrian", lateral=0.5, lane_relation="crossing_ego_path")]
        tracker, feedback = evaluate_execution_feedback(
            intent,
            active_plan(intent, "step_1"),
            decision(intent, crossing, "step_1", matched_entity_id="carla_actor_42"),
            crossing,
        )
        self.assertTrue(tracker["target_observed_crossing"])
        self.assertIsNone(feedback)
        cleared = self.next_frame(crossing, 124)
        cleared["ego"]["speed_mps"] = 4.9
        cleared["objects"][0]["relative_position_ego_m"]["lateral"] = -3.0
        cleared["objects"][0]["lane_relation"] = "roadside"
        _, feedback = evaluate_execution_feedback(
            intent,
            active_plan(intent, "step_1"),
            decision(
                intent,
                cleared,
                "step_1",
                matched_entity_id=None,
                decision_status="BLOCKED",
                action="stop",
                blocked_reason_codes=["no_matching_entity", "safe_stop"],
            ),
            cleared,
            tracker=tracker,
        )
        self.assertEqual(feedback["reason_codes"], ["pedestrian_crossing_cleared", "ego_speed_reduced", "collision_free"])

    def test_clearance_cannot_bypass_risk_blocked_decision(self):
        intent = driving_intent(
            "ADJUST_SPEED",
            {"change": "DECREASE"},
            target={"type": "PEDESTRIAN", "relation": "AHEAD_CROSSING"},
        )
        intent["intent"]["steps"][0]["completion"] = {"type": "TARGET_CLEARED"}
        crossing = deepcopy(self.world)
        crossing["ego"]["speed_mps"] = 8.0
        crossing["objects"] = [self.target_object(category="pedestrian", lateral=0.5, lane_relation="crossing_ego_path")]
        tracker, _ = evaluate_execution_feedback(
            intent,
            active_plan(intent, "step_1"),
            decision(intent, crossing, "step_1", matched_entity_id="carla_actor_42"),
            crossing,
        )
        cleared = self.next_frame(crossing, 124)
        cleared["ego"]["speed_mps"] = 4.9
        cleared["objects"][0]["relative_position_ego_m"]["lateral"] = -3.0
        cleared["objects"][0]["lane_relation"] = "roadside"
        _, feedback = evaluate_execution_feedback(
            intent,
            active_plan(intent, "step_1"),
            decision(
                intent,
                cleared,
                "step_1",
                decision_status="BLOCKED",
                action="emergency_brake",
                blocked_reason_codes=["risk_requires_emergency_brake"],
            ),
            cleared,
            tracker=tracker,
        )
        self.assertIsNone(feedback)

    def test_bridge_and_feedback_complete_cleared_pedestrian_without_terminal_block(self):
        intent = driving_intent(
            "ADJUST_SPEED",
            {"change": "DECREASE"},
            target={"type": "PEDESTRIAN", "relation": "AHEAD_CROSSING"},
            on_blocked="SAFE_STOP",
        )
        intent["intent"]["steps"][0]["completion"] = {"type": "TARGET_CLEARED"}
        crossing = deepcopy(self.world)
        crossing["ego"]["speed_mps"] = 8.0
        crossing["objects"] = [self.target_object(category="pedestrian", lateral=0.5, lane_relation="crossing_ego_path")]
        state, first_decision, _, _ = build_decision(intent, crossing)
        tracker, feedback = evaluate_execution_feedback(intent, state, first_decision, crossing)
        self.assertIsNone(feedback)
        self.assertEqual(first_decision["matched_entity_id"], "carla_actor_42")

        cleared = self.next_frame(crossing, 124)
        cleared["ego"]["speed_mps"] = 4.9
        cleared["objects"][0]["relative_position_ego_m"]["lateral"] = -3.0
        cleared["objects"][0]["lane_relation"] = "roadside"
        waiting_state, blocked_decision, _, _ = build_decision(
            intent,
            cleared,
            prior_plan_state=state,
        )
        self.assertEqual(waiting_state["plan_status"], "ACTIVE")
        self.assertIn(waiting_state["step_states"][0]["status"], {"ACTIVE", "WAITING"})
        self.assertIn(blocked_decision["decision_status"], {"READY", "BLOCKED"})
        tracker, feedback = evaluate_execution_feedback(
            intent,
            waiting_state,
            blocked_decision,
            cleared,
            tracker=tracker,
        )
        self.assertEqual(feedback["outcome"], "COMPLETED")

        final_world = self.next_frame(cleared, 125)
        final_state, _, _, _ = build_decision(
            intent,
            final_world,
            prior_plan_state=waiting_state,
            feedback=feedback,
        )
        self.assertEqual(final_state["plan_status"], "COMPLETED")

    def test_overtake_target_cleared_requires_observation_and_stable_rear_clearance(self):
        intent = driving_intent(
            "OVERTAKE",
            {},
            target={"type": "SLOW_VEHICLE", "relation": "AHEAD"},
        )
        intent["intent"]["steps"][0]["completion"] = {"type": "TARGET_CLEARED"}
        ahead = deepcopy(self.world)
        ahead["objects"] = [self.target_object(longitudinal=12.0)]
        tracker, feedback = evaluate_execution_feedback(
            intent,
            active_plan(intent, "step_1"),
            decision(intent, ahead, "step_1", matched_entity_id="carla_actor_42"),
            ahead,
            required_stable_frames=2,
        )
        self.assertTrue(tracker["target_observed_ahead"])
        self.assertIsNone(feedback)
        for frame in (124, 125):
            cleared = self.next_frame(ahead, frame)
            cleared["objects"][0]["relative_position_ego_m"]["longitudinal"] = -9.0
            tracker, feedback = evaluate_execution_feedback(
                intent,
                active_plan(intent, "step_1"),
                decision(intent, cleared, "step_1", matched_entity_id="carla_actor_42"),
                cleared,
                tracker=tracker,
                required_stable_frames=2,
            )
        self.assertEqual(feedback["reason_codes"], ["slow_vehicle_passed_with_rear_clearance", "clearance_stable", "collision_free"])

    def test_file_service_writes_terminal_feedback_and_tracker(self):
        intent = driving_intent("SET_SPEED", {"target_speed_mps": 10.0})
        intent["intent"]["steps"][0]["completion"] = {"type": "TARGET_SPEED_REACHED"}
        self.world["ego"]["speed_mps"] = 10.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "intent": root / "intent.json",
                "world": root / "world.json",
                "decision": root / "decision.json",
                "plan": root / "plan.json",
                "tracker": root / "tracker.json",
                "feedback": root / "feedback.json",
            }
            paths["intent"].write_text(json.dumps(intent), encoding="utf-8")
            paths["world"].write_text(json.dumps(self.world), encoding="utf-8")
            paths["decision"].write_text(
                json.dumps(decision(intent, self.world, "step_1")), encoding="utf-8"
            )
            paths["plan"].write_text(
                json.dumps(active_plan(intent, "step_1")), encoding="utf-8"
            )
            frame_id, feedback = process_current_frame(
                driving_intent_path=paths["intent"],
                world_state_path=paths["world"],
                control_decision_path=paths["decision"],
                plan_state_path=paths["plan"],
                tracker_path=paths["tracker"],
                feedback_output_path=paths["feedback"],
                speed_tolerance_mps=0.5,
                stop_speed_threshold_mps=0.2,
                target_tolerance_m=3.0,
                pedestrian_clearance_lateral_m=2.5,
                minimum_speed_reduction_mps=3.0,
                overtake_rear_clearance_m=8.0,
                required_stable_frames=5,
            )

            self.assertEqual(frame_id, self.world["frame_id"])
            self.assertEqual(feedback["outcome"], "COMPLETED")
            self.assertTrue(paths["tracker"].is_file())
            self.assertEqual(
                json.loads(paths["feedback"].read_text(encoding="utf-8"))["reason_codes"],
                ["target_speed_reached"],
            )


if __name__ == "__main__":
    unittest.main()
