import json
import os
import tempfile
import unittest
from pathlib import Path

from control.decision_provider import JsonFileDecisionPolicy
from control.placeholder_following_policy import (
    PlaceholderFollowingPolicy,
    build_control_decision,
)


def write_json(path, document):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle)


class JsonFileDecisionPolicyTest(unittest.TestCase):
    def test_reads_checked_in_scene_understanding_contract(self):
        root = Path(__file__).resolve().parents[3]
        document = json.loads(
            (root / "scene_understanding" / "schemas" / "examples" / "control_decision.example.json")
            .read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "decision.json")
            write_json(path, document)
            decision = JsonFileDecisionPolicy(path, max_age_frames=3).decide({
                "simulation_frame": 125,
            })
        self.assertEqual(decision["action"], "decelerate")
        self.assertEqual(decision["decision_frame_id"], "carla_000123")
        self.assertEqual(decision["decision_age_frames"], 2)

    def test_reads_scene_understanding_control_decision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "decision.json")
            write_json(path, {
                "action": "lane_change_left",
                "target_speed_kmh": 32.0,
                "target_lane": "left",
                "target_location": None,
                "emergency": False,
                "reason": "external_policy",
                "request_id": "decision-001",
                "parse_status": "VALID",
                "parse_confidence": 0.96,
                "source_step_id": "step_2",
                "source_step_action": "CHANGE_LANE",
                "source_step_count": 3,
            })
            decision = JsonFileDecisionPolicy(path).decide({})
        self.assertEqual(decision["action"], "lane_change_left")
        self.assertEqual(decision["target_speed_kmh"], 32.0)
        self.assertEqual(decision["source_step_id"], "step_2")

    def test_malformed_external_document_stops_safely(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "decision.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("not-json")
            decision = JsonFileDecisionPolicy(path).decide({})
        self.assertEqual(decision["action"], "stop")
        self.assertEqual(decision["target_speed_kmh"], 0.0)
        self.assertIn("external_decision_unavailable", decision["reason"])

    def test_accepts_utf8_bom_external_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "decision.json")
            with open(path, "w", encoding="utf-8-sig") as handle:
                json.dump({"action": "keep_lane", "target_speed_kmh": 35.0}, handle)
            decision = JsonFileDecisionPolicy(path).decide({})
        self.assertEqual(decision["action"], "keep_lane")
        self.assertEqual(decision["target_speed_kmh"], 35.0)

    def test_rejects_stale_or_future_external_decision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "decision.json")
            document = {
                "action": "keep_lane",
                "target_speed_kmh": 30.0,
                "frame_id": "carla_000100",
            }
            write_json(path, document)
            policy = JsonFileDecisionPolicy(path, max_age_frames=3)
            stale = policy.decide({"simulation_frame": 104})
            self.assertEqual(stale["action"], "stop")
            self.assertEqual(stale["reason"], "external_decision_stale")

            document["frame_id"] = "carla_000106"
            write_json(path, document)
            future = policy.decide({"simulation_frame": 104})
            self.assertEqual(future["action"], "stop")
            self.assertEqual(future["reason"], "external_decision_future_frame")

    def test_placeholder_policy_brakes_for_closing_front_vehicle(self):
        decision = build_control_decision({
            "ego": {"speed(km/h)": 25.0},
            "vehicles": [{
                "id": 42,
                "distance": 20.0,
                "speed_kmh": 10.0,
                "relative_position": {"x": 20.0, "y": 0.0},
            }],
        }, "carla_123")
        self.assertEqual(decision["action"], "emergency_brake")
        self.assertTrue(decision["emergency"])
        self.assertEqual(decision["risk_level"], "high")

    def test_placeholder_policy_latches_emergency_state(self):
        policy = PlaceholderFollowingPolicy()
        policy.decide({
            "ego": {"speed(km/h)": 25.0},
            "vehicles": [{
                "distance": 20.0,
                "speed_kmh": 10.0,
                "relative_position": {"x": 20.0, "y": 0.0},
            }],
        }, "carla_123")
        decision = policy.decide({"ego": {"speed(km/h)": 0.0}}, "carla_124")
        self.assertEqual(decision["action"], "emergency_brake")
        self.assertEqual(decision["reason"], "placeholder_emergency_brake_latched")
