from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


CARLA_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CARLA_DIR.parents[1]
for path in (CARLA_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scene3_vla_controller import (
    COMMAND_PROFILES,
    active_text_command,
    apply_scene3_liveness_gate,
    assess_scene3_risk,
    build_canonical_decision,
    waypoint_lane_relation,
)
from scene_understanding.src.control_decision import validate_control_decision


class Scene3VlaControllerTests(unittest.TestCase):
    def test_waypoint_lane_identity_beats_curved_road_lateral_threshold(self):
        ego = SimpleNamespace(road_id=34, section_id=0, lane_id=-1)
        cone = SimpleNamespace(road_id=34, section_id=0, lane_id=-2)
        self.assertEqual(
            waypoint_lane_relation(ego, cone, 1.375),
            "right_adjacent_lane",
        )

    def test_text_schedule_selects_latest_trigger(self):
        commands = [
            {"id": "first", "trigger_progress_m": 10.0, "text": "a"},
            {"id": "second", "trigger_progress_m": 20.0, "text": "b"},
        ]
        self.assertEqual(active_text_command(commands, 5.0)["id"], "scene3_cruise")
        self.assertEqual(active_text_command(commands, 15.0)["id"], "first")
        self.assertEqual(active_text_command(commands, 20.0)["id"], "second")

    def test_text_schedule_expires_event_command_to_cruise(self):
        commands = [
            {
                "id": "temporary_hazard",
                "trigger_progress_m": 10.0,
                "end_progress_m": 20.0,
                "text": "slow down",
            }
        ]
        self.assertEqual(
            active_text_command(commands, 19.9)["id"],
            "temporary_hazard",
        )
        self.assertEqual(
            active_text_command(commands, 20.0)["id"],
            "scene3_cruise",
        )

    def test_text_schedule_falls_back_to_overlapping_active_command(self):
        commands = [
            {
                "id": "work_zone",
                "trigger_progress_m": 10.0,
                "end_progress_m": 40.0,
                "text": "work zone",
            },
            {
                "id": "crossing_worker",
                "trigger_progress_m": 20.0,
                "end_progress_m": 30.0,
                "text": "worker",
            },
        ]
        self.assertEqual(
            active_text_command(commands, 25.0)["id"],
            "crossing_worker",
        )
        self.assertEqual(
            active_text_command(commands, 35.0)["id"],
            "work_zone",
        )

    def test_front_collision_preempts_text_envelope(self):
        world_state = {
            "ego": {"speed_mps": 10.0},
            "objects": [
                {
                    "entity_id": "vehicle-front",
                    "lane_relation": "ego_lane",
                    "relative_position_m": {"x": 8.0, "y": 0.0, "z": 0.0},
                    "relative_velocity_mps": {"x": -5.0, "y": 0.0},
                }
            ],
        }
        risk = assess_scene3_risk(world_state)
        self.assertEqual(risk["recommended_action"], "emergency_brake")
        decision = build_canonical_decision(
            command_id="scene3_resume_normal_driving",
            frame_id="carla_10",
            profile=COMMAND_PROFILES["scene3_resume_normal_driving"],
            parse_result={"status": "VALID", "confidence": 0.9},
            risk=risk,
        )
        self.assertEqual(decision["action"], "emergency_brake")
        self.assertEqual(decision["target_speed_kmh"], 0.0)
        self.assertEqual(validate_control_decision(decision), [])

    def test_safe_left_lane_produces_valid_lane_change_envelope(self):
        risk = assess_scene3_risk(
            {"ego": {"speed_mps": 5.0}, "objects": []}
        )
        decision = build_canonical_decision(
            command_id="scene3_blocked_lane_change_left",
            frame_id="carla_20",
            profile=COMMAND_PROFILES["scene3_blocked_lane_change_left"],
            parse_result={"status": "NEEDS_CLARIFICATION", "confidence": 0.8},
            risk=risk,
        )
        self.assertEqual(decision["action"], "lane_change_left")
        self.assertEqual(decision["target_lane"], "left")
        self.assertEqual(validate_control_decision(decision), [])

    def test_liveness_gate_rejects_unprompted_low_risk_stop(self):
        canonical = build_canonical_decision(
            command_id="scene3_cruise",
            frame_id="carla_30",
            profile=COMMAND_PROFILES["scene3_cruise"],
            parse_result={"status": "VALID", "confidence": 0.9},
            risk={
                "risk_level": "low",
                "recommended_action": "keep_lane",
                "reason_codes": [],
                "matched_entity_id": None,
                "lane_change": {"left": {"is_safe": True}},
            },
        )
        stopped = dict(canonical)
        stopped.update(
            {
                "action": "stop",
                "target_speed_kmh": 0.0,
                "reason": "vla_accepted_model",
            }
        )
        decision, override = apply_scene3_liveness_gate(
            stopped,
            canonical,
            {
                "risk_level": "low",
                "recommended_action": "keep_lane",
            },
        )
        self.assertEqual(override, "unprompted_stop")
        self.assertEqual(decision["action"], "keep_lane")


if __name__ == "__main__":
    unittest.main()
