"""Generic supervisor/FSM tests replacing the former scene-specific gates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


CARLA_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CARLA_DIR.parents[1]
for path in (CARLA_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from control.generic_instruction_fsm import GenericInstructionFSM  # noqa: E402
from control.generic_temporal_risk_supervisor import (  # noqa: E402
    GenericTemporalRiskSupervisor,
    TemporalRiskSupervisorConfig,
)
from universal_vla_controller import build_sensor_policy_state  # noqa: E402


class GenericControllerTests(unittest.TestCase):
    def test_sensor_policy_state_never_enumerates_carla_actors(self):
        world = mock.Mock()
        world.get_snapshot.return_value = SimpleNamespace(
            timestamp=SimpleNamespace(elapsed_seconds=12.5)
        )
        world.get_actors.side_effect = AssertionError("actor truth accessed")
        world.get_map.return_value.get_waypoint.return_value = SimpleNamespace(
            is_junction=False
        )
        ego = mock.Mock()
        ego.get_velocity.return_value = SimpleNamespace(x=3.0, y=4.0, z=0.0)
        ego.get_acceleration.return_value = SimpleNamespace(x=0.1, y=0.2, z=0.0)
        ego.get_angular_velocity.return_value = SimpleNamespace(z=2.0)
        ego.get_speed_limit.return_value = 60.0
        ego.get_control.return_value = SimpleNamespace(
            steer=0.1, throttle=0.2, brake=0.0
        )
        ego.get_location.return_value = SimpleNamespace(x=0.0, y=0.0, z=0.0)

        state = build_sensor_policy_state(world, ego, frame_id="carla_42")

        self.assertEqual(state["objects"], [])
        self.assertEqual(state["ego"]["speed_mps"], 5.0)
        self.assertEqual(state["timestamp_s"], 12.5)
        world.get_actors.assert_not_called()

    def test_text_schedule_selects_latest_trigger_generic_fields(self):
        fsm = GenericInstructionFSM(default_speed_kmh=32.0)
        commands = [
            {"id": "first", "trigger_progress_m": 10.0, "text": "a"},
            {"id": "second", "trigger_progress_m": 20.0, "text": "b"},
            {
                "id": "announced",
                "announce_at_m": 30.0,
                "activate_at_m": 35.0,
                "text": "c",
            },
        ]
        self.assertNotIn("id", fsm.active_command(commands, 5.0))
        self.assertEqual(fsm.active_command(commands, 15.0)["id"], "first")
        self.assertEqual(fsm.active_command(commands, 20.0)["id"], "second")
        self.assertEqual(fsm.active_command(commands, 36.0)["id"], "announced")

    def test_text_schedule_expires_event_command_to_default(self):
        fsm = GenericInstructionFSM(default_speed_kmh=32.0)
        commands = [
            {
                "id": "temporary_hazard",
                "trigger_progress_m": 10.0,
                "end_progress_m": 20.0,
                "text": "slow down",
            }
        ]
        self.assertEqual(
            fsm.active_command(commands, 19.9)["id"],
            "temporary_hazard",
        )
        self.assertNotIn("id", fsm.active_command(commands, 20.0))
        self.assertNotIn("id", fsm.active_command(commands, 21.0))

    def test_generic_gate_converts_unconfirmed_low_risk_model_stop_to_crawl(self):
        supervisor = GenericTemporalRiskSupervisor()
        canonical = {
            "action": "keep_lane",
            "target_speed_kmh": 32.0,
        }
        stopped = dict(canonical)
        stopped.update(
            {
                "action": "stop",
                "target_speed_kmh": 0.0,
                "reason": "vla_accepted_model",
            }
        )
        decision, override = supervisor.apply(
            stopped,
            canonical,
            {"risk_level": "low", "recommended_action": "keep_lane"},
            parsed_intent="KEEP_LANE",
            requested_lane_direction=None,
            target_lane_risk=None,
            stationary_elapsed_s=0.0,
            resume_active=False,
            resume_speed_kmh=32.0,
        )
        self.assertEqual(override, "unconfirmed_stop_crawl_floor")
        self.assertEqual(decision["action"], "decelerate")
        self.assertEqual(decision["target_speed_kmh"], 10.0)

        # High risk still stops immediately.
        decision, override = supervisor.apply(
            stopped,
            canonical,
            {"risk_level": "high", "recommended_action": "emergency_brake"},
            parsed_intent="KEEP_LANE",
            requested_lane_direction=None,
            target_lane_risk=None,
            stationary_elapsed_s=0.0,
            resume_active=False,
            resume_speed_kmh=32.0,
        )
        self.assertIsNone(override)
        self.assertEqual(decision["action"], "stop")

    def test_generic_liveness_gate_adds_crawl_floor_to_low_risk_deceleration(self):
        supervisor = GenericTemporalRiskSupervisor()
        canonical = {"action": "decelerate", "target_speed_kmh": 30.0}
        stopped = dict(canonical)
        stopped["target_speed_kmh"] = 0.0
        decision, override = supervisor.apply(
            stopped,
            canonical,
            {"risk_level": "low", "recommended_action": "keep_lane"},
            parsed_intent="DECELERATE",
            requested_lane_direction=None,
            target_lane_risk=None,
            stationary_elapsed_s=0.0,
            resume_active=False,
            resume_speed_kmh=32.0,
        )
        self.assertEqual(override, "low_risk_deceleration_crawl")
        self.assertEqual(decision["action"], "decelerate")
        self.assertEqual(decision["target_speed_kmh"], 10.0)

    def test_generic_liveness_gate_never_floors_non_low_risk_deceleration(self):
        supervisor = GenericTemporalRiskSupervisor()
        canonical = {"action": "decelerate", "target_speed_kmh": 30.0}
        stopped = dict(canonical)
        stopped["target_speed_kmh"] = 0.0
        decision, override = supervisor.apply(
            stopped,
            canonical,
            {"risk_level": "medium", "recommended_action": "decelerate"},
            parsed_intent="DECELERATE",
            requested_lane_direction=None,
            target_lane_risk=None,
            stationary_elapsed_s=0.0,
            resume_active=False,
            resume_speed_kmh=32.0,
        )
        self.assertIsNone(override)
        self.assertEqual(decision["target_speed_kmh"], 0.0)

    def test_generic_hazard_clearance_debounces_risk_flicker(self):
        supervisor = GenericTemporalRiskSupervisor(
            TemporalRiskSupervisorConfig(hold_seconds=1.0, min_samples=3)
        )
        for frame in range(4):
            supervisor.observe(
                frame=frame,
                timestamp_s=float(frame) * 0.05,
                parsed_intent="YIELD",
                risk_level="low",
                target_lane_risk_level=None,
                ego_speed_kmh=0.0,
                requested_lane_direction=None,
            )
        stopped = {
            "action": "emergency_brake",
            "target_speed_kmh": 0.0,
            "target_lane": None,
            "emergency": True,
            "reason": "vla_accepted_model",
            "blocked_reason_codes": [],
        }
        held, override = supervisor.apply(
            stopped,
            stopped,
            {"risk_level": "low"},
            parsed_intent="YIELD",
            requested_lane_direction=None,
            target_lane_risk=None,
            stationary_elapsed_s=0.5,
            resume_active=False,
            resume_speed_kmh=20.0,
            hold_seconds=1.0,
        )
        self.assertIsNone(override)

        resumed, override = supervisor.apply(
            stopped,
            stopped,
            {"risk_level": "low"},
            parsed_intent="YIELD",
            requested_lane_direction=None,
            target_lane_risk=None,
            stationary_elapsed_s=3.0,
            resume_active=False,
            resume_speed_kmh=20.0,
            hold_seconds=1.0,
        )
        self.assertEqual(override, "temporal_hazard_clearance")
        self.assertEqual(resumed["action"], "keep_lane")
        self.assertGreaterEqual(resumed["target_speed_kmh"], 10.0)

    def test_generic_blocked_lane_resume_requires_clear_camera_history(self):
        supervisor = GenericTemporalRiskSupervisor(
            TemporalRiskSupervisorConfig(hold_seconds=1.0, min_samples=3)
        )
        for frame in range(4):
            supervisor.observe(
                frame=frame,
                timestamp_s=float(frame) * 0.05,
                parsed_intent="CHANGE_LANE_LEFT",
                risk_level="high",
                target_lane_risk_level="low",
                ego_speed_kmh=0.0,
                requested_lane_direction="left",
            )
        held = {
            "action": "decelerate",
            "target_speed_kmh": 4.0,
            "target_lane": None,
            "emergency": False,
            "reason": "vla_accepted_model",
            "blocked_reason_codes": [],
        }
        unchanged, override = supervisor.apply(
            held,
            held,
            {"risk_level": "high"},
            parsed_intent="CHANGE_LANE_LEFT",
            requested_lane_direction="left",
            target_lane_risk={"risk_level": "low"},
            stationary_elapsed_s=0.5,
            resume_active=False,
            resume_speed_kmh=20.0,
            hold_seconds=1.0,
        )
        self.assertIsNone(override)

        changed, override = supervisor.apply(
            held,
            held,
            {"risk_level": "high"},
            parsed_intent="CHANGE_LANE_LEFT",
            requested_lane_direction="left",
            target_lane_risk={"risk_level": "low"},
            stationary_elapsed_s=3.0,
            resume_active=False,
            resume_speed_kmh=20.0,
            hold_seconds=1.0,
        )
        self.assertEqual(override, "target_lane_visual_clearance")
        self.assertEqual(changed["action"], "lane_change_left")
        self.assertEqual(changed["target_lane"], "left")
        self.assertEqual(changed["target_speed_kmh"], 10.0)


if __name__ == "__main__":
    unittest.main()
