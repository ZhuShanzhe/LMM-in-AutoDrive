import json
import tempfile
import unittest
from pathlib import Path

from evaluation.run_preflight import evaluate_run


class RunPreflightTests(unittest.TestCase):
    def _write_fixture(self, directory, brake=1.0, completed=None, failed=None):
        Path(directory, "metrics.json").write_text(json.dumps({
            "collision_free": True,
            "illegal_lane_invasion_free": True,
            "collision_events": 0,
            "illegal_lane_invasion_events": 0,
            "scenario_status": {"scenario_events": {
                "completed": completed if completed is not None else ["lead"],
                "failed": failed if failed is not None else [],
            }},
        }), encoding="utf-8")
        Path(directory, "events.jsonl").write_text(json.dumps({
            "type": "scenario_event", "transition": "activated", "event_id": "lead",
        }) + "\n", encoding="utf-8")
        frames = [
            {"scenario_status": {"traffic": {"background_actor_count": 9}, "pedestrians": {"walker_count": 3}, "scenario_events": {"active_details": [{"id": "lead", "status": "BRAKING"}]}}, "ego": {"speed_kmh": 40.0}, "control": {"throttle": 0.4, "brake": 0.0}},
            {"scenario_status": {"traffic": {"background_actor_count": 12}, "pedestrians": {"walker_count": 5}}, "ego": {"speed_kmh": 0.0}, "control": {"throttle": 0.0, "brake": brake}},
        ]
        Path(directory, "frames.jsonl").write_text(
            "".join(json.dumps(frame) + "\n" for frame in frames), encoding="utf-8"
        )

    def test_passing_run_reports_event_traffic_and_control_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_fixture(directory)
            report = evaluate_run(
                directory,
                ["lead"],
                require_emergency_brake=True,
                require_fresh_event_evidence=True,
                require_event_behavior=True,
            )
        self.assertTrue(report["passed"])
        self.assertEqual(report["traffic_actor_count"], {"min": 9, "max": 12})
        self.assertEqual(report["pedestrian_count"], {"min": 3, "max": 5})
        self.assertEqual(report["control"]["simultaneous_throttle_brake_frames"], 0)
        self.assertTrue(report["checks"]["required_events_activated_this_run"])
        self.assertTrue(report["checks"]["required_event_behavior_observed"])
        self.assertEqual(report["control"]["strong_brake_windows"][0]["min_speed_next_2s_kmh"], 0.0)

    def test_missing_event_or_missing_brake_fails_the_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_fixture(directory, brake=0.4, completed=[])
            report = evaluate_run(directory, ["lead"], require_emergency_brake=True)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["required_events_completed"])
        self.assertFalse(report["checks"]["emergency_brake_observed"])


if __name__ == "__main__":
    unittest.main()
