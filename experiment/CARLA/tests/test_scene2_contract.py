import importlib.util
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
if HERE.name == "tests":
    ROOT = HERE.parent
    SCRIPT = ROOT / "maps" / "decorate_complex_scene.py"
    INTERFACE = ROOT / "scene2_runtime_interface.py"
    CONFIG = (
        ROOT
        / "configs"
        / "scene_2_complex_avoidance_8km_runtime.json"
    )
    XODR = (
        ROOT
        / "maps"
        / "maps"
        / "output"
        / "VLA_ComplexRoad_8km.xodr"
    )
else:
    ROOT = HERE
    SCRIPT = ROOT / "decorate_complex_scene.py"
    INTERFACE = ROOT / "scene2_runtime_interface.py"
    CONFIG = ROOT / "scene_2_complex_avoidance_8km_runtime.json"
    XODR = ROOT / "VLA_ComplexRoad_8km.xodr"

spec = importlib.util.spec_from_file_location(
    "scene2_runner",
    SCRIPT,
)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class Scene2ContractTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_runtime_contract_is_valid(self):
        runner.validate_runtime_config(self.config)

    def test_route_is_exactly_eight_kilometres(self):
        self.assertEqual(self.config["route"]["length_m"], 8000.0)
        self.assertEqual(len(self.config["segments"]), 6)
        self.assertEqual(
            [segment["range_m"] for segment in self.config["segments"]],
            [
                [0, 1100],
                [1100, 2300],
                [2300, 3600],
                [3600, 5200],
                [5200, 6800],
                [6800, 8000],
            ],
        )

    def test_fifteen_commands_are_monotonic_and_composite(self):
        commands = self.config["voice_commands"]
        self.assertEqual(len(commands), 15)
        positions = [item["announce_at_m"] for item in commands]
        self.assertEqual(positions, sorted(positions))
        for command in commands:
            self.assertGreaterEqual(len(command["steps"]), 3)
            self.assertTrue(command["spoken_text"].strip())
            self.assertNotIn("?", command["spoken_text"])

    def test_required_special_events_are_bound_to_visible_actors(self):
        events = self.config["events"]
        self.assertEqual(len(events), 6)
        self.assertEqual(
            {event["id"] for event in events},
            {
                "s2_evt_crosswalk_pedestrian",
                "s2_evt_bus_stop_alighting",
                "s2_evt_left_turn_signal",
                "s2_evt_cyclist_pass",
                "s2_evt_overtake_gap",
                "s2_evt_final_u_turn",
            },
        )
        for event in events:
            self.assertTrue(event["actors"])
            self.assertLess(
                event["activate_at_m"],
                event["resolve_at_m"],
            )

    def test_competition_traffic_counts_are_exact(self):
        self.assertEqual(
            self.config["traffic"],
            {
                "private_cars": 24,
                "city_buses": 3,
                "bicycles": 6,
                "sidewalk_pedestrians": 18,
                "seed": 2026,
            },
        )

    def test_multimodal_interface_uses_exact_simulation_frame(self):
        interfaces = self.config["interfaces"]
        self.assertEqual(
            interfaces["driving_intent_schema"],
            "DrivingIntent/1.2.0",
        )
        self.assertEqual(
            interfaces["multimodal_bundle_schema"],
            "MultimodalFrameBundle/1.0.0",
        )
        self.assertEqual(
            interfaces["vla_proposal_schema"],
            "VlaActionProposal/1.0.0",
        )
        self.assertEqual(
            interfaces["control_decision_schema"],
            "ControlDecision/1.0.0",
        )
        self.assertTrue(
            interfaces["simulation_frame_is_sync_key"]
        )
        self.assertFalse(interfaces["allow_adjacent_frame_fill"])

    def test_control_interface_rejects_stale_or_ungated_decisions(self):
        interface_spec = importlib.util.spec_from_file_location(
            "scene2_runtime_interface_test",
            INTERFACE,
        )
        module = importlib.util.module_from_spec(interface_spec)
        assert interface_spec.loader is not None
        interface_spec.loader.exec_module(module)
        good = {
            "simulation_frame": 42,
            "action": "decelerate",
            "target_speed_kmh": 30,
            "safety_gate_status": "APPROVED",
        }
        result = module.validate_control_decision(good, 42)
        self.assertEqual(result["action"], "decelerate")
        with self.assertRaises(ValueError):
            module.validate_control_decision(good, 43)
        with self.assertRaises(ValueError):
            module.validate_control_decision(
                {**good, "safety_gate_status": "PENDING"},
                42,
            )

    def test_chase_view_matches_shared_experiment_camera(self):
        view = self.config["sensors"]["presentation_view"]
        self.assertEqual(view["x"], -10.0)
        self.assertEqual(view["z"], 3.0)
        self.assertEqual(view["pitch"], -8.0)
        self.assertEqual(
            view["attachment_type"],
            "Rigid",
        )
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("camera_pose=(-10.0, 0.0, 3.0, -8.0, 0.0)", source)
        self.assertIn(
            "RGB camera: chase_rgb (third-person shared view)",
            source,
        )

    def test_xodr_has_required_lane_families(self):
        contract = runner.validate_xodr(XODR)
        self.assertEqual(contract["main_road_length_m"], 8000.0)
        self.assertGreaterEqual(
            contract["lane_types"]["driving"],
            4,
        )
        self.assertGreaterEqual(
            contract["lane_types"]["biking"],
            2,
        )
        self.assertGreaterEqual(
            contract["lane_types"]["sidewalk"],
            2,
        )

    def test_xodr_has_real_junction_and_bus_stop(self):
        root = ET.parse(XODR).getroot()
        self.assertGreaterEqual(len(root.findall("junction")), 1)
        objects = root.findall(
            "./road[@id='1']/objects/object"
        )
        bus_stops = [
            item for item in objects
            if item.get("type") == "busStop"
        ]
        self.assertEqual(len(bus_stops), 1)
        self.assertEqual(float(bus_stops[0].get("s")), 1770.0)
        object_ids = [item.get("id") for item in objects]
        self.assertEqual(
            len(object_ids),
            len(set(object_ids)),
        )

    def test_acceptance_thresholds_match_repository_plan(self):
        acceptance = self.config["acceptance"]
        self.assertEqual(
            acceptance["collision_count_equals"],
            0,
        )
        self.assertEqual(
            acceptance["violation_count_at_most"],
            1,
        )
        self.assertEqual(
            acceptance["asr_accuracy_min"],
            0.96,
        )
        self.assertEqual(
            acceptance["semantic_alignment_accuracy_min"],
            0.985,
        )
        self.assertEqual(
            acceptance["ordered_step_omissions"],
            0,
        )

    def test_legal_broken_line_change_is_not_a_violation(self):
        monitor = runner.SafetyMonitor(None, None, None)
        monitor.lane_invasions = [
            {"crossed_lane_markings": ["LaneMarkingType.Broken"]},
            {"crossed_lane_markings": ["LaneMarkingType.Solid"]},
        ]
        self.assertEqual(monitor.violation_count, 1)


if __name__ == "__main__":
    unittest.main()
