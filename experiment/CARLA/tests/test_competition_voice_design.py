import json
import unittest
from pathlib import Path


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
DESIGN_ROOT = CONFIG_ROOT / "competition_voice_design"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class CompetitionVoiceDesignTests(unittest.TestCase):
    def test_runtime_basic_scene_has_fifteen_monotonic_commands(self):
        config = load_json(CONFIG_ROOT / "basic_voice_control_5km.json")
        commands = config["commands"]
        self.assertEqual(len(commands), 15)
        self.assertEqual(
            [item["announce_at_m"] for item in commands],
            sorted(item["announce_at_m"] for item in commands),
        )
        self.assertEqual(commands[0]["announce_at_m"], 0)
        self.assertLessEqual(commands[-1]["announce_at_m"], config["route"]["length_m"])
        for command in commands:
            if command["id"].startswith("decelerate_to_"):
                self.assertEqual(command["action"], "keep_lane")

    def test_each_competition_design_has_fifteen_ordered_demo_commands(self):
        for filename in (
            "scene_1_basic_voice_5km.json",
            "scene_2_complex_avoidance_8km.json",
            "scene_3_emergency_6km.json",
        ):
            config = load_json(DESIGN_ROOT / filename)
            commands = config["voice_command_schedule"]
            self.assertEqual(len(commands), 15, filename)
            self.assertEqual(config["evaluation_protocol"]["online_demo_commands"], 15, filename)
            self.assertEqual(
                [item["announce_at_m"] for item in commands],
                sorted(item["announce_at_m"] for item in commands),
                filename,
            )

    def test_complex_and_emergency_commands_preserve_ordered_actions(self):
        for filename in (
            "scene_2_complex_avoidance_8km.json",
            "scene_3_emergency_6km.json",
        ):
            config = load_json(DESIGN_ROOT / filename)
            for command in config["voice_command_schedule"]:
                self.assertGreaterEqual(
                    len(command["expected_parse"]["steps"]),
                    3,
                    "{0}: {1}".format(filename, command["id"]),
                )


if __name__ == "__main__":
    unittest.main()
