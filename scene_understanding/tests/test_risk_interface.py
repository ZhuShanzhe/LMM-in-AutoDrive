import json
import tempfile
import unittest
from pathlib import Path

from scene_understanding.scripts.assess_risk import main
from scene_understanding.src.risk_interface import assess_scene_risk
from scene_understanding.core.risk_assessment import validate_risk_assessment


ROOT = Path(__file__).resolve().parents[2]
WORLD_STATE_EXAMPLE = ROOT / "scene_understanding" / "schemas" / "examples" / "world_state.example.json"


class RiskInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world_state = json.loads(WORLD_STATE_EXAMPLE.read_text(encoding="utf-8"))

    def test_returns_existing_valid_risk_contract(self):
        result = assess_scene_risk(self.world_state)
        self.assertEqual(validate_risk_assessment(result), [])
        self.assertEqual(result["frame_id"], self.world_state["frame_id"])
        self.assertEqual(result["risk_level"], "medium")

    def test_command_writes_json_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "risk_assessment.json"
            code = main(
                [
                    "--world-state",
                    str(WORLD_STATE_EXAMPLE),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(code, 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(validate_risk_assessment(result), [])


if __name__ == "__main__":
    unittest.main()
