from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CARLA_DIR = Path(__file__).resolve().parents[1]
if str(CARLA_DIR) not in sys.path:
    sys.path.insert(0, str(CARLA_DIR))

from tools.prepare_scene3_event_manifest import build_records


class PrepareScene3EventManifestTests(unittest.TestCase):
    def test_selects_nearest_sparse_frame_for_active_event(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            frames = run_dir / "camera_frames"
            frames.mkdir()
            (frames / "00001000.png").write_bytes(b"png")
            (frames / "00001200.png").write_bytes(b"png")
            event = {
                "event_id": "scene3_cut_in",
                "scenario": "cut_in_vehicle",
                "voice_command_id": "scene3_cut_in_decelerate",
                "state": "ACTIVE",
                "simulation_frame": 1110,
                "route_s_m": 1250.0,
                "elapsed_s": 100.0,
            }
            (run_dir / "event_timeline.jsonl").write_text(
                json.dumps(event) + "\n",
                encoding="utf-8",
            )
            prompt = run_dir / "prompt.txt"
            prompt.write_text(
                '{"frame": "{frame_id}"} {source} {camera_name}',
                encoding="utf-8",
            )

            records = build_records(
                run_dir,
                prompt,
                maximum_frame_gap=120,
                camera_name="front_rgb",
                image_dir=frames,
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["frame_id"], "carla_1200")
            self.assertEqual(records[0]["camera_name"], "front_rgb")
            self.assertEqual(records[0]["scene3_event"]["frame_gap"], 90)
            self.assertEqual(
                records[0]["prompt"],
                '{"frame": "carla_1200"} carla front_rgb',
            )


if __name__ == "__main__":
    unittest.main()
