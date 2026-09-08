from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CARLA_DIR = Path(__file__).resolve().parents[1]
if str(CARLA_DIR) not in sys.path:
    sys.path.insert(0, str(CARLA_DIR))

from tools.prepare_scene3_event_manifest import build_records, portable_image_path


class PrepareScene3EventManifestTests(unittest.TestCase):
    def test_repository_image_path_is_portable(self):
        path = CARLA_DIR / "outputs" / "sample" / "00000001.png"
        self.assertEqual(
            portable_image_path(path),
            "experiment/CARLA/outputs/sample/00000001.png",
        )

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
            self.assertEqual(records[0]["scene3_event"]["selection"], "activation")
            self.assertEqual(
                records[0]["prompt"],
                '{"frame": "carla_1200"} carla front_rgb',
            )

    def test_observed_target_prefers_actor_near_requested_distance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            frames = run_dir / "rgb" / "front_rgb"
            frames.mkdir(parents=True)
            for frame in (1000, 1100, 1200):
                (frames / f"{frame:08d}.png").write_bytes(b"png")
            event = {
                "event_id": "scene3_cut_in",
                "scenario": "cut_in_vehicle",
                "voice_command_id": "scene3_cut_in_decelerate",
                "state": "ACTIVE",
                "simulation_frame": 1000,
                "route_s_m": 1250.0,
                "elapsed_s": 100.0,
            }
            (run_dir / "event_timeline.jsonl").write_text(
                json.dumps(event) + "\n",
                encoding="utf-8",
            )
            truth_rows = []
            for frame, distance in ((1100, 40.0), (1200, 21.0)):
                truth_rows.append(
                    {
                        "simulation_frame": frame,
                        "route_s_m": 1300.0,
                        "timestamp_s": 105.0,
                        "active_events": [
                            {
                                "event_id": "scene3_cut_in",
                                "scenario": "cut_in_vehicle",
                                "evidence": {"observed": ["cut_in_vehicle"]},
                            }
                        ],
                        "actors": {
                            "cut_in_vehicle": {
                                "relation_to_ego": {
                                    "longitudinal_m": distance,
                                    "lateral_m": 2.0,
                                    "euclidean_distance_m": distance,
                                }
                            }
                        },
                    }
                )
            (run_dir / "frame_ground_truth.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in truth_rows),
                encoding="utf-8",
            )
            prompt = run_dir / "prompt.txt"
            prompt.write_text("{frame_id} {source} {camera_name}", encoding="utf-8")

            records = build_records(
                run_dir,
                prompt,
                maximum_frame_gap=0,
                camera_name="front_rgb",
                image_dir=frames,
                selection="observed-target",
                target_distance_m=20.0,
            )

            scene_event = records[0]["scene3_event"]
            self.assertEqual(scene_event["selected_image_frame"], 1200)
            self.assertEqual(scene_event["ground_truth_frame"], 1200)
            self.assertEqual(scene_event["selected_actor"]["distance_m"], 21.0)


if __name__ == "__main__":
    unittest.main()
