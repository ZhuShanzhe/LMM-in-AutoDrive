import contextlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from scene_understanding.core.multimodal_frame_bundle import (
    validate_multimodal_frame_bundle,
)
from scene_understanding.scripts.build_multimodal_vla_input import (
    build_parser,
    main,
)


class BuildMultimodalVlaInputCommandTests(
    unittest.TestCase
):
    def _scene_root(self):
        return Path(__file__).resolve().parents[1]

    def _read_example(self, name):
        path = (
            self._scene_root()
            / "schemas"
            / "examples"
            / name
        )
        return json.loads(
            path.read_text(encoding="utf-8")
        )

    def _write_json(self, path, value):
        path.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _input_paths(self, temp_dir):
        root = Path(temp_dir)

        world = self._read_example(
            "world_state.example.json"
        )
        example = self._read_example(
            "multimodal_frame_bundle.example.json"
        )

        asr_result = {
            "audio_file": "command.wav",
            "chinese_text": (
                "看到前方行人后减速避让，然后向左变道"
            ),
            "english_translation": (
                "Yield to the pedestrian and change left"
            ),
            "asr_processing_time_seconds": 1.234,
        }

        sensor_snapshot = {
            "simulation_frame": world[
                "simulation_frame"
            ],
            "complete": True,
            "missing_modalities": [],
            "cameras": deepcopy(
                example["cameras"]
            ),
            "lidar": deepcopy(
                example["lidar"]
            ),
        }

        asr_path = root / "asr_result.json"
        sensor_path = root / "sensor_snapshot.json"
        world_path = root / "world_state.json"

        self._write_json(asr_path, asr_result)
        self._write_json(
            sensor_path,
            sensor_snapshot,
        )
        self._write_json(world_path, world)

        return asr_path, sensor_path, world_path

    def test_parser_exposes_required_interfaces(self):
        parser = build_parser()
        destinations = {
            action.dest
            for action in parser._actions
        }

        self.assertTrue(
            {
                "request_id",
                "asr_result",
                "instruction_timestamp_s",
                "instruction_confidence",
                "language",
                "sensor_snapshot",
                "world_state",
                "output_dir",
                "output",
                "tolerance_ms",
            }.issubset(destinations)
        )

    def test_command_writes_valid_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (
                asr_path,
                sensor_path,
                world_path,
            ) = self._input_paths(temp_dir)

            root = Path(temp_dir)
            output_dir = root / "artifacts"
            output_path = (
                output_dir
                / "multimodal_frame_bundle.json"
            )

            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "--request-id",
                        "cli-request-1",
                        "--asr-result",
                        str(asr_path),
                        "--instruction-timestamp-s",
                        "6.1",
                        "--instruction-confidence",
                        "0.98",
                        "--language",
                        "zh-CN",
                        "--sensor-snapshot",
                        str(sensor_path),
                        "--world-state",
                        str(world_path),
                        "--output-dir",
                        str(output_dir),
                        "--output",
                        str(output_path),
                        "--tolerance-ms",
                        "50",
                    ]
                )

            self.assertEqual(status, 0)
            self.assertTrue(output_path.is_file())

            bundle = json.loads(
                output_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                validate_multimodal_frame_bundle(
                    bundle
                ),
                [],
            )
            self.assertEqual(
                bundle["request_id"],
                "cli-request-1",
            )
            self.assertEqual(
                bundle["instruction"]["text"],
                "看到前方行人后减速避让，然后向左变道",
            )
            self.assertEqual(
                bundle["synchronization"]["status"],
                "EXACT",
            )
            self.assertTrue(
                Path(
                    bundle["world_state"]["path"]
                ).is_file()
            )
            self.assertIn(
                "Wrote multimodal VLA input",
                stdout.getvalue(),
            )

    def test_invalid_json_returns_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (
                asr_path,
                sensor_path,
                world_path,
            ) = self._input_paths(temp_dir)

            asr_path.write_text(
                "{invalid-json",
                encoding="utf-8",
            )

            root = Path(temp_dir)
            output_path = root / "bundle.json"
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                status = main(
                    [
                        "--request-id",
                        "cli-request-invalid",
                        "--asr-result",
                        str(asr_path),
                        "--instruction-timestamp-s",
                        "6.1",
                        "--instruction-confidence",
                        "0.98",
                        "--language",
                        "zh-CN",
                        "--sensor-snapshot",
                        str(sensor_path),
                        "--world-state",
                        str(world_path),
                        "--output-dir",
                        str(root / "artifacts"),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(status, 1)
            self.assertFalse(output_path.exists())
            self.assertIn(
                "ERROR:",
                stderr.getvalue(),
            )


if __name__ == "__main__":
    unittest.main()
