import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from scene_understanding.core.multimodal_frame_bundle import (
    validate_multimodal_frame_bundle,
)
from scene_understanding.core.multimodal_input_adapter import (
    assemble_vla_multimodal_input,
    normalize_asr_instruction,
    write_world_state_reference,
)


class MultimodalInputAdapterTests(unittest.TestCase):
    def _scene_root(self):
        return Path(__file__).resolve().parents[2]

    def _world_state(self):
        path = (
            self._scene_root()
            / "schemas"
            / "examples"
            / "world_state.example.json"
        )
        return json.loads(
            path.read_text(encoding="utf-8")
        )

    def _bundle_example(self):
        path = (
            self._scene_root()
            / "schemas"
            / "examples"
            / "multimodal_frame_bundle.example.json"
        )
        return json.loads(
            path.read_text(encoding="utf-8")
        )

    def test_normalizes_pipeline_chinese_text(self):
        result = {
            "audio_file": "command.wav",
            "chinese_text": "前方路口右转",
            "english_translation": (
                "Turn right at the intersection"
            ),
            "asr_processing_time_seconds": 1.234,
            "total_time_seconds": 1.8,
        }

        instruction = normalize_asr_instruction(
            result,
            timestamp_s=6.1,
            confidence=0.93,
            language="zh-CN",
        )

        self.assertEqual(
            instruction,
            {
                "source": "asr",
                "text": "前方路口右转",
                "language": "zh-CN",
                "confidence": 0.93,
                "timestamp_s": 6.1,
            },
        )

        # 模型耗时不能冒充语音采集时间。
        self.assertNotEqual(
            instruction["timestamp_s"],
            result["asr_processing_time_seconds"],
        )

    def test_normalizes_service_text_fallback(self):
        result = {
            "audio_file": "command.wav",
            "text": "减速至四十公里每小时",
            "processing_time_seconds": 0.8,
        }

        instruction = normalize_asr_instruction(
            result,
            timestamp_s=12.5,
            confidence=0.88,
            language="zh-CN",
        )

        self.assertEqual(
            instruction["text"],
            result["text"],
        )
        self.assertEqual(
            instruction["timestamp_s"],
            12.5,
        )

    def test_rejects_missing_confidence(self):
        result = {
            "audio_file": "command.wav",
            "chinese_text": "停车",
        }

        with self.assertRaisesRegex(
            ValueError,
            "confidence",
        ):
            normalize_asr_instruction(
                result,
                timestamp_s=2.0,
                language="zh-CN",
            )

    def test_rejects_empty_transcription(self):
        with self.assertRaisesRegex(
            ValueError,
            "text",
        ):
            normalize_asr_instruction(
                {
                    "audio_file": "command.wav",
                    "chinese_text": "   ",
                },
                timestamp_s=2.0,
                confidence=0.5,
                language="zh-CN",
            )

    def test_writes_valid_world_state_reference(self):
        world = self._world_state()

        with tempfile.TemporaryDirectory() as temp_dir:
            reference = write_world_state_reference(
                world,
                output_dir=temp_dir,
            )

            self.assertEqual(
                set(reference),
                {
                    "frame_id",
                    "simulation_frame",
                    "timestamp_s",
                    "path",
                },
            )
            self.assertEqual(
                reference["frame_id"],
                world["frame_id"],
            )
            self.assertEqual(
                reference["simulation_frame"],
                world["simulation_frame"],
            )

            saved_path = Path(reference["path"])

            self.assertTrue(saved_path.is_file())
            self.assertEqual(
                json.loads(
                    saved_path.read_text(
                        encoding="utf-8"
                    )
                ),
                world,
            )

    def test_assembles_asr_sensors_and_world_state(self):
        world = self._world_state()
        example = self._bundle_example()

        sensor_snapshot = {
            "simulation_frame": world[
                "simulation_frame"
            ],
            "complete": True,
            "missing_modalities": [],
            "cameras": deepcopy(
                example["cameras"]
            ),
            "lidar": deepcopy(example["lidar"]),
        }

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

        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = assemble_vla_multimodal_input(
                request_id="req-1",
                asr_result=asr_result,
                instruction_timestamp_s=6.1,
                instruction_confidence=0.98,
                language="zh-CN",
                sensor_snapshot=sensor_snapshot,
                world_state=world,
                output_dir=temp_dir,
                tolerance_ms=50.0,
            )

            errors = validate_multimodal_frame_bundle(
                bundle
            )

            self.assertEqual(errors, [])
            self.assertEqual(
                bundle["instruction"]["text"],
                asr_result["chinese_text"],
            )
            self.assertEqual(
                bundle["simulation_frame"],
                world["simulation_frame"],
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


if __name__ == "__main__":
    unittest.main()
