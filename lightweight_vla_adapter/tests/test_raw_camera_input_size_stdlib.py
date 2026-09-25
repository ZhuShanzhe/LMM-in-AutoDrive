"""Regression tests for configurable raw-camera encoder resolution."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.src.raw_sensor_encoder import MultiviewImageEncoder


class RawCameraInputSizeTests(unittest.TestCase):
    def test_build_model_uses_configured_height_and_width(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (root / "lightweight_vla_adapter/configs/challenge_sequence_v2.json").read_text(
                encoding="utf-8"
            )
        )
        config["camera_input_height"] = 160
        config["camera_input_width"] = 192
        model = build_model(config)
        self.assertEqual(model.raw_camera_encoder.input_size, (160, 192))

    def test_default_size_remains_224_square(self) -> None:
        encoder = MultiviewImageEncoder(hidden_size=32)
        self.assertEqual(encoder.input_size, (224, 224))

    def test_invalid_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive dimensions"):
            MultiviewImageEncoder(hidden_size=32, input_size=(0, 224))


if __name__ == "__main__":
    unittest.main()
