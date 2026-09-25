"""Full configured-adapter regression for inference Conv/BN folding."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import unittest

import torch
from torch import nn

from lightweight_vla_adapter.scripts.benchmark_latency import configured_inputs
from lightweight_vla_adapter.scripts.run_offline_inference import build_model
from lightweight_vla_adapter.src.inference_optimization import fuse_inference_conv_bn


class FullAdapterFusionTests(unittest.TestCase):
    def test_full_challenge_adapter_preserves_outputs(self) -> None:
        torch.manual_seed(911)
        torch.set_num_threads(2)
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (root / "lightweight_vla_adapter/configs/challenge_sequence_v2.json").read_text(
                encoding="utf-8"
            )
        )
        model = build_model(config).eval()
        source_state = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        source_batch_norms = sum(
            isinstance(module, nn.BatchNorm2d) for module in model.modules()
        )
        inputs = configured_inputs(config, torch.device("cpu"), torch.float32)
        fused, count = fuse_inference_conv_bn(model)

        self.assertGreater(count, 0)
        self.assertEqual(count, source_batch_norms)
        self.assertEqual(
            sum(isinstance(module, nn.BatchNorm2d) for module in fused.modules()), 0
        )
        with torch.inference_mode():
            expected = model(**inputs)
            actual = fused(**inputs)
        for field in fields(type(expected)):
            expected_value = getattr(expected, field.name)
            actual_value = getattr(actual, field.name)
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(
                    actual_value, expected_value, rtol=3e-4, atol=3e-5
                )
            else:
                self.assertEqual(actual_value, expected_value, field.name)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, source_state[name]), name)


if __name__ == "__main__":
    unittest.main()
