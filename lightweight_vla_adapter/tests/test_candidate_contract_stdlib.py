"""Regression tests for the challenge-track candidate tensor contract."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from lightweight_vla_adapter.src.structured_bev import StructuredBEVRasterizer


class CandidateContractTests(unittest.TestCase):
    def test_structured_proxy_has_stable_semantics_and_alignment(self) -> None:
        rasterizer = StructuredBEVRasterizer(max_candidates=2)
        intent_tokens = torch.zeros(1, 32, 768)
        intent_mask = torch.ones(1, 32, dtype=torch.bool)
        batch, entity_ids = rasterizer.build(
            {
                "objects": [
                    {
                        "entity_id": "vehicle-7",
                        "relative_position_m": [12.0, -2.0, 0.5],
                        "relative_velocity_mps": {"x": 1.0, "y": -0.2},
                        "confidence": 0.8,
                        "lane_relation": "left",
                        "class": "vehicle",
                       },
                    {
                        "entity_id": "ped-3",
                        "relative_position_m": [8.0, 3.0, 0.0],
                        "confidence": 0.6,
                        "lane_relation": "same",
                        "class": "pedestrian",
                       },
                ],
                "ego": {},
                "environment": {},
            },
            intent_tokens=intent_tokens,
            intent_mask=intent_mask,
        )
        self.assertEqual(tuple(batch.candidate_features.shape), (1, 2, 12))
        self.assertEqual(tuple(batch.candidate_mask.shape), (1, 2))
        self.assertEqual(entity_ids, [["vehicle-7", "ped-3"]])
        self.assertTrue(torch.equal(batch.candidate_mask, torch.tensor([[True, True]])))
        expected = torch.tensor(
            [12.0, -2.0, 0.5, (12.0**2 + 2.0**2 + 0.5**2) ** 0.5,
             1.0, -0.2, 0.8, 1.0, 0.0, 0.0, 0.0, 1.0]
        )
        torch.testing.assert_close(batch.candidate_features[0, 0], expected)
        self.assertEqual(float(batch.candidate_features[0, 1, 10]), 1.0)
        self.assertEqual(float(batch.candidate_features[0, 1, 11]), 0.0)

    def test_challenge_config_keeps_candidate_entities_disabled(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (root / "lightweight_vla_adapter/configs/challenge_sequence_v2.json").read_text(encoding="utf-8")
        )
        self.assertIs(config["use_candidate_entities"], False)
        self.assertIs(config["policy_input_contract"]["candidate_entities_allowed"], False)

if __name__ == "__main__":
    unittest.main()
