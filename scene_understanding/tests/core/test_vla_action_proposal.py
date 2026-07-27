import copy
import unittest

from scene_understanding.core.vla_action_proposal import (
    validate_vla_action_proposal,
)


class VlaActionProposalTests(unittest.TestCase):
    def _proposal(self):
        return {
            "schema_version": "1.0.0",
            "proposal_id": "vla-proposal-001",
            "request_id": "request-001",
            "bundle_id": (
                "bundle_carla_000123_request_001"
            ),
            "frame_id": "carla_000123",
            "simulation_frame": 123,
            "status": "VALID",
            "action": "lane_change_left",
            "target_speed_kmh": 25.0,
            "target_lane": "left",
            "target_location": None,
            "confidence": 0.91,
            "model_name": "multimodal-vla",
            "inference_latency_ms": 42.5,
            "matched_entity_id": None,
            "evidence_modalities": [
                "instruction",
                "front_rgb",
                "left_rgb",
                "right_rgb",
                "rear_rgb",
                "lidar",
                "world_state",
            ],
            "reason_codes": [
                "multimodal_context_supports_lane_change"
            ],
        }

    def test_accepts_valid_multimodal_proposal(self):
        self.assertEqual(
            validate_vla_action_proposal(
                self._proposal()
            ),
            [],
        )

    def test_rejects_unexpected_field(self):
        proposal = self._proposal()
        proposal["raw_model_output"] = "unsafe"

        errors = validate_vla_action_proposal(
            proposal
        )

        self.assertTrue(
            any(
                "unexpected fields" in error
                for error in errors
            ),
            errors,
        )

    def test_lane_change_requires_matching_lane(self):
        proposal = self._proposal()
        proposal["target_lane"] = "right"

        errors = validate_vla_action_proposal(
            proposal
        )

        self.assertIn(
            "target_lane: must be 'left' "
            "for lane_change_left",
            errors,
        )

    def test_stop_requires_zero_speed(self):
        proposal = self._proposal()
        proposal["action"] = "stop"
        proposal["target_lane"] = None
        proposal["target_speed_kmh"] = 10.0

        errors = validate_vla_action_proposal(
            proposal
        )

        self.assertIn(
            "target_speed_kmh: must be 0 "
            "for stop or emergency_brake",
            errors,
        )

    def test_rejects_duplicate_evidence(self):
        proposal = self._proposal()
        proposal["evidence_modalities"].append(
            "front_rgb"
        )

        errors = validate_vla_action_proposal(
            proposal
        )

        self.assertIn(
            "evidence_modalities: "
            "entries must be unique",
            errors,
        )

    def test_rejects_invalid_confidence(self):
        proposal = self._proposal()
        proposal["confidence"] = 1.1

        errors = validate_vla_action_proposal(
            proposal
        )

        self.assertIn(
            "confidence: expected a finite number "
            "between 0 and 1",
            errors,
        )

    def test_rejects_non_object(self):
        self.assertEqual(
            validate_vla_action_proposal([]),
            ["root: expected an object"],
        )


if __name__ == "__main__":
    unittest.main()
