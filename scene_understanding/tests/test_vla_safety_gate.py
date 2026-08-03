import copy
import json
import unittest
from pathlib import Path

from scene_understanding.src.control_decision import (
    validate_control_decision,
)
from scene_understanding.src.vla_safety_gate import (
    gate_vla_action_proposal,
)


class VlaSafetyGateTests(unittest.TestCase):
    def _scene_root(self):
        return Path(__file__).resolve().parents[1]

    def _example(self, name):
        path = (
            self._scene_root()
            / "schemas"
            / "examples"
            / name
        )
        return json.loads(
            path.read_text(encoding="utf-8")
        )

    def _inputs(self):
        bundle = self._example(
            "multimodal_frame_bundle.example.json"
        )
        proposal = self._example(
            "vla_action_proposal.example.json"
        )
        world = self._example(
            "world_state.example.json"
        )
        risk = self._example(
            "risk_assessment.example.json"
        )

        proposal["request_id"] = bundle["request_id"]
        proposal["bundle_id"] = bundle["bundle_id"]
        proposal["frame_id"] = bundle["frame_id"]
        proposal["simulation_frame"] = bundle[
            "simulation_frame"
        ]

        risk["frame_id"] = bundle["frame_id"]
        world["frame_id"] = bundle["frame_id"]
        world["simulation_frame"] = bundle[
            "simulation_frame"
        ]

        # 默认构造一个允许正常决策通过的低风险状态。
        risk["risk_level"] = "low"
        risk["recommended_action"] = "monitor"
        risk["reason_codes"] = [
            "distant_object_monitoring"
        ]
        risk["lane_change"]["left"]["is_safe"] = True
        risk["lane_change"]["left"][
            "reason_codes"
        ] = ["target_lane_clear"]
        risk["lane_change"]["right"]["is_safe"] = True
        risk["lane_change"]["right"][
            "reason_codes"
        ] = ["target_lane_clear"]

        return proposal, bundle, world, risk

    def _gate(
        self,
        proposal,
        bundle,
        world,
        risk,
    ):
        decision = gate_vla_action_proposal(
            proposal,
            bundle,
            world,
            risk,
            min_confidence=0.70,
        )
        self.assertEqual(
            validate_control_decision(decision),
            [],
        )
        return decision

    def test_safe_lane_change_becomes_ready(self):
        proposal, bundle, world, risk = (
            self._inputs()
        )

        decision = self._gate(
            proposal,
            bundle,
            world,
            risk,
        )

        self.assertEqual(
            decision["decision_status"],
            "READY",
        )
        self.assertEqual(
            decision["action"],
            "lane_change_left",
        )
        self.assertEqual(
            decision["target_lane"],
            "left",
        )
        self.assertFalse(decision["emergency"])

    def test_high_risk_forces_emergency_brake(self):
        proposal, bundle, world, risk = (
            self._inputs()
        )
        proposal["action"] = "keep_lane"
        proposal["target_lane"] = None
        risk["risk_level"] = "high"
        risk["recommended_action"] = (
            "emergency_brake"
        )
        risk["reason_codes"] = [
            "ttc_below_1s"
        ]

        decision = self._gate(
            proposal,
            bundle,
            world,
            risk,
        )

        self.assertEqual(
            decision["decision_status"],
            "BLOCKED",
        )
        self.assertEqual(
            decision["action"],
            "emergency_brake",
        )
        self.assertTrue(decision["emergency"])
        self.assertEqual(
            decision["reason"],
            "risk_requires_emergency_brake",
        )

    def test_risk_deceleration_overrides_lane_change(self):
        proposal, bundle, world, risk = (
            self._inputs()
        )
        risk["risk_level"] = "medium"
        risk["recommended_action"] = "decelerate"
        risk["reason_codes"] = [
            "predicted_path_conflict"
        ]

        decision = self._gate(
            proposal,
            bundle,
            world,
            risk,
        )

        self.assertEqual(
            decision["decision_status"],
            "BLOCKED",
        )
        self.assertEqual(
            decision["action"],
            "decelerate",
        )
        self.assertEqual(
            decision["reason"],
            "risk_requires_deceleration",
        )

    def test_unsafe_lane_change_is_stopped(self):
        proposal, bundle, world, risk = (
            self._inputs()
        )
        risk["lane_change"]["left"]["is_safe"] = (
            False
        )
        risk["lane_change"]["left"][
            "reason_codes"
        ] = ["front_gap_below_15m"]

        decision = self._gate(
            proposal,
            bundle,
            world,
            risk,
        )

        self.assertEqual(
            decision["decision_status"],
            "BLOCKED",
        )
        self.assertEqual(
            decision["action"],
            "stop",
        )
        self.assertEqual(
            decision["reason"],
            "lane_change_left_blocked",
        )
        self.assertIn(
            "front_gap_below_15m",
            decision["blocked_reason_codes"],
        )

    def test_low_confidence_proposal_safe_stops(self):
        proposal, bundle, world, risk = (
            self._inputs()
        )
        proposal["confidence"] = 0.40

        decision = self._gate(
            proposal,
            bundle,
            world,
            risk,
        )

        self.assertEqual(
            decision["decision_status"],
            "SAFE_FALLBACK",
        )
        self.assertEqual(
            decision["action"],
            "stop",
        )
        self.assertEqual(
            decision["reason"],
            "vla_proposal_low_confidence",
        )

    def test_stale_frame_proposal_safe_stops(self):
        proposal, bundle, world, risk = (
            self._inputs()
        )
        proposal["frame_id"] = "carla_stale"
        proposal["simulation_frame"] = 122

        decision = self._gate(
            proposal,
            bundle,
            world,
            risk,
        )

        self.assertEqual(
            decision["decision_status"],
            "SAFE_FALLBACK",
        )
        self.assertEqual(
            decision["action"],
            "stop",
        )
        self.assertEqual(
            decision["reason"],
            "vla_proposal_identity_mismatch",
        )

    def test_incomplete_bundle_safe_stops(self):
        proposal, bundle, world, risk = (
            self._inputs()
        )
        bundle["lidar"] = None
        bundle["synchronization"]["status"] = (
            "INCOMPLETE"
        )
        bundle["synchronization"][
            "missing_modalities"
        ] = ["lidar"]

        decision = self._gate(
            proposal,
            bundle,
            world,
            risk,
        )

        self.assertEqual(
            decision["decision_status"],
            "SAFE_FALLBACK",
        )
        self.assertEqual(
            decision["action"],
            "stop",
        )
        self.assertEqual(
            decision["reason"],
            "multimodal_bundle_incomplete",
        )

    def test_unknown_matched_entity_safe_stops(self):
        proposal, bundle, world, risk = (
            self._inputs()
        )
        proposal["matched_entity_id"] = (
            "carla_actor_missing"
        )

        decision = self._gate(
            proposal,
            bundle,
            world,
            risk,
        )

        self.assertEqual(
            decision["decision_status"],
            "SAFE_FALLBACK",
        )
        self.assertEqual(
            decision["action"],
            "stop",
        )
        self.assertEqual(
            decision["reason"],
            "matched_entity_not_in_world_state",
        )


if __name__ == "__main__":
    unittest.main()
