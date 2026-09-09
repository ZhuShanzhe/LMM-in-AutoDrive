from __future__ import annotations

import sys
import unittest
import copy
from pathlib import Path

from lightweight_vla_adapter.src.safety_bridge import (
    advance_vla_control_plan,
    gate_vla_proposal,
)
from lightweight_vla_adapter.tests.fixtures import integration_documents, proposal
from scene_understanding.src.control_decision import (
    build_control_decision,
    validate_control_decision,
)


CARLA_ROOT = Path(__file__).resolve().parents[2] / "experiment" / "CARLA"
if str(CARLA_ROOT) not in sys.path:
    sys.path.insert(0, str(CARLA_ROOT))

from control.protocol import normalize_intent


class SafetyBridgeTest(unittest.TestCase):
    def test_compatible_proposal_reaches_existing_carla_protocol(self):
        intent, world, alignment, risk = integration_documents()
        canonical = build_control_decision(intent, world, alignment, risk)
        final = gate_vla_proposal(proposal(), canonical, risk)
        normalized = normalize_intent(final)
        self.assertEqual(final["action"], "lane_change_left")
        self.assertEqual(normalized["action"], "lane_change_left")
        self.assertLessEqual(
            final["target_speed_kmh"], canonical["target_speed_kmh"]
        )

    def test_opposite_lane_change_is_rejected(self):
        intent, world, alignment, risk = integration_documents()
        canonical = build_control_decision(intent, world, alignment, risk)
        final = gate_vla_proposal(
            proposal(action="lane_change_right"),
            canonical,
            risk,
        )
        self.assertEqual(final["action"], "lane_change_left")
        self.assertIn(
            "vla_incompatible_with_active_intent",
            final["blocked_reason_codes"],
        )

    def test_emergency_risk_overrides_model(self):
        intent, world, alignment, risk = integration_documents(
            recommended_action="emergency_brake",
            risk_level="high",
        )
        canonical = build_control_decision(intent, world, alignment, risk)
        final = gate_vla_proposal(
            proposal(action="accelerate", target_speed_kmh=60.0),
            canonical,
            risk,
        )
        self.assertEqual(final["action"], "emergency_brake")
        self.assertEqual(final["target_speed_kmh"], 0.0)

    def test_existing_fsm_remains_authoritative(self):
        intent, world, alignment, risk = integration_documents()
        state, decision = advance_vla_control_plan(
            intent,
            world,
            alignment,
            risk,
            proposal(action="keep_lane"),
        )
        self.assertEqual(state["plan_status"], "ACTIVE")
        self.assertEqual(state["active_step_id"], "step_1")
        self.assertEqual(decision["action"], "lane_change_left")
        self.assertIn("vla_incompatible_with_active_intent", decision["blocked_reason_codes"])

    def test_safe_requested_lane_is_followed_for_non_braking_proposals(self):
        for direction in ("LEFT", "RIGHT"):
            for action in ("keep_lane", "accelerate", "lane_change_left", "lane_change_right"):
                with self.subTest(direction=direction, proposal=action):
                    intent, world, alignment, risk = integration_documents(direction=direction)
                    canonical = build_control_decision(intent, world, alignment, risk)
                    candidate = proposal(action=action, target_speed_kmh=0.0 if action in {"stop", "emergency_brake"} else 18.0)
                    final = gate_vla_proposal(candidate, canonical, risk)
                    self.assertEqual(final["action"], "lane_change_" + direction.lower())
                    self.assertEqual(validate_control_decision(final), [])

    def test_conservative_braking_is_not_replaced_by_lane_change(self):
        intent, world, alignment, risk = integration_documents()
        canonical = build_control_decision(intent, world, alignment, risk)
        for action in ("decelerate", "stop", "emergency_brake"):
            final = gate_vla_proposal(
                proposal(action=action, target_speed_kmh=0.0), canonical, risk,
            )
            self.assertEqual(final["action"], action)
            self.assertIsNone(final["target_lane"])

    def test_unsafe_requested_lane_never_switches_to_safe_opposite_lane(self):
        for direction in ("LEFT", "RIGHT"):
            for action in ("keep_lane", "accelerate", "lane_change_left", "lane_change_right"):
                with self.subTest(direction=direction, proposal=action):
                    intent, world, alignment, risk = integration_documents(direction=direction)
                    risk["lane_change"][direction.lower()] = {
                        "is_safe": False, "reason_codes": ["rear_vehicle_ttc_low"],
                    }
                    canonical = build_control_decision(intent, world, alignment, risk)
                    final = gate_vla_proposal(proposal(action=action), canonical, risk)
                    self.assertEqual(final["decision_status"], "BLOCKED")
                    self.assertEqual(final["action"], "decelerate")
                    self.assertIsNone(final["target_lane"])
                    self.assertIn("rear_vehicle_ttc_low", final["blocked_reason_codes"])
                    self.assertEqual(validate_control_decision(final), [])

    def test_lane_change_resumes_only_after_requested_lane_becomes_safe(self):
        intent, world, alignment, risk = integration_documents()
        risk["lane_change"]["left"]["is_safe"] = False
        state, blocked = advance_vla_control_plan(intent, world, alignment, risk, proposal())
        self.assertNotIn(blocked["action"], {"lane_change_left", "lane_change_right"})
        risk["lane_change"]["left"]["is_safe"] = True
        _, ready = advance_vla_control_plan(
            intent, world, alignment, risk, proposal(action="keep_lane"), prior_state=state,
        )
        self.assertEqual(ready["action"], "lane_change_left")

    def test_unrequested_lane_change_is_rejected(self):
        for action in ("lane_change_left", "lane_change_right"):
            intent, world, alignment, risk = integration_documents(parser_action="KEEP_LANE")
            canonical = build_control_decision(intent, world, alignment, risk)
            final = gate_vla_proposal(proposal(action=action), canonical, risk)
            self.assertEqual(final["action"], "keep_lane")

    def test_deceleration_risk_overrides_safe_lane_command(self):
        intent, world, alignment, risk = integration_documents(recommended_action="decelerate")
        canonical = build_control_decision(intent, world, alignment, risk)
        final = gate_vla_proposal(proposal(), canonical, risk)
        self.assertEqual(final["action"], "decelerate")
        self.assertIsNone(final["target_lane"])

    def test_ready_canonical_cannot_bypass_missing_or_unsafe_lane_judgment(self):
        for judgment in ({}, {"is_safe": False}, {"is_safe": "true"}):
            intent, world, alignment, risk = integration_documents()
            canonical = build_control_decision(intent, world, alignment, risk)
            risk["lane_change"]["left"] = judgment
            final = gate_vla_proposal(proposal(), canonical, risk)
            self.assertEqual(final["action"], "decelerate")
            self.assertEqual(final["decision_status"], "BLOCKED")
            self.assertIsNone(final["target_lane"])
            self.assertEqual(validate_control_decision(final), [])

    def test_lane_change_speed_does_not_exceed_canonical_limit(self):
        intent, world, alignment, risk = integration_documents()
        canonical = build_control_decision(intent, world, alignment, risk)
        final = gate_vla_proposal(proposal(target_speed_kmh=60.0), canonical, risk)
        self.assertLessEqual(final["target_speed_kmh"], canonical["target_speed_kmh"])

    def test_risk_frame_must_match(self):
        intent, world, alignment, risk = integration_documents()
        canonical = build_control_decision(intent, world, alignment, risk)
        risk["frame_id"] = "old_frame"
        with self.assertRaisesRegex(ValueError, "frame_id mismatch"):
            gate_vla_proposal(proposal(), canonical, risk)

    def test_gate_does_not_mutate_inputs(self):
        intent, world, alignment, risk = integration_documents()
        canonical = build_control_decision(intent, world, alignment, risk)
        candidate = proposal(action="lane_change_right")
        before = copy.deepcopy((candidate, canonical, risk))
        gate_vla_proposal(candidate, canonical, risk)
        self.assertEqual((candidate, canonical, risk), before)


if __name__ == "__main__":
    unittest.main()
