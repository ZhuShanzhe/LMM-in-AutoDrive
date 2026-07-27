from __future__ import annotations

import sys
import unittest
from pathlib import Path

from lightweight_vla_adapter.src.safety_bridge import (
    advance_vla_control_plan,
    gate_vla_proposal,
)
from lightweight_vla_adapter.tests.fixtures import integration_documents, proposal
from scene_understanding.src.control_decision import build_control_decision


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
        self.assertEqual(decision["action"], "keep_lane")


if __name__ == "__main__":
    unittest.main()
