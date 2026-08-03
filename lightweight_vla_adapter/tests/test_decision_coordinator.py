from __future__ import annotations

import unittest

from lightweight_vla_adapter.src.decision_coordinator import (
    VLAFirstDecisionCoordinator,
)


def intent() -> dict:
    return {
        "request_id": "request-1",
        "parse_result": {"status": "VALID", "confidence": 0.94},
        "intent": {
            "steps": [
                {"step_id": "step-1", "action": "CHANGE_LANE"},
                {"step_id": "step-2", "action": "OVERTAKE"},
                {"step_id": "step-3", "action": "CHANGE_LANE"},
            ]
        },
    }


def world(frame: int = 1) -> dict:
    return {
        "frame_id": f"frame-{frame}",
        "timestamp_s": frame * 0.05,
        "ego": {"speed_mps": 8.0},
        "objects": [],
    }


def risk(*, left_safe: bool = True, recommended: str = "keep_lane") -> dict:
    return {
        "risk_level": "low",
        "reason_codes": [],
        "recommended_action": recommended,
        "lane_change": {
            "left": {"is_safe": left_safe, "reason_codes": []},
            "right": {"is_safe": True, "reason_codes": []},
        },
    }


def proposal(frame: int = 1, action: str = "lane_change_left") -> dict:
    return {
        "schema_version": "1.0.0",
        "request_id": "request-1",
        "frame_id": f"frame-{frame}",
        "action": action,
        "target_speed_kmh": 25.0,
        "target_lane": "left" if action == "lane_change_left" else None,
        "target_location": None,
        "target_entity_id": None,
        "confidence": 0.91,
        "model": "test-vla",
        "latency_ms": 18.0,
    }


class VLAFirstCoordinatorTests(unittest.TestCase):
    def test_vla_owns_nominal_lane_change(self):
        state, decision = VLAFirstDecisionCoordinator().coordinate(
            proposal(), intent(), world(), {}, risk()
        )
        self.assertEqual(decision["decision_status"], "READY")
        self.assertEqual(decision["action"], "lane_change_left")
        self.assertEqual(state["decision_source"], "vla")
        self.assertEqual(state["maneuver"]["phase"], "EXECUTING")

    def test_unsafe_lane_is_a_hard_constraint(self):
        state, decision = VLAFirstDecisionCoordinator().coordinate(
            proposal(), intent(), world(), {}, risk(left_safe=False)
        )
        self.assertEqual(decision["decision_status"], "SAFE_FALLBACK")
        self.assertEqual(decision["action"], "stop")
        self.assertIn("target_lane_left_unsafe", decision["blocked_reason_codes"])
        self.assertEqual(state["decision_source"], "safety_fallback")

    def test_action_must_match_active_semantic_step(self):
        _, decision = VLAFirstDecisionCoordinator().coordinate(
            proposal(action="turn_right"), intent(), world(), {}, risk()
        )
        self.assertEqual(decision["decision_status"], "SAFE_FALLBACK")
        self.assertIn(
            "vla_action_not_aligned_with_active_step",
            decision["blocked_reason_codes"],
        )

    def test_repeated_block_requests_dynamic_replan(self):
        coordinator = VLAFirstDecisionCoordinator()
        state = None
        for frame in range(1, 4):
            state, _ = coordinator.coordinate(
                proposal(frame), intent(), world(frame), {}, risk(left_safe=False)
            )
        assert state is not None
        self.assertTrue(state["replan_requested"])
        self.assertEqual(state["blocked_frames"], 3)

    def test_step_feedback_advances_compound_command(self):
        coordinator = VLAFirstDecisionCoordinator()
        coordinator.coordinate(proposal(), intent(), world(), {}, risk())
        state, decision = coordinator.coordinate(
            proposal(2, action="accelerate"),
            intent(),
            world(2),
            {},
            risk(),
            feedback={
                "request_id": "request-1",
                "step_id": "step-1",
                "outcome": "COMPLETED",
            },
        )
        self.assertEqual(state["active_step_id"], "step-2")
        self.assertEqual(decision["source_step_id"], "step-2")


if __name__ == "__main__":
    unittest.main()
