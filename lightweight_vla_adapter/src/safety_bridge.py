from __future__ import annotations

import copy
from typing import Any, Mapping

from scene_understanding.src.control_decision import validate_control_decision
from scene_understanding.src.control_plan_executor import advance_control_plan

from .contracts import validate_vla_proposal


def _canonical(
    decision: Mapping[str, Any], reason_code: str
) -> dict[str, Any]:
    result = copy.deepcopy(dict(decision))
    reasons = list(result.get("blocked_reason_codes", []))
    if reason_code not in reasons:
        reasons.append(reason_code)
    result["blocked_reason_codes"] = reasons
    return result


def gate_vla_proposal(
    proposal: dict[str, Any],
    canonical_decision: dict[str, Any],
    risk_assessment: dict[str, Any],
) -> dict[str, Any]:
    """Apply only safety-critical constraints to a learned proposal.

    Semantic command alignment is deliberately evaluated in the model output;
    it is not repaired by a rule template at inference time.
    """

    proposal_errors = validate_vla_proposal(proposal)
    if proposal_errors:
        raise ValueError("invalid VLADecisionProposal: " + "; ".join(proposal_errors))
    canonical_errors = validate_control_decision(canonical_decision)
    if canonical_errors:
        raise ValueError("invalid canonical ControlDecision: " + "; ".join(canonical_errors))
    if proposal["request_id"] != canonical_decision["request_id"]:
        raise ValueError("request_id mismatch between proposal and ControlDecision")
    if proposal["frame_id"] != canonical_decision["frame_id"]:
        raise ValueError("frame_id mismatch between proposal and ControlDecision")

    recommended = risk_assessment.get("recommended_action")
    if recommended == "emergency_brake":
        return _canonical(canonical_decision, "vla_overridden_by_emergency_risk")
    if recommended == "decelerate" and proposal["action"] not in {
        "decelerate",
        "stop",
        "emergency_brake",
    }:
        return _canonical(canonical_decision, "vla_overridden_by_deceleration_risk")
    if canonical_decision["decision_status"] != "READY":
        return _canonical(canonical_decision, "vla_blocked_by_canonical_gate")
    proposed_action = proposal["action"]
    if proposed_action in {"lane_change_left", "lane_change_right"}:
        direction = proposed_action.removeprefix("lane_change_")
        judgment = risk_assessment.get("lane_change", {}).get(direction, {})
        if judgment.get("is_safe") is not True:
            return _canonical(canonical_decision, f"vla_{direction}_lane_unsafe")

    result = copy.deepcopy(canonical_decision)
    result["action"] = proposed_action
    result["target_speed_kmh"] = round(
        min(max(float(proposal["target_speed_kmh"]), 0.0), 60.0),
        6,
    )
    if proposed_action in {"stop", "emergency_brake"}:
        result["target_speed_kmh"] = 0.0
    result["target_lane"] = (
        proposed_action.removeprefix("lane_change_")
        if proposed_action.startswith("lane_change_")
        else None
    )
    if proposed_action not in {"turn_left", "turn_right"}:
        result["target_location"] = None
    result["emergency"] = proposed_action == "emergency_brake"
    result["reason"] = f"vla_accepted_{proposal['model']}"
    result["blocked_reason_codes"] = []
    errors = validate_control_decision(result)
    if errors:
        raise ValueError("invalid gated ControlDecision: " + "; ".join(errors))
    return result


def advance_vla_control_plan(
    driving_intent: dict[str, Any],
    world_state: dict[str, Any],
    semantic_alignment: dict[str, Any],
    risk_assessment: dict[str, Any],
    proposal: dict[str, Any],
    *,
    prior_state: dict[str, Any] | None = None,
    feedback: dict[str, Any] | None = None,
    planner_target_location: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Advance the existing FSM and safety-gate one learned model proposal."""

    state, canonical_decision = advance_control_plan(
        driving_intent,
        world_state,
        semantic_alignment,
        risk_assessment,
        prior_state=prior_state,
        feedback=feedback,
        planner_target_location=planner_target_location,
    )
    final_decision = gate_vla_proposal(
        proposal,
        canonical_decision,
        risk_assessment,
    )
    return state, final_decision
