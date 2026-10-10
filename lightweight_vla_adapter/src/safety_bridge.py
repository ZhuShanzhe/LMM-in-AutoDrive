from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from scene_understanding.src.control_decision import validate_control_decision
from scene_understanding.src.control_plan_executor import advance_control_plan

from .contracts import validate_vla_proposal
from .risk_behavior import cautious_permission


RUNTIME_FIELDS = frozenset({
    'allow_positive_acceleration', 'longitudinal_control_mode',
    'passing_lane_established', 'deferred_lane_change',
    'deferred_current_lane_evidence', 'risk_motion_permission',
})


def validate_runtime_control_decision(decision, risk):
    """Keep the flat wire contract strict and validate runtime authority separately."""
    core = {key: value for key, value in decision.items() if key not in RUNTIME_FIELDS}
    errors = validate_control_decision(core)
    for key in ('allow_positive_acceleration', 'passing_lane_established', 'deferred_lane_change'):
        if key in decision and not isinstance(decision[key], bool):
            errors.append(key + ': must be a boolean')
    if ('longitudinal_control_mode' in decision and
            decision['longitudinal_control_mode'] not in {'STOP', 'CAUTIOUS_SPEED_CAP'}):
        errors.append('longitudinal_control_mode: unsupported runtime mode')
    if 'passing_lane_established' in decision and decision.get('source_step_action') not in {'OVERTAKE','PASS_BY'}:
        errors.append('passing_lane_established: requires OVERTAKE or PASS_BY step')
    deferred = decision.get('deferred_lane_change') is True
    if ('deferred_current_lane_evidence' in decision) != deferred:
        errors.append('deferred_current_lane_evidence: requires deferred navigation')
    if deferred:
        evidence = decision.get('deferred_current_lane_evidence') or {}
        now = (risk.get('risk_evidence') or {}).get('decision_timestamp_s')
        observed = evidence.get('observed_at_s')
        expires = evidence.get('valid_until_s')
        numeric = lambda value: (isinstance(value, (int, float)) and
                                 not isinstance(value, bool) and math.isfinite(value))
        speed_limit = 10.
        condition_ready=evidence.get('condition')=='PATH_CLEAR'
        if evidence.get('condition')=='CURRENT_LANE_MOTION':
            policy=risk.get('behavior_policy') or {};lidar=policy.get('lidar') or {}
            radar=risk.get('physical_forward_radar') or {};sensor=risk.get('risk_evidence') or {}
            required=evidence.get('required_clearance_m');physical=policy.get('current_speed_required_clearance_m')
            clear=lidar.get('clear_distance_m');nearest=radar.get('nearest_distance_m');cap=policy.get('max_speed_kmh')
            condition_ready=(evidence.get('semantic_scope')=='bounded_current_lane_motion_not_semantic_path_clear'
                and all(numeric(v) for v in (required,physical,clear,cap))
                and required>=.5 and math.isclose(required,physical,rel_tol=0.,abs_tol=1e-6)
                and clear>=required and sensor.get('is_fresh') is True and lidar.get('valid') is True
                and evidence.get('sensor_frame')==lidar.get('sensor_frame')==radar.get('sensor_frame')==sensor.get('sensor_frame')
                and evidence.get('observed_at_s')==lidar.get('observed_at_s')
                and policy.get('disposition') in {'CLEAR','CAUTIOUS_PROCEED'}
                and not policy.get('physical_stop') and not policy.get('absolute_stop_active')
                and numeric(decision.get('target_speed_kmh')) and decision['target_speed_kmh']<=cap
                and (radar.get('obstacle_candidate_count')==0 and nearest is None or numeric(nearest) and nearest>required))
            target = decision.get('target_speed_kmh')
            if condition_ready and numeric(target) and target > 10.:
                from .risk_behavior import stopping_clearance
                approach_clearance = stopping_clearance(target)
                if (clear >= approach_clearance
                        and (radar.get('obstacle_candidate_count') == 0 and nearest is None
                             or numeric(nearest) and nearest > approach_clearance)):
                    speed_limit = min(30., cap)
        if not (decision.get('source_step_action') == 'CHANGE_LANE'
                and decision.get('decision_status') == 'READY'
                and decision.get('action') == 'keep_lane'
                and decision.get('target_lane') is None and decision.get('target_location') is None
                and numeric(decision.get('target_speed_kmh'))
                and 0 < decision['target_speed_kmh'] <= speed_limit
                and not decision.get('emergency')
                and evidence.get('request_id') == decision.get('request_id')
                and evidence.get('frame_id') == decision.get('frame_id') == risk.get('frame_id')
                and condition_ready
                and evidence.get('source') == 'fresh_lidar_radar_driving_corridor'
                and evidence.get('valid') is True and evidence.get('satisfied') is True
                and all(numeric(value) for value in (now, observed, expires))
                and 0 <= now - observed <= .25 and now <= expires):
            errors.append('deferred_lane_change: invalid bounded current-lane evidence')
    if 'risk_motion_permission' in decision:
        actual = cautious_permission(risk, frame_id=decision.get('frame_id'))
        if not deferred or actual is None or decision['risk_motion_permission'] != actual:
            errors.append('risk_motion_permission: must match independently verified current risk')
    if decision.get('allow_positive_acceleration') is True:
        policy = risk.get('behavior_policy') or {}
        if (decision.get('emergency') or decision.get('action') in {'stop', 'emergency_brake'}
                or policy.get('physical_stop') or policy.get('absolute_stop_active')
                or policy.get('allow_positive_acceleration') is False):
            errors.append('allow_positive_acceleration: conflicts with stop/slowdown authority')
    return errors


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
    """Prioritize risk gates, then enforce the active lane-change instruction.

    Keep raw proposals for model evaluation: a canonical fallback is not a
    correct model prediction. The FSM remains authoritative for active steps.
    """

    proposal_errors = validate_vla_proposal(proposal)
    if proposal_errors:
        raise ValueError("invalid VLADecisionProposal: " + "; ".join(proposal_errors))
    canonical_errors = validate_runtime_control_decision(canonical_decision, risk_assessment)
    if canonical_errors:
        raise ValueError("invalid canonical ControlDecision: " + "; ".join(canonical_errors))
    if proposal["request_id"] != canonical_decision["request_id"]:
        raise ValueError("request_id mismatch between proposal and ControlDecision")
    if proposal["frame_id"] != canonical_decision["frame_id"]:
        raise ValueError("frame_id mismatch between proposal and ControlDecision")
    if risk_assessment.get("frame_id") != canonical_decision["frame_id"]:
        raise ValueError("frame_id mismatch between risk and ControlDecision")

    recommended = risk_assessment.get("recommended_action")
    if recommended == "emergency_brake":
        return _canonical(canonical_decision, "vla_overridden_by_emergency_risk")
    if canonical_decision.get('deferred_lane_change') is True:
        # Deferred means current-lane motion only, never permission to change lanes.
        bounded = copy.deepcopy(canonical_decision)
        policy = risk_assessment.get('behavior_policy') or {}
        if (policy.get('physical_stop') or policy.get('absolute_stop_active')
                or recommended == 'decelerate' or proposal['action'] in {'stop', 'emergency_brake'}):
            bounded.update(action=proposal['action'] if proposal['action'] in {'stop', 'emergency_brake'}
                           else 'stop', target_speed_kmh=0.,
                           emergency=proposal['action'] == 'emergency_brake',
                           allow_positive_acceleration=False, longitudinal_control_mode='STOP')
        return bounded
    from .task_feasibility import task_permission
    if (canonical_decision['action'] in {'lane_change_left','lane_change_right'}
            and proposal['action'] not in {'stop','emergency_brake'}
            and canonical_decision['decision_status']=='READY'
            and not canonical_decision.get('blocked_reason_codes')
            and not canonical_decision.get('emergency')
            and task_permission(risk_assessment,'CHANGE_LANE',frame_id=proposal['frame_id'])
            and task_permission(risk_assessment,'PATH_CLEAR',frame_id=proposal['frame_id'])):
        # Speed advisories cannot erase a sensor-authorized lateral task.
        # Keep this rule-produced decision separate from the raw model output.
        bounded=copy.deepcopy(canonical_decision)
        cap=(risk_assessment.get('lane_change') or {}).get(
            canonical_decision['action'].removeprefix('lane_change_'),{}).get('speed_cap_kmh',8.)
        bounded['target_speed_kmh']=min(float(bounded['target_speed_kmh']),float(cap or 8.))
        bounded['reason']='task_feasibility_bounded_lane_execution'
        return bounded
    if recommended == "decelerate" and proposal["action"] not in {
        "decelerate",
        "stop",
        "emergency_brake",
    }:
        return _canonical(canonical_decision, "vla_overridden_by_deceleration_risk")
    if canonical_decision["decision_status"] != "READY":
        return _canonical(canonical_decision, "vla_blocked_by_canonical_gate")
    permission = cautious_permission(risk_assessment, frame_id=proposal['frame_id'])
    if (permission and permission['schema_version'] == 'cautious_motion/2.0'
            and canonical_decision['action'] in {'accelerate', 'decelerate', 'keep_lane', 'turn_left', 'turn_right'}
            and not canonical_decision.get('blocked_reason_codes')
            and not canonical_decision.get('emergency')):
        bounded = copy.deepcopy(canonical_decision)
        bounded['target_speed_kmh'] = min(float(bounded['target_speed_kmh']), permission['max_speed_kmh'])
        bounded['reason'] = 'risk_bounded_instruction_execution'
        return bounded
    proposed_action = proposal["action"]
    canonical_action = canonical_decision["action"]
    lane_actions = {"lane_change_left", "lane_change_right"}
    navigation_actions = lane_actions | {"turn_left", "turn_right"}
    braking_actions = {"decelerate", "stop", "emergency_brake"}
    longitudinal_turn = (canonical_action in {'turn_left','turn_right'}
                         and (proposed_action in {'accelerate','keep_lane'} or (
                             proposed_action == 'decelerate'
                             and cautious_permission(risk_assessment,frame_id=proposal['frame_id']) is not None)))
    if longitudinal_turn:
        # A longitudinal proposal can change speed, not the active route turn.
        # Keep this adaptation distinct from an unmodified model prediction.
        proposed_action=canonical_action
    if canonical_action in lane_actions:
        direction = canonical_action.removeprefix("lane_change_")
        judgment = risk_assessment.get("lane_change", {}).get(direction, {})
        if judgment.get("is_safe") is not True:
            # Also fail closed if a caller supplied a READY canonical decision
            # inconsistent with the current lane assessment.
            blocked = _canonical(canonical_decision, f"vla_{direction}_lane_unsafe")
            blocked.update(
                decision_status="BLOCKED",
                action="decelerate",
                target_lane=None,
                target_location=None,
                emergency=False,
                reason=f"vla_{direction}_lane_unsafe_wait_for_safe",
            )
            return blocked
    if (canonical_action in navigation_actions or proposed_action in navigation_actions) and (
        proposed_action != canonical_action
    ) and proposed_action not in braking_actions:
        return _canonical(canonical_decision, "vla_incompatible_with_active_intent")

    result = copy.deepcopy(canonical_decision)
    result["action"] = proposed_action
    result["target_speed_kmh"] = round(
        min(
            max(float(proposal["target_speed_kmh"]), 0.0),
            60.0,
            float(canonical_decision["target_speed_kmh"])
            if proposed_action in lane_actions else 60.0,
        ),
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
    if longitudinal_turn:
        result['reason']='vla_longitudinal_only_turn_target_preserved'
    result["blocked_reason_codes"] = []
    # A braking model output retires earlier positive-motion authority.
    if result['action'] in {'stop', 'emergency_brake'} and 'allow_positive_acceleration' in result:
        result['allow_positive_acceleration'] = False
        result['longitudinal_control_mode'] = 'STOP'
    errors = validate_runtime_control_decision(result, risk_assessment)
    if errors:
        raise ValueError("invalid gated ControlDecision: " + "; ".join(errors))
    return result


def enforce_final_lane_policy(
    decision: dict[str, Any],
    canonical_decision: dict[str, Any],
    risk_assessment: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Prevent downstream resume/escape policies from bypassing lane gates."""
    if decision.get("action") not in {"lane_change_left", "lane_change_right"}:
        return copy.deepcopy(decision), None
    proposal = {
        "schema_version": "1.0.0",
        "request_id": decision["request_id"],
        "frame_id": decision["frame_id"],
        "action": decision["action"],
        "target_speed_kmh": decision["target_speed_kmh"],
        "target_lane": decision["action"].removeprefix("lane_change_"),
        "target_location": None,
        "target_entity_id": decision.get("matched_entity_id"),
        "confidence": 1.0,
        "model": "final_lane_policy_guard",
        "latency_ms": 0.0,
    }
    guarded = gate_vla_proposal(proposal, canonical_decision, risk_assessment)
    if guarded["action"] != decision["action"]:
        return guarded, "final_lane_policy_guard"
    result = copy.deepcopy(decision)
    result["target_speed_kmh"] = guarded["target_speed_kmh"]
    return result, None


def advance_vla_control_plan(
    driving_intent: dict[str, Any],
    world_state: dict[str, Any],
    semantic_alignment: dict[str, Any],
    risk_assessment: dict[str, Any],
    proposal: dict[str, Any],
    *,
    prior_state: dict[str, Any] | None = None,
    feedback: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Advance the existing FSM and safety-gate one learned model proposal."""

    state, canonical_decision = advance_control_plan(
        driving_intent,
        world_state,
        semantic_alignment,
        risk_assessment,
        prior_state=prior_state,
        feedback=feedback,
    )
    final_decision = gate_vla_proposal(
        proposal,
        canonical_decision,
        risk_assessment,
    )
    return state, final_decision
