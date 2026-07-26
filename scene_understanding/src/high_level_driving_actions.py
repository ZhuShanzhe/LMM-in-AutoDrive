"""High-level driving action utilities for scene understanding decision logic."""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

CONTROL_ACTIONS = {
    "keep_lane",
    "accelerate",
    "decelerate",
    "stop",
    "emergency_brake",
    "lane_change_left",
    "lane_change_right",
    "turn_left",
    "turn_right",
}

POLICY_WAIT = {"WAIT_FOR_SAFE", "WAIT", "SLOW_DOWN"}
POLICY_SKIP = {"SKIP", "SKIP_STEP", "CONTINUE"}

RISK_RECOMMENDED_ACTIONS = {
    "maintain_speed",
    "monitor",
    "decelerate",
    "emergency_brake",
}


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _current_speed_kmh(world_state: Mapping[str, Any]) -> float:
    speed_mps = world_state.get("ego", {}).get("speed_mps", 0.0)
    if not _is_number(speed_mps) or float(speed_mps) < 0:
        raise ValueError("WorldState ego.speed_mps must be a finite non-negative number")
    return round(min(float(speed_mps) * 3.6, 100.0), 6)


def fallback_action(on_blocked: Any) -> Tuple[str, float | None, str]:
    policy = str(on_blocked or "SAFE_STOP").strip().upper()
    if policy in POLICY_WAIT:
        return "decelerate", None, "wait_for_safe"
    if policy in POLICY_SKIP:
        return "keep_lane", None, "skip_blocked_step"
    return "stop", 0.0, "safe_stop"


def map_step_action(
    step: Mapping[str, Any], current_speed_kmh: float
) -> Tuple[str, float, str | None, dict[str, float] | None]:
    parser_action = str(step.get("action", "")).strip().upper()
    parameters = step.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError(
            f"DrivingIntent step {step.get('step_id')!r} parameters must be an object"
        )

    action = {
        "KEEP_LANE": "keep_lane",
        "STOP": "stop",
        "EMERGENCY_BRAKE": "emergency_brake",
        "RESUME": "keep_lane",
        "CANCEL": "keep_lane",
    }.get(parser_action)
    target_speed_kmh = current_speed_kmh
    target_lane = None
    target_location = None

    if parser_action == "SET_SPEED":
        value = parameters.get("target_speed_mps")
        if not _is_number(value) or float(value) < 0:
            raise ValueError("SET_SPEED requires finite non-negative target_speed_mps")
        action = "keep_lane"
        target_speed_kmh = round(min(float(value) * 3.6, 100.0), 6)
    elif parser_action == "ADJUST_SPEED":
        change = str(parameters.get("change", "HOLD")).strip().upper()
        action = {
            "INCREASE": "accelerate",
            "DECREASE": "decelerate",
            "HOLD": "keep_lane",
        }.get(change)
    elif parser_action == "CHANGE_LANE":
        direction = str(parameters.get("direction", "")).strip().upper()
        if direction in {"LEFT", "RIGHT"}:
            target_lane = direction.lower()
            action = f"lane_change_{target_lane}"
    elif parser_action == "TURN":
        direction = str(parameters.get("direction", "")).strip().upper()
        if direction == "STRAIGHT":
            action = "keep_lane"
        elif direction in {"LEFT", "RIGHT"}:
            action = f"turn_{direction.lower()}"
        location = parameters.get("target_location")
        if location is not None:
            if not isinstance(location, dict) or not all(
                key in location and _is_number(location[key]) for key in ("x", "y")
            ):
                raise ValueError("TURN target_location requires finite x and y")
            target_location = {
                "x": float(location["x"]),
                "y": float(location["y"]),
                "z": float(location.get("z", 0.0)),
            }
    elif parser_action == "OVERTAKE":
        direction = str(parameters.get("direction", "")).strip().upper()
        if direction in {"LEFT", "RIGHT"}:
            target_lane = direction.lower()
            action = f"lane_change_{target_lane}"
        else:
            action = "accelerate"
    elif parser_action in {"YIELD", "PULL_OVER", "AVOID"}:
        direction = str(parameters.get("direction", "")).strip().upper()
        if direction in {"LEFT", "RIGHT"}:
            target_lane = direction.lower()
            action = f"lane_change_{target_lane}"
        else:
            action = "decelerate"

    if action not in CONTROL_ACTIONS:
        raise ValueError(f"unsupported DrivingIntent action {parser_action!r}")
    if action in {"stop", "emergency_brake"}:
        target_speed_kmh = 0.0
    return action, target_speed_kmh, target_lane, target_location


def alignment_requires_action(alignment: Mapping[str, Any]) -> bool:
    return alignment.get("alignment_required") is True


def alignment_successful(alignment: Mapping[str, Any]) -> bool:
    return alignment.get("alignment_success") is True


def blocked_alignment_reason(alignment: Mapping[str, Any]) -> str:
    return str(alignment.get("reason_code") or "target_not_aligned")


def lane_change_safe(
    action: str, risk_assessment: Mapping[str, Any]
) -> Tuple[bool, list[str]]:
    direction = action.removeprefix("lane_change_")
    judgment = risk_assessment["lane_change"][direction]
    return judgment["is_safe"], list(judgment["reason_codes"])


def validate_risk_assessment(risk_assessment: Mapping[str, Any]) -> None:
    if risk_assessment.get("recommended_action") not in RISK_RECOMMENDED_ACTIONS:
        raise ValueError("RiskAssessment recommended_action is invalid")
    if risk_assessment.get("risk_level") not in {"none", "low", "medium", "high"}:
        raise ValueError("RiskAssessment risk_level is invalid")
    if not isinstance(risk_assessment.get("reason_codes"), list):
        raise ValueError("RiskAssessment reason_codes must be an array")
    lane_change = risk_assessment.get("lane_change")
    if not isinstance(lane_change, dict):
        raise ValueError("RiskAssessment lane_change must be an object")
    for direction in ("left", "right"):
        judgment = lane_change.get(direction)
        if not isinstance(judgment, dict) or not isinstance(judgment.get("is_safe"), bool):
            raise ValueError(
                f"RiskAssessment lane_change.{direction}.is_safe must be a boolean"
            )
        if not isinstance(judgment.get("reason_codes"), list):
            raise ValueError(
                f"RiskAssessment lane_change.{direction}.reason_codes must be an array"
            )


def evaluate_step_decision(
    step: Mapping[str, Any],
    alignment: Mapping[str, Any],
    risk_assessment: Mapping[str, Any],
    current_speed_kmh: float,
) -> dict[str, Any]:
    validate_risk_assessment(risk_assessment)
    action, target_speed_kmh, target_lane, target_location = map_step_action(
        step, current_speed_kmh
    )
    parser_action = str(step.get("action", "")).strip().upper()
    blocked_reason_codes: list[str] = []

    if action in {"stop", "emergency_brake"}:
        return {
            "status": "READY",
            "action": action,
            "target_speed_kmh": target_speed_kmh,
            "target_lane": None,
            "target_location": None,
            "reason": f"driving_intent_{parser_action.lower()}",
            "blocked_reason_codes": [],
        }

    recommended = risk_assessment["recommended_action"]
    if recommended == "emergency_brake":
        return {
            "status": "BLOCKED",
            "action": "emergency_brake",
            "target_speed_kmh": 0.0,
            "target_lane": None,
            "target_location": None,
            "reason": "risk_requires_emergency_brake",
            "blocked_reason_codes": ["risk_requires_emergency_brake"],
        }

    if recommended == "decelerate" and action not in {"decelerate", "stop"}:
        return {
            "status": "BLOCKED",
            "action": "decelerate",
            "target_speed_kmh": current_speed_kmh,
            "target_lane": None,
            "target_location": None,
            "reason": "risk_requires_deceleration",
            "blocked_reason_codes": ["risk_requires_deceleration"],
        }

    if alignment_requires_action(alignment) and not alignment_successful(alignment):
        fallback_action_name, fallback_speed, policy_reason = fallback_action(
            step.get("on_blocked")
        )
        reason_code = blocked_alignment_reason(alignment)
        return {
            "status": "BLOCKED",
            "action": fallback_action_name,
            "target_speed_kmh": (
                current_speed_kmh if fallback_speed is None else fallback_speed
            ),
            "target_lane": None,
            "target_location": None,
            "reason": f"{reason_code}_{policy_reason}",
            "blocked_reason_codes": [reason_code, policy_reason],
        }

    if action in {"lane_change_left", "lane_change_right"}:
        safe, lane_reasons = lane_change_safe(action, risk_assessment)
        if not safe:
            fallback_action_name, fallback_speed, policy_reason = fallback_action(
                step.get("on_blocked")
            )
            return {
                "status": "BLOCKED",
                "action": fallback_action_name,
                "target_speed_kmh": (
                    current_speed_kmh if fallback_speed is None else fallback_speed
                ),
                "target_lane": None,
                "target_location": None,
                "reason": f"lane_change_{action.removeprefix('lane_change_')}_blocked_{policy_reason}",
                "blocked_reason_codes": (
                    lane_reasons or [f"lane_change_{action.removeprefix('lane_change_')}_unsafe"]
                ),
            }

    if action in {"turn_left", "turn_right"} and target_location is None:
        fallback_action_name, fallback_speed, policy_reason = fallback_action(
            step.get("on_blocked")
        )
        return {
            "status": "BLOCKED",
            "action": fallback_action_name,
            "target_speed_kmh": (
                current_speed_kmh if fallback_speed is None else fallback_speed
            ),
            "target_lane": None,
            "target_location": None,
            "reason": f"turn_target_location_missing_{policy_reason}",
            "blocked_reason_codes": ["turn_target_location_missing", policy_reason],
        }

    return {
        "status": "READY",
        "action": action,
        "target_speed_kmh": target_speed_kmh,
        "target_lane": target_lane,
        "target_location": target_location,
        "reason": f"driving_intent_{parser_action.lower()}",
        "blocked_reason_codes": [],
    }


def classify_scenario(
    action: str, risk_assessment: Mapping[str, Any], alignment: Mapping[str, Any]
) -> str:
    if risk_assessment["recommended_action"] == "emergency_brake":
        return "emergency_response"
    if action in {"lane_change_left", "lane_change_right"}:
        return "complex_avoidance"
    if action in {"stop", "decelerate"} and alignment_requires_action(alignment):
        return "complex_avoidance"
    return "basic_control"
