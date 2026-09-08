"""High-level DrivingIntent action mapping and deterministic safety helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping


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


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def fallback_action(on_blocked: Any) -> tuple[str, float | None, str]:
    policy = str(on_blocked or "SAFE_STOP").strip().upper()
    if policy in {"WAIT_FOR_SAFE", "WAIT", "SLOW_DOWN"}:
        return "decelerate", None, "wait_for_safe"
    if policy in {"SKIP", "SKIP_STEP", "CONTINUE"}:
        return "keep_lane", None, "skip_blocked_step"
    return "stop", 0.0, "safe_stop"


def map_step_action(
    step: Mapping[str, Any],
    current_speed_kmh: float,
) -> tuple[str, float, str | None, dict[str, float] | None]:
    """Map every DrivingIntent 1.2 action to the stable CARLA control protocol."""

    parser_action = str(step.get("action", "")).strip().upper()
    parameters = step.get("parameters") or {}
    if not isinstance(parameters, Mapping):
        raise ValueError(
            f"DrivingIntent step {step.get('step_id')!r} parameters must be an object"
        )

    action = {
        "KEEP_LANE": "keep_lane",
        "STOP": "stop",
        "WAIT": "stop",
        "FOLLOW": "keep_lane",
        "APPROACH": "decelerate",
        "NAVIGATE_TO": "keep_lane",
        "U_TURN": "turn_left",
        "PROCEED": "keep_lane",
        "YIELD": "decelerate",
        "PARK": "stop",
        "PASS_BY": "decelerate",
        "REVERSE": "stop",
        "ENTER_AREA": "keep_lane",
        "EXIT_AREA": "keep_lane",
        "EMERGENCY_BRAKE": "emergency_brake",
        "RESUME": "keep_lane",
        "CANCEL": "keep_lane",
    }.get(parser_action)
    target_speed_kmh = float(current_speed_kmh)
    target_lane: str | None = None
    target_location: dict[str, float] | None = None

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
    elif parser_action in {"CHANGE_LANE", "MERGE"}:
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
    elif parser_action == "OVERTAKE":
        direction = str(parameters.get("direction", "")).strip().upper()
        if direction in {"LEFT", "RIGHT"}:
            target_lane = direction.lower()
            action = f"lane_change_{target_lane}"
        else:
            action = "accelerate"
    elif parser_action in {"PULL_OVER", "AVOID", "PASS_BY"}:
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


def validate_risk_assessment(risk_assessment: Mapping[str, Any]) -> None:
    if risk_assessment.get("recommended_action") not in {
        "maintain_speed",
        "monitor",
        "decelerate",
        "emergency_brake",
    }:
        raise ValueError("RiskAssessment recommended_action is invalid")
    if risk_assessment.get("risk_level") not in {"none", "low", "medium", "high"}:
        raise ValueError("RiskAssessment risk_level is invalid")
    if not isinstance(risk_assessment.get("reason_codes"), list):
        raise ValueError("RiskAssessment reason_codes must be an array")
    lane_change = risk_assessment.get("lane_change")
    if not isinstance(lane_change, Mapping):
        raise ValueError("RiskAssessment lane_change must be an object")
    for direction in ("left", "right"):
        judgment = lane_change.get(direction)
        if not isinstance(judgment, Mapping) or not isinstance(
            judgment.get("is_safe"), bool
        ):
            raise ValueError(
                f"RiskAssessment lane_change.{direction}.is_safe must be a boolean"
            )
        if not isinstance(judgment.get("reason_codes"), list):
            raise ValueError(
                f"RiskAssessment lane_change.{direction}.reason_codes must be an array"
            )
