"""Build a deterministic, safety-gated control decision from integration JSON.

The module is the boundary between scene understanding and the CARLA control
layer.  It consumes parser intent, metric world state, semantic alignment and
risk assessment documents.  It emits exactly one flat action compatible with
``control.protocol.normalize_intent`` on the control branch.

Only the first intent step is selected here.  Sequencing several steps requires
a stateful executor and is intentionally outside this stateless JSON adapter.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from scene_understanding.src.high_level_driving_actions import evaluate_step_decision


CONTROL_DECISION_SCHEMA_VERSION = "1.0.0"
DECISION_STATUSES = {"READY", "BLOCKED", "SAFE_FALLBACK"}
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
PARSE_STATUSES = {"VALID", "NEEDS_CLARIFICATION", "UNSUPPORTED", "INVALID"}
RISK_LEVELS = {"none", "low", "medium", "high"}

DIRECT_ACTION_MAP = {
    "KEEP_LANE": "keep_lane",
    "STOP": "stop",
    "EMERGENCY_BRAKE": "emergency_brake",
    "RESUME": "keep_lane",
    "CANCEL": "keep_lane",
}


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _speed_kmh(world_state: Mapping[str, Any]) -> float:
    speed_mps = world_state.get("ego", {}).get("speed_mps", 0.0)
    if not _is_number(speed_mps) or float(speed_mps) < 0:
        raise ValueError("WorldState ego.speed_mps must be a finite non-negative number")
    return round(min(float(speed_mps) * 3.6, 100.0), 6)


def _parse_confidence(driving_intent: Mapping[str, Any]) -> float | None:
    value = driving_intent["parse_result"].get("confidence")
    if value is None:
        return None
    if not _is_number(value) or not 0 <= float(value) <= 1:
        raise ValueError("DrivingIntent parse_result.confidence must be null or between 0 and 1")
    return round(float(value), 6)


def _string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{path} must be an array of non-empty strings")
    return list(value)


def _validate_inputs(
    driving_intent: Any,
    world_state: Any,
    semantic_alignment: Any,
    risk_assessment: Any,
) -> None:
    documents = {
        "DrivingIntent": driving_intent,
        "WorldState": world_state,
        "SemanticAlignment": semantic_alignment,
        "RiskAssessment": risk_assessment,
    }
    for name, document in documents.items():
        if not isinstance(document, dict):
            raise ValueError(f"{name} must be a JSON object")

    request_id = driving_intent.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("DrivingIntent request_id must be a non-empty string")
    if semantic_alignment.get("request_id") != request_id:
        raise ValueError("request_id mismatch between DrivingIntent and SemanticAlignment")

    frame_id = world_state.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("WorldState frame_id must be a non-empty string")
    if semantic_alignment.get("world_state_frame_id") != frame_id:
        raise ValueError("frame_id mismatch between WorldState and SemanticAlignment")
    if risk_assessment.get("frame_id") != frame_id:
        raise ValueError("frame_id mismatch between WorldState and RiskAssessment")

    parse_result = driving_intent.get("parse_result")
    if not isinstance(parse_result, dict):
        raise ValueError("DrivingIntent parse_result must be an object")
    parse_status = parse_result.get("status")
    if parse_status not in PARSE_STATUSES:
        raise ValueError("DrivingIntent parse_result.status is invalid")
    if semantic_alignment.get("parse_status") != parse_status:
        raise ValueError("parse_status mismatch between DrivingIntent and SemanticAlignment")

    intent = driving_intent.get("intent")
    if not isinstance(intent, dict) or not isinstance(intent.get("steps"), list):
        raise ValueError("DrivingIntent intent.steps must be an array")
    if parse_status == "VALID" and not intent["steps"]:
        raise ValueError("VALID DrivingIntent must contain at least one step")

    alignments = semantic_alignment.get("step_alignments")
    if not isinstance(alignments, list):
        raise ValueError("SemanticAlignment step_alignments must be an array")
    if any(not isinstance(item, dict) for item in alignments):
        raise ValueError("SemanticAlignment step_alignments entries must be objects")

    if risk_assessment.get("risk_level") not in RISK_LEVELS:
        raise ValueError("RiskAssessment risk_level is invalid")
    if risk_assessment.get("recommended_action") not in {
        "maintain_speed", "monitor", "decelerate", "emergency_brake"
    }:
        raise ValueError("RiskAssessment recommended_action is invalid")
    _string_list(risk_assessment.get("reason_codes"), "RiskAssessment reason_codes")

    lane_change = risk_assessment.get("lane_change")
    if not isinstance(lane_change, dict):
        raise ValueError("RiskAssessment lane_change must be an object")
    for direction in ("left", "right"):
        judgment = lane_change.get(direction)
        if not isinstance(judgment, dict) or not isinstance(judgment.get("is_safe"), bool):
            raise ValueError(
                f"RiskAssessment lane_change.{direction}.is_safe must be a boolean"
            )
        _string_list(
            judgment.get("reason_codes"),
            f"RiskAssessment lane_change.{direction}.reason_codes",
        )


def _alignment_for_step(
    semantic_alignment: Mapping[str, Any], step_id: str
) -> Mapping[str, Any]:
    matches = [
        item
        for item in semantic_alignment["step_alignments"]
        if item.get("step_id") == step_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"SemanticAlignment must contain exactly one entry for step {step_id!r}"
        )
    return matches[0]


def _step_decision(
    step: Mapping[str, Any],
    alignment: Mapping[str, Any],
    risk_assessment: Mapping[str, Any],
    current_speed_kmh: float,
) -> dict[str, Any]:
    return evaluate_step_decision(
        step=step,
        alignment=alignment,
        risk_assessment=risk_assessment,
        current_speed_kmh=current_speed_kmh,
    )


def _decision(
    *,
    driving_intent: Mapping[str, Any],
    world_state: Mapping[str, Any],
    step: Mapping[str, Any] | None,
    status: str,
    action: str,
    target_speed_kmh: float,
    target_lane: str | None,
    target_location: dict[str, float] | None,
    reason: str,
    matched_entity_id: str | None,
    risk_assessment: Mapping[str, Any],
    blocked_reason_codes: list[str],
) -> dict[str, Any]:
    parse_result = driving_intent["parse_result"]
    result = {
        "schema_version": CONTROL_DECISION_SCHEMA_VERSION,
        "request_id": driving_intent["request_id"],
        "frame_id": world_state["frame_id"],
        "decision_status": status,
        "action": action,
        "target_speed_kmh": round(float(target_speed_kmh), 6),
        "target_lane": target_lane,
        "target_location": target_location,
        "emergency": action == "emergency_brake",
        "reason": reason,
        "parse_status": parse_result["status"],
        "parse_confidence": _parse_confidence(driving_intent),
        "source_step_id": None if step is None else step.get("step_id"),
        "source_step_action": None if step is None else step.get("action"),
        "source_step_count": len(driving_intent["intent"]["steps"]),
        "matched_entity_id": matched_entity_id,
        "risk_level": risk_assessment["risk_level"],
        "risk_reason_codes": list(risk_assessment["reason_codes"]),
        "blocked_reason_codes": blocked_reason_codes,
    }
    errors = validate_control_decision(result)
    if errors:
        raise ValueError("invalid ControlDecision: " + "; ".join(errors))
    return result


def build_control_decision(
    driving_intent: dict[str, Any],
    world_state: dict[str, Any],
    semantic_alignment: dict[str, Any],
    risk_assessment: dict[str, Any],
    *,
    source_step_id: str | None = None,
) -> dict[str, Any]:
    """Return one validated flat action after deterministic safety gating.

    By default the first step is selected for backward compatibility.  A
    stateful plan executor may pass ``source_step_id`` to evaluate a later
    step without mutating the DrivingIntent document.
    """

    _validate_inputs(driving_intent, world_state, semantic_alignment, risk_assessment)
    current_speed = _speed_kmh(world_state)
    parse_status = driving_intent["parse_result"]["status"]
    if parse_status != "VALID":
        return _decision(
            driving_intent=driving_intent,
            world_state=world_state,
            step=None,
            status="SAFE_FALLBACK",
            action="stop",
            target_speed_kmh=0.0,
            target_lane=None,
            target_location=None,
            reason=f"parse_status_{parse_status.lower()}",
            matched_entity_id=None,
            risk_assessment=risk_assessment,
            blocked_reason_codes=[f"parse_status_{parse_status.lower()}"],
        )

    steps = driving_intent["intent"]["steps"]
    if source_step_id is None:
        step = steps[0]
    else:
        selected = [item for item in steps if item.get("step_id") == source_step_id]
        if len(selected) != 1:
            raise ValueError(
                "DrivingIntent must contain exactly one step matching "
                f"source_step_id {source_step_id!r}"
            )
        step = selected[0]
    if not isinstance(step, dict):
        raise ValueError("DrivingIntent first step must be an object")
    step_id = step.get("step_id")
    if not isinstance(step_id, str) or not step_id:
        raise ValueError("DrivingIntent first step step_id must be a non-empty string")
    alignment = _alignment_for_step(semantic_alignment, step_id)
    matched_entity = alignment.get("matched_entity")
    matched_entity_id = (
        matched_entity.get("entity_id") if isinstance(matched_entity, dict) else None
    )

    decision = evaluate_step_decision(
        step=step,
        alignment=alignment,
        risk_assessment=risk_assessment,
        current_speed_kmh=current_speed,
    )
    return _decision(
        driving_intent=driving_intent,
        world_state=world_state,
        step=step,
        status=decision["status"],
        action=decision["action"],
        target_speed_kmh=decision["target_speed_kmh"],
        target_lane=decision["target_lane"],
        target_location=decision["target_location"],
        reason=decision["reason"],
        matched_entity_id=matched_entity_id,
        risk_assessment=risk_assessment,
        blocked_reason_codes=decision["blocked_reason_codes"],
    )


def validate_control_decision(data: Any) -> list[str]:
    """Return structural errors for the stable ControlDecision contract."""

    if not isinstance(data, dict):
        return ["root: expected an object"]
    expected = {
        "schema_version", "request_id", "frame_id", "decision_status",
        "action", "target_speed_kmh", "target_lane", "target_location",
        "emergency", "reason", "parse_status", "parse_confidence",
        "source_step_id", "source_step_action", "source_step_count",
        "matched_entity_id", "risk_level", "risk_reason_codes",
        "blocked_reason_codes",
    }
    errors: list[str] = []
    missing = sorted(expected - data.keys())
    extra = sorted(data.keys() - expected)
    if missing:
        errors.append("root: missing fields: " + ", ".join(missing))
    if extra:
        errors.append("root: unexpected fields: " + ", ".join(extra))
    if data.get("schema_version") != CONTROL_DECISION_SCHEMA_VERSION:
        errors.append("schema_version: expected '1.0.0'")
    for key in ("request_id", "frame_id", "reason"):
        if not isinstance(data.get(key), str) or not data[key]:
            errors.append(f"{key}: expected a non-empty string")
    if data.get("decision_status") not in DECISION_STATUSES:
        errors.append("decision_status: invalid value")
    if data.get("action") not in CONTROL_ACTIONS:
        errors.append("action: invalid value")
    speed = data.get("target_speed_kmh")
    if not _is_number(speed) or not 0 <= float(speed) <= 100:
        errors.append("target_speed_kmh: expected a finite number between 0 and 100")
    if data.get("target_lane") not in {None, "left", "right"}:
        errors.append("target_lane: expected null, 'left', or 'right'")
    location = data.get("target_location")
    if location is not None and (
        not isinstance(location, dict)
        or set(location) != {"x", "y", "z"}
        or any(not _is_number(location.get(key)) for key in ("x", "y", "z"))
    ):
        errors.append("target_location: expected null or finite x/y/z object")
    if not isinstance(data.get("emergency"), bool):
        errors.append("emergency: expected a boolean")
    elif data.get("emergency") != (data.get("action") == "emergency_brake"):
        errors.append("emergency: must be true exactly for emergency_brake")
    if data.get("parse_status") not in PARSE_STATUSES:
        errors.append("parse_status: invalid value")
    confidence = data.get("parse_confidence")
    if confidence is not None and (
        not _is_number(confidence) or not 0 <= float(confidence) <= 1
    ):
        errors.append("parse_confidence: expected null or a number between 0 and 1")
    for key in ("source_step_id", "source_step_action", "matched_entity_id"):
        value = data.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            errors.append(f"{key}: expected null or a non-empty string")
    count = data.get("source_step_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        errors.append("source_step_count: expected a non-negative integer")
    if data.get("risk_level") not in RISK_LEVELS:
        errors.append("risk_level: invalid value")
    for key in ("risk_reason_codes", "blocked_reason_codes"):
        value = data.get(key)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            errors.append(f"{key}: expected an array of non-empty strings")
        elif len(set(value)) != len(value):
            errors.append(f"{key}: values must be unique")
    return errors
