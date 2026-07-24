"""Align structured DrivingIntent targets with entities in a WorldState.

This module is the JSON-file integration boundary between the command parser
and scene understanding.  It never turns language-model output into vehicle
controls.  A target that is absent or unsupported remains explicitly
unmatched so the downstream decision module can apply ``on_blocked`` safely.
"""

from __future__ import annotations

from typing import Any, Mapping

from scene_understanding.core.object_matcher import relative_position_label, select_world_object
from scene_understanding.core.risk_assessment import assess_world_state, distance_risk_level
from scene_understanding.core.world_state import validate_world_state


DRIVING_INTENT_SCHEMA_VERSION = "1.1.0"
SUPPORTED_DRIVING_INTENT_SCHEMA_VERSIONS = {
    "1.0.0",
    DRIVING_INTENT_SCHEMA_VERSION,
}
ALIGNMENT_SCHEMA_VERSION = "1.0.0"
PARSE_STATUSES = {"VALID", "NEEDS_CLARIFICATION", "UNSUPPORTED", "INVALID"}
ALIGNMENT_STATUSES = {"COMPLETE", "PARTIAL", "FAILED", "NOT_REQUIRED", "SKIPPED"}

TARGET_TYPE_MAP = {
    "PEDESTRIAN": "pedestrian",
    "CYCLIST": "cyclist",
    "SLOW_VEHICLE": "slow_vehicle",
    "VEHICLE": "vehicle",
    "OBSTACLE": "obstacle",
    "TRAFFIC_CONE": "traffic_cone",
    "ROAD_HAZARD": "road_hazard",
    "TRAFFIC_LIGHT": "traffic_light",
    "TRAFFIC_SIGN": "traffic_sign",
    "JUNCTION": "junction",
}

# These target types are valid in DrivingIntent 1.1, but the current
# WorldState contract does not expose corresponding map or route entities.
# Keep them distinct from genuinely unknown target types so downstream
# modules can wait, request another capability, or stop safely without
# treating a valid parser result as malformed.
WORLD_STATE_CAPABILITY_UNAVAILABLE_TARGET_TYPES = {
    "AREA",
    "CONSTRUCTION_ZONE",
    "COORDINATE",
    "CROSSWALK",
    "CURB",
    "DESTINATION",
    "DROPOFF_POINT",
    "LANDMARK",
    "PARKING_AREA",
    "PARKING_SPACE",
    "PICKUP_POINT",
    "ROAD",
    "STOP_LINE",
}
WORLD_STATE_CAPABILITY_UNAVAILABLE_TARGET = (
    "world_state_capability_unavailable"
)

RELATION_HINTS = {
    "AHEAD": ("front", "ego_lane"),
    "BEHIND": ("rear", "unknown"),
    "LEFT": ("left", "unknown"),
    "RIGHT": ("right", "unknown"),
    "FRONT_LEFT": ("front_left", "unknown"),
    "FRONT_RIGHT": ("front_right", "unknown"),
    "REAR_LEFT": ("rear_left", "unknown"),
    "REAR_RIGHT": ("rear_right", "unknown"),
    "AHEAD_CROSSING": ("front", "crossing_ego_path"),
    "ADJACENT": ("unknown", "unknown"),
    "NEXT_TO": ("unknown", "unknown"),
    "IN_FRONT_OF": ("front", "unknown"),
    "NEAR": ("unknown", "unknown"),
    "AT_JUNCTION": ("front", "ego_lane"),
    "NEAR_DESTINATION": ("front", "unknown"),
    "INSIDE": ("unknown", "unknown"),
    "PAST": ("rear", "unknown"),
    "UNSPECIFIED": ("unknown", "unknown"),
}


def validate_driving_intent(document: Any) -> None:
    """Validate the subset of DrivingIntent required by this integration."""

    if not isinstance(document, dict):
        raise ValueError("DrivingIntent must be a JSON object")
    schema_version = document.get("schema_version")
    if schema_version not in SUPPORTED_DRIVING_INTENT_SCHEMA_VERSIONS:
        supported = ", ".join(
            sorted(SUPPORTED_DRIVING_INTENT_SCHEMA_VERSIONS)
        )
        raise ValueError(
            "DrivingIntent schema_version must be one of: "
            + supported
        )
    request_id = document.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("DrivingIntent request_id must be a non-empty string")

    parse_result = document.get("parse_result")
    if not isinstance(parse_result, dict):
        raise ValueError("DrivingIntent parse_result must be an object")
    status = parse_result.get("status")
    if status not in PARSE_STATUSES:
        raise ValueError("DrivingIntent parse_result.status is invalid")

    intent = document.get("intent")
    if not isinstance(intent, dict):
        raise ValueError("DrivingIntent intent must be an object")
    steps = intent.get("steps")
    if not isinstance(steps, list):
        raise ValueError("DrivingIntent intent.steps must be an array")
    if status == "VALID" and not steps:
        raise ValueError("VALID DrivingIntent must contain at least one step")

    step_ids: set[str] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"DrivingIntent intent.steps[{index}] must be an object")
        step_id = step.get("step_id")
        if not isinstance(step_id, str) or not step_id:
            raise ValueError(f"DrivingIntent intent.steps[{index}].step_id is invalid")
        if step_id in step_ids:
            raise ValueError(f"DrivingIntent contains duplicate step_id {step_id!r}")
        step_ids.add(step_id)
        action = step.get("action")
        if not isinstance(action, str) or not action:
            raise ValueError(f"DrivingIntent step {step_id!r} action is invalid")
        target = step.get("target")
        if target is not None:
            if not isinstance(target, dict):
                raise ValueError(f"DrivingIntent step {step_id!r} target must be an object")
            if not isinstance(target.get("type"), str) or not target["type"]:
                raise ValueError(f"DrivingIntent step {step_id!r} target.type is invalid")
            if not isinstance(target.get("relation"), str) or not target["relation"]:
                raise ValueError(f"DrivingIntent step {step_id!r} target.relation is invalid")


def target_to_reference(
    target: Mapping[str, Any],
    *,
    action: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Convert one DrivingIntent target to the existing matcher vocabulary."""

    target_type_raw = str(target.get("type", "UNKNOWN")).upper()
    relation_raw = str(target.get("relation", "UNSPECIFIED")).upper()
    position_hint, lane_hint = RELATION_HINTS.get(
        relation_raw, ("unknown", "unknown")
    )

    if target_type_raw == "LANE":
        direction = relation_raw
        if direction not in {"LEFT", "RIGHT"} and action == "CHANGE_LANE":
            direction = str((parameters or {}).get("direction", "")).upper()
        if direction == "LEFT":
            target_type = "left_lane"
            position_hint = "left"
            lane_hint = "left_adjacent_lane"
        elif direction == "RIGHT":
            target_type = "right_lane"
            position_hint = "right"
            lane_hint = "right_adjacent_lane"
        else:
            target_type = "unknown"
    elif (
        target_type_raw
        in WORLD_STATE_CAPABILITY_UNAVAILABLE_TARGET_TYPES
    ):
        target_type = WORLD_STATE_CAPABILITY_UNAVAILABLE_TARGET
    else:
        target_type = TARGET_TYPE_MAP.get(target_type_raw, "unknown")

    description = target.get("description")
    raw_text = f"{target_type_raw}/{relation_raw}"
    if isinstance(description, str) and description.strip():
        raw_text = description.strip()
    return {
        "raw_text": raw_text,
        "target_type": target_type,
        "position_hint": position_hint,
        "lane_hint": lane_hint,
    }


def _risk_by_object_id(risk: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(item["object_id"]): str(item["risk_level"])
        for item in risk["object_assessments"]
    }


def _actor_entity(
    obj: Mapping[str, Any],
    risk_by_id: Mapping[str, str],
) -> dict[str, Any]:
    distance = obj.get("distance_m")
    return {
        "entity_type": "actor",
        "entity_id": str(obj["object_id"]),
        "category": str(obj["category"]),
        "distance_m": round(float(distance), 6) if distance is not None else None,
        "relative_position": relative_position_label(obj),
        "lane_relation": str(obj["lane_relation"]),
        "risk_level": risk_by_id.get(
            str(obj["object_id"]), distance_risk_level(distance)
        ),
    }


def _lane_entity(direction: str, lane: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entity_type": "lane",
        "entity_id": f"lane:{lane.get('road_id')}:{lane.get('lane_id')}",
        "category": "lane",
        "distance_m": None,
        "relative_position": direction,
        "lane_relation": f"{direction}_adjacent_lane",
        "risk_level": "none",
    }


def _junction_entity(world_state: Mapping[str, Any]) -> dict[str, Any]:
    ego = world_state["ego"]
    return {
        "entity_type": "junction",
        "entity_id": f"junction:{ego['road_id']}:{ego['section_id']}",
        "category": "junction",
        "distance_m": 0.0,
        "relative_position": "front",
        "lane_relation": "ego_lane",
        "risk_level": "medium",
    }


def _align_reference(
    reference: Mapping[str, str],
    world_state: Mapping[str, Any],
    risk_by_id: Mapping[str, str],
) -> tuple[bool, int, dict[str, Any] | None, str]:
    target_type = reference["target_type"]
    if target_type == WORLD_STATE_CAPABILITY_UNAVAILABLE_TARGET:
        return (
            False,
            0,
            None,
            "world_state_capability_unavailable",
        )
    if target_type == "unknown":
        return False, 0, None, "unsupported_target_type"
    if target_type in {"left_lane", "right_lane"}:
        direction = "left" if target_type == "left_lane" else "right"
        lane = world_state["ego"]["adjacent_lanes"][direction]
        if lane is None:
            return False, 0, None, f"{direction}_lane_unavailable"
        return True, 1, _lane_entity(direction, lane), "matched_adjacent_lane"
    if target_type == "junction":
        available = world_state["ego"]["is_junction"] is True or (
            world_state["environment"]["is_intersection"] is True
        )
        if not available:
            return False, 0, None, "junction_not_currently_available"
        return True, 1, _junction_entity(world_state), "matched_current_junction"

    obj, candidate_count = select_world_object(reference, world_state)
    if obj is None:
        return False, candidate_count, None, "no_matching_entity"
    return (
        True,
        candidate_count,
        _actor_entity(obj, risk_by_id),
        "matched_world_object",
    )


def align_driving_intent(
    driving_intent: dict[str, Any],
    world_state: dict[str, Any],
) -> dict[str, Any]:
    """Align every target in one DrivingIntent against one WorldState frame."""

    validate_driving_intent(driving_intent)
    world_errors = validate_world_state(world_state)
    if world_errors:
        raise ValueError("invalid WorldState: " + "; ".join(world_errors))

    parse_status = driving_intent["parse_result"]["status"]
    if parse_status != "VALID":
        return {
            "schema_version": ALIGNMENT_SCHEMA_VERSION,
            "request_id": driving_intent["request_id"],
            "world_state_frame_id": world_state["frame_id"],
            "parse_status": parse_status,
            "alignment_status": "SKIPPED",
            "target_count": 0,
            "matched_target_count": 0,
            "step_alignments": [],
        }

    risk_by_id = _risk_by_object_id(assess_world_state(world_state))
    step_alignments: list[dict[str, Any]] = []
    target_count = 0
    matched_count = 0
    for step in driving_intent["intent"]["steps"]:
        target = step.get("target")
        if target is None:
            step_alignments.append(
                {
                    "step_id": step["step_id"],
                    "action": step["action"],
                    "target": None,
                    "alignment_required": False,
                    "alignment_success": None,
                    "candidate_count": 0,
                    "matched_entity": None,
                    "reason_code": "target_not_required",
                }
            )
            continue

        target_count += 1
        reference = target_to_reference(
            target,
            action=step["action"],
            parameters=step.get("parameters"),
        )
        success, candidates, entity, reason = _align_reference(
            reference, world_state, risk_by_id
        )
        if success:
            matched_count += 1
        step_alignments.append(
            {
                "step_id": step["step_id"],
                "action": step["action"],
                "target": target,
                "alignment_required": True,
                "alignment_success": success,
                "candidate_count": candidates,
                "matched_entity": entity,
                "reason_code": reason,
            }
        )

    if target_count == 0:
        alignment_status = "NOT_REQUIRED"
    elif matched_count == target_count:
        alignment_status = "COMPLETE"
    elif matched_count == 0:
        alignment_status = "FAILED"
    else:
        alignment_status = "PARTIAL"

    result = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "request_id": driving_intent["request_id"],
        "world_state_frame_id": world_state["frame_id"],
        "parse_status": parse_status,
        "alignment_status": alignment_status,
        "target_count": target_count,
        "matched_target_count": matched_count,
        "step_alignments": step_alignments,
    }
    errors = validate_alignment_result(result)
    if errors:
        raise ValueError("invalid semantic-alignment output: " + "; ".join(errors))
    return result


def validate_alignment_result(data: Any) -> list[str]:
    """Return structural and cross-reference errors for an alignment result."""

    errors: list[str] = []
    if not isinstance(data, dict):
        return ["root: expected an object"]
    expected = {
        "schema_version",
        "request_id",
        "world_state_frame_id",
        "parse_status",
        "alignment_status",
        "target_count",
        "matched_target_count",
        "step_alignments",
    }
    missing = sorted(expected - data.keys())
    extra = sorted(data.keys() - expected)
    if missing:
        errors.append("root: missing fields: " + ", ".join(missing))
    if extra:
        errors.append("root: unexpected fields: " + ", ".join(extra))
    if data.get("schema_version") != ALIGNMENT_SCHEMA_VERSION:
        errors.append("schema_version: expected '1.0.0'")
    for key in ("request_id", "world_state_frame_id"):
        if not isinstance(data.get(key), str) or not data[key]:
            errors.append(f"{key}: expected a non-empty string")
    if data.get("parse_status") not in PARSE_STATUSES:
        errors.append("parse_status: invalid value")
    if data.get("alignment_status") not in ALIGNMENT_STATUSES:
        errors.append("alignment_status: invalid value")

    target_count = data.get("target_count")
    matched_count = data.get("matched_target_count")
    for key, value in (
        ("target_count", target_count),
        ("matched_target_count", matched_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"{key}: expected a non-negative integer")
    if isinstance(target_count, int) and isinstance(matched_count, int):
        if matched_count > target_count:
            errors.append("matched_target_count: cannot exceed target_count")

    alignments = data.get("step_alignments")
    if not isinstance(alignments, list):
        errors.append("step_alignments: expected an array")
        return errors
    step_ids: set[str] = set()
    successful = 0
    required = 0
    for index, alignment in enumerate(alignments):
        path = f"step_alignments[{index}]"
        if not isinstance(alignment, dict):
            errors.append(f"{path}: expected an object")
            continue
        step_id = alignment.get("step_id")
        if not isinstance(step_id, str) or not step_id:
            errors.append(f"{path}.step_id: expected a non-empty string")
        elif step_id in step_ids:
            errors.append(f"{path}.step_id: duplicate value {step_id!r}")
        else:
            step_ids.add(step_id)
        is_required = alignment.get("alignment_required")
        is_success = alignment.get("alignment_success")
        if not isinstance(is_required, bool):
            errors.append(f"{path}.alignment_required: expected a boolean")
        elif is_required:
            required += 1
            if not isinstance(is_success, bool):
                errors.append(f"{path}.alignment_success: expected a boolean")
            elif is_success:
                successful += 1
                if alignment.get("matched_entity") is None:
                    errors.append(f"{path}.matched_entity: successful match requires entity")
        elif is_success is not None:
            errors.append(f"{path}.alignment_success: non-required target must use null")
        if not isinstance(alignment.get("reason_code"), str) or not alignment["reason_code"]:
            errors.append(f"{path}.reason_code: expected a non-empty string")
    if isinstance(target_count, int) and required != target_count:
        errors.append("target_count: does not equal required step alignments")
    if isinstance(matched_count, int) and successful != matched_count:
        errors.append("matched_target_count: does not equal successful alignments")
    return errors
