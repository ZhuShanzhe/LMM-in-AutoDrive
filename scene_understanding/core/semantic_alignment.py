"""Align command references with actors, lanes and junctions in WorldState."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

from scene_understanding.core.object_matcher import (
    TARGET_TYPES,
    normalize_instruction_reference,
    relative_position_label,
    select_world_object,
)
from scene_understanding.core.risk_assessment import (
    RISK_RANK,
    distance_risk_level,
    validate_risk_assessment,
)
from scene_understanding.core.world_state import LANE_RELATIONS, validate_world_state


TOP_LEVEL_KEYS = {
    "schema_version",
    "frame_id",
    "reference",
    "alignment_success",
    "candidate_count",
    "matched_entity",
    "reason_code",
}
REFERENCE_KEYS = {"raw_text", "target_type", "position_hint", "lane_hint"}
ENTITY_KEYS = {
    "entity_type",
    "entity_id",
    "category",
    "distance_m",
    "relative_position",
    "lane_relation",
    "risk_level",
}
ENTITY_TYPES = {"actor", "lane", "junction"}
POSITION_LABELS = {
    "front",
    "front_left",
    "front_right",
    "left",
    "right",
    "rear",
    "rear_left",
    "rear_right",
    "unknown",
}


def _risk_by_object_id(risk_assessment: Mapping[str, Any] | None) -> dict[str, str]:
    if risk_assessment is None:
        return {}
    return {
        assessment["object_id"]: assessment["risk_level"]
        for assessment in risk_assessment["object_assessments"]
    }


def _actor_entity(
    obj: Mapping[str, Any],
    risk_by_id: Mapping[str, str],
) -> dict[str, Any]:
    distance = obj.get("distance_m")
    risk_level = risk_by_id.get(str(obj["object_id"]), distance_risk_level(distance))
    return {
        "entity_type": "actor",
        "entity_id": str(obj["object_id"]),
        "category": obj["category"],
        "distance_m": round(float(distance), 6) if distance is not None else None,
        "relative_position": relative_position_label(obj),
        "lane_relation": obj["lane_relation"],
        "risk_level": risk_level,
    }


def _lane_entity(direction: str, lane: Mapping[str, Any]) -> dict[str, Any]:
    road_id = lane.get("road_id")
    lane_id = lane.get("lane_id")
    return {
        "entity_type": "lane",
        "entity_id": f"lane:{road_id}:{lane_id}",
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


def align_instruction_reference(
    reference_input: str | Mapping[str, Any],
    world_state: dict[str, Any],
    *,
    risk_assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Align one parser reference with the current validated world snapshot."""

    world_errors = validate_world_state(world_state)
    if world_errors:
        raise ValueError("invalid WorldState: " + "; ".join(world_errors))
    if risk_assessment is not None:
        risk_errors = validate_risk_assessment(risk_assessment)
        if risk_errors:
            raise ValueError("invalid risk assessment: " + "; ".join(risk_errors))
        if risk_assessment["frame_id"] != world_state["frame_id"]:
            raise ValueError("risk assessment and WorldState frame_id must match")

    reference = normalize_instruction_reference(reference_input)
    matched_entity: dict[str, Any] | None = None
    candidate_count = 0
    reason_code = "no_matching_entity"

    target_type = reference["target_type"]
    if target_type == "unknown":
        reason_code = "unknown_reference"
    elif target_type in {"left_lane", "right_lane"}:
        direction = "left" if target_type == "left_lane" else "right"
        lane = world_state["ego"]["adjacent_lanes"][direction]
        if lane is None:
            reason_code = f"{direction}_lane_unavailable"
        else:
            candidate_count = 1
            matched_entity = _lane_entity(direction, lane)
            reason_code = "matched_adjacent_lane"
    elif target_type == "junction":
        is_junction = world_state["ego"]["is_junction"] is True or (
            world_state["environment"]["is_intersection"] is True
        )
        if is_junction:
            candidate_count = 1
            matched_entity = _junction_entity(world_state)
            reason_code = "matched_current_junction"
        else:
            reason_code = "junction_not_currently_available"
    else:
        obj, candidate_count = select_world_object(reference, world_state)
        if obj is not None:
            matched_entity = _actor_entity(obj, _risk_by_object_id(risk_assessment))
            reason_code = "matched_world_object"

    result = {
        "schema_version": "1.0",
        "frame_id": world_state["frame_id"],
        "reference": reference,
        "alignment_success": matched_entity is not None,
        "candidate_count": candidate_count,
        "matched_entity": matched_entity,
        "reason_code": reason_code,
    }
    errors = validate_semantic_alignment(result)
    if errors:
        raise ValueError("invalid semantic alignment: " + "; ".join(errors))
    return result


def _exact_keys(value: Any, expected: set[str], path: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{path}: expected an object")
        return False
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    if missing:
        errors.append(f"{path}: missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"{path}: unexpected fields: {', '.join(extra)}")
    return not missing and not extra


def _finite_nonnegative_or_null(value: Any, path: str, errors: list[str]) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        errors.append(f"{path}: expected a finite non-negative number or null")


def validate_semantic_alignment(data: Any) -> list[str]:
    """Validate the semantic alignment contract without third-party packages."""

    errors: list[str] = []
    if not _exact_keys(data, TOP_LEVEL_KEYS, "root", errors):
        if not isinstance(data, dict):
            return errors
    if data.get("schema_version") != "1.0":
        errors.append("schema_version: expected '1.0'")
    if not isinstance(data.get("frame_id"), str) or not data["frame_id"]:
        errors.append("frame_id: expected a non-empty string")

    reference = data.get("reference")
    if _exact_keys(reference, REFERENCE_KEYS, "reference", errors):
        if not isinstance(reference["raw_text"], str):
            errors.append("reference.raw_text: expected a string")
        if reference["target_type"] not in TARGET_TYPES:
            errors.append("reference.target_type: invalid value")
        if reference["position_hint"] not in POSITION_LABELS:
            errors.append("reference.position_hint: invalid value")
        if reference["lane_hint"] not in LANE_RELATIONS:
            errors.append("reference.lane_hint: invalid value")

    success = data.get("alignment_success")
    if not isinstance(success, bool):
        errors.append("alignment_success: expected a boolean")
    count = data.get("candidate_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        errors.append("candidate_count: expected a non-negative integer")
    if not isinstance(data.get("reason_code"), str) or not data["reason_code"]:
        errors.append("reason_code: expected a non-empty string")

    entity = data.get("matched_entity")
    if entity is not None and _exact_keys(entity, ENTITY_KEYS, "matched_entity", errors):
        if entity["entity_type"] not in ENTITY_TYPES:
            errors.append("matched_entity.entity_type: invalid value")
        for key in ("entity_id", "category"):
            if not isinstance(entity[key], str) or not entity[key]:
                errors.append(f"matched_entity.{key}: expected a non-empty string")
        _finite_nonnegative_or_null(entity["distance_m"], "matched_entity.distance_m", errors)
        if entity["relative_position"] not in POSITION_LABELS:
            errors.append("matched_entity.relative_position: invalid value")
        if entity["lane_relation"] not in LANE_RELATIONS:
            errors.append("matched_entity.lane_relation: invalid value")
        if entity["risk_level"] not in RISK_RANK:
            errors.append("matched_entity.risk_level: invalid value")

    if success is True:
        if entity is None:
            errors.append("matched_entity: successful alignment requires an entity")
        if isinstance(count, int) and count < 1:
            errors.append("candidate_count: successful alignment requires at least one candidate")
    elif success is False and entity is not None:
        errors.append("matched_entity: failed alignment must use null")
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="reference text, for example 前车 or 左车道")
    parser.add_argument("world_state", type=Path, help="WorldState JSON file")
    parser.add_argument("--risk", type=Path, help="optional risk assessment JSON")
    parser.add_argument("--output", type=Path, help="output JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    world_state = json.loads(args.world_state.read_text(encoding="utf-8"))
    risk = json.loads(args.risk.read_text(encoding="utf-8")) if args.risk else None
    try:
        result = align_instruction_reference(
            args.reference,
            world_state,
            risk_assessment=risk,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote semantic alignment to {args.output}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
