"""Normalize language references and ground them to WorldState entities."""

from __future__ import annotations

import re
from typing import Any, Mapping


TARGET_TYPES = {
    "pedestrian",
    "cyclist",
    "front_vehicle",
    "slow_vehicle",
    "vehicle",
    "obstacle",
    "traffic_cone",
    "road_hazard",
    "left_lane",
    "right_lane",
    "junction",
    "traffic_light",
    "traffic_sign",
    "unknown",
}

REFERENCE_ALIASES: tuple[tuple[str, str], ...] = (
    ("change_lane_left", "left_lane"),
    ("向左变道", "left_lane"),
    ("左侧车道", "left_lane"),
    ("左车道", "left_lane"),
    ("left lane", "left_lane"),
    ("change_lane_right", "right_lane"),
    ("向右变道", "right_lane"),
    ("右侧车道", "right_lane"),
    ("右车道", "right_lane"),
    ("right lane", "right_lane"),
    ("intersection", "junction"),
    ("junction", "junction"),
    ("路口", "junction"),
    ("slow vehicle", "slow_vehicle"),
    ("slow car", "slow_vehicle"),
    ("慢车", "slow_vehicle"),
    ("front vehicle", "front_vehicle"),
    ("vehicle ahead", "front_vehicle"),
    ("前车", "front_vehicle"),
    ("pedestrian", "pedestrian"),
    ("walker", "pedestrian"),
    ("行人", "pedestrian"),
    ("cyclist", "cyclist"),
    ("traffic cone", "traffic_cone"),
    ("traffic light", "traffic_light"),
    ("信号灯", "traffic_light"),
    ("traffic sign", "traffic_sign"),
    ("vehicle", "vehicle"),
    ("车辆", "vehicle"),
)

REFERENCE_FIELDS = (
    "target_object",
    "object",
    "target",
    "entity",
    "reference_text",
    "raw_text",
    "text",
    "command",
    "intent",
)

COLOR_TERMS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "orange",
    "red",
    "silver",
    "white",
    "yellow",
}

VEHICLE_SUBTYPE_ALIASES = {
    "bus": {"bus", "coach"},
    "car": {"car", "vehicle"},
    "hatchback": {"hatchback"},
    "pickup": {"pickup"},
    "sedan": {"sedan", "saloon"},
    "suv": {"suv", "sport utility"},
    "taxi": {"taxi", "cab"},
    "truck": {"truck", "lorry"},
    "van": {"van", "minivan"},
    "unspecified": set(),
}


def _reference_text(reference_input: str | Mapping[str, Any]) -> str:
    if isinstance(reference_input, str):
        return reference_input.strip()
    if not isinstance(reference_input, Mapping):
        raise TypeError("reference input must be text or a mapping")
    for key in REFERENCE_FIELDS:
        value = reference_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_instruction_reference(
    reference_input: str | Mapping[str, Any],
) -> dict[str, Any]:
    """Return target vocabulary, position hints, and optional attributes."""

    raw_text = _reference_text(reference_input)
    normalized = raw_text.casefold().replace("-", " ")
    target_type = "unknown"
    for alias, candidate in REFERENCE_ALIASES:
        if alias in normalized:
            target_type = candidate
            break

    if target_type == "left_lane":
        position_hint, lane_hint = "left", "left_adjacent_lane"
    elif target_type == "right_lane":
        position_hint, lane_hint = "right", "right_adjacent_lane"
    elif target_type == "front_vehicle":
        position_hint, lane_hint = "front", "ego_lane"
    else:
        if any(token in normalized for token in ("左侧", "左边", "left")):
            position_hint = "left"
        elif any(token in normalized for token in ("右侧", "右边", "right")):
            position_hint = "right"
        elif any(token in normalized for token in ("后方", "后面", "rear", "behind")):
            position_hint = "rear"
        elif any(token in normalized for token in ("前方", "前面", "front", "ahead")):
            position_hint = "front"
        else:
            position_hint = "unknown"
        lane_hint = "unknown"

    result: dict[str, Any] = {
        "raw_text": raw_text,
        "target_type": target_type,
        "position_hint": position_hint,
        "lane_hint": lane_hint,
    }
    if isinstance(reference_input, Mapping):
        for key in ("canonical_attributes", "open_descriptors"):
            if key in reference_input:
                result[key] = reference_input[key]
    return result


def relative_position_label(obj: Mapping[str, Any]) -> str:
    position = obj.get("relative_position_ego_m")
    if not isinstance(position, Mapping):
        return "unknown"
    longitudinal = position.get("longitudinal")
    lateral = position.get("lateral")
    if not isinstance(longitudinal, (int, float)) or isinstance(longitudinal, bool):
        return "unknown"
    if not isinstance(lateral, (int, float)) or isinstance(lateral, bool):
        return "unknown"
    if abs(float(lateral)) <= 2.5:
        return "front" if longitudinal >= 0 else "rear"
    if longitudinal >= 0:
        return "front_right" if lateral > 0 else "front_left"
    return "rear_right" if lateral > 0 else "rear_left"


def _candidate_score(
    obj: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[float, float, str]:
    score = 0.0
    relation = obj.get("lane_relation")
    direction = relative_position_label(obj)
    lane_hint = reference["lane_hint"]
    position_hint = reference["position_hint"]
    if lane_hint != "unknown" and relation == lane_hint:
        score -= 100.0
    if position_hint in {
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
    } and direction == position_hint:
        score -= 50.0
    elif position_hint == "front" and direction.startswith("front"):
        score -= 50.0
    elif position_hint == "rear" and direction.startswith("rear"):
        score -= 50.0
    elif position_hint == "left" and "left" in direction:
        score -= 50.0
    elif position_hint == "right" and "right" in direction:
        score -= 50.0
    distance = obj.get("distance_m")
    numeric_distance = (
        float(distance) if isinstance(distance, (int, float)) else float("inf")
    )
    return score, numeric_distance, str(obj.get("object_id", ""))


def candidate_world_objects(
    reference: Mapping[str, Any],
    world_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return geometrically compatible candidates in deterministic order."""

    target_type = reference["target_type"]
    categories = {
        "pedestrian": {"pedestrian"},
        "cyclist": {"cyclist"},
        "traffic_light": {"traffic_light"},
        "traffic_sign": {"traffic_sign"},
        "traffic_cone": {"traffic_cone"},
        "obstacle": {"road_barrier", "traffic_cone", "other"},
        "road_hazard": {"road_barrier", "traffic_cone", "other"},
        "front_vehicle": {"vehicle"},
        "slow_vehicle": {"vehicle"},
        "vehicle": {"vehicle"},
    }.get(target_type)
    if categories is None:
        return []

    candidates: list[dict[str, Any]] = []
    ego_speed = world_state.get("ego", {}).get("speed_mps")
    for obj in world_state.get("objects", []):
        if not isinstance(obj, dict) or obj.get("category") not in categories:
            continue
        direction = relative_position_label(obj)
        position_hint = reference["position_hint"]
        if position_hint in {
            "front_left",
            "front_right",
            "rear_left",
            "rear_right",
        } and direction != position_hint:
            continue
        if position_hint == "front" and not direction.startswith("front"):
            continue
        if position_hint == "rear" and not direction.startswith("rear"):
            continue
        if position_hint == "left" and "left" not in direction:
            continue
        if position_hint == "right" and "right" not in direction:
            continue
        lane_hint = reference["lane_hint"]
        if (
            lane_hint != "unknown"
            and obj.get("lane_relation") not in {lane_hint, "unknown"}
        ):
            continue
        if target_type == "front_vehicle":
            position = obj.get("relative_position_ego_m")
            if not isinstance(position, Mapping) or position.get("longitudinal", 0) <= 0:
                continue
            if obj.get("lane_relation") not in {"ego_lane", "unknown"}:
                continue
            if (
                obj.get("lane_relation") == "unknown"
                and abs(position.get("lateral", 99)) > 2.5
            ):
                continue
        if target_type == "slow_vehicle":
            speed = obj.get("speed_mps")
            if not isinstance(speed, (int, float)):
                continue
            slower = (
                isinstance(ego_speed, (int, float))
                and speed < ego_speed - 0.5
            )
            if not slower and speed * 3.6 > 30.0:
                continue
        candidates.append(obj)
    candidates.sort(key=lambda item: _candidate_score(item, reference))
    return candidates


def _object_semantic_text(obj: Mapping[str, Any]) -> str:
    parts = [str(obj.get("subtype", ""))]
    for match in obj.get("semantic_matches", []):
        if isinstance(match, Mapping):
            parts.append(str(match.get("description", "")))
    return re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+",
        " ",
        " ".join(parts).casefold().replace("_", " "),
    ).strip()


def _attribute_evidence(
    obj: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[bool, list[str], list[str]]:
    requested = reference.get("canonical_attributes")
    if not isinstance(requested, Mapping):
        requested = {}
    text = _object_semantic_text(obj)
    words = set(text.split())
    matched: list[str] = []
    missing: list[str] = []

    color = requested.get("color")
    if isinstance(color, str):
        expected = color.casefold()
        observed = COLOR_TERMS & words
        if expected in observed:
            matched.append("color")
        else:
            missing.append("color")

    subtype = requested.get("vehicle_subtype")
    if isinstance(subtype, str) and subtype.casefold() != "unspecified":
        expected_terms = VEHICLE_SUBTYPE_ALIASES.get(
            subtype.casefold(), {subtype.casefold()}
        )
        if any(term in text for term in expected_terms):
            matched.append("vehicle_subtype")
        else:
            missing.append("vehicle_subtype")

    descriptors = reference.get("open_descriptors")
    if isinstance(descriptors, list):
        descriptor_tokens = {
            token
            for descriptor in descriptors
            if isinstance(descriptor, str)
            for token in descriptor.casefold().split()
            if len(token) > 2
        }
        if descriptor_tokens:
            if descriptor_tokens & words:
                matched.append("open_descriptors")
            else:
                missing.append("open_descriptors")
    return not missing, matched, missing


def match_world_object(
    reference: Mapping[str, Any],
    world_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Match with explicit evidence and reject unresolved ambiguity."""

    candidates = candidate_world_objects(reference, world_state)
    eligible: list[tuple[dict[str, Any], list[str]]] = []
    missing_fields: set[str] = set()
    for obj in candidates:
        matches, evidence, missing = _attribute_evidence(obj, reference)
        missing_fields.update(missing)
        if matches:
            eligible.append((obj, evidence))

    requested = reference.get("canonical_attributes")
    has_attributes = isinstance(requested, Mapping) and bool(requested)
    has_open_descriptors = bool(reference.get("open_descriptors"))
    requires_semantics = has_attributes or has_open_descriptors
    if requires_semantics and not eligible:
        return {
            "object": None,
            "candidate_count": len(candidates),
            "reason_code": "attribute_evidence_unavailable",
            "matched_attributes": [],
            "missing_attributes": sorted(missing_fields),
        }
    if not requires_semantics:
        eligible = [(obj, []) for obj in candidates]

    ordinal = requested.get("ordinal") if isinstance(requested, Mapping) else None
    if isinstance(ordinal, int):
        if ordinal < 1 or ordinal > len(eligible):
            return {
                "object": None,
                "candidate_count": len(eligible),
                "reason_code": "ordinal_target_unavailable",
                "matched_attributes": ["ordinal"],
                "missing_attributes": [],
            }
        selected, evidence = eligible[ordinal - 1]
        return {
            "object": selected,
            "candidate_count": len(eligible),
            "reason_code": "matched_ordinal_world_object",
            "matched_attributes": sorted({*evidence, "ordinal"}),
            "missing_attributes": [],
        }

    if requires_semantics and len(eligible) > 1:
        return {
            "object": None,
            "candidate_count": len(eligible),
            "reason_code": "ambiguous_matching_entities",
            "matched_attributes": sorted(
                {field for _, evidence in eligible for field in evidence}
            ),
            "missing_attributes": [],
        }
    if not eligible:
        return {
            "object": None,
            "candidate_count": 0,
            "reason_code": "no_matching_entity",
            "matched_attributes": [],
            "missing_attributes": [],
        }
    selected, evidence = eligible[0]
    return {
        "object": selected,
        "candidate_count": len(eligible),
        "reason_code": "matched_world_object",
        "matched_attributes": evidence,
        "missing_attributes": [],
    }


def select_world_object(
    reference: Mapping[str, Any],
    world_state: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, int]:
    result = match_world_object(reference, world_state)
    return result["object"], result["candidate_count"]
