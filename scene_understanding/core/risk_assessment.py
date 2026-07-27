"""Deterministic TTC, distance-risk and lane-change assessment.

All metric inputs must come from a validated WorldState.  Visual descriptions
may explain a matched object, but they never provide distance, speed or TTC.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from scene_understanding.core.world_state import validate_world_state


RISK_LEVELS = ("none", "low", "medium", "high")
RISK_RANK = {level: rank for rank, level in enumerate(RISK_LEVELS)}
RECOMMENDED_ACTIONS = {"maintain_speed", "monitor", "decelerate", "emergency_brake"}

EGO_PATH_HALF_WIDTH_M = 2.5
PATH_CONFLICT_HORIZON_S = 4.0
DRIVER_REACTION_TIME_S = 1.0
COMFORTABLE_DECELERATION_MPS2 = 4.0
EMERGENCY_DECELERATION_MPS2 = 8.0
STOPPING_DISTANCE_BUFFER_M = 2.0
STATIC_HAZARD_CATEGORIES = {
    "traffic_cone",
    "road_barrier",
    "obstacle",
    "other",
}

TOP_LEVEL_KEYS = {
    "schema_version",
    "frame_id",
    "risk_level",
    "reason_codes",
    "recommended_action",
    "safe_following_distance_m",
    "object_assessments",
    "lane_change",
}
OBJECT_ASSESSMENT_KEYS = {
    "object_id",
    "relevant_to_ego_path",
    "distance_m",
    "safe_distance_m",
    "distance_is_safe",
    "closing_speed_mps",
    "ttc_s",
    "ttc_risk_level",
    "risk_level",
    "reason_codes",
}
LANE_JUDGMENT_KEYS = {
    "direction",
    "is_safe",
    "reason_codes",
    "blocking_object_ids",
    "closest_front_gap_m",
    "closest_rear_gap_m",
}


def safe_following_distance_m(ego_speed_mps: float) -> float:
    """Return the plan-defined safe distance for the ego speed."""

    if ego_speed_mps < 0 or not math.isfinite(ego_speed_mps):
        raise ValueError("ego_speed_mps must be a finite non-negative number")
    # Stabilize exact plan boundaries after the common km/h -> m/s -> km/h
    # round trip (for example 60 / 3.6 * 3.6 can be 60.00000000000001).
    speed_kmh = round(ego_speed_mps * 3.6, 9)
    if speed_kmh < 30.0:
        return 10.0
    if speed_kmh <= 60.0:
        return 20.0
    return 40.0


def compute_ttc_s(distance_m: float | None, closing_speed_mps: float | None) -> float | None:
    """Return TTC in seconds, or ``None`` if the separation is not shrinking."""

    if distance_m is None or closing_speed_mps is None:
        return None
    if not math.isfinite(distance_m) or not math.isfinite(closing_speed_mps):
        return None
    if distance_m < 0 or closing_speed_mps <= 1e-6:
        return None
    return distance_m / closing_speed_mps


def ttc_risk_level(ttc_s: float | None) -> str:
    """Map TTC to the exact bands in the team plan."""

    if ttc_s is None or ttc_s > 4.0:
        return "none"
    if ttc_s >= 2.0:
        return "low"
    if ttc_s >= 1.0:
        return "medium"
    return "high"


def distance_risk_level(distance_m: float | None) -> str:
    """Map a relevant object's metric distance to the plan-defined bands."""

    if distance_m is None:
        return "none"
    if distance_m < 10.0:
        return "high"
    if distance_m <= 25.0:
        return "medium"
    return "low"


def _max_risk(*levels: str) -> str:
    return max(levels, key=RISK_RANK.__getitem__)


def required_stopping_distance_m(
    ego_speed_mps: float,
    *,
    reaction_time_s: float = DRIVER_REACTION_TIME_S,
    deceleration_mps2: float = COMFORTABLE_DECELERATION_MPS2,
    buffer_m: float = STOPPING_DISTANCE_BUFFER_M,
) -> float:
    """Return reaction distance plus braking distance and a small buffer."""

    values = {
        "ego_speed_mps": ego_speed_mps,
        "reaction_time_s": reaction_time_s,
        "deceleration_mps2": deceleration_mps2,
        "buffer_m": buffer_m,
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values.values()
    ):
        raise ValueError("stopping-distance inputs must be finite numbers")
    if ego_speed_mps < 0 or reaction_time_s < 0 or buffer_m < 0:
        raise ValueError(
            "speed, reaction time and buffer must be non-negative"
        )
    if deceleration_mps2 <= 0:
        raise ValueError("deceleration_mps2 must be positive")

    reaction_distance = ego_speed_mps * reaction_time_s
    braking_distance = (
        ego_speed_mps * ego_speed_mps
        / (2.0 * deceleration_mps2)
    )
    return reaction_distance + braking_distance + buffer_m


def _corridor_entry_time_s(
    obj: dict[str, Any],
) -> float | None:
    """Predict when a laterally approaching object enters the ego corridor."""

    position = obj.get("relative_position_ego_m")
    velocity = obj.get("relative_velocity_ego_mps")
    if not isinstance(position, dict) or not isinstance(
        velocity,
        dict,
    ):
        return None

    lateral = position.get("lateral")
    longitudinal = position.get("longitudinal")
    lateral_velocity = velocity.get("lateral")
    longitudinal_velocity = velocity.get("longitudinal")

    values = (
        lateral,
        longitudinal,
        lateral_velocity,
        longitudinal_velocity,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        return None
    if longitudinal <= 0:
        return None

    lateral_distance = abs(lateral)
    if lateral_distance <= EGO_PATH_HALF_WIDTH_M:
        entry_time = 0.0
    else:
        if (
            abs(lateral_velocity) <= 1e-6
            or lateral * lateral_velocity >= 0
        ):
            return None
        entry_time = (
            lateral_distance - EGO_PATH_HALF_WIDTH_M
        ) / abs(lateral_velocity)

    if entry_time > PATH_CONFLICT_HORIZON_S:
        return None

    predicted_longitudinal = (
        longitudinal
        + longitudinal_velocity * entry_time
    )
    if predicted_longitudinal <= 0:
        return None
    return entry_time


def _path_conflict_kind(
    obj: dict[str, Any],
) -> str | None:
    category = obj["category"]
    relation = obj["lane_relation"]

    if (
        category == "vehicle"
        and relation
        in {"left_adjacent_lane", "right_adjacent_lane"}
    ):
        return "cut_in"

    if (
        category in {"pedestrian", "cyclist"}
        and relation
        in {"roadside", "unknown", "crossing_ego_path"}
    ):
        return category

    return None


def _predicted_longitudinal_gap_at_entry_m(
    obj: dict[str, Any],
    conflict_time_s: float | None,
) -> float | None:
    """Return predicted forward separation at corridor entry."""

    if conflict_time_s is None:
        return None

    position = obj.get("relative_position_ego_m")
    velocity = obj.get("relative_velocity_ego_mps")
    if not isinstance(position, dict) or not isinstance(
        velocity,
        dict,
    ):
        return None

    longitudinal = position.get("longitudinal")
    longitudinal_velocity = velocity.get("longitudinal")
    values = (
        longitudinal,
        longitudinal_velocity,
        conflict_time_s,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        return None

    return (
        longitudinal
        + longitudinal_velocity * conflict_time_s
    )


def _conflict_risk_level(
    conflict_time_s: float | None,
    predicted_gap_m: float | None,
    *,
    safe_distance_m: float,
) -> str:
    if (
        conflict_time_s is None
        or predicted_gap_m is None
        or predicted_gap_m > safe_distance_m
    ):
        return "none"
    if conflict_time_s <= 1.0:
        return "high"
    if conflict_time_s <= 2.0:
        return "medium"
    if conflict_time_s <= PATH_CONFLICT_HORIZON_S:
        return "low"
    return "none"


def _is_path_relevant(obj: dict[str, Any]) -> bool:
    if obj["category"] in {"traffic_light", "traffic_sign"}:
        return False

    position = obj.get("relative_position_ego_m")
    if (
        not isinstance(position, dict)
        or position["longitudinal"] <= 0
    ):
        return False

    relation = obj["lane_relation"]
    if relation in {"ego_lane", "crossing_ego_path"}:
        return True

    conflict_kind = _path_conflict_kind(obj)
    if (
        conflict_kind is not None
        and _corridor_entry_time_s(obj) is not None
    ):
        return True

    if relation in {
        "left_adjacent_lane",
        "right_adjacent_lane",
        "oncoming_lane",
        "roadside",
    }:
        return False

    return abs(position["lateral"]) <= EGO_PATH_HALF_WIDTH_M


def _is_static_path_hazard(
    obj: dict[str, Any],
) -> bool:
    if obj["category"] in STATIC_HAZARD_CATEGORIES:
        return True
    if obj["category"] != "vehicle":
        return False

    speed = obj.get("speed_mps")
    return (
        isinstance(speed, (int, float))
        and not isinstance(speed, bool)
        and math.isfinite(speed)
        and speed <= 0.5
    )


def assess_object(
    obj: dict[str, Any],
    *,
    safe_distance_m: float,
    ego_speed_mps: float,
) -> dict[str, Any]:
    relevant = _is_path_relevant(obj)
    distance = obj.get("distance_m")
    closing_speed = obj.get("closing_speed_mps")
    ttc = (
        compute_ttc_s(distance, closing_speed)
        if relevant
        else None
    )
    ttc_level = ttc_risk_level(ttc)

    conflict_kind = (
        _path_conflict_kind(obj)
        if relevant
        else None
    )
    conflict_time = (
        _corridor_entry_time_s(obj)
        if conflict_kind is not None
        else None
    )
    predicted_conflict_gap = (
        _predicted_longitudinal_gap_at_entry_m(
            obj,
            conflict_time,
        )
    )
    conflict_level = _conflict_risk_level(
        conflict_time,
        predicted_conflict_gap,
        safe_distance_m=safe_distance_m,
    )

    reasons: list[str] = []

    if not relevant:
        risk_level = "none"
        distance_is_safe = None
    else:
        distance_level = distance_risk_level(distance)
        risk_level = _max_risk(
            distance_level,
            ttc_level,
            conflict_level,
        )
        distance_is_safe = (
            distance is not None
            and distance >= safe_distance_m
        )

        if distance is None:
            reasons.append("metric_distance_unavailable")
        elif distance < 10.0:
            reasons.append("distance_below_10m")
        elif distance <= 25.0:
            reasons.append("distance_10_to_25m")
        else:
            reasons.append("distance_above_25m")

        if distance_is_safe is False:
            risk_level = _max_risk(
                risk_level,
                "medium",
            )
            reasons.append(
                "below_speed_based_safe_distance"
            )

        if ttc_level == "high":
            reasons.append("ttc_below_1s")
        elif ttc_level == "medium":
            reasons.append("ttc_1_to_2s")
        elif ttc_level == "low":
            reasons.append("ttc_2_to_4s")

        if conflict_kind is not None:
            if conflict_level == "high":
                reasons.append(
                    f"{conflict_kind}_path_conflict_imminent"
                )
            elif conflict_level in {"medium", "low"}:
                reasons.append(
                    f"{conflict_kind}_path_conflict_predicted"
                )

        if (
            distance is not None
            and _is_static_path_hazard(obj)
        ):
            comfortable_stopping_distance = (
                required_stopping_distance_m(
                    ego_speed_mps
                )
            )
            emergency_stopping_distance = (
                required_stopping_distance_m(
                    ego_speed_mps,
                    deceleration_mps2=(
                        EMERGENCY_DECELERATION_MPS2
                    ),
                )
            )

            if distance < emergency_stopping_distance:
                risk_level = "high"
                reasons.append(
                    "insufficient_emergency_stopping_distance"
                )
            elif distance < comfortable_stopping_distance:
                risk_level = _max_risk(
                    risk_level,
                    "medium",
                )
                reasons.append(
                    "insufficient_stopping_distance"
                )

    return {
        "object_id": obj["object_id"],
        "relevant_to_ego_path": relevant,
        "distance_m": (
            round(distance, 6)
            if distance is not None
            else None
        ),
        "safe_distance_m": safe_distance_m,
        "distance_is_safe": distance_is_safe,
        "closing_speed_mps": (
            round(closing_speed, 6)
            if closing_speed is not None
            else None
        ),
        "ttc_s": (
            round(ttc, 6)
            if ttc is not None
            else None
        ),
        "ttc_risk_level": ttc_level,
        "risk_level": risk_level,
        "reason_codes": reasons,
    }


def assess_lane_change(
    world_state: dict[str, Any],
    direction: str,
    *,
    safe_distance_m: float,
) -> dict[str, Any]:
    """Conservatively judge one target lane using map relation, gap and TTC."""

    if direction not in {"left", "right"}:
        raise ValueError("direction must be 'left' or 'right'")
    permission = world_state["ego"]["lane_change"]
    allowed = permission in {direction, "both"}
    reasons: list[str] = []
    blocking_ids: list[str] = []
    if not allowed:
        reasons.append("lane_change_not_permitted")

    target_relation = f"{direction}_adjacent_lane"
    front_gaps: list[float] = []
    rear_gaps: list[float] = []
    for obj in world_state["objects"]:
        if obj["lane_relation"] != target_relation:
            continue
        position = obj.get("relative_position_ego_m")
        if not isinstance(position, dict):
            blocking_ids.append(obj["object_id"])
            reasons.append("target_lane_object_position_unknown")
            continue
        longitudinal = position["longitudinal"]
        gap = abs(longitudinal)
        if longitudinal >= 0:
            front_gaps.append(gap)
            gap_reason = "target_lane_front_gap_too_small"
        else:
            rear_gaps.append(gap)
            gap_reason = "target_lane_rear_gap_too_small"
        relative_speed = obj.get(
            "relative_longitudinal_speed_mps"
        )
        if isinstance(relative_speed, (int, float)) and not isinstance(
            relative_speed,
            bool,
        ):
            if longitudinal >= 0:
                lane_closing_speed = max(
                    0.0,
                    -float(relative_speed),
                )
            else:
                lane_closing_speed = max(
                    0.0,
                    float(relative_speed),
                )
        else:
            lane_closing_speed = obj.get(
                "closing_speed_mps"
            )

        ttc = compute_ttc_s(
            gap,
            lane_closing_speed,
        )
        if gap < safe_distance_m:
            blocking_ids.append(obj["object_id"])
            reasons.append(gap_reason)
        if ttc is not None and ttc <= 4.0:
            blocking_ids.append(obj["object_id"])
            reasons.append("target_lane_ttc_at_most_4s")

    reasons = list(dict.fromkeys(reasons))
    blocking_ids = list(dict.fromkeys(blocking_ids))
    is_safe = allowed and not blocking_ids
    if is_safe:
        reasons.append("target_lane_clear")
    return {
        "direction": direction,
        "is_safe": is_safe,
        "reason_codes": reasons,
        "blocking_object_ids": blocking_ids,
        "closest_front_gap_m": round(min(front_gaps), 6) if front_gaps else None,
        "closest_rear_gap_m": round(min(rear_gaps), 6) if rear_gaps else None,
    }


def assess_world_state(world_state: dict[str, Any]) -> dict[str, Any]:
    """Generate the final deterministic risk and lane-change judgment."""

    input_errors = validate_world_state(world_state)
    if input_errors:
        raise ValueError("invalid WorldState: " + "; ".join(input_errors))

    safe_distance = safe_following_distance_m(world_state["ego"]["speed_mps"])
    ego_speed = world_state["ego"]["speed_mps"]
    object_assessments = [
        assess_object(
            obj,
            safe_distance_m=safe_distance,
            ego_speed_mps=ego_speed,
        )
        for obj in world_state["objects"]
    ]
    overall = "none"
    for assessment in object_assessments:
        overall = _max_risk(overall, assessment["risk_level"])

    event_reasons: list[str] = []
    collisions = world_state["sensor_events"]["collisions"]
    lane_invasions = world_state["sensor_events"]["lane_invasions"]
    if collisions:
        overall = "high"
        event_reasons.append("collision_detected")
    if lane_invasions:
        overall = _max_risk(overall, "medium")
        event_reasons.append("lane_invasion_detected")

    top_reasons = list(event_reasons)
    for assessment in object_assessments:
        if assessment["risk_level"] == overall:
            top_reasons.extend(assessment["reason_codes"])
    top_reasons = list(dict.fromkeys(top_reasons))

    emergency_reason_codes = {
        "cut_in_path_conflict_imminent",
        "pedestrian_path_conflict_imminent",
        "cyclist_path_conflict_imminent",
        "insufficient_emergency_stopping_distance",
    }
    emergency_required = any(
        (
            assessment["ttc_s"] is not None
            and assessment["ttc_s"] < 1.0
        )
        or bool(
            emergency_reason_codes
            & set(assessment["reason_codes"])
        )
        for assessment in object_assessments
    )

    if collisions or emergency_required:
        action = "emergency_brake"
    elif overall in {"high", "medium"}:
        action = "decelerate"
    elif overall == "low":
        action = "monitor"
    else:
        action = "maintain_speed"

    result = {
        "schema_version": "1.0",
        "frame_id": world_state["frame_id"],
        "risk_level": overall,
        "reason_codes": top_reasons,
        "recommended_action": action,
        "safe_following_distance_m": safe_distance,
        "object_assessments": object_assessments,
        "lane_change": {
            "left": assess_lane_change(
                world_state, "left", safe_distance_m=safe_distance
            ),
            "right": assess_lane_change(
                world_state, "right", safe_distance_m=safe_distance
            ),
        },
    }
    errors = validate_risk_assessment(result)
    if errors:
        raise ValueError("invalid risk output: " + "; ".join(errors))
    return result


def _exact_keys(value: Any, keys: set[str], path: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{path}: expected an object")
        return False
    missing = sorted(keys - value.keys())
    extra = sorted(value.keys() - keys)
    if missing:
        errors.append(f"{path}: missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"{path}: unexpected fields: {', '.join(extra)}")
    return not missing and not extra


def _optional_number(value: Any, path: str, errors: list[str], *, minimum: float | None = None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append(f"{path}: expected a finite number or null")
    elif minimum is not None and value < minimum:
        errors.append(f"{path}: must be at least {minimum}")


def _string_array(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        errors.append(f"{path}: expected an array of non-empty strings")
    elif len(value) != len(set(value)):
        errors.append(f"{path}: duplicate values")


def validate_risk_assessment(data: Any) -> list[str]:
    """Return structural and cross-reference errors for a risk output."""

    errors: list[str] = []
    if not _exact_keys(data, TOP_LEVEL_KEYS, "root", errors):
        if not isinstance(data, dict):
            return errors
    if data.get("schema_version") != "1.0":
        errors.append("schema_version: expected '1.0'")
    if not isinstance(data.get("frame_id"), str) or not data["frame_id"]:
        errors.append("frame_id: expected a non-empty string")
    if data.get("risk_level") not in RISK_RANK:
        errors.append("risk_level: invalid value")
    _string_array(data.get("reason_codes"), "reason_codes", errors)
    if data.get("recommended_action") not in RECOMMENDED_ACTIONS:
        errors.append("recommended_action: invalid value")
    _optional_number(
        data.get("safe_following_distance_m"),
        "safe_following_distance_m",
        errors,
        minimum=0,
    )

    assessments = data.get("object_assessments")
    object_ids: set[str] = set()
    if not isinstance(assessments, list):
        errors.append("object_assessments: expected an array")
    else:
        for index, assessment in enumerate(assessments):
            path = f"object_assessments[{index}]"
            if not _exact_keys(assessment, OBJECT_ASSESSMENT_KEYS, path, errors):
                if not isinstance(assessment, dict):
                    continue
            object_id = assessment.get("object_id")
            if not isinstance(object_id, str) or not object_id:
                errors.append(f"{path}.object_id: expected a non-empty string")
            elif object_id in object_ids:
                errors.append(f"{path}.object_id: duplicate ID {object_id}")
            else:
                object_ids.add(object_id)
            if not isinstance(assessment.get("relevant_to_ego_path"), bool):
                errors.append(f"{path}.relevant_to_ego_path: expected a boolean")
            for key in ("distance_m", "safe_distance_m", "closing_speed_mps", "ttc_s"):
                _optional_number(assessment.get(key), f"{path}.{key}", errors, minimum=0)
            if assessment.get("distance_is_safe") is not None and not isinstance(
                assessment.get("distance_is_safe"), bool
            ):
                errors.append(f"{path}.distance_is_safe: expected a boolean or null")
            for key in ("ttc_risk_level", "risk_level"):
                if assessment.get(key) not in RISK_RANK:
                    errors.append(f"{path}.{key}: invalid value")
            _string_array(assessment.get("reason_codes"), f"{path}.reason_codes", errors)

    lane_change = data.get("lane_change")
    if not isinstance(lane_change, dict) or set(lane_change) != {"left", "right"}:
        errors.append("lane_change: expected exactly left and right")
    else:
        for direction in ("left", "right"):
            judgment = lane_change[direction]
            path = f"lane_change.{direction}"
            if not _exact_keys(judgment, LANE_JUDGMENT_KEYS, path, errors):
                if not isinstance(judgment, dict):
                    continue
            if judgment.get("direction") != direction:
                errors.append(f"{path}.direction: expected {direction!r}")
            if not isinstance(judgment.get("is_safe"), bool):
                errors.append(f"{path}.is_safe: expected a boolean")
            _string_array(judgment.get("reason_codes"), f"{path}.reason_codes", errors)
            _string_array(
                judgment.get("blocking_object_ids"), f"{path}.blocking_object_ids", errors
            )
            unknown_ids = set(judgment.get("blocking_object_ids") or []) - object_ids
            if unknown_ids:
                errors.append(f"{path}.blocking_object_ids: unknown IDs {sorted(unknown_ids)}")
            for key in ("closest_front_gap_m", "closest_rear_gap_m"):
                _optional_number(judgment.get(key), f"{path}.{key}", errors, minimum=0)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="WorldState JSON file")
    parser.add_argument("--output", type=Path, help="Optional risk JSON output path")
    args = parser.parse_args()
    try:
        world_state = json.loads(args.input.read_text(encoding="utf-8"))
        result = assess_world_state(world_state)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"RISK ASSESSMENT FAILED: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote risk assessment to {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
