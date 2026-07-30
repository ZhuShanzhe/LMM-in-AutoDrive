"""Final actuator-side guards for short-horizon trajectory conflicts."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping


_CONFLICT_RADIUS_M = {
    "vehicle": 3.0,
    "motorcycle": 2.5,
    "cyclist": 2.0,
    "pedestrian": 2.0,
}


def preserve_safe_lateral_maneuver(
    decision: Mapping[str, Any],
    active_step: Mapping[str, Any] | None,
    risk_assessment: Mapping[str, Any],
    *,
    speed_setpoint_kmh: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Allow an explicitly requested safe lane change to make progress."""

    audit = {
        "source": "driving_intent_lateral_progress_gate",
        "override_applied": False,
        "direction": None,
    }
    result = copy.deepcopy(dict(decision))
    if not active_step or str(active_step.get("action", "")).upper() != "CHANGE_LANE":
        return result, audit
    if result.get("action") != "decelerate":
        return result, audit
    if result.get("reason") != "risk_requires_deceleration":
        return result, audit

    direction = str(
        (active_step.get("parameters") or {}).get("direction", "")
    ).strip().lower()
    audit["direction"] = direction or None
    if direction not in {"left", "right"}:
        return result, audit
    lane_judgment = risk_assessment.get("lane_change", {}).get(direction, {})
    if lane_judgment.get("is_safe") is not True:
        return result, audit

    relevant = [
        item
        for item in risk_assessment.get("object_assessments", [])
        if item.get("relevant_to_ego_path") is True
    ]
    unsafe_object = any(
        item.get("distance_is_safe") is False
        or item.get("ttc_risk_level") in {"high", "emergency"}
        for item in relevant
    )
    if unsafe_object or risk_assessment.get("recommended_action") == "emergency_brake":
        return result, audit

    target_speed = (
        float(speed_setpoint_kmh)
        if speed_setpoint_kmh is not None
        else max(20.0, float(result.get("target_speed_kmh", 0.0)))
    )
    result["action"] = f"lane_change_{direction}"
    result["target_lane"] = direction
    result["target_location"] = None
    result["target_speed_kmh"] = round(max(20.0, min(target_speed, 45.0)), 6)
    result["emergency"] = False
    result["reason"] = "safe_lateral_maneuver_preserved"
    reasons = list(result.get("blocked_reason_codes", []))
    if "medium_longitudinal_risk_did_not_block_safe_lane_change" not in reasons:
        reasons.append(
            "medium_longitudinal_risk_did_not_block_safe_lane_change"
        )
    result["blocked_reason_codes"] = reasons
    audit["override_applied"] = True
    audit["target_speed_kmh"] = result["target_speed_kmh"]
    return result, audit


def _closest_approach(
    obj: Mapping[str, Any],
    horizon_s: float,
) -> tuple[float, float] | None:
    position = obj.get("relative_position_ego_m")
    velocity = obj.get("relative_velocity_ego_mps")
    if not isinstance(position, Mapping) or not isinstance(velocity, Mapping):
        return None
    px = float(position.get("longitudinal", 0.0))
    py = float(position.get("lateral", 0.0))
    vx = float(velocity.get("longitudinal", 0.0))
    vy = float(velocity.get("lateral", 0.0))
    # Braking cannot avoid a vehicle approaching from behind and instead
    # increases rear-end collision risk. Rear objects remain visible to the
    # scene-risk log but do not own the longitudinal actuator.
    if px <= 1.5:
        return None
    speed_sq = vx * vx + vy * vy
    if speed_sq < 0.04:
        return None
    time_s = max(0.0, min(float(horizon_s), -(px * vx + py * vy) / speed_sq))
    if time_s <= 0.0:
        return None
    distance_m = math.hypot(px + vx * time_s, py + vy * time_s)
    return time_s, distance_m


def apply_kinematic_conflict_guard(
    decision: Mapping[str, Any],
    world_state: Mapping[str, Any],
    risk_assessment: Mapping[str, Any] | None = None,
    *,
    horizon_s: float = 4.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stop for objects whose relative trajectory intersects the ego corridor."""

    relevant_object_ids = {
        str(item.get("object_id", ""))
        for item in (risk_assessment or {}).get("object_assessments", [])
        if item.get("relevant_to_ego_path") is True
    }
    conflicts = []
    for obj in world_state.get("objects", []):
        object_id = str(obj.get("object_id", ""))
        if risk_assessment is not None and object_id not in relevant_object_ids:
            continue
        category = str(obj.get("category", ""))
        radius_m = _CONFLICT_RADIUS_M.get(category)
        if radius_m is None:
            continue
        approach = _closest_approach(obj, horizon_s)
        if approach is None:
            continue
        time_s, distance_m = approach
        if distance_m > radius_m:
            continue
        conflicts.append(
            {
                "object_id": object_id,
                "category": category,
                "time_to_closest_s": round(time_s, 3),
                "closest_distance_m": round(distance_m, 3),
            }
        )

    conflicts.sort(key=lambda item: item["time_to_closest_s"])
    audit = {
        "source": "scene_understanding_world_state_kinematics",
        "horizon_s": float(horizon_s),
        "conflicts": conflicts,
        "override_applied": False,
    }
    result = copy.deepcopy(dict(decision))
    if not conflicts or result.get("action") == "emergency_brake":
        return result, audit

    assessments = {
        str(item.get("object_id", "")): item
        for item in (risk_assessment or {}).get(
            "object_assessments",
            [],
        )
    }
    primary_assessment = assessments.get(
        str(conflicts[0]["object_id"]),
        {},
    )
    imminent = conflicts[0]["time_to_closest_s"] <= 1.0
    medium_safe_caution = (
        imminent
        and primary_assessment.get("distance_is_safe") is True
        and primary_assessment.get("ttc_risk_level") == "medium"
    )
    if medium_safe_caution:
        result["action"] = "decelerate"
        result["target_speed_kmh"] = round(
            min(float(result.get("target_speed_kmh", 15.0)), 15.0),
            6,
        )
        result["target_lane"] = None
        result["target_location"] = None
        result["emergency"] = False
        result["reason"] = "kinematic_conflict_medium_caution"
        reasons = list(result.get("blocked_reason_codes", []))
        if result["reason"] not in reasons:
            reasons.append(result["reason"])
        result["blocked_reason_codes"] = reasons
        audit["override_applied"] = True
        audit["severity"] = "CONTROLLED_DECELERATION"
        return result, audit

    result["action"] = "emergency_brake" if imminent else "stop"
    result["target_speed_kmh"] = 0.0
    result["target_lane"] = None
    result["target_location"] = None
    result["emergency"] = imminent
    result["reason"] = (
        "kinematic_conflict_imminent"
        if imminent
        else "kinematic_conflict_predicted"
    )
    reasons = list(result.get("blocked_reason_codes", []))
    if result["reason"] not in reasons:
        reasons.append(result["reason"])
    result["blocked_reason_codes"] = reasons
    audit["override_applied"] = True
    return result, audit


def apply_adaptive_cruise_guard(
    decision: Mapping[str, Any],
    world_state: Mapping[str, Any],
    risk_assessment: Mapping[str, Any],
    *,
    speed_limit_kmh: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace cumulative medium-risk braking with a stable following target."""

    objects = {
        str(obj.get("object_id")): obj
        for obj in world_state.get("objects", [])
    }
    candidates = []
    for assessment in risk_assessment.get("object_assessments", []):
        if assessment.get("relevant_to_ego_path") is not True:
            continue
        if assessment.get("distance_is_safe") is not True:
            continue
        if assessment.get("ttc_risk_level") != "none":
            continue
        obj = objects.get(str(assessment.get("object_id")))
        if obj is None or obj.get("category") != "vehicle":
            continue
        candidates.append((float(assessment["distance_m"]), assessment, obj))

    candidates.sort(key=lambda item: item[0])
    audit = {
        "source": "scene_understanding_safe_gap_acc",
        "override_applied": False,
        "target_object_id": None,
        "target_speed_kmh": None,
    }
    result = copy.deepcopy(dict(decision))
    if result.get("action") != "decelerate" or not candidates:
        return result, audit

    distance_m, assessment, obj = candidates[0]
    lead_speed_kmh = max(0.0, float(obj.get("speed_mps", 0.0)) * 3.6)
    desired_gap_m = float(assessment.get("safe_distance_m", 10.0)) + 2.0
    gap_correction_kmh = max(0.0, distance_m - desired_gap_m) * 1.2
    target_speed_kmh = lead_speed_kmh + gap_correction_kmh
    if speed_limit_kmh is not None:
        target_speed_kmh = min(target_speed_kmh, float(speed_limit_kmh))

    result["action"] = "keep_lane"
    result["target_speed_kmh"] = round(max(0.0, target_speed_kmh), 6)
    result["target_lane"] = None
    result["target_location"] = None
    result["emergency"] = False
    result["reason"] = "adaptive_cruise_safe_gap"
    reasons = list(result.get("blocked_reason_codes", []))
    if "cumulative_deceleration_replaced_by_acc" not in reasons:
        reasons.append("cumulative_deceleration_replaced_by_acc")
    result["blocked_reason_codes"] = reasons
    audit.update(
        {
            "override_applied": True,
            "target_object_id": str(obj.get("object_id")),
            "distance_m": round(distance_m, 3),
            "lead_speed_kmh": round(lead_speed_kmh, 3),
            "desired_gap_m": round(desired_gap_m, 3),
            "target_speed_kmh": result["target_speed_kmh"],
        }
    )
    return result, audit
